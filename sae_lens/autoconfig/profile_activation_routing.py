#!/usr/bin/env python3
"""Fast routing-only profiler for SAELens distributed_v2 ordinary routing.

Profiles the interval:

    producer activations already ready
        -> producer row slicing / multi-hook packing
        -> NCCL P2P routing (including the current per-consumer barrier)
        -> consumer torch.cat assembly
        -> SAE-TP broadcast
        -> consumer activation ready on every SAE-TP rank

It intentionally excludes:
  * vLLM forward / vLLM TP step barrier
  * mixing_buffer / filtering / train-batch preparation
  * SAE forward/backward/optimizer
  * streaming/disjoint topology and SAE pipeline parallelism

The replay uses SAELens' current ``compute_routing_table`` implementation and
recreates the current overlapping rank layout.  One torchrun launch scans many
routing topologies and activation-row anchors, so startup cost stays small.

Recommended use: profile a sparse geometric set of ``rows`` anchors and later
piecewise-linearly interpolate *within the same discrete topology, H, d_in,
dtype and hardware*.  Do not interpolate across TP/DP topology or H.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

# Allow running the downloaded script from the SAELens repository root even
# before copying it into sae_lens/autoconfig/. Editable installs also work.
for _candidate in (Path.cwd(), Path(__file__).resolve().parent):
    if (_candidate / "sae_lens").is_dir():
        _candidate_text = str(_candidate)
        if _candidate_text not in sys.path:
            sys.path.insert(0, _candidate_text)
        break

SCRIPT_VERSION = "routing_only_v1"
DEFAULT_ROWS = "1024,2048,4096,8192,16384,32768,65536"
DEFAULT_OUTPUT_DIR = "sae_lens/autoconfig/profile_results/routing_v1"


@dataclass(frozen=True, order=True)
class Topology:
    vllm_dp: int
    vllm_tp: int
    sae_dp: int
    sae_tp: int

    @property
    def active_world_size(self) -> int:
        return max(self.vllm_dp * self.vllm_tp, self.sae_dp * self.sae_tp)

    @property
    def name(self) -> str:
        return (
            f"vdp{self.vllm_dp}_vtp{self.vllm_tp}_"
            f"sdp{self.sae_dp}_stp{self.sae_tp}"
        )


# ---------------------------------------------------------------------------
# CLI / utilities
# ---------------------------------------------------------------------------


def _parse_int_csv(text: str, *, name: str) -> list[int]:
    try:
        values = sorted({int(x) for x in text.replace(" ", "").split(",") if x})
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if not values or any(x <= 0 for x in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_topologies(text: str) -> list[Topology]:
    """Parse vDP:vTP:sDP:sTP tuples separated by commas."""
    out: list[Topology] = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                "--topologies entries must be vDP:vTP:sDP:sTP, e.g. "
                "1:1:1:1,1:2:1:2"
            )
        vals = [int(x) for x in parts]
        if any(x <= 0 for x in vals):
            raise ValueError("topology values must be positive")
        out.append(Topology(*vals))
    if not out:
        raise ValueError("--topologies is empty")
    return sorted(set(out), key=lambda t: (t.active_world_size, t))


def _dtype_from_name(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _dtype_bytes(name: str) -> int:
    return {"bfloat16": 2, "float16": 2, "float32": 4}[name]


def _visible_gpu_count(cuda_devices: str) -> int:
    devices = [x for x in cuda_devices.split(",") if x.strip()]
    if devices:
        return len(devices)
    import torch

    return torch.cuda.device_count()


def _build_topologies(args: argparse.Namespace) -> list[Topology]:
    if args.topologies:
        topologies = _parse_topologies(args.topologies)
    else:
        topologies = [
            Topology(vdp, vtp, sdp, stp)
            for vdp in args.vllm_dp_values
            for vtp in args.vllm_tp_values
            for sdp in args.sae_dp_values
            for stp in args.sae_tp_values
            if max(vdp * vtp, sdp * stp) <= args.gpu_count
        ]
        topologies = sorted(set(topologies), key=lambda t: (t.active_world_size, t))

    bad = [t for t in topologies if t.active_world_size > args.gpu_count]
    if bad:
        raise ValueError(
            "topologies exceed --gpu-count: " + ", ".join(t.name for t in bad)
        )
    if len(topologies) > args.max_topologies and not args.allow_large_scan:
        raise ValueError(
            f"scan has {len(topologies)} topologies > --max-topologies="
            f"{args.max_topologies}. Narrow the value lists / use --topologies, "
            "or pass --allow-large-scan explicitly."
        )
    return topologies


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Routing-table helpers
# ---------------------------------------------------------------------------


def _routes_for_producer(table: Sequence[Any], producer_idx: int) -> list[Any]:
    return [r for r in table if int(r.producer_idx) == producer_idx]


def _routes_for_consumer(table: Sequence[Any], consumer_idx: int) -> list[Any]:
    return [r for r in table if int(r.consumer_idx) == consumer_idx]


def _producer_root(topo: Topology, p: int) -> int:
    return p * topo.vllm_tp


def _consumer_root(topo: Topology, c: int) -> int:
    return c * topo.sae_tp


def _consumer_tp_ranks(topo: Topology, c: int) -> list[int]:
    base = c * topo.sae_tp
    return list(range(base, base + topo.sae_tp))


def _p2p_members(topo: Topology, table: Sequence[Any], c: int) -> tuple[int, ...]:
    sources = {int(r.producer_idx) for r in _routes_for_consumer(table, c)}
    return tuple(
        sorted(
            {_consumer_root(topo, c)}
            | {_producer_root(topo, p) for p in sources}
        )
    )


def _has_remote_route(topo: Topology, table: Sequence[Any], c: int) -> bool:
    croot = _consumer_root(topo, c)
    return any(
        _producer_root(topo, int(r.producer_idx)) != croot
        for r in _routes_for_consumer(table, c)
    )


def _route_stats(
    topo: Topology,
    table: Sequence[Any],
    *,
    rows_per_producer: int,
    d_in: int,
    num_hooks: int,
    element_bytes: int,
) -> dict[str, Any]:
    row_bytes = d_in * num_hooks * element_bytes
    local_rows = 0
    remote_rows = 0
    remote_edges = 0
    local_edges = 0
    remote_fanin: dict[int, int] = {c: 0 for c in range(topo.sae_dp)}
    remote_fanout: dict[int, int] = {p: 0 for p in range(topo.vllm_dp)}
    consumer_rows: dict[int, int] = {c: 0 for c in range(topo.sae_dp)}

    seen_pc: set[tuple[int, int]] = set()
    for r in table:
        p = int(r.producer_idx)
        c = int(r.consumer_idx)
        key = (p, c)
        if key in seen_pc:
            raise RuntimeError(
                f"routing table has multiple edges for producer={p}, consumer={c}; "
                "the current v2 outgoing dict assumes at most one edge per pair"
            )
        seen_pc.add(key)
        n = int(r.row_end) - int(r.row_start)
        consumer_rows[c] += n
        if _producer_root(topo, p) == _consumer_root(topo, c):
            local_rows += n
            local_edges += 1
        else:
            remote_rows += n
            remote_edges += 1
            remote_fanin[c] += 1
            remote_fanout[p] += 1

    total_rows = topo.vllm_dp * rows_per_producer
    if sum(consumer_rows.values()) != total_rows:
        raise RuntimeError(
            f"routing table row mismatch: consumers={sum(consumer_rows.values())}, "
            f"producers={total_rows}"
        )

    consumer_values = list(consumer_rows.values())
    broadcast_payload_bytes_total = (
        sum(consumer_values) * row_bytes if topo.sae_tp > 1 else 0
    )
    broadcast_payload_bytes_max_consumer = (
        max(consumer_values) * row_bytes if topo.sae_tp > 1 else 0
    )

    return {
        "total_rows_global": total_rows,
        "logical_activation_bytes_global": total_rows * row_bytes,
        "local_rows": local_rows,
        "remote_rows": remote_rows,
        "local_bytes": local_rows * row_bytes,
        "remote_bytes": remote_rows * row_bytes,
        "local_edge_count": local_edges,
        "remote_edge_count": remote_edges,
        "max_remote_fanin": max(remote_fanin.values(), default=0),
        "max_remote_fanout": max(remote_fanout.values(), default=0),
        "consumer_rows_min": min(consumer_values, default=0),
        "consumer_rows_max": max(consumer_values, default=0),
        "broadcast_payload_bytes_total": broadcast_payload_bytes_total,
        "broadcast_payload_bytes_max_consumer": broadcast_payload_bytes_max_consumer,
    }


def _estimate_rank_working_set_bytes(
    topo: Topology,
    table: Sequence[Any],
    *,
    rank: int,
    rows: int,
    d_in: int,
    num_hooks: int,
    element_bytes: int,
) -> int:
    """Conservative allocation estimate used only to avoid profiler-caused OOM.

    Includes synthetic raw activation on producer roots and major route temporaries.
    It is not a memory model and is deliberately conservative.
    """
    one_row = d_in * num_hooks * element_bytes
    raw = 0
    p_idx = None
    if rank < topo.vllm_dp * topo.vllm_tp and rank % topo.vllm_tp == 0:
        p_idx = rank // topo.vllm_tp
        raw = rows * one_row

    recv = 0
    assembled = 0
    if rank < topo.sae_dp * topo.sae_tp:
        c = rank // topo.sae_tp
        c_routes = _routes_for_consumer(table, c)
        croot = _consumer_root(topo, c)
        total_c_rows = sum(int(r.row_end) - int(r.row_start) for r in c_routes)
        assembled = total_c_rows * one_row
        if rank == croot:
            recv = sum(
                (int(r.row_end) - int(r.row_start)) * one_row
                for r in c_routes
                if _producer_root(topo, int(r.producer_idx)) != croot
            )

    # Multi-hook P2P packs H row-slices into one contiguous send tensor.
    pack = 0
    if num_hooks > 1 and p_idx is not None:
        remote_sizes = [
            (int(r.row_end) - int(r.row_start)) * one_row
            for r in _routes_for_producer(table, p_idx)
            if _producer_root(topo, p_idx)
            != _consumer_root(topo, int(r.consumer_idx))
        ]
        pack = max(remote_sizes, default=0)

    # P2P phase roughly raw + recv + pack; assembly/broadcast phase roughly
    # raw + recv + assembled. Add 15% headroom for allocator/protocol extras.
    return int((raw + recv + max(pack, assembled)) * 1.15)


# ---------------------------------------------------------------------------
# Process-group construction
# ---------------------------------------------------------------------------


@dataclass
class TopologyGroups:
    sae_tp_groups: dict[int, Any]
    p2p_groups: dict[tuple[int, tuple[int, ...]], Any]


def _create_groups_for_topology(
    topo: Topology,
    tables_by_rows: dict[int, Sequence[Any]],
) -> TopologyGroups:
    import torch.distributed as dist

    # All world ranks call new_group in the same deterministic order. Ranks outside
    # topo.active_world_size simply receive NON_GROUP_MEMBER handles and stay idle.
    sae_tp_groups: dict[int, Any] = {}
    for c in range(topo.sae_dp):
        ranks = _consumer_tp_ranks(topo, c)
        sae_tp_groups[c] = dist.new_group(ranks=ranks, backend="nccl")

    keys: set[tuple[int, tuple[int, ...]]] = set()
    for table in tables_by_rows.values():
        for c in range(topo.sae_dp):
            if _has_remote_route(topo, table, c):
                keys.add((c, _p2p_members(topo, table, c)))

    p2p_groups: dict[tuple[int, tuple[int, ...]], Any] = {}
    for c, members in sorted(keys, key=lambda x: (x[0], x[1])):
        p2p_groups[(c, members)] = dist.new_group(
            ranks=list(members), backend="nccl"
        )

    return TopologyGroups(
        sae_tp_groups=sae_tp_groups,
        p2p_groups=p2p_groups,
    )


# ---------------------------------------------------------------------------
# Exact ordinary-routing replay
# ---------------------------------------------------------------------------


def _route_once(
    *,
    topo: Topology,
    table: Sequence[Any],
    groups: TopologyGroups,
    raw_hooks: list[Any] | None,
    rows: int,
    d_in: int,
    num_hooks: int,
    dtype: Any,
    device: Any,
) -> list[Any] | None:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()

    # ------------------------------------------------------------------
    # Producer: current v2 row slicing.  T_route starts with raw activation
    # already materialized, so the vLLM TP step barrier / forward are excluded.
    # ------------------------------------------------------------------
    local_slices: dict[int, list[Any]] = {}
    outgoing: dict[int, list[Any]] = {}

    producer_idx: int | None = None
    if (
        rank < topo.vllm_dp * topo.vllm_tp
        and rank % topo.vllm_tp == 0
    ):
        producer_idx = rank // topo.vllm_tp
        if raw_hooks is None:
            raise RuntimeError("producer root has no synthetic raw activation")
        p_routes = _routes_for_producer(table, producer_idx)
        seen_consumers: set[int] = set()
        for route in p_routes:
            c = int(route.consumer_idx)
            if c in seen_consumers:
                raise RuntimeError(
                    f"multiple routes from producer {producer_idx} to consumer {c}"
                )
            seen_consumers.add(c)
            start = int(route.row_start)
            end = int(route.row_end)
            payload = [x[start:end].contiguous() for x in raw_hooks]
            if _producer_root(topo, producer_idx) == _consumer_root(topo, c):
                local_slices[producer_idx] = payload
            outgoing[c] = payload

    # ------------------------------------------------------------------
    # Current v2 NCCL P2P: per-consumer group barrier, then batched irecv/isend,
    # then work.wait(). Local self-routes are excluded from P2P.
    # ------------------------------------------------------------------
    remote_slices: dict[int, list[Any]] = {}

    for c in range(topo.sae_dp):
        consumer_root = _consumer_root(topo, c)
        c_routes = _routes_for_consumer(table, c)
        remote_routes = [
            r
            for r in c_routes
            if _producer_root(topo, int(r.producer_idx)) != consumer_root
        ]
        if not remote_routes:
            continue

        send_route = None
        if producer_idx is not None:
            for r in _routes_for_producer(table, producer_idx):
                if int(r.consumer_idx) == c:
                    send_route = r
                    break
        should_send = (
            send_route is not None
            and _producer_root(topo, int(send_route.producer_idx)) != consumer_root
        )
        should_recv = rank == consumer_root
        if not should_send and not should_recv:
            continue

        members = _p2p_members(topo, table, c)
        p2p_group = groups.p2p_groups[(c, members)]
        dist.barrier(
            group=p2p_group,
            device_ids=[torch.cuda.current_device()],
        )

        ops: list[Any] = []
        recv_meta: list[tuple[int, Any, int]] = []
        if should_recv:
            for route in remote_routes:
                p = int(route.producer_idx)
                n_rows = int(route.row_end) - int(route.row_start)
                recv_buf = torch.empty(
                    (n_rows * num_hooks, d_in),
                    dtype=dtype,
                    device=device,
                )
                recv_meta.append((p, recv_buf, n_rows))
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_buf,
                        _producer_root(topo, p),
                        group=p2p_group,
                    )
                )

        send_buf = None
        if should_send:
            payload = outgoing[c]
            send_buf = payload[0] if num_hooks == 1 else torch.cat(payload, dim=0)
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_buf,
                    consumer_root,
                    group=p2p_group,
                )
            )

        for work in dist.batch_isend_irecv(ops):
            work.wait()

        if should_recv:
            for p, recv_buf, n_rows in recv_meta:
                if num_hooks == 1:
                    remote_slices[p] = [recv_buf]
                else:
                    remote_slices[p] = list(recv_buf.split(n_rows, dim=0))

        # The current helper returns after all work.wait() calls; a packed send
        # buffer need not survive past this point.
        del send_buf

    # ------------------------------------------------------------------
    # Consumer root assembly + current per-hook SAE-TP broadcast.
    # Followers allocate their receive tensors and participate in broadcasts.
    # ------------------------------------------------------------------
    if rank >= topo.sae_dp * topo.sae_tp:
        return None

    consumer_idx = rank // topo.sae_tp
    sae_tp_rank = rank % topo.sae_tp
    consumer_root = _consumer_root(topo, consumer_idx)
    c_routes = _routes_for_consumer(table, consumer_idx)
    n_consumer_rows = sum(
        int(r.row_end) - int(r.row_start) for r in c_routes
    )

    assembled: list[Any]
    if sae_tp_rank == 0:
        buf_map: dict[int, list[Any]] = {}
        for route in c_routes:
            p = int(route.producer_idx)
            if _producer_root(topo, p) == consumer_root:
                buf_map[p] = local_slices[p]
            else:
                buf_map[p] = remote_slices[p]
        assembled = [
            torch.cat([buf_map[int(r.producer_idx)][h] for r in c_routes], dim=0)
            for h in range(num_hooks)
        ]
    else:
        assembled = [
            torch.empty((n_consumer_rows, d_in), dtype=dtype, device=device)
            for _ in range(num_hooks)
        ]

    if topo.sae_tp > 1:
        sae_tp_group = groups.sae_tp_groups[consumer_idx]
        for hook_acts in assembled:
            dist.broadcast(
                hook_acts,
                src=consumer_root,
                group=sae_tp_group,
            )

    return assembled


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _measure_case(
    *,
    topo: Topology,
    table: Sequence[Any],
    groups: TopologyGroups,
    rows: int,
    d_in: int,
    num_hooks: int,
    dtype_name: str,
    warmup: int,
    repeats: int,
    memory_safety_fraction: float,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    dtype = _dtype_from_name(dtype_name)
    elem_bytes = _dtype_bytes(dtype_name)

    # Avoid a distributed hang from one rank OOMing while another is inside NCCL.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    estimated = _estimate_rank_working_set_bytes(
        topo,
        table,
        rank=rank,
        rows=rows,
        d_in=d_in,
        num_hooks=num_hooks,
        element_bytes=elem_bytes,
    )
    ratio = estimated / max(1, free_bytes)
    ratio_t = torch.tensor([ratio], dtype=torch.float64, device=device)
    free_t = torch.tensor([float(free_bytes)], dtype=torch.float64, device=device)
    est_t = torch.tensor([float(estimated)], dtype=torch.float64, device=device)
    dist.all_reduce(ratio_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(free_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(est_t, op=dist.ReduceOp.MAX)
    max_ratio = float(ratio_t.item())
    min_free = float(free_t.item())
    max_est = float(est_t.item())

    if max_ratio > memory_safety_fraction:
        return {
            "status": "skipped_oom_risk",
            "estimated_profile_working_set_mib_max": max_est / (1024**2),
            "free_mib_min_before_case": min_free / (1024**2),
            "estimated_free_fraction_max": max_ratio,
        }

    # Raw producer activation is allocated once, outside the measured interval:
    # the definition starts when activation is already ready.
    raw_hooks: list[Any] | None = None
    if (
        rank < topo.vllm_dp * topo.vllm_tp
        and rank % topo.vllm_tp == 0
    ):
        raw_hooks = [
            torch.empty((rows, d_in), dtype=dtype, device=device)
            for _ in range(num_hooks)
        ]

    # Warm up lazy NCCL communicator setup, allocator paths, cat kernels, etc.
    for _ in range(warmup):
        dist.barrier()
        torch.cuda.synchronize()
        out = _route_once(
            topo=topo,
            table=table,
            groups=groups,
            raw_hooks=raw_hooks,
            rows=rows,
            d_in=d_in,
            num_hooks=num_hooks,
            dtype=dtype,
            device=device,
        )
        torch.cuda.synchronize()
        del out

    times_ms: list[float] = []
    peak_delta_mib: list[float] = []
    baseline_allocated_mib: list[float] = []

    metric = torch.empty(3, dtype=torch.float64, device=device)
    for _ in range(repeats):
        # Profiler synchronization only; deliberately outside T_route.
        dist.barrier()
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        out = _route_once(
            topo=topo,
            table=table,
            groups=groups,
            raw_hooks=raw_hooks,
            rows=rows,
            d_in=d_in,
            num_hooks=num_hooks,
            dtype=dtype,
            device=device,
        )
        torch.cuda.synchronize()
        local_ms = (time.perf_counter() - start) * 1000.0
        local_peak = torch.cuda.max_memory_allocated(device)
        local_delta = max(0, local_peak - base_alloc) / (1024**2)
        local_base = base_alloc / (1024**2)

        # Critical-path metric across all launched ranks. Idle ranks contribute
        # ~0, so using the full torchrun group is safe and avoids extra groups.
        metric[0] = local_ms
        metric[1] = local_delta
        metric[2] = local_base
        dist.all_reduce(metric, op=dist.ReduceOp.MAX)
        times_ms.append(float(metric[0].item()))
        peak_delta_mib.append(float(metric[1].item()))
        baseline_allocated_mib.append(float(metric[2].item()))
        del out

    median_ms = statistics.median(times_ms)
    global_rows = topo.vllm_dp * rows
    result = {
        "status": "ok",
        "warmup": warmup,
        "repeats": repeats,
        "route_wall_ms_median": median_ms,
        "route_wall_ms_mean": statistics.mean(times_ms),
        "route_wall_ms_min": min(times_ms),
        "route_wall_ms_max": max(times_ms),
        "route_wall_ms_cv": (
            statistics.pstdev(times_ms) / statistics.mean(times_ms)
            if len(times_ms) > 1 and statistics.mean(times_ms) > 0
            else 0.0
        ),
        "route_global_rows_per_s": global_rows / (median_ms / 1000.0),
        "route_peak_delta_mib_max_median": statistics.median(peak_delta_mib),
        "baseline_allocated_mib_max_median": statistics.median(
            baseline_allocated_mib
        ),
        "estimated_profile_working_set_mib_max": max_est / (1024**2),
        "free_mib_min_before_case": min_free / (1024**2),
        "estimated_free_fraction_max": max_ratio,
    }

    del raw_hooks
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Worker / controller
# ---------------------------------------------------------------------------


def _worker(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    from sae_lens.shard_routing import compute_routing_table

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.gpu_count:
        raise RuntimeError(
            f"torchrun world_size={world_size} != requested gpu_count={args.gpu_count}"
        )

    topologies = _build_topologies(args)
    output_dir = Path(args.output_dir)
    rows_out: list[dict[str, Any]] = []

    if rank == 0:
        print(
            f"[ROUTING PROFILE] gpus={args.gpu_count} topologies={len(topologies)} "
            f"rows={args.rows} H={args.num_hooks} d_in={args.d_in} "
            f"dtype={args.dtype}",
            flush=True,
        )

    for topo_idx, topo in enumerate(topologies, start=1):
        # Recompute the actual SAELens routing table for every size anchor.
        tables_by_rows: dict[int, Sequence[Any]] = {}
        for rows in args.rows:
            tables_by_rows[rows] = compute_routing_table(
                topo.vllm_dp,
                topo.sae_dp,
                rows,
            )

        # Create exactly the SAE-TP groups and every distinct per-consumer P2P
        # membership set required by this topology's requested row anchors.
        groups = _create_groups_for_topology(topo, tables_by_rows)
        dist.barrier()

        if rank == 0:
            print(
                f"[{topo_idx}/{len(topologies)}] {topo.name} "
                f"active_world={topo.active_world_size}",
                flush=True,
            )

        for rows in args.rows:
            table = tables_by_rows[rows]
            route_stats = _route_stats(
                topo,
                table,
                rows_per_producer=rows,
                d_in=args.d_in,
                num_hooks=args.num_hooks,
                element_bytes=_dtype_bytes(args.dtype),
            )
            measured = _measure_case(
                topo=topo,
                table=table,
                groups=groups,
                rows=rows,
                d_in=args.d_in,
                num_hooks=args.num_hooks,
                dtype_name=args.dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                memory_safety_fraction=args.memory_safety_fraction,
            )

            if rank == 0:
                row = {
                    "script_version": SCRIPT_VERSION,
                    **asdict(topo),
                    "topology": topo.name,
                    "active_world_size": topo.active_world_size,
                    "profile_world_size": world_size,
                    "rows_per_producer": rows,
                    "d_in": args.d_in,
                    "H": args.num_hooks,
                    "dtype": args.dtype,
                    "element_bytes": _dtype_bytes(args.dtype),
                    **route_stats,
                    **measured,
                }
                rows_out.append(row)
                if measured["status"] == "ok":
                    print(
                        f"  rows={rows:7d} route={measured['route_wall_ms_median']:.3f} ms "
                        f"remote={route_stats['remote_bytes']/(1024**2):.1f} MiB "
                        f"bcast(max)={route_stats['broadcast_payload_bytes_max_consumer']/(1024**2):.1f} MiB",
                        flush=True,
                    )
                else:
                    print(
                        f"  rows={rows:7d} {measured['status']} "
                        f"estimated={measured['estimated_profile_working_set_mib_max']:.1f} MiB "
                        f"free_min={measured['free_mib_min_before_case']:.1f} MiB",
                        flush=True,
                    )
                # Incremental write makes long scans resumable by preserving data
                # if a later case fails, without adding synchronization to timings.
                _write_csv(output_dir / "routing_profile.csv", rows_out)

        dist.barrier()

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        metadata = {
            "script_version": SCRIPT_VERSION,
            "gpu_name_rank0": gpu_name,
            "gpu_count": args.gpu_count,
            "cuda_visible_devices": args.cuda_devices,
            "rows": args.rows,
            "d_in": args.d_in,
            "H": args.num_hooks,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "topologies": [asdict(t) for t in topologies],
            "timing_scope": (
                "activation_ready -> producer slice/pack -> NCCL P2P -> "
                "consumer cat -> SAE-TP broadcast -> activation_ready_on_all_sae_tp"
            ),
            "excluded": [
                "vLLM forward",
                "vLLM TP step barrier",
                "mixing_buffer/filter/train-batch preparation",
                "SAE compute",
                "streaming/disjoint topology",
                "SAE pipeline parallelism",
            ],
            "interpolation_guidance": (
                "Use piecewise-linear interpolation in rows (equivalently payload "
                "bytes when d_in/H/dtype are fixed) only within the exact same "
                "GPU/topology/H/d_in/dtype. Do not interpolate across TP/DP topology "
                "or H. Prefer interpolation inside the measured range; avoid "
                "unvalidated extrapolation."
            ),
        }
        _write_json(output_dir / "run_metadata.json", metadata)
        print(
            f"Wrote {len(rows_out)} rows to {output_dir / 'routing_profile.csv'}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    return 0


def _controller(args: argparse.Namespace, original_argv: list[str]) -> int:
    topologies = _build_topologies(args)
    print("=== SAELens ordinary routing-only profile plan ===")
    print(f"CUDA_VISIBLE_DEVICES : {args.cuda_devices}")
    print(f"GPU count            : {args.gpu_count}")
    print(f"d_in / H / dtype     : {args.d_in} / {args.num_hooks} / {args.dtype}")
    print(f"rows anchors         : {args.rows}")
    print(f"warmup / repeats     : {args.warmup} / {args.repeats}")
    print(f"topology count       : {len(topologies)}")
    for topo in topologies:
        print(f"  {topo.name} active_world={topo.active_world_size}")
    print(
        "Timing scope          : activation ready -> slice/pack -> P2P -> cat -> "
        "SAE-TP broadcast"
    )
    print(
        "Size strategy         : sparse geometric anchors; interpolate later within "
        "each discrete topology"
    )
    if args.dry_run:
        return 0

    # One torchrun for the entire scan. Extra ranks are idle for smaller topologies;
    # all measured routing collectives use exact custom groups on the active ranks.
    filtered_argv = [x for x in original_argv if x != "--worker"]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpu_count}",
        os.path.abspath(__file__),
        "--worker",
        *filtered_argv,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.perf_counter()
    completed = subprocess.run(command, env=env)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"Routing profile finished in {elapsed:.1f}s")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        help="Visible physical GPUs, e.g. 0,1. One torchrun process is launched per entry.",
    )
    parser.add_argument("--gpu-count", type=int, default=None)

    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--H", "--num-hooks", dest="num_hooks", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--rows",
        default=DEFAULT_ROWS,
        help=(
            "Rows per vLLM-DP producer per routing cycle. Geometric anchors are "
            "recommended because later interpolation is done within each topology."
        ),
    )

    parser.add_argument("--vllm-dp-values", default="1,2")
    parser.add_argument("--vllm-tp-values", default="1,2")
    parser.add_argument("--sae-dp-values", default="1,2")
    parser.add_argument("--sae-tp-values", default="1,2")
    parser.add_argument(
        "--topologies",
        default=None,
        help=(
            "Optional explicit vDP:vTP:sDP:sTP list, e.g. "
            "1:1:1:1,1:2:1:2,2:1:1:2. Overrides the four value lists."
        ),
    )
    parser.add_argument("--max-topologies", type=int, default=32)
    parser.add_argument("--allow-large-scan", action="store_true")

    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=0.80,
        help=(
            "Skip a size before collectives if a conservative routing-only working-set "
            "estimate exceeds this fraction of the minimum free memory across ranks."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.gpu_count is None:
        args.gpu_count = _visible_gpu_count(args.cuda_devices)

    try:
        args.rows = _parse_int_csv(args.rows, name="--rows")
        args.vllm_dp_values = _parse_int_csv(
            args.vllm_dp_values, name="--vllm-dp-values"
        )
        args.vllm_tp_values = _parse_int_csv(
            args.vllm_tp_values, name="--vllm-tp-values"
        )
        args.sae_dp_values = _parse_int_csv(
            args.sae_dp_values, name="--sae-dp-values"
        )
        args.sae_tp_values = _parse_int_csv(
            args.sae_tp_values, name="--sae-tp-values"
        )
        _build_topologies(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.gpu_count <= 0:
        parser.error("--gpu-count must be positive")
    if args.d_in <= 0 or args.num_hooks <= 0:
        parser.error("--d-in and --H must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be >=0 and --repeats must be >0")
    if not 0.0 < args.memory_safety_fraction < 1.0:
        parser.error("--memory-safety-fraction must be in (0,1)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    original_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.worker:
        return _worker(args)
    return _controller(args, original_argv)


if __name__ == "__main__":
    raise SystemExit(main())
