#!/usr/bin/env python3
"""Per-topology routing profiler driving the real SAELens ordinary-routing code.

Scope
=====
Measures one full ordinary (non-streaming) shard-routing cycle:

    raw activation already materialized on the producer TP root
      -> producer row slicing (``_run_producer_phase2_v2``)
      -> per-endpoint NCCL P2P barrier + batch_isend_irecv
      -> consumer-root ``torch.cat`` assembly
      -> per-hook SAE-TP broadcast
      -> activation ready on every SAE endpoint rank

Unlike ``profile_activation_routing.py``, which reimplements the routing logic,
this profiler calls ``ActivationsStore._produce_one_v2_assembled_batch`` — the
exact method training runs — on top of real ``init_distributed_v2`` process
groups and the real ``compute_routing_table``. SAE pipeline parallelism is
supported, so the per-endpoint fan-out that PP introduces is measured rather
than assumed away.

Excluded: vLLM forward (stubbed, see below), mixing buffer, activation
filtering, SAE compute, streaming/GPU-direct paths.

Why vLLM is stubbed
===================
``_get_raw_llm_batch_with_epoch_restart`` is replaced by a pre-allocated tensor
of exactly the shape the configured model/batch produces. Routing cost depends
on payload bytes, device placement and topology, never on activation values, so
stubbing keeps every routing kernel identical while removing an engine load per
topology. The vLLM step itself is already covered by
``profile_vllm_two_stage_v4.py``, and ``vllm_generate`` is still timed and
subtracted so the reported cycle stays a routing-only interval.

Sizes come from the configuration
=================================
``rows_per_producer = store_batch_size_prompts * context_size`` and
``payload_row_bytes = d_in * len(hook_names) * dtype_bytes`` are derived from the
run configuration, matching what training would move. ``--rows-per-producer``
overrides the row count for byte-driven sweeps.

One torchrun launch per topology
================================
``init_distributed_v2`` asserts an exact world size, which differs per topology,
so each topology gets its own launch. No engine is loaded, so startup is seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

SCRIPT_VERSION = "routing_real_v1"
DEFAULT_OUTPUT_DIR = "sae_lens/autoconfig/profile_results/routing_real_v1"

# Phases recorded by RoutingPhaseProbe, in execution order. `vllm_generate` is
# the stub and is reported separately so it can be subtracted.
ROUTING_PHASES = (
    "vllm_tp_barrier",
    "producer_slice",
    "p2p_barrier",
    "p2p_setup",
    "p2p_exchange",
    "p2p_unpack",
    "consumer_assemble",
    "sae_tp_broadcast",
)
STUB_PHASE = "vllm_generate"

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


@dataclass(frozen=True, order=True)
class Topology:
    """A vLLM/SAE parallel layout. Field order matches the CLI spec order."""

    vllm_dp: int
    vllm_tp: int
    sae_dp: int
    sae_tp: int
    sae_pp: int = 1

    @property
    def world_size(self) -> int:
        """World size ``init_distributed_v2`` requires for the overlapping layout."""
        return max(
            self.vllm_dp * self.vllm_tp,
            self.sae_dp * self.sae_pp * self.sae_tp,
        )

    @property
    def num_endpoints(self) -> int:
        return self.sae_dp * self.sae_pp

    @property
    def name(self) -> str:
        return (
            f"vdp{self.vllm_dp}_vtp{self.vllm_tp}_"
            f"sdp{self.sae_dp}_stp{self.sae_tp}_spp{self.sae_pp}"
        )


def parse_topologies(text: str) -> list[Topology]:
    """Parse ``vDP:vTP:sDP:sTP[:sPP]`` entries separated by commas."""
    out: list[Topology] = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (4, 5):
            raise ValueError(
                "--topologies entries must be vDP:vTP:sDP:sTP[:sPP], e.g. "
                "1:1:1:1 or 2:1:1:2:2"
            )
        values = [int(x) for x in parts]
        if any(x <= 0 for x in values):
            raise ValueError(f"topology values must be positive: {item}")
        out.append(Topology(*values))
    if not out:
        raise ValueError("--topologies is empty")
    # Preserve the caller's order after de-duplication so scan output is predictable.
    seen: set[Topology] = set()
    unique: list[Topology] = []
    for topo in out:
        if topo not in seen:
            seen.add(topo)
            unique.append(topo)
    return unique


# ---------------------------------------------------------------------------
# Route structure derived from the configuration
# ---------------------------------------------------------------------------


def route_structure(
    topo: Topology,
    *,
    rows_per_producer: int,
    payload_row_bytes: int,
) -> dict[str, Any]:
    """Bytes and edge counts each rank moves for one routing cycle.

    ``payload_row_bytes`` is ``d_in * H_payload * dtype_bytes``: the producer
    always sends every hook, since PP stages of one DP replica share the routes.

    A route is local when the producer TP root and the endpoint TP root are the
    same rank, in which case the slice is handed over in memory instead of sent.
    Under PP, one DP consumer's rows are sent once per endpoint, so
    ``sae_pp > 1`` multiplies the remote traffic.
    """
    from sae_lens.shard_routing import compute_routing_table

    table = compute_routing_table(topo.vllm_dp, topo.sae_dp, rows_per_producer)
    producer_root = {p: p * topo.vllm_tp for p in range(topo.vllm_dp)}
    endpoint_root = {e: e * topo.sae_tp for e in range(topo.num_endpoints)}

    local_rows = 0
    remote_rows = 0
    local_edges = 0
    remote_edges = 0
    rows_per_consumer: dict[int, int] = {c: 0 for c in range(topo.sae_dp)}
    remote_fanin: dict[int, int] = {e: 0 for e in range(topo.num_endpoints)}
    remote_fanout: dict[int, int] = {p: 0 for p in range(topo.vllm_dp)}

    for route in table:
        n_rows = route.row_end - route.row_start
        rows_per_consumer[route.consumer_idx] += n_rows
        # Each PP stage of the target DP replica is a separate endpoint that
        # needs the same rows.
        for stage in range(topo.sae_pp):
            e = route.consumer_idx * topo.sae_pp + stage
            if producer_root[route.producer_idx] == endpoint_root[e]:
                local_rows += n_rows
                local_edges += 1
            else:
                remote_rows += n_rows
                remote_edges += 1
                remote_fanin[e] += 1
                remote_fanout[route.producer_idx] += 1

    consumer_row_counts = list(rows_per_consumer.values())
    # Broadcast replicates a consumer's assembled rows to sae_tp - 1 followers.
    broadcast_bytes_max_endpoint = (
        max(consumer_row_counts) * payload_row_bytes if topo.sae_tp > 1 else 0
    )
    return {
        "rows_per_producer": rows_per_producer,
        "payload_row_bytes": payload_row_bytes,
        "total_rows_global": topo.vllm_dp * rows_per_producer,
        "logical_activation_bytes_global": (
            topo.vllm_dp * rows_per_producer * payload_row_bytes
        ),
        "local_rows": local_rows,
        "remote_rows": remote_rows,
        "local_bytes": local_rows * payload_row_bytes,
        "remote_bytes": remote_rows * payload_row_bytes,
        "local_edge_count": local_edges,
        "remote_edge_count": remote_edges,
        "max_remote_fanin": max(remote_fanin.values(), default=0),
        "max_remote_fanout": max(remote_fanout.values(), default=0),
        "consumer_rows_min": min(consumer_row_counts, default=0),
        "consumer_rows_max": max(consumer_row_counts, default=0),
        "broadcast_bytes_max_endpoint": broadcast_bytes_max_endpoint,
        "num_endpoints": topo.num_endpoints,
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Median/mean/min/max/cv for one repeated measurement, in milliseconds."""
    if not values:
        return {}
    mean = statistics.mean(values)
    return {
        "median_ms": statistics.median(values) * 1000.0,
        "mean_ms": mean * 1000.0,
        "min_ms": min(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
        "cv": (statistics.pstdev(values) / mean) if len(values) > 1 and mean > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Worker: build the real store on real process groups and drive real routing
# ---------------------------------------------------------------------------


def _build_store(args: argparse.Namespace, topo: Topology, device: Any) -> Any:
    """Construct an ActivationsStore with the dataset and model paths bypassed.

    ``skip_raw_dataset_load=True`` keeps the store from touching HF datasets;
    the raw batch is stubbed anyway. Hook subsetting mirrors what the runner
    does for PP: the store carries all hooks as the payload and this endpoint's
    stage subset as ``hook_names``.
    """
    import torch

    import sae_lens.distributed_v2 as v2
    from sae_lens.distributed_v2 import hooks_for_pp_rank
    from sae_lens.training.activations_store import ActivationsStore

    all_hooks = [f"blocks.{i}.hook_resid_post" for i in range(args.num_hooks)]
    rows = args.rows_per_producer

    store = ActivationsStore(
        model=None,  # type: ignore[arg-type]
        dataset="",
        streaming=False,
        hook_name=all_hooks[0],
        hook_names=list(all_hooks),
        hook_head_index=None,
        context_size=args.context_size,
        d_in=args.d_in,
        n_batches_in_buffer=1,
        total_training_tokens=rows,
        store_batch_size_prompts=args.store_batch_size_prompts,
        train_batch_size_tokens=rows,
        prepend_bos=False,
        normalize_activations="none",
        device=device,
        dtype=args.dtype,
        skip_raw_dataset_load=True,
    )
    store.is_multi_hook = args.num_hooks > 1
    store._all_hook_names = list(all_hooks)
    if v2.is_consumer() and topo.sae_pp > 1:
        store.hook_names = hooks_for_pp_rank(
            v2.get_sae_pp_rank(), topo.sae_pp, all_hooks
        )
    else:
        store.hook_names = list(all_hooks)

    # Stub the vLLM step with a pre-allocated raw batch of exactly the shape the
    # configured model would produce. Allocated once so the measured interval
    # contains no allocation the real path would not have; the real path
    # receives a fresh tensor per step, but routing only reads from it.
    raw: Any
    if store.is_multi_hook:
        raw = {
            hook: torch.empty(rows, args.d_in, dtype=store.dtype, device=device)
            for hook in all_hooks
        }
    else:
        raw = torch.empty(rows, args.d_in, dtype=store.dtype, device=device)
    store._get_raw_llm_batch_with_epoch_restart = lambda: (raw, None)  # type: ignore[method-assign]
    return store


def _role_name() -> str:
    """This rank's routing role, used to attribute per-phase costs."""
    import sae_lens.distributed_v2 as v2

    producer = v2.is_producer()
    consumer = v2.is_consumer()
    parts: list[str] = []
    if producer:
        parts.append("ptp_root" if v2.get_vllm_tp_rank() == 0 else "ptp_follower")
    if consumer:
        parts.append("stp_root" if v2.get_sae_tp_rank() == 0 else "stp_follower")
    return "+".join(parts) if parts else "idle"


def _measure(
    args: argparse.Namespace,
    topo: Topology,
    store: Any,
    device: Any,
) -> dict[str, Any]:
    """Time ``repeats`` routing cycles and reduce them across ranks.

    Each repeat is bracketed by a global barrier and a device sync so a cycle
    never absorbs the previous cycle's tail. The reported cycle time is the max
    across ranks: routing is not finished until the slowest participant is done.
    """
    import torch
    import torch.distributed as dist

    from sae_lens.training.routing_phase_probe import RoutingPhaseProbe

    for _ in range(args.warmup):
        store._produce_one_v2_assembled_batch()
    dist.barrier()
    torch.cuda.synchronize(device)

    probe = RoutingPhaseProbe(device=device)
    store._routing_probe = probe

    cycle_s: list[float] = []
    stub_s: list[float] = []
    phase_wall: dict[str, list[float]] = {name: [] for name in ROUTING_PHASES}
    phase_gpu: dict[str, list[float]] = {name: [] for name in ROUTING_PHASES}
    peak_delta_mib: list[float] = []

    for _ in range(args.repeats):
        dist.barrier()
        torch.cuda.synchronize(device)
        base_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        assembled = store._produce_one_v2_assembled_batch()
        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - t0

        peak_delta_mib.append(
            max(0, torch.cuda.max_memory_allocated(device) - base_alloc) / (1024**2)
        )
        del assembled
        totals = probe.flush()
        stub = totals.pop(STUB_PHASE, None)
        stub_s.append(stub.wall_s if stub is not None else 0.0)
        # The stub stands in for the vLLM step, so remove it from the cycle to
        # leave a routing-only interval.
        cycle_s.append(wall_s - (stub.wall_s if stub is not None else 0.0))
        for name in ROUTING_PHASES:
            entry = totals.get(name)
            phase_wall[name].append(entry.wall_s if entry is not None else 0.0)
            phase_gpu[name].append(entry.gpu_s if entry is not None else 0.0)
        unexpected = set(totals) - set(ROUTING_PHASES)
        if unexpected:
            raise RuntimeError(
                f"routing probe reported unknown phases {sorted(unexpected)}; "
                "ROUTING_PHASES is out of sync with the instrumented code"
            )

    store._routing_probe = None

    local: dict[str, Any] = {
        "role": _role_name(),
        "cycle": summarize(cycle_s),
        "vllm_stub": summarize(stub_s),
        "peak_delta_mib": statistics.median(peak_delta_mib),
        "phases": {
            name: {
                "wall": summarize(phase_wall[name]),
                "gpu": summarize(phase_gpu[name]),
            }
            for name in ROUTING_PHASES
        },
    }

    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return reduce_ranks([g for g in gathered if g is not None], topo)


def reduce_ranks(
    per_rank: Sequence[dict[str, Any]], topo: Topology
) -> dict[str, Any]:
    """Collapse per-rank measurements into the critical path plus a skew split.

    The cycle time is the max across ranks. Each phase is reported as the max
    across the ranks that entered it, since that is what the cycle pays.

    Barrier phases are additionally split. Every participant leaves a barrier at
    the same moment, so a rank's barrier wall time is roughly
    ``t_last_arrival - t_own_arrival``. The minimum across participants is the
    barrier's own cost, and the difference up to the maximum is arrival skew —
    time one rank spent waiting for another, which no amount of faster routing
    would remove.
    """
    result: dict[str, Any] = {
        "routing_cycle_ms_max": max(
            r["cycle"].get("median_ms", 0.0) for r in per_rank
        ),
        "routing_cycle_ms_min": min(
            r["cycle"].get("median_ms", 0.0) for r in per_rank
        ),
        "routing_cycle_cv_max": max(r["cycle"].get("cv", 0.0) for r in per_rank),
        "vllm_stub_ms_max": max(
            r["vllm_stub"].get("median_ms", 0.0) for r in per_rank
        ),
        "peak_delta_mib_max": max(r["peak_delta_mib"] for r in per_rank),
        "roles": ",".join(sorted({r["role"] for r in per_rank})),
    }

    for name in ROUTING_PHASES:
        # A rank that never entered the phase contributes 0 calls; including it
        # would drag a max/min toward 0 and misreport the phase.
        active = [
            r for r in per_rank if r["phases"][name]["wall"].get("median_ms", 0.0) > 0.0
        ]
        if not active:
            result[f"{name}_wall_ms_max"] = 0.0
            result[f"{name}_gpu_ms_max"] = 0.0
            result[f"{name}_ranks"] = 0
            if name.endswith("barrier"):
                result[f"{name}_own_ms"] = 0.0
                result[f"{name}_skew_ms"] = 0.0
            continue
        walls = [r["phases"][name]["wall"]["median_ms"] for r in active]
        gpus = [r["phases"][name]["gpu"].get("median_ms", 0.0) for r in active]
        result[f"{name}_wall_ms_max"] = max(walls)
        result[f"{name}_gpu_ms_max"] = max(gpus)
        result[f"{name}_ranks"] = len(active)
        # A phase costs whichever of the two is real: wall for CPU-blocking
        # sections (barriers), GPU span for async collectives whose wall time is
        # only the launch. Taking the max per rank avoids having to classify.
        result[f"{name}_cost_ms_max"] = max(
            max(
                r["phases"][name]["wall"]["median_ms"],
                r["phases"][name]["gpu"].get("median_ms", 0.0),
            )
            for r in active
        )
        if name.endswith("barrier"):
            result[f"{name}_own_ms"] = min(walls)
            result[f"{name}_skew_ms"] = max(walls) - min(walls)

    # Exposed routing: what the cycle pays for moving data, with barrier skew
    # attributed separately. Barrier own-cost stays in, arrival skew does not.
    skew = sum(
        result.get(f"{name}_skew_ms", 0.0)
        for name in ROUTING_PHASES
        if name.endswith("barrier")
    )
    result["barrier_skew_ms_total"] = skew
    result["routing_ms_excl_skew"] = result["routing_cycle_ms_max"] - skew
    result["phase_sum_ms_max"] = sum(
        result[f"{name}_wall_ms_max"] for name in ROUTING_PHASES
    )
    result["world_size"] = topo.world_size
    return result


def _worker(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    import sae_lens.distributed_v2 as v2

    topo = parse_topologies(args.topology)[0]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    if dist.get_world_size() != topo.world_size:
        raise RuntimeError(
            f"world_size={dist.get_world_size()} != required {topo.world_size} "
            f"for topology {topo.name}"
        )

    v2.init_distributed_v2(
        P=topo.vllm_dp,
        Q=topo.sae_dp,
        vllm_tp_size=topo.vllm_tp,
        sae_tp_size=topo.sae_tp,
        sae_pp_size=topo.sae_pp,
        batch_size=args.rows_per_producer,
    )

    store = _build_store(args, topo, device)
    measured = _measure(args, topo, store, device)
    structure = route_structure(
        topo,
        rows_per_producer=args.rows_per_producer,
        payload_row_bytes=args.d_in * args.num_hooks * DTYPE_BYTES[args.dtype],
    )

    if dist.get_rank() == 0:
        row = {
            "script_version": SCRIPT_VERSION,
            "topology": topo.name,
            **asdict(topo),
            "d_in": args.d_in,
            "H": args.num_hooks,
            "context_size": args.context_size,
            "store_batch_size_prompts": args.store_batch_size_prompts,
            "dtype": args.dtype,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            **structure,
            **measured,
        }
        out_path = Path(args.output_dir) / "routing_real_profile.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as handle:
            json.dump(row, handle)
            handle.write("\n")
        print(
            f"[{topo.name}] routing={row['routing_cycle_ms_max']:.3f} ms "
            f"(excl skew {row['routing_ms_excl_skew']:.3f} ms) "
            f"p2p={row['p2p_exchange_gpu_ms_max']:.3f} ms "
            f"bcast={row['sae_tp_broadcast_gpu_ms_max']:.3f} ms "
            f"remote={structure['remote_bytes'] / 1024**2:.1f} MiB",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    return 0


# ---------------------------------------------------------------------------
# Controller: one torchrun launch per topology
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

def _controller(args: argparse.Namespace) -> int:
    topologies = parse_topologies(args.topologies)
    gpu_count = len(args.cuda_devices.split(","))
    payload_row_bytes = args.d_in * args.num_hooks * DTYPE_BYTES[args.dtype]

    print("=== SAELens real ordinary-routing profile ===")
    print(f"CUDA_VISIBLE_DEVICES : {args.cuda_devices}")
    print(f"rows per producer    : {args.rows_per_producer}")
    print(f"d_in / H / dtype     : {args.d_in} / {args.num_hooks} / {args.dtype}")
    print(f"payload row bytes    : {payload_row_bytes}")
    for topo in topologies:
        structure = route_structure(
            topo,
            rows_per_producer=args.rows_per_producer,
            payload_row_bytes=payload_row_bytes,
        )
        print(
            f"  {topo.name:34s} world={topo.world_size} "
            f"endpoints={topo.num_endpoints} "
            f"remote={structure['remote_bytes'] / 1024**2:8.1f} MiB "
            f"edges={structure['remote_edge_count']}"
        )
    too_big = [t for t in topologies if t.world_size > gpu_count]
    if too_big:
        raise SystemExit(
            "topologies need more GPUs than --cuda-devices provides: "
            + ", ".join(f"{t.name}(world={t.world_size})" for t in too_big)
        )
    if args.dry_run:
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "routing_real_profile.jsonl"
    jsonl_path.write_text("")

    for idx, topo in enumerate(topologies, start=1):
        print(f"[{idx}/{len(topologies)}] launching {topo.name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={topo.world_size}",
            os.path.abspath(__file__),
            "--worker",
            "--topology",
            f"{topo.vllm_dp}:{topo.vllm_tp}:{topo.sae_dp}:{topo.sae_tp}:{topo.sae_pp}",
            "--rows-per-producer",
            str(args.rows_per_producer),
            "--d-in",
            str(args.d_in),
            "--H",
            str(args.num_hooks),
            "--context-size",
            str(args.context_size),
            "--store-batch-size-prompts",
            str(args.store_batch_size_prompts),
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--output-dir",
            str(output_dir),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        completed = subprocess.run(command, env=env)
        if completed.returncode != 0:
            raise SystemExit(
                f"topology {topo.name} failed with exit code {completed.returncode}"
            )

    rows = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line
    ]
    write_csv(output_dir / "routing_real_profile.csv", rows)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "cuda_visible_devices": args.cuda_devices,
        "rows_per_producer": args.rows_per_producer,
        "d_in": args.d_in,
        "H": args.num_hooks,
        "context_size": args.context_size,
        "store_batch_size_prompts": args.store_batch_size_prompts,
        "dtype": args.dtype,
        "payload_row_bytes": payload_row_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "topologies": [t.name for t in topologies],
        "timing_scope": (
            "raw activation ready -> producer slice -> per-endpoint P2P barrier "
            "+ batch_isend_irecv -> consumer cat -> per-hook SAE-TP broadcast "
            "-> ready on all SAE endpoint ranks"
        ),
        "excluded": [
            "vLLM forward (stubbed and subtracted as vllm_generate)",
            "mixing_buffer",
            "activation filtering",
            "SAE compute",
            "streaming / GPU-direct paths",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(f"Wrote {len(rows)} rows to {output_dir / 'routing_real_profile.csv'}")
    return 0

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--topology", default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--topologies",
        default="1:1:1:1,2:1:1:1,1:1:2:1,2:1:2:1",
        help="vDP:vTP:sDP:sTP[:sPP] entries, e.g. 2:1:1:2:2. sPP defaults to 1.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        help="Visible physical GPUs. The largest topology must fit in this count.",
    )

    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--H", "--num-hooks", dest="num_hooks", type=int, default=2)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--store-batch-size-prompts", type=int, default=1)
    parser.add_argument(
        "--rows-per-producer",
        type=int,
        default=None,
        help=(
            "Rows each producer routes per cycle. Defaults to "
            "store_batch_size_prompts * context_size, which is what training moves."
        ),
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="float32")

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.rows_per_producer is None:
        args.rows_per_producer = args.store_batch_size_prompts * args.context_size
    if args.worker and args.topology is None:
        parser.error("--worker requires --topology")
    if args.d_in <= 0 or args.num_hooks <= 0 or args.rows_per_producer <= 0:
        parser.error("--d-in, --H and --rows-per-producer must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be >= 0 and --repeats > 0")
    try:
        parse_topologies(args.topology or args.topologies)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return _worker(args)
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
