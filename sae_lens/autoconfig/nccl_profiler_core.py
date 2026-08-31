#!/usr/bin/env python3
"""External profiler for NCCL TP collectives, keyed by buffer_size and tp.

Profiles the communication collectives the SAE training step uses
(``dist.all_gather`` in encode, ``dist.all_reduce`` in decode / grad-sync /
clip-norm; see ``sae_lens.distributed``), plus ``broadcast`` and
``reduce_scatter`` for completeness. Parameterized by:

  * buffer_bytes  local buffer bytes each rank contributes (NOT B/d_in/d_sae)
  * tp/group_size process-group size (== world_size for the run; tp kept for CSV compatibility)
  * dtype         element type (buffer element count = buffer_bytes / itemsize)
  * collective    allgather | allreduce | broadcast | reduce_scatter
  * topology      NCCL_ALGO / NCCL_PROTO / NCCL_P2P_LEVEL (optional; recorded)

The tensor is 1-D so shape semantics are reduced to payload bytes. The output
contains every rank's isolated barrier-synchronized service samples plus the
per-repeat maximum across ranks; wrapper-local clone/cat/slice work is excluded.

Must be launched multi-rank (torchrun / mp.spawn) with world_size == tp. NCCL
cannot be profiled on a single process; a gloo path exists for CPU smoke tests.

Topology ordering constraint: NCCL reads NCCL_ALGO/PROTO/P2P_LEVEL only at
communicator init, so the worker sets them BEFORE init_process_group. One run
profiles one topo config; sweep topo across multiple launches.
"""

from __future__ import annotations

# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================
# Cover both latency-dominated control collectives and bandwidth-dominated
# activation/decoder payloads.  Important current SAE points include 4 B
# (gradient-norm scalar), 16 KiB (b_dec gradient at D=4096, fp32), and
# 128 MiB (large TP2 activation all-gather).
_KIB = 1024
_MIB = 1024 * 1024
DEFAULT_BUFFER_BYTES: list[int] = [
    4,
    16,
    64,
    256,
    1 * _KIB,
    4 * _KIB,
    16 * _KIB,
    64 * _KIB,
    256 * _KIB,
    1 * _MIB,
    2 * _MIB,
    4 * _MIB,
    8 * _MIB,
    16 * _MIB,
    32 * _MIB,
    64 * _MIB,
    128 * _MIB,
    256 * _MIB,
    1024 * _MIB,
    2048 * _MIB,
]
DEFAULT_TP_VALUES: list[int] = [2, 4]
DEFAULT_COLLECTIVES: list[str] = ["allgather", "allreduce", "broadcast", "reduce_scatter"]
DEFAULT_DTYPES: list[str] = ["float32"]

DEFAULT_WARMUP: int = 3
DEFAULT_REPEATS: int = 5
DEFAULT_OUTPUT_DIR: str = "sae_lens/autoconfig/profile_results"
DEFAULT_OUTPUT_NAME: str = "nccl_comm_profile"
# =============================================================================

import argparse
import csv
import gc
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

COLLECTIVES = ("allgather", "allreduce", "broadcast", "reduce_scatter")

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def canonical_dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    return mapping.get(dtype, str(dtype).removeprefix("torch."))


def parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in DTYPE_ALIASES:
        supported = ", ".join(sorted(DTYPE_ALIASES))
        raise ValueError(f"Unsupported dtype {name!r}. Supported aliases: {supported}")
    return DTYPE_ALIASES[key]


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


@dataclass(frozen=True)
class NcclCase:
    """One NCCL profile point: (collective, buffer_bytes, tp, dtype)."""

    collective: str
    buffer_bytes: int
    tp: int
    dtype_name: str
    device: str

    @property
    def dtype(self) -> torch.dtype:
        return parse_dtype(self.dtype_name)

    @property
    def buffer_numel(self) -> int:
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        if self.buffer_bytes % itemsize != 0:
            raise ValueError(
                f"buffer_bytes={self.buffer_bytes} not divisible by "
                f"{self.dtype_name} itemsize={itemsize}"
            )
        return self.buffer_bytes // itemsize

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def case_id(self) -> str:
        return f"{self.collective}_buf{self.buffer_bytes}_tp{self.tp}_{self.dtype_name}"


@dataclass
class NcclSweepConfig:
    buffer_bytes_values: list[int]
    tp_values: list[int]
    collectives: list[str]
    dtypes: list[str]

    warmup: int = 20
    repeats: int = 50
    output_dir: str = "profile_results"
    output_name: str = "nccl_comm_profile"
    backend: str = "nccl"  # "nccl" (GPU) or "gloo" (CPU smoke test)

    # Topology knobs. Empty string == inherit / auto. Set before init.
    nccl_algo: str = ""
    nccl_proto: str = ""
    nccl_p2p_level: str = ""

    def normalize(self) -> "NcclSweepConfig":
        self.buffer_bytes_values = unique_preserve_order(
            int(x) for x in self.buffer_bytes_values
        )
        self.tp_values = unique_preserve_order(int(x) for x in self.tp_values)
        self.collectives = unique_preserve_order(str(x).lower() for x in self.collectives)
        self.dtypes = unique_preserve_order(
            canonical_dtype_name(parse_dtype(str(x))) for x in self.dtypes
        )
        return self

    def validate_inputs(self) -> None:
        if not self.buffer_bytes_values:
            raise ValueError("buffer_bytes_values must not be empty")
        if not self.tp_values:
            raise ValueError("tp_values must not be empty")
        if not self.collectives:
            raise ValueError("collectives must not be empty")
        for c in self.collectives:
            if c not in COLLECTIVES:
                raise ValueError(f"Unknown collective {c!r}; choices={COLLECTIVES}")
        for b in self.buffer_bytes_values:
            if b <= 0:
                raise ValueError(f"buffer_bytes must be positive; got {b}")
        for tp in self.tp_values:
            if tp < 2:
                raise ValueError(f"tp must be >= 2 for NCCL; got {tp}")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")

    def device_for(self, backend: str) -> str:
        return "cpu" if backend == "gloo" else "cuda"

    def iter_cases(self, tp: int, device: str) -> Iterable[NcclCase]:
        import itertools

        for collective, buffer_bytes, dtype_name in itertools.product(
            self.collectives, self.buffer_bytes_values, self.dtypes
        ):
            # allgather output and reduce_scatter input must split evenly across tp.
            yield NcclCase(
                collective=collective,
                buffer_bytes=buffer_bytes,
                tp=tp,
                dtype_name=dtype_name,
                device=device,
            )


@dataclass(frozen=True)
class TimingStats:
    samples: int
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    p10_ms: float
    p90_ms: float
    max_ms: float


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_samples(samples_ms: Sequence[float]) -> TimingStats:
    if not samples_ms:
        raise ValueError("No timing samples were collected")
    ordered = sorted(float(x) for x in samples_ms)
    return TimingStats(
        samples=len(ordered),
        median_ms=float(statistics.median(ordered)),
        mean_ms=float(statistics.fmean(ordered)),
        std_ms=float(statistics.pstdev(ordered) if len(ordered) > 1 else 0.0),
        min_ms=float(ordered[0]),
        p10_ms=percentile(ordered, 0.10),
        p90_ms=percentile(ordered, 0.90),
        max_ms=float(ordered[-1]),
    )


def build_collective_op(
    case: NcclCase,
    group: Any,
    world_size: int,
) -> Callable[[], None]:
    """Build one isolated collective without unrelated spare buffers.

    ``buffer_bytes`` is the LOCAL payload each rank contributes. Tensor setup is
    outside the timed window. Wrapper-local work such as the TP all-reduce clone,
    all-gather ``zeros_like`` allocation/``cat`` and backward slicing belongs to
    the single-device local profiler, not this NCCL primitive profiler.
    """
    import torch.distributed as dist

    device = case.torch_device
    numel = case.buffer_numel
    dtype = case.dtype

    if case.collective == "allgather":
        local = torch.ones(numel, device=device, dtype=dtype)
        out = [torch.empty(numel, device=device, dtype=dtype) for _ in range(world_size)]

        def op() -> None:
            dist.all_gather(out, local, group=group)

    elif case.collective == "allreduce":
        # One in-place NCCL buffer. The original pre-clone tensor and clone kernel
        # are intentionally handled by local_compute.
        buf = torch.ones(numel, device=device, dtype=dtype)

        def op() -> None:
            dist.all_reduce(buf, group=group)

    elif case.collective == "broadcast":
        # Broadcast is in-place and needs only one payload buffer.
        buf = torch.ones(numel, device=device, dtype=dtype)
        root = dist.get_global_rank(group, 0) if hasattr(dist, "get_global_rank") else 0

        def op() -> None:
            dist.broadcast(buf, src=root, group=group)

    elif case.collective == "reduce_scatter":
        # The list API genuinely requires tp input shards plus one output shard.
        inputs = [torch.ones(numel, device=device, dtype=dtype) for _ in range(world_size)]
        out_rs = torch.empty(numel, device=device, dtype=dtype)

        def op() -> None:
            dist.reduce_scatter(out_rs, inputs, group=group)

    else:
        raise ValueError(f"Unknown collective {case.collective!r}")

    return op


def measure_collective(
    op: Callable[[], None],
    case: NcclCase,
    group: Any,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Return this rank's per-repeat isolated collective samples in milliseconds."""
    import torch.distributed as dist

    device = case.torch_device
    is_cuda = device.type == "cuda"

    for _ in range(warmup):
        dist.barrier(group=group)
        op()
    if is_cuda:
        torch.cuda.synchronize(device)
    dist.barrier(group=group)

    samples_ms: list[float] = []
    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            dist.barrier(group=group)
            start.record()
            op()
            end.record()
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(repeats):
            dist.barrier(group=group)
            t0 = time.perf_counter_ns()
            op()
            t1 = time.perf_counter_ns()
            samples_ms.append((t1 - t0) / 1_000_000.0)
    return samples_ms


def _cleanup_case(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def profile_case(
    case: NcclCase,
    config: NcclSweepConfig,
    group: Any,
    control_group: Any,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Profile one case; rank 0 returns all-rank and group-max rows.

    Buffer construction is coordinated through a CPU/Gloo control group before
    any rank enters the measured collective. If one rank cannot allocate its
    legitimate primitive buffers, every rank skips the case instead of leaving
    peers blocked inside NCCL.
    """
    import torch.distributed as dist

    op: Callable[[], None] | None = None
    local_status: dict[str, Any]
    try:
        _ = case.buffer_numel
        op = build_collective_op(case, group, world_size)
        local_status = {"ok": True, "rank": rank, "error_type": "", "error_message": "", "error_traceback": ""}
    except Exception as exc:  # noqa: BLE001
        local_status = {
            "ok": False,
            "rank": rank,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(limit=12),
        }

    statuses: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(statuses, local_status, group=control_group)
    failed = [status for status in statuses if status is not None and not status["ok"]]
    if failed:
        op = None
        _cleanup_case(case.torch_device)
        dist.barrier(group=control_group)
        if rank != 0:
            return []
        rows: list[dict[str, Any]] = []
        for peer_rank, status in enumerate(statuses):
            assert status is not None
            if status["ok"]:
                rows.append(
                    _peer_abort_row(
                        case,
                        config,
                        peer_rank,
                        world_size,
                        failed,
                    )
                )
            else:
                rows.append(
                    _error_row_from_status(case, config, status, world_size)
                )
        return rows

    assert op is not None
    try:
        local_samples = measure_collective(
            op, case, group, config.warmup, config.repeats
        )
        payload = {
            "ok": True,
            "rank": rank,
            "samples_ms": local_samples,
            "error_type": "",
            "error_message": "",
            "error_traceback": "",
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "ok": False,
            "rank": rank,
            "samples_ms": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(limit=12),
        }

    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    # This gather is reached only if every rank returned from the measured loop.
    dist.all_gather_object(gathered, payload, group=control_group)
    op = None
    _cleanup_case(case.torch_device)
    dist.barrier(group=control_group)

    if rank != 0:
        return []

    rows: list[dict[str, Any]] = []
    failures = [item for item in gathered if item is not None and not item["ok"]]
    if failures:
        for item in gathered:
            assert item is not None
            if item["ok"]:
                rows.append(
                    _peer_abort_row(case, config, int(item["rank"]), world_size, failures)
                )
            else:
                rows.append(_error_row_from_status(case, config, item, world_size))
        return rows

    rank_samples: list[list[float]] = []
    for item in gathered:
        assert item is not None
        samples = [float(value) for value in item["samples_ms"]]
        rank_samples.append(samples)
        rows.append(
            _success_row(
                case,
                config,
                summarize_samples(samples),
                rank=int(item["rank"]),
                world_size=world_size,
                row_kind="rank",
                rank_label=f"rank_{int(item['rank'])}",
                samples_ms=samples,
            )
        )

    if any(len(samples) != config.repeats for samples in rank_samples):
        raise RuntimeError("Ranks returned different NCCL sample counts")
    group_max_samples = [
        max(rank_samples[r][i] for r in range(world_size))
        for i in range(config.repeats)
    ]
    rows.append(
        _success_row(
            case,
            config,
            summarize_samples(group_max_samples),
            rank=-1,
            world_size=world_size,
            row_kind="group_max",
            rank_label="group_max_per_repeat",
            samples_ms=group_max_samples,
        )
    )
    return rows


def _topo_fields(config: NcclSweepConfig) -> dict[str, Any]:
    return {
        "nccl_algo": config.nccl_algo or "auto",
        "nccl_proto": config.nccl_proto or "auto",
        "nccl_p2p_level": config.nccl_p2p_level or "auto",
    }


def _safe_buffer_numel(case: NcclCase) -> int | str:
    try:
        return case.buffer_numel
    except Exception:
        return ""


def _success_row(
    case: NcclCase,
    config: NcclSweepConfig,
    stats: TimingStats,
    rank: int,
    world_size: int,
    *,
    row_kind: str,
    rank_label: str,
    samples_ms: Sequence[float],
) -> dict[str, Any]:
    return {
        "profiler": "nccl",
        "status": "ok",
        "error_type": "",
        "error_message": "",
        "error_traceback": "",
        "case_id": case.case_id,
        "collective": case.collective,
        "buffer_bytes": case.buffer_bytes,
        "buffer_numel": case.buffer_numel,
        "tp": case.tp,
        "group_size": case.tp,
        "dtype": case.dtype_name,
        "backend": config.backend,
        **_topo_fields(config),
        "row_kind": row_kind,
        "rank": rank,
        "rank_label": rank_label,
        "world_size": world_size,
        "warmup": config.warmup,
        "repeats": config.repeats,
        "samples_ms": list(samples_ms),
        **asdict(stats),
    }


def _error_row_from_status(
    case: NcclCase,
    config: NcclSweepConfig,
    status: dict[str, Any],
    world_size: int,
) -> dict[str, Any]:
    row = blank_row()
    row.update(
        {
            "profiler": "nccl",
            "status": "error",
            "error_type": status.get("error_type", "RuntimeError"),
            "error_message": status.get("error_message", ""),
            "error_traceback": status.get("error_traceback", ""),
            "case_id": case.case_id,
            "collective": case.collective,
            "buffer_bytes": case.buffer_bytes,
            "buffer_numel": _safe_buffer_numel(case),
            "tp": case.tp,
            "group_size": case.tp,
            "dtype": case.dtype_name,
            "backend": config.backend,
            **_topo_fields(config),
            "row_kind": "rank",
            "rank": int(status.get("rank", -1)),
            "rank_label": f"rank_{int(status.get('rank', -1))}",
            "world_size": world_size,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "samples": 0,
            "samples_ms": [],
        }
    )
    return row


def _peer_abort_row(
    case: NcclCase,
    config: NcclSweepConfig,
    rank: int,
    world_size: int,
    failed_statuses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failed_ranks = [int(status.get("rank", -1)) for status in failed_statuses]
    status = {
        "rank": rank,
        "error_type": "PeerAllocationOrCollectiveFailure",
        "error_message": (
            f"case skipped on rank {rank} because peer rank(s) {failed_ranks} failed "
            "before all ranks could safely complete the case"
        ),
        "error_traceback": "",
    }
    return _error_row_from_status(case, config, status, world_size)


CSV_FIELDS: list[str] = [
    "profiler",
    "status",
    "error_type",
    "error_message",
    "error_traceback",
    "case_id",
    "collective",
    "buffer_bytes",
    "buffer_numel",
    "tp",
    "group_size",
    "dtype",
    "backend",
    "nccl_algo",
    "nccl_proto",
    "nccl_p2p_level",
    "row_kind",
    "rank",
    "rank_label",
    "world_size",
    "warmup",
    "repeats",
    "samples_ms",
    "samples",
    "median_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "p10_ms",
    "p90_ms",
    "max_ms",
]


def blank_row() -> dict[str, Any]:
    return {key: "" for key in CSV_FIELDS}


def csv_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def unique_output_paths(output_dir: Path, requested_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(requested_name).stem or "nccl_comm_profile"
    suffix = 0
    while True:
        candidate = stem if suffix == 0 else f"{stem}_{suffix}"
        csv_path = output_dir / f"{candidate}.csv"
        json_path = output_dir / f"{candidate}.json"
        if not csv_path.exists() and not json_path.exists():
            return csv_path, json_path
        suffix += 1


def _nvidia_smi_topo() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}>"


def _nccl_version() -> str:
    try:
        ver = torch.cuda.nccl.version()  # type: ignore[attr-defined]
        return ".".join(str(x) for x in ver) if isinstance(ver, tuple) else str(ver)
    except Exception:  # noqa: BLE001
        return "<unknown>"


def build_metadata(config: NcclSweepConfig, started_at: datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profiler": "nccl",
        "started_at_utc": started_at.isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": _nccl_version() if config.backend == "nccl" else "n/a",
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi_topo": _nvidia_smi_topo() if config.backend == "nccl" else "n/a",
        "active_topo_env": {
            "NCCL_ALGO": os.environ.get("NCCL_ALGO", "<unset>"),
            "NCCL_PROTO": os.environ.get("NCCL_PROTO", "<unset>"),
            "NCCL_P2P_LEVEL": os.environ.get("NCCL_P2P_LEVEL", "<unset>"),
        },
        "config": asdict(config),
        "semantics": {
            "buffer_bytes": "LOCAL bytes each rank contributes (not B/d_in/d_sae)",
            "latency": "isolated barrier-synchronized collective service latency",
            "aggregation": "all rank samples plus per-repeat group maximum",
            "collectives": list(COLLECTIVES),
            "wrapper_local_work": "excluded here and profiled single-device in local_compute",
            "backward": "collective communication excluded where backward is local/identity",
            "topo_ordering": "NCCL_ALGO/PROTO/P2P_LEVEL set before init_process_group",
        },
    }


def apply_topo_env(config: NcclSweepConfig) -> None:
    """Set NCCL topology env vars BEFORE any NCCL init. No-op for empty values."""
    if config.nccl_algo:
        os.environ["NCCL_ALGO"] = config.nccl_algo
    if config.nccl_proto:
        os.environ["NCCL_PROTO"] = config.nccl_proto
    if config.nccl_p2p_level:
        os.environ["NCCL_P2P_LEVEL"] = config.nccl_p2p_level


def run_worker(config: NcclSweepConfig) -> list[dict[str, Any]]:
    """One rank. Rank 0 returns all per-rank and group-max rows.

    The NCCL payload group and Gloo control group are explicitly destroyed at
    the end of the run so repeated profiler launches do not leak communicators.
    """
    import torch.distributed as dist

    config = config.normalize()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks = list(range(world_size))
    group: Any | None = None
    control_group: Any | None = None
    try:
        group = dist.new_group(ranks)
        # A CPU control plane coordinates allocation failures and gathers raw
        # samples without consuming the NCCL payload buffers being measured.
        control_group = dist.new_group(ranks, backend="gloo")

        device = config.device_for(config.backend)
        if device == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
                raise RuntimeError("NCCL backend requested but no CUDA device is available")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
                )
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"

        rows: list[dict[str, Any]] = []
        for case in config.iter_cases(tp=world_size, device=device):
            rows.extend(
                profile_case(
                    case,
                    config,
                    group,
                    control_group,
                    rank,
                    world_size,
                )
            )
        return rows
    finally:
        # All ranks execute the same case loop.  Best-effort subgroup teardown is
        # sufficient here; the default process group is destroyed in main().
        if group is not None:
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        if control_group is not None:
            try:
                dist.destroy_process_group(control_group)
            except Exception:
                pass


def write_outputs(
    config: NcclSweepConfig, rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    csv_path, json_path = unique_output_paths(Path(config.output_dir), config.output_name)
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key, "")) for key in CSV_FIELDS})
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {"metadata": build_metadata(config, datetime.now(timezone.utc)), "results": rows},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    return csv_path, json_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile NCCL collectives by payload bytes and process-group size (+ topo)."
    )
    parser.add_argument(
        "--buffer-bytes", dest="buffer_bytes_values", type=int, nargs="+",
        default=list(DEFAULT_BUFFER_BYTES),
        help="Local buffer bytes per rank",
    )
    parser.add_argument(
        "--tp", "--group-size", dest="tp_values", type=int, nargs="+", default=list(DEFAULT_TP_VALUES),
        help="Process-group sizes (legacy name: --tp); world_size must equal the selected size",
    )
    parser.add_argument(
        "--collectives", nargs="+", choices=list(COLLECTIVES),
        default=list(DEFAULT_COLLECTIVES),
    )
    parser.add_argument(
        "--dtype", "--dtypes", dest="dtypes", nargs="+", default=list(DEFAULT_DTYPES),
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--nccl-algo", default="", help="NCCL_ALGO (e.g. Ring, Tree)")
    parser.add_argument("--nccl-proto", default="", help="NCCL_PROTO (e.g. LL, LL128, Simple)")
    parser.add_argument("--nccl-p2p-level", default="", help="NCCL_P2P_LEVEL (e.g. NVL, PIX, SYS)")
    parser.add_argument(
        "--auto-launch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When invoked with plain python and exactly one --tp/--group-size value, "
            "relaunch automatically through torch.distributed.run. "
            "Use --no-auto-launch inside an existing launcher/debug session."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> NcclSweepConfig:
    return NcclSweepConfig(
        buffer_bytes_values=list(args.buffer_bytes_values),
        tp_values=list(args.tp_values),
        collectives=list(args.collectives),
        dtypes=list(args.dtypes),
        warmup=args.warmup,
        repeats=args.repeats,
        output_dir=args.output_dir,
        output_name=args.output_name,
        backend=args.backend,
        nccl_algo=args.nccl_algo,
        nccl_proto=args.nccl_proto,
        nccl_p2p_level=args.nccl_p2p_level,
    )


def _has_torchrun_environment() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def _auto_launch_if_needed(
    args: argparse.Namespace,
    config: NcclSweepConfig,
    argv: Sequence[str] | None,
) -> int | None:
    if _has_torchrun_environment():
        return None
    if not args.auto_launch:
        raise RuntimeError(
            "Distributed environment is not initialized. Run with "
            "`torchrun --standalone --nproc_per_node=<tp> "
            "sae_lens/autoconfig/external_nccl_profiler.py ...` or omit "
            "--no-auto-launch."
        )
    if len(config.tp_values) != 1:
        raise RuntimeError(
            "Plain-python auto-launch requires exactly one --tp/--group-size value. "
            "Run one profiler launch per TP degree so each world_size equals TP."
        )
    tp = int(config.tp_values[0])
    original_args = list(sys.argv[1:] if argv is None else argv)
    # BooleanOptionalAction accepts the final occurrence; force child mode.
    child_args = [*original_args, "--no-auto-launch"]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={tp}",
        str(Path(__file__).resolve()),
        *child_args,
    ]
    print(
        "[nccl-profiler] no torchrun environment detected; launching:\n  "
        + " ".join(command),
        flush=True,
    )
    return int(subprocess.run(command, check=False).returncode)


def main(argv: Sequence[str] | None = None) -> int:
    import torch.distributed as dist

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    initialized_here = False
    local_rank: int | None = None
    try:
        config = config_from_args(args)
        config.normalize().validate_inputs()

        launched = _auto_launch_if_needed(args, config, argv)
        if launched is not None:
            return launched

        # Topology env must be set before init_process_group.
        apply_topo_env(config)

        if config.backend == "nccl":
            local_rank = int(os.environ["LOCAL_RANK"])
            if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} is invalid for visible CUDA device_count="
                    f"{torch.cuda.device_count()}"
                )
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            if config.backend == "nccl":
                device = torch.device(f"cuda:{local_rank}")
                try:
                    dist.init_process_group(backend=config.backend, device_id=device)
                except TypeError:
                    # Compatibility with PyTorch versions predating device_id.
                    dist.init_process_group(backend=config.backend)
            else:
                dist.init_process_group(backend=config.backend)
            initialized_here = True

        world_size = dist.get_world_size()
        if world_size not in config.tp_values:
            if dist.get_rank() == 0:
                print(
                    f"[nccl-profiler] world_size={world_size} not in tp_values="
                    f"{config.tp_values}; nothing to profile",
                    file=sys.stderr,
                )
            if config.backend == "nccl" and local_rank is not None:
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
            return 2

        rows = run_worker(config)
        if dist.get_rank() == 0:
            csv_path, _ = write_outputs(config.normalize(), rows)
            print(f"[nccl-profiler] done: {len(rows)} rows -> {csv_path}")
        if config.backend == "nccl" and local_rank is not None:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        return 0
    except KeyboardInterrupt:
        print("\n[nccl-profiler] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[nccl-profiler] fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if initialized_here and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())



