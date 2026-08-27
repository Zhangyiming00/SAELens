#!/usr/bin/env python3
"""v4.1 two-stage vLLM activation prefill profiler.

Purpose
=======
Profile the activation-producer side of SAE training with a small number of
vLLM measurements while separating two different questions:

Stage A -- saturation scale
---------------------------
For each TP degree, start ONE vLLM engine with a large, fixed
``max_num_batched_tokens`` capacity and a KV pool large enough for the largest
workload in the scan.  Sweep the actual workload size

    N = B * context_size

while keeping the engine capacity non-binding.  Define

    L_m = smallest N whose throughput >= saturation_fraction * observed_max.

This stage is for locating the useful workload scale.  Its memory numbers are
NOT minimal standalone configuration memory.

Stage B -- sparse local (N, MBT) frontier
-----------------------------------------
Around L_m, build exact-fill workload anchors at approximately

    {L_m/4, L_m/2, L_m, 2*L_m}

and add chunked-prefill points only at L_m/2 and L_m:

    MBT in {N/2, N/4}.

With the default power-of-two Stage-A scan this gives, for B_s = L_m / S,

    B_s/4 : MBT=N
    B_s/2 : MBT=N, N/2, N/4
    B_s   : MBT=N, N/2, N/4
    2B_s  : MBT=N

= 8 Stage-B points per TP.

Runtime MBT switching
---------------------
The pinned vLLM used by this repository allocates model-runner buffers from the
initial ``scheduler_config.max_num_batched_tokens`` capacity, while the
scheduler separately stores the active per-step token budget in
``scheduler.max_num_scheduled_tokens``.  vLLM does not expose a supported API
for resizing max_num_batched_tokens at runtime, and its generic update_config
path does not accept scheduler_config.

Therefore this profiler uses a deliberately narrow "soft MBT" switch:

    scheduler.max_num_scheduled_tokens = active_mbt

only while the engine is idle, and never above the immutable engine capacity.
This changes chunking/scheduling behavior without resizing runner buffers.
An extra warmup is discarded after every switch.

Memory correction for reused engines
-------------------------------------
Stage B reuses one capacity engine per TP.  Runtime is measured directly, but
raw allocated memory contains group-sized KV and runner-capacity allocations.
The script also reports an estimated standalone allocated-memory value by:

1. replacing the group KV pool with the exact activation-capture pool required
   by that case; and
2. correcting immutable MBT-capacity buffers with a runtime-calibrated linear
   slope (optional; enabled by default).

The corrected memory is a model estimate and should be validated against a few
real standalone launches before it is treated as ground truth.

The script records measurements and Pareto flags.  It does not choose the final
producer configuration for a particular SAE consumer rate.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "two_stage_v4.1"

# Defaults pinned to the single operating point used by
# scripts/run_step_window_sweep.py: B=1 prompt, context 2048, MBT 4096,
# float32, hook blocks.21.hook_resid_post (stop_at_layer 22). The B/MBT sweep
# lists are collapsed to that one point so a default invocation measures the
# operating point rather than a frontier; widen them explicitly to sweep again.
DEFAULT_MODEL = "/data/models/Llama-3.1-8B"
DEFAULT_STOP_AT_LAYER = 22
DEFAULT_HOOK_NAME = "blocks.21.hook_resid_post"
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_BATCH_VALUES = "1"
DEFAULT_TP_VALUES = "1,2"
DEFAULT_STAGE_A_MBT = 4096
DEFAULT_SATURATION_FRACTION = 0.99
DEFAULT_EXACT_B_RATIOS = "1"
DEFAULT_CHUNKED_B_RATIOS = "1"
DEFAULT_CHUNK_DIVISORS = "1"
# The runner's --dtype is the activation-store dtype and is never forwarded to
# vLLM, so the engine runs at HookedVLLMModel's bfloat16 default.
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5
DEFAULT_SWITCH_WARMUP = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.50
DEFAULT_BLOCK_SIZE = 16
DEFAULT_CUDA_DEVICES = "0,1"
DEFAULT_EPSILON_PARETO = 0.005
DEFAULT_OUTPUT_DIR = "sae_lens/autoconfig/profile_results/vllm_two_stage_v4"

_TORCHRUN_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@dataclass(frozen=True)
class Case:
    stage: str
    tp: int
    B: int
    context_size: int
    total_tokens: int
    mbt: int
    kind: str

    @property
    def name(self) -> str:
        return (
            f"{self.stage}_tp{self.tp}_B{self.B}_S{self.context_size}"
            f"_N{self.total_tokens}_M{self.mbt}_{self.kind}"
        )

    @property
    def signature(self) -> str:
        return (
            f"{SCRIPT_VERSION}|{self.stage}|tp{self.tp}|B{self.B}|"
            f"S{self.context_size}|N{self.total_tokens}|M{self.mbt}|{self.kind}"
        )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _parse_int_csv(text: str, *, name: str) -> list[int]:
    try:
        values = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_float_csv(text: str, *, name: str) -> list[float]:
    try:
        values = [float(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive numbers")
    return values


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _pool_blocks_for_tokens(tokens: int, block_size: int) -> int:
    # Matches the repository's activation-capture vLLM fork:
    # ceil(required_tokens / block_size) + one safety block.
    return math.ceil(tokens / block_size) + 1


def _sync_all() -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        torch.cuda.synchronize()


def _all_reduce_max_float(value: float) -> float:
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(
        [value], dtype=torch.float64, device=torch.device("cuda", _local_rank())
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _make_tokens(B: int, S: int, vocab_size: int):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(B * 100_003 + S * 101)
    tokens = torch.randint(
        low=100,
        high=max(101, vocab_size - 256),
        size=(B, S),
        generator=generator,
        dtype=torch.long,
    )
    return tokens.to(torch.device("cuda", _local_rank()))


# ---------------------------------------------------------------------------
# Stage-A selection and Stage-B design
# ---------------------------------------------------------------------------


def build_stage_a_cases(*, tp: int, args: argparse.Namespace) -> list[Case]:
    """Scan actual workload N while a large MBT capacity stays non-binding."""
    cases: list[Case] = []
    for B in sorted(set(args.batch_values)):
        total = B * args.context_size
        if total > args.stage_a_mbt:
            continue
        cases.append(
            Case(
                stage="stage_a_saturation",
                tp=tp,
                B=B,
                context_size=args.context_size,
                total_tokens=total,
                mbt=args.stage_a_mbt,
                kind="single_iteration_nonbinding_mbt",
            )
        )
    if not cases:
        raise ValueError(
            f"TP={tp}: no Stage-A case fits stage_a_mbt={args.stage_a_mbt} "
            f"with context_size={args.context_size}"
        )
    return cases


def select_saturation(
    rows: Sequence[dict[str, Any]], *, fraction: float
) -> dict[str, Any]:
    """Return the smallest measured N that reaches fraction * observed max."""
    ok = [
        row
        for row in rows
        if row.get("status") == "ok"
        and math.isfinite(float(row.get("tokens_per_s") or float("nan")))
    ]
    if not ok:
        raise RuntimeError("No successful Stage-A rows to select from")

    ok = sorted(ok, key=lambda row: int(row["total_tokens"]))
    peak_row = max(ok, key=lambda row: float(row["tokens_per_s"]))
    peak = float(peak_row["tokens_per_s"])
    threshold = fraction * peak
    saturated = [row for row in ok if float(row["tokens_per_s"]) >= threshold]
    selected = min(saturated, key=lambda row: int(row["total_tokens"]))

    max_N = max(int(row["total_tokens"]) for row in ok)
    boundary_peak = int(peak_row["total_tokens"]) == max_N
    previous = ok[-2] if len(ok) >= 2 else None
    boundary_growth = None
    if boundary_peak and previous is not None:
        prev_r = float(previous["tokens_per_s"])
        boundary_growth = peak / prev_r - 1.0 if prev_r > 0 else None

    return {
        "max_tokens_per_s": peak,
        "max_B": int(peak_row["B"]),
        "max_total_tokens": int(peak_row["total_tokens"]),
        "threshold_tokens_per_s": threshold,
        "saturation_fraction": fraction,
        "num_saturated": len(saturated),
        "selected_B": int(selected["B"]),
        "L_m_tokens": int(selected["total_tokens"]),
        "selected_tokens_per_s": float(selected["tokens_per_s"]),
        "selected_fraction_of_max": float(selected["tokens_per_s"]) / peak,
        "boundary_peak": boundary_peak,
        "boundary_growth_fraction": boundary_growth,
        "saturated_points": [
            {
                "B": int(row["B"]),
                "total_tokens": int(row["total_tokens"]),
                "tokens_per_s": float(row["tokens_per_s"]),
                "fraction_of_max": float(row["tokens_per_s"]) / peak,
            }
            for row in saturated
        ],
    }


def _scaled_B(B_s: int, ratio: float) -> int:
    # Default B_s is a power of two.  round() keeps this function usable for a
    # refined/non-power-of-two saturation search while always producing a real
    # integer request batch.
    return max(1, int(round(B_s * ratio)))


def build_stage_b_cases(
    *, tp: int, saturation: dict[str, Any], args: argparse.Namespace
) -> list[Case]:
    """Sparse local frontier around the selected saturation workload."""
    B_s = int(saturation["selected_B"])
    seen: set[tuple[int, int]] = set()
    cases: list[Case] = []

    # Exact-fill anchors at ~{Lm/4, Lm/2, Lm, 2Lm}.
    for ratio in args.exact_b_ratios:
        B = _scaled_B(B_s, ratio)
        N = B * args.context_size
        mbt = N
        if mbt < args.min_mbt or (B, mbt) in seen:
            continue
        seen.add((B, mbt))
        cases.append(
            Case(
                stage="stage_b_frontier",
                tp=tp,
                B=B,
                context_size=args.context_size,
                total_tokens=N,
                mbt=mbt,
                kind=f"exact_fill_B_ratio_{ratio:g}",
            )
        )

    # Chunk only the two workload scales most likely to matter for Pareto:
    # B_s/2 and B_s.  Exact fill is already present above.
    for ratio in args.chunked_b_ratios:
        B = _scaled_B(B_s, ratio)
        N = B * args.context_size
        for divisor in args.chunk_divisors:
            mbt = N // divisor
            if mbt < args.min_mbt or mbt < 1 or (B, mbt) in seen:
                continue
            seen.add((B, mbt))
            cases.append(
                Case(
                    stage="stage_b_frontier",
                    tp=tp,
                    B=B,
                    context_size=args.context_size,
                    total_tokens=N,
                    mbt=mbt,
                    kind=f"chunk_B_ratio_{ratio:g}_N_over_{divisor}",
                )
            )

    # Stable order: increasing workload, and for a workload exact/larger MBT first.
    cases.sort(key=lambda case: (case.total_tokens, -case.mbt))
    return cases


# ---------------------------------------------------------------------------
# Narrow runtime scheduler-budget switch
# ---------------------------------------------------------------------------


def _resolve_scheduler(model: Any) -> Any:
    """Resolve the in-process vLLM V1 scheduler without depending on one path.

    SAELens sets VLLM_ENABLE_V1_MULTIPROCESSING=0 and, under torchrun TP,
    external_launcher.  In the pinned vLLM this makes the scheduler reachable
    from the local LLM object.  We intentionally only inspect a few known V1
    layouts and fail loudly if vLLM changes them.
    """
    roots = [model.llm, getattr(model.llm, "llm_engine", None)]
    paths = [
        ("llm_engine", "engine_core", "engine_core", "scheduler"),
        ("llm_engine", "engine_core", "scheduler"),
        ("engine_core", "engine_core", "scheduler"),
        ("engine_core", "scheduler"),
        ("scheduler",),
    ]
    for root in roots:
        if root is None:
            continue
        for path in paths:
            obj = root
            try:
                for attr in path:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            if hasattr(obj, "max_num_scheduled_tokens") and hasattr(obj, "schedule"):
                return obj
    raise RuntimeError(
        "Could not resolve the in-process vLLM scheduler. The pinned repository "
        "layout may have changed. Do not fake a hot MBT update: either update "
        "_resolve_scheduler() for the new vLLM layout or use standalone launches."
    )


def _scheduler_is_idle(scheduler: Any) -> bool:
    for attr in ("running", "waiting"):
        value = getattr(scheduler, attr, None)
        if value is None:
            continue
        try:
            if len(value) != 0:
                return False
        except TypeError:
            pass
    requests = getattr(scheduler, "requests", None)
    if requests is not None:
        try:
            if len(requests) != 0:
                return False
        except TypeError:
            pass
    return True


def _engine_mbt_capacity(scheduler: Any) -> int:
    cfg = getattr(scheduler, "scheduler_config", None)
    capacity = getattr(cfg, "max_num_batched_tokens", None)
    if capacity is None:
        raise RuntimeError("Unable to read scheduler_config.max_num_batched_tokens")
    return int(capacity)


def _set_active_mbt(model: Any, new_mbt: int) -> tuple[int, int]:
    """Change only the scheduler's active per-step token budget.

    This is NOT a buffer resize.  The immutable model-runner capacity remains
    the max_num_batched_tokens used at engine construction.
    """
    scheduler = _resolve_scheduler(model)
    if not _scheduler_is_idle(scheduler):
        raise RuntimeError("Refusing to switch active MBT while scheduler is not idle")
    capacity = _engine_mbt_capacity(scheduler)
    if not 1 <= new_mbt <= capacity:
        raise ValueError(f"active MBT {new_mbt} must be in [1, capacity={capacity}]")
    old = int(scheduler.max_num_scheduled_tokens)
    scheduler.max_num_scheduled_tokens = int(new_mbt)
    return old, int(scheduler.max_num_scheduled_tokens)


# ---------------------------------------------------------------------------
# Measurement and memory correction
# ---------------------------------------------------------------------------


def _measure_kv_pool(model: Any) -> tuple[int, int]:
    """Return (pool_blocks, KV bytes on this rank) for the live engine."""
    import torch

    def collect(inner: Any) -> int:
        total = 0
        for module in inner.modules():
            kv = getattr(module, "kv_cache", None)
            if kv is None:
                continue
            tensors = kv if isinstance(kv, (list, tuple)) else [kv]
            for tensor in tensors:
                if torch.is_tensor(tensor):
                    total += tensor.numel() * tensor.element_size()
        return total

    kv_bytes = int(model.llm.apply_model(collect)[0])
    scheduler = _resolve_scheduler(model)
    blocks = int(scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
    return blocks, kv_bytes


def _measure_case(
    *,
    model: Any,
    case: Case,
    tokens: Any,
    hook_name: str,
    stop_at_layer: int,
    warmup: int,
    switch_warmup: int,
    repeats: int,
    did_switch: bool,
) -> dict[str, Any]:
    import torch

    def run() -> Any:
        _, cache = model.run_with_cache(
            tokens,
            names_filter=[hook_name],
            stop_at_layer=stop_at_layer,
            prepend_bos=False,
        )
        return cache

    for _ in range(warmup + (switch_warmup if did_switch else 0)):
        _sync_all()
        cache = run()
        _sync_all()
        del cache

    times_ms: list[float] = []
    baseline_allocated: list[float] = []
    baseline_reserved: list[float] = []
    peak_allocated: list[float] = []
    peak_reserved: list[float] = []
    activation_mib = float("nan")
    activation_shape = ""

    for _ in range(repeats):
        gc.collect()
        _sync_all()
        base_alloc = torch.cuda.memory_allocated() / (1024**2)
        base_res = torch.cuda.memory_reserved() / (1024**2)
        torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        cache = run()
        _sync_all()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        act = cache[hook_name]
        if tuple(act.shape[:2]) != (case.B, case.context_size):
            raise RuntimeError(
                f"{case.name}: activation shape {tuple(act.shape)} does not "
                f"match expected leading dims {(case.B, case.context_size)}"
            )
        activation_mib = act.numel() * act.element_size() / (1024**2)
        activation_shape = "x".join(map(str, act.shape))

        times_ms.append(_all_reduce_max_float(elapsed_ms))
        baseline_allocated.append(_all_reduce_max_float(base_alloc))
        baseline_reserved.append(_all_reduce_max_float(base_res))
        peak_allocated.append(
            _all_reduce_max_float(torch.cuda.max_memory_allocated() / (1024**2))
        )
        peak_reserved.append(
            _all_reduce_max_float(torch.cuda.max_memory_reserved() / (1024**2))
        )
        del cache, act

    median_ms = statistics.median(times_ms)
    return {
        "status": "ok",
        "repeats": repeats,
        "wall_ms_median": median_ms,
        "wall_ms_mean": statistics.mean(times_ms),
        "wall_ms_min": min(times_ms),
        "wall_ms_max": max(times_ms),
        "wall_ms_cv": (
            statistics.pstdev(times_ms) / statistics.mean(times_ms)
            if len(times_ms) > 1 and statistics.mean(times_ms) > 0
            else 0.0
        ),
        "tokens_per_s": case.total_tokens / (median_ms / 1000.0),
        "ms_per_1k_tokens": median_ms / case.total_tokens * 1000.0,
        "naive_iterations": math.ceil(case.total_tokens / case.mbt),
        "mbt_fraction_of_N": case.mbt / case.total_tokens,
        "activation_mib": activation_mib,
        "activation_shape": activation_shape,
        "measured_baseline_allocated_mib": statistics.median(baseline_allocated),
        "measured_baseline_reserved_mib": statistics.median(baseline_reserved),
        "measured_peak_allocated_mib": statistics.median(peak_allocated),
        "measured_peak_reserved_mib": statistics.median(peak_reserved),
    }


def calibrate_capacity_buffer_mib_per_token(
    *, probe_mbt: Sequence[int], probe_baselines_mib: Sequence[float]
) -> float:
    """Least-squares slope of persistent allocated memory vs MBT capacity."""
    if len(probe_mbt) != len(probe_baselines_mib):
        raise ValueError("probe_mbt and probe_baselines_mib must have equal length")
    if len(probe_mbt) < 2:
        return 0.0
    mean_x = sum(probe_mbt) / len(probe_mbt)
    mean_y = sum(probe_baselines_mib) / len(probe_baselines_mib)
    covariance = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(probe_mbt, probe_baselines_mib)
    )
    variance = sum((x - mean_x) ** 2 for x in probe_mbt)
    return covariance / variance if variance > 0 else 0.0


def _worker(task_file: Path, result_file: Path) -> int:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from sae_lens.vllm_model import HookedVLLMModel

    task = json.loads(task_file.read_text())
    tp = int(task["tp"])
    context_size = int(task["context_size"])
    block_size = int(task["block_size"])
    mbt_capacity = int(task["mbt_capacity"])
    pool_capacity_tokens = int(task["pool_capacity_tokens"])
    cases = [Case(**entry) for entry in task["cases"]]

    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    if _world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = _rank()

    tokenizer = AutoTokenizer.from_pretrained(task["model_name"])
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[task["dtype"]]

    if pool_capacity_tokens % context_size != 0:
        raise ValueError("pool_capacity_tokens must be a multiple of context_size")

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": tp,
        "max_model_len": context_size + 1,
        "max_num_batched_tokens": mbt_capacity,
        "block_size": block_size,
        "gpu_memory_utilization": float(task["gpu_memory_utilization"]),
        "enable_chunked_prefill": True,
        "capture_batch_size": pool_capacity_tokens // context_size,
        "capture_context_size": context_size,
    }
    if tp > 1:
        llm_kwargs["distributed_executor_backend"] = "external_launcher"
        llm_kwargs["device"] = f"cuda:{local_rank}"

    model = HookedVLLMModel(task["model_name"], tokenizer, dtype=dtype, **llm_kwargs)
    hook_name = task["hook_name"]
    stop_at_layer = int(task["stop_at_layer"])
    vocab_size = int(getattr(tokenizer, "vocab_size", 128256) or 128256)

    scheduler = _resolve_scheduler(model)
    actual_capacity = _engine_mbt_capacity(scheduler)
    if actual_capacity != mbt_capacity:
        raise RuntimeError(
            f"Engine MBT capacity mismatch: expected {mbt_capacity}, got {actual_capacity}"
        )

    group_pool_blocks, group_kv_bytes = _measure_kv_pool(model)
    bytes_per_block = group_kv_bytes / group_pool_blocks if group_pool_blocks else 0.0
    group_pool_mib = group_kv_bytes / (1024**2)
    capacity_slope = float(task.get("capacity_buffer_mib_per_token") or 0.0)

    if rank == 0:
        print(
            f"[ENGINE] TP={tp} capacity={mbt_capacity} "
            f"active={scheduler.max_num_scheduled_tokens} "
            f"pool_blocks={group_pool_blocks} pool={group_pool_mib:.1f}MiB "
            f"capacity_slope={capacity_slope:.8f}MiB/token",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        _sync_all()
        switch_started = time.perf_counter()
        old_mbt, active_mbt = _set_active_mbt(model, case.mbt)
        _sync_all()
        switch_ms = _all_reduce_max_float(
            (time.perf_counter() - switch_started) * 1000.0
        )
        did_switch = old_mbt != active_mbt

        tokens = _make_tokens(case.B, case.context_size, vocab_size)
        row = _measure_case(
            model=model,
            case=case,
            tokens=tokens,
            hook_name=hook_name,
            stop_at_layer=stop_at_layer,
            warmup=int(task["warmup"]),
            switch_warmup=int(task["switch_warmup"]),
            repeats=int(task["repeats"]),
            did_switch=did_switch,
        )
        del tokens
        gc.collect()

        if rank == 0:
            single_blocks = _pool_blocks_for_tokens(case.total_tokens, block_size)
            single_pool_mib = single_blocks * bytes_per_block / (1024**2)
            pool_delta_mib = single_pool_mib - group_pool_mib
            # Standalone case would build persistent runner buffers at case.mbt.
            capacity_delta_mib = capacity_slope * (case.mbt - mbt_capacity)
            total_delta_mib = pool_delta_mib + capacity_delta_mib
            row.update(
                {
                    "script_version": SCRIPT_VERSION,
                    "tp": tp,
                    "stage": case.stage,
                    "case_name": case.name,
                    "case_signature": case.signature,
                    "B": case.B,
                    "context_size": case.context_size,
                    "total_tokens": case.total_tokens,
                    "mbt": case.mbt,
                    "kind": case.kind,
                    "mbt_capacity": mbt_capacity,
                    "active_mbt_before": old_mbt,
                    "active_mbt_after": active_mbt,
                    "budget_switch_wall_ms": switch_ms,
                    "block_size": block_size,
                    "group_pool_capacity_tokens": pool_capacity_tokens,
                    "group_pool_blocks": group_pool_blocks,
                    "group_pool_mib": group_pool_mib,
                    "kv_bytes_per_block": bytes_per_block,
                    "single_launch_pool_blocks": single_blocks,
                    "single_launch_pool_mib": single_pool_mib,
                    "pool_correction_mib": pool_delta_mib,
                    "capacity_buffer_mib_per_token": capacity_slope,
                    "capacity_correction_mib": capacity_delta_mib,
                    "total_correction_mib": total_delta_mib,
                    "corrected_baseline_allocated_mib": (
                        row["measured_baseline_allocated_mib"] + total_delta_mib
                    ),
                    "corrected_peak_allocated_mib": (
                        row["measured_peak_allocated_mib"] + total_delta_mib
                    ),
                    "memory_scope": (
                        "fixed_large_engine_not_minimal_config"
                        if case.stage == "stage_a_saturation"
                        else "reused_engine_with_standalone_allocated_estimate"
                    ),
                }
            )
            rows.append(row)
            print(
                f"[RESULT] {case.name} tok/s={row['tokens_per_s']:.1f} "
                f"raw_peak={row['measured_peak_allocated_mib']:.1f}MiB "
                f"corrected_peak={row['corrected_peak_allocated_mib']:.1f}MiB",
                flush=True,
            )
            _atomic_write_json(
                result_file,
                {
                    "status": "partial",
                    "task_hash": task["task_hash"],
                    "task": task,
                    "rows": rows,
                },
            )

    if rank == 0:
        _atomic_write_json(
            result_file,
            {
                "status": "ok",
                "task_hash": task["task_hash"],
                "task": task,
                "rows": rows,
            },
        )
    return 0


# ---------------------------------------------------------------------------
# Controller subprocess groups / calibration
# ---------------------------------------------------------------------------


def _run_group(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    task_name: str,
    tp: int,
    cases: Sequence[Case],
    mbt_capacity_override: int | None = None,
    pool_capacity_override: int | None = None,
    capacity_slope: float = 0.0,
) -> list[dict[str, Any]]:
    if not cases:
        return []

    mbt_capacity = (
        int(mbt_capacity_override)
        if mbt_capacity_override is not None
        else max(case.mbt for case in cases)
    )
    if any(case.mbt > mbt_capacity for case in cases):
        raise ValueError(f"{task_name}: a case MBT exceeds capacity {mbt_capacity}")

    pool_capacity_tokens = (
        int(pool_capacity_override)
        if pool_capacity_override is not None
        else max(case.total_tokens for case in cases)
    )
    if pool_capacity_tokens % args.context_size != 0:
        raise ValueError(
            f"{task_name}: pool capacity {pool_capacity_tokens} is not a multiple "
            f"of context_size {args.context_size}"
        )
    if any(case.total_tokens > pool_capacity_tokens for case in cases):
        raise ValueError(f"{task_name}: a case workload exceeds pool capacity")

    task_base = {
        "script_version": SCRIPT_VERSION,
        "task_name": task_name,
        "tp": tp,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "hook_name": args.hook_name,
        "stop_at_layer": args.stop_at_layer,
        "context_size": args.context_size,
        "block_size": args.block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup": args.warmup,
        "switch_warmup": args.switch_warmup,
        "repeats": args.repeats,
        "mbt_capacity": mbt_capacity,
        "pool_capacity_tokens": pool_capacity_tokens,
        "capacity_buffer_mib_per_token": capacity_slope,
        "cases": [asdict(case) for case in cases],
    }
    task_hash = _canonical_hash(task_base)
    task = {**task_base, "task_hash": task_hash}

    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_file = task_dir / f"{task_name}.{task_hash}.task.json"
    result_file = task_dir / f"{task_name}.{task_hash}.result.json"
    log_file = task_dir / f"{task_name}.{task_hash}.log"
    _atomic_write_json(task_file, task)

    if args.resume and result_file.exists():
        payload = json.loads(result_file.read_text())
        if payload.get("status") == "ok" and payload.get("task_hash") == task_hash:
            print(f"[SKIP] {task_name}: exact task hash {task_hash}", flush=True)
            return list(payload.get("rows", []))

    if tp > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={tp}",
            os.path.abspath(__file__),
        ]
    else:
        command = [sys.executable, os.path.abspath(__file__)]
    command += [
        "--worker",
        "--task-file",
        str(task_file),
        "--result-file",
        str(result_file),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["VLLM_ACTIVATION_CAPTURE_MODE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if tp == 1:
        for key in _TORCHRUN_ENV_KEYS:
            env.pop(key, None)

    print(
        f"[RUN] {task_name}: TP={tp}, cases={len(cases)}, "
        f"mbt_capacity={mbt_capacity}, pool_capacity={pool_capacity_tokens}, "
        f"hash={task_hash}",
        flush=True,
    )
    started = time.perf_counter()
    with log_file.open("w") as handle:
        completed = subprocess.run(
            command, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0 or not result_file.exists():
        print(
            f"[FAIL] {task_name}: exit={completed.returncode}, see {log_file}",
            flush=True,
        )
        if args.fail_fast:
            raise RuntimeError(f"{task_name} failed; see {log_file}")
        return []

    payload = json.loads(result_file.read_text())
    rows = list(payload.get("rows", []))
    print(f"[OK] {task_name}: {elapsed:.1f}s, {len(rows)} row(s)", flush=True)
    return rows


def _calibrate_capacity_slope(
    *, args: argparse.Namespace, output_dir: Path, tp: int
) -> tuple[float, list[dict[str, Any]]]:
    if args.capacity_slope_mib_per_token is not None:
        slope = float(args.capacity_slope_mib_per_token)
        print(f"TP={tp}: using supplied capacity slope {slope:.8f} MiB/token")
        return slope, []

    probe_B = min(args.batch_values)
    probe_tokens = probe_B * args.context_size
    capacities = sorted({max(probe_tokens, cap) for cap in args.calibrate_mbt})
    rows: list[dict[str, Any]] = []
    for capacity in capacities:
        case = Case(
            stage="calibration",
            tp=tp,
            B=probe_B,
            context_size=args.context_size,
            total_tokens=probe_tokens,
            mbt=probe_tokens,
            kind=f"capacity_probe_{capacity}",
        )
        probe_rows = _run_group(
            args=args,
            output_dir=output_dir,
            task_name=f"calib_tp{tp}_S{args.context_size}_cap{capacity}",
            tp=tp,
            cases=[case],
            mbt_capacity_override=capacity,
            pool_capacity_override=probe_tokens,
            capacity_slope=0.0,
        )
        for row in probe_rows:
            row["calibration_capacity"] = capacity
        rows.extend(probe_rows)

    usable = [row for row in rows if row.get("status") == "ok"]
    if len(usable) < 2:
        print(
            "[WARN] capacity calibration needs >=2 successful probes; "
            "capacity correction disabled",
            flush=True,
        )
        return 0.0, rows

    slope = calibrate_capacity_buffer_mib_per_token(
        probe_mbt=[int(row["calibration_capacity"]) for row in usable],
        probe_baselines_mib=[
            float(row["measured_baseline_allocated_mib"]) for row in usable
        ],
    )
    print(
        f"TP={tp}: capacity-buffer slope={slope:.8f} MiB/token "
        f"({slope * 1024:.3f} KiB/token) from {len(usable)} probes",
        flush=True,
    )
    return slope, rows


# ---------------------------------------------------------------------------
# Pareto annotation
# ---------------------------------------------------------------------------


def _annotate_pareto(
    rows: list[dict[str, Any]], *, epsilon: float
) -> list[dict[str, Any]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    for row in rows:
        row["is_strict_pareto"] = False
        row["is_epsilon_pareto"] = False
        row["throughput_epsilon"] = epsilon

    for a in ok:
        ma = float(a["corrected_peak_allocated_mib"])
        ra = float(a["tokens_per_s"])
        strict_dominated = False
        eps_dominated = False
        for b in ok:
            if a is b:
                continue
            mb = float(b["corrected_peak_allocated_mib"])
            rb = float(b["tokens_per_s"])
            if mb <= ma and rb >= ra and (mb < ma or rb > ra):
                strict_dominated = True
            if mb < ma and rb * (1.0 + epsilon) >= ra:
                eps_dominated = True
        a["is_strict_pareto"] = not strict_dominated
        a["is_epsilon_pareto"] = not eps_dominated
    return rows


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------


def _print_plan(args: argparse.Namespace) -> None:
    print("=== vLLM two-stage v4.1 plan ===")
    print(f"model                 : {args.model_name}")
    print(f"dtype                 : {args.dtype}")
    print(f"hook / stop_at_layer  : {args.hook_name} / {args.stop_at_layer}")
    print(f"context_size          : {args.context_size}")
    print(f"TP values             : {args.tp_values}")
    print(f"Stage-A B values      : {sorted(args.batch_values)}")
    print(f"Stage-A fixed MBT     : {args.stage_a_mbt}")
    print(f"saturation fraction   : {args.saturation_fraction}")
    print(f"exact B ratios        : {args.exact_b_ratios}")
    print(f"chunked B ratios      : {args.chunked_b_ratios}")
    print(f"chunk divisors        : {args.chunk_divisors}")
    print(f"Stage-B min MBT       : {args.min_mbt}")
    print(f"capacity calibration  : {args.calibrate_capacity}")
    print(f"calibration capacities: {args.calibrate_mbt}")
    print(f"supplied slope        : {args.capacity_slope_mib_per_token}")
    print(f"warmup / repeats      : {args.warmup} / {args.repeats}")
    print(f"switch warmup         : {args.switch_warmup}")
    print(f"epsilon Pareto        : {args.epsilon_pareto}")
    print(f"CUDA_VISIBLE_DEVICES  : {args.cuda_devices}")
    print(f"output directory      : {args.output_dir}")
    print("Stage B uses scheduler soft-budget switching; runner capacity is immutable.")


def _controller(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_plan(args)

    for tp in args.tp_values:
        cases = build_stage_a_cases(tp=tp, args=args)
        print(f"TP={tp} Stage-A cases ({len(cases)}):")
        for case in cases:
            print(f"  B={case.B:4d} N={case.total_tokens:7d} active_MBT={case.mbt}")
    if args.dry_run:
        print(
            "Stage-B is derived after L_m is measured. Example for B_s=16: "
            "B4:M4096; B8:M8192/4096/2048; "
            "B16:M16384/8192/4096; B32:M32768."
        )
        return 0

    stage_a_rows: list[dict[str, Any]] = []
    stage_b_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    slopes: dict[str, float] = {}

    for tp in args.tp_values:
        if args.calibrate_capacity:
            slope, probe_rows = _calibrate_capacity_slope(
                args=args, output_dir=output_dir, tp=tp
            )
            calibration_rows.extend(probe_rows)
        else:
            slope = float(args.capacity_slope_mib_per_token or 0.0)
            print(
                f"TP={tp}: calibration disabled; capacity slope={slope:.8f}",
                flush=True,
            )
        slopes[str(tp)] = slope

        a_cases = build_stage_a_cases(tp=tp, args=args)
        a_rows = _run_group(
            args=args,
            output_dir=output_dir,
            task_name=f"stage_a_tp{tp}_S{args.context_size}_M{args.stage_a_mbt}",
            tp=tp,
            cases=a_cases,
            mbt_capacity_override=args.stage_a_mbt,
            pool_capacity_override=max(case.total_tokens for case in a_cases),
            capacity_slope=slope,
        )
        stage_a_rows.extend(a_rows)
        if not a_rows:
            print(f"[WARN] TP={tp}: Stage-A produced no rows; skipping Stage-B")
            continue

        selection = select_saturation(a_rows, fraction=args.saturation_fraction)
        selections[str(tp)] = selection
        print(
            f"TP={tp}: observed max={selection['max_tokens_per_s']:.1f} tok/s "
            f"at B={selection['max_B']}; threshold={selection['threshold_tokens_per_s']:.1f}; "
            f"selected B_s={selection['selected_B']}, L_m={selection['L_m_tokens']}",
            flush=True,
        )
        if selection["boundary_peak"]:
            growth = selection["boundary_growth_fraction"]
            growth_text = "unknown" if growth is None else f"{100*growth:.2f}%"
            print(
                f"[WARN] TP={tp}: observed throughput maximum is at the largest "
                f"Stage-A workload (growth vs previous={growth_text}). Increase "
                "--stage-a-mbt/--batch-values if you need proof that the plateau "
                "has been bracketed.",
                flush=True,
            )

        b_cases = build_stage_b_cases(
            tp=tp, saturation=selection, args=args
        )
        print(f"TP={tp} Stage-B cases ({len(b_cases)}):")
        for case in b_cases:
            print(
                f"  B={case.B:4d} N={case.total_tokens:7d} "
                f"MBT={case.mbt:7d} iters~{math.ceil(case.total_tokens/case.mbt):2d} "
                f"{case.kind}"
            )

        b_rows = _run_group(
            args=args,
            output_dir=output_dir,
            task_name=f"stage_b_tp{tp}_S{args.context_size}_Lm{selection['L_m_tokens']}",
            tp=tp,
            cases=b_cases,
            capacity_slope=slope,
        )
        for row in b_rows:
            row["selected_saturation_B"] = selection["selected_B"]
            row["L_m_tokens"] = selection["L_m_tokens"]
            row["saturation_max_tokens_per_s"] = selection["max_tokens_per_s"]
            row["throughput_fraction_of_saturation_max"] = (
                float(row["tokens_per_s"]) / float(selection["max_tokens_per_s"])
            )
            row["meets_saturation_threshold"] = (
                float(row["tokens_per_s"])
                >= float(selection["threshold_tokens_per_s"])
            )
            row["B_ratio_to_saturation"] = (
                float(row["B"]) / float(selection["selected_B"])
            )
            row["N_ratio_to_L_m"] = (
                float(row["total_tokens"]) / float(selection["L_m_tokens"])
            )
        _annotate_pareto(b_rows, epsilon=args.epsilon_pareto)
        stage_b_rows.extend(b_rows)

    _write_csv(output_dir / "stage_a_saturation.csv", stage_a_rows)
    _write_csv(output_dir / "stage_b_frontier.csv", stage_b_rows)
    if calibration_rows:
        _write_csv(output_dir / "capacity_calibration.csv", calibration_rows)
    _write_csv(output_dir / "all_rows.csv", stage_a_rows + stage_b_rows)

    strict = [row for row in stage_b_rows if row.get("is_strict_pareto")]
    eps = [row for row in stage_b_rows if row.get("is_epsilon_pareto")]
    _write_csv(output_dir / "strict_pareto.csv", strict)
    _write_csv(output_dir / "epsilon_pareto.csv", eps)

    _atomic_write_json(
        output_dir / "run_metadata.json",
        {
            "script_version": SCRIPT_VERSION,
            "args": {
                key: value
                for key, value in vars(args).items()
                if key not in {"worker", "task_file", "result_file"}
            },
            "saturation_selection": selections,
            "capacity_buffer_mib_per_token": slopes,
            "strict_pareto_count": len(strict),
            "epsilon_pareto_count": len(eps),
        },
    )
    print(
        f"Wrote {len(stage_a_rows)} Stage-A rows, {len(stage_b_rows)} Stage-B rows; "
        f"strict Pareto={len(strict)}, epsilon Pareto={len(eps)} to {output_dir}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dtype", default=DEFAULT_DTYPE, choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--hook-name", default=DEFAULT_HOOK_NAME)
    parser.add_argument("--stop-at-layer", type=int, default=DEFAULT_STOP_AT_LAYER)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--tp-values", default=DEFAULT_TP_VALUES)
    parser.add_argument("--batch-values", default=DEFAULT_BATCH_VALUES)
    parser.add_argument(
        "--stage-a-mbt",
        type=int,
        default=DEFAULT_STAGE_A_MBT,
        help="Immutable MBT capacity and active budget used by the Stage-A engine.",
    )
    parser.add_argument(
        "--saturation-fraction", type=float, default=DEFAULT_SATURATION_FRACTION
    )
    parser.add_argument(
        "--exact-b-ratios",
        default=DEFAULT_EXACT_B_RATIOS,
        help="Exact-fill Stage-B B/B_s ratios, default 0.25,0.5,1,2.",
    )
    parser.add_argument(
        "--chunked-b-ratios",
        default=DEFAULT_CHUNKED_B_RATIOS,
        help="B/B_s ratios that receive extra chunked MBT points.",
    )
    parser.add_argument(
        "--chunk-divisors",
        default=DEFAULT_CHUNK_DIVISORS,
        help="Extra Stage-B MBTs are N divided by these values.",
    )
    parser.add_argument("--min-mbt", type=int, default=1)

    parser.add_argument(
        "--no-calibrate-capacity",
        dest="calibrate_capacity",
        action="store_false",
        help="Skip MBT-capacity buffer calibration.",
    )
    parser.set_defaults(calibrate_capacity=True)
    parser.add_argument(
        "--calibrate-mbt",
        default="2048,16384,65536",
        help="Immutable MBT capacities used to fit persistent buffer slope.",
    )
    parser.add_argument(
        "--capacity-slope-mib-per-token",
        type=float,
        default=None,
        help="Use a prevalidated capacity slope instead of launching probes.",
    )

    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--switch-warmup",
        type=int,
        default=DEFAULT_SWITCH_WARMUP,
        help="Extra discarded runs after changing scheduler soft MBT.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--epsilon-pareto", type=float, default=DEFAULT_EPSILON_PARETO)
    parser.add_argument("--cuda-devices", default=DEFAULT_CUDA_DEVICES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fail-fast", action="store_true")

    args = parser.parse_args(argv)
    if args.worker:
        return args

    try:
        args.tp_values = _parse_int_csv(args.tp_values, name="--tp-values")
        args.batch_values = _parse_int_csv(args.batch_values, name="--batch-values")
        args.exact_b_ratios = _parse_float_csv(
            args.exact_b_ratios, name="--exact-b-ratios"
        )
        args.chunked_b_ratios = _parse_float_csv(
            args.chunked_b_ratios, name="--chunked-b-ratios"
        )
        args.chunk_divisors = _parse_int_csv(
            args.chunk_divisors, name="--chunk-divisors"
        )
        args.calibrate_mbt = _parse_int_csv(args.calibrate_mbt, name="--calibrate-mbt")
    except ValueError as exc:
        parser.error(str(exc))

    if args.context_size <= 0:
        parser.error("--context-size must be positive")
    if args.stage_a_mbt < args.context_size:
        parser.error("--stage-a-mbt must be >= --context-size")
    if not 0.0 < args.saturation_fraction <= 1.0:
        parser.error("--saturation-fraction must be in (0, 1]")
    if args.repeats < 1 or args.warmup < 0 or args.switch_warmup < 0:
        parser.error("--repeats must be >=1; warmup values must be >=0")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if not 0.0 <= args.epsilon_pareto < 1.0:
        parser.error("--epsilon-pareto must be in [0,1)")
    if args.capacity_slope_mib_per_token is not None:
        args.calibrate_capacity = False
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        if not args.task_file or not args.result_file:
            raise SystemExit("--worker requires --task-file and --result-file")
        return _worker(Path(args.task_file), Path(args.result_file))
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
