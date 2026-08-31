#!/usr/bin/env python3
"""
v3.2: focused vLLM activation-capture saturation scan for ctx=2048.

This is a deliberately narrow derivative of explore_vllm_activation_profile_v3.py.
It keeps the original worker isolation, timing, memory accounting, resume logic,
CSV/JSON output, and plotting code, but changes the experiment grid to answer one
question cleanly on the current A40 setup:

    with context_size fixed at 2048, how do larger Batch and MBT values move
    throughput toward / across the saturation plateau?

Default scan
------------
context_size:
    2048 only

Batch:
    4, 8, 16, 32, 64

max_num_batched_tokens (MBT):
    8192, 16384, 32768, 65536

fixed batch-fill KV-pool capacity:
    131072 tokens

This gives 20 measured cases per TP and 40 cases with the default TP={1,2}.

Why these limits
----------------
* ctx=2048 is safely below the script's default max_model_len=4097.
* MBT is an aggregate batched-token scheduler capacity, not a per-sequence
  context length.  Therefore MBT can be >4097 while every individual sequence
  remains length 2048.
* B is capped at 64 by default, well below the common vLLM max_num_seqs=256.
* B*ctx is capped at 131072 tokens, so B=64 is the largest default case.
* MBT=65536 with B=64 reaches 2x MBT fill; smaller MBTs reach still larger
  overfill ratios.  This extends beyond the old ctx=2048 scan (which stopped at
  B=32 and MBT=16384) without jumping to B=128 / pool=262144, which is a much
  more aggressive memory regime for a single A40.

Typical commands
----------------
Inspect the exact plan without loading the model:

    python sae_lens/autoconfig/explore_vllm_activation_profile_v3_2.py --dry-run

Run both TP1 and TP2 (default):

    python sae_lens/autoconfig/explore_vllm_activation_profile_v3_2.py

Run only one A40 / TP1:

    python sae_lens/autoconfig/explore_vllm_activation_profile_v3_2.py \
      --tp-values 1

Run only TP2 on the two-A40 machine:

    python sae_lens/autoconfig/explore_vllm_activation_profile_v3_2.py \
      --tp-values 2

Outputs
-------
sae_lens/autoconfig/profile_results/
    vllm_activation_explore_v3_2.csv
    vllm_activation_explore_v3_2.json
    vllm_activation_explore_v3_2_tasks/
    vllm_activation_explore_v3_2_plots/
"""


from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_VERSION = "v3.2"

# Defaults pinned to the single operating point used by
# scripts/run_step_window_sweep.py: context 2048 (max_model_len 2049), float32,
# hook blocks.21.hook_resid_post (stop_at_layer 22).
DEFAULT_MODEL = "/data/models/Llama-3.1-8B"
DEFAULT_STOP_AT_LAYER = 22
DEFAULT_HOOK_NAME = "blocks.21.hook_resid_post"
# The runner's --dtype is the activation-store dtype and is never forwarded to
# vLLM, so the engine runs at HookedVLLMModel's bfloat16 default.
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5
DEFAULT_MAX_MODEL_LEN = 2049
DEFAULT_GPU_MEMORY_UTILIZATION = 0.50
DEFAULT_MAX_BATCH = 1
DEFAULT_MAX_BATCH_TOTAL_TOKENS = 4096
DEFAULT_CAPTURE_CONTEXT_FOR_POOL = 2048


@dataclass(frozen=True)
class ScaleSpec:
    contexts: tuple[int, ...]
    batch_mbts: tuple[int, ...]
    fill_ratios: tuple[float, ...]
    tradeoff_total_tokens: tuple[int, ...]
    tradeoff_mbts: tuple[int, ...]


SCALE_SPECS: dict[str, ScaleSpec] = {
    "a40_ctx2048": ScaleSpec(
        contexts=(2048,),
        batch_mbts=(8192, 16384, 32768, 65536),
        # v3.2 uses an explicit Batch list in _build_batch_fill_cases().
        fill_ratios=(),
        # No separate fixed-N tradeoff family in this focused scan.
        tradeoff_total_tokens=(),
        tradeoff_mbts=(),
    ),
}

# Exact default Batch values for the focused saturation sweep.
SATURATION_BATCH_VALUES: tuple[int, ...] = (4, 8, 16, 32, 64)



@dataclass(frozen=True)
class Case:
    case_name: str
    family: str
    experiment_id: str
    B: int
    context_size: int
    mbt: int
    kv_pool_tokens: int
    target_total_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.B * self.context_size

    @property
    def causal_pairs(self) -> int:
        return self.B * self.context_size * (self.context_size + 1) // 2

    @property
    def naive_iterations(self) -> int:
        return math.ceil(self.total_tokens / self.mbt)

    @property
    def full_mbt_iterations(self) -> int:
        return self.total_tokens // self.mbt

    @property
    def remainder_tokens(self) -> int:
        return self.total_tokens % self.mbt

    @property
    def total_tokens_over_mbt(self) -> float:
        return self.total_tokens / self.mbt

    @property
    def mbt_over_context(self) -> float:
        return self.mbt / self.context_size

    @property
    def fill_batch(self) -> float:
        return self.mbt / self.context_size

    @property
    def batch_over_fill_batch(self) -> float:
        return self.B / self.fill_batch

    @property
    def workload_regime(self) -> str:
        if self.total_tokens < self.mbt:
            return "underfilled_N_lt_MBT"
        if self.total_tokens == self.mbt:
            return "one_full_N_eq_MBT"
        return "multi_iter_N_gt_MBT"

    @property
    def chunk_regime(self) -> str:
        if self.mbt < self.context_size:
            return "intra_sequence_MBT_lt_ctx"
        if self.mbt == self.context_size:
            return "one_sequence_MBT_eq_ctx"
        return "multi_sequence_MBT_gt_ctx"

    @property
    def signature(self) -> str:
        return (
            f"{self.family}|{self.experiment_id}|B{self.B}|S{self.context_size}|"
            f"M{self.mbt}|P{self.kv_pool_tokens}"
        )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _nearest_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    exponent = round(math.log2(value))
    return 2**exponent


def _build_batch_fill_cases(
    spec: ScaleSpec,
    max_batch: int,
    max_total_tokens: int,
) -> list[Case]:
    """Build the focused ctx=2048 saturation grid.

    v3.2 intentionally uses explicit Batch values rather than deriving them from
    B_fill.  This makes the experiment easy to read and guarantees that each MBT
    is compared at the same B values.

    Safety / validity constraints:
      * one sequence is always ctx=2048;
      * B <= --max-batch (default 64);
      * B*ctx <= fixed pool capacity (default 131072);
      * MBT <= fixed pool capacity;
      * B is power-of-two.
    """
    cases: list[Case] = []
    pool_tokens = max_total_tokens

    if pool_tokens % DEFAULT_CAPTURE_CONTEXT_FOR_POOL != 0:
        raise ValueError(
            "--max-batch-total-tokens must be divisible by "
            f"{DEFAULT_CAPTURE_CONTEXT_FOR_POOL}"
        )

    for ctx in spec.contexts:
        for mbt in spec.batch_mbts:
            if mbt < ctx:
                continue
            if mbt > pool_tokens:
                continue
            if mbt % ctx != 0:
                continue

            valid_batches = [
                B
                for B in SATURATION_BATCH_VALUES
                if _is_power_of_two(B)
                and 1 <= B <= max_batch
                and B * ctx <= pool_tokens
            ]

            experiment_id = f"batch_ctx{ctx}_mbt{mbt}"
            for B in valid_batches:
                cases.append(
                    Case(
                        case_name=f"batch_ctx{ctx}_B{B}_M{mbt}_pool{pool_tokens}",
                        family="batch_fill",
                        experiment_id=experiment_id,
                        B=B,
                        context_size=ctx,
                        mbt=mbt,
                        kv_pool_tokens=pool_tokens,
                    )
                )
    return cases


def _build_mbt_tradeoff_cases(
    spec: ScaleSpec,
    max_batch: int,
) -> list[Case]:
    """Build fixed-N MBT sweeps for direct time-memory tradeoff plots."""
    cases: list[Case] = []

    for total_tokens in spec.tradeoff_total_tokens:
        if total_tokens % DEFAULT_CAPTURE_CONTEXT_FOR_POOL != 0:
            raise ValueError(
                f"tradeoff N={total_tokens} must be divisible by "
                f"{DEFAULT_CAPTURE_CONTEXT_FOR_POOL}"
            )

        for ctx in spec.contexts:
            if total_tokens % ctx != 0:
                continue
            B = total_tokens // ctx
            if B < 1 or B > max_batch or not _is_power_of_two(B):
                continue

            experiment_id = f"mbt_tradeoff_N{total_tokens}_ctx{ctx}"
            for mbt in spec.tradeoff_mbts:
                cases.append(
                    Case(
                        case_name=(
                            f"tradeoff_N{total_tokens}_B{B}_S{ctx}_M{mbt}_"
                            f"pool{total_tokens}"
                        ),
                        family="mbt_tradeoff",
                        experiment_id=experiment_id,
                        B=B,
                        context_size=ctx,
                        mbt=mbt,
                        kv_pool_tokens=total_tokens,
                        target_total_tokens=total_tokens,
                    )
                )
    return cases


def _build_runmatch_cases(max_total_tokens: int) -> list[Case]:
    """One case reproducing a training run's own activation-generation call.

    The training loop feeds the mixing buffer from
    `store_batch_size_prompts * context_size` tokens per vLLM invocation, so a
    run with `store_batch_size_prompts=1, context_size=2048` generates 2048
    tokens per call. `mbt` is left at the runner's default rather than raised:
    mbt only bounds how many chunks one prefill is split into, and changing it
    would measure a different call than the run makes.
    """
    ctx = DEFAULT_CAPTURE_CONTEXT_FOR_POOL
    return [
        Case(
            case_name=f"runmatch_ctx{ctx}_B1_M{DEFAULT_MAX_BATCH_TOTAL_TOKENS}",
            family="runmatch",
            experiment_id=f"runmatch_ctx{ctx}",
            B=1,
            context_size=ctx,
            mbt=DEFAULT_MAX_BATCH_TOTAL_TOKENS,
            kv_pool_tokens=max_total_tokens,
        )
    ]


def build_cases(
    scale: str,
    suite: str,
    max_batch: int,
    max_total_tokens: int,
) -> list[Case]:
    spec = SCALE_SPECS[scale]
    batch_cases = _build_batch_fill_cases(
        spec,
        max_batch=max_batch,
        max_total_tokens=max_total_tokens,
    )
    tradeoff_cases = _build_mbt_tradeoff_cases(spec, max_batch=max_batch)

    if suite == "batch":
        selected = batch_cases
    elif suite == "tradeoff":
        selected = tradeoff_cases
    elif suite == "runmatch":
        selected = _build_runmatch_cases(max_total_tokens)
    elif suite == "all":
        selected = batch_cases + tradeoff_cases
    else:
        raise ValueError(f"Unsupported suite={suite!r}")

    unique: dict[str, Case] = {}
    for case in selected:
        unique.setdefault(case.signature, case)
    return list(unique.values())


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ys = sorted(values)
    if len(ys) == 1:
        return ys[0]
    position = (len(ys) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ys[lo]
    alpha = position - lo
    return ys[lo] * (1.0 - alpha) + ys[hi] * alpha


def coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _init_distributed_if_needed() -> None:
    import torch
    import torch.distributed as dist

    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    if _world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


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
        [value],
        dtype=torch.float64,
        device=torch.device("cuda", _local_rank()),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _all_reduce_min_float(value: float) -> float:
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device=torch.device("cuda", _local_rank()),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return float(tensor.item())


def _make_tokens(B: int, S: int, vocab_size: int, seed: int = 1234):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + B * 100_003 + S * 101)
    low = 100
    high = max(low + 1, vocab_size - 256)
    tokens = torch.randint(
        low=low,
        high=high,
        size=(B, S),
        generator=generator,
        dtype=torch.long,
    )
    return tokens.to(torch.device("cuda", _local_rank()))


def _capture_pair_for_pool(kv_pool_tokens: int) -> tuple[int, int]:
    if kv_pool_tokens % DEFAULT_CAPTURE_CONTEXT_FOR_POOL != 0:
        raise ValueError(
            f"kv_pool_tokens={kv_pool_tokens} must be divisible by "
            f"{DEFAULT_CAPTURE_CONTEXT_FOR_POOL}"
        )
    return (
        kv_pool_tokens // DEFAULT_CAPTURE_CONTEXT_FOR_POOL,
        DEFAULT_CAPTURE_CONTEXT_FOR_POOL,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_existing_worker_rows(
    result_file: Path,
    expected_task_hash: str,
) -> list[dict[str, Any]]:
    if not result_file.exists():
        return []
    try:
        payload = json.loads(result_file.read_text())
    except Exception:
        return []
    if payload.get("task_hash") != expected_task_hash:
        return []
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _case_row_base(
    case: Case,
    task: dict[str, Any],
    gpu_name: str,
) -> dict[str, Any]:
    capture_batch_size = int(task["capture_batch_size"])
    capture_context_size = int(task["capture_context_size"])
    return {
        "case_signature": case.signature,
        "case_name": case.case_name,
        "family": case.family,
        "experiment_id": case.experiment_id,
        "script_version": SCRIPT_VERSION,
        "model": task["model_name"],
        "gpu": gpu_name,
        "dtype": task["dtype"],
        "tp": int(task["tp"]),
        "stop_at_layer": int(task["stop_at_layer"]),
        "hook_name": task["hook_name"],
        "B": case.B,
        "context_size": case.context_size,
        "mbt": case.mbt,
        "kv_pool_tokens": case.kv_pool_tokens,
        "capture_batch_size": capture_batch_size,
        "capture_context_size": capture_context_size,
        "target_total_tokens": case.target_total_tokens,
        "total_tokens": case.total_tokens,
        "causal_pairs": case.causal_pairs,
        "naive_iterations": case.naive_iterations,
        "full_mbt_iterations": case.full_mbt_iterations,
        "remainder_tokens": case.remainder_tokens,
        "total_tokens_over_mbt": case.total_tokens_over_mbt,
        "mbt_over_context": case.mbt_over_context,
        "fill_batch_mbt_over_ctx": case.fill_batch,
        "batch_over_fill_batch": case.batch_over_fill_batch,
        "workload_regime": case.workload_regime,
        "chunk_regime": case.chunk_regime,
        "warmup": int(task["warmup"]),
        "repeats": int(task["repeats"]),
    }


def _worker(task_file: Path, result_file: Path, no_resume: bool) -> int:
    task = json.loads(task_file.read_text())
    task_hash = str(task["task_hash"])

    tp = int(task["tp"])
    mbt = int(task["mbt"])
    kv_pool_tokens = int(task["kv_pool_tokens"])
    capture_batch_size = int(task["capture_batch_size"])
    capture_context_size = int(task["capture_context_size"])
    model_name = str(task["model_name"])
    stop_at_layer = int(task["stop_at_layer"])
    hook_name = str(task["hook_name"])
    dtype_name = str(task["dtype"])
    warmup = int(task["warmup"])
    repeats = int(task["repeats"])
    max_model_len = int(task["max_model_len"])
    gpu_memory_utilization = float(task["gpu_memory_utilization"])
    case_order = str(task["case_order"])
    order_seed = int(task["order_seed"])
    cases = [Case(**raw) for raw in task["cases"]]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    rank = _rank()
    rows: list[dict[str, Any]] = []
    if rank == 0 and not no_resume:
        rows = _load_existing_worker_rows(result_file, task_hash)
    completed_signatures = {
        str(row.get("case_signature"))
        for row in rows
        if row.get("status") in {"ok", "error"}
    }

    try:
        _init_distributed_if_needed()

        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer

        from sae_lens.vllm_model import HookedVLLMModel

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype={dtype_name!r}")
        dtype = dtype_map[dtype_name]

        if _world_size() != tp:
            raise RuntimeError(
                f"WORLD_SIZE={_world_size()} but task requests tp={tp}"
            )

        for case in cases:
            if case.mbt != mbt or case.kv_pool_tokens != kv_pool_tokens:
                raise RuntimeError(
                    f"Task grouping error for {case.case_name}: "
                    f"case MBT/pool={case.mbt}/{case.kv_pool_tokens}, "
                    f"task MBT/pool={mbt}/{kv_pool_tokens}"
                )
            if case.total_tokens > kv_pool_tokens:
                raise ValueError(
                    f"{case.case_name}: total_tokens={case.total_tokens} exceeds "
                    f"fixed task KV pool={kv_pool_tokens}"
                )
            if case.context_size + 1 > max_model_len:
                raise ValueError(
                    f"{case.case_name}: ctx={case.context_size} requires "
                    f"max_model_len>={case.context_size + 1}, got {max_model_len}"
                )

        device = f"cuda:{_local_rank()}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=Path(model_name).exists(),
        )

        model = HookedVLLMModel(
            model_name,
            tokenizer,
            dtype=dtype,
            capture_batch_size=capture_batch_size,
            capture_context_size=capture_context_size,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            max_num_batched_tokens=mbt,
            enable_chunked_prefill=True,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
        )

        vocab_size = int(getattr(tokenizer, "vocab_size", 128256) or 128256)
        gpu_name = torch.cuda.get_device_name(_local_rank())

        # Warm-up with up to 8192 tokens while respecting the fixed pool.
        warm_total = min(8192, kv_pool_tokens)
        warm_ctx = min(2048, max_model_len - 1, warm_total)
        while warm_total % warm_ctx != 0:
            warm_ctx //= 2
        warm_B = warm_total // warm_ctx
        warm_tokens = _make_tokens(warm_B, warm_ctx, vocab_size, seed=99)
        for _ in range(warmup):
            _sync_all()
            _, warm_cache = model.run_with_cache(
                warm_tokens,
                names_filter=[hook_name],
                stop_at_layer=stop_at_layer,
                prepend_bos=False,
            )
            _sync_all()
            warm_act = warm_cache[hook_name]
            if tuple(warm_act.shape[:2]) != (warm_B, warm_ctx):
                raise RuntimeError(
                    f"Warmup activation shape mismatch: {tuple(warm_act.shape)}"
                )
            del warm_cache, warm_act
        del warm_tokens
        gc.collect()
        _sync_all()

        ordered_cases = list(cases)
        if case_order == "random":
            rng = random.Random(order_seed + tp * 10_000 + mbt + kv_pool_tokens)
            rng.shuffle(ordered_cases)
        elif case_order == "descending":
            ordered_cases.sort(
                key=lambda c: (c.total_tokens, c.context_size, c.B),
                reverse=True,
            )
        else:
            ordered_cases.sort(
                key=lambda c: (c.total_tokens, c.context_size, c.B)
            )

        for execution_index, case in enumerate(ordered_cases):
            # All ranks must make the same skip decision.  completed_signatures
            # originates from rank 0's result file, so broadcast it implicitly by
            # checking only rank 0 and then reducing a scalar.
            local_skip = 1.0 if rank == 0 and case.signature in completed_signatures else 0.0
            skip_value = _all_reduce_max_float(local_skip)
            if skip_value > 0.5:
                if rank == 0:
                    print(f"[SKIP] {case.case_name}", flush=True)
                continue

            row_base = _case_row_base(case, task, gpu_name)
            row_base["execution_index"] = execution_index

            try:
                batch_tokens = _make_tokens(
                    case.B,
                    case.context_size,
                    vocab_size,
                )

                times_ms: list[float] = []
                baseline_allocated_mib: list[float] = []
                baseline_reserved_mib: list[float] = []
                peak_allocated_mib: list[float] = []
                peak_reserved_mib: list[float] = []
                peak_incremental_allocated_mib: list[float] = []
                peak_incremental_reserved_mib: list[float] = []
                activation_mib: float | None = None
                activation_shape: list[int] | None = None

                for repeat_index in range(repeats):
                    gc.collect()
                    _sync_all()

                    base_a = torch.cuda.memory_allocated() / (1024**2)
                    base_r = torch.cuda.memory_reserved() / (1024**2)
                    torch.cuda.reset_peak_memory_stats()

                    started = time.perf_counter()
                    _, cache = model.run_with_cache(
                        batch_tokens,
                        names_filter=[hook_name],
                        stop_at_layer=stop_at_layer,
                        prepend_bos=False,
                    )
                    _sync_all()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0

                    act = cache[hook_name]
                    if tuple(act.shape[:2]) != (
                        case.B,
                        case.context_size,
                    ):
                        raise RuntimeError(
                            f"{case.case_name}: activation shape mismatch: "
                            f"got {tuple(act.shape)}, expected leading dims "
                            f"{(case.B, case.context_size)}"
                        )

                    current_activation_mib = (
                        act.numel() * act.element_size() / (1024**2)
                    )
                    if activation_mib is None:
                        activation_mib = current_activation_mib
                        activation_shape = list(act.shape)

                    elapsed_ms = _all_reduce_max_float(elapsed_ms)
                    base_a = _all_reduce_max_float(base_a)
                    base_r = _all_reduce_max_float(base_r)
                    peak_a = _all_reduce_max_float(
                        torch.cuda.max_memory_allocated() / (1024**2)
                    )
                    peak_r = _all_reduce_max_float(
                        torch.cuda.max_memory_reserved() / (1024**2)
                    )

                    times_ms.append(elapsed_ms)
                    baseline_allocated_mib.append(base_a)
                    baseline_reserved_mib.append(base_r)
                    peak_allocated_mib.append(peak_a)
                    peak_reserved_mib.append(peak_r)
                    peak_incremental_allocated_mib.append(
                        max(0.0, peak_a - base_a)
                    )
                    peak_incremental_reserved_mib.append(
                        max(0.0, peak_r - base_r)
                    )

                    del cache, act

                if rank == 0:
                    median_ms = statistics.median(times_ms)
                    row = {
                        **row_base,
                        "status": "ok",
                        "error_type": "",
                        "error_message": "",
                        "wall_ms_median": median_ms,
                        "wall_ms_mean": statistics.mean(times_ms),
                        "wall_ms_min": min(times_ms),
                        "wall_ms_p10": percentile(times_ms, 0.10),
                        "wall_ms_p90": percentile(times_ms, 0.90),
                        "wall_ms_max": max(times_ms),
                        "wall_ms_cv": coefficient_of_variation(times_ms),
                        "tokens_per_s": case.total_tokens / (median_ms / 1000.0),
                        "ms_per_1k_tokens": (
                            median_ms / case.total_tokens * 1000.0
                        ),
                        "activation_shape": "x".join(
                            map(str, activation_shape or [])
                        ),
                        "activation_mib": activation_mib,
                        "baseline_allocated_mib_max_rank_median": statistics.median(
                            baseline_allocated_mib
                        ),
                        "baseline_reserved_mib_max_rank_median": statistics.median(
                            baseline_reserved_mib
                        ),
                        "peak_allocated_mib_max_rank_median": statistics.median(
                            peak_allocated_mib
                        ),
                        "peak_reserved_mib_max_rank_median": statistics.median(
                            peak_reserved_mib
                        ),
                        "peak_incremental_allocated_mib_max_rank_median": statistics.median(
                            peak_incremental_allocated_mib
                        ),
                        "peak_incremental_reserved_mib_max_rank_median": statistics.median(
                            peak_incremental_reserved_mib
                        ),
                        "times_ms": times_ms,
                    }
                    rows.append(row)
                    completed_signatures.add(case.signature)
                    _atomic_write_json(
                        result_file,
                        {
                            "status": "partial",
                            "task_hash": task_hash,
                            "task": task,
                            "rows": rows,
                        },
                    )
                    print(
                        "[RESULT] "
                        f"tp={tp} family={case.family} "
                        f"B={case.B} ctx={case.context_size} MBT={case.mbt} "
                        f"N={case.total_tokens} "
                        f"fill={case.batch_over_fill_batch:.4g}x "
                        f"median={median_ms:.3f}ms "
                        f"tok/s={row['tokens_per_s']:.1f} "
                        f"peak={row['peak_allocated_mib_max_rank_median']:.1f}MiB",
                        flush=True,
                    )

                del batch_tokens
                gc.collect()

            except BaseException as case_exc:
                # Keep the experiment moving.  OOMs and unsupported extreme
                # configurations are useful feasibility observations.
                error_name = type(case_exc).__name__
                error_message = str(case_exc)
                if rank == 0:
                    row = {
                        **row_base,
                        "status": "error",
                        "error_type": error_name,
                        "error_message": error_message,
                        "error_traceback": traceback.format_exc(),
                    }
                    rows.append(row)
                    completed_signatures.add(case.signature)
                    _atomic_write_json(
                        result_file,
                        {
                            "status": "partial",
                            "task_hash": task_hash,
                            "task": task,
                            "rows": rows,
                        },
                    )
                    print(
                        f"[CASE ERROR] {case.case_name}: "
                        f"{error_name}: {error_message}",
                        file=sys.stderr,
                        flush=True,
                    )

                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    _sync_all()
                except Exception:
                    # If the CUDA context/process group is no longer healthy,
                    # abort this task.  Completed case rows remain resumable.
                    raise

        if rank == 0:
            _atomic_write_json(
                result_file,
                {
                    "status": "ok",
                    "task_hash": task_hash,
                    "task": task,
                    "rows": rows,
                },
            )

        _sync_all()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return 0

    except BaseException as exc:
        if rank == 0:
            _atomic_write_json(
                result_file,
                {
                    "status": "error",
                    "task_hash": task_hash,
                    "task": task,
                    "rows": rows,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def _group_cases(
    cases: Iterable[Case],
) -> dict[tuple[int, int], list[Case]]:
    groups: dict[tuple[int, int], list[Case]] = {}
    for case in cases:
        groups.setdefault((case.mbt, case.kv_pool_tokens), []).append(case)
    return dict(sorted(groups.items()))


def _git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _task_hash(task_without_hash: dict[str, Any]) -> str:
    raw = json.dumps(task_without_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    scalar_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"times_ms", "error_traceback"}
        }
        for row in rows
    ]

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in scalar_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _print_case_plan(
    cases: list[Case],
    tp_values: list[int],
    repeats: int,
) -> None:
    groups = _group_cases(cases)
    by_family: dict[str, int] = {}
    for case in cases:
        by_family[case.family] = by_family.get(case.family, 0) + 1

    print("\n=== Experiment plan ===")
    print(f"script_version: {SCRIPT_VERSION}")
    print(f"cases per TP: {len(cases)}")
    print(f"TP values: {tp_values}")
    print(f"total measured cases: {len(cases) * len(tp_values)}")
    print(f"timed run_with_cache calls: {len(cases) * len(tp_values) * repeats}")
    print(f"model-load groups per TP: {len(groups)}")
    print(f"family counts: {by_family}")

    print("\n=== Model-load groups per TP ===")
    for (mbt, pool), group_cases in groups.items():
        family_counts: dict[str, int] = {}
        for case in group_cases:
            family_counts[case.family] = family_counts.get(case.family, 0) + 1
        print(
            f"MBT={mbt:6d}, pool={pool:6d}: "
            f"{len(group_cases):3d} case(s), {family_counts}"
        )

    print("\n=== Batch-fill ranges ===")
    batch_groups: dict[tuple[int, int], list[Case]] = {}
    for case in cases:
        if case.family == "batch_fill":
            batch_groups.setdefault((case.context_size, case.mbt), []).append(case)
    for (ctx, mbt), group in sorted(batch_groups.items()):
        Bs = sorted(case.B for case in group)
        fill = mbt // ctx
        ratios = [B / fill for B in Bs]
        print(
            f"ctx={ctx:4d}, MBT={mbt:6d}, B_fill={fill:4d}: "
            f"B={Bs}, fill range={min(ratios):.4g}x..{max(ratios):.4g}x"
        )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _pareto_flags(
    rows: list[dict[str, Any]],
    time_key: str,
    memory_key: str,
) -> list[bool]:
    flags: list[bool] = []
    for i, row in enumerate(rows):
        t_i = _safe_float(row.get(time_key))
        m_i = _safe_float(row.get(memory_key))
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            t_j = _safe_float(other.get(time_key))
            m_j = _safe_float(other.get(memory_key))
            if (
                t_j <= t_i
                and m_j <= m_i
                and (t_j < t_i or m_j < m_i)
            ):
                dominated = True
                break
        flags.append(not dominated)
    return flags


def generate_plots_and_summaries(
    rows: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """Generate plots without fitting any performance model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if str(row.get("status")) == "ok"]
    if not ok_rows:
        print("No successful rows available for plotting.", file=sys.stderr)
        return

    df = pd.DataFrame(ok_rows)
    numeric_columns = [
        "tp",
        "B",
        "context_size",
        "mbt",
        "kv_pool_tokens",
        "total_tokens",
        "total_tokens_over_mbt",
        "mbt_over_context",
        "fill_batch_mbt_over_ctx",
        "batch_over_fill_batch",
        "wall_ms_median",
        "tokens_per_s",
        "ms_per_1k_tokens",
        "peak_allocated_mib_max_rank_median",
        "peak_incremental_allocated_mib_max_rank_median",
        "activation_mib",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    batch_time_dir = plot_dir / "batch_time_memory"
    batch_throughput_dir = plot_dir / "batch_throughput_memory"
    batch_summary_dir = plot_dir / "batch_fill_summary"
    mbt_time_dir = plot_dir / "mbt_time_memory"
    mbt_pareto_dir = plot_dir / "mbt_pareto"
    mbt_summary_dir = plot_dir / "mbt_summary"
    for directory in (
        batch_time_dir,
        batch_throughput_dir,
        batch_summary_dir,
        mbt_time_dir,
        mbt_pareto_dir,
        mbt_summary_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    batch_df = df[df["family"] == "batch_fill"].copy()
    tradeoff_df = df[df["family"] == "mbt_tradeoff"].copy()

    # ------------------------------------------------------------------
    # Batch plots: one dual-axis chart per exact TP/ctx/MBT curve.
    # ------------------------------------------------------------------
    if not batch_df.empty:
        batch_group_columns = ["tp", "context_size", "mbt"]
        for (tp, ctx, mbt), group in batch_df.groupby(batch_group_columns):
            group = group.sort_values("B")
            b_fill = float(mbt) / float(ctx)

            fig, ax_time = plt.subplots(figsize=(8.4, 5.2))
            ax_memory = ax_time.twinx()
            line_time = ax_time.plot(
                group["B"],
                group["wall_ms_median"],
                marker="o",
                label="Wall time",
            )
            line_memory = ax_memory.plot(
                group["B"],
                group["peak_allocated_mib_max_rank_median"],
                marker="s",
                label="Peak allocated",
            )
            ax_time.axvline(b_fill, linestyle="--", linewidth=1.0, label="B_fill")
            ax_time.set_xscale("log", base=2)
            ax_time.set_xlabel("Batch size B")
            ax_time.set_ylabel("Wall time (ms)")
            ax_memory.set_ylabel("Peak allocated per rank (MiB)")
            ax_time.set_title(
                f"Batch time-memory — TP{int(tp)}, ctx={int(ctx)}, MBT={int(mbt)}"
            )
            ax_time.grid(True, alpha=0.25)
            lines = line_time + line_memory + [ax_time.lines[-1]]
            ax_time.legend(lines, [line.get_label() for line in lines], loc="best")
            fig.tight_layout()
            fig.savefig(
                batch_time_dir
                / f"tp{int(tp)}_ctx{int(ctx)}_mbt{int(mbt)}.png",
                dpi=160,
            )
            plt.close(fig)

            fig, ax_throughput = plt.subplots(figsize=(8.4, 5.2))
            ax_memory = ax_throughput.twinx()
            line_throughput = ax_throughput.plot(
                group["B"],
                group["tokens_per_s"],
                marker="o",
                label="Throughput",
            )
            line_memory = ax_memory.plot(
                group["B"],
                group["peak_allocated_mib_max_rank_median"],
                marker="s",
                label="Peak allocated",
            )
            ax_throughput.axvline(
                b_fill,
                linestyle="--",
                linewidth=1.0,
                label="B_fill",
            )
            ax_throughput.set_xscale("log", base=2)
            ax_throughput.set_xlabel("Batch size B")
            ax_throughput.set_ylabel("Throughput (tokens/s)")
            ax_memory.set_ylabel("Peak allocated per rank (MiB)")
            ax_throughput.set_title(
                f"Batch saturation — TP{int(tp)}, ctx={int(ctx)}, MBT={int(mbt)}"
            )
            ax_throughput.grid(True, alpha=0.25)
            lines = line_throughput + line_memory + [ax_throughput.lines[-1]]
            ax_throughput.legend(
                lines,
                [line.get_label() for line in lines],
                loc="best",
            )
            fig.tight_layout()
            fig.savefig(
                batch_throughput_dir
                / f"tp{int(tp)}_ctx{int(ctx)}_mbt{int(mbt)}.png",
                dpi=160,
            )
            plt.close(fig)

        # One summary per TP/ctx: all MBTs on normalized fill ratio.
        for (tp, ctx), group in batch_df.groupby(["tp", "context_size"]):
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            for mbt, mbt_group in group.groupby("mbt"):
                mbt_group = mbt_group.sort_values("batch_over_fill_batch")
                ax.plot(
                    mbt_group["batch_over_fill_batch"],
                    mbt_group["tokens_per_s"],
                    marker="o",
                    label=f"MBT={int(mbt)}",
                )
            ax.axvline(1.0, linestyle="--", linewidth=1.0, label="One full iteration")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Batch / B_fill = B × ctx / MBT")
            ax.set_ylabel("Throughput (tokens/s)")
            ax.set_title(f"Batch fill summary — TP{int(tp)}, ctx={int(ctx)}")
            ax.grid(True, alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(
                batch_summary_dir / f"tp{int(tp)}_ctx{int(ctx)}.png",
                dpi=170,
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # MBT tradeoff plots: exact fixed-N, fixed-ctx curves.
    # ------------------------------------------------------------------
    if not tradeoff_df.empty:
        for (tp, total_tokens, ctx), group in tradeoff_df.groupby(
            ["tp", "target_total_tokens", "context_size"]
        ):
            group = group.sort_values("mbt")

            fig, ax_time = plt.subplots(figsize=(8.4, 5.2))
            ax_memory = ax_time.twinx()
            line_time = ax_time.plot(
                group["mbt"],
                group["wall_ms_median"],
                marker="o",
                label="Wall time",
            )
            line_memory = ax_memory.plot(
                group["mbt"],
                group["peak_allocated_mib_max_rank_median"],
                marker="s",
                label="Peak allocated",
            )
            ax_time.axvline(
                float(total_tokens),
                linestyle="--",
                linewidth=1.0,
                label="MBT=N",
            )
            ax_time.set_xscale("log", base=2)
            ax_time.set_xlabel("max_num_batched_tokens (MBT)")
            ax_time.set_ylabel("Wall time (ms)")
            ax_memory.set_ylabel("Peak allocated per rank (MiB)")
            ax_time.set_title(
                f"MBT time-memory — TP{int(tp)}, N={int(total_tokens)}, "
                f"ctx={int(ctx)}, B={int(group.iloc[0]['B'])}"
            )
            ax_time.grid(True, alpha=0.25)
            lines = line_time + line_memory + [ax_time.lines[-1]]
            ax_time.legend(lines, [line.get_label() for line in lines], loc="best")
            fig.tight_layout()
            fig.savefig(
                mbt_time_dir
                / f"tp{int(tp)}_N{int(total_tokens)}_ctx{int(ctx)}.png",
                dpi=170,
            )
            plt.close(fig)

            # Memory-time Pareto view with MBT labels.
            fig, ax = plt.subplots(figsize=(7.6, 5.4))
            ax.scatter(
                group["peak_allocated_mib_max_rank_median"],
                group["wall_ms_median"],
            )
            for _, point in group.iterrows():
                ax.annotate(
                    f"{int(point['mbt'])}",
                    (
                        point["peak_allocated_mib_max_rank_median"],
                        point["wall_ms_median"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xlabel("Peak allocated per rank (MiB)")
            ax.set_ylabel("Wall time (ms)")
            ax.set_title(
                f"MBT Pareto — TP{int(tp)}, N={int(total_tokens)}, ctx={int(ctx)}"
            )
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                mbt_pareto_dir
                / f"tp{int(tp)}_N{int(total_tokens)}_ctx{int(ctx)}.png",
                dpi=170,
            )
            plt.close(fig)

        # Summary per TP/N: one time line per ctx and one memory line per ctx.
        for (tp, total_tokens), group in tradeoff_df.groupby(
            ["tp", "target_total_tokens"]
        ):
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            for ctx, ctx_group in group.groupby("context_size"):
                ctx_group = ctx_group.sort_values("mbt")
                ax.plot(
                    ctx_group["mbt"],
                    ctx_group["wall_ms_median"],
                    marker="o",
                    label=f"ctx={int(ctx)}",
                )
            ax.axvline(
                float(total_tokens),
                linestyle="--",
                linewidth=1.0,
                label="MBT=N",
            )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("max_num_batched_tokens (MBT)")
            ax.set_ylabel("Wall time (ms)")
            ax.set_title(
                f"MBT timing summary — TP{int(tp)}, N={int(total_tokens)}"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(
                mbt_summary_dir / f"time_tp{int(tp)}_N{int(total_tokens)}.png",
                dpi=170,
            )
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            for ctx, ctx_group in group.groupby("context_size"):
                ctx_group = ctx_group.sort_values("mbt")
                ax.plot(
                    ctx_group["mbt"],
                    ctx_group["peak_allocated_mib_max_rank_median"],
                    marker="s",
                    label=f"ctx={int(ctx)}",
                )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("max_num_batched_tokens (MBT)")
            ax.set_ylabel("Peak allocated per rank (MiB)")
            ax.set_title(
                f"MBT memory summary — TP{int(tp)}, N={int(total_tokens)}"
            )
            ax.grid(True, alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(
                mbt_summary_dir / f"memory_tp{int(tp)}_N{int(total_tokens)}.png",
                dpi=170,
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # Recommendation/summary CSVs.  These report measured points only.
    # ------------------------------------------------------------------
    batch_summary_rows: list[dict[str, Any]] = []
    if not batch_df.empty:
        for (tp, ctx, mbt), group in batch_df.groupby(
            ["tp", "context_size", "mbt"]
        ):
            group = group.sort_values("B")
            best_index = group["tokens_per_s"].idxmax()
            best = group.loc[best_index]
            best_throughput = float(best["tokens_per_s"])
            threshold = best_throughput * 0.99
            within_99 = group[group["tokens_per_s"] >= threshold].sort_values("B")
            smallest_99 = within_99.iloc[0]
            largest = group.iloc[-1]
            fill_rows = group[
                (group["batch_over_fill_batch"] - 1.0).abs() < 1e-12
            ]
            fill = fill_rows.iloc[0] if not fill_rows.empty else None

            batch_summary_rows.append(
                {
                    "tp": int(tp),
                    "context_size": int(ctx),
                    "mbt": int(mbt),
                    "B_fill": int(mbt // ctx),
                    "measured_B_min": int(group["B"].min()),
                    "measured_B_max": int(group["B"].max()),
                    "best_B": int(best["B"]),
                    "best_fill_ratio": float(best["batch_over_fill_batch"]),
                    "best_tokens_per_s": best_throughput,
                    "smallest_B_within_99pct_best": int(smallest_99["B"]),
                    "smallest_99_fill_ratio": float(
                        smallest_99["batch_over_fill_batch"]
                    ),
                    "smallest_99_peak_allocated_mib": float(
                        smallest_99["peak_allocated_mib_max_rank_median"]
                    ),
                    "fill_tokens_per_s": (
                        float(fill["tokens_per_s"]) if fill is not None else None
                    ),
                    "fill_peak_allocated_mib": (
                        float(fill["peak_allocated_mib_max_rank_median"])
                        if fill is not None
                        else None
                    ),
                    "largest_B_tokens_per_s": float(largest["tokens_per_s"]),
                    "largest_B_change_from_best_pct": (
                        float(largest["tokens_per_s"]) / best_throughput - 1.0
                    )
                    * 100.0,
                }
            )

    batch_summary_path = plot_dir / "batch_recommendations.csv"
    if batch_summary_rows:
        pd.DataFrame(batch_summary_rows).sort_values(
            ["tp", "context_size", "mbt"]
        ).to_csv(batch_summary_path, index=False)

    tradeoff_summary_rows: list[dict[str, Any]] = []
    if not tradeoff_df.empty:
        for (tp, total_tokens, ctx), group in tradeoff_df.groupby(
            ["tp", "target_total_tokens", "context_size"]
        ):
            group = group.sort_values("mbt").copy()
            records = group.to_dict("records")
            pareto = _pareto_flags(
                records,
                "wall_ms_median",
                "peak_allocated_mib_max_rank_median",
            )
            fastest_time = float(group["wall_ms_median"].min())
            minimum_memory = float(
                group["peak_allocated_mib_max_rank_median"].min()
            )
            for record, is_pareto in zip(records, pareto):
                tradeoff_summary_rows.append(
                    {
                        "tp": int(tp),
                        "target_total_tokens": int(total_tokens),
                        "context_size": int(ctx),
                        "B": int(record["B"]),
                        "mbt": int(record["mbt"]),
                        "wall_ms_median": float(record["wall_ms_median"]),
                        "tokens_per_s": float(record["tokens_per_s"]),
                        "peak_allocated_mib": float(
                            record["peak_allocated_mib_max_rank_median"]
                        ),
                        "time_over_fastest_pct": (
                            float(record["wall_ms_median"]) / fastest_time - 1.0
                        )
                        * 100.0,
                        "memory_over_minimum_mib": (
                            float(record["peak_allocated_mib_max_rank_median"])
                            - minimum_memory
                        ),
                        "pareto_optimal": bool(is_pareto),
                    }
                )

    tradeoff_summary_path = plot_dir / "mbt_tradeoff_summary.csv"
    if tradeoff_summary_rows:
        pd.DataFrame(tradeoff_summary_rows).sort_values(
            ["tp", "target_total_tokens", "context_size", "mbt"]
        ).to_csv(tradeoff_summary_path, index=False)

    # Compact Markdown index/report.
    summary_lines = [
        "# vLLM activation-profile plot index",
        "",
        f"Successful measured rows: {len(ok_rows)}",
        f"Batch-fill rows: {len(batch_df)}",
        f"MBT-tradeoff rows: {len(tradeoff_df)}",
        "",
        "No curve fitting is used. All figures connect directly measured points.",
        "",
        "## Plot directories",
        "",
        "- `batch_time_memory/`: B vs wall time and peak allocated memory.",
        "- `batch_throughput_memory/`: B vs throughput and peak allocated memory.",
        "- `batch_fill_summary/`: normalized B/B_fill throughput curves.",
        "- `mbt_time_memory/`: MBT vs wall time and peak allocated memory.",
        "- `mbt_pareto/`: measured memory-time points labelled by MBT.",
        "- `mbt_summary/`: cross-context MBT time and memory summaries.",
        "",
        "## Summary tables",
        "",
        "- `batch_recommendations.csv`: best measured B and the smallest B within 99% of best throughput.",
        "- `mbt_tradeoff_summary.csv`: measured time/memory deltas and Pareto flags.",
        "",
    ]
    (plot_dir / "summary.md").write_text("\n".join(summary_lines))

    print(f"Plots and summaries written to: {plot_dir}")


def _load_rows_from_task_results(task_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_file in sorted(task_dir.glob("*.result.json")):
        try:
            payload = json.loads(result_file.read_text())
        except Exception:
            continue
        result_rows = payload.get("rows", [])
        if isinstance(result_rows, list):
            rows.extend(result_rows)
    return rows


def _controller(args: argparse.Namespace) -> int:
    repo_root = Path.cwd()
    script_path = Path(__file__).resolve()

    cases = build_cases(
        scale=args.scale,
        suite=args.suite,
        max_batch=args.max_batch,
        max_total_tokens=args.max_batch_total_tokens,
    )
    _print_case_plan(cases, args.tp_values, args.repeats)
    if args.dry_run:
        return 0

    if (
        args.hook_name == DEFAULT_HOOK_NAME
        and args.stop_at_layer != DEFAULT_STOP_AT_LAYER
    ):
        print(
            "[WARN] stop_at_layer changed while the default hook name remained. "
            "Confirm that the hook lies before the stop boundary.",
            file=sys.stderr,
        )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    task_dir = Path(args.task_dir)
    plot_dir = Path(args.plot_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_cases(cases)
    all_rows: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []

    for tp in args.tp_values:
        for (mbt, kv_pool_tokens), group_cases in groups.items():
            capture_batch_size, capture_context_size = _capture_pair_for_pool(
                kv_pool_tokens
            )
            task_name = f"tp{tp}_mbt{mbt}_pool{kv_pool_tokens}"
            task_file = task_dir / f"{task_name}.task.json"
            result_file = task_dir / f"{task_name}.result.json"
            log_file = task_dir / f"{task_name}.log"

            task_without_hash = {
                "script_version": SCRIPT_VERSION,
                "tp": tp,
                "mbt": mbt,
                "kv_pool_tokens": kv_pool_tokens,
                "capture_batch_size": capture_batch_size,
                "capture_context_size": capture_context_size,
                "model_name": args.model_name,
                "stop_at_layer": args.stop_at_layer,
                "hook_name": args.hook_name,
                "dtype": args.dtype,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "case_order": args.case_order,
                "order_seed": args.order_seed,
                "cases": [asdict(case) for case in group_cases],
            }
            task = {
                **task_without_hash,
                "task_hash": _task_hash(task_without_hash),
            }
            task_file.write_text(json.dumps(task, indent=2) + "\n")

            # Whole-task resume: only reuse an exact task hash marked complete.
            if result_file.exists() and not args.no_resume:
                try:
                    existing = json.loads(result_file.read_text())
                except Exception:
                    existing = {}
                if (
                    existing.get("status") == "ok"
                    and existing.get("task_hash") == task["task_hash"]
                ):
                    existing_rows = existing.get("rows", [])
                    all_rows.extend(existing_rows)
                    task_records.append(
                        {
                            "task_name": task_name,
                            "tp": tp,
                            "mbt": mbt,
                            "kv_pool_tokens": kv_pool_tokens,
                            "status": "reused",
                            "rows": len(existing_rows),
                            "log_file": str(log_file),
                            "result_file": str(result_file),
                        }
                    )
                    print(
                        f"\n[REUSE] {task_name}: {len(existing_rows)} row(s)",
                        flush=True,
                    )
                    continue

            if tp == 1:
                command = [
                    sys.executable,
                    str(script_path),
                    "--worker",
                    "--task-file",
                    str(task_file),
                    "--result-file",
                    str(result_file),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={tp}",
                    str(script_path),
                    "--worker",
                    "--task-file",
                    str(task_file),
                    "--result-file",
                    str(result_file),
                ]
            if args.no_resume:
                command.append("--no-resume")

            print(
                f"\n[RUN] {task_name}: {len(group_cases)} case(s)",
                flush=True,
            )
            started = time.perf_counter()
            log_mode = "w" if args.no_resume else "a"
            with log_file.open(log_mode) as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=os.environ.copy(),
                )
            task_wall_s = time.perf_counter() - started

            if result_file.exists():
                result = json.loads(result_file.read_text())
            else:
                result = {
                    "status": "error",
                    "error_type": "MissingResult",
                    "error_message": (
                        f"worker exited {process.returncode} without result file"
                    ),
                    "rows": [],
                }

            result_rows = result.get("rows", [])
            all_rows.extend(result_rows)
            task_record = {
                "task_name": task_name,
                "tp": tp,
                "mbt": mbt,
                "kv_pool_tokens": kv_pool_tokens,
                "returncode": process.returncode,
                "task_wall_s": task_wall_s,
                "log_file": str(log_file),
                "result_file": str(result_file),
                "status": result.get("status", "unknown"),
                "rows": len(result_rows),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
            task_records.append(task_record)

            if result.get("status") == "ok":
                print(
                    f"[OK] {task_name}: rows={len(result_rows)}, "
                    f"task_wall={task_wall_s:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"[TASK ERROR] {task_name}: {result.get('error_type')}: "
                    f"{result.get('error_message')} (see {log_file})",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    break

        if args.fail_fast and task_records and task_records[-1]["status"] not in {
            "ok",
            "reused",
        }:
            break

    # De-duplicate rows after resumed/reused tasks.
    deduplicated: dict[tuple[int, str], dict[str, Any]] = {}
    for row in all_rows:
        key = (_safe_int(row.get("tp")), str(row.get("case_signature")))
        deduplicated[key] = row
    all_rows = list(deduplicated.values())
    all_rows.sort(
        key=lambda row: (
            _safe_int(row.get("tp")),
            str(row.get("family", "")),
            _safe_int(row.get("context_size")),
            _safe_int(row.get("mbt")),
            _safe_int(row.get("B")),
        )
    )

    payload = {
        "kind": "exploratory_vllm_activation_capture_profile_v3_2",
        "script_version": SCRIPT_VERSION,
        "scale": args.scale,
        "suite": args.suite,
        "repo_git_head": _git_head(repo_root),
        "model": args.model_name,
        "stop_at_layer": args.stop_at_layer,
        "hook_name": args.hook_name,
        "dtype": args.dtype,
        "tp_values": args.tp_values,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_batch": args.max_batch,
        "max_batch_total_tokens": args.max_batch_total_tokens,
        "case_order": args.case_order,
        "order_seed": args.order_seed,
        "tasks": task_records,
        "rows": all_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_json, payload)
    _write_csv(output_csv, all_rows)

    if not args.no_plots:
        generate_plots_and_summaries(all_rows, plot_dir)

    successful = sum(1 for row in all_rows if row.get("status") == "ok")
    failed_cases = sum(1 for row in all_rows if row.get("status") == "error")
    print(f"\nSuccessful cases: {successful}")
    print(f"Failed cases: {failed_cases}")
    print(f"JSON: {output_json}")
    print(f"CSV : {output_csv}")
    print(f"logs: {task_dir}")
    if not args.no_plots:
        print(f"plots: {plot_dir}")

    task_failures = [
        record
        for record in task_records
        if record.get("status") not in {"ok", "reused"}
    ]
    return 1 if task_failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--scale",
        choices=sorted(SCALE_SPECS),
        default="a40_ctx2048",
        help="Focused ctx=2048 A40 saturation grid.",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "batch", "tradeoff", "runmatch"],
        default="batch",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--stop-at-layer", type=int, default=DEFAULT_STOP_AT_LAYER)
    parser.add_argument("--hook-name", default=DEFAULT_HOOK_NAME)
    parser.add_argument(
        "--tp-values",
        default="1,2",
        help="Comma-separated exact TP values. This script supports 1 and 2.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default=DEFAULT_DTYPE,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=DEFAULT_MAX_BATCH,
        help="Maximum B included in the focused grid; default 64.",
    )
    parser.add_argument(
        "--max-batch-total-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_TOTAL_TOKENS,
        help=(
            "Maximum B*ctx and fixed KV-pool capacity for batch-fill curves; "
            "default 131072."
        ),
    )
    parser.add_argument(
        "--case-order",
        choices=["random", "ascending", "descending"],
        default="random",
        help="Randomized order reduces monotonic thermal/order bias.",
    )
    parser.add_argument("--order-seed", type=int, default=20260802)
    parser.add_argument(
        "--output-json",
        default=(
            "sae_lens/autoconfig/profile_results/"
            "vllm_activation_explore_v3_2.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "sae_lens/autoconfig/profile_results/"
            "vllm_activation_explore_v3_2.csv"
        ),
    )
    parser.add_argument(
        "--task-dir",
        default=(
            "sae_lens/autoconfig/profile_results/"
            "vllm_activation_explore_v3_2_tasks"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        default=(
            "sae_lens/autoconfig/profile_results/"
            "vllm_activation_explore_v3_2_plots"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard task-level/case-level resume and rerun exact tasks.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read --input-csv and generate plots without running vLLM.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV used by --plot-only.",
    )

    args = parser.parse_args()

    if args.worker:
        if args.task_file is None or args.result_file is None:
            parser.error("--worker requires --task-file and --result-file")
        return args

    try:
        args.tp_values = [
            int(value.strip())
            for value in str(args.tp_values).split(",")
            if value.strip()
        ]
    except ValueError as exc:
        parser.error(f"invalid --tp-values: {exc}")

    if not args.tp_values:
        parser.error("--tp-values must not be empty")
    if any(tp not in (1, 2) for tp in args.tp_values):
        parser.error("this exploratory script supports only TP=1 and TP=2")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.max_batch < 1 or not _is_power_of_two(args.max_batch):
        parser.error("--max-batch must be a positive power of two")
    if args.max_batch_total_tokens < 4096:
        parser.error("--max-batch-total-tokens must be >= 4096")
    if (
        args.max_batch_total_tokens % DEFAULT_CAPTURE_CONTEXT_FOR_POOL != 0
    ):
        parser.error(
            "--max-batch-total-tokens must be divisible by "
            f"{DEFAULT_CAPTURE_CONTEXT_FOR_POOL}"
        )
    if args.max_model_len < 2049:
        parser.error(
            "v3.2 fixes context_size=2048, so --max-model-len must be >= 2049"
        )
    if args.max_batch > 256:
        parser.error(
            "v3.2 intentionally caps --max-batch at 256; the default is 64"
        )
    if args.plot_only and args.input_csv is None:
        parser.error("--plot-only requires --input-csv")
    return args


def main() -> int:
    args = parse_args()
    if args.worker:
        return _worker(
            task_file=args.task_file,
            result_file=args.result_file,
            no_resume=args.no_resume,
        )
    if args.plot_only:
        rows = _read_csv_rows(args.input_csv)
        generate_plots_and_summaries(rows, Path(args.plot_dir))
        return 0
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
