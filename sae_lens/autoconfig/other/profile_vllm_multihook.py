#!/usr/bin/env python3
# ruff: noqa: T201
"""Multi-hook vLLM activation-prefill profiler at a single fixed (B, MBT) point.

Motivation
==========
`profile_vllm_two_stage_v4.py` sweeps B and MBT for one hook. The step-window
sweep in `scripts/run_step_window_sweep.py` needs the opposite: B and MBT are
pinned to the runner's operating point, and what varies is the *hook set*
(1..4 hooks) and TP degree.

Because B and MBT are fixed, each distinct (hook set, TP) needs exactly one
measurement -- d_sae and SAE parallel mode do not affect the producer at all.

What is measured
================
One `run_with_cache(names_filter=hook_names, stop_at_layer=max(layer)+1)` call,
which is exactly what `ActivationsStore._get_activations_local` issues. Reported
per case: median/mean/min/max wall ms over `--repeats`, throughput, and the
captured activation bytes. Memory is recorded as-observed (allocated/reserved,
baseline and peak) without any standalone-configuration correction -- this
profiler does not model minimal memory.

Defaults are pinned to the operating point of scripts/run_step_window_sweep.py:
Llama-3.1-8B, B=1 prompt, context 2048, MBT 4096, block 16, float32,
gpu_memory_utilization 0.5, capture pool sized to B*context (the same
`capture_batch_size`/`capture_context_size` the runner derives in
`LanguageModelSAERunnerConfig.__post_init__`).

`stop_at_layer` is derived from the hook set as `max(layer) + 1`, matching
`ActivationsStore._get_activations_local`, so a shallow hook set genuinely runs
fewer layers.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "vllm_multihook_v1"

# --- Defaults: the single operating point used by run_step_window_sweep.py ---
DEFAULT_MODEL = "/data/models/Llama-3.1-8B"
DEFAULT_HOOK_SETS = (
    "blocks.21.hook_resid_post"
    "|blocks.21.hook_resid_post,blocks.31.hook_resid_post"
    "|blocks.21.hook_resid_post,blocks.31.hook_resid_post,blocks.11.hook_resid_post"
    "|blocks.21.hook_resid_post,blocks.31.hook_resid_post,"
    "blocks.11.hook_resid_post,blocks.26.hook_resid_post"
)
DEFAULT_TP_VALUES = "1,2"
DEFAULT_BATCH_PROMPTS = 1
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_MAX_NUM_BATCHED_TOKENS = 4096
DEFAULT_MAX_MODEL_LEN = 2049
# The runner's --dtype is the ACTIVATION-STORE dtype; it is never forwarded to
# vLLM (load_model passes only model_from_pretrained_kwargs, which carries no
# dtype), so the engine runs at HookedVLLMModel's bfloat16 default. Profiling at
# float32 would load 8B weights in fp32 and inflate prefill ~3x.
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.50
DEFAULT_BLOCK_SIZE = 16
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 5
DEFAULT_CUDA_DEVICES = "0,1"
DEFAULT_OUTPUT_DIR = "sae_lens/autoconfig/profile_results/vllm_multihook_v1"

_TORCHRUN_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "OMP_NUM_THREADS",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
)


@dataclass
class Case:
    tp: int
    hook_names: list[str] = field(default_factory=list)
    B: int = DEFAULT_BATCH_PROMPTS
    context_size: int = DEFAULT_CONTEXT_SIZE
    mbt: int = DEFAULT_MAX_NUM_BATCHED_TOKENS

    @property
    def num_hooks(self) -> int:
        return len(self.hook_names)

    @property
    def total_tokens(self) -> int:
        return self.B * self.context_size

    @property
    def stop_at_layer(self) -> int:
        """max(hook layer) + 1, matching ActivationsStore._get_activations_local."""
        layers = [_hook_layer(name) for name in self.hook_names]
        layers = [layer for layer in layers if layer is not None]
        if not layers:
            raise ValueError(f"no layer could be parsed from {self.hook_names}")
        return max(layers) + 1

    @property
    def name(self) -> str:
        return (
            f"tp{self.tp}_H{self.num_hooks}_B{self.B}_S{self.context_size}"
            f"_M{self.mbt}_L{self.stop_at_layer}"
        )


def _hook_layer(hook_name: str) -> int | None:
    import re

    match = re.search(r"\.(\d+)\.", hook_name)
    return None if match is None else int(match.group(1))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _parse_int_csv(text: str, *, name: str) -> list[int]:
    values = [int(part) for part in text.replace(" ", "").split(",") if part]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_hook_sets(text: str) -> list[list[str]]:
    """Parse '|'-separated hook sets, each a ','-separated hook list."""
    sets: list[list[str]] = []
    for group in text.split("|"):
        hooks = [part.strip() for part in group.split(",") if part.strip()]
        if hooks:
            sets.append(hooks)
    if not sets:
        raise ValueError("--hook-sets produced no hook set")
    return sets


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


def _pool_blocks_for_tokens(tokens: int, block_size: int) -> int:
    # Matches the repository's activation-capture vLLM fork:
    # ceil(required_tokens / block_size) + one safety block.
    return math.ceil(tokens / block_size) + 1


def _resolve_scheduler(model: Any) -> Any:
    """Reach the V1 scheduler through the engine core, as v4 does."""
    llm = getattr(model, "llm", model)
    engine = getattr(llm, "llm_engine", None)
    for path in (
        ("engine_core", "engine_core", "scheduler"),
        ("engine_core", "scheduler"),
        ("scheduler",),
    ):
        node = engine
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None:
            return node
    raise RuntimeError("could not resolve the vLLM V1 scheduler")


def _measure_kv_pool(model: Any) -> tuple[int, int]:
    scheduler = _resolve_scheduler(model)
    blocks = int(scheduler.kv_cache_manager.block_pool.num_gpu_blocks)
    kv_bytes = 0
    try:
        caches = (
            model.llm.llm_engine.engine_core.engine_core.model_executor.driver_worker
        )
        kv_bytes = 0
        for tensor in getattr(caches, "kv_cache_list", []) or []:
            kv_bytes += tensor.numel() * tensor.element_size()
    except Exception:
        kv_bytes = 0
    return blocks, kv_bytes


def _measure_case(
    *,
    model: Any,
    case: Case,
    tokens: Any,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    hook_names = list(case.hook_names)
    stop_at_layer = case.stop_at_layer

    def run() -> Any:
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_names,
            stop_at_layer=stop_at_layer,
            prepend_bos=False,
        )
        return cache

    for _ in range(warmup):
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
    activation_shapes: dict[str, str] = {}

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

        total_bytes = 0
        for hook_name in hook_names:
            act = cache[hook_name]
            if tuple(act.shape[:2]) != (case.B, case.context_size):
                raise RuntimeError(
                    f"{case.name}: {hook_name} activation shape "
                    f"{tuple(act.shape)} does not match expected leading dims "
                    f"{(case.B, case.context_size)}"
                )
            total_bytes += act.numel() * act.element_size()
            activation_shapes[hook_name] = "x".join(map(str, act.shape))
        activation_mib = total_bytes / (1024**2)

        times_ms.append(_all_reduce_max_float(elapsed_ms))
        baseline_allocated.append(_all_reduce_max_float(base_alloc))
        baseline_reserved.append(_all_reduce_max_float(base_res))
        peak_allocated.append(
            _all_reduce_max_float(torch.cuda.max_memory_allocated() / (1024**2))
        )
        peak_reserved.append(
            _all_reduce_max_float(torch.cuda.max_memory_reserved() / (1024**2))
        )
        del cache

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)
    return {
        "status": "ok",
        "repeats": repeats,
        "wall_ms_median": median_ms,
        "wall_ms_mean": mean_ms,
        "wall_ms_min": min(times_ms),
        "wall_ms_max": max(times_ms),
        "wall_ms_cv": (
            statistics.pstdev(times_ms) / mean_ms
            if len(times_ms) > 1 and mean_ms > 0
            else 0.0
        ),
        "tokens_per_s": case.total_tokens / (median_ms / 1000.0),
        "ms_per_1k_tokens": median_ms / case.total_tokens * 1000.0,
        "activation_total_mib": activation_mib,
        "activation_shapes": json.dumps(activation_shapes, sort_keys=True),
        "measured_baseline_allocated_mib": statistics.median(baseline_allocated),
        "measured_baseline_reserved_mib": statistics.median(baseline_reserved),
        "measured_peak_allocated_mib": statistics.median(peak_allocated),
        "measured_peak_reserved_mib": statistics.median(peak_reserved),
    }


def _worker(task_file: Path, result_file: Path) -> int:
    """Run every case of one TP group inside a single engine.

    All cases in a group share B, context_size and MBT, so one engine serves
    them; only names_filter/stop_at_layer differ per case.
    """
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
    mbt = int(task["mbt"])
    batch_prompts = int(task["B"])
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

    # Mirrors what LanguageModelSAERunnerConfig.__post_init__ derives for the
    # runner: max_num_batched_tokens from the CLI, and a capture pool sized to
    # store_batch_size_prompts * context_size rather than to MBT.
    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": tp,
        "max_model_len": int(task["max_model_len"]),
        "max_num_batched_tokens": mbt,
        "block_size": block_size,
        "gpu_memory_utilization": float(task["gpu_memory_utilization"]),
        "enable_chunked_prefill": True,
        "capture_batch_size": batch_prompts,
        "capture_context_size": context_size,
    }
    if tp > 1:
        llm_kwargs["distributed_executor_backend"] = "external_launcher"
        llm_kwargs["device"] = f"cuda:{local_rank}"

    model = HookedVLLMModel(task["model_name"], tokenizer, dtype=dtype, **llm_kwargs)
    vocab_size = int(getattr(tokenizer, "vocab_size", 128256) or 128256)

    pool_blocks, kv_bytes = _measure_kv_pool(model)
    if rank == 0:
        print(
            f"[ENGINE] TP={tp} mbt={mbt} B={batch_prompts} S={context_size} "
            f"pool_blocks={pool_blocks} "
            f"expected_blocks={_pool_blocks_for_tokens(batch_prompts * context_size, block_size)}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    tokens = _make_tokens(batch_prompts, context_size, vocab_size)
    for case in cases:
        row = _measure_case(
            model=model,
            case=case,
            tokens=tokens,
            warmup=int(task["warmup"]),
            repeats=int(task["repeats"]),
        )
        if rank == 0:
            row.update(
                {
                    "script_version": SCRIPT_VERSION,
                    "case_name": case.name,
                    "tp": tp,
                    "num_hooks": case.num_hooks,
                    "hook_names": ",".join(case.hook_names),
                    "stop_at_layer": case.stop_at_layer,
                    "B": case.B,
                    "context_size": case.context_size,
                    "total_tokens": case.total_tokens,
                    "mbt": mbt,
                    "max_model_len": int(task["max_model_len"]),
                    "block_size": block_size,
                    "dtype": task["dtype"],
                    "gpu_memory_utilization": task["gpu_memory_utilization"],
                    "kv_pool_blocks": pool_blocks,
                    "kv_pool_bytes": kv_bytes,
                    "model_name": task["model_name"],
                    "task_hash": task["task_hash"],
                }
            )
            rows.append(row)
            print(
                f"[CASE] {case.name}: median={row['wall_ms_median']:.2f}ms "
                f"cv={row['wall_ms_cv']:.4f} "
                f"act={row['activation_total_mib']:.1f}MiB",
                flush=True,
            )
        gc.collect()
    del tokens

    if rank == 0:
        _atomic_write_json(
            result_file,
            {"status": "ok", "task_hash": task["task_hash"], "rows": rows},
        )
    _sync_all()
    return 0


def build_cases(*, tp: int, args: argparse.Namespace) -> list[Case]:
    hook_sets = _parse_hook_sets(args.hook_sets)
    return [
        Case(
            tp=tp,
            hook_names=hooks,
            B=args.batch_prompts,
            context_size=args.context_size,
            mbt=args.max_num_batched_tokens,
        )
        for hooks in hook_sets
    ]


def _run_group(*, tp: int, args: argparse.Namespace, output_dir: Path) -> list[dict]:
    cases = build_cases(tp=tp, args=args)
    task_base = {
        "tp": tp,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "context_size": args.context_size,
        "B": args.batch_prompts,
        "mbt": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "script_version": SCRIPT_VERSION,
        "cases": [asdict(case) for case in cases],
    }
    task_hash = _canonical_hash(task_base)
    task = {**task_base, "task_hash": task_hash}

    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"tp{tp}"
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
        f"B={args.batch_prompts}, S={args.context_size}, "
        f"mbt={args.max_num_batched_tokens}, hash={task_hash}",
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


def _controller(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tp_values = _parse_int_csv(args.tp_values, name="--tp-values")

    if args.dry_run:
        for tp in tp_values:
            for case in build_cases(tp=tp, args=args):
                print(
                    f"{case.name}: hooks={case.hook_names} "
                    f"stop_at_layer={case.stop_at_layer} "
                    f"tokens={case.total_tokens}"
                )
        return 0

    all_rows: list[dict[str, Any]] = []
    for tp in tp_values:
        all_rows.extend(_run_group(tp=tp, args=args, output_dir=output_dir))

    if not all_rows:
        print("[WARN] no rows produced", flush=True)
        return 1

    _write_csv(output_dir / "vllm_multihook_profile.csv", all_rows)
    _atomic_write_json(
        output_dir / "vllm_multihook_profile.json",
        {"script_version": SCRIPT_VERSION, "rows": all_rows},
    )

    header = (
        f"{'case':34s} {'tp':>3s} {'H':>2s} {'stopL':>6s} {'B':>3s} "
        f"{'S':>6s} {'mbt':>6s} {'median_ms':>10s} {'mean_ms':>9s} "
        f"{'cv':>7s} {'tok/s':>10s} {'act_MiB':>8s}"
    )
    lines = [header, "-" * len(header)]
    for row in all_rows:
        lines.append(
            f"{row['case_name']:34s} {row['tp']:3d} {row['num_hooks']:2d} "
            f"{row['stop_at_layer']:6d} {row['B']:3d} {row['context_size']:6d} "
            f"{row['mbt']:6d} {row['wall_ms_median']:10.2f} "
            f"{row['wall_ms_mean']:9.2f} {row['wall_ms_cv']:7.4f} "
            f"{row['tokens_per_s']:10.1f} {row['activation_total_mib']:8.1f}"
        )
    table = "\n".join(lines)
    (output_dir / "vllm_multihook_table.txt").write_text(table + "\n")
    print(table, flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--hook-sets",
        default=DEFAULT_HOOK_SETS,
        help=(
            "'|'-separated hook sets, each a ','-separated hook list. "
            "stop_at_layer is derived per set as max(layer)+1."
        ),
    )
    parser.add_argument("--tp-values", default=DEFAULT_TP_VALUES)
    parser.add_argument(
        "--batch-prompts",
        type=int,
        default=DEFAULT_BATCH_PROMPTS,
        help="B: prompts per prefill (the runner's store_batch_size_prompts).",
    )
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help="MBT: the engine's per-step token budget.",
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cuda-devices", default=DEFAULT_CUDA_DEVICES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        if not args.task_file or not args.result_file:
            raise ValueError("--worker requires --task-file and --result-file")
        return _worker(Path(args.task_file), Path(args.result_file))
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
