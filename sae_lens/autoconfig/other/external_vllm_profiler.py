#!/usr/bin/env python3
"""External-launcher vLLM activation profiler for exact-point lookup.

Profile semantics
-----------------
The profile table is intentionally simple.  A measured row is identified by
three *exact* simulator dimensions plus a measured forward depth:

    (model_name, dtype, tp, stop_at_layer)

The hook names used while profiling are recorded for provenance.  batch_size
and mbt are optional profiling metadata/configuration knobs and are deliberately
NOT simulator dimensions.

Important behavior:
* dtype defaults to float16.
* TP>1 is launched with torchrun and vLLM external_launcher.
* gpu_memory_utilization is NEVER passed.
* If --batch-size is omitted, capture_batch_size/capture_context_size are NOT
  passed to HookedVLLMModel.  The input workload still needs a real batch, so
  the profiler sends one prompt internally (input_batch_size=1); the CSV
  batch_size field remains blank to mean "not explicitly configured".
* If --mbt is omitted, max_num_batched_tokens is NOT passed.
* stop_at_layer is derived from each --hook-set as max(block_index) + 1.

Examples
--------
python sae_lens/autoconfig/external_vllm_profiler.py \
  --model-name /data/models/Llama-3.1-8B \
  --dtype float16 \
  --tp-values 1 2 \
  --hook-set blocks.5.hook_resid_post \
  --hook-set blocks.10.hook_resid_post \
  --hook-set blocks.21.hook_resid_post \
  --context-size 2048

A multi-hook profile point is also allowed:

  --hook-set blocks.10.hook_resid_post,blocks.21.hook_resid_post

The simulator does not require the exact profiled hook names; it uses their
count only as a tie-breaker when several rows share the requested stop layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent.parent if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
DEFAULT_OUTPUT = Path("sae_lens/autoconfig/profile_results/external_vllm_profile.csv")
HOOK_LAYER_RE = re.compile(r"\bblocks\.(\d+)\.")

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

DTYPE_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}

CSV_FIELDS = [
    "status",
    "error_type",
    "error_message",
    "model_name",
    "dtype",
    "tp",
    "hook_names",
    "hook_count",
    "stop_at_layer",
    "batch_size",
    "mbt",
    "context_size",
    "input_batch_size",
    "input_total_tokens",
    "warmup",
    "repeats",
    "wall_ms_median",
    "wall_ms_mean",
    "wall_ms_std",
    "wall_ms_min",
    "wall_ms_max",
    "activation_mib",
    "activation_shapes",
    "gpu_name",
]


@dataclass(frozen=True)
class HookCase:
    hook_names: tuple[str, ...]
    stop_at_layer: int


@dataclass(frozen=True)
class GroupCase:
    model_name: str
    dtype: str
    tp: int


def canonical_dtype(value: str) -> str:
    key = value.strip().lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"unsupported dtype {value!r}; choose float16/bfloat16/float32")
    return DTYPE_ALIASES[key]


def parse_hook_set(text: str) -> HookCase:
    hooks = tuple(part.strip() for part in text.split(",") if part.strip())
    if not hooks:
        raise ValueError("--hook-set cannot be empty")
    layers: list[int] = []
    for hook in hooks:
        match = HOOK_LAYER_RE.search(hook)
        if match is None:
            raise ValueError(
                f"cannot infer stop_at_layer from hook {hook!r}; "
                "use blocks.<N>.<hook> names"
            )
        layers.append(int(match.group(1)))
    return HookCase(hook_names=hooks, stop_at_layer=max(layers) + 1)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


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
        return float(value)
    tensor = torch.tensor(
        [value], dtype=torch.float64, device=torch.device("cuda", _local_rank())
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _make_tokens(batch_size: int, context_size: int, vocab_size: int):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(batch_size * 100_003 + context_size * 101)
    high = max(101, int(vocab_size) - 256)
    tokens = torch.randint(
        low=100,
        high=high,
        size=(batch_size, context_size),
        generator=generator,
        dtype=torch.long,
    )
    return tokens.to(torch.device("cuda", _local_rank()))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _task_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in CSV_FIELDS}


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(dict(row)))


def _hook_names_key(hooks: Iterable[str]) -> str:
    return json.dumps(list(hooks), separators=(",", ":"))


def _resume_key_from_row(row: dict[str, Any]) -> tuple[str, str, int, str, str, str, int]:
    return (
        str(row.get("model_name", "")),
        canonical_dtype(str(row.get("dtype", "float16"))),
        int(row.get("tp") or 0),
        str(row.get("hook_names", "")),
        str(row.get("batch_size", "")),
        str(row.get("mbt", "")),
        int(row.get("context_size") or 0),
    )


def _resume_key(
    group: GroupCase,
    hook_case: HookCase,
    batch_size: int | None,
    mbt: int | None,
    context_size: int,
) -> tuple[str, str, int, str, str, str, int]:
    return (
        group.model_name,
        group.dtype,
        group.tp,
        _hook_names_key(hook_case.hook_names),
        "" if batch_size is None else str(batch_size),
        "" if mbt is None else str(mbt),
        context_size,
    )


def _worker(task_path: Path, result_path: Path) -> int:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from sae_lens.vllm_model import HookedVLLMModel

    task = json.loads(task_path.read_text())
    rank = _rank()
    local_rank = _local_rank()
    tp = int(task["tp"])
    torch.cuda.set_device(local_rank)
    if _world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    dtype_name = canonical_dtype(task["dtype"])
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(task["model_name"])
    context_size = int(task["context_size"])
    configured_batch_size = task.get("batch_size")
    input_batch_size = int(configured_batch_size) if configured_batch_size is not None else 1
    mbt = task.get("mbt")

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": tp,
        "max_model_len": context_size + 1,
    }
    # Intentionally NO gpu_memory_utilization.
    if mbt is not None:
        llm_kwargs["max_num_batched_tokens"] = int(mbt)
    if tp > 1:
        llm_kwargs["distributed_executor_backend"] = "external_launcher"
        llm_kwargs["device"] = f"cuda:{local_rank}"

    model_kwargs: dict[str, Any] = {}
    # Only explicit --batch-size pins the capture pool sizing knobs.
    if configured_batch_size is not None:
        model_kwargs["capture_batch_size"] = int(configured_batch_size)
        model_kwargs["capture_context_size"] = context_size

    model = HookedVLLMModel(
        task["model_name"],
        tokenizer,
        dtype=torch_dtype,
        **model_kwargs,
        **llm_kwargs,
    )

    vocab_size = int(getattr(tokenizer, "vocab_size", 128256) or 128256)
    tokens = _make_tokens(input_batch_size, context_size, vocab_size)
    warmup = int(task["warmup"])
    repeats = int(task["repeats"])
    rows: list[dict[str, Any]] = []

    try:
        for hook_spec in task["hook_cases"]:
            hooks = list(hook_spec["hook_names"])
            stop_at_layer = int(hook_spec["stop_at_layer"])

            def run_once():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=hooks,
                    stop_at_layer=stop_at_layer,
                    prepend_bos=False,
                )
                return cache

            for _ in range(warmup):
                _sync_all()
                cache = run_once()
                _sync_all()
                del cache

            samples: list[float] = []
            activation_mib = 0.0
            activation_shapes: dict[str, list[int]] = {}
            for _ in range(repeats):
                _sync_all()
                started = time.perf_counter()
                cache = run_once()
                _sync_all()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                samples.append(_all_reduce_max_float(elapsed_ms))
                if rank == 0:
                    activation_mib = sum(
                        tensor.numel() * tensor.element_size() for tensor in cache.values()
                    ) / (1024**2)
                    activation_shapes = {
                        name: list(tensor.shape) for name, tensor in cache.items()
                    }
                del cache

            if rank == 0:
                row = {
                    "status": "ok",
                    "error_type": "",
                    "error_message": "",
                    "model_name": task["model_name"],
                    "dtype": dtype_name,
                    "tp": tp,
                    "hook_names": _hook_names_key(hooks),
                    "hook_count": len(hooks),
                    "stop_at_layer": stop_at_layer,
                    "batch_size": "" if configured_batch_size is None else int(configured_batch_size),
                    "mbt": "" if mbt is None else int(mbt),
                    "context_size": context_size,
                    "input_batch_size": input_batch_size,
                    "input_total_tokens": input_batch_size * context_size,
                    "warmup": warmup,
                    "repeats": repeats,
                    "wall_ms_median": statistics.median(samples),
                    "wall_ms_mean": statistics.fmean(samples),
                    "wall_ms_std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
                    "wall_ms_min": min(samples),
                    "wall_ms_max": max(samples),
                    "activation_mib": activation_mib,
                    "activation_shapes": json.dumps(activation_shapes, separators=(",", ":")),
                    "gpu_name": torch.cuda.get_device_name(local_rank),
                }
                rows.append(row)
                _atomic_json(
                    result_path,
                    {"status": "partial", "rows": rows, "task": task},
                )
                print(
                    f"[RESULT] tp={tp} stop={stop_at_layer} hooks={len(hooks)} "
                    f"median={row['wall_ms_median']:.3f} ms",
                    flush=True,
                )

        if rank == 0:
            _atomic_json(result_path, {"status": "ok", "rows": rows, "task": task})
        return 0
    except BaseException as exc:
        if rank == 0:
            err = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=20),
                "rows": rows,
                "task": task,
            }
            _atomic_json(result_path, err)
        raise


def _run_group(
    *,
    group: GroupCase,
    hook_cases: Sequence[HookCase],
    args: argparse.Namespace,
    task_dir: Path,
) -> list[dict[str, Any]]:
    payload = {
        "model_name": group.model_name,
        "dtype": group.dtype,
        "tp": group.tp,
        "hook_cases": [
            {"hook_names": list(case.hook_names), "stop_at_layer": case.stop_at_layer}
            for case in hook_cases
        ],
        "batch_size": args.batch_size,
        "mbt": args.mbt,
        "context_size": args.context_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    digest = _task_hash(payload)
    stem = f"tp{group.tp}_{digest}"
    task_path = task_dir / f"{stem}.task.json"
    result_path = task_dir / f"{stem}.result.json"
    log_path = task_dir / f"{stem}.log"
    _atomic_json(task_path, payload)

    if group.tp > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={group.tp}",
            os.path.abspath(__file__),
        ]
    else:
        command = [sys.executable, os.path.abspath(__file__)]
    command += ["--worker", "--task-file", str(task_path), "--result-file", str(result_path)]

    env = os.environ.copy()
    if args.cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["VLLM_ACTIVATION_CAPTURE_MODE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if group.tp == 1:
        for key in _TORCHRUN_ENV_KEYS:
            env.pop(key, None)

    print(
        f"[RUN] model={group.model_name} dtype={group.dtype} tp={group.tp} "
        f"hook_cases={len(hook_cases)} batch_size={args.batch_size} mbt={args.mbt}",
        flush=True,
    )
    with log_path.open("w") as handle:
        completed = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)

    if not result_path.exists():
        raise RuntimeError(f"worker produced no result; see {log_path}")
    result = json.loads(result_path.read_text())
    rows = list(result.get("rows", []))
    if completed.returncode != 0 or result.get("status") != "ok":
        message = result.get("error_message") or f"worker exit={completed.returncode}"
        if args.fail_fast:
            raise RuntimeError(f"{message}; see {log_path}")
        print(f"[WARN] {message}; partial rows kept; see {log_path}", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--model-name",
        nargs="+",
        required=False,
        help="One or more exact model names/paths. Required in controller mode.",
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        default=["float16"],
        help="One or more dtypes; default float16.",
    )
    parser.add_argument("--tp-values", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--hook-set",
        action="append",
        default=[],
        help=(
            "Comma-separated hook names for one measured profile row. Repeat this "
            "flag to profile several stop depths and/or hook counts."
        ),
    )
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional explicit capture batch size. Omitted => not passed to vLLM.",
    )
    parser.add_argument(
        "--mbt",
        type=int,
        default=None,
        help="Optional max_num_batched_tokens. Omitted => not passed to vLLM.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.task_file is None or args.result_file is None:
            raise SystemExit("--worker requires --task-file and --result-file")
        return _worker(args.task_file, args.result_file)

    if not args.model_name:
        raise SystemExit("--model-name is required")
    if not args.hook_set:
        raise SystemExit("at least one --hook-set is required")
    if args.context_size <= 0:
        raise SystemExit("--context-size must be > 0")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.mbt is not None and args.mbt <= 0:
        raise SystemExit("--mbt must be > 0")
    if args.mbt is not None:
        effective_b = args.batch_size if args.batch_size is not None else 1
        if args.mbt < effective_b * args.context_size:
            print(
                "[WARN] MBT is smaller than input_batch_size*context_size; this will "
                "exercise chunked prefill rather than one exact prefill iteration.",
                flush=True,
            )
    if any(tp <= 0 for tp in args.tp_values):
        raise SystemExit("all --tp-values must be > 0")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("--warmup must be >=0 and --repeats >=1")

    dtypes = [canonical_dtype(value) for value in args.dtype]
    hook_cases = [parse_hook_set(text) for text in args.hook_set]
    # Preserve order while removing identical hook sets.
    hook_cases = list(dict.fromkeys(hook_cases))
    groups = [
        GroupCase(model_name=model, dtype=dtype, tp=tp)
        for model, dtype, tp in itertools.product(args.model_name, dtypes, args.tp_values)
    ]

    existing = _read_existing(args.output) if args.resume else []
    existing_ok_keys = {
        _resume_key_from_row(row)
        for row in existing
        if str(row.get("status", "ok")) == "ok"
    }
    all_rows = list(existing)
    task_dir = args.output.parent / (args.output.stem + "_tasks")
    task_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        pending = [
            hook_case
            for hook_case in hook_cases
            if _resume_key(group, hook_case, args.batch_size, args.mbt, args.context_size)
            not in existing_ok_keys
        ]
        if not pending:
            print(
                f"[SKIP] model={group.model_name} dtype={group.dtype} tp={group.tp}: "
                "all requested hook sets already profiled",
                flush=True,
            )
            continue
        new_rows = _run_group(group=group, hook_cases=pending, args=args, task_dir=task_dir)
        all_rows.extend(new_rows)
        _write_csv(args.output, all_rows)
        for row in new_rows:
            if row.get("status") == "ok":
                existing_ok_keys.add(_resume_key_from_row(row))

    _write_csv(args.output, all_rows)
    print(f"Wrote {len(all_rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
