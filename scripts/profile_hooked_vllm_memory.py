"""Profile Hooked vLLM activation-generation GPU memory as a PyTorch snapshot.

Produces ``.pickle`` files loadable at https://pytorch.org/memory_viz that
capture the allocation lifecycle of:

1. vLLM model weights,
2. the minimized KV cache, and
3. the activation capture buffers ``HookedVLLMModel.run_with_cache`` keeps
   alive for the current batch.

Recording is started BEFORE the model is constructed, so the allocation stacks
for weights and KV cache appear in the snapshot (not only the activation
buffers). The run path is deliberately minimal and does NOT touch the SAE
trainer, activations store, or on-disk activation cache:

    start_memory_history()
      -> HookedVLLMModel(...)          # vLLM load_model + minimized KV init
      -> model.run_with_cache(...)     # one real prefill, hooks capture acts
      -> keep activations alive
      -> dump_memory_snapshot(...)     # weights + KV + capture all still live
    stop_memory_history()

Cumulative snapshots (all share one recording lifecycle):
  vllm_after_weight_load_global{g}_tp{t}_pp{p}.pickle   (via worker patch)
  vllm_after_kv_init_global{g}_tp{t}_pp{p}.pickle       (via worker patch)
  vllm_full_lifecycle_global{g}_tp{t}_pp{p}.pickle      (after capture)

TP1 runs in one process. For TP>1 / PP>1, launch under torchrun so every rank
loads its model inline (vLLM external_launcher backend) and each rank starts
recording before load and dumps its own snapshot. See the module docstring at
the bottom / --help for commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.profiler import record_function
from transformers import AutoConfig, AutoTokenizer

from sae_lens.vllm_memory_snapshot import (
    STAGE_CAPTURE_MERGE,
    STAGE_KV_CACHE_INIT,
    STAGE_MODEL_LOAD,
    STAGE_RUN_WITH_CACHE,
    STAGE_SNAPSHOT_DUMP,
    collect_worker_weight_kv_bytes,
    dump_memory_snapshot,
    start_memory_history,
    stop_memory_history,
    sum_unique_cuda_storage_bytes,
    validate_snapshot,
)
from sae_lens.vllm_model import HookedVLLMModel

# Set before importing/constructing vLLM: keep the scheduler in-process and the
# KV cache minimized. HookedVLLMModel also sets these, but we set them here so
# the very first worker init (which may run during import) sees them too.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Populated from CLI before the model is built; the worker monkeypatches read
# it to know where to dump the cumulative snapshots.
_WORKER_SNAPSHOT_STATE: dict[str, Any] = {"output_dir": None, "enabled": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=sorted(_DTYPE_MAP)
    )
    parser.add_argument(
        "--hook-names",
        default="blocks.0.hook_resid_post",
        help="Comma-separated TransformerLens-style hook names to capture.",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/vllm_memory_snapshot")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the synthetic-but-valid token ids.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Rank resolution
# ---------------------------------------------------------------------------


def _resolve_ranks() -> tuple[int, int, int]:
    """Return (global_rank, tp_rank, pp_rank) for the current process."""
    global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    tp_rank = 0
    pp_rank = 0
    try:
        from vllm.distributed.parallel_state import (
            get_pipeline_model_parallel_rank,
            get_tensor_model_parallel_rank,
        )

        try:
            tp_rank = int(get_tensor_model_parallel_rank())
        except Exception:
            tp_rank = 0
        try:
            pp_rank = int(get_pipeline_model_parallel_rank())
        except Exception:
            pp_rank = 0
    except ImportError:
        pass
    return global_rank, tp_rank, pp_rank


def _rank_suffix() -> str:
    g, t, p = _resolve_ranks()
    return f"global{g}_tp{t}_pp{p}"


# ---------------------------------------------------------------------------
# Worker monkeypatches: dump cumulative snapshots right after weight load and
# right after KV cache initialization. These run in the worker process, which
# for TP1 (UniProcExecutor) and TP>1-under-torchrun (external_launcher) is the
# same process that started recording — so the shared allocation history is
# dumped correctly. If patching fails (vLLM version drift) we log and continue;
# the full-lifecycle snapshot is produced regardless.
# ---------------------------------------------------------------------------


def _install_worker_snapshot_patches(output_dir: Path) -> None:
    _WORKER_SNAPSHOT_STATE["output_dir"] = output_dir
    _WORKER_SNAPSHOT_STATE["enabled"] = True

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as exc:  # pragma: no cover - version guard
        print(f"[WARN] could not import vLLM Worker for patching: {exc!r}")
        return

    if getattr(Worker, "_sae_snapshot_patched", False):
        return

    orig_load_model = Worker.load_model
    orig_init_from_config = Worker.initialize_from_config

    def patched_load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        with record_function(STAGE_MODEL_LOAD):
            result = orig_load_model(self, *args, **kwargs)
        _dump_worker_snapshot("vllm_after_weight_load")
        return result

    def patched_init_from_config(self: Any, *args: Any, **kwargs: Any) -> Any:
        with record_function(STAGE_KV_CACHE_INIT):
            result = orig_init_from_config(self, *args, **kwargs)
        _dump_worker_snapshot("vllm_after_kv_init")
        return result

    Worker.load_model = patched_load_model  # type: ignore[method-assign]
    Worker.initialize_from_config = patched_init_from_config  # type: ignore[method-assign]
    Worker._sae_snapshot_patched = True  # type: ignore[attr-defined]


def _dump_worker_snapshot(name: str) -> None:
    if not _WORKER_SNAPSHOT_STATE["enabled"]:
        return
    output_dir = _WORKER_SNAPSHOT_STATE["output_dir"]
    if output_dir is None:
        return
    path = Path(output_dir) / f"{name}_{_rank_suffix()}.pickle"
    try:
        dump_memory_snapshot(path)
        print(f"[INFO] dumped cumulative snapshot: {path}")
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[WARN] failed to dump {name} snapshot: {exc!r}")


# ---------------------------------------------------------------------------
# Token construction
# ---------------------------------------------------------------------------


def _build_token_ids(
    *, model_name: str, batch_size: int, sequence_length: int, seed: int
) -> torch.Tensor:
    """Build real, in-range token ids of shape (batch_size, sequence_length).

    Uses a deterministic CPU generator (seeded from CLI, not a global test
    seed) drawing from ``[0, vocab_size)`` so every id is a valid token the
    model can embed. Not natural text — but a genuine forward pass, which is
    what the memory profile needs. Returned on CPU; run_with_cache converts to
    Python lists for vLLM prompts.
    """
    cfg = AutoConfig.from_pretrained(
        model_name, local_files_only=Path(model_name).exists()
    )
    vocab_size = int(getattr(cfg, "vocab_size"))
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _collect_weight_kv_bytes(model: HookedVLLMModel) -> dict[str, int]:
    """Aggregate weight/KV byte totals across all workers of this process's LLM.

    For TP1 there is one worker; for external_launcher TP>1 each rank's LLM has
    one local worker. We sum weights over workers (each holds a shard) and take
    the KV total the same way. ``collective_rpc`` runs the collector inside the
    worker(s).
    """
    try:
        per_worker = model.llm.collective_rpc(collect_worker_weight_kv_bytes)
    except Exception as exc:  # pragma: no cover - version guard
        print(f"[WARN] collective_rpc weight/kv collection failed: {exc!r}")
        return {
            "weight_bytes": 0,
            "weight_bytes_param_scan": 0,
            "kv_cache_bytes": 0,
        }
    weight_bytes = sum(int(r.get("weight_bytes", 0)) for r in per_worker)
    weight_scan = sum(
        int(r.get("weight_bytes_param_scan", 0)) for r in per_worker
    )
    kv_bytes = sum(int(r.get("kv_cache_bytes", 0)) for r in per_worker)
    return {
        "weight_bytes": weight_bytes,
        "weight_bytes_param_scan": weight_scan,
        "kv_cache_bytes": kv_bytes,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This profiling script is GPU-only.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    hook_names = [h.strip() for h in args.hook_names.split(",") if h.strip()]
    if not hook_names:
        raise ValueError("--hook-names must list at least one hook")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = _DTYPE_MAP[args.dtype]

    # (1) START RECORDING BEFORE THE MODEL IS BUILT. This is the whole point:
    # weights and KV cache must be allocated while recording is live.
    start_memory_history()

    # Install worker patches so cumulative after-load / after-kv snapshots dump
    # inside the same process that owns the CUDA allocations.
    _install_worker_snapshot_patches(output_dir)

    allocated_before = int(torch.cuda.memory_allocated(device))

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": args.block_size,
        "seed": args.seed,
        "device": str(device),
    }
    if args.data_parallel_size > 1:
        llm_kwargs["data_parallel_size"] = args.data_parallel_size
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # (2) BUILD THE MODEL: load_model + minimized KV init happen here. The
    # worker patches wrap them in record_function stages and dump cumulative
    # snapshots. record_function here brackets the whole construction.
    # The minimal KV pool is sized to this profile's actual workload
    # (batch_size * sequence_length) via capture_batch_size/context_size.
    with record_function("vllm_model_initialization"):
        model = HookedVLLMModel(
            args.model,
            tokenizer,
            dtype=dtype,
            capture_batch_size=args.batch_size,
            capture_context_size=args.sequence_length,
            **llm_kwargs,
        )

    torch.cuda.synchronize(device)
    allocated_after_init = int(torch.cuda.memory_allocated(device))
    reserved_after_init = int(torch.cuda.memory_reserved(device))

    weight_kv = _collect_weight_kv_bytes(model)

    # (3) REAL run_with_cache. Keep the returned activations alive.
    tokens = _build_token_ids(
        model_name=args.model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    with record_function(STAGE_RUN_WITH_CACHE):
        _logits, activations = model.run_with_cache(
            tokens, names_filter=hook_names
        )

    with record_function(STAGE_CAPTURE_MERGE):
        capture_bytes = sum_unique_cuda_storage_bytes(activations)

    torch.cuda.synchronize(device)
    allocated_after_capture = int(torch.cuda.memory_allocated(device))
    reserved_after_capture = int(torch.cuda.memory_reserved(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))

    # (4) DUMP THE FULL-LIFECYCLE SNAPSHOT while activations are still alive.
    full_path = output_dir / f"vllm_full_lifecycle_{_rank_suffix()}.pickle"
    with record_function(STAGE_SNAPSHOT_DUMP):
        dump_memory_snapshot(full_path)

    # activations MUST stay referenced until after the dump above.
    assert activations is not None and len(activations) > 0

    global_rank, tp_rank, pp_rank = _resolve_ranks()
    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "hook_names": hook_names,
        "global_rank": global_rank,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "weight_bytes": weight_kv["weight_bytes"],
        "weight_bytes_param_scan": weight_kv["weight_bytes_param_scan"],
        "kv_cache_bytes": weight_kv["kv_cache_bytes"],
        "capture_bytes": capture_bytes,
        "allocated_before": allocated_before,
        "allocated_after_init": allocated_after_init,
        "allocated_after_capture": allocated_after_capture,
        "reserved_after_init": reserved_after_init,
        "reserved_after_capture": reserved_after_capture,
        "peak_allocated": peak_allocated,
        "full_lifecycle_snapshot": str(full_path),
    }
    summary_path = output_dir / f"vllm_memory_summary_rank{global_rank}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[INFO] wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))

    # (5) STOP RECORDING now that every snapshot is dumped.
    stop_memory_history()

    # (6) VALIDATE the full-lifecycle snapshot automatically.
    report = validate_snapshot(full_path)
    print("[INFO] snapshot validation report:")
    print(json.dumps(report, indent=2))

    weight_kv_bytes = weight_kv["weight_bytes"] + weight_kv["kv_cache_bytes"]
    checks = {
        "snapshot_valid": report["ok"],
        "allocated_after_init >= weight+kv": (
            allocated_after_init >= weight_kv_bytes
        ),
        "allocated_after_capture >= weight+kv+capture": (
            allocated_after_capture >= weight_kv_bytes + capture_bytes
        ),
        "capture_bytes > 0": capture_bytes > 0,
    }
    print("[INFO] acceptance checks:")
    print(json.dumps(checks, indent=2))
    # Keep activations referenced to here so nothing is freed before the dump.
    del activations
    if not all(checks.values()):
        raise SystemExit(f"Acceptance checks failed: {checks}")


if __name__ == "__main__":
    main()
