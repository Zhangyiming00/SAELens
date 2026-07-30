"""Verify the real behaviour of vLLM's minimized (activation-capture) KV pool.

Runs ONE case: builds a HookedVLLMModel engine (via the same vLLM path the SAE
activation store uses), records requested-vs-resolved config and the physical
KV storage footprint at every lifecycle point, submits one or more real
requests through vLLM's normal generate() path, and — with
``SAELENS_KV_POOL_INSTRUMENT=1`` — logs the per-scheduler-iteration KV block
manager state to ``events.jsonl``.

It answers, empirically:
  * how ``num_gpu_blocks`` is determined in capture mode (formula check),
  * whether the physical KV pool ever expands at runtime,
  * what happens under chunked prefill and block contention (preempt vs. lower
    concurrency vs. init failure),
  * release semantics after a request finishes.

Instrumentation is entirely env-gated and patches nothing in the vendored vLLM
fork. Nothing here is imported by the SAE trainer.

Outputs (under ``--output-dir``):
  events.jsonl   — one record per scheduled request per scheduler iteration
  summary.json   — resolved config, lifecycle memory records, verdicts

Multiple cases are aggregated into ``all_cases.csv`` by ``run_kv_matrix.sh``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

# Instrumentation must be importable and its patches installed BEFORE the vLLM
# engine is built, so schedule()/KV-alloc patches are in place from the start.
from sae_lens import kv_pool_instrument

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_false",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--case-name", default="case")
    parser.add_argument("--output-dir", default="results/kv_pool_verification")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_token_ids(
    *, model_name: str, batch_size: int, context_len: int, seed: int
) -> torch.Tensor:
    """Valid, in-range token ids of shape exactly (batch_size, context_len)."""
    cfg = AutoConfig.from_pretrained(
        model_name, local_files_only=Path(model_name).exists()
    )
    vocab_size = int(getattr(cfg, "vocab_size"))
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_len),
        generator=generator,
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# Engine internals access (in-process; TP1 / external_launcher)
# ---------------------------------------------------------------------------


def _get_engine_core(model: Any) -> Any:
    """Reach the in-process EngineCore (owns scheduler, vllm_config, executor).

    Chain: LLM.llm_engine.engine_core (InprocClient) .engine_core (EngineCore).
    """
    client = model.llm.llm_engine.engine_core
    return getattr(client, "engine_core", client)


def _resolved_config(model: Any) -> dict[str, Any]:
    core = _get_engine_core(model)
    vcfg = core.vllm_config
    sched = vcfg.scheduler_config
    cache = vcfg.cache_config
    mcfg = vcfg.model_config
    return {
        "resolved_max_num_batched_tokens": int(sched.max_num_batched_tokens),
        "resolved_max_num_seqs": int(sched.max_num_seqs),
        "resolved_max_model_len": int(mcfg.max_model_len),
        "resolved_block_size": int(cache.block_size),
        "resolved_num_gpu_blocks": int(cache.num_gpu_blocks or 0),
        "resolved_enable_chunked_prefill": bool(
            sched.enable_chunked_prefill
        ),
    }


def _model_kv_dims(model_name: str) -> dict[str, Any]:
    """Read attention dims from HF config for the KV-bytes cross-check."""
    cfg = AutoConfig.from_pretrained(
        model_name, local_files_only=Path(model_name).exists()
    )
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    num_kv_heads = int(
        getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))
    )
    hidden = int(getattr(cfg, "hidden_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = int(getattr(cfg, "head_dim", hidden // num_heads))
    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def _kv_storage_from_runner(model: Any) -> dict[str, Any]:
    """Read the physical KV storage stats from the worker's model_runner."""
    core = _get_engine_core(model)
    try:
        runner = core.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        try:
            runner = core.model_executor.driver_worker.model_runner
        except AttributeError:
            return {
                "kv_storage_bytes": 0,
                "kv_storage_data_ptrs": [],
                "kv_storage_count": 0,
            }
    return kv_pool_instrument.kv_storage_stats(getattr(runner, "kv_caches", None))


def _pool_and_mem(model: Any) -> dict[str, Any]:
    core = _get_engine_core(model)
    out: dict[str, Any] = {}
    try:
        pool = core.scheduler.kv_cache_manager.block_pool
        total = int(pool.num_gpu_blocks)
        free = int(pool.get_num_free_blocks())
        out.update(
            {"total_blocks": total, "free_blocks": free, "used_blocks": total - free}
        )
    except Exception:
        out.update({"total_blocks": 0, "free_blocks": 0, "used_blocks": 0})
    out.update(kv_pool_instrument.cuda_mem_stats())
    out.update(_kv_storage_from_runner(model))
    return out


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This verification script is GPU-only.")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    out_dir = Path(args.output_dir) / args.case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    summary_path = out_dir / "summary.json"

    # Install instrumentation + fresh recorder BEFORE building the engine so the
    # schedule()/KV-alloc patches capture from the very first iteration.
    instrumented = kv_pool_instrument.maybe_install()
    kv_pool_instrument.reset_recorder(events_path if instrumented else None)

    dtype = torch.bfloat16
    block_size = args.block_size

    summary: dict[str, Any] = {
        "case_name": args.case_name,
        "instrumented": instrumented,
        "requested": {
            "model": args.model,
            "requested_max_num_batched_tokens": args.max_num_batched_tokens,
            "requested_max_model_len": args.max_model_len,
            "requested_max_num_seqs": args.max_num_seqs,
            "requested_block_size": args.block_size,
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "max_new_tokens": args.max_new_tokens,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "lifecycle": {},
        "init_result": "pending",
        "run_result": "pending",
        "failure_stage": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }

    # -- engine initialization ------------------------------------------------
    from sae_lens.vllm_model import HookedVLLMModel

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": block_size,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "seed": args.seed,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = HookedVLLMModel(args.model, tokenizer, dtype=dtype, **llm_kwargs)
    except Exception as exc:
        summary["init_result"] = "failed"
        summary["failure_stage"] = "engine initialization"
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[RESULT] init FAILED: {type(exc).__name__}: {exc}")
        print(f"[INFO] summary: {summary_path}")
        raise SystemExit(0)

    summary["init_result"] = "ok"
    summary["resolved"] = _resolved_config(model)
    summary["lifecycle"]["after_engine_init"] = _pool_and_mem(model)

    # Physical-formula cross-check.
    resolved = summary["resolved"]
    rmbt = resolved["resolved_max_num_batched_tokens"]
    rbs = resolved["resolved_block_size"]
    expected_blocks = math.ceil(rmbt / rbs) + 1
    dims = _model_kv_dims(args.model)
    kv_dtype_bytes = torch.finfo(dtype).bits // 8
    # KV bytes = num_blocks * block_size * layers * kv_heads * (d_k+d_v) * dtype
    expected_kv_bytes = (
        expected_blocks
        * rbs
        * dims["num_layers"]
        * dims["num_kv_heads"]
        * (2 * dims["head_dim"])
        * kv_dtype_bytes
    )
    actual_kv_bytes = summary["lifecycle"]["after_engine_init"]["kv_storage_bytes"]
    summary["formula_check"] = {
        "expected_blocks": expected_blocks,
        "actual_num_gpu_blocks": resolved["resolved_num_gpu_blocks"],
        "blocks_match": resolved["resolved_num_gpu_blocks"] == expected_blocks,
        "expected_kv_bytes": expected_kv_bytes,
        "actual_kv_storage_bytes": actual_kv_bytes,
        "kv_bytes_match": expected_kv_bytes == actual_kv_bytes,
        "kv_dims": dims,
        "kv_dtype_bytes": kv_dtype_bytes,
    }

    # -- run requests ---------------------------------------------------------
    tokens = _build_token_ids(
        model_name=args.model,
        batch_size=args.batch_size,
        context_len=args.context_len,
        seed=args.seed,
    )
    from vllm import SamplingParams

    prompts = [
        {"prompt_token_ids": tokens[i].tolist()} for i in range(args.batch_size)
    ]
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    summary["lifecycle"]["before_request"] = _pool_and_mem(model)
    try:
        model.llm.generate(prompts, sampling, use_tqdm=False)
        summary["run_result"] = "ok"
    except Exception as exc:
        summary["run_result"] = "failed"
        summary["failure_stage"] = "run (scheduler/attention/completion)"
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        summary["traceback"] = traceback.format_exc()

    summary["lifecycle"]["after_request_finish"] = _pool_and_mem(model)

    # -- second request (reuse / release check) -------------------------------
    if summary["run_result"] == "ok":
        try:
            model.llm.generate(prompts, sampling, use_tqdm=False)
            summary["lifecycle"]["after_second_request"] = _pool_and_mem(model)
        except Exception as exc:  # pragma: no cover - defensive
            summary["second_request_error"] = f"{type(exc).__name__}: {exc}"

    # -- recorder-derived aggregates ------------------------------------------
    recorder = kv_pool_instrument.get_recorder()
    if recorder is not None:
        summary["instrumentation"] = {
            "num_iterations": recorder.iteration,
            "preemption_count": recorder.preemption_count,
            "max_used_blocks": recorder.max_used_blocks,
            "max_concurrent_running_requests": recorder.max_running_requests,
            "kv_physical_allocation_call_count": (
                recorder.kv_physical_allocation_call_count
            ),
        }

    # -- dynamic-expansion verdict (evidence-based) ---------------------------
    life = summary["lifecycle"]
    init_rec = life.get("after_engine_init", {})
    max_rec = life.get("after_second_request", life.get("after_request_finish", {}))
    kv_alloc_calls = (
        summary.get("instrumentation", {}).get(
            "kv_physical_allocation_call_count", 0
        )
    )
    dyn = {
        "num_gpu_blocks_changed": (
            init_rec.get("total_blocks") != max_rec.get("total_blocks")
        ),
        "kv_storage_bytes_changed": (
            init_rec.get("kv_storage_bytes") != max_rec.get("kv_storage_bytes")
        ),
        "kv_storage_ptrs_changed": (
            init_rec.get("kv_storage_data_ptrs")
            != max_rec.get("kv_storage_data_ptrs")
        ),
        "kv_alloc_calls_after_init": max(0, kv_alloc_calls - 1),
    }
    dyn["dynamic_expansion"] = bool(
        dyn["num_gpu_blocks_changed"]
        or dyn["kv_storage_bytes_changed"]
        or dyn["kv_storage_ptrs_changed"]
        or dyn["kv_alloc_calls_after_init"] > 0
    )
    summary["dynamic_expansion_verdict"] = dyn

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[RESULT] init={summary['init_result']} run={summary['run_result']} "
          f"dynamic_expansion={dyn['dynamic_expansion']} "
          f"preemptions={summary.get('instrumentation', {}).get('preemption_count')}")
    print(f"[INFO] summary: {summary_path}")
    print(f"[INFO] events: {events_path}")


if __name__ == "__main__":
    main()
