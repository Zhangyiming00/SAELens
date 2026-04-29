"""Minimal GPU entrypoint for caching activations to disk via CacheActivationsRunner.

Uses VLLMModel by default for fast inference with tensor parallelism.
Supports multi-hook caching via --hook-names.

Example (single hook):
    python3 scripts/run_cache_activations_runner_gpu.py \
        --model-name /data/models/Llama-3.1-8B \
        --dataset-path /tmp/saelens_e2e_ds \
        --hook-name blocks.21.hook_resid_post \
        --training-tokens 1000000 \
        --context-size 2048 \
        --output-path /tmp/cached_activations

Example (multi-hook):
    python3 scripts/run_cache_activations_runner_gpu.py \
        --hook-names blocks.16.hook_resid_post,blocks.21.hook_resid_post,blocks.26.hook_resid_post \
        --training-tokens 204800 \
        --tp-size 2 \
        --output-path checkpoints/cached_activations/1

For multi-GPU TP:
    python3 scripts/run_cache_activations_runner_gpu.py ... --tp-size 4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/models/Llama-3.1-8B")
    parser.add_argument("--dataset-path", default="../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048")
    parser.add_argument("--hook-name", default="blocks.21.hook_resid_post")
    parser.add_argument(
        "--hook-names",
        default=None,
        help="Comma-separated hook names for multi-hook caching.",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--training-tokens", type=int, default=1_000_000)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--model-batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--autocast-lm", action="store_true")
    parser.add_argument("--buffer-size-gb", type=float, default=2.0)
    parser.add_argument(
        "--is-dataset-tokenized",
        dest="is_dataset_tokenized",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-is-dataset-tokenized",
        dest="is_dataset_tokenized",
        action="store_false",
        help="Dataset has a 'text' column (not pre-tokenized).",
    )
    parser.add_argument("--output-path", required=True, help="Directory to save cached activations.")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo-id", default=None, help="Push cached dataset to this HF repo.")
    parser.add_argument("--hf-num-shards", type=int, default=None)
    parser.add_argument(
        "--model-class-name",
        default="VLLMModel",
        choices=["VLLMModel", "HookedTransformer"],
        help="Model backend. VLLMModel is faster for large models.",
    )
    return parser.parse_args()


def _resolve_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is GPU-only.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _resolve_hidden_size(model_name: str) -> int:
    local_files_only = Path(model_name).exists()
    hf_cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    if not hasattr(hf_cfg, "hidden_size"):
        raise ValueError(f"Could not infer hidden_size from model config: {model_name}")
    return int(hf_cfg.hidden_size)


def main() -> None:
    args = parse_args()
    device = _resolve_device()
    d_in = _resolve_hidden_size(args.model_name)

    hook_names: list[str] | None = None
    if args.hook_names is not None:
        hook_names = [h.strip() for h in args.hook_names.split(",") if h.strip()]
        if len(hook_names) <= 1:
            hook_names = None

    model_from_pretrained_kwargs: dict = {}
    if args.model_class_name == "VLLMModel":
        model_from_pretrained_kwargs = {
            "tensor_parallel_size": args.tp_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }

    # With VLLMModel TP>1 (no torchrun), vLLM's MultiprocExecutor assigns
    # devices to workers internally.  Passing "cuda:0" would force all TP
    # workers onto GPU 0.  Use bare "cuda" so vLLM can distribute.
    in_torchrun = "RANK" in os.environ
    if args.model_class_name == "VLLMModel" and args.tp_size > 1 and not in_torchrun:
        model_device = "cuda"
    else:
        model_device = device

    cfg = CacheActivationsRunnerConfig(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        model_class_name=args.model_class_name,
        model_batch_size=args.model_batch_size,
        hook_name=args.hook_name,
        hook_names=hook_names,
        d_in=d_in,
        training_tokens=args.training_tokens,
        context_size=args.context_size,
        new_cached_activations_path=args.output_path,
        shuffle=args.shuffle,
        seed=args.seed,
        dtype=args.dtype,
        device=model_device,
        buffer_size_gb=args.buffer_size_gb,
        autocast_lm=args.autocast_lm,
        streaming=not args.is_dataset_tokenized,
        model_from_pretrained_kwargs=model_from_pretrained_kwargs,
        hf_repo_id=args.hf_repo_id,
        hf_num_shards=args.hf_num_shards,
    )

    print("Starting cache activations runner with:")
    print(f"  device={model_device}")
    print(f"  model={args.model_name} (class={args.model_class_name})")
    print(f"  dataset={args.dataset_path}")
    print(f"  hook={args.hook_name}")
    if hook_names is not None:
        print(f"  hooks={','.join(hook_names)}")
    print(f"  d_in={d_in}")
    print(f"  training_tokens={args.training_tokens}")
    print(f"  context_size={args.context_size}")
    print(f"  model_batch_size={args.model_batch_size}")
    if args.model_class_name == "VLLMModel":
        print(f"  tp_size={args.tp_size}")
    print(f"  output_path={args.output_path}")
    print(f"  n_buffers={cfg.n_buffers}")
    print(f"  n_batches_in_buffer={cfg.n_batches_in_buffer}")

    t0 = time.perf_counter()
    runner = CacheActivationsRunner(cfg)
    cached = runner.run()
    elapsed = time.perf_counter() - t0

    if isinstance(cached, dict):
        first_dataset = next(iter(cached.values()))
        print(
            f"\nDone in {elapsed:.1f}s. Cached {len(first_dataset)} sequences per hook to {args.output_path}"
        )
        for hook_name, dataset in cached.items():
            print(f"  {hook_name}: columns={dataset.column_names}")
    else:
        print(
            f"\nDone in {elapsed:.1f}s. Cached {len(cached)} sequences to {args.output_path}"
        )
        print(f"  columns: {cached.column_names}")


if __name__ == "__main__":
    main()
