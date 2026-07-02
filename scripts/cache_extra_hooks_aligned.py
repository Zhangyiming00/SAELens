"""Cache activations for extra hooks, row-aligned to an existing split cache.

Multi-hook cached training reads every hook dir at the same row index and
assumes all hooks share the same token sequence per row. To add new layers to
an existing cache we therefore must reuse the *exact* token_ids already stored
(in their stored order, shuffle off) rather than re-running the shuffled input
pipeline. Activations are deterministic per sequence, so this yields rows that
align with the existing hooks one-for-one.

Run, verify token_ids match, then merge the new hook dirs into the existing
cache and extend its manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.training.multi_sae_trainer import sanitize_hook_name_for_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="/data/models/Llama-3.1-8B")
    p.add_argument(
        "--existing-cache",
        default="/home/zhangyiming/datasets/cached_activation.new_20260509_140126",
    )
    p.add_argument(
        "--ref-hook",
        default="blocks.21.hook_resid_post",
        help="Existing hook whose token_ids define the row order to reuse.",
    )
    p.add_argument(
        "--new-hooks",
        default="blocks.11.hook_resid_post,blocks.26.hook_resid_post",
    )
    p.add_argument("--context-size", type=int, default=2048)
    p.add_argument("--model-batch-size", type=int, default=16)
    p.add_argument("--d-in", type=int, default=4096)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--output-path", required=True, help="Fresh temp dir for new hooks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    existing = Path(args.existing_cache)
    new_hooks = [h.strip() for h in args.new_hooks.split(",") if h.strip()]

    ref_dir = existing / sanitize_hook_name_for_path(args.ref_hook)
    ref_ds = load_from_disk(str(ref_dir))
    n_rows = ref_ds.num_rows
    print(f"Reference {args.ref_hook}: {n_rows} rows, ctx={args.context_size}")

    # Build an in-order, tokenized override dataset from the existing token_ids.
    token_ids = [
        np.asarray(t, dtype=np.int64).tolist() for t in ref_ds["token_ids"]
    ]
    override = Dataset.from_dict({"input_ids": token_ids})

    cfg = CacheActivationsRunnerConfig(
        dataset_path="__in_memory_token_ids__",
        model_name=args.model_name,
        model_class_name="VLLMModel",
        model_batch_size=args.model_batch_size,
        hook_name=new_hooks[0],
        hook_names=new_hooks,
        d_in=args.d_in,
        training_tokens=n_rows * args.context_size,
        context_size=args.context_size,
        new_cached_activations_path=args.output_path,
        shuffle=False,
        dtype=args.dtype,
        device="cuda",
        streaming=False,
        model_from_pretrained_kwargs={
            "tensor_parallel_size": args.tp_size,
            "max_model_len": args.context_size + 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )

    runner = CacheActivationsRunner(cfg, override_dataset=override)
    print(runner)
    cached = runner.run()
    print(f"Cached new hooks to {args.output_path}: {list(cached.keys())}")


if __name__ == "__main__":
    main()
