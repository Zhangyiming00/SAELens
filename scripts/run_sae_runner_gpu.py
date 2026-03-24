"""Minimal GPU entrypoint for real SAE training via LanguageModelSAETrainingRunner.

This is intentionally simpler than ``scripts/train_tp.py``:
- defaults to single-GPU
- uses the real runner / activation-store / trainer flow
- keeps eval / wandb / compilation off
- aims to be easy to start, not maximally fast

Example:
    python3 scripts/run_sae_runner_gpu.py \
        --model-name /data/models/Llama-3.1-8B \
        --dataset-path /tmp/saelens_e2e_ds \
        --hook-name blocks.1.hook_resid_post \
        --d-sae 8192 \
        --k 32 \
        --training-tokens 128 \
        --train-batch-size-tokens 64 \
        --context-size 32 \
        --max-model-len 128 \
        --output-path /tmp/saelens_runner_gpu_smoke

For multi-GPU shared TP:
    torchrun --nproc_per_node=2 scripts/run_sae_runner_gpu.py ... --tp-size 2
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoConfig

from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/models/Llama-3.1-8B")
    parser.add_argument("--dataset-path", default="../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048")
    parser.add_argument("--hook-name", default="blocks.21.hook_resid_post")
    parser.add_argument("--d-sae", type=int, default=131072)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--vllm-tp-size", type=int, default=1)
    parser.add_argument("--sae-tp-size", type=int, default=1)
    parser.add_argument("--training-tokens", type=int, default=2048*100)
    parser.add_argument("--train-batch-size-tokens", type=int, default=2048)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--store-batch-size-prompts", type=int, default=4)
    parser.add_argument("--n-batches-in-buffer", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--autocast-lm", action="store_true")
    parser.add_argument("--act-store-device", default="cuda")
    parser.add_argument(
        "--output-path",
        default=f"results/res11/saelens_runner_gpu_{datetime.now().strftime('%y%m%d_%H%M%S')}",
    )
    parser.add_argument("--save-mse-every-n-steps", type=int, default=1)
    parser.add_argument("--save-timing-every-n-steps", type=int, default=1)
    parser.add_argument("--synchronize-timing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
    vllm_tp_size = args.vllm_tp_size or args.tp_size
    sae_tp_size = args.sae_tp_size or args.tp_size
    if vllm_tp_size < 1:
        raise ValueError("--vllm-tp-size must be >= 1")
    if sae_tp_size < 1:
        raise ValueError("--sae-tp-size must be >= 1")
    if args.context_size < 1:
        raise ValueError("--context-size must be >= 1")
    if args.train_batch_size_tokens < 1:
        raise ValueError("--train-batch-size-tokens must be >= 1")
    expected_world_size = max(vllm_tp_size, sae_tp_size)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if expected_world_size > 1 and world_size == 1:
        raise ValueError(
            "This configuration requires torchrun. "
            f"Expected WORLD_SIZE={expected_world_size}, got {world_size}."
        )
    if world_size not in (1, expected_world_size):
        raise ValueError(
            f"WORLD_SIZE={world_size} does not match "
            f"max(vllm_tp_size, sae_tp_size)={expected_world_size}."
        )

    min_n_batches_in_buffer = math.ceil(
        args.train_batch_size_tokens / args.context_size
    )
    n_batches_in_buffer = (
        max(2, min_n_batches_in_buffer)
        if args.n_batches_in_buffer is None
        else args.n_batches_in_buffer
    )
    if n_batches_in_buffer * args.context_size < args.train_batch_size_tokens:
        raise ValueError(
            "n_batches_in_buffer * context_size must be >= train_batch_size_tokens"
        )

    device = _resolve_device()
    d_in = _resolve_hidden_size(args.model_name)
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=args.d_sae,
            k=args.k,
            device=device,
            dtype=args.dtype,
        ),
        model_name=args.model_name,
        model_class_name="VLLMModel",
        model_from_pretrained_kwargs={
            "tensor_parallel_size": vllm_tp_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        hook_name=args.hook_name,
        dataset_path=args.dataset_path,
        dataset_trust_remote_code=False,
        streaming=False,
        is_dataset_tokenized=True,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.train_batch_size_tokens,
        store_batch_size_prompts=args.store_batch_size_prompts,
        n_batches_in_buffer=n_batches_in_buffer,
        activations_mixing_fraction=0.0,
        device=device,
        act_store_device=args.act_store_device,
        dtype=args.dtype,
        autocast=args.autocast,
        autocast_lm=args.autocast_lm,
        compile_llm=False,
        compile_sae=False,
        n_eval_batches=0,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=0,
        save_final_checkpoint=False,
        output_path=args.output_path,
        save_mse_every_n_steps=args.save_mse_every_n_steps,
        save_timing_every_n_steps=args.save_timing_every_n_steps,
        synchronize_timing=args.synchronize_timing,
        seed=args.seed,
        verbose=True,
    )

    print("Starting runner with:")
    print(f"  device={device}")
    print(f"  model={args.model_name}")
    print(f"  dataset={args.dataset_path}")
    print(f"  hook={args.hook_name}")
    print(f"  d_in={d_in} d_sae={args.d_sae} k={args.k}")
    print(
        "  training_tokens="
        f"{args.training_tokens} train_batch_size_tokens={args.train_batch_size_tokens}"
    )
    print(
        f"  vllm_tp_size={vllm_tp_size} sae_tp_size={sae_tp_size} "
        f"output_path={args.output_path}"
    )
    if args.save_mse_every_n_steps > 0:
        print(f"  save_mse_every_n_steps={args.save_mse_every_n_steps}")
    if args.save_timing_every_n_steps > 0:
        print(f"  save_timing_every_n_steps={args.save_timing_every_n_steps}")
    if args.synchronize_timing:
        print("  synchronize_timing=True")

    runner = LanguageModelSAETrainingRunner(
        cfg=cfg,
        vllm_tp_size=vllm_tp_size,
        sae_tp_size=sae_tp_size,
        dp_size=1,
    )
    runner.run()


if __name__ == "__main__":
    main()
