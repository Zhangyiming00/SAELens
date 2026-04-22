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
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.constants import SAE_WEIGHTS_FILENAME, TRAINER_STATE_FILENAME
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/models/Llama-3.1-8B")
    parser.add_argument("--dataset-path", default="../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048")
    parser.add_argument("--hook-name", default="blocks.21.hook_resid_post")
    parser.add_argument(
        "--hook-names",
        default=None,
        # default="blocks.16.hook_resid_post,blocks.21.hook_resid_post,blocks.26.hook_resid_post,blocks.31.hook_resid_post",        
        help="Comma-separated hook names for multi-layer independent SAE training.",
    )
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--vllm-tp-size", type=int, default=None)
    parser.add_argument("--sae-tp-size", type=int, default=None)
    parser.add_argument("--vllm-dp-size", type=int, default=1)
    parser.add_argument("--sae-dp-size", type=int, default=1)
    parser.add_argument("--training-tokens", type=int, default=2048*48)
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
        default=f"results/results_1.45_hook31/saelens_runner_gpu_{datetime.now().strftime('%y%m%d_%H%M%S')}",
    )
    parser.add_argument("--save-mse-every-n-steps", type=int, default=1)
    parser.add_argument("--save-timing-every-n-steps", type=int, default=1)
    parser.add_argument("--save-memory-every-n-steps", type=int, default=1)
    parser.add_argument(
        "--synchronize-timing",
        action="store_true",
        default=False,
        help=(
            "If set, force CUDA sync around timed regions for measurement accuracy. "
            "This can perturb runtime; keep disabled for throughput/overlap runs."
        ),
    )
    parser.add_argument("--checkpoint-path", default="checkpoints/")
    parser.add_argument("--n-checkpoints", type=int, default=0)
    parser.add_argument("--save-final-checkpoint", action="store_true", default=True)
    parser.add_argument(
        "--no-save-final-checkpoint",
        dest="save_final_checkpoint",
        action="store_false",
        help="Do not write the final training checkpoint.",
    )
    parser.add_argument(
        "--no-save-final-sae",
        action="store_true",
        help="Do not write final SAE weights to output_path. Useful for smoke tests.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-shard-routing", action="store_true",default=True,
                        help="Use unified shard-routing DP (supports arbitrary vllm_dp:sae_dp ratios).")
    parser.add_argument(
        "--sae-dp-mode",
        default="manual",
        choices=["manual", "ddp", "fsdp"],
        help=(
            "SAE data-parallel sync mode. 'ddp' replicates SAE parameters across DP "
            "replicas; 'fsdp' shards them. Multi-layer SAE defaults manual to ddp; "
            "with --sae-dp-size 1 this runs without DP communication."
        ),
    )
    parser.add_argument(
        "--multi-sae-backward-mode",
        default="combined",
        choices=["combined", "sequential"],
        help=(
            "Multi-layer SAE backward mode. 'combined' keeps all layer graphs "
            "until one backward to allow DDP/FSDP communication overlap with "
            "later-layer backward compute. 'sequential' uses less memory."
        ),
    )
    parser.add_argument(
        "--multi-sae-backward-order",
        default="forward",
        choices=["forward", "reverse", "largest_first"],
        help="Backward order for sequential multi-layer backward mode.",
    )
    parser.add_argument(
        "--multi-sae-stats-sync-mode",
        default="immediate",
        choices=["immediate", "deferred", "periodic"],
        help="When to DP-sync per-layer firing/token stats in multi-layer mode.",
    )
    parser.add_argument(
        "--multi-sae-stats-sync-interval",
        type=int,
        default=1,
        help="Sync interval for --multi-sae-stats-sync-mode=periodic.",
    )
    parser.add_argument(
        "--multi-sae-seed-mode",
        default="same",
        choices=["same", "offset"],
        help=(
            "How to seed independent SAE initializations in multi-layer mode. "
            "'same' matches separate single-layer runs with the same --seed; "
            "'offset' uses seed + hook_index for each hook."
        ),
    )
    parser.add_argument(
        "--ddp-broadcast-buffers",
        dest="ddp_broadcast_buffers",
        action="store_true",
        default=None,
        help="Explicitly set DDP broadcast_buffers=True.",
    )
    parser.add_argument(
        "--no-ddp-broadcast-buffers",
        dest="ddp_broadcast_buffers",
        action="store_false",
        help="Explicitly set DDP broadcast_buffers=False.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        default=None,
        help="Explicitly set DDP find_unused_parameters=True.",
    )
    parser.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Explicitly set DDP find_unused_parameters=False.",
    )
    parser.add_argument(
        "--ddp-gradient-as-bucket-view",
        dest="ddp_gradient_as_bucket_view",
        action="store_true",
        default=None,
        help="Explicitly set DDP gradient_as_bucket_view=True.",
    )
    parser.add_argument(
        "--no-ddp-gradient-as-bucket-view",
        dest="ddp_gradient_as_bucket_view",
        action="store_false",
        help="Explicitly set DDP gradient_as_bucket_view=False.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        dest="ddp_static_graph",
        action="store_true",
        default=None,
        help="Explicitly set DDP static_graph=True.",
    )
    parser.add_argument(
        "--no-ddp-static-graph",
        dest="ddp_static_graph",
        action="store_false",
        help="Explicitly set DDP static_graph=False.",
    )
    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=int,
        default=None,
        help="Explicitly set DDP bucket_cap_mb.",
    )
    parser.add_argument(
        "--ddp-config-strict",
        action="store_true",
        default=False,
        help="Fail fast on invalid DDP config combinations instead of fallback.",
    )
    # Streaming mode (v1): vLLM and SAE processes on separate GPU sets via /dev/shm.
    # Requires sae_dp_size=1. World size = vllm_tp * vllm_dp + sae_tp * 1.
    parser.add_argument(
        "--streaming-mode",
        action="store_true",
        default=False,
        help="Enable streaming_mode v1 (vLLM producers + SAE consumers via /dev/shm).",
    )
    parser.add_argument(
        "--streaming-chunk-size-tokens",
        type=int,
        default=4096,
        help="Tokens per shared-memory chunk in streaming_mode.",
    )
    parser.add_argument(
        "--streaming-num-chunks",
        type=int,
        default=32,
        help="Number of shared-memory chunk slots in streaming_mode.",
    )
    parser.add_argument(
        "--streaming-prefetch-chunks",
        type=int,
        default=2,
        help="Max chunks to acquire per consumer refill in streaming_mode.",
    )
    parser.add_argument(
        "--streaming-buffer-name",
        type=str,
        default="",
        help="Shared buffer name (auto-generated if empty) in streaming_mode.",
    )
    parser.add_argument(
        "--no-streaming-shuffle",
        action="store_false",
        dest="streaming_shuffle",
        help="Disable per-refill token shuffle in streaming_mode.",
    )
    parser.set_defaults(streaming_shuffle=True)
    parser.add_argument(
        "--no-streaming-random-chunks",
        action="store_false",
        dest="streaming_random_chunks",
        help="Disable random chunk selection in streaming_mode (use lowest-index READY slots).",
    )
    parser.set_defaults(streaming_random_chunks=True)
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


def _validate_checkpoint_args(args: argparse.Namespace) -> None:
    if args.n_checkpoints < 0:
        raise ValueError("--n-checkpoints must be >= 0")
    needs_checkpoint_path = args.n_checkpoints > 0 or args.save_final_checkpoint
    if needs_checkpoint_path and args.checkpoint_path is None:
        raise ValueError(
            "--checkpoint-path is required when --n-checkpoints > 0 or "
            "--save-final-checkpoint is set."
        )
    if args.resume_from_checkpoint is None:
        return

    resume_path = Path(args.resume_from_checkpoint)
    if not resume_path.exists():
        raise ValueError(f"--resume-from-checkpoint does not exist: {resume_path}")
    if not resume_path.is_dir():
        raise ValueError(f"--resume-from-checkpoint must be a directory: {resume_path}")
    missing = [
        filename
        for filename in (TRAINER_STATE_FILENAME, SAE_WEIGHTS_FILENAME)
        if not (resume_path / filename).exists()
    ]
    if missing:
        raise ValueError(
            "--resume-from-checkpoint is missing required file(s): "
            + ", ".join(missing)
        )


def _is_writer_rank() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _append_total_runtime_record(
    *,
    output_path_arg: str | None,
    run_id: str,
    total_time_s: float,
    vllm_tp_size: int,
    sae_tp_size: int,
    vllm_dp_size: int,
    sae_dp_size: int,
    sae_dp_mode: str,
    status: str,
    error: str | None,
) -> None:
    if output_path_arg is None or not _is_writer_rank():
        return
    output_path = Path(output_path_arg)
    log_path = output_path.parent / "total_time_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "status": status,
        "total_time_s": total_time_s,
        "config": {
            "vllm_tp_size": vllm_tp_size,
            "sae_tp_size": sae_tp_size,
            "vllm_dp_size": vllm_dp_size,
            "sae_dp_size": sae_dp_size,
            "sae_dp_mode": sae_dp_mode,
        },
    }
    if error is not None:
        record["error"] = error
    with open(log_path, "a") as f:
        json.dump(record, f)
        f.write("\n")
    print(f"[INFO] Appended total runtime record to {log_path}")


def main() -> None:
    args = parse_args()
    _validate_checkpoint_args(args)
    vllm_tp_size = (
        args.vllm_tp_size if args.vllm_tp_size is not None else args.tp_size
    )
    sae_tp_size = args.sae_tp_size if args.sae_tp_size is not None else args.tp_size
    if vllm_tp_size < 1:
        raise ValueError("--vllm-tp-size must be >= 1")
    if sae_tp_size < 1:
        raise ValueError("--sae-tp-size must be >= 1")
    if args.vllm_dp_size < 1:
        raise ValueError("--vllm-dp-size must be >= 1")
    if args.sae_dp_size < 1:
        raise ValueError("--sae-dp-size must be >= 1")
    if args.context_size < 1:
        raise ValueError("--context-size must be >= 1")
    if args.train_batch_size_tokens < 1:
        raise ValueError("--train-batch-size-tokens must be >= 1")
    if args.sae_dp_size > 1 and args.vllm_dp_size == 1:
        print(
            f"[INFO] vllm_dp_size=1, sae_dp_size={args.sae_dp_size} (1:m topology) — "
            "automatically enabling --use-shard-routing."
        )
        args.use_shard_routing = True
    if not args.use_shard_routing and args.vllm_dp_size > 1 and args.sae_dp_size > 1:
        large = max(args.vllm_dp_size, args.sae_dp_size)
        small = min(args.vllm_dp_size, args.sae_dp_size)
        needs_shard_routing = (large % small != 0) or (args.sae_dp_size > args.vllm_dp_size)
        if needs_shard_routing:
            print(
                f"[INFO] vllm_dp_size={args.vllm_dp_size}, sae_dp_size={args.sae_dp_size} "
                "is not an integer-multiple ratio — automatically enabling --use-shard-routing."
            )
            args.use_shard_routing = True
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.streaming_mode:
        if args.sae_dp_size != 1:
            raise ValueError("--streaming-mode requires --sae-dp-size 1")
        args.use_shard_routing = False
        expected_world_size = vllm_tp_size * args.vllm_dp_size + sae_tp_size * 1
        if world_size not in (1, expected_world_size):
            raise ValueError(
                f"streaming_mode: WORLD_SIZE={world_size} does not match "
                f"vllm_tp*vllm_dp + sae_tp = {expected_world_size}."
            )
    else:
        expected_world_size = max(
            vllm_tp_size * args.vllm_dp_size, sae_tp_size * args.sae_dp_size
        )
        if expected_world_size > 1 and world_size == 1:
            raise ValueError(
                "This configuration requires torchrun. "
                f"Expected WORLD_SIZE={expected_world_size}, got {world_size}."
            )
        if world_size not in (1, expected_world_size):
            raise ValueError(
                f"WORLD_SIZE={world_size} does not match "
                f"the expected world size {expected_world_size}."
            )

    training_tokens = args.training_tokens
    train_batch_size_tokens = args.train_batch_size_tokens
    if args.sae_dp_size > 1:
        if args.training_tokens % args.sae_dp_size != 0:
            raise ValueError(
                "--training-tokens must be divisible by --sae-dp-size so each "
                "SAE DP replica gets the same number of local tokens."
            )
        if args.train_batch_size_tokens % args.sae_dp_size != 0:
            raise ValueError(
                "--train-batch-size-tokens must be divisible by --sae-dp-size; "
                "the argument is treated as the global SAE batch size."
            )
        training_tokens = args.training_tokens // args.sae_dp_size
        train_batch_size_tokens = args.train_batch_size_tokens // args.sae_dp_size
        print(
            f"[INFO] sae_dp_size={args.sae_dp_size}: scaling training_tokens "
            f"{args.training_tokens} -> {training_tokens} per replica."
        )
        print(
            f"[INFO] sae_dp_size={args.sae_dp_size}: scaling train_batch_size_tokens "
            f"{args.train_batch_size_tokens} -> {train_batch_size_tokens} per replica "
            f"(global batch stays {args.train_batch_size_tokens})."
        )

    min_n_batches_in_buffer = math.ceil(
        train_batch_size_tokens / args.context_size
    )
    n_batches_in_buffer = (
        max(2, min_n_batches_in_buffer)
        if args.n_batches_in_buffer is None
        else args.n_batches_in_buffer
    )
    if n_batches_in_buffer * args.context_size < train_batch_size_tokens:
        raise ValueError(
            "n_batches_in_buffer * context_size must be >= train_batch_size_tokens"
        )

    output_path = None if args.no_save_final_sae else args.output_path

    device = _resolve_device()
    d_in = _resolve_hidden_size(args.model_name)
    hook_names = (
        [hook.strip() for hook in args.hook_names.split(",") if hook.strip()]
        if args.hook_names is not None
        else None
    )
    if hook_names is not None and len(hook_names) <= 1:
        hook_names = None
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
        hook_names=hook_names,
        dataset_path=args.dataset_path,
        dataset_trust_remote_code=False,
        streaming=False,
        is_dataset_tokenized=True,
        context_size=args.context_size,
        training_tokens=training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        store_batch_size_prompts=args.store_batch_size_prompts,
        n_batches_in_buffer=n_batches_in_buffer,
        activations_mixing_fraction=0.5,
        device=device,
        act_store_device=args.act_store_device,
        dtype=args.dtype,
        autocast=args.autocast,
        autocast_lm=args.autocast_lm,
        compile_llm=False,
        compile_sae=False,
        n_eval_batches=0,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        save_final_checkpoint=args.save_final_checkpoint,
        output_path=output_path,
        save_mse_every_n_steps=args.save_mse_every_n_steps,
        save_timing_every_n_steps=args.save_timing_every_n_steps,
        save_memory_every_n_steps=args.save_memory_every_n_steps,
        synchronize_timing=args.synchronize_timing,
        seed=args.seed,
        verbose=True,
        sae_dp_mode=args.sae_dp_mode,
        multi_sae_backward_mode=args.multi_sae_backward_mode,
        multi_sae_backward_order=args.multi_sae_backward_order,
        multi_sae_stats_sync_mode=args.multi_sae_stats_sync_mode,
        multi_sae_stats_sync_interval=args.multi_sae_stats_sync_interval,
        multi_sae_seed_mode=args.multi_sae_seed_mode,
        ddp_broadcast_buffers=args.ddp_broadcast_buffers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        ddp_static_graph=args.ddp_static_graph,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        ddp_config_strict=args.ddp_config_strict,
        streaming_mode=args.streaming_mode,
        streaming_chunk_size_tokens=args.streaming_chunk_size_tokens,
        streaming_num_chunks=args.streaming_num_chunks,
        streaming_prefetch_chunks=args.streaming_prefetch_chunks,
        streaming_buffer_name=args.streaming_buffer_name,
        streaming_shuffle=args.streaming_shuffle,
        streaming_random_chunks=args.streaming_random_chunks,
    )

    print("Starting runner with:")
    print(f"  device={device}")
    print(f"  model={args.model_name}")
    print(f"  dataset={args.dataset_path}")
    print(f"  hook={args.hook_name}")
    if hook_names is not None:
        print(f"  hooks={','.join(hook_names)}")
    print(f"  d_in={d_in} d_sae={args.d_sae} k={args.k}")
    print(
        "  training_tokens="
        f"{args.training_tokens} train_batch_size_tokens={args.train_batch_size_tokens} "
        f"(per_replica={training_tokens}/{train_batch_size_tokens})"
    )
    print(
        f"  vllm_tp_size={vllm_tp_size} vllm_dp_size={args.vllm_dp_size} "
        f"sae_tp_size={sae_tp_size} sae_dp_size={args.sae_dp_size} "
        f"output_path={output_path}"
    )
    print(f"  sae_dp_mode={cfg.sae_dp_mode}")
    if hook_names is not None:
        print(f"  multi_sae_backward_mode={cfg.multi_sae_backward_mode}")
        print(f"  multi_sae_backward_order={cfg.multi_sae_backward_order}")
        print(f"  multi_sae_stats_sync_mode={cfg.multi_sae_stats_sync_mode}")
        print(f"  multi_sae_stats_sync_interval={cfg.multi_sae_stats_sync_interval}")
        print(f"  multi_sae_seed_mode={cfg.multi_sae_seed_mode}")
    if args.ddp_broadcast_buffers is not None:
        print(f"  ddp_broadcast_buffers={args.ddp_broadcast_buffers}")
    if args.ddp_find_unused_parameters is not None:
        print(f"  ddp_find_unused_parameters={args.ddp_find_unused_parameters}")
    if args.ddp_gradient_as_bucket_view is not None:
        print(f"  ddp_gradient_as_bucket_view={args.ddp_gradient_as_bucket_view}")
    if args.ddp_static_graph is not None:
        print(f"  ddp_static_graph={args.ddp_static_graph}")
    if args.ddp_bucket_cap_mb is not None:
        print(f"  ddp_bucket_cap_mb={args.ddp_bucket_cap_mb}")
    if args.ddp_config_strict:
        print("  ddp_config_strict=True")
    if args.save_mse_every_n_steps > 0:
        print(f"  save_mse_every_n_steps={args.save_mse_every_n_steps}")
    if args.save_timing_every_n_steps > 0:
        print(f"  save_timing_every_n_steps={args.save_timing_every_n_steps}")
    if args.synchronize_timing:
        print("  synchronize_timing=True")
    if args.checkpoint_path is not None:
        print(f"  checkpoint_path={args.checkpoint_path}")
    if args.n_checkpoints > 0:
        print(f"  n_checkpoints={args.n_checkpoints}")
    if args.save_final_checkpoint:
        print("  save_final_checkpoint=True")
    if args.resume_from_checkpoint is not None:
        print(f"  resume_from_checkpoint={args.resume_from_checkpoint}")

    runner = LanguageModelSAETrainingRunner(
        cfg=cfg,
        resume_from_checkpoint=args.resume_from_checkpoint,
        vllm_tp_size=vllm_tp_size,
        sae_tp_size=sae_tp_size,
        vllm_dp_size=args.vllm_dp_size,
        sae_dp_size=args.sae_dp_size,
        use_shard_routing=args.use_shard_routing,
        streaming_mode=args.streaming_mode,
    )
    run_id = Path(args.output_path).name if args.output_path is not None else "unknown_run_id"
    run_t0 = time.perf_counter()
    run_status = "ok"
    run_error: str | None = None
    try:
        runner.run()
    except Exception as exc:
        run_status = "error"
        run_error = repr(exc)
        raise
    finally:
        _append_total_runtime_record(
            output_path_arg=args.output_path,
            run_id=run_id,
            total_time_s=time.perf_counter() - run_t0,
            vllm_tp_size=vllm_tp_size,
            sae_tp_size=sae_tp_size,
            vllm_dp_size=args.vllm_dp_size,
            sae_dp_size=args.sae_dp_size,
            sae_dp_mode=args.sae_dp_mode,
            status=run_status,
            error=run_error,
        )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
