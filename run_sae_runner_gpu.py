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

Streaming startup removes stale topology-switch shared-memory files by default.
Pass ``--no_cleanup`` with ``--streaming-mode`` to preserve them.

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
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
from sae_lens.training.multi_sae_trainer import MULTI_SAE_MANIFEST_FILENAME
from sae_lens.topology_control import BufferParams, read_control_state, write_control_state
from sae_lens.util import extract_layer_from_tlens_hook_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name","--model", default="/root/models/Llama-3.1-8B")
    parser.add_argument("--dataset-path","--dataset", default="/mnt/L202500425/dzl/datasets/wikitext2_tokenized_llama31_ctx2048")
    parser.add_argument("--hook-name","--hook", default="blocks.21.hook_resid_post")
    parser.add_argument("--hook-names","--hooks",
        # default=None,
        default="blocks.21.hook_resid_post,blocks.31.hook_resid_post",
        help="Comma-separated hook names for multi-layer independent SAE training.",
    )
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--use-sparse-activations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use COO sparse Top-K activations during SAE training (default: disabled).",
    )
    parser.add_argument(
        "--no-rescale-acts-by-decoder-norm",
        dest="rescale_acts_by_decoder_norm",
        action="store_false",
        default=True,
        help="Disable TopK rescale_acts_by_decoder_norm (default: enabled).",
    )
    parser.add_argument("--tp-size", "-tp", type=int, default=1)
    parser.add_argument("--vllm-tp-size","-vtp", type=int, default=None)
    parser.add_argument("--sae-tp-size", "-stp", type=int, default=None)
    parser.add_argument("--vllm-dp-size","-vdp", type=int, default=1)
    parser.add_argument("--sae-dp-size", "-sdp", type=int, default=1)
    # DP convenience aliases. These are normalized after parsing so the original
    # --sae-dp-size + --sae-dp-mode interface remains fully supported.
    parser.add_argument(
        "--ddp","-ddp",
        action="store_true",
        help="Shortcut for --sae-dp-mode ddp.",
    )
    parser.add_argument(
        "--fsdp","-fsdp",
        action="store_true",
        help="Shortcut for --sae-dp-mode fsdp.",
    )
    parser.add_argument(
        "-sddp",
        dest="sae_ddp_size",
        type=int,
        default=None,
        metavar="N",
        help="Shortcut for --sae-dp-size N --sae-dp-mode ddp. Also accepts -sddpN.",
    )
    parser.add_argument(
        "-sfsdp",
        dest="sae_fsdp_size",
        type=int,
        default=None,
        metavar="N",
        help="Shortcut for --sae-dp-size N --sae-dp-mode fsdp. Also accepts -sfsdpN.",
    )
    parser.add_argument("--sae-pp-size", "-spp", type=int, default=1)
    parser.add_argument("--training-tokens", type=int, default=2048*4096)
    parser.add_argument("--train-batch-size-tokens", type=int, default=2048)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument(
        "--store-batch-size-prompts",
        type=int,
        default=8,
        help=(
            "Baseline number of prompts fetched by one vLLM DP producer per batch. "
            "In non-streaming mode it is automatically scaled by "
            "sae_dp_size / vllm_dp_size by default."
        ),
    )
    parser.add_argument(
        "--auto-scale-store-batch-size-prompts",
        dest="auto_scale_store_batch_size_prompts",
        action="store_true",
        default=True,
        help=(
            "Automatically scale --store-batch-size-prompts for non-streaming DP "
            "topologies (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-auto-scale-store-batch-size-prompts",
        dest="auto_scale_store_batch_size_prompts",
        action="store_false",
        help="Keep --store-batch-size-prompts unchanged in all non-streaming topologies.",
    )
    parser.add_argument("--n-batches-in-buffer", type=int, default=None)
    parser.add_argument("--activations-mixing-fraction", type=float, default=0.5)
    parser.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help=(
            "Training steps before a feature is considered dead for TopK aux loss. "
            "Use a negative value for profiling the all-dead aux-loss worst case."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--autocast-lm", action="store_true")
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
    parser.add_argument("--act-store-device", default="cuda")
    parser.add_argument(
        "--output-path",
        default=f"results/results_2.3_H5_asynctpddp_long1/saelens_runner_gpu_{datetime.now().strftime('%y%m%d_%H%M%S')}",
    )
    parser.add_argument("--save-mse-every-n-steps", type=int, default=32)
    parser.add_argument("--save-timing-every-n-steps", type=int, default=512)
    parser.add_argument("--save-memory-every-n-steps", type=int, default=512)
    parser.add_argument(
        "--save-vllm-memory-every-n-steps",
        type=int,
        default=0,
        help=(
            "Save vLLM decoder substage memory records every N capture calls. "
            "0 disables it. Writes vllm_memory_history_rank{rank}.jsonl."
        ),
    )
    parser.add_argument(
        "--vllm-memory-probe-layer",
        type=int,
        default=None,
        help=(
            "Decoder layer index for vLLM substage memory profiling "
            "(ln1/attn/ln2/mlp). Defaults to the layer in --hook-name when "
            "--save-vllm-memory-every-n-steps is enabled."
        ),
    )
    parser.add_argument(
        "--record-memory-empty-cache",
        action="store_true",
        help="When memory profiling is on, call torch.cuda.empty_cache() before "
        "each per-phase snapshot so reserved/driver_used reflect current live "
        "tensors, not the historical watermark. Adds tens of ms per phase; "
        "profiling-only.",
    )
    parser.add_argument(
        "--record-memory-timeline-step",
        type=int,
        default=-1,
        help="When >= 0, record the full PyTorch alloc/free history (every "
        "event with its Python stack, plus the peak-moment snapshot) for that "
        "single training step and dump it to memory_timeline_rank{rank}.pickle "
        "in output_path. Open at https://pytorch.org/memory_viz. Profiling-only; "
        "-1 disables.",
    )
    parser.add_argument(
        "--record-vllm-memory-timeline-step",
        type=int,
        default=-1,
        help="When >= 0, record the full vLLM forward allocator history for that "
        "single activation-generation step and dump it to "
        "memory_timeline_vllm[_tp{rank}].pickle in output_path (one per TP rank). "
        "Requires --save-vllm-memory-every-n-steps > 0 (live vLLM). "
        "Open at https://pytorch.org/memory_viz. -1 disables.",
    )
    parser.add_argument(
        "--append-history-logs",
        action="store_true",
        default=False,
        help=(
            "Append mse/timing history logs instead of truncating them at trainer "
            "startup. Topology-supervisor phases use this to preserve each phase."
        ),
    )
    parser.add_argument(
        "--synchronize-timing",
        action="store_true",
        default=False,
        help=(
            "If set, force CUDA sync around timed regions for measurement accuracy. "
            "This can perturb runtime; keep disabled for throughput/overlap runs."
        ),
    )
    parser.add_argument(
        "--step-window-profile-start-step",
        type=int,
        default=65,
        help=(
            "First step of the first step-window profiling window (1-based). "
            "Windows are contiguous, so --step-window-profile-start-step 11 with "
            "--step-window-profile-window-steps 20 --step-window-profile-window-count 4 "
            "measures steps 11-30, 31-50, 51-70 and 71-90. The device is "
            "synchronized only at each window's two boundaries, so overlap inside "
            "a window is preserved and the recorded interval is a true end-to-end "
            "wall time. Pick a start step past warmup (step 1 is far slower). "
            "0 disables it."
        ),
    )
    parser.add_argument(
        "--step-window-profile-window-steps",
        type=int,
        default=64,
        help="Steps per step-window profiling window.",
    )
    parser.add_argument(
        "--step-window-profile-window-count",
        type=int,
        default=14,
        help="Number of consecutive step-window profiling windows to record.",
    )
    parser.add_argument(
        "--step-window-profile-vllm-start-step",
        type=int,
        default=0,
        help=(
            "Override --step-window-profile-start-step on vLLM producer ranks in "
            "streaming/split-role modes, where a step is one produced chunk or "
            "batch rather than an SAE step. In co-located mode vLLM runs inside "
            "the SAE step and needs no separate setting. 0 reuses the shared value."
        ),
    )
    parser.add_argument(
        "--step-window-profile-vllm-window-steps",
        type=int,
        default=0,
        help="Override --step-window-profile-window-steps on vLLM producer ranks.",
    )
    parser.add_argument(
        "--step-window-profile-vllm-window-count",
        type=int,
        default=0,
        help="Override --step-window-profile-window-count on vLLM producer ranks.",
    )
    parser.add_argument("--checkpoint-path", default="checkpoints/1.60/")
    parser.add_argument(
        "--checkpoint-storage",
        choices=["memory", "disk"],
        default="memory",
        help="Storage backend for quiesce checkpoints in topology-supervisor mode.",
    )
    parser.add_argument(
        "--quiesce-checkpoint-path",
        default=None,
        help=(
            "Checkpoint base path used for quiesce checkpoints when "
            "--checkpoint-storage=memory. The supervisor supplies this."
        ),
    )
    parser.add_argument("--n-checkpoints", type=int, default=0)
    parser.add_argument(
        "--save-final-checkpoint",
        dest="save_final_checkpoint",
        action="store_true",
        default=False,
        help="Write the final training checkpoint (default: disabled).",
    )
    parser.add_argument(
        "--no-save-final-checkpoint", "--no-final-checkpoint", "-nsfc", "-nfc",
        dest="save_final_checkpoint",
        action="store_false",
        help="Do not write the final training checkpoint (default; short: -nsfc / -nfc).",
    )
    parser.add_argument(
        "--no-save-final-sae", "--no-final-sae", "-nsfs", "-nfs",
        dest="no_save_final_sae",
        action="store_true",
        default=False,
        help="Do not write final SAE weights to output_path (short: -nsfs / -nfs).",
    )
    parser.add_argument(
        "--no-save-final", "-nsf",
        action="store_true",
        default=False,
        help="Disable both the final checkpoint and final SAE save.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-shard-routing", action="store_true",default=True,
                        help="Use unified shard-routing DP (supports arbitrary vllm_dp:sae_dp ratios).")
    parser.add_argument(
        "--sae-dp-mode",
        default=None,
        choices=["manual", "ddp", "fsdp"],
        help=(
            "SAE data-parallel sync mode. 'ddp' replicates SAE parameters across DP "
            "replicas; 'fsdp' shards them. Default: ddp. Use "
            "'--sae-dp-mode manual' explicitly to select manual mode."
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
        "--multi-sae-distributed-architecture",
        default="unified_multi_hook",
        choices=["legacy_per_hook_wrapper", "unified_multi_hook"],
        help=(
            "Multi-layer SAE distributed wrapper architecture. The default "
            "unified_multi_hook trains through one MultiHookSAE owner and enables "
            "cross-hook TP wavefront forward when TP>1. legacy_per_hook_wrapper "
            "keeps the old path. FSDP automatically falls back to legacy."
        ),
    )
    parser.add_argument(
        "--multi-sae-tp-phase-fence",
        default="auto",
        choices=["auto", "always", "off"],
        help=(
            "Host-visible fence between SAE-TP work and a different NCCL phase. "
            "auto fences only for cross-hook TP when a distinct DDP group or "
            "co-located producer path can interleave; always forces the fence; "
            "off disables it."
        ),
    )
    parser.add_argument(
        "--multi-sae-optimizer-overlap",
        default="on",
        choices=["off", "on", "non_tp_only"],
        help=(
            "Experimental per-hook DDP bucket reduction -> optimizer overlap. "
            "Buckets launch during combined backward; 'on' also supports SAE-TP "
            "through CPU shared-memory TP post; 'non_tp_only' keeps TP x DDP on "
            "the normal optimizer path."
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
        default=True,
        help=(
            "Use DDP bucket-backed gradients (the effective default, avoiding a "
            "second gradient-sized allocation)."
        ),
    )
    parser.add_argument(
        "--no-ddp-gradient-as-bucket-view",
        dest="ddp_gradient_as_bucket_view",
        action="store_false",
        help=(
            "Disable bucket-backed gradients on the standard DDP/off path. "
            "The optimizer-overlap path requires and forces bucket views."
        ),
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
    parser.add_argument(
        "--fsdp-backward-prefetch",
        default="backward_pre",
        choices=["backward_pre", "backward_post", "none"],
        help=(
            "FSDP backward prefetch policy. Use 'none' to disable FSDP's default "
            "BACKWARD_PRE full-parameter prefetch/caching behavior."
        ),
    )
    parser.add_argument(
        "--fsdp-forward-prefetch",
        dest="fsdp_forward_prefetch",
        action="store_true",
        default=False,
        help="Enable FSDP forward prefetch for unified multi-hook static execution order.",
    )
    parser.add_argument(
        "--no-fsdp-forward-prefetch",
        dest="fsdp_forward_prefetch",
        action="store_false",
        help="Disable FSDP forward prefetch.",
    )
    parser.add_argument(
        "--fsdp-sharding-strategy",
        default="shard_grad_op",
        choices=["shard_grad_op", "full_shard", "no_shard"],
        help=(
            "FSDP sharding strategy. 'shard_grad_op' keeps full parameters after "
            "forward and shards gradients/optimizer state, avoiding a backward "
            "parameter all-gather. 'full_shard' reshards parameters after forward."
        ),
    )
    # Streaming mode (v1): vLLM and SAE processes on separate GPU sets via /dev/shm.
    # Requires sae_dp_size=1. World size = vllm_tp * vllm_dp + sae_tp * 1.
    parser.add_argument(
        "--streaming-mode", "--streaming",
        action="store_true",
        default=False,
        help="Enable streaming_mode v1 (vLLM producers + SAE consumers via /dev/shm).",
    )
    parser.add_argument(
        "--no_cleanup",
        "--no-cleanup",
        dest="streaming_cleanup",
        action="store_false",
        default=True,
        help=(
            "Disable topology-switch runner shared-memory cleanup at streaming "
            "startup (cleanup is enabled by default)."
        ),
    )
    parser.add_argument(
        "--streaming-chunk-size-tokens",
        type=int,
        default=8192,
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
        "--streaming-mix-chunks",
        type=int,
        default=8,
        help=(
            "Consumer-local rolling mixing window in shared-memory chunks. "
            "Set 0 to disable and serve each prefetch pool directly."
        ),
    )
    parser.add_argument(
        "--streaming-mix-fraction",
        type=float,
        default=0.5,
        help="Fraction of the local rolling mix window kept for the next refill.",
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
    parser.add_argument(
        "--streaming-use-gpu-direct",
        action="store_true",
        default=False,
        help="Enable GPU direct NCCL streaming (vLLM→SAE GPU-to-GPU transfer, no CPU copy).",
    )
    parser.add_argument(
        "--streaming-staging-queue-capacity",
        type=int,
        default=4,
        help="GPU staging queue capacity (chunks) for GPU direct streaming.",
    )
    parser.add_argument(
        "--streaming-consumer-prefill-chunks",
        type=int,
        default=0,
        help=(
            "GPU direct consumer prefill target in chunks. Set 0 to disable; "
            "values >0 wait for post-mixing serving tokens before training starts."
        ),
    )
    parser.add_argument(
        "--control-state-path",
        type=str,
        default=None,
        help=(
            "Path to control_state.json written by topology_supervisor.py. "
            "When provided, topology (vllm_tp, vllm_dp, sae_tp), buffer_name, "
            "and checkpoint_path are read from this file and override CLI args."
        ),
    )
    parser.add_argument(
        "--use-cached-activations",
        action="store_true",
        default=False,
        help=(
            "Train SAE(s) from a pre-computed activations cache produced by "
            "CacheActivationsRunner. vLLM is not loaded. Mutually exclusive with "
            "--streaming-mode and --control-state-path."
        ),
    )
    parser.add_argument(
        "--cached-activations-path",
        type=str,
        default=None,
        help=(
            "Path to the cached activations directory (split-by-hook or monolithic "
            "HuggingFace Dataset). Required when --use-cached-activations is set."
        ),
    )
    # argparse does not split custom compact options such as ``-sddp2`` into
    # ``-sddp 2``. Normalize those two convenience spellings before parsing.
    argv = []
    for token in sys.argv[1:]:
        if token.startswith("-sddp") and token != "-sddp":
            suffix = token[len("-sddp"):]
            if suffix.isdigit():
                argv.extend(["-sddp", suffix])
                continue
        if token.startswith("-sfsdp") and token != "-sfsdp":
            suffix = token[len("-sfsdp"):]
            if suffix.isdigit():
                argv.extend(["-sfsdp", suffix])
                continue
        argv.append(token)

    args = parser.parse_args(argv)

    shortcut_modes = []
    if args.ddp:
        shortcut_modes.append("ddp")
    if args.fsdp:
        shortcut_modes.append("fsdp")
    if args.sae_ddp_size is not None:
        shortcut_modes.append("ddp")
    if args.sae_fsdp_size is not None:
        shortcut_modes.append("fsdp")

    distinct_shortcut_modes = set(shortcut_modes)
    if len(distinct_shortcut_modes) > 1:
        parser.error("DDP and FSDP shortcut options cannot be used together")
    shortcut_mode = next(iter(distinct_shortcut_modes), None)

    if args.sae_ddp_size is not None and args.sae_fsdp_size is not None:
        parser.error("-sddp and -sfsdp cannot be used together")

    shortcut_size = (
        args.sae_ddp_size
        if args.sae_ddp_size is not None
        else args.sae_fsdp_size
    )
    if shortcut_size is not None:
        if shortcut_size < 1:
            parser.error("-sddp/-sfsdp size must be >= 1")
        # Reject an explicitly different -sdp value rather than silently overriding it.
        explicit_sdp = any(
            token in ("-sdp", "--sae-dp-size")
            or token.startswith("--sae-dp-size=")
            for token in argv
        )
        if explicit_sdp and args.sae_dp_size != shortcut_size:
            parser.error(
                f"conflicting SAE DP sizes: -sdp/--sae-dp-size={args.sae_dp_size} "
                f"but shortcut requests {shortcut_size}"
            )
        args.sae_dp_size = shortcut_size

    if args.sae_dp_mode is not None and shortcut_mode is not None:
        if args.sae_dp_mode != shortcut_mode:
            parser.error(
                f"conflicting SAE DP modes: --sae-dp-mode={args.sae_dp_mode} "
                f"but shortcut requests {shortcut_mode}"
            )
    elif args.sae_dp_mode is None:
        args.sae_dp_mode = shortcut_mode or "ddp"

    if args.no_save_final:
        args.save_final_checkpoint = False
        args.no_save_final_sae = True

    return args


def _resolve_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is GPU-only.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _cleanup_topology_runner_shm() -> int:
    """Remove shared-memory files left by a previous topology-switch run."""
    removed = 0
    for path in Path("/dev/shm").glob("sae_buf_*"):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            # Another rank or a concurrently exiting process removed it first.
            pass
        except OSError as exc:
            print(f"[WARNING] Could not remove streaming shared-memory file {path}: {exc}")
    return removed


def _cleanup_topology_runner_runtime() -> int:
    """Stop a stale topology runner and remove its shared-memory artifacts."""
    try:
        try:
            from scripts.run_topology_switch_runner_gpu import (
                _resolve_run_dir,
                _terminate_existing_run_processes,
            )
        except ImportError:
            launcher_path = Path(__file__).resolve().parent / "scripts" / "run_topology_switch_runner_gpu.py"
            spec = importlib.util.spec_from_file_location(
                "_saelens_topology_switch_runner", launcher_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load {launcher_path}")
            launcher = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = launcher
            spec.loader.exec_module(launcher)
            _resolve_run_dir = launcher._resolve_run_dir
            _terminate_existing_run_processes = launcher._terminate_existing_run_processes

        _terminate_existing_run_processes(run_dir=_resolve_run_dir())
    except Exception as exc:
        # Cleanup should not prevent a standalone streaming run from starting
        # when the optional topology-runner launcher is unavailable.
        print(f"[WARNING] Could not clean stale topology-runner processes: {exc}")
    return _cleanup_topology_runner_shm()


def _prepare_streaming_startup(*, cleanup: bool, world_size: int) -> None:
    """Run topology-runner cleanup once before streaming buffers are created.

    ``run_sae_runner_gpu.py`` is commonly launched under ``torchrun``.  The
    process group is initialized here so all ranks can wait until rank 0 has
    finished cleaning stale topology-runner processes and ``sae_buf_*`` files
    before constructing the streaming runner.  When cleanup is disabled,
    initialization is left to the normal runner path.
    """
    if not cleanup:
        return

    rank = int(os.environ.get("RANK", "0"))
    if dist.is_initialized():
        rank = dist.get_rank()
    if rank == 0:
        removed = _cleanup_topology_runner_runtime()
        print(
            f"[INFO] streaming startup cleanup removed {removed} "
            "shared-memory file(s)."
        )

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if dist.is_initialized() and world_size > 1:
        dist.barrier()


def _resolve_store_batch_size_prompts(
    baseline_prompts: int,
    *,
    vllm_dp_size: int,
    sae_dp_size: int,
    auto_scale: bool,
) -> int:
    """Resolve the per-vLLM-producer prompt batch for non-streaming DP.

    ``baseline_prompts`` is defined for the 1-vLLM-DP -> 1-SAE-DP topology.
    A producer batch is partitioned/fanned in across SAE DP replicas, so the
    equivalent per-producer batch is proportional to ``sae_dp / vllm_dp``.
    ``ceil`` avoids underfeeding an SAE replica when that ratio is fractional.
    Streaming has a separate chunk allocator and intentionally does not use
    this scaling helper.
    """
    if baseline_prompts < 1:
        raise ValueError("--store-batch-size-prompts must be >= 1")
    if not auto_scale or vllm_dp_size <= 0 or sae_dp_size <= 0:
        return baseline_prompts
    return max(
        1,
        (baseline_prompts * sae_dp_size + vllm_dp_size - 1) // vllm_dp_size,
    )


def _resolve_hidden_size(model_name: str) -> int:
    local_files_only = Path(model_name).exists()
    hf_cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    if not hasattr(hf_cfg, "hidden_size"):
        raise ValueError(f"Could not infer hidden_size from model config: {model_name}")
    return int(hf_cfg.hidden_size)


def _resolve_d_in_for_cached(args: argparse.Namespace) -> int:
    """Resolve d_in for cached mode.

    The cached activations are the ground truth for d_in (the hook type may be
    attn_v/mlp/etc. whose width differs from the model hidden_size), so read the
    cache's dataset_info first and only fall back to the model config.
    """
    cache_dir = Path(args.cached_activations_path)
    manifest_path = cache_dir / "cache_activations_manifest.json"
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            hook_to_dir = manifest.get("hook_to_dir", {})
            hook_dirs = list(hook_to_dir.values())
            if not hook_dirs:
                raise ValueError(f"manifest at {manifest_path} has empty hook_to_dir")
            first_hook_dir = cache_dir / hook_dirs[0]
            first_hook_name = manifest["hook_names"][0]
        else:
            first_hook_dir = cache_dir
            first_hook_name = args.hook_name
        info_path = first_hook_dir / "dataset_info.json"
        if not info_path.exists():
            raise ValueError(f"{info_path} missing")
        info = json.loads(info_path.read_text())
        feat = info["features"][first_hook_name]
        return int(feat["shape"][-1])
    except Exception as cache_err:
        try:
            return _resolve_hidden_size(args.model_name)
        except Exception as model_err:
            raise ValueError(
                f"Could not infer d_in from cache ({cache_err}) or model "
                f"({model_err})"
            ) from cache_err


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
    is_multi_sae = (resume_path / MULTI_SAE_MANIFEST_FILENAME).exists()
    required_files = (
        [TRAINER_STATE_FILENAME, MULTI_SAE_MANIFEST_FILENAME]
        if is_multi_sae
        else [TRAINER_STATE_FILENAME, SAE_WEIGHTS_FILENAME]
    )
    missing = [
        filename
        for filename in required_files
        if not (resume_path / filename).exists()
    ]
    if missing:
        raise ValueError(
            "--resume-from-checkpoint is missing required file(s): "
            + ", ".join(missing)
        )


def _normalize_checkpoint_storage_args(args: argparse.Namespace) -> None:
    """Apply topology-supervisor memory checkpoint semantics to parsed args.

    ``checkpoint_storage=memory`` is only actionable when the supervisor provides
    a quiesce checkpoint path.  Plain standalone runs should keep the normal
    checkpoint path and final checkpoint behavior.
    """
    if args.checkpoint_storage != "memory":
        return
    if args.quiesce_checkpoint_path is None:
        return
    args.checkpoint_path = args.quiesce_checkpoint_path
    args.save_final_checkpoint = False


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
    vllm_dp_size: int,
    sae_tp_size: int,
    sae_dp_size: int,
    sae_pp_size: int,
    sae_dp_mode: str,
    hooks: list[str],
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
            "vllm_dp_size": vllm_dp_size,
            "sae_tp_size": sae_tp_size,
            "sae_dp_size": sae_dp_size,
            "sae_pp_size": sae_pp_size,
            "sae_dp_mode": sae_dp_mode,
            "hooks": hooks,
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
    os.environ.setdefault("SAE_ADAM_IMPL", "fused")

    # Cached-mode validation must run before control-state processing so the
    # mutual-exclusion errors fire even if the user passes both.
    if args.use_cached_activations:
        if args.streaming_mode:
            raise ValueError(
                "--use-cached-activations is incompatible with --streaming-mode"
            )
        if args.control_state_path is not None:
            raise ValueError(
                "--use-cached-activations is incompatible with --control-state-path "
                "(topology supervisor mode)"
            )
        if not args.cached_activations_path:
            raise ValueError(
                "--use-cached-activations requires --cached-activations-path"
            )
        cache_dir = Path(args.cached_activations_path)
        if not cache_dir.exists():
            raise ValueError(
                f"--cached-activations-path does not exist: {cache_dir}"
            )
        if args.sae_dp_size < 1:
            raise ValueError(
                "--use-cached-activations requires --sae-dp-size >= 1"
            )
        if args.sae_pp_size > 1:
            if args.hook_names is None:
                raise ValueError(
                    "--sae-pp-size > 1 with cached activations requires "
                    "--hook-names (multi-hook)"
                )
            n_hooks = len(
                [h.strip() for h in args.hook_names.split(",") if h.strip()]
            )
            if n_hooks < args.sae_pp_size:
                raise ValueError(
                    f"--sae-pp-size={args.sae_pp_size} > number of hooks ({n_hooks})"
                )
        # Force topology shape: no producers, all ranks are SAE.
        if args.vllm_dp_size != 0:
            print(
                f"[INFO] cached mode: forcing --vllm-dp-size 0 (was {args.vllm_dp_size})"
            )
            args.vllm_dp_size = 0
        if args.vllm_tp_size not in (None, 1):
            print(
                f"[INFO] cached mode: forcing --vllm-tp-size 1 (was {args.vllm_tp_size})"
            )
        args.vllm_tp_size = 1
        args.use_shard_routing = True

    # If a control state file is provided, override topology and checkpoint args.
    if args.control_state_path is not None:
        ctrl = read_control_state(args.control_state_path)
        args.vllm_tp_size = ctrl.topology.vllm_tp
        args.vllm_dp_size = ctrl.topology.vllm_dp
        args.sae_tp_size = ctrl.topology.sae_tp
        args.sae_dp_size = ctrl.topology.sae_dp
        args.sae_pp_size = ctrl.topology.sae_pp_size
        args.streaming_buffer_name = ctrl.buffer_name
        if args.resume_from_checkpoint is None and ctrl.checkpoint_path is not None:
            args.resume_from_checkpoint = ctrl.checkpoint_path
        print(
            f"[INFO] control_state_path={args.control_state_path}: "
            f"topology=vllm_tp={ctrl.topology.vllm_tp} vllm_dp={ctrl.topology.vllm_dp} "
            f"sae_tp={ctrl.topology.sae_tp} sae_pp={ctrl.topology.sae_pp_size} "
            f"buffer={ctrl.buffer_name} checkpoint={ctrl.checkpoint_path}"
        )

    # Derive quiesce_dir from control_state_path so workers look for quiesce
    # signals in run_dir (where the supervisor writes them), not in checkpoint_path.
    quiesce_dir = (
        Path(args.control_state_path).parent
        if args.control_state_path is not None
        else None
    )
    _normalize_checkpoint_storage_args(args)

    _validate_checkpoint_args(args)
    vllm_tp_size = (
        args.vllm_tp_size if args.vllm_tp_size is not None else args.tp_size
    )
    sae_tp_size = args.sae_tp_size if args.sae_tp_size is not None else args.tp_size
    if vllm_tp_size < 1:
        raise ValueError("--vllm-tp-size must be >= 1")
    if sae_tp_size < 1:
        raise ValueError("--sae-tp-size must be >= 1")
    if args.vllm_dp_size < 0:
        raise ValueError("--vllm-dp-size must be >= 0")
    if args.sae_dp_size < 0:
        raise ValueError("--sae-dp-size must be >= 0")
    if args.context_size < 1:
        raise ValueError("--context-size must be >= 1")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        raise ValueError("--max-num-batched-tokens must be >= 1")
    if args.train_batch_size_tokens < 1:
        raise ValueError("--train-batch-size-tokens must be >= 1")
    if args.store_batch_size_prompts < 1:
        raise ValueError("--store-batch-size-prompts must be >= 1")
    if args.save_vllm_memory_every_n_steps < 0:
        raise ValueError("--save-vllm-memory-every-n-steps must be >= 0")
    if not args.streaming_mode and args.sae_dp_size > 1 and args.vllm_dp_size == 1:
        print(
            f"[INFO] vllm_dp_size=1, sae_dp_size={args.sae_dp_size} (1:m topology) — "
            "automatically enabling --use-shard-routing."
        )
        args.use_shard_routing = True
    if (
        not args.streaming_mode
        and not args.use_shard_routing
        and args.vllm_dp_size > 1
        and args.sae_dp_size > 1
    ):
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
    # Hook names are needed early to validate sae_pp_size in streaming_mode.
    hook_names = (
        [hook.strip() for hook in args.hook_names.split(",") if hook.strip()]
        if args.hook_names is not None
        else None
    )
    if hook_names is not None and len(hook_names) == 0:
        hook_names = None
    if args.streaming_mode:
        args.use_shard_routing = False
        expected_world_size = (
            vllm_tp_size * args.vllm_dp_size
            + sae_tp_size * args.sae_dp_size * args.sae_pp_size
        )
        if world_size not in (1, expected_world_size):
            raise ValueError(
                f"streaming_mode: WORLD_SIZE={world_size} does not match "
                f"vllm_tp*vllm_dp + sae_tp*sae_dp*sae_pp_size = {expected_world_size}."
            )
        if args.sae_pp_size > 1:
            if hook_names is None or len(hook_names) < args.sae_pp_size:
                raise ValueError(
                    f"streaming_mode with --sae-pp-size={args.sae_pp_size} requires "
                    "--hook-names with at least sae_pp_size hooks."
                )
    else:
        expected_world_size = max(
            vllm_tp_size * args.vllm_dp_size, sae_tp_size * args.sae_dp_size * args.sae_pp_size
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

    # Treat the CLI value as the DP=1 -> DP=1 baseline. In non-streaming mode
    # each vLLM producer batch is partitioned/fanned in across SAE DP replicas,
    # so scale the producer batch to keep the per-SAE input volume equivalent.
    # Streaming uses a shared chunk allocator and deliberately keeps the value
    # per producer unchanged.
    store_batch_size_prompts = _resolve_store_batch_size_prompts(
        args.store_batch_size_prompts,
        vllm_dp_size=args.vllm_dp_size,
        sae_dp_size=args.sae_dp_size,
        auto_scale=(
            args.auto_scale_store_batch_size_prompts and not args.streaming_mode
        ),
    )
    if (
        not args.streaming_mode
        and args.auto_scale_store_batch_size_prompts
        and args.vllm_dp_size > 0
        and args.sae_dp_size > 0
        and store_batch_size_prompts != args.store_batch_size_prompts
    ):
        print(
            f"[INFO] auto-scaling store_batch_size_prompts "
            f"{args.store_batch_size_prompts} -> {store_batch_size_prompts} "
            f"for vllm_dp_size={args.vllm_dp_size}, "
            f"sae_dp_size={args.sae_dp_size}."
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
    vllm_memory_probe_layer = args.vllm_memory_probe_layer
    model_kwargs: dict[str, object] = {}
    if args.save_vllm_memory_every_n_steps > 0:
        if args.use_cached_activations:
            raise ValueError(
                "--save-vllm-memory-every-n-steps requires live vLLM activation generation"
            )
        if output_path is None:
            raise ValueError(
                "--save-vllm-memory-every-n-steps requires output_path to be set"
            )
        if vllm_memory_probe_layer is None:
            default_probe_hook = hook_names[0] if hook_names is not None else args.hook_name
            vllm_memory_probe_layer = extract_layer_from_tlens_hook_name(default_probe_hook)
        if vllm_memory_probe_layer is None:
            raise ValueError(
                "--vllm-memory-probe-layer is required when --hook-name has no layer"
            )
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        model_kwargs.update(
            {
                "vllm_memory_every_n_steps": args.save_vllm_memory_every_n_steps,
                "vllm_memory_probe_layer": vllm_memory_probe_layer,
                "vllm_memory_history_path": str(
                    Path(output_path) / f"vllm_memory_history_rank{rank}.jsonl"
                ),
            }
        )
        if args.record_vllm_memory_timeline_step >= 0:
            # Pickle allocator timeline for one vLLM forward. activations_store
            # pops these kwargs. Under external_launcher TP>1 every torchrun rank
            # runs generate() inline and would write the SAME file, so suffix the
            # path by rank here (mirrors vllm_memory_history_path above).
            model_kwargs.update(
                {
                    "vllm_memory_timeline_step": args.record_vllm_memory_timeline_step,
                    "vllm_memory_timeline_path": str(
                        Path(output_path) / f"memory_timeline_vllm_rank{rank}.pickle"
                    ),
                }
            )

    device = _resolve_device()
    if args.streaming_mode and args.control_state_path is None:
        _prepare_streaming_startup(
            cleanup=args.streaming_cleanup,
            world_size=world_size,
        )
    d_in = (
        _resolve_d_in_for_cached(args)
        if args.use_cached_activations
        else _resolve_hidden_size(args.model_name)
    )
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=args.d_sae,
            k=args.k,
            device=device,
            dtype=args.dtype,
            use_sparse_activations=args.use_sparse_activations,
            rescale_acts_by_decoder_norm=args.rescale_acts_by_decoder_norm,
        ),
        model_name=args.model_name,
        model_class_name="VLLMModel",
        model_from_pretrained_kwargs={
            "tensor_parallel_size": vllm_tp_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        vllm_max_num_batched_tokens=args.max_num_batched_tokens,
        hook_name=args.hook_name,
        hook_names=hook_names,
        dataset_path=args.dataset_path,
        dataset_trust_remote_code=False,
        streaming=False,
        is_dataset_tokenized=args.is_dataset_tokenized,
        context_size=args.context_size,
        training_tokens=training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        store_batch_size_prompts=store_batch_size_prompts,
        n_batches_in_buffer=n_batches_in_buffer,
        activations_mixing_fraction=args.activations_mixing_fraction,
        device=device,
        act_store_device=args.act_store_device,
        dtype=args.dtype,
        autocast=args.autocast,
        autocast_lm=args.autocast_lm,
        compile_llm=False,
        compile_sae=False,
        model_kwargs=model_kwargs,
        dead_feature_window=args.dead_feature_window,
        n_eval_batches=0,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        quiesce_checkpoint_path=args.quiesce_checkpoint_path,
        save_final_checkpoint=args.save_final_checkpoint,
        output_path=output_path,
        save_mse_every_n_steps=args.save_mse_every_n_steps,
        save_timing_every_n_steps=args.save_timing_every_n_steps,
        save_memory_every_n_steps=args.save_memory_every_n_steps,
        record_memory_empty_cache=args.record_memory_empty_cache,
        record_memory_timeline_step=args.record_memory_timeline_step,
        append_history_logs=args.append_history_logs,
        synchronize_timing=args.synchronize_timing,
        step_window_profile_start_step=args.step_window_profile_start_step,
        step_window_profile_window_steps=args.step_window_profile_window_steps,
        step_window_profile_window_count=args.step_window_profile_window_count,
        step_window_profile_vllm_start_step=args.step_window_profile_vllm_start_step,
        step_window_profile_vllm_window_steps=args.step_window_profile_vllm_window_steps,
        step_window_profile_vllm_window_count=args.step_window_profile_vllm_window_count,
        seed=args.seed,
        verbose=True,
        sae_dp_mode=args.sae_dp_mode,
        sae_pp_size=args.sae_pp_size,
        multi_sae_backward_mode=args.multi_sae_backward_mode,
        multi_sae_backward_order=args.multi_sae_backward_order,
        multi_sae_stats_sync_mode=args.multi_sae_stats_sync_mode,
        multi_sae_stats_sync_interval=args.multi_sae_stats_sync_interval,
        multi_sae_seed_mode=args.multi_sae_seed_mode,
        multi_sae_distributed_architecture=args.multi_sae_distributed_architecture,
        multi_sae_tp_phase_fence=args.multi_sae_tp_phase_fence,
        multi_sae_optimizer_overlap=args.multi_sae_optimizer_overlap,
        ddp_broadcast_buffers=args.ddp_broadcast_buffers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        ddp_static_graph=args.ddp_static_graph,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        ddp_config_strict=args.ddp_config_strict,
        fsdp_backward_prefetch=args.fsdp_backward_prefetch,
        fsdp_forward_prefetch=args.fsdp_forward_prefetch,
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
        streaming_mode=args.streaming_mode,
        streaming_chunk_size_tokens=args.streaming_chunk_size_tokens,
        streaming_num_chunks=args.streaming_num_chunks,
        streaming_prefetch_chunks=args.streaming_prefetch_chunks,
        streaming_mix_chunks=args.streaming_mix_chunks,
        streaming_mix_fraction=args.streaming_mix_fraction,
        streaming_buffer_name=args.streaming_buffer_name,
        streaming_shuffle=args.streaming_shuffle,
        streaming_random_chunks=args.streaming_random_chunks,
        streaming_use_gpu_direct=args.streaming_use_gpu_direct,
        streaming_staging_queue_capacity=args.streaming_staging_queue_capacity,
        streaming_consumer_prefill_chunks=args.streaming_consumer_prefill_chunks,
        use_cached_activations=args.use_cached_activations,
        cached_activations_path=args.cached_activations_path,
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
    if args.streaming_mode:
        print(
            f"  store_batch_size_prompts={store_batch_size_prompts} "
            "(streaming; DP auto-scaling not applied)"
        )
    elif args.auto_scale_store_batch_size_prompts:
        print(
            f"  store_batch_size_prompts={store_batch_size_prompts} "
            f"(baseline={args.store_batch_size_prompts}, auto_scaled=True)"
        )
    else:
        print(f"  store_batch_size_prompts={store_batch_size_prompts}")
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
        print(
            "  multi_sae_distributed_architecture="
            f"{cfg.multi_sae_distributed_architecture}"
        )
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
    if args.sae_dp_mode == "fsdp":
        print(f"  fsdp_backward_prefetch={args.fsdp_backward_prefetch}")
        print(f"  fsdp_forward_prefetch={args.fsdp_forward_prefetch}")
        print(f"  fsdp_sharding_strategy={args.fsdp_sharding_strategy}")
    if args.save_mse_every_n_steps > 0:
        print(f"  save_mse_every_n_steps={args.save_mse_every_n_steps}")
    if args.save_timing_every_n_steps > 0:
        print(f"  save_timing_every_n_steps={args.save_timing_every_n_steps}")
    if args.save_vllm_memory_every_n_steps > 0:
        print(f"  save_vllm_memory_every_n_steps={args.save_vllm_memory_every_n_steps}")
        print(f"  vllm_memory_probe_layer={vllm_memory_probe_layer}")
    if args.synchronize_timing:
        print("  synchronize_timing=True")
    if args.step_window_profile_start_step > 0:
        sae_start, sae_steps, sae_count = cfg.resolved_step_window_profile(role="sae")
        vllm_start, vllm_steps, vllm_count = cfg.resolved_step_window_profile(
            role="vllm"
        )
        sae_last = sae_start + sae_steps * sae_count - 1
        vllm_last = vllm_start + vllm_steps * vllm_count - 1
        print(
            f"  step_window_profile(sae)=start{sae_start} x{sae_steps}steps "
            f"x{sae_count}windows (through step {sae_last})"
        )
        print(
            f"  step_window_profile(vllm)=start{vllm_start} x{vllm_steps}steps "
            f"x{vllm_count}windows (through step {vllm_last}); "
            "only used by streaming/split-role vLLM ranks"
        )
        if args.save_memory_every_n_steps > 0:
            print(
                "  WARNING: save_memory_every_n_steps > 0 syncs the device at memory "
                "phases on sampled steps, which inflates those steps. Pass "
                "--save-memory-every-n-steps 0 for throughput measurements."
            )
        if args.synchronize_timing:
            print(
                "  WARNING: --synchronize-timing syncs inside each step, which "
                "inflates step-window times. Drop it for throughput measurements."
            )
    if args.max_num_batched_tokens is not None:
        print(f"  max_num_batched_tokens={args.max_num_batched_tokens}")
    if args.checkpoint_path is not None:
        print(f"  checkpoint_path={args.checkpoint_path}")
    print(f"  checkpoint_storage={args.checkpoint_storage}")
    if args.quiesce_checkpoint_path is not None:
        print(f"  quiesce_checkpoint_path={args.quiesce_checkpoint_path}")
    if args.streaming_mode:
        print(f"  streaming_mix_chunks={args.streaming_mix_chunks}")
        print(f"  streaming_mix_fraction={args.streaming_mix_fraction}")
        if args.streaming_use_gpu_direct:
            print("  streaming_use_gpu_direct=True")
            print(f"  activations_mixing_fraction={args.activations_mixing_fraction}")
            print(f"  n_batches_in_buffer={n_batches_in_buffer}")
            print(f"  streaming_staging_queue_capacity={args.streaming_staging_queue_capacity}")
            print(
                "  streaming_consumer_prefill_chunks="
                f"{args.streaming_consumer_prefill_chunks}"
            )
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
        quiesce_dir=quiesce_dir,
        sae_pp_size=args.sae_pp_size,
    )

    # Write buffer name and params to control state on first run (rank 0 only).
    # The supervisor needs this to reset the buffer on topology switch.
    if (
        args.control_state_path is not None
        and args.streaming_mode
        and (not dist.is_initialized() or dist.get_rank() == 0)
        and hasattr(runner, "_streaming_buffer_name")
        and runner._streaming_buffer_name
    ):
        ctrl = read_control_state(args.control_state_path)
        if not ctrl.buffer_name:
            ctrl.buffer_name = runner._streaming_buffer_name
            num_hooks = len(hook_names) if hook_names is not None else 1
            ctrl.buffer_params = BufferParams(
                num_chunks=args.streaming_num_chunks,
                chunk_size_tokens=args.streaming_chunk_size_tokens * num_hooks,
                d_model=d_in,
                dtype=args.dtype,
                num_hooks=num_hooks,
            )
            write_control_state(args.control_state_path, ctrl)
            print(f"[INFO] Wrote buffer name to control state: {runner._streaming_buffer_name}")
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
            vllm_dp_size=args.vllm_dp_size,
            sae_tp_size=sae_tp_size,
            sae_dp_size=args.sae_dp_size,
            sae_pp_size=args.sae_pp_size,
            sae_dp_mode=args.sae_dp_mode,
            hooks=hook_names if hook_names is not None else [args.hook_name],
            status=run_status,
            error=run_error,
        )
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
