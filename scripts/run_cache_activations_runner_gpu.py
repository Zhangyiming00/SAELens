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

For cache-time DP:
    torchrun --nproc_per_node=2 scripts/run_cache_activations_runner_gpu.py ... --dp-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.util import extract_layer_from_tlens_hook_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/data/models/Llama-3.1-8B")
    parser.add_argument("--dataset-path", default="../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048")
    parser.add_argument("--hook-name", default="blocks.21.hook_resid_post")
    parser.add_argument(
        "--hook-names",
        default="blocks.21.hook_resid_post,blocks.31.hook_resid_post",
        help="Comma-separated hook names for multi-hook caching.",
    )
    parser.add_argument(
        "--d-in",
        type=int,
        default=None,
        help="Activation feature dimension written to disk. Defaults to the "
        "model's hidden_size. Must be set explicitly when the hook captures a "
        "tensor whose last dim differs from hidden_size (e.g. mlp.hook_pre = "
        "intermediate_size, attn.hook_k/v = num_kv_heads*head_dim). All hooks "
        "in a single run must share this d_in.",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Cache-time data parallel size. Requires torchrun when > 1.",
    )
    parser.add_argument("--training-tokens", type=int, default=2048*4096)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--autocast-lm", action="store_true")
    parser.add_argument("--buffer-size-gb", type=float, default=4.0)
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
    parser.add_argument(
        "--save-vllm-memory-every-n-steps",
        type=int,
        default=0,
        help="When > 0 (VLLMModel only), probe per-substage GPU memory on one "
        "decoder layer every N buffer batches and write "
        "vllm_memory_history_rank{N}.jsonl into --output-path. 0 disables.",
    )
    parser.add_argument(
        "--vllm-memory-probe-layer",
        type=int,
        default=None,
        help="Decoder layer index to install the memory probe on. Defaults to "
        "the layer parsed from --hook-name (or the first of --hook-names).",
    )
    parser.add_argument(
        "--record-vllm-memory-timeline-step",
        type=int,
        default=-1,
        help="When >= 0 (VLLMModel only), record the full PyTorch CUDA "
        "allocator history for that 0-based cache-batch step and write "
        "vllm_cache_memory_timeline_rank{N}.pickle. -1 disables.",
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


def _distributed_rank(dp_size: int) -> tuple[int, int]:
    if dp_size < 1:
        raise ValueError("--dp-size must be >= 1")
    if dp_size == 1:
        return 0, 1

    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            "--dp-size > 1 requires torchrun. Missing environment variables: "
            + ", ".join(missing)
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != dp_size:
        raise RuntimeError(
            f"--dp-size={dp_size} requires WORLD_SIZE={dp_size}, got {world_size}. "
            f"Launch with torchrun --nproc_per_node={dp_size}."
        )
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return rank, world_size


def _format_total_completion_time(elapsed_s: float) -> str:
    return f"Total completion time: {elapsed_s:.1f}s"


def _write_cache_activation_time(
    *,
    output_path: Path,
    total_elapsed_s: float,
    cache_elapsed_s: float,
    rank: int,
    dp_size: int,
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    timing_path = output_path / "cache_activation_time.json"
    timing = {
        "total_completion_time_s": round(total_elapsed_s, 3),
        "cache_runner_time_s": round(cache_elapsed_s, 3),
        "rank": rank,
        "dp_size": dp_size,
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    return timing_path


def _cleanup_dp_tmp_cache(dp_tmp_root: Path) -> None:
    if dp_tmp_root.exists():
        shutil.rmtree(dp_tmp_root)


def _vllm_memory_timeline_path(
    *,
    output_path: Path,
    dp_tmp_root: Path,
    rank: int,
    dp_size: int,
) -> Path:
    parent = dp_tmp_root if dp_size > 1 else output_path
    return parent / f"vllm_cache_memory_timeline_rank{rank}.pickle"


def _vllm_model_kwargs(
    *,
    save_vllm_memory_every_n_steps: int,
    vllm_memory_probe_layer: int | None,
    hook_names: list[str] | None,
    hook_name: str,
    model_class_name: str,
    output_path: Path,
    rank: int,
    record_vllm_memory_timeline_step: int,
    vllm_memory_timeline_path: Path,
) -> dict:
    model_kwargs: dict = {}
    if save_vllm_memory_every_n_steps > 0:
        if model_class_name != "VLLMModel":
            raise RuntimeError(
                "--save-vllm-memory-every-n-steps requires --model-class-name VLLMModel"
            )
        probe_layer = vllm_memory_probe_layer
        if probe_layer is None:
            probe_hook = hook_names[0] if hook_names is not None else hook_name
            probe_layer = extract_layer_from_tlens_hook_name(probe_hook)
        if probe_layer is None:
            raise RuntimeError(
                "--vllm-memory-probe-layer is required when --hook-name has no layer"
            )
        model_kwargs.update(
            {
                "vllm_memory_every_n_steps": save_vllm_memory_every_n_steps,
                "vllm_memory_probe_layer": int(probe_layer),
                "vllm_memory_history_path": str(
                    output_path / f"vllm_memory_history_rank{rank}.jsonl"
                ),
            }
        )

    if record_vllm_memory_timeline_step >= 0:
        if model_class_name != "VLLMModel":
            raise RuntimeError(
                "--record-vllm-memory-timeline-step requires --model-class-name VLLMModel"
            )
        model_kwargs.update(
            {
                "vllm_memory_timeline_step": record_vllm_memory_timeline_step,
                "vllm_memory_timeline_path": str(vllm_memory_timeline_path),
            }
        )
    return model_kwargs


def _resolve_output_path_for_new_run(
    output_path: Path,
    *,
    timestamp: str | None = None,
) -> Path:
    dp_tmp_root = output_path.with_name(output_path.name + ".tmp_dp_cache")
    output_conflicts = output_path.exists() and any(output_path.iterdir())
    tmp_conflicts = dp_tmp_root.exists() and any(dp_tmp_root.iterdir())
    if not output_conflicts and not tmp_conflicts:
        return output_path

    suffix = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_new_output_path = output_path.with_name(f"{output_path.name}.new_{suffix}")
    new_output_path = base_new_output_path
    counter = 1
    while new_output_path.exists():
        new_output_path = output_path.with_name(f"{base_new_output_path.name}_{counter}")
        counter += 1

    return new_output_path


def main() -> None:
    total_t0 = time.perf_counter()
    args = parse_args()
    rank, world_size = _distributed_rank(args.dp_size)
    if args.dp_size > 1 and args.tp_size != 1:
        raise RuntimeError("--dp-size > 1 currently requires --tp-size 1.")

    device = _resolve_device()
    d_in = args.d_in if args.d_in is not None else _resolve_hidden_size(args.model_name)

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

    requested_output_path = Path(args.output_path)
    output_path = requested_output_path
    if rank == 0:
        output_path = _resolve_output_path_for_new_run(requested_output_path)
        if output_path != requested_output_path:
            print(
                f"Requested output_path is non-empty; writing this run to {output_path}"
            )
    if args.dp_size > 1:
        resolved_output_paths = [str(output_path)]
        dist.broadcast_object_list(resolved_output_paths, src=0)
        output_path = Path(resolved_output_paths[0])
    dp_tmp_root = output_path.with_name(output_path.name + ".tmp_dp_cache")
    if args.dp_size > 1:
        dist.barrier()
    cfg_output_path = str(output_path)
    if args.dp_size > 1:
        cfg_output_path = str(dp_tmp_root / f"dp_{rank:05d}")

    vllm_memory_timeline_path = _vllm_memory_timeline_path(
        output_path=output_path,
        dp_tmp_root=dp_tmp_root,
        rank=rank,
        dp_size=args.dp_size,
    )
    model_kwargs = _vllm_model_kwargs(
        save_vllm_memory_every_n_steps=args.save_vllm_memory_every_n_steps,
        vllm_memory_probe_layer=args.vllm_memory_probe_layer,
        hook_names=hook_names,
        hook_name=args.hook_name,
        model_class_name=args.model_class_name,
        output_path=output_path,
        rank=rank,
        record_vllm_memory_timeline_step=args.record_vllm_memory_timeline_step,
        vllm_memory_timeline_path=vllm_memory_timeline_path,
    )

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
        new_cached_activations_path=cfg_output_path,
        shuffle=args.shuffle,
        seed=args.seed,
        dtype=args.dtype,
        device=model_device,
        buffer_size_gb=args.buffer_size_gb,
        autocast_lm=args.autocast_lm,
        streaming=not args.is_dataset_tokenized,
        dataset_shard_index=rank,
        dataset_shard_count=world_size,
        model_from_pretrained_kwargs=model_from_pretrained_kwargs,
        model_kwargs=model_kwargs,
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
    print(f"  dp_size={args.dp_size}")
    if args.dp_size > 1:
        print(f"  dp_rank={rank}")
    print(f"  output_path={args.output_path}")
    if args.dp_size > 1:
        print(f"  rank_output_path={cfg_output_path}")
    print(f"  n_buffers={cfg.n_buffers}")
    print(f"  n_batches_in_buffer={cfg.n_batches_in_buffer}")

    t0 = time.perf_counter()
    runner = CacheActivationsRunner(cfg)
    cached = runner.run()
    elapsed = time.perf_counter() - t0

    if args.dp_size > 1:
        dist.barrier()
        if rank == 0:
            shard_dirs = [
                dp_tmp_root / f"dp_{i:05d}"
                for i in range(world_size)
            ]
            cached = CacheActivationsRunner.consolidate_dp_shards(
                shard_dirs,
                output_path,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            for dp_rank in range(world_size):
                src = _vllm_memory_timeline_path(
                    output_path=output_path,
                    dp_tmp_root=dp_tmp_root,
                    rank=dp_rank,
                    dp_size=args.dp_size,
                )
                if src.exists():
                    shutil.move(
                        str(src),
                        str(
                            output_path
                            / f"vllm_cache_memory_timeline_rank{dp_rank}.pickle"
                        ),
                    )
        dist.barrier()
        if rank != 0:
            total_elapsed = time.perf_counter() - total_t0
            print(
                f"\nRank {rank} done in {elapsed:.1f}s. "
                f"Cached local shard to {cfg_output_path}"
            )
            print(_format_total_completion_time(total_elapsed))
            return

    total_elapsed = time.perf_counter() - total_t0
    timing_path = _write_cache_activation_time(
        output_path=output_path,
        total_elapsed_s=total_elapsed,
        cache_elapsed_s=elapsed,
        rank=rank,
        dp_size=args.dp_size,
    )
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
    print(_format_total_completion_time(total_elapsed))
    print(f"  timing_file={timing_path}")
    if args.dp_size > 1 and rank == 0:
        _cleanup_dp_tmp_cache(dp_tmp_root)
        print(f"  cleaned_tmp_cache={dp_tmp_root}")


if __name__ == "__main__":
    main()
