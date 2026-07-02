"""Synthetic SAE-only memory profiling harness for the v5 phase study.

Drives the *real* ``MultiSAETrainer.fit()`` loop on randomly generated
activations (no vLLM, no Llama weights — matching the SAE-only memory that the
sae_phase_v4 topology-supervised run measured). Because it goes through
``fit()``, the per-phase peak recorder and the ~1 Hz background device sampler
both run, producing:

    <output>/memory_phase_history_rank{N}.jsonl   (per-phase peak_allocated etc.)
    <output>/device_history_rank{N}.jsonl         (~1 Hz device-used timeline)

TP=1:
    python3 scripts/profile_sae_phase_v5.py --output-path results/memory_model/sae_phase_v5/tp1

TP=2 (sharded SAEs across 2 GPUs):
    torchrun --nproc_per_node=2 scripts/profile_sae_phase_v5.py \
        --output-path results/memory_model/sae_phase_v5/tp2
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator

import torch
import torch.distributed as dist

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.training.multi_sae_trainer import MultiSAETrainer

DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--d-sae", type=int, default=65536)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--dtype", default="float32", choices=list(DTYPE))
    parser.add_argument("--hooks", type=int, default=2)
    parser.add_argument("--train-batch-size-tokens", type=int, default=2048)
    parser.add_argument("--n-steps", type=int, default=50)
    return parser.parse_args()


def make_data_provider(
    hook_names: list[str],
    *,
    n_steps: int,
    batch: int,
    d_in: int,
    dtype: torch.dtype,
) -> Iterator[dict[str, torch.Tensor]]:
    # Fresh CPU tensors per step (the trainer moves them to device in
    # after_scale_to_device), mirroring how real activations stream in.
    for _ in range(n_steps):
        yield {
            hook: torch.randn(batch, d_in, dtype=dtype) for hook in hook_names
        }


def build_saes(
    hook_names: list[str],
    *,
    d_in: int,
    d_sae: int,
    k: int,
    dtype_name: str,
    device: torch.device,
    dtype: torch.dtype,
    tp_group: dist.ProcessGroup | None,
) -> dict[str, TopKTrainingSAE]:
    saes: dict[str, TopKTrainingSAE] = {}
    for hook in hook_names:
        cfg = TopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            dtype=dtype_name,
            device=str(device),
            normalize_activations="none",
            apply_b_dec_to_input=True,
            rescale_acts_by_decoder_norm=True,
        )
        if tp_group is not None:
            sae = TopKTrainingSAE.from_config_sharded(cfg, tp_group)
        else:
            sae = TopKTrainingSAE(cfg)
        sae.to(device=device, dtype=dtype)
        sae.train()
        saes[hook] = sae
    return saes


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    dtype = DTYPE[args.dtype]

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tp_group: dist.ProcessGroup | None = None
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        tp_group = dist.new_group(list(range(world_size)), backend="nccl")

    hook_names = [f"blocks.{21 + 10 * i}.hook_resid_post" for i in range(args.hooks)]

    saes = build_saes(
        hook_names,
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        dtype_name=args.dtype,
        device=device,
        dtype=dtype,
        tp_group=tp_group,
    )

    provider = make_data_provider(
        hook_names,
        n_steps=args.n_steps,
        batch=args.train_batch_size_tokens,
        d_in=args.d_in,
        dtype=dtype,
    )

    total_samples = args.n_steps * args.train_batch_size_tokens
    cfg = SAETrainerConfig(
        n_checkpoints=0,
        checkpoint_path=None,
        quiesce_checkpoint_path=None,
        save_final_checkpoint=False,
        output_path=args.output_path,
        save_mse_every_n_steps=1,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=1,
        synchronize_timing=False,
        multi_sae_backward_order="forward",
        multi_sae_stats_sync_mode="immediate",
        multi_sae_stats_sync_interval=1,
        total_training_samples=total_samples,
        device=str(device),
        autocast=False,
        lr=3e-4,
        lr_end=None,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        train_batch_size_samples=args.train_batch_size_tokens,
        dead_feature_window=1000,
        feature_sampling_window=1000,
        logger=LoggingConfig(log_to_wandb=False),
    )

    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=saes,
        base_sae_by_hook=saes,
        data_provider=provider,
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
        backward_mode="combined",
    )
    trainer.fit()

    if rank == 0:
        print(f"[rank{rank}] wrote phase + device history to {args.output_path}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
