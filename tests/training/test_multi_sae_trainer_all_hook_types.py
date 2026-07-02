from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from tests.helpers import build_topk_sae_training_cfg

# Mirror the hook registry in sae_lens/vllm_model.py:_LLAMA_LIKE_HOOKS but
# parameterised with concrete d_in values so the trainer sees realistic per-hook
# activation widths. Sizes follow Llama-3.1-8B (D=4096, H=32, H_kv=8, d_h=128,
# I=14336) but shrunk to keep the test cheap.
D = 64  # stand-in for d_model
H = 8  # n_heads
H_KV = 2  # n_kv_heads (GQA)
D_H = 8  # head_dim  (so H * D_H == D)
I = 96  # intermediate size (multiple of 32)

# (hook_name, d_in_for_that_hook)
HOOK_TYPES: list[tuple[str, int]] = [
    ("hook_embed", D),
    ("blocks.0.hook_resid_pre", D),
    ("blocks.0.hook_resid_mid", D),
    ("blocks.0.hook_resid_post", D),
    ("blocks.0.hook_attn_out", D),
    ("blocks.0.hook_mlp_out", D),
    ("blocks.0.attn.hook_q", H * D_H),
    ("blocks.0.attn.hook_k", H_KV * D_H),
    ("blocks.0.attn.hook_v", H_KV * D_H),
    ("blocks.0.attn.hook_z", H * D_H),
    ("blocks.0.mlp.hook_pre", I),
    ("blocks.0.mlp.hook_post", I),
]

D_SAE_MULTIPLIER = 4
K = 8
BATCH = 32
N_STEPS = 60


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_sae(d_in: int, device: str) -> TopKTrainingSAE:
    # Use the SAE's default init (Kaiming-like) — random_params would give
    # uniform [0,1) weights, which is a pathological starting point and won't
    # converge in the short training budget we use here.
    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_in * D_SAE_MULTIPLIER,
        k=K,
        device=device,
    )
    return TopKTrainingSAE(cfg)


def _make_trainer_cfg(
    tmp_path: Path, total_samples: int, device: str
) -> SAETrainerConfig:
    return SAETrainerConfig(
        device=device,
        n_checkpoints=0,
        total_training_samples=total_samples,
        train_batch_size_samples=BATCH,
        output_path=None,
        save_mse_every_n_steps=0,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=0,
        synchronize_timing=False,
        multi_sae_backward_order="forward",
        multi_sae_stats_sync_mode="immediate",
        multi_sae_stats_sync_interval=1,
        lr=3e-3,
        lr_end=None,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        dead_feature_window=1000,
        feature_sampling_window=1000,
        autocast=False,
        checkpoint_path=str(tmp_path / "checkpoints"),
        quiesce_checkpoint_path=None,
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
    )


def _structured_batches(
    hook_to_din: dict[str, int],
    n_batches: int,
    device: str,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches drawn from a fixed sparse-code generative model per hook.

    Each hook's data lives on a low-rank K-sparse manifold, so a TopK SAE has
    a real target to converge towards (not pure i.i.d. noise where loss only
    drifts).
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    # One fixed "true" decoder per hook, plus a noise scale.
    n_true_features_per_hook = {
        hook: max(d_in * 2, K * 4) for hook, d_in in hook_to_din.items()
    }
    true_decoders = {
        hook: torch.randn(n_true_features_per_hook[hook], d_in, generator=gen)
        for hook, d_in in hook_to_din.items()
    }
    for _ in range(n_batches):
        batch: dict[str, torch.Tensor] = {}
        for hook, d_in in hook_to_din.items():
            n_feat = n_true_features_per_hook[hook]
            # Sample a K-sparse code per token.
            code = torch.zeros(BATCH, n_feat)
            idx = torch.stack(
                [torch.randperm(n_feat, generator=gen)[:K] for _ in range(BATCH)]
            )
            magnitudes = torch.randn(BATCH, K, generator=gen).abs() + 0.5
            code.scatter_(1, idx, magnitudes)
            acts = code @ true_decoders[hook]
            acts = acts + 0.05 * torch.randn(BATCH, d_in, generator=gen)
            batch[hook] = acts.to(device)
        yield batch


def _snapshot_params(
    sae_by_hook: dict[str, TopKTrainingSAE],
) -> dict[str, torch.Tensor]:
    return {hook: sae.W_enc.detach().clone() for hook, sae in sae_by_hook.items()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("hook_name,d_in", HOOK_TYPES, ids=[h for h, _ in HOOK_TYPES])
def test_each_hook_type_trains(tmp_path: Path, hook_name: str, d_in: int) -> None:
    """Mini training loop for a single hook of each registered type.

    Confirms MultiSAETrainer can drive a training step end-to-end for every
    activation shape produced by sae_lens/vllm_model.py:_LLAMA_LIKE_HOOKS.
    """
    device = _device()
    sae_by_hook = {hook_name: _make_sae(d_in, device)}
    initial = _snapshot_params(sae_by_hook)

    total_samples = BATCH * N_STEPS
    cfg = _make_trainer_cfg(tmp_path, total_samples, device)
    provider = _structured_batches({hook_name: d_in}, n_batches=N_STEPS, device=device)
    trainer = MultiSAETrainer(
        hook_names=[hook_name],
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
        data_provider=provider,
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    # Drive the real fit() loop, not just _train_step, so we exercise scaling,
    # stats sync, lr scheduler, and the memory bookkeeping path together.
    losses: list[float] = []
    orig_train_step = trainer._train_step

    def _capture(batch_by_hook, local_n):  # type: ignore[no-untyped-def]
        outputs, timing = orig_train_step(batch_by_hook, local_n)
        if outputs:
            losses.append(float(outputs[hook_name].loss.detach().item()))
        return outputs, timing

    trainer._train_step = _capture  # type: ignore[method-assign]
    trainer.fit()

    assert trainer.n_training_steps == N_STEPS, (
        f"expected {N_STEPS} steps, got {trainer.n_training_steps}"
    )
    assert len(losses) == N_STEPS

    # Loss must be finite throughout.
    assert all(torch.isfinite(torch.tensor(losses)).tolist()), losses

    # Parameters must have actually moved.
    final = _snapshot_params(sae_by_hook)
    delta = (final[hook_name] - initial[hook_name]).abs().max().item()
    assert delta > 0.0, f"W_enc unchanged after {N_STEPS} steps for {hook_name}"

    # Average loss in the second half should be lower than the first half.
    mid = N_STEPS // 2
    first_half = sum(losses[:mid]) / mid
    second_half = sum(losses[mid:]) / (N_STEPS - mid)
    assert second_half < first_half, (
        f"loss did not decrease for {hook_name}: "
        f"first_half={first_half:.4f}, second_half={second_half:.4f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_all_hook_types_train_together(tmp_path: Path) -> None:
    """All hook types as one MultiSAETrainer — exercises mixed-d_in dispatch."""
    device = _device()
    hook_to_din = dict(HOOK_TYPES)
    sae_by_hook = {hook: _make_sae(d_in, device) for hook, d_in in hook_to_din.items()}
    initial = _snapshot_params(sae_by_hook)

    total_samples = BATCH * N_STEPS
    cfg = _make_trainer_cfg(tmp_path, total_samples, device)
    provider = _structured_batches(hook_to_din, n_batches=N_STEPS, device=device)
    trainer = MultiSAETrainer(
        hook_names=list(hook_to_din.keys()),
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
        data_provider=provider,
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    per_hook_losses: dict[str, list[float]] = {hook: [] for hook in hook_to_din}
    orig_train_step = trainer._train_step

    def _capture(batch_by_hook, local_n):  # type: ignore[no-untyped-def]
        outputs, timing = orig_train_step(batch_by_hook, local_n)
        for hook, out in outputs.items():
            per_hook_losses[hook].append(float(out.loss.detach().item()))
        return outputs, timing

    trainer._train_step = _capture  # type: ignore[method-assign]
    trainer.fit()

    assert trainer.n_training_steps == N_STEPS

    final = _snapshot_params(sae_by_hook)
    mid = N_STEPS // 2
    for hook in hook_to_din:
        losses = per_hook_losses[hook]
        assert len(losses) == N_STEPS, hook
        assert all(torch.isfinite(torch.tensor(losses)).tolist()), (hook, losses)

        delta = (final[hook] - initial[hook]).abs().max().item()
        assert delta > 0.0, f"W_enc unchanged for {hook}"

        first_half = sum(losses[:mid]) / mid
        second_half = sum(losses[mid:]) / (N_STEPS - mid)
        assert second_half < first_half, (
            f"loss did not decrease for {hook}: "
            f"first_half={first_half:.4f}, second_half={second_half:.4f}"
        )
