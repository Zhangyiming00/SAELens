from __future__ import annotations

import contextlib
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.shared_activation_buffer import SharedActivationBuffer
from sae_lens.training.streaming_activation_provider import StreamingActivationProvider
from tests.helpers import assert_close, build_topk_sae_training_cfg, random_params

HOOK_NAMES = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
D_IN = 8
D_SAE = 16
K = 4
BATCH = 4


def _make_sae() -> TopKTrainingSAE:
    cfg = build_topk_sae_training_cfg(d_in=D_IN, d_sae=D_SAE, k=K)
    sae = TopKTrainingSAE(cfg)
    random_params(sae)
    return sae


def _make_trainer_cfg(
    tmp_path: Path,
    total_training_samples: int = 100,
) -> SAETrainerConfig:
    return SAETrainerConfig(
        device="cpu",
        n_checkpoints=0,
        total_training_samples=total_training_samples,
        train_batch_size_samples=BATCH,
        output_path=None,
        save_mse_every_n_steps=0,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=0,
        synchronize_timing=False,
        multi_sae_backward_order="forward",
        multi_sae_stats_sync_mode="immediate",
        multi_sae_stats_sync_interval=1,
        lr=1e-3,
        lr_end=None,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        dead_feature_window=100,
        feature_sampling_window=100,
        autocast=False,
        checkpoint_path=str(tmp_path / "checkpoints"),
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
    )


def _make_data_provider(
    n_batches: int,
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        yield {
            hook: torch.randn(BATCH, D_IN, generator=gen) for hook in HOOK_NAMES
        }


def _build_trainer(
    tmp_path: Path,
    total_samples: int,
    n_batches: int,
    seed: int = 42,
    sae_by_hook: dict[str, TopKTrainingSAE] | None = None,
    base_sae_by_hook: dict[str, TopKTrainingSAE] | None = None,
) -> MultiSAETrainer:
    if sae_by_hook is None:
        sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    if base_sae_by_hook is None:
        base_sae_by_hook = sae_by_hook
    provider = _make_data_provider(n_batches, seed=seed)
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=total_samples)
    return MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=provider,
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )


def test_checkpoint_round_trip_preserves_all_state(tmp_path: Path) -> None:
    sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    trainer = _build_trainer(
        tmp_path,
        total_samples=1000,
        n_batches=5,
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
    )
    # Run 5 steps
    for _ in range(5):
        batch = next(_make_data_provider(1, seed=99))
        batch_by_hook = {
            hook: trainer.activation_scaler_by_hook[hook](batch[hook])
            for hook in HOOK_NAMES
        }
        trainer._train_step(batch_by_hook, BATCH)
        trainer.n_training_samples += BATCH
        trainer.n_training_steps += 1
        trainer.lr_scheduler.step()

    # Manually poke stats so they're non-trivial
    for hook in HOOK_NAMES:
        trainer.act_freq_scores_by_hook[hook] += torch.rand(D_SAE)
        trainer.n_forward_passes_since_fired_by_hook[hook] += torch.randint(0, 10, (D_SAE,))
        trainer.n_frac_active_samples_by_hook[hook] = 42

    checkpoint_path = tmp_path / "ckpt"
    trainer.save_trainer_state(checkpoint_path)
    trainer.save_checkpoint(checkpoint_name="test_ckpt")

    ckpt_dir = Path(trainer.cfg.checkpoint_path) / "test_ckpt"

    # Build a fresh trainer and load
    fresh_sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    fresh_trainer = _build_trainer(
        tmp_path / "fresh",
        total_samples=1000,
        n_batches=0,
        sae_by_hook=fresh_sae_by_hook,
        base_sae_by_hook=fresh_sae_by_hook,
    )
    fresh_trainer.load_trainer_state(ckpt_dir)

    # Verify training counters
    assert fresh_trainer.n_training_samples == trainer.n_training_samples
    assert fresh_trainer.n_training_steps == trainer.n_training_steps

    # Verify per-hook weights
    for hook in HOOK_NAMES:
        orig_params = dict(trainer.base_sae_by_hook[hook].named_parameters())
        loaded_params = dict(fresh_trainer.base_sae_by_hook[hook].named_parameters())
        for name in orig_params:
            assert_close(loaded_params[name], orig_params[name], msg=f"{hook}/{name}")

    # Verify per-hook stats
    for hook in HOOK_NAMES:
        assert_close(
            fresh_trainer.act_freq_scores_by_hook[hook],
            trainer.act_freq_scores_by_hook[hook],
            msg=f"act_freq_scores {hook}",
        )
        assert_close(
            fresh_trainer.n_forward_passes_since_fired_by_hook[hook],
            trainer.n_forward_passes_since_fired_by_hook[hook],
            msg=f"n_forward_passes_since_fired {hook}",
        )
        assert (
            fresh_trainer.n_frac_active_samples_by_hook[hook]
            == trainer.n_frac_active_samples_by_hook[hook]
        )

    # Verify optimizer state exists for all params
    for hook in HOOK_NAMES:
        for param in fresh_trainer.base_sae_by_hook[hook].parameters():
            assert param in fresh_trainer.optimizer.state, (
                f"Missing optimizer state for param in {hook}"
            )

    # Verify lr_scheduler state
    assert (
        fresh_trainer.lr_scheduler.state_dict()
        == trainer.lr_scheduler.state_dict()
    )


def test_checkpoint_resume_mismatched_hooks_raises(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, total_samples=1000, n_batches=1)
    batch = next(_make_data_provider(1))
    batch_by_hook = {
        hook: trainer.activation_scaler_by_hook[hook](batch[hook])
        for hook in HOOK_NAMES
    }
    trainer._train_step(batch_by_hook, BATCH)
    trainer.n_training_samples += BATCH
    trainer.n_training_steps += 1

    ckpt_dir = tmp_path / "ckpt_mismatch"
    trainer.save_trainer_state(ckpt_dir)
    for hook in HOOK_NAMES:
        trainer._save_one_checkpoint_model(ckpt_dir, hook)

    # Build trainer with different hooks
    different_hooks = ["blocks.2.hook_resid_post", "blocks.3.hook_resid_post"]
    different_sae_by_hook = {hook: _make_sae() for hook in different_hooks}
    cfg = _make_trainer_cfg(tmp_path / "mismatch", total_training_samples=1000)
    bad_trainer = MultiSAETrainer(
        hook_names=different_hooks,
        sae_by_hook=different_sae_by_hook,
        base_sae_by_hook=different_sae_by_hook,
        data_provider=iter([]),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    with pytest.raises((ValueError, FileNotFoundError)):
        bad_trainer.load_trainer_state(ckpt_dir)


def _run_n_steps(
    trainer: MultiSAETrainer,
    data: list[dict[str, torch.Tensor]],
    n_steps: int,
) -> None:
    for i in range(n_steps):
        batch = data[i]
        batch_by_hook = {
            hook: trainer.activation_scaler_by_hook[hook](batch[hook])
            for hook in HOOK_NAMES
        }
        trainer._train_step(batch_by_hook, BATCH)
        trainer.n_training_samples += BATCH
        trainer.n_training_steps += 1
        trainer.lr_scheduler.step()


def test_checkpoint_resume_continues_training_correctly(tmp_path: Path) -> None:
    total_steps = 6
    split_at = 3
    data = list(_make_data_provider(total_steps, seed=123))

    # Run A: continuous training for all steps
    sae_a = {hook: _make_sae() for hook in HOOK_NAMES}
    # Clone initial weights so run B starts from the same point
    initial_state = {
        hook: deepcopy(sae_a[hook].state_dict()) for hook in HOOK_NAMES
    }
    trainer_a = _build_trainer(
        tmp_path / "a",
        total_samples=total_steps * BATCH,
        n_batches=0,
        sae_by_hook=sae_a,
        base_sae_by_hook=sae_a,
    )
    _run_n_steps(trainer_a, data, total_steps)

    # Run B: train split_at steps, checkpoint, resume, train remaining
    sae_b = {hook: TopKTrainingSAE(sae_a[hook].cfg) for hook in HOOK_NAMES}
    for hook in HOOK_NAMES:
        sae_b[hook].load_state_dict(initial_state[hook])
    trainer_b = _build_trainer(
        tmp_path / "b",
        total_samples=total_steps * BATCH,
        n_batches=0,
        sae_by_hook=sae_b,
        base_sae_by_hook=sae_b,
    )
    _run_n_steps(trainer_b, data[:split_at], split_at)

    ckpt_dir = tmp_path / "b_ckpt"
    trainer_b.save_trainer_state(ckpt_dir)
    for hook in HOOK_NAMES:
        trainer_b._save_one_checkpoint_model(ckpt_dir, hook)

    # Resume into fresh trainer
    sae_c = {hook: _make_sae() for hook in HOOK_NAMES}
    trainer_c = _build_trainer(
        tmp_path / "c",
        total_samples=total_steps * BATCH,
        n_batches=0,
        sae_by_hook=sae_c,
        base_sae_by_hook=sae_c,
    )
    trainer_c.load_trainer_state(ckpt_dir)
    assert trainer_c.n_training_steps == split_at
    _run_n_steps(trainer_c, data[split_at:], total_steps - split_at)

    # Final weights should match continuous run
    for hook in HOOK_NAMES:
        params_a = dict(trainer_a.base_sae_by_hook[hook].named_parameters())
        params_c = dict(trainer_c.base_sae_by_hook[hook].named_parameters())
        for name in params_a:
            assert_close(
                params_c[name],
                params_a[name],
                atol=1e-5,
                rtol=1e-4,
                msg=f"Weight mismatch after resume: {hook}/{name}",
            )


def _fill_buffer(
    buf: SharedActivationBuffer,
    n_chunks: int,
    chunk_tokens: int,
    d_model: int,
    num_hooks: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    written: list[torch.Tensor] = []
    for _ in range(n_chunks):
        acts = torch.randn(chunk_tokens * num_hooks, d_model, generator=gen)
        result = buf.allocate_write_chunk()
        assert result is not None
        chunk_idx, _ = result
        buf.write_chunk(chunk_idx, acts, valid_tokens=chunk_tokens * num_hooks)
        buf.mark_ready(chunk_idx)
        written.append(acts)
    buf.signal_done()
    return written


def _make_streaming_provider(
    buf: SharedActivationBuffer,
    hook_names: list[str],
    batch_size: int,
    d_model: int,
) -> StreamingActivationProvider:
    return StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=batch_size,
        prefetch_chunks=4,
        device=torch.device("cpu"),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=d_model,
        dtype=torch.float32,
        shuffle=False,
        random_chunks=False,
        hook_names=hook_names,
    )


def test_streaming_multi_hook_checkpoint_round_trip(tmp_path: Path) -> None:
    num_chunks = 4
    chunk_tokens = BATCH * 2
    total_tokens = num_chunks * chunk_tokens * BATCH
    buf_dir = tmp_path / "shm"
    buf_dir.mkdir()

    buf = SharedActivationBuffer(
        name="test_stream",
        num_chunks=num_chunks,
        chunk_size_tokens=chunk_tokens * len(HOOK_NAMES),
        d_model=D_IN,
        num_producers=1,
        target_chunks=num_chunks,
        create=True,
        base_dir=str(buf_dir),
        dtype=torch.float32,
    )
    _fill_buffer(buf, num_chunks, chunk_tokens, D_IN, len(HOOK_NAMES), seed=77)

    provider = _make_streaming_provider(buf, HOOK_NAMES, BATCH, D_IN)

    sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=total_tokens)
    trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
        data_provider=provider,
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    # Train until data runs out
    with contextlib.suppress(StopIteration):
        trainer.fit()

    assert trainer.n_training_steps > 0

    # Save checkpoint
    ckpt_dir = tmp_path / "stream_ckpt"
    trainer.save_trainer_state(ckpt_dir)
    for hook in HOOK_NAMES:
        trainer._save_one_checkpoint_model(ckpt_dir, hook)

    # Load into fresh trainer with a fresh buffer (no data — just verifying state)
    fresh_sae = {hook: _make_sae() for hook in HOOK_NAMES}
    fresh_cfg = _make_trainer_cfg(tmp_path / "fresh", total_training_samples=total_tokens)
    fresh_trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=fresh_sae,
        base_sae_by_hook=fresh_sae,
        data_provider=iter([]),
        save_checkpoint_fn=None,
        cfg=fresh_cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    fresh_trainer.load_trainer_state(ckpt_dir)

    assert fresh_trainer.n_training_steps == trainer.n_training_steps
    assert fresh_trainer.n_training_samples == trainer.n_training_samples

    for hook in HOOK_NAMES:
        orig = dict(trainer.base_sae_by_hook[hook].named_parameters())
        loaded = dict(fresh_trainer.base_sae_by_hook[hook].named_parameters())
        for name in orig:
            assert_close(loaded[name], orig[name], msg=f"streaming {hook}/{name}")

    for hook in HOOK_NAMES:
        assert_close(
            fresh_trainer.act_freq_scores_by_hook[hook],
            trainer.act_freq_scores_by_hook[hook],
        )
        assert (
            fresh_trainer.n_frac_active_samples_by_hook[hook]
            == trainer.n_frac_active_samples_by_hook[hook]
        )

    buf.close()


def test_streaming_multi_hook_resume_continues_training(tmp_path: Path) -> None:
    num_chunks = 6
    chunk_tokens = BATCH * 2
    total_tokens_per_chunk = chunk_tokens * len(HOOK_NAMES)

    # Create two identical buffers with the same data for run A and run B
    def make_filled_buffer(name: str, subdir: str) -> SharedActivationBuffer:
        d = tmp_path / subdir
        d.mkdir()
        b = SharedActivationBuffer(
            name=name,
            num_chunks=num_chunks,
            chunk_size_tokens=total_tokens_per_chunk,
            d_model=D_IN,
            num_producers=1,
            target_chunks=num_chunks,
            create=True,
            base_dir=str(d),
            dtype=torch.float32,
        )
        _fill_buffer(b, num_chunks, chunk_tokens, D_IN, len(HOOK_NAMES), seed=55)
        return b

    big_total = num_chunks * chunk_tokens * 100

    # Run A: continuous training
    buf_a = make_filled_buffer("buf_a", "shm_a")
    prov_a = _make_streaming_provider(buf_a, HOOK_NAMES, BATCH, D_IN)
    sae_a = {hook: _make_sae() for hook in HOOK_NAMES}
    initial_state = {hook: deepcopy(sae_a[hook].state_dict()) for hook in HOOK_NAMES}
    cfg_a = _make_trainer_cfg(tmp_path / "a", total_training_samples=big_total)
    trainer_a = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_a,
        base_sae_by_hook=sae_a,
        data_provider=prov_a,
        save_checkpoint_fn=None,
        cfg=cfg_a,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    with contextlib.suppress(StopIteration):
        trainer_a.fit()
    total_steps_a = trainer_a.n_training_steps
    assert total_steps_a > 2
    buf_a.close()

    # Run B: train half, checkpoint, resume with second half
    split_chunks = num_chunks // 2

    # First half buffer
    buf_b1_dir = tmp_path / "shm_b1"
    buf_b1_dir.mkdir()
    buf_b1 = SharedActivationBuffer(
        name="buf_b1",
        num_chunks=split_chunks,
        chunk_size_tokens=total_tokens_per_chunk,
        d_model=D_IN,
        num_producers=1,
        target_chunks=split_chunks,
        create=True,
        base_dir=str(buf_b1_dir),
        dtype=torch.float32,
    )
    _fill_buffer(buf_b1, split_chunks, chunk_tokens, D_IN, len(HOOK_NAMES), seed=55)

    prov_b1 = _make_streaming_provider(buf_b1, HOOK_NAMES, BATCH, D_IN)
    sae_b = {hook: TopKTrainingSAE(sae_a[hook].cfg) for hook in HOOK_NAMES}
    for hook in HOOK_NAMES:
        sae_b[hook].load_state_dict(initial_state[hook])
    cfg_b = _make_trainer_cfg(tmp_path / "b", total_training_samples=big_total)
    trainer_b = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_b,
        base_sae_by_hook=sae_b,
        data_provider=prov_b1,
        save_checkpoint_fn=None,
        cfg=cfg_b,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    with contextlib.suppress(StopIteration):
        trainer_b.fit()
    split_steps = trainer_b.n_training_steps
    assert split_steps > 0
    buf_b1.close()

    # Checkpoint
    ckpt_dir = tmp_path / "b_ckpt"
    trainer_b.save_trainer_state(ckpt_dir)
    for hook in HOOK_NAMES:
        trainer_b._save_one_checkpoint_model(ckpt_dir, hook)

    # Second half buffer — use seed offset to produce the remaining chunks
    # We need the same data as chunks [split_chunks:num_chunks] from seed=55
    # Recreate full sequence and skip first split_chunks
    buf_b2_dir = tmp_path / "shm_b2"
    buf_b2_dir.mkdir()
    remaining = num_chunks - split_chunks
    buf_b2 = SharedActivationBuffer(
        name="buf_b2",
        num_chunks=remaining,
        chunk_size_tokens=total_tokens_per_chunk,
        d_model=D_IN,
        num_producers=1,
        target_chunks=remaining,
        create=True,
        base_dir=str(buf_b2_dir),
        dtype=torch.float32,
    )
    # Regenerate data with same seed, skip first split_chunks
    gen = torch.Generator().manual_seed(55)
    for _ in range(split_chunks):
        torch.randn(total_tokens_per_chunk, D_IN, generator=gen)
    for _ in range(remaining):
        acts = torch.randn(total_tokens_per_chunk, D_IN, generator=gen)
        result = buf_b2.allocate_write_chunk()
        assert result is not None
        idx, _ = result
        buf_b2.write_chunk(idx, acts, valid_tokens=total_tokens_per_chunk)
        buf_b2.mark_ready(idx)
    buf_b2.signal_done()

    prov_b2 = _make_streaming_provider(buf_b2, HOOK_NAMES, BATCH, D_IN)
    sae_c = {hook: _make_sae() for hook in HOOK_NAMES}
    cfg_c = _make_trainer_cfg(tmp_path / "c", total_training_samples=big_total)
    trainer_c = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_c,
        base_sae_by_hook=sae_c,
        data_provider=prov_b2,
        save_checkpoint_fn=None,
        cfg=cfg_c,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    trainer_c.load_trainer_state(ckpt_dir)
    assert trainer_c.n_training_steps == split_steps
    with contextlib.suppress(StopIteration):
        trainer_c.fit()
    buf_b2.close()

    assert trainer_c.n_training_steps == total_steps_a

    for hook in HOOK_NAMES:
        params_a = dict(trainer_a.base_sae_by_hook[hook].named_parameters())
        params_c = dict(trainer_c.base_sae_by_hook[hook].named_parameters())
        for name in params_a:
            assert_close(
                params_c[name],
                params_a[name],
                atol=1e-5,
                rtol=1e-4,
                msg=f"Streaming resume mismatch: {hook}/{name}",
            )


# ---------------------------------------------------------------------------
# GPU streaming tests
# ---------------------------------------------------------------------------

GPU_HOOK_NAMES = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
GPU_D_IN = 128
GPU_D_SAE = 512
GPU_K = 16
GPU_BATCH = 64


def _make_gpu_sae(device: str = "cuda:0") -> TopKTrainingSAE:
    cfg = build_topk_sae_training_cfg(
        d_in=GPU_D_IN, d_sae=GPU_D_SAE, k=GPU_K, device=device
    )
    sae = TopKTrainingSAE(cfg)
    random_params(sae)
    return sae


def _make_gpu_trainer_cfg(
    tmp_path: Path,
    device: str = "cuda:0",
    total_training_samples: int = 10000,
) -> SAETrainerConfig:
    return SAETrainerConfig(
        device=device,
        n_checkpoints=0,
        total_training_samples=total_training_samples,
        train_batch_size_samples=GPU_BATCH,
        output_path=None,
        save_mse_every_n_steps=0,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=0,
        synchronize_timing=False,
        multi_sae_backward_order="forward",
        multi_sae_stats_sync_mode="immediate",
        multi_sae_stats_sync_interval=1,
        lr=1e-3,
        lr_end=None,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        dead_feature_window=1000,
        feature_sampling_window=100,
        autocast=False,
        checkpoint_path=str(tmp_path / "checkpoints"),
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
    )


def _make_gpu_buffer(
    buf_dir: Path,
    name: str,
    num_chunks: int,
    chunk_tokens: int,
    num_hooks: int,
) -> SharedActivationBuffer:
    buf_dir.mkdir(parents=True, exist_ok=True)
    return SharedActivationBuffer(
        name=name,
        num_chunks=num_chunks,
        chunk_size_tokens=chunk_tokens * num_hooks,
        d_model=GPU_D_IN,
        num_producers=1,
        target_chunks=num_chunks,
        create=True,
        base_dir=str(buf_dir),
        dtype=torch.float32,
    )


def _fill_gpu_buffer(
    buf: SharedActivationBuffer,
    n_chunks: int,
    chunk_tokens: int,
    num_hooks: int,
    seed: int,
) -> None:
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_chunks):
        acts = torch.randn(chunk_tokens * num_hooks, GPU_D_IN, generator=gen)
        result = buf.allocate_write_chunk()
        assert result is not None
        idx, _ = result
        buf.write_chunk(idx, acts, valid_tokens=chunk_tokens * num_hooks)
        buf.mark_ready(idx)
    buf.signal_done()


def _make_gpu_streaming_provider(
    buf: SharedActivationBuffer,
    device: str = "cuda:0",
) -> StreamingActivationProvider:
    return StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=GPU_BATCH,
        prefetch_chunks=4,
        device=torch.device(device),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=GPU_D_IN,
        dtype=torch.float32,
        shuffle=False,
        random_chunks=False,
        hook_names=GPU_HOOK_NAMES,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_streaming_multi_hook_checkpoint_resume(tmp_path: Path) -> None:
    device = "cuda:0"
    num_hooks = len(GPU_HOOK_NAMES)
    num_chunks = 8
    chunk_tokens = GPU_BATCH * 4
    big_total = num_chunks * chunk_tokens * 100

    # --- Phase 1: train for a while, checkpoint ---
    buf1 = _make_gpu_buffer(
        tmp_path / "shm1", "phase1", num_chunks, chunk_tokens, num_hooks
    )
    _fill_gpu_buffer(buf1, num_chunks, chunk_tokens, num_hooks, seed=42)
    prov1 = _make_gpu_streaming_provider(buf1, device)

    sae_by_hook = {hook: _make_gpu_sae(device) for hook in GPU_HOOK_NAMES}
    cfg1 = _make_gpu_trainer_cfg(tmp_path / "run1", device, big_total)
    trainer1 = MultiSAETrainer(
        hook_names=GPU_HOOK_NAMES,
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
        data_provider=prov1,
        save_checkpoint_fn=None,
        cfg=cfg1,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    with contextlib.suppress(StopIteration):
        trainer1.fit()
    buf1.close()

    steps_before = trainer1.n_training_steps
    samples_before = trainer1.n_training_samples
    assert steps_before > 0, "should have trained at least one step"

    # Snapshot pre-checkpoint state for comparison
    weights_before = {
        hook: {
            name: param.detach().cpu().clone()
            for name, param in trainer1.base_sae_by_hook[hook].named_parameters()
        }
        for hook in GPU_HOOK_NAMES
    }
    stats_before = {
        hook: {
            "act_freq": trainer1.act_freq_scores_by_hook[hook].cpu().clone(),
            "n_fired": trainer1.n_forward_passes_since_fired_by_hook[hook].cpu().clone(),
            "n_frac": trainer1.n_frac_active_samples_by_hook[hook],
        }
        for hook in GPU_HOOK_NAMES
    }
    lr_state_before = deepcopy(trainer1.lr_scheduler.state_dict())

    # Save checkpoint
    ckpt_dir = tmp_path / "gpu_ckpt"
    trainer1.save_trainer_state(ckpt_dir)
    for hook in GPU_HOOK_NAMES:
        trainer1._save_one_checkpoint_model(ckpt_dir, hook)

    # --- Phase 2: load checkpoint into fresh trainer, verify exact match ---
    fresh_sae = {hook: _make_gpu_sae(device) for hook in GPU_HOOK_NAMES}
    cfg2 = _make_gpu_trainer_cfg(tmp_path / "run2", device, big_total)
    trainer2 = MultiSAETrainer(
        hook_names=GPU_HOOK_NAMES,
        sae_by_hook=fresh_sae,
        base_sae_by_hook=fresh_sae,
        data_provider=iter([]),
        save_checkpoint_fn=None,
        cfg=cfg2,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    trainer2.load_trainer_state(ckpt_dir)

    assert trainer2.n_training_steps == steps_before
    assert trainer2.n_training_samples == samples_before

    for hook in GPU_HOOK_NAMES:
        for name, expected in weights_before[hook].items():
            actual = dict(trainer2.base_sae_by_hook[hook].named_parameters())[name]
            assert_close(
                actual.cpu(), expected,
                msg=f"Phase 2 weight mismatch: {hook}/{name}",
            )

    for hook in GPU_HOOK_NAMES:
        assert_close(
            trainer2.act_freq_scores_by_hook[hook].cpu(),
            stats_before[hook]["act_freq"],
            msg=f"Phase 2 act_freq_scores mismatch: {hook}",
        )
        assert_close(
            trainer2.n_forward_passes_since_fired_by_hook[hook].cpu(),
            stats_before[hook]["n_fired"],
            msg=f"Phase 2 n_forward_passes_since_fired mismatch: {hook}",
        )
        assert (
            trainer2.n_frac_active_samples_by_hook[hook]
            == stats_before[hook]["n_frac"]
        )

    for hook in GPU_HOOK_NAMES:
        for param in trainer2.base_sae_by_hook[hook].parameters():
            assert param in trainer2.optimizer.state, (
                f"Phase 2: missing optimizer state for param in {hook}"
            )

    assert trainer2.lr_scheduler.state_dict() == lr_state_before

    # --- Phase 3: continue training on new data, verify loss decreases ---
    buf3 = _make_gpu_buffer(
        tmp_path / "shm3", "phase3", num_chunks, chunk_tokens, num_hooks
    )
    _fill_gpu_buffer(buf3, num_chunks, chunk_tokens, num_hooks, seed=999)
    prov3 = _make_gpu_streaming_provider(buf3, device)

    # Swap in the new provider
    trainer2.data_provider = prov3

    # Collect losses over resumed training
    losses: list[float] = []
    step_count = 0
    for batch_by_hook in prov3:
        scaled = {
            hook: trainer2.activation_scaler_by_hook[hook](
                batch_by_hook[hook].to(device)
            )
            for hook in GPU_HOOK_NAMES
        }
        local_n = next(iter(batch_by_hook.values())).shape[0]
        outputs, _ = trainer2._train_step(scaled, local_n)
        trainer2.n_training_samples += local_n
        trainer2.n_training_steps += 1
        trainer2.lr_scheduler.step()
        avg_loss = sum(o.loss.item() for o in outputs.values()) / len(outputs)
        losses.append(avg_loss)
        step_count += 1
    buf3.close()

    assert step_count > 0, "should have trained on new data"
    assert trainer2.n_training_steps == steps_before + step_count

    # Weights should have changed from the checkpoint
    for hook in GPU_HOOK_NAMES:
        for name, before_val in weights_before[hook].items():
            after_val = dict(trainer2.base_sae_by_hook[hook].named_parameters())[name]
            assert not torch.allclose(after_val.cpu(), before_val, atol=1e-7), (
                f"Phase 3: weights didn't change after resumed training: {hook}/{name}"
            )

    torch.cuda.empty_cache()
