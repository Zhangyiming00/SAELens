from __future__ import annotations

import contextlib
import json
import pickle
import time
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.multi_hook_sae import MultiHookSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.multi_sae_trainer import _load_hook_optimizer_state_safetensors
from sae_lens.training.multi_sae_trainer import _load_tp_sharded_state_dict
from sae_lens.training.multi_sae_trainer import _save_hook_optimizer_state_safetensors
from sae_lens.training.multi_sae_trainer import sanitize_hook_name_for_path
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
        record_memory_empty_cache=False,
        record_memory_timeline_step=-1,
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
        quiesce_checkpoint_path=None,
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


def _make_step_input(sae: TopKTrainingSAE, batch: int = BATCH) -> TrainStepInput:
    return TrainStepInput(
        sae_in=torch.randn(batch, sae.cfg.d_in),
        dead_neuron_mask=torch.zeros(sae.cfg.d_sae, dtype=torch.bool),
        coefficients={},
        n_training_steps=0,
        is_logging_step=False,
    )


class _FailingLookup(torch.nn.Module):
    def __init__(self, base_sae: TopKTrainingSAE) -> None:
        super().__init__()
        self.base_sae = base_sae

    def forward(self, step_input: TrainStepInput) -> TrainStepOutput:
        raise AssertionError("trainer must enter through MultiHookSAE owner")


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


def test_multi_hook_sae_requires_exact_hook_set_and_splits_state_dict() -> None:
    base_sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    owner = MultiHookSAE(HOOK_NAMES, base_sae_by_hook)
    inputs = {
        hook: _make_step_input(base_sae_by_hook[hook])
        for hook in reversed(HOOK_NAMES)
    }

    outputs = owner(inputs)

    assert list(outputs.keys()) == HOOK_NAMES
    root_state = owner.state_dict()
    split_state = owner.split_state_dict_by_hook(root_state)
    assert set(split_state) == set(HOOK_NAMES)
    for hook in HOOK_NAMES:
        assert sorted(split_state[hook]) == sorted(base_sae_by_hook[hook].state_dict())
    merged_state = owner.merge_state_dict_by_hook(split_state)
    assert sorted(merged_state) == sorted(root_state)

    with pytest.raises(ValueError, match="exact hook set"):
        owner({HOOK_NAMES[0]: _make_step_input(base_sae_by_hook[HOOK_NAMES[0]])})


def test_unified_trainer_enters_multi_hook_owner_not_compat_lookup(tmp_path: Path) -> None:
    base_sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    owner = MultiHookSAE(HOOK_NAMES, base_sae_by_hook)
    failing_lookup = {
        hook: _FailingLookup(base_sae_by_hook[hook])
        for hook in HOOK_NAMES
    }
    provider = _make_data_provider(1, seed=123)
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=BATCH)
    cfg.multi_sae_distributed_architecture = "unified_multi_hook"
    trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=failing_lookup,
        base_sae_by_hook=base_sae_by_hook,
        multi_hook_sae=owner,
        data_provider=provider,
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    batch = next(_make_data_provider(1, seed=456))

    outputs, _ = trainer._train_step(batch, BATCH)

    assert list(outputs.keys()) == HOOK_NAMES


def test_unified_trainer_requires_exact_hook_batch_set(tmp_path: Path) -> None:
    base_sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    owner = MultiHookSAE(HOOK_NAMES, base_sae_by_hook)
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=BATCH)
    cfg.multi_sae_distributed_architecture = "unified_multi_hook"
    trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        multi_hook_sae=owner,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    valid_batch = next(_make_data_provider(1, seed=654))

    with pytest.raises(ValueError, match="exactly"):
        trainer._train_step({HOOK_NAMES[0]: valid_batch[HOOK_NAMES[0]]}, BATCH)
    with pytest.raises(ValueError, match="exactly"):
        trainer._train_step(
            {
                **valid_batch,
                "blocks.2.hook_resid_post": torch.randn(BATCH, D_IN),
            },
            BATCH,
        )


def test_unified_checkpoint_round_trip_preserves_per_hook_files_and_architecture(
    tmp_path: Path,
) -> None:
    base_sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    owner = MultiHookSAE(HOOK_NAMES, base_sae_by_hook)
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=1000)
    cfg.checkpoint_path = str(tmp_path / "checkpoints")
    cfg.multi_sae_distributed_architecture = "unified_multi_hook"
    trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        multi_hook_sae=owner,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    batch = next(_make_data_provider(1, seed=987))
    trainer._train_step(batch, BATCH)
    trainer.n_training_samples = BATCH
    trainer.n_training_steps = 1

    trainer.save_checkpoint("4")

    checkpoint_path = Path(cfg.checkpoint_path) / "4"
    trainer_state = torch.load(checkpoint_path / "trainer_state.pt", map_location="cpu")
    assert trainer_state["multi_sae_distributed_architecture"] == "unified_multi_hook"
    for hook_name in HOOK_NAMES:
        hook_dir = checkpoint_path / sanitize_hook_name_for_path(hook_name)
        assert (hook_dir / "sae_weights.safetensors").exists()
        assert (hook_dir / "cfg.json").exists()

    fresh_base_sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    fresh_owner = MultiHookSAE(HOOK_NAMES, fresh_base_sae_by_hook)
    fresh_trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=fresh_base_sae_by_hook,
        base_sae_by_hook=fresh_base_sae_by_hook,
        multi_hook_sae=fresh_owner,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    fresh_trainer.load_trainer_state(checkpoint_path)

    for hook_name in HOOK_NAMES:
        expected = dict(trainer.base_sae_by_hook[hook_name].named_parameters())
        actual = dict(fresh_trainer.base_sae_by_hook[hook_name].named_parameters())
        for name, expected_param in expected.items():
            assert_close(actual[name], expected_param, msg=f"{hook_name}/{name}")


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device sampler is cuda-only")
def test_device_sampler_writes_timestamped_records(tmp_path: Path) -> None:
    sae_by_hook = {hook: _make_sae().to("cuda") for hook in HOOK_NAMES}
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=100)
    cfg.device = "cuda"
    cfg.output_path = str(tmp_path / "out")
    cfg.save_memory_every_n_steps = 1
    trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
        data_provider=iter([]),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    trainer._device_sampler_interval_s = 0.02

    trainer._start_device_sampler()
    # Allocate after sampling starts so device_used must reflect a real load.
    scratch = torch.empty(64 * 1024 * 1024 // 4, device="cuda")  # 64 MB
    time.sleep(0.25)
    del scratch
    trainer._stop_device_sampler()

    assert trainer._device_sampler_thread is None
    rows = [
        json.loads(line)
        for line in trainer.device_history_path.read_text().splitlines()
    ]
    # ~0.25s at 0.02s interval -> several samples; require at least a handful.
    assert len(rows) >= 5
    times = [r["t_s"] for r in rows]
    assert times == sorted(times)
    assert all(r["device_used_mb"] > 0 for r in rows)
    # device_used is the OS watermark and must be >= this process' reserved pool.
    assert all(r["device_used_mb"] >= r["reserved_mb"] for r in rows)
    assert all(r["reserved_mb"] >= r["allocated_mb"] for r in rows)


def test_append_history_logs_preserves_existing_multi_sae_history(tmp_path: Path) -> None:
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=100)
    cfg.output_path = str(tmp_path / "out")
    cfg.save_mse_every_n_steps = 1
    cfg.save_timing_every_n_steps = 1
    output = Path(cfg.output_path)
    output.mkdir()
    (output / "mse_history.jsonl").write_text('{"existing": "mse"}\n')
    (output / "timing_history.jsonl").write_text('{"existing": "timing"}\n')

    sae_by_hook = {hook: _make_sae() for hook in HOOK_NAMES}
    MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook=sae_by_hook,
        base_sae_by_hook=sae_by_hook,
        data_provider=_make_data_provider(1),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
        append_logs=True,
    )

    assert (output / "mse_history.jsonl").read_text() == '{"existing": "mse"}\n'
    assert (output / "timing_history.jsonl").read_text() == '{"existing": "timing"}\n'


def test_save_final_writes_pp_local_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=100)
    cfg.output_path = str(tmp_path / "out")
    trainer = MultiSAETrainer(
        hook_names=HOOK_NAMES,
        sae_by_hook={hook: _make_sae() for hook in HOOK_NAMES},
        base_sae_by_hook={hook: _make_sae() for hook in HOOK_NAMES},
        data_provider=_make_data_provider(1),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    monkeypatch.setattr("sae_lens.training.multi_sae_trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("sae_lens.training.multi_sae_trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr("sae_lens.training.multi_sae_trainer.dist.get_rank", lambda group=None: 1)
    monkeypatch.setattr(trainer, "_pp_rank", lambda: 1)
    monkeypatch.setattr(trainer, "_tp_rank", lambda: 0)
    monkeypatch.setattr(trainer, "_dp_rank", lambda: 0)

    trainer.save_final(cfg.output_path)

    local_manifest = Path(cfg.output_path) / "multi_sae_manifest_pp1_rank1.json"
    assert local_manifest.exists()
    manifest = json.loads(local_manifest.read_text())
    assert manifest["hook_names"] == HOOK_NAMES


def test_record_mse_non_writer_pp_stage_writes_rank_local_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=100)
    cfg.output_path = str(tmp_path / "out")
    cfg.save_mse_every_n_steps = 1
    trainer = MultiSAETrainer(
        hook_names=[HOOK_NAMES[1]],
        sae_by_hook={HOOK_NAMES[1]: _make_sae()},
        base_sae_by_hook={HOOK_NAMES[1]: _make_sae()},
        data_provider=_make_data_provider(1),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    class Output:
        loss = torch.tensor(2.25)
        losses = {"mse_loss": torch.tensor(2.0)}
        sae_out = torch.zeros(1, D_IN)
        sae_in = torch.ones(1, D_IN)

    monkeypatch.setattr("sae_lens.training.multi_sae_trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("sae_lens.training.multi_sae_trainer.dist.is_initialized", lambda: True)
    monkeypatch.setattr("sae_lens.training.multi_sae_trainer.dist.get_rank", lambda group=None: 1)
    monkeypatch.setattr(trainer, "_pp_rank", lambda: 1)
    monkeypatch.setattr(trainer, "_tp_rank", lambda: 0)
    monkeypatch.setattr(trainer, "_dp_rank", lambda: 0)

    trainer.n_training_samples = BATCH
    trainer._record_mse_if_needed({HOOK_NAMES[1]: Output()}, local_n=BATCH)  # type: ignore[arg-type]

    records = [
        json.loads(line)
        for line in (
            Path(cfg.output_path) / "mse_history_pp1_rank1.jsonl"
        ).read_text().splitlines()
    ]
    assert records[0]["pp_rank"] == 1
    assert set(records[0]["hooks"]) == {HOOK_NAMES[1]}
    assert records[0]["hooks"][HOOK_NAMES[1]]["mse_loss"] == 2.0


def test_multi_sae_checkpoint_stores_adam_outside_root_trainer_state(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path, total_samples=100, n_batches=1)
    batch = next(_make_data_provider(1))
    batch_by_hook = {
        hook: trainer.activation_scaler_by_hook[hook](batch[hook])
        for hook in HOOK_NAMES
    }
    trainer._train_step(batch_by_hook, BATCH)
    trainer.n_training_samples += BATCH
    trainer.n_training_steps += 1

    ckpt_dir = tmp_path / "ckpt"
    trainer.save_trainer_state(ckpt_dir)
    for hook_name in HOOK_NAMES:
        trainer._save_one_checkpoint_model(ckpt_dir, hook_name)

    root_state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu")
    assert "optimizer_by_hook_by_name" not in root_state
    for hook_name in HOOK_NAMES:
        hook_dir = ckpt_dir / hook_name.replace(".", "_")
        assert (hook_dir / "optimizer_state.safetensors").exists()
        hook_state = torch.load(hook_dir / "hook_state.pt", map_location="cpu")
        assert "optimizer_state" not in hook_state

    resumed = _build_trainer(tmp_path / "resume", total_samples=100, n_batches=1)
    resumed.load_trainer_state(ckpt_dir)
    assert resumed.n_training_samples == trainer.n_training_samples
    assert resumed.n_training_steps == trainer.n_training_steps
    assert len(resumed.optimizer.state) == len(trainer.optimizer.state)


def test_load_tp_sharded_state_dict_reads_only_local_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(
        {
            "W_enc": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "W_dec": torch.arange(48, dtype=torch.float32).reshape(8, 6),
            "b_enc": torch.arange(8, dtype=torch.float32),
            "b_dec": torch.arange(4, dtype=torch.float32),
        },
        weights_path,
    )
    base_sae = SimpleNamespace(
        _tp_param_shard_dims=lambda: {
            "W_enc": 1,
            "W_dec": 0,
            "b_enc": 0,
            "b_dec": None,
        }
    )
    monkeypatch.setattr(
        "sae_lens.training.multi_sae_trainer.dist.get_world_size",
        lambda group: 2,
    )
    monkeypatch.setattr(
        "sae_lens.training.multi_sae_trainer.dist.get_rank",
        lambda group: 1,
    )
    monkeypatch.setattr(
        "sae_lens.training.multi_sae_trainer.dist.scatter",
        lambda *args, **kwargs: pytest.fail(
            "TP checkpoint load must not scatter CPU tensors"
        ),
    )
    monkeypatch.setattr(
        "sae_lens.training.multi_sae_trainer.dist.broadcast",
        lambda *args, **kwargs: pytest.fail(
            "TP checkpoint load must not broadcast CPU tensors"
        ),
    )

    state = _load_tp_sharded_state_dict(weights_path, base_sae, object())

    assert torch.equal(
        state["W_enc"],
        torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, 4:8],
    )
    assert torch.equal(
        state["W_dec"],
        torch.arange(48, dtype=torch.float32).reshape(8, 6)[4:8, :],
    )
    assert torch.equal(state["b_enc"], torch.arange(8, dtype=torch.float32)[4:8])
    assert torch.equal(state["b_dec"], torch.arange(4, dtype=torch.float32))


def test_hook_optimizer_safetensors_loads_tp2_slices_and_tp1_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    optimizer_state = {
        "W_enc": {
            "step": torch.tensor(3.0),
            "exp_avg": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "exp_avg_sq": torch.arange(100, 132, dtype=torch.float32).reshape(4, 8),
        },
        "W_dec": {
            "step": torch.tensor(3.0),
            "exp_avg": torch.arange(48, dtype=torch.float32).reshape(8, 6),
        },
        "b_enc": {
            "step": torch.tensor(3.0),
            "exp_avg": torch.arange(8, dtype=torch.float32),
        },
        "b_dec": {
            "step": torch.tensor(3.0),
            "exp_avg": torch.arange(4, dtype=torch.float32),
        },
    }
    _save_hook_optimizer_state_safetensors(hook_dir, optimizer_state)
    base_sae = SimpleNamespace(
        _tp_param_shard_dims=lambda: {
            "W_enc": 1,
            "W_dec": 0,
            "b_enc": 0,
            "b_dec": None,
        }
    )

    full = _load_hook_optimizer_state_safetensors(
        hook_dir,
        base_sae,
        tp_group=None,
    )
    assert torch.equal(full["W_enc"]["exp_avg"], optimizer_state["W_enc"]["exp_avg"])
    assert torch.equal(full["W_dec"]["exp_avg"], optimizer_state["W_dec"]["exp_avg"])
    assert torch.equal(full["b_enc"]["exp_avg"], optimizer_state["b_enc"]["exp_avg"])
    assert torch.equal(full["b_dec"]["exp_avg"], optimizer_state["b_dec"]["exp_avg"])

    monkeypatch.setattr(
        "sae_lens.training.multi_sae_trainer.dist.get_world_size",
        lambda group: 2,
    )
    monkeypatch.setattr(
        "sae_lens.training.multi_sae_trainer.dist.get_rank",
        lambda group: 1,
    )
    local = _load_hook_optimizer_state_safetensors(hook_dir, base_sae, object())

    assert torch.equal(
        local["W_enc"]["exp_avg"],
        optimizer_state["W_enc"]["exp_avg"][:, 4:8],
    )
    assert torch.equal(
        local["W_enc"]["exp_avg_sq"],
        optimizer_state["W_enc"]["exp_avg_sq"][:, 4:8],
    )
    assert torch.equal(
        local["W_dec"]["exp_avg"],
        optimizer_state["W_dec"]["exp_avg"][4:8, :],
    )
    assert torch.equal(
        local["b_enc"]["exp_avg"],
        optimizer_state["b_enc"]["exp_avg"][4:8],
    )
    assert torch.equal(local["b_dec"]["exp_avg"], optimizer_state["b_dec"]["exp_avg"])
    assert torch.equal(local["W_enc"]["step"], optimizer_state["W_enc"]["step"])


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
        record_memory_empty_cache=False,
        record_memory_timeline_step=-1,
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
        quiesce_checkpoint_path=None,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_data_provider_buffer_bytes_sums_gpu_streaming_shapes(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, total_samples=BATCH, n_batches=1)
    # GpuStreamingActivationProvider-shaped buffers: two dicts of tensors plus
    # a list-of-dicts chunk buffer. 4 + 8 + 2 = 14 fp32 rows of width D_IN.
    pool = {h: torch.zeros(4, D_IN, device="cuda") for h in HOOK_NAMES}
    serving = {h: torch.zeros(8, D_IN, device="cuda") for h in HOOK_NAMES}
    chunk_buffer = [{h: torch.zeros(2, D_IN, device="cuda") for h in HOOK_NAMES}]
    trainer.data_provider = SimpleNamespace(
        _pool_by_hook=pool,
        _serving_by_hook=serving,
        _chunk_buffer=chunk_buffer,
    )
    expected = len(HOOK_NAMES) * (4 + 8 + 2) * D_IN * 4
    assert trainer._data_provider_buffer_bytes(set()) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_data_provider_buffer_bytes_handles_single_mixing_pool(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path, total_samples=BATCH, n_batches=1)
    # StreamingActivationProvider-shaped buffer: one _mixing_pool tensor.
    trainer.data_provider = SimpleNamespace(
        _mixing_pool=torch.zeros(7, D_IN, device="cuda")
    )
    assert trainer._data_provider_buffer_bytes(set()) == 7 * D_IN * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_data_provider_buffer_bytes_recurses_into_inner(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, total_samples=BATCH, n_batches=1)
    # GpuDirectDataProvider wraps an inner provider; buffers live on _inner.
    inner = SimpleNamespace(
        _pool_by_hook={h: torch.zeros(3, D_IN, device="cuda") for h in HOOK_NAMES}
    )
    trainer.data_provider = SimpleNamespace(_inner=inner)
    assert trainer._data_provider_buffer_bytes(set()) == (
        len(HOOK_NAMES) * 3 * D_IN * 4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_data_provider_buffer_bytes_dedupes_against_seen_batch(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path, total_samples=BATCH, n_batches=1)
    served = torch.zeros(4, D_IN, device="cuda")
    residual = torch.zeros(5, D_IN, device="cuda")
    # The pool still references `served` (already counted as the batch) plus a
    # not-yet-served `residual`. With a shared `seen`, only the residual counts.
    trainer.data_provider = SimpleNamespace(
        _pool_by_hook={HOOK_NAMES[0]: served, HOOK_NAMES[1]: residual}
    )
    seen: set[int] = set()
    trainer._tensor_tree_bytes(served, seen)  # pretend already counted as batch
    assert trainer._data_provider_buffer_bytes(seen) == 5 * D_IN * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_data_provider_buffer_bytes_ignores_non_cuda_tensors(
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path, total_samples=BATCH, n_batches=1)
    # A cross-process pool is not Python-reachable; a CPU pool stands in for
    # "not on this process's cuda device" and must contribute nothing.
    trainer.data_provider = SimpleNamespace(
        _mixing_pool=torch.zeros(7, D_IN, device="cpu")
    )
    assert trainer._data_provider_buffer_bytes(set()) == 0


def test_memory_timeline_disabled_by_default(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, total_samples=BATCH, n_batches=1)
    assert trainer._memory_timeline_step == -1
    assert trainer.memory_timeline_path is None
    # Hooks must be inert when disabled (no recorder started, no dump).
    trainer._maybe_start_memory_timeline()
    assert trainer._memory_timeline_active is False
    trainer._maybe_stop_memory_timeline()  # no-op, must not raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memory_timeline_dumps_pickle_for_target_step(tmp_path: Path) -> None:
    sae_by_hook = {hook: _make_sae().to("cuda") for hook in HOOK_NAMES}
    cfg = _make_trainer_cfg(tmp_path, total_training_samples=3 * BATCH)
    cfg.device = "cuda"
    cfg.output_path = str(tmp_path / "out")
    cfg.save_memory_every_n_steps = 1
    cfg.record_memory_timeline_step = 1
    provider = _make_data_provider(3, seed=7)
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
    expected = Path(cfg.output_path) / f"memory_timeline_rank{trainer._memory_rank}.pickle"
    assert trainer.memory_timeline_path == expected

    trainer.fit()

    # Recording is scoped to exactly the target step and stopped afterwards.
    assert trainer._memory_timeline_active is False
    assert expected.exists()
    assert expected.stat().st_size > 0
    # The dumped snapshot must be a loadable allocator history with events.
    with open(expected, "rb") as f:
        snapshot = pickle.load(f)
    assert snapshot["device_traces"] or snapshot["segments"]
