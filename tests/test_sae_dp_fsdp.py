"""Tests for sae_dp_mode config validation, forward() dispatch, clip_grad_norm_
with dp_group, base_sae accessor, and FSDP save/load weight format."""

import os
import socket
import unittest.mock as mock
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from sae_lens.config import (
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    SAETrainerConfig,
)
from sae_lens.constants import (
    FSDP_OPTIMIZER_STATE_FILENAME_TEMPLATE,
    SAE_WEIGHTS_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.sae_trainer import SAETrainer
from tests.helpers import build_topk_sae_training_cfg, random_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sae(d_in: int = 8, d_sae: int = 16, k: int = 4) -> TopKTrainingSAE:
    cfg = build_topk_sae_training_cfg(d_in=d_in, d_sae=d_sae, k=k)
    sae = TopKTrainingSAE(cfg)
    random_params(sae)
    return sae


def _make_step_input(sae: TopKTrainingSAE, batch: int = 4) -> TrainStepInput:
    return TrainStepInput(
        sae_in=torch.randn(batch, sae.cfg.d_in),
        dead_neuron_mask=torch.zeros(sae.cfg.d_sae, dtype=torch.bool),
        coefficients={},
        n_training_steps=0,
        is_logging_step=False,
    )


def _make_trainer_cfg(
    *,
    device: str = "cpu",
    total_training_samples: int = 100,
    train_batch_size_samples: int = 4,
) -> SAETrainerConfig:
    return SAETrainerConfig(
        device=device,
        n_checkpoints=0,
        total_training_samples=total_training_samples,
        train_batch_size_samples=train_batch_size_samples,
        output_path=None,
        save_mse_every_n_steps=0,
        save_timing_every_n_steps=0,
        synchronize_timing=False,
        lr=1e-3,
        lr_end=1e-4,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        dead_feature_window=100,
        feature_sampling_window=100,
        autocast=False,
        checkpoint_path=None,
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


class _RecordingWrapper(torch.nn.Module):
    def __init__(self, base_sae: TopKTrainingSAE) -> None:
        super().__init__()
        self.base_sae = base_sae
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, step_input: TrainStepInput) -> TrainStepOutput:
        self.calls += 1
        sae_in = step_input.sae_in
        loss = self.weight * sae_in.sum()
        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_in,
            feature_acts=torch.zeros(
                sae_in.shape[0],
                self.base_sae.cfg.d_sae,
                device=sae_in.device,
            ),
            hidden_pre=torch.zeros(
                sae_in.shape[0],
                self.base_sae.cfg.d_sae,
                device=sae_in.device,
            ),
            loss=loss,
            losses={"mse_loss": loss},
        )


def _ddp_worker(
    rank: int,
    world_size: int,
    d_in: int,
    d_sae: int,
    k: int,
    state_dict: dict[str, torch.Tensor],
    x_per_rank: list[torch.Tensor],
    result_list: list,
    port: int,
    save_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        sae = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
        sae.load_state_dict(state_dict)
        ddp_sae = DDP(sae, process_group=dist.group.WORLD)
        trainer = SAETrainer(
            cfg=_make_trainer_cfg(
                total_training_samples=x_per_rank[rank].shape[0] * world_size,
                train_batch_size_samples=x_per_rank[rank].shape[0],
            ),
            sae=ddp_sae,
            base_sae=sae,
            data_provider=mock.MagicMock(),
            dp_group=dist.group.WORLD,
        )
        trainer._train_step(ddp_sae, x_per_rank[rank])
        trainer.n_training_steps = 1
        trainer.n_training_samples = x_per_rank[rank].shape[0]

        saved_keys = None
        saved_mode = None
        if rank == 0:
            trainer._save_model(Path(save_dir))
            trainer.save_trainer_state(Path(save_dir))
            saved_state = load_file(Path(save_dir) / SAE_WEIGHTS_FILENAME)
            saved_keys = sorted(saved_state.keys())
            trainer_state = torch.load(Path(save_dir) / TRAINER_STATE_FILENAME)
            saved_mode = trainer_state["sae_dp_mode"]
        dist.barrier()

        resumed_sae = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
        resumed_ddp_sae = DDP(resumed_sae, process_group=dist.group.WORLD)
        resumed_trainer = SAETrainer(
            cfg=_make_trainer_cfg(
                total_training_samples=x_per_rank[rank].shape[0] * world_size,
                train_batch_size_samples=x_per_rank[rank].shape[0],
            ),
            sae=resumed_ddp_sae,
            base_sae=resumed_sae,
            data_provider=mock.MagicMock(),
            dp_group=dist.group.WORLD,
        )
        resumed_trainer.load_trainer_state(Path(save_dir))
        resumed_sae.load_weights_from_checkpoint(Path(save_dir))

        params = {
            name: param.detach().clone()
            for name, param in trainer.base_sae.named_parameters()
        }
        resumed_params = {
            name: param.detach().clone()
            for name, param in resumed_trainer.base_sae.named_parameters()
        }
        resumed_has_optimizer_state = all(
            param in resumed_trainer.optimizer.state
            for param in resumed_trainer.base_sae.parameters()
        )
        result_list.append(
            (
                rank,
                params,
                trainer.sae_dp_mode,
                saved_keys,
                saved_mode,
                resumed_params,
                resumed_trainer.sae_dp_mode,
                resumed_trainer.n_training_steps,
                resumed_trainer.n_training_samples,
                resumed_has_optimizer_state,
            )
        )
    finally:
        dist.destroy_process_group()


def _tp2_ddp2_worker(
    rank: int,
    world_size: int,
    d_in: int,
    d_sae: int,
    k: int,
    state_dict: dict[str, torch.Tensor],
    x_per_dp_rank: list[torch.Tensor],
    result_list: list,
    port: int,
    save_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        tp_groups = [
            dist.new_group([0, 1], backend="gloo"),
            dist.new_group([2, 3], backend="gloo"),
        ]
        dp_groups = [
            dist.new_group([0, 2], backend="gloo"),
            dist.new_group([1, 3], backend="gloo"),
        ]
        dp_rank = rank // 2
        tp_rank = rank % 2
        tp_group = tp_groups[dp_rank]
        dp_group = dp_groups[tp_rank]

        sae = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
        sae.load_state_dict(state_dict)
        sae.shard_weights(tp_group)
        ddp_sae = DDP(sae, process_group=dp_group)
        trainer = SAETrainer(
            cfg=_make_trainer_cfg(
                total_training_samples=x_per_dp_rank[dp_rank].shape[0] * 2,
                train_batch_size_samples=x_per_dp_rank[dp_rank].shape[0],
            ),
            sae=ddp_sae,
            base_sae=sae,
            data_provider=mock.MagicMock(),
            dp_group=dp_group,
        )
        trainer._train_step(ddp_sae, x_per_dp_rank[dp_rank])

        if dist.get_rank(dp_group) == 0:
            trainer._save_model(Path(save_dir))
        dist.barrier()

        saved_shapes = None
        saved_keys = None
        if rank == 0:
            saved_state = load_file(Path(save_dir) / SAE_WEIGHTS_FILENAME)
            saved_keys = sorted(saved_state.keys())
            saved_shapes = {name: tuple(value.shape) for name, value in saved_state.items()}

        params = {
            name: param.detach().clone()
            for name, param in trainer.base_sae.named_parameters()
        }
        result_list.append((rank, params, saved_keys, saved_shapes))
    finally:
        dist.destroy_process_group()


def _fsdp_resume_worker(
    rank: int,
    world_size: int,
    d_in: int,
    d_sae: int,
    k: int,
    state_dict: dict[str, torch.Tensor],
    x_per_rank: list[torch.Tensor],
    result_list: list,
    port: int,
    save_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"
    try:
        sae = _make_sae(d_in=d_in, d_sae=d_sae, k=k).to(device)
        sae.load_state_dict({name: value.to(device) for name, value in state_dict.items()})
        fsdp_sae = FSDP(
            sae,
            process_group=dist.group.WORLD,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=torch.device(device),
        )
        trainer = SAETrainer(
            cfg=_make_trainer_cfg(
                device=device,
                total_training_samples=x_per_rank[rank].shape[0] * world_size,
                train_batch_size_samples=x_per_rank[rank].shape[0],
            ),
            sae=fsdp_sae,
            base_sae=sae,
            data_provider=mock.MagicMock(),
            dp_group=dist.group.WORLD,
        )
        trainer._train_step(fsdp_sae, x_per_rank[rank].to(device))
        trainer.n_training_steps = 1
        trainer.n_training_samples = x_per_rank[rank].shape[0]
        trainer._save_model(Path(save_dir))
        trainer.save_trainer_state(Path(save_dir))
        dist.barrier()

        resumed_sae = _make_sae(d_in=d_in, d_sae=d_sae, k=k).to(device)
        resumed_sae.load_weights_from_checkpoint(Path(save_dir))
        resumed_fsdp_sae = FSDP(
            resumed_sae,
            process_group=dist.group.WORLD,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=torch.device(device),
        )
        resumed_trainer = SAETrainer(
            cfg=_make_trainer_cfg(
                device=device,
                total_training_samples=x_per_rank[rank].shape[0] * world_size,
                train_batch_size_samples=x_per_rank[rank].shape[0],
            ),
            sae=resumed_fsdp_sae,
            base_sae=resumed_sae,
            data_provider=mock.MagicMock(),
            dp_group=dist.group.WORLD,
        )
        resumed_trainer.load_trainer_state(Path(save_dir))

        full_state_matches_saved = True
        saved_mode = None
        saved_format = None
        saved_dp_size = None
        if rank == 0:
            saved_state = load_file(Path(save_dir) / SAE_WEIGHTS_FILENAME)
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                resumed_fsdp_sae,
                StateDictType.FULL_STATE_DICT,
                fsdp_cfg,
            ):
                resumed_state = resumed_fsdp_sae.state_dict()
            for name, expected in saved_state.items():
                if not torch.allclose(resumed_state[name], expected):
                    full_state_matches_saved = False
                    break
            trainer_state = torch.load(Path(save_dir) / TRAINER_STATE_FILENAME)
            saved_mode = trainer_state["sae_dp_mode"]
            saved_format = trainer_state["optimizer_state_format"]
            saved_dp_size = trainer_state["fsdp_dp_size"]
        else:
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                resumed_fsdp_sae,
                StateDictType.FULL_STATE_DICT,
                fsdp_cfg,
            ):
                resumed_fsdp_sae.state_dict()

        resumed_trainer._train_step(resumed_fsdp_sae, x_per_rank[rank].to(device))

        shard_path = Path(save_dir) / FSDP_OPTIMIZER_STATE_FILENAME_TEMPLATE.format(
            rank=rank
        )
        result_list.append(
            (
                rank,
                shard_path.exists(),
                resumed_trainer.sae_dp_mode,
                resumed_trainer.n_training_steps,
                resumed_trainer.n_training_samples,
                bool(resumed_trainer.optimizer.state),
                full_state_matches_saved,
                saved_mode,
                saved_format,
                saved_dp_size,
            )
        )
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_sae_dp_mode_ddp_accepted_without_dist() -> None:
    assert not dist.is_initialized(), "dist must not be initialized for this test"
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="ddp",
    )
    assert cfg.sae_dp_mode == "ddp"


def test_multi_hook_defaults_manual_to_ddp_without_dist() -> None:
    assert not dist.is_initialized(), "dist must not be initialized for this test"
    with pytest.warns(UserWarning, match="defaulting sae_dp_mode to 'ddp'"):
        cfg = LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            hook_names=[
                "blocks.20.hook_resid_post",
                "blocks.21.hook_resid_post",
            ],
        )
    assert cfg.sae_dp_mode == "ddp"


def test_multi_sae_trainer_ddp_mode_allows_dp1_without_dist() -> None:
    assert not dist.is_initialized(), "dist must not be initialized for this test"
    hook_names = ["blocks.20.hook_resid_post", "blocks.21.hook_resid_post"]
    base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=_make_trainer_cfg(total_training_samples=4),
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    batch_by_hook = {
        hook_name: torch.randn(4, base_sae_by_hook[hook_name].cfg.d_in)
        for hook_name in hook_names
    }

    outputs = trainer._train_step(batch_by_hook, local_n=4)

    assert set(outputs) == set(hook_names)


def test_multi_sae_trainer_token_weighting_only_for_ddp_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_names = ["blocks.20.hook_resid_post", "blocks.21.hook_resid_post"]
    base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    batch_by_hook = {
        hook_name: torch.randn(4, base_sae_by_hook[hook_name].cfg.d_in)
        for hook_name in hook_names
    }

    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=_make_trainer_cfg(total_training_samples=4),
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    all_reduce_sum_calls = mock.Mock(side_effect=trainer._all_reduce_sum)
    monkeypatch.setattr(trainer, "_all_reduce_sum", all_reduce_sum_calls)
    trainer._train_step(batch_by_hook, local_n=4)
    non_weighted_calls = all_reduce_sum_calls.call_count

    trainer_weighted = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=_make_trainer_cfg(total_training_samples=4),
        dp_group=None,
        token_count_weighted_dp=True,
        sae_dp_mode="ddp",
    )
    all_reduce_sum_calls_weighted = mock.Mock(side_effect=trainer_weighted._all_reduce_sum)
    monkeypatch.setattr(trainer_weighted, "_all_reduce_sum", all_reduce_sum_calls_weighted)
    trainer_weighted._train_step(batch_by_hook, local_n=4)
    # Weighted mode performs one additional global-token all-reduce.
    assert all_reduce_sum_calls_weighted.call_count == non_weighted_calls + 1


def test_multi_sae_trainer_checkpoint_saves_per_hook_weights_without_dist(
    tmp_path: Path,
) -> None:
    assert not dist.is_initialized(), "dist must not be initialized for this test"
    hook_names = ["blocks.20.hook_resid_post", "blocks.21.hook_resid_post"]
    base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    cfg = _make_trainer_cfg(total_training_samples=4)
    cfg.checkpoint_path = str(tmp_path)
    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    batch_by_hook = {
        hook_name: torch.randn(4, base_sae_by_hook[hook_name].cfg.d_in)
        for hook_name in hook_names
    }
    trainer._train_step(batch_by_hook, local_n=4)
    trainer.n_training_samples = 4
    trainer.n_training_steps = 1

    trainer.save_checkpoint("4")

    checkpoint_path = tmp_path / "4"
    assert (checkpoint_path / "multi_sae_manifest.json").exists()
    assert (checkpoint_path / TRAINER_STATE_FILENAME).exists()
    for hook_name in hook_names:
        hook_dir = checkpoint_path / hook_name.replace(".", "_")
        assert (hook_dir / SAE_WEIGHTS_FILENAME).exists()
        assert (hook_dir / "cfg.json").exists()

    resumed_base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    resumed = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=resumed_base_sae_by_hook,
        base_sae_by_hook=resumed_base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    resumed.load_trainer_state(checkpoint_path)
    assert resumed.n_training_samples == 4
    assert resumed.n_training_steps == 1


def test_multi_sae_tp_rank_falls_back_to_distributed_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    hook_names = ["blocks.21.hook_resid_post"]
    base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=_make_trainer_cfg(total_training_samples=4),
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )
    assert getattr(base_sae_by_hook[hook_names[0]], "_tp_group", None) is None

    import sae_lens.distributed_v2 as v2_mod

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(v2_mod, "_initialized", True)
    monkeypatch.setattr(v2_mod, "is_consumer", lambda: True)
    monkeypatch.setattr(v2_mod, "get_sae_tp_rank", lambda: 1)

    assert trainer._tp_rank() == 1


def test_multi_sae_metric_writer_uses_fallback_tp_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    hook_names = ["blocks.21.hook_resid_post"]
    base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=_make_trainer_cfg(total_training_samples=4),
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    import sae_lens.distributed_v2 as v2_mod

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(v2_mod, "_initialized", True)
    monkeypatch.setattr(v2_mod, "is_consumer", lambda: True)
    monkeypatch.setattr(v2_mod, "get_sae_tp_rank", lambda: 1)

    # dp_rank defaults to 0 when dp_group is None, so writer gating depends on tp_rank.
    assert trainer._is_metric_writer_rank() is False


def test_multi_sae_global_timing_uses_local_writer_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook_names = ["blocks.21.hook_resid_post"]
    base_sae_by_hook = {hook_name: _make_sae() for hook_name in hook_names}
    trainer = MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        data_provider=iter(()),
        save_checkpoint_fn=None,
        cfg=_make_trainer_cfg(total_training_samples=4),
        dp_group=mock.MagicMock(),
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
    )

    all_gather_mock = mock.Mock(side_effect=AssertionError("all_gather should not be called"))
    monkeypatch.setattr(dist, "all_gather", all_gather_mock)

    timing = trainer._global_timing_if_needed(
        vllm_step_time_s=1.0,
        transfer_time_s=2.0,
        sae_time_s=3.0,
    )

    assert timing["step_time_s"] == pytest.approx(6.0)
    assert timing["vllm_step_time_s"] == pytest.approx(1.0)
    assert timing["transfer_time_s"] == pytest.approx(2.0)
    assert timing["sae_time_s"] == pytest.approx(3.0)
    all_gather_mock.assert_not_called()


def test_sae_dp_mode_ddp_rejects_compile_sae() -> None:
    with pytest.raises(ValueError, match="compile_sae"):
        LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            sae_dp_mode="ddp",
            compile_sae=True,
        )


def test_sae_dp_mode_ddp_allows_checkpoint_options() -> None:
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="ddp",
        n_checkpoints=1,
        resume_from_checkpoint="/some/path",
        save_final_checkpoint=True,
    )
    assert cfg.sae_dp_mode == "ddp"


def test_runner_resume_constructor_argument_sets_cfg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        device="cpu",
    )
    monkeypatch.setattr(
        "sae_lens.llm_sae_training_runner.ActivationsStore.from_config",
        mock.Mock(return_value=mock.MagicMock()),
    )

    runner = LanguageModelSAETrainingRunner(
        cfg=cfg,
        override_model=mock.MagicMock(),
        resume_from_checkpoint=tmp_path,
    )

    assert runner.cfg.resume_from_checkpoint == str(tmp_path)


def test_checkpoint_path_uses_timestamp_run_id_with_collision_suffix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("sae_lens.config._timestamp_run_id", lambda: "260413_151500")
    (tmp_path / "260413_151500").mkdir()

    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        checkpoint_path=str(tmp_path),
    )

    assert cfg.checkpoint_path == str(tmp_path / "260413_151500_1")


def test_runner_syncs_run_paths_from_rank0(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LanguageModelSAETrainingRunner.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = mock.MagicMock()
    runner.cfg.checkpoint_path = "checkpoints/ck1/wrong_rank_path"
    runner.cfg.output_path = "results/wrong_rank_path"

    def _fake_broadcast_object_list(objects, src):
        assert src == 0
        objects[:] = [
            "checkpoints/ck1/260413_151500",
            "results/results_1.22/saelens_runner_gpu_260413_151500",
        ]

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "broadcast_object_list", _fake_broadcast_object_list)

    runner._sync_run_paths_across_ranks()

    assert runner.cfg.checkpoint_path == "checkpoints/ck1/260413_151500"
    assert (
        runner.cfg.output_path
        == "results/results_1.22/saelens_runner_gpu_260413_151500"
    )


def test_sae_dp_mode_fsdp_accepted_without_dist() -> None:
    assert not dist.is_initialized(), "dist must not be initialized for this test"
    # Config construction succeeds even before dist is initialized;
    # the dist check is deferred to the runner.
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="fsdp",
    )
    assert cfg.sae_dp_mode == "fsdp"


def test_sae_dp_mode_fsdp_allows_compile_sae() -> None:
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="fsdp",
        compile_sae=True,
    )
    assert cfg.sae_dp_mode == "fsdp"
    assert cfg.compile_sae is True


def test_sae_dp_mode_fsdp_allows_checkpoint_options() -> None:
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="fsdp",
        n_checkpoints=1,
        resume_from_checkpoint="/some/path",
        save_final_checkpoint=True,
    )
    assert cfg.sae_dp_mode == "fsdp"
    assert cfg.n_checkpoints == 1
    assert cfg.resume_from_checkpoint == "/some/path"
    assert cfg.save_final_checkpoint is True


def test_sae_dp_mode_fsdp_does_not_reject_save_final_checkpoint() -> None:
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="fsdp",
        save_final_checkpoint=True,
    )
    assert cfg.save_final_checkpoint is True


def test_sae_dp_mode_manual_is_default_and_valid() -> None:
    cfg = LanguageModelSAERunnerConfig(sae=build_topk_sae_training_cfg())
    assert cfg.sae_dp_mode == "manual"


# ---------------------------------------------------------------------------
# compile_sae placement / wrapper call path
# ---------------------------------------------------------------------------


def test_compile_sae_targets_base_sae_training_forward_pass(monkeypatch) -> None:
    sae = _make_sae()
    runner = LanguageModelSAETrainingRunner.__new__(LanguageModelSAETrainingRunner)
    runner._base_sae = sae
    runner.cfg = mock.MagicMock()
    runner.cfg.compile_sae = True
    runner.cfg.device = "cpu"
    runner.cfg.sae_compilation_mode = "default"

    original_forward = sae.training_forward_pass
    compile_calls = []

    def _fake_compile(fn, *, mode, backend):
        compile_calls.append((fn, mode, backend))

        def _compiled(step_input):
            return fn(step_input)

        return _compiled

    monkeypatch.setattr(torch, "compile", _fake_compile)

    runner._compile_sae_if_needed()

    assert len(compile_calls) == 1
    compiled_fn, mode, backend = compile_calls[0]
    assert compiled_fn.__self__ is sae
    assert compiled_fn.__func__ is original_forward.__func__
    assert mode == "default"
    assert backend == "inductor"


def test_invalid_fsdp_compile_sae_validates_dist_before_compile(monkeypatch) -> None:
    assert not dist.is_initialized(), "dist must not be initialized for this test"
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        sae_dp_mode="fsdp",
        compile_sae=True,
    )
    compile_mock = mock.Mock(
        side_effect=AssertionError("torch.compile should not run before FSDP validation")
    )
    monkeypatch.setattr(torch, "compile", compile_mock)
    monkeypatch.setattr(
        "sae_lens.llm_sae_training_runner.ActivationsStore.from_config",
        mock.Mock(return_value=mock.MagicMock()),
    )

    with pytest.raises(ValueError, match="requires an initialized distributed process group"):
        LanguageModelSAETrainingRunner(cfg=cfg, override_model=mock.MagicMock())

    compile_mock.assert_not_called()


def test_resume_limits_producer_helper_loop_to_remaining_steps(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    torch.save({"n_training_samples": 32}, checkpoint_path / TRAINER_STATE_FILENAME)

    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        training_tokens=64,
        context_size=4,
        store_batch_size_prompts=2,
        train_batch_size_tokens=8,
        n_batches_in_buffer=2,
        activations_mixing_fraction=0.0,
        resume_from_checkpoint=str(checkpoint_path),
    )
    runner = object.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = cfg
    runner.vllm_dp_size = 2
    runner.sae_dp_size = 1
    runner.activations_store = mock.MagicMock()
    runner.activations_store.training_context_size = 4
    runner.activations_store._run_producer_phase2_v2.return_value = ({}, {})

    runner._run_producer_helper_loop_v2()

    assert runner.activations_store._run_producer_phase2_v2.call_count == 2
    assert runner.activations_store._run_nccl_p2p_exchange_v2.call_count == 2


def test_producer_helper_loop_accounts_for_mixing_buffer_retained_samples() -> None:
    cfg = LanguageModelSAERunnerConfig(
        sae=build_topk_sae_training_cfg(),
        training_tokens=196608,
        context_size=2048,
        store_batch_size_prompts=4,
        train_batch_size_tokens=2048,
        n_batches_in_buffer=2,
        activations_mixing_fraction=0.5,
    )
    runner = object.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = cfg
    runner.vllm_dp_size = 1
    runner.sae_dp_size = 1
    runner.activations_store = mock.MagicMock()
    runner.activations_store.training_context_size = 2048
    runner.activations_store._run_producer_phase2_v2.return_value = ({}, {})

    runner._run_producer_helper_loop_v2()

    assert runner.activations_store._run_producer_phase2_v2.call_count == 25
    assert runner.activations_store._run_nccl_p2p_exchange_v2.call_count == 25


def test_train_step_enters_wrapped_sae_not_base_training_forward_pass() -> None:
    base_sae = _make_sae()
    base_sae.training_forward_pass = mock.Mock(
        side_effect=AssertionError("base SAE should not be called directly")
    )
    wrapped_sae = _RecordingWrapper(base_sae)
    trainer = SAETrainer(
        cfg=_make_trainer_cfg(),
        sae=wrapped_sae,  # type: ignore[arg-type]
        base_sae=base_sae,
        data_provider=mock.MagicMock(),
    )

    trainer._train_step(wrapped_sae, torch.randn(4, base_sae.cfg.d_in))  # type: ignore[arg-type]

    assert wrapped_sae.calls == 1
    base_sae.training_forward_pass.assert_not_called()


# ---------------------------------------------------------------------------
# forward() dispatch
# ---------------------------------------------------------------------------


def test_forward_with_tensor_returns_tensor() -> None:
    sae = _make_sae()
    x = torch.randn(4, sae.cfg.d_in)
    out = sae(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


def test_forward_with_trainstepinput_returns_trainstepoutput() -> None:
    sae = _make_sae()
    step_input = _make_step_input(sae)
    out = sae(step_input)
    assert isinstance(out, TrainStepOutput)
    assert out.loss.ndim == 0


def test_forward_dispatch_matches_training_forward_pass() -> None:
    sae = _make_sae()
    step_input = _make_step_input(sae)
    out_via_call = sae(step_input)
    out_via_method = sae.training_forward_pass(step_input)
    assert out_via_call.loss.item() == pytest.approx(out_via_method.loss.item())


# ---------------------------------------------------------------------------
# base_sae accessor
# ---------------------------------------------------------------------------


def test_trainer_base_sae_equals_sae_in_manual_mode() -> None:
    sae = _make_sae()
    data_provider = mock.MagicMock()
    data_provider.consume_last_data_timing = None

    trainer = SAETrainer(cfg=_make_trainer_cfg(), sae=sae, data_provider=data_provider)
    assert trainer.base_sae is sae
    assert trainer._base_sae is sae


def test_sae_trainer_metric_writer_falls_back_to_distributed_v2_tp_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sae = _make_sae()
    data_provider = mock.MagicMock()
    data_provider.consume_last_data_timing = None
    trainer = SAETrainer(cfg=_make_trainer_cfg(), sae=sae, data_provider=data_provider)
    assert getattr(sae, "_tp_group", None) is None

    import sae_lens.distributed_v2 as v2_mod

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(v2_mod, "_initialized", True)
    monkeypatch.setattr(v2_mod, "is_consumer", lambda: True)
    monkeypatch.setattr(v2_mod, "get_sae_tp_rank", lambda: 1)

    assert trainer._is_metric_writer_rank() is False


def test_ddp_single_step_matches_full_batch_reference_and_saves_base_keys(
    tmp_path: Path,
) -> None:
    d_in, d_sae, k = 8, 16, 4
    world_size = 2
    sae_ref = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
    state_dict = {name: value.clone() for name, value in sae_ref.state_dict().items()}
    x_per_rank = [torch.randn(4, d_in), torch.randn(4, d_in)]

    ref_sae = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
    ref_sae.load_state_dict(state_dict)
    ref_trainer = SAETrainer(
        cfg=_make_trainer_cfg(
            total_training_samples=sum(x.shape[0] for x in x_per_rank),
            train_batch_size_samples=sum(x.shape[0] for x in x_per_rank),
        ),
        sae=ref_sae,
        data_provider=mock.MagicMock(),
    )
    ref_trainer._train_step(ref_sae, torch.cat(x_per_rank, dim=0))
    expected_params = {
        name: param.detach().clone()
        for name, param in ref_sae.named_parameters()
    }

    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _ddp_worker,
        args=(
            world_size,
            d_in,
            d_sae,
            k,
            state_dict,
            x_per_rank,
            result_list,
            _find_free_port(),
            str(tmp_path),
        ),
        nprocs=world_size,
        join=True,
    )
    results = sorted(result_list, key=lambda item: item[0])
    assert len(results) == world_size

    expected_keys = sorted(state_dict.keys())
    for (
        rank,
        params,
        mode,
        saved_keys,
        saved_mode,
        resumed_params,
        resumed_mode,
        resumed_steps,
        resumed_samples,
        resumed_has_optimizer_state,
    ) in results:
        assert mode == "ddp"
        assert resumed_mode == "ddp"
        assert resumed_steps == 1
        assert resumed_samples == x_per_rank[rank].shape[0]
        assert resumed_has_optimizer_state
        for name, expected in expected_params.items():
            torch.testing.assert_close(
                params[name],
                expected,
                atol=1e-5,
                rtol=1e-4,
                msg=f"rank {rank} DDP param {name} differs from full-batch reference",
            )
            torch.testing.assert_close(
                resumed_params[name],
                expected,
                atol=1e-5,
                rtol=1e-4,
                msg=f"rank {rank} resumed DDP param {name} differs from saved weights",
            )
        if saved_keys is not None:
            assert saved_keys == expected_keys
            assert all(not key.startswith("module.") for key in saved_keys)
            assert saved_mode == "ddp"


def test_ddp_supports_tp_sharded_sae_and_saves_full_weight_shapes(
    tmp_path: Path,
) -> None:
    d_in, d_sae, k = 8, 16, 4
    world_size = 4
    sae_ref = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
    state_dict = {name: value.clone() for name, value in sae_ref.state_dict().items()}
    x_per_dp_rank = [torch.randn(4, d_in), torch.randn(4, d_in)]

    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _tp2_ddp2_worker,
        args=(
            world_size,
            d_in,
            d_sae,
            k,
            state_dict,
            x_per_dp_rank,
            result_list,
            _find_free_port(),
            str(tmp_path),
        ),
        nprocs=world_size,
        join=True,
    )
    results = sorted(result_list, key=lambda item: item[0])
    assert len(results) == world_size
    params_by_rank = {rank: params for rank, params, _, _ in results}

    for name in ("W_enc", "W_dec", "b_enc"):
        torch.testing.assert_close(params_by_rank[0][name], params_by_rank[2][name])
        torch.testing.assert_close(params_by_rank[1][name], params_by_rank[3][name])
    for rank in range(1, world_size):
        torch.testing.assert_close(params_by_rank[0]["b_dec"], params_by_rank[rank]["b_dec"])

    expected_keys = sorted(state_dict.keys())
    expected_shapes = {name: tuple(value.shape) for name, value in state_dict.items()}
    for _, _, saved_keys, saved_shapes in results:
        if saved_keys is not None:
            assert saved_keys == expected_keys
            assert all(not key.startswith("module.") for key in saved_keys)
            assert saved_shapes == expected_shapes


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="FSDP resume integration test requires at least 2 CUDA devices",
)
def test_fsdp_saves_and_loads_sharded_optimizer_state(tmp_path: Path) -> None:
    d_in, d_sae, k = 8, 16, 4
    world_size = 2
    sae_ref = _make_sae(d_in=d_in, d_sae=d_sae, k=k)
    state_dict = {name: value.clone() for name, value in sae_ref.state_dict().items()}
    x_per_rank = [torch.randn(4, d_in), torch.randn(4, d_in)]

    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _fsdp_resume_worker,
        args=(
            world_size,
            d_in,
            d_sae,
            k,
            state_dict,
            x_per_rank,
            result_list,
            _find_free_port(),
            str(tmp_path),
        ),
        nprocs=world_size,
        join=True,
    )
    results = sorted(result_list, key=lambda item: item[0])
    assert len(results) == world_size

    for (
        rank,
        shard_exists,
        resumed_mode,
        resumed_steps,
        resumed_samples,
        optimizer_loaded,
        full_state_matches_saved,
        saved_mode,
        saved_format,
        saved_dp_size,
    ) in results:
        assert shard_exists, f"rank {rank} did not save an FSDP optimizer shard"
        assert resumed_mode == "fsdp"
        assert resumed_steps == 1
        assert resumed_samples == x_per_rank[rank].shape[0]
        assert optimizer_loaded
        if rank == 0:
            assert full_state_matches_saved
            assert saved_mode == "fsdp"
            assert saved_format == "fsdp_sharded"
            assert saved_dp_size == world_size


# ---------------------------------------------------------------------------
# clip_grad_norm_ with dp_group
# ---------------------------------------------------------------------------


def test_clip_grad_norm_no_tp_no_dp() -> None:
    sae = _make_sae()
    for p in sae.parameters():
        p.grad = torch.ones_like(p) * 10.0
    norm = sae.clip_grad_norm_(1.0)
    assert norm.item() > 1.0
    clipped_norm = sum(p.grad.norm() ** 2 for p in sae.parameters() if p.grad is not None) ** 0.5
    assert clipped_norm <= 1.0 + 1e-4


def test_clip_grad_norm_dp_group_none_unchanged() -> None:
    sae = _make_sae()
    for p in sae.parameters():
        p.grad = torch.ones_like(p) * 5.0
    norm_default = sae.clip_grad_norm_(1.0)

    sae2 = _make_sae()
    for p in sae2.parameters():
        p.grad = torch.ones_like(p) * 5.0
    norm_explicit_none = sae2.clip_grad_norm_(1.0, dp_group=None)

    assert norm_default == pytest.approx(norm_explicit_none.item())


# ---------------------------------------------------------------------------
# Weight export format: base_sae.process_state_dict_for_saving produces
# the same shape state dict regardless of TP size (no-TP baseline).
# ---------------------------------------------------------------------------


def test_process_state_dict_for_saving_no_tp_identity() -> None:
    sae = _make_sae()
    state_dict = sae.state_dict()
    original_shapes = {k: v.shape for k, v in state_dict.items()}
    sae.process_state_dict_for_saving(state_dict)
    for k, shape in original_shapes.items():
        assert state_dict[k].shape == shape


def test_save_load_roundtrip_preserves_weights(tmp_path: pytest.TempPathFactory) -> None:
    sae = _make_sae()
    sae.save_model(str(tmp_path))

    sae2 = _make_sae()
    sae2.load_weights_from_checkpoint(tmp_path)

    for (name, p1), (_, p2) in zip(sae.named_parameters(), sae2.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter {name} differs after roundtrip"
