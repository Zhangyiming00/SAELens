"""Tests for sae_dp_mode config validation, forward() dispatch, clip_grad_norm_
with dp_group, base_sae accessor, and FSDP save/load weight format."""

import unittest.mock as mock

import pytest
import torch
import torch.distributed as dist

from sae_lens.config import LanguageModelSAERunnerConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKTrainingSAE
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


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_sae_dp_mode_ddp_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="ddp"):
        LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            sae_dp_mode="ddp",
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


def test_sae_dp_mode_fsdp_rejects_compile_sae() -> None:
    with pytest.raises(ValueError, match="compile_sae"):
        LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            sae_dp_mode="fsdp",
            compile_sae=True,
        )


def test_sae_dp_mode_fsdp_rejects_n_checkpoints() -> None:
    with pytest.raises(ValueError, match="n_checkpoints"):
        LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            sae_dp_mode="fsdp",
            n_checkpoints=1,
        )


def test_sae_dp_mode_fsdp_rejects_resume_from_checkpoint() -> None:
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            sae_dp_mode="fsdp",
            resume_from_checkpoint="/some/path",
        )


def test_sae_dp_mode_fsdp_rejects_save_final_checkpoint() -> None:
    with pytest.raises(ValueError, match="save_final_checkpoint"):
        LanguageModelSAERunnerConfig(
            sae=build_topk_sae_training_cfg(),
            sae_dp_mode="fsdp",
            save_final_checkpoint=True,
        )


def test_sae_dp_mode_manual_is_default_and_valid() -> None:
    cfg = LanguageModelSAERunnerConfig(sae=build_topk_sae_training_cfg())
    assert cfg.sae_dp_mode == "manual"


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
    cfg = mock.MagicMock(spec=SAETrainerConfig)
    cfg.device = "cpu"
    cfg.n_checkpoints = 0
    cfg.total_training_samples = 100
    cfg.train_batch_size_samples = 4
    cfg.output_path = None
    cfg.save_mse_every_n_steps = 0
    cfg.save_timing_every_n_steps = 0
    cfg.synchronize_timing = False
    cfg.lr = 1e-3
    cfg.lr_end = 1e-4
    cfg.lr_scheduler_name = "constant"
    cfg.lr_warm_up_steps = 0
    cfg.lr_decay_steps = 0
    cfg.n_restart_cycles = 1
    cfg.adam_beta1 = 0.9
    cfg.adam_beta2 = 0.999
    cfg.dead_feature_window = 100
    cfg.feature_sampling_window = 100
    cfg.autocast = False
    cfg.checkpoint_path = None
    cfg.save_final_checkpoint = False
    cfg.logger = mock.MagicMock()
    cfg.logger.log_to_wandb = False

    data_provider = mock.MagicMock()
    data_provider.consume_last_data_timing = None

    trainer = SAETrainer(cfg=cfg, sae=sae, data_provider=data_provider)
    assert trainer.base_sae is sae
    assert trainer._base_sae is sae


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
