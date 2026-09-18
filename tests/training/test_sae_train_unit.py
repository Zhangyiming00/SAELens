"""Per-hook optimizer ownership and compatibility with existing checkpoint readers."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sae_lens.sae_runtime import SAERuntime
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.training.megatron_ddp import supports_early_grad_sync
from sae_lens.training.megatron_optimizer import (
    MEGATRON_GROUP_METADATA,
    build_runtime_optimizer,
)
from sae_lens.training.optim import get_lr_scheduler
from sae_lens.training.optimizer_checkpoint import optimizer_state_for_loading
from sae_lens.training.sae_train_unit import SAETrainUnit, UnitOptimizers
from tests.saes.test_megatron_sae_boundaries import sae  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron optimizer requires CUDA")
def test_native_optimizer_raw_state_round_trip_and_legacy_groups(sae):  # noqa: F811
    from megatron.core.optimizer.optimizer import FP32Optimizer

    sae.to("cuda")
    runtime = SimpleNamespace(require_local=lambda: SimpleNamespace(tp_group=sae._tp_group))
    optimizer = build_runtime_optimizer(sae, runtime, adam_kwargs={"lr": 3e-4})
    assert isinstance(optimizer, FP32Optimizer)
    assert type(optimizer).step is FP32Optimizer.step
    scheduler = get_lr_scheduler("cosineannealing", optimizer, 8, 3e-4, 0, 0, 3e-5, 1)
    assert scheduler.optimizer is optimizer.optimizer
    for p in sae.parameters():
        p.grad = torch.ones_like(p)
    success, norm, _ = optimizer.step()
    assert torch.is_tensor(norm) and norm.is_cuda and norm.ndim == 0
    assert success and norm > 1
    scheduler.step()
    saved = copy.deepcopy(optimizer.state_dict())
    optimizer.state.clear()
    optimizer.load_state_dict(copy.deepcopy(saved))
    torch.testing.assert_close(optimizer.state_dict(), saved, atol=0, rtol=0)
    legacy = copy.deepcopy(saved)
    for group in legacy["param_groups"]:
        for key in MEGATRON_GROUP_METADATA:
            del group[key]
    optimizer.state.clear()
    optimizer.load_state_dict(optimizer_state_for_loading(optimizer, legacy))
    torch.testing.assert_close(optimizer.state_dict(), saved, atol=0, rtol=0)
    assert scheduler.optimizer.param_groups is optimizer.param_groups


def test_unit_optimizer_state_round_trip_and_independent_updates(sae):  # noqa: F811
    runtime = object.__new__(SAERuntime)
    runtime._closed = False
    runtime.local = SimpleNamespace(tp_group=sae._tp_group, dp_group=sae._tp_group)
    second = MegatronTopKSAE(sae.cfg, tp_group=sae._tp_group)
    models = {"h1": sae, "h2": second}
    units = {
        h: SAETrainUnit(
            h, m, m, torch.optim.Adam(m.parameters(), lr=(i + 1) * 1e-3), runtime
        )
        for i, (h, m) in enumerate(models.items())
    }
    optimizer = UnitOptimizers(units)
    batch = torch.linspace(-1, 1, 176).reshape(11, 16)
    for unit in units.values():
        unit.forward(batch).square().sum().backward()
        unit.finish_grad_sync()
        unit.clip_grad_norm()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    optimizer.state.clear()
    assert all(not u.optimizer.state for u in units.values())
    optimizer.load_state_dict(saved)
    assert all(len(u.optimizer.state) == 4 for u in units.values())
    for live, expected in zip(
        optimizer.state_dict()["state"].values(), saved["state"].values(), strict=True
    ):
        for name, value in live.items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert optimizer.param_groups[1] is units["h2"].optimizer.param_groups[0]
    assert optimizer.param_groups[1]["lr"] == 2e-3
    before = copy.deepcopy(second.state_dict())
    before_state = copy.deepcopy(units["h2"].optimizer.state_dict())
    optimizer.zero_grad()
    sae(batch).square().sum().backward()
    units["h1"].clip_grad_norm()
    units["h1"].step()
    for name, value in second.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for key, values in before_state["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(
                units["h2"].optimizer.state_dict()["state"][key][name],
                value,
                rtol=0,
                atol=0,
            )
    with pytest.raises(ValueError, match="must not share parameters"):
        UnitOptimizers({"h1": units["h1"], "h2": units["h1"]})


def test_pending_reduction_owns_gradient_buffer(sae):  # noqa: F811
    """API state transitions, separate from the real NCCL acceptance below."""
    runtime = object.__new__(SAERuntime)
    runtime._closed = False
    runtime.local = SimpleNamespace(tp_group=sae._tp_group, dp_group=sae._tp_group)
    for p in sae.parameters():
        p.main_grad = torch.ones_like(p)
    ddp = SimpleNamespace(
        _sae_megatron_ddp=True, module=sae, dp_group=sae._tp_group,
        start_grad_sync=Mock(), finish_grad_sync=Mock(), zero_grad_buffer=Mock(),
    )
    unit = SAETrainUnit("h", sae, ddp, torch.optim.Adam(sae.parameters()), runtime)
    with pytest.raises(RuntimeError, match="Start hook"):
        unit.finish_grad_sync()
    unit.start_grad_sync()
    unit.start_grad_sync()
    ddp.start_grad_sync.assert_called_once()
    assert all(p.grad is None for p in sae.parameters())
    with pytest.raises(RuntimeError, match="Cannot clear"):
        unit.zero_grad()

    with pytest.raises(RuntimeError, match="Cannot accumulate"):
        unit.backward(None, None)
    with pytest.raises(RuntimeError, match="before clipping"):
        unit.clip_grad_norm()
    with pytest.raises(RuntimeError, match="before optimizer"):
        unit.step()
    ddp.zero_grad_buffer.assert_not_called()
    assert all(p.main_grad.eq(1).all() for p in sae.parameters())
    unit.finish_window(4)
    ddp.start_grad_sync.assert_called_once()
    ddp.finish_grad_sync.assert_called_once()
    assert all(p.grad is p.main_grad and p.grad.eq(0.25).all() for p in sae.parameters())
    unit.finish_grad_sync()
    ddp.finish_grad_sync.assert_called_once()
    unit.zero_grad()
    unit.start_grad_sync()
    assert ddp.start_grad_sync.call_count == 2
    unit.finish_grad_sync()
    unit.zero_grad()
    ddp.start_grad_sync.side_effect = RuntimeError("second bucket failed")
    with pytest.raises(RuntimeError, match="second bucket"):
        unit.start_grad_sync()
    with pytest.raises(RuntimeError, match="Cannot clear"):
        unit.zero_grad()


def test_native_final_backward_owns_sync_even_on_failure(sae):  # noqa: F811
    runtime = object.__new__(SAERuntime)
    runtime._closed = False
    runtime.local = SimpleNamespace(tp_group=sae._tp_group, dp_group=sae._tp_group)
    for parameter in sae.parameters():
        parameter.main_grad = torch.ones_like(parameter)
    ddp = SimpleNamespace(
        _sae_megatron_ddp=True, module=sae, dp_group=sae._tp_group,
        start_grad_sync=Mock(), finish_grad_sync=Mock(), zero_grad_buffer=Mock(),
    )
    unit = SAETrainUnit("h", sae, ddp, torch.optim.Adam(sae.parameters()), runtime)
    scaled_loss = Mock()
    scaler = Mock(scale=Mock(return_value=scaled_loss))
    unit.backward(None, scaler, sync_gradients=True)
    scaled_loss.backward.assert_called_once()
    with pytest.raises(RuntimeError, match="Cannot accumulate"):
        unit.backward(None, scaler)
    with pytest.raises(RuntimeError, match="Cannot clear"):
        unit.zero_grad()
    unit.start_grad_sync()
    ddp.start_grad_sync.assert_not_called()
    unit.finish_window(4)
    ddp.start_grad_sync.assert_not_called()
    ddp.finish_grad_sync.assert_called_once()
    assert all(p.grad is p.main_grad and p.grad.eq(0.25).all() for p in sae.parameters())
    unit.zero_grad()
    scaled_loss.backward.side_effect = RuntimeError("backward failed after one bucket")
    with pytest.raises(RuntimeError, match="after one bucket"):
        unit.backward(None, scaler, sync_gradients=True)
    with pytest.raises(RuntimeError, match="Cannot clear"):
        unit.zero_grad()


@pytest.mark.parametrize("tp,dp,implicit,nccl,expected", [
    (1, 3, "0", (2, 25, 0), True),
    (2, 1, "0", (2, 25, 0), True),
    (2, 2, "0", (2, 27, 5), False),
    (2, 2, "1", (2, 25, 0), False),
    (2, 2, "1", (2, 27, 5), True),
])
def test_cross_group_early_reduction_requires_nccl_ordering(monkeypatch, tp, dp, implicit, nccl, expected):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", implicit)
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: nccl)
    context = SimpleNamespace(tp_group=Mock(size=lambda: tp), dp_group=Mock(size=lambda: dp))
    runtime = SimpleNamespace(require_local=lambda: context)
    assert supports_early_grad_sync(SimpleNamespace(_sae_megatron_ddp=True), runtime) is expected
    assert not supports_early_grad_sync(SimpleNamespace(), runtime)
