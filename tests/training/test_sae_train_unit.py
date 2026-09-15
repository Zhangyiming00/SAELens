"""Per-hook optimizer ownership and compatibility with existing checkpoint readers."""

import copy
from types import SimpleNamespace

import pytest
import torch

from sae_lens.sae_runtime import SAERuntime
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.training.sae_train_unit import SAETrainUnit, UnitOptimizers
from tests.saes.test_megatron_sae_boundaries import sae  # noqa: F401


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
