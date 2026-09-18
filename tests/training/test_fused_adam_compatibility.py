"""Extension installation must preserve coupled Adam math and checkpoint clocks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from sae_lens.training.distributed_optimizer import GPUClipDistributedOptimizer


def test_torch_checkpoint_to_fused_adam_preserves_moments_and_clock():
    moments = {"exp_avg": torch.arange(5.0), "exp_avg_sq": torch.arange(5.0).square()}
    saved = dict(
        state={0: {**moments, "step": torch.tensor(7.0)}},
        param_groups=[dict(params=[0], lr=0.002)],
    )
    target = SimpleNamespace(
        optimizer=SimpleNamespace(
            adam_w_mode=0, param_groups=[dict(params=[], bias_correction=True)]
        )
    )
    converted = GPUClipDistributedOptimizer._adapt_local_adam_state(target, saved)
    assert converted["param_groups"][0]["step"] == 7
    assert converted["param_groups"][0]["bias_correction"] is True
    assert converted["state"][0] == moments
    assert "step" in saved["state"][0]  # Conversion must not mutate the source.


def test_fused_adam_rejects_distinct_parameter_clocks():
    target = SimpleNamespace(
        optimizer=SimpleNamespace(adam_w_mode=0, param_groups=[dict(params=[])])
    )
    saved = dict(
        state={0: {"step": 2}, 1: {"step": 3}}, param_groups=[dict(params=[0, 1])]
    )
    with pytest.raises(ValueError, match="common step"):
        GPUClipDistributedOptimizer._adapt_local_adam_state(target, saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA extensions")
@pytest.mark.parametrize("backend", ["transformer_engine", "apex"])
@pytest.mark.parametrize("decay", [0.0, 0.02])
def test_fused_adam_matches_torch_and_resumes(backend, decay):
    module = pytest.importorskip(
        backend
        + (".pytorch.optimizers" if backend == "transformer_engine" else ".optimizers")
    )
    initial = torch.linspace(-1, 1, 257, device="cuda")
    reference, actual = (torch.nn.Parameter(initial.clone()) for _ in range(2))
    kwargs = dict(lr=3e-4, betas=(0.85, 0.97), eps=1e-7, weight_decay=decay)
    torch_adam = torch.optim.Adam([reference], fused=True, **kwargs)
    fused_adam = module.FusedAdam([actual], adam_w_mode=False, **kwargs)
    for iteration in range(12):
        grad = torch.sin(initial * (iteration + 1)) * (0.02 if iteration % 2 else 10)
        reference.grad, actual.grad = grad.clone(), grad.clone()
        torch_adam.step()
        fused_adam.step()
        torch.testing.assert_close(actual, reference, atol=3e-7, rtol=3e-6)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                fused_adam.state[actual][name],
                torch_adam.state[reference][name],
                atol=2e-6,
                rtol=2e-5,
            )
        assert int(fused_adam.param_groups[0]["step"]) == iteration + 1
        if iteration == 5:
            # Load the real torch format into the extension and continue updates.
            target = SimpleNamespace(optimizer=fused_adam)
            saved = GPUClipDistributedOptimizer._adapt_local_adam_state(
                target, deepcopy(torch_adam.state_dict())
            )
            fused_adam.load_state_dict(saved)
            assert fused_adam.adam_w_mode == 0
