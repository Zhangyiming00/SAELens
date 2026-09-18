"""Shared-scaler bookkeeping when token normalization uses the unscale pass."""

import pytest
import torch

from sae_lens.training.runtime_amp import unscale_with_token_mean


@pytest.mark.parametrize("tokens", [3, 7, 2048])
@pytest.mark.parametrize("scale", [64.0, 96.0])
def test_token_mean_keeps_shared_scale_and_independent_hook_overflow(tokens, scale):
    torch.manual_seed(789)
    models = [[torch.nn.Parameter(torch.ones(32)) for _ in range(2)] for _ in range(2)]
    optimizers = [[torch.optim.Adam([p], lr=3e-4) for p in ps] for ps in models]
    scalers = [torch.amp.GradScaler("cpu", init_scale=scale, growth_interval=2) for _ in models]
    for window in range(3):
        grads = [torch.randn(32), torch.randn(32)]
        if window == 1:
            grads[0][0] = float("inf")
        for candidate, (params, opts, scaler) in enumerate(zip(models, optimizers, scalers)):
            before = scaler.get_scale()
            for param, opt, grad in zip(params, opts, grads):
                opt.zero_grad(set_to_none=True)
                scaler.scale(param.sum()).backward()
                param.grad = grad.clone()
                if candidate:
                    unscale_with_token_mean(scaler, opt, tokens)
                    with pytest.raises(RuntimeError, match="already been unscaled"):
                        unscale_with_token_mean(scaler, opt, tokens)
                else:
                    param.grad.div_(tokens)
                    scaler.unscale_(opt)
                assert scaler.get_scale() == before
                scaler.step(opt)
            scaler.update()
        for left, right in zip(models[0], models[1]):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
        for left, right in zip(optimizers[0], optimizers[1]):
            torch.testing.assert_close(left.state_dict(), right.state_dict(), atol=2e-6, rtol=2e-5)
        assert scalers[0].state_dict() == scalers[1].state_dict()
    assert optimizers[1][0].state[models[1][0]]["step"] == 2
    assert optimizers[1][1].state[models[1][1]]["step"] == 3
