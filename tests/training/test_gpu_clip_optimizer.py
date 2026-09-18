"""Analytic clipping boundaries and a guard against scalar host reads."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer.optimizer import FP32Optimizer
from torch.utils._python_dispatch import TorchDispatchMode

from sae_lens.training.gpu_clip_optimizer import (
    GPUClipFP32Optimizer,
    clip_grads_on_device,
)
from sae_lens.training.megatron_optimizer import build_runtime_optimizer
from tests.saes.test_megatron_sae_boundaries import sae  # noqa: F401


class RejectScalarHostRead(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("GPU clipping/update must not read a tensor scalar on CPU")
        return func(*args, **(kwargs or {}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA optimizer")
@pytest.mark.parametrize("scale,expected_norm,coefficient", [
    (0.0, 0.0, 1.0),
    (0.1, 0.5, 1.0),
    (1.0, 5.0, 1.0 / (5.0 + 1e-6)),
])
def test_gpu_clip_boundaries_and_native_step_without_scalar_reads(
    sae, scale, expected_norm, coefficient  # noqa: F811
):
    sae.to("cuda")
    runtime = SimpleNamespace(require_local=lambda: SimpleNamespace(tp_group=sae._tp_group))
    optimizer = build_runtime_optimizer(sae, runtime, adam_kwargs={"lr": 3e-4})
    assert type(optimizer) is GPUClipFP32Optimizer
    assert GPUClipFP32Optimizer.step is FP32Optimizer.step
    assert GPUClipFP32Optimizer.prepare_grads is FP32Optimizer.prepare_grads
    assert GPUClipFP32Optimizer.step_with_ready_grads is FP32Optimizer.step_with_ready_grads
    for parameter in sae.parameters():
        parameter.grad = torch.zeros_like(parameter)
    sae.b_dec.grad[:2] = torch.tensor([3.0 * scale, 4.0 * scale], device="cuda")
    with RejectScalarHostRead():
        success, norm, zeros = optimizer.step()
    assert success and zeros is None
    assert norm.is_cuda and norm.ndim == 0
    torch.testing.assert_close(norm, torch.tensor(expected_norm, device="cuda"))
    expected = torch.zeros_like(sae.b_dec)
    expected[:2] = torch.tensor(
        [3.0 * scale * coefficient, 4.0 * scale * coefficient], device="cuda"
    )
    torch.testing.assert_close(sae.b_dec.grad, expected)
    for name, parameter in sae.named_parameters():
        if name != "b_dec":
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("value", [0.0, 0.1, 3.0, float("nan"), float("inf"), 1e20])
def test_multi_tensor_clip_matches_analytic_reference_on_views(value):
    storage = torch.full((40,), float("nan"), device="cuda")
    # Non-contiguous and disjoint valid views; unused storage must stay NaN.
    grads = [storage[:12:2], storage[20:32].reshape(3, 4).t()]
    for grad in grads:
        grad.fill_(value)
    expected = [g.clone() for g in grads]
    reference = sum(g.square().sum() for g in expected).sqrt()
    coefficient = (1.0 / (reference + 1e-6)).clamp(max=1.0)
    for grad in expected:
        grad.mul_(coefficient)
    with RejectScalarHostRead():
        norm = clip_grads_on_device(grads, grads, device=storage.device, group=None, max_norm=1.0)
    torch.testing.assert_close(norm, reference, equal_nan=True)
    torch.testing.assert_close(grads, expected, equal_nan=True)
    assert storage[12:20].isnan().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_multi_tensor_clip_empty_local_shard_returns_device_zero():
    with RejectScalarHostRead():
        norm = clip_grads_on_device([], [], device="cuda", group=None, max_norm=1.0)
    torch.testing.assert_close(norm, torch.zeros((), device="cuda"))
