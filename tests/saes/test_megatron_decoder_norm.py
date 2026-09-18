"""Sharing norm vectors must retain all branches and isolate forward graphs."""

import copy
from types import MethodType

import pytest
import torch
import torch.distributed as dist

from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.sae import TrainingSAE, TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig


@pytest.fixture
def tp1():
    pytest.importorskip("megatron.core")
    import torch.testing._internal.distributed.fake_pg  # noqa: F401

    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rescale", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_shared_norm_retains_gradients_and_multiple_forward_graphs(tp1, rescale, sparse):
    torch.manual_seed(2026)
    cfg = TopKTrainingSAEConfig(
        d_in=16, d_sae=48, k=4, device="cpu", normalize_activations="none",
        rescale_acts_by_decoder_norm=rescale, use_sparse_activations=sparse,
    )
    reference = MegatronTopKSAE(cfg, tp_group=tp1)
    shared = MegatronTopKSAE(cfg, tp_group=tp1)
    shared.load_state_dict(copy.deepcopy(reference.state_dict()))
    # Bypass only the new scope, preserving the previous independent norms.
    reference.training_forward_pass = MethodType(
        TrainingSAE.training_forward_pass, reference
    )
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in (reference, shared)]
    for step in range(3):
        torch.manual_seed(44 + step)
        batches = [torch.randn(n, 16) for n in (5, 7)]
        all_inputs, all_outputs = [], []
        for model, optimizer in zip((reference, shared), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            inputs = [x.clone().requires_grad_() for x in batches]
            outputs = [model(TrainStepInput(
                x, {}, torch.ones(48, dtype=torch.bool), step, False,
            )) for x in inputs]
            # hidden_pre has a real norm derivative that cannot cancel the
            # decode normalization; both graphs remain live before backward.
            loss = sum(o.loss + 0.2 * o.hidden_pre.square().mean() for o in outputs)
            loss.backward()
            assert getattr(model, "_decoder_norm_scope", None) is None
            all_inputs.append(inputs)
            all_outputs.append(outputs)
        for a, b in zip(all_outputs[0], all_outputs[1], strict=True):
            # Reassociating the norm gradient can round differently after
            # Adam updates; the initial forward is still bitwise identical.
            tolerance = dict(atol=0, rtol=0) if step == 0 else dict(atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a.sae_out, b.sae_out, **tolerance)
            torch.testing.assert_close(a.loss, b.loss, **tolerance)
        for a, b in zip(all_inputs[0], all_inputs[1], strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
        for a, b in zip(reference.parameters(), shared.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(reference.state_dict(), shared.state_dict(),
                                   atol=2e-6, rtol=2e-5)


def test_norm_scope_does_not_survive_forward_failure(tp1):
    model = MegatronTopKSAE(
        TopKTrainingSAEConfig(d_in=16, d_sae=48, k=4, device="cpu"), tp_group=tp1
    )
    def fail(_module, _args, _output):
        raise RuntimeError("probe forward failure")
    handle = model.hook_sae_acts_post.register_forward_hook(fail)
    step_input = TrainStepInput(torch.randn(5, 16), {}, None, 0, False)
    try:
        with pytest.raises(RuntimeError, match="probe forward failure"):
            model(step_input)
        assert model._decoder_norm_scope is None
    finally:
        handle.remove()
    with torch.no_grad():
        model.decoder.weight.mul_(1.1)
    model(step_input).loss.backward()
    assert torch.isfinite(model.decoder.weight.grad).all()
    assert model._decoder_norm_scope is None
