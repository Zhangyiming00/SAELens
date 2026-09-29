from dataclasses import replace

import pytest
import torch
from helpers import Input, dense_reference, make_harness, reference_forward

from sae_lens.sharded_sparse import sparse_decode
from sae_lens.sharded_topk import feature_counts_from_output, score_keys, sharded_topk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 19), (2, 3, 19), (0, 19)])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("k", [1, 5, 19])
def test_local_topk(dtype, shape, sparse, k):
    torch.manual_seed(7)
    x = torch.randint(-3, 5, shape).to(dtype).requires_grad_()
    y = sharded_topk(x, k, sparse=sparse)
    expected = dense_reference(x, k)
    actual = y.to_dense() if y.is_sparse else y
    torch.testing.assert_close(actual, expected)
    g1 = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(g1, g2)


def test_score_order_and_ties():
    x = torch.tensor(
        [
            [
                -float("inf"),
                -3.0,
                -1.0e-30,
                -0.0,
                0.0,
                1.0e-30,
                3.0,
                float("inf"),
                float("nan"),
            ]
        ]
    )
    key = score_keys(x)
    assert torch.equal(
        key.argsort(descending=True), torch.tensor([[8, 7, 6, 5, 3, 4, 2, 1, 0]])
    )
    # Adjacent floating-point values must not be changed by an index epsilon.
    x = torch.tensor(
        [[1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()]]
    )
    assert sharded_topk(x, 1, sparse=False).nonzero().tolist() == [[0, 1]]


@pytest.mark.parametrize("dead_ids", [[], [0], [0, 3], [1, 4, 9, 10, 16], list(range(19))])
@pytest.mark.parametrize("n", [1, 5, 19])
def test_aux_candidates_preserve_global_ids_and_unique_padding(dead_ids, n):
    from sae_lens.sharded_topk import _local_candidates

    scores = torch.tensor([[0., -0., 3., -2., float('inf'), -1., 4., 5., 1.,
                            float('nan'), -float('inf'), 2., 3., 4., 5., -3., 0., 2., 1.]])
    mask = torch.zeros(19, dtype=torch.bool)
    mask[dead_ids] = True
    keys, ids = _local_candidates(scores, n, 37, mask, 'torch')
    reference = score_keys(scores, 37).masked_fill(~mask, -(1 << 63))
    torch.testing.assert_close(keys.sort(dim=1).values, reference.topk(n, dim=1).values.sort(dim=1).values)
    torch.testing.assert_close(keys, reference.gather(1, ids))
    # Dense scatter must never overwrite a real winner with padded zero.
    assert ids.unique().numel() == n


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_sparse_decoder_forward_backward(dtype, strided):
    torch.manual_seed(23)
    idx = torch.tensor([[0, 0, 1, 2, 2], [1, 6, 3, 2, 5]])
    v = torch.tensor([1.0, -2.0, 0.0, 3.0, -0.5], dtype=dtype, requires_grad=True)
    w = (
        torch.randn(8, 7, dtype=dtype).T if strided else torch.randn(7, 8, dtype=dtype)
    ).requires_grad_()
    a = torch.sparse_coo_tensor(idx, v, (3, 8)).coalesce()
    y = sparse_decode(a, w)
    ref = a.to_dense() @ w.T
    g = torch.randn_like(y)
    torch.testing.assert_close(y, ref)
    grads = torch.autograd.grad((y * g).sum(), (v, w), retain_graph=True)
    refs = torch.autograd.grad((ref * g).sum(), (v, w))
    for actual, expected in zip(grads, refs):
        torch.testing.assert_close(actual, expected)
        assert actual.layout == torch.strided


def test_decoder_gradcheck():
    idx = torch.tensor([[0, 0, 1], [1, 4, 2]])
    v = torch.randn(3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 6, dtype=torch.float64, requires_grad=True)

    def op(v, w):
        a = torch.sparse_coo_tensor(idx, v, (2, 6)).coalesce()
        return sparse_decode(a, w)

    assert torch.autograd.gradcheck(op, (v, w))


@pytest.mark.parametrize("backend", ["sharded_dense", "sharded_sparse"])
@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("rows", [0, 5])
@pytest.mark.parametrize("aux", [False, True])
def test_real_model_methods_cpu(backend, rescale, rows, aux):
    torch.manual_seed(58)
    weights = [
        torch.randn(16, 7) * 0.2,
        torch.randn(16) * 0.2,
        torch.randn(7, 16) * 0.2,
        torch.randn(7) * 0.1,
    ]
    reference_weights = [x.clone().requires_grad_() for x in weights]
    model = make_harness(weights, None, backend, rescale)
    x = torch.randn(rows, 7, requires_grad=True)
    mask = torch.arange(16) % 2 == 0 if aux else None
    result = model.training_forward_pass(Input(x, mask))
    ref_out, ref_acts, ref_loss = reference_forward(
        reference_weights, x, 2, mask, rescale
    )
    torch.testing.assert_close(result.sae_out, ref_out, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.loss, ref_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        feature_counts_from_output(result), (ref_acts != 0).float().sum(0)
    )
    result.loss.backward(retain_graph=True)
    actual = [
        model.encoder.weight.grad,
        model.encoder.bias.grad,
        model.decoder.weight.grad,
        model.b_dec.grad,
    ]
    ref_grads = torch.autograd.grad(ref_loss, reference_weights)
    for grad, expected in zip(actual, ref_grads):
        assert grad is not None and grad.layout == torch.strided
        torch.testing.assert_close(grad, expected, atol=2e-5, rtol=3e-5)
    # New summary is handled by the trainer's existing top-level tensor detach.
    detached = replace(
        result, feature_firing_counts=result.feature_firing_counts.detach()
    )
    assert not detached.feature_firing_counts.requires_grad


def test_wrong_dtype_and_width_fail_explicitly():
    with pytest.raises(TypeError):
        sharded_topk(torch.ones(3, 8, dtype=torch.float64), 2)
    with pytest.raises(ValueError):
        sparse_decode(torch.ones(3, 9).to_sparse(), torch.ones(7, 8))


def test_empty_sparse_decoder():
    a = torch.sparse_coo_tensor(
        torch.empty(2, 0, dtype=torch.long), torch.empty(0, requires_grad=True), (0, 8)
    ).coalesce()
    w = torch.randn(7, 8, requires_grad=True)
    sparse_decode(a, w).sum().backward()
    assert w.grad is not None
    assert torch.equal(w.grad, torch.zeros_like(w))
