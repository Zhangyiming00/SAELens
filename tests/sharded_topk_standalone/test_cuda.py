"""These tests are SKIPPED on CPU; they are not evidence of GPU validation."""

import pytest
import torch

from sae_lens.sharded_sparse import sparse_decode
from sae_lens.sharded_topk import score_keys, sharded_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU unavailable"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_triton_decoder_and_keys(dtype, strided):
    pytest.importorskip("triton")
    from sae_lens.sharded_triton import make_score_keys

    torch.manual_seed(44)
    score = torch.randn(11, 193, device="cuda", dtype=dtype)
    score[0, :5] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")], device="cuda"
    )
    torch.testing.assert_close(make_score_keys(score, 1234), score_keys(score, 1234))
    # Odd sizes, stride variants, zero padding and negative selected AuxK values.
    scores = torch.randn(11, 193, device="cuda", dtype=dtype, requires_grad=True)
    w = (
        torch.randn(193, 97, device="cuda", dtype=dtype).T
        if strided
        else torch.randn(97, 193, device="cuda", dtype=dtype)
    ).requires_grad_()
    a = sharded_topk(scores, 17, relu=False, sparse=True)
    y = sparse_decode(a, w, backend="triton")
    reference = sparse_decode(a, w, backend="torch")
    tol = 3e-4 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(y, reference, atol=tol, rtol=tol)
    g = torch.randn_like(y)
    actual = torch.autograd.grad((y * g).sum(), (scores, w), retain_graph=True)
    expected = torch.autograd.grad((reference * g).sum(), (scores, w))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
