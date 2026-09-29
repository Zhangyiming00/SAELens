"""These tests are SKIPPED on CPU; they are not evidence of GPU validation."""

import pytest
import torch

from sae_lens.sharded_sparse import sparse_decode
from sae_lens.sharded_topk import score_keys, sharded_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU unavailable"
)


@pytest.mark.parametrize("k", [1, 128, 512])
@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("workspace_mib", [256, 8192])
def test_candidate_workspace_preserves_exact_winners(k, backend, workspace_mib):
    from sae_lens.sharded_topk import _local_candidates

    torch.manual_seed(45)
    scores = torch.randint(-10, 11, (257, 16384), device="cuda").float()
    scores[0, :5] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")], device="cuda"
    )
    eligible = torch.arange(16384, device="cuda") % 3 != 2
    reference = score_keys(scores, 16384).masked_fill(~eligible, -(1 << 63))
    ref_keys, ref_ids = reference.topk(k, dim=1, sorted=False)
    keys, ids = _local_candidates(
        scores, k, 16384, eligible, backend, workspace_bytes=workspace_mib * 1024**2
    )
    torch.testing.assert_close(ids.sort(1).values, ref_ids.sort(1).values)
    torch.testing.assert_close(keys.sort(1).values, ref_keys.sort(1).values)


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("local_dead", [0, 17, 128])
def test_compacted_aux_candidates_gpu_padding(backend, local_dead):
    from sae_lens.sharded_topk import _local_candidates

    scores = torch.zeros(97, 1024, device="cuda")
    scores[:, 3] = float('nan')
    scores[:, 4] = -float('inf')
    eligible = torch.arange(1024, device="cuda") < local_dead
    keys, ids = _local_candidates(scores, 512, 4096, eligible, backend)
    reference = score_keys(scores, 4096).masked_fill(~eligible, -(1 << 63))
    torch.testing.assert_close(keys.sort(1).values, reference.topk(512, dim=1).values.sort(1).values)
    torch.testing.assert_close(keys, reference.gather(1, ids))
    assert all(row.unique().numel() == 512 for row in ids)


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
