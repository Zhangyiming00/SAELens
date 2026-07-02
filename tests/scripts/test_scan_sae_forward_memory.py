"""Tests for scripts/scan_sae_forward_memory + analyze_sae_forward_memory.

Closed forms validated against real measurements on GPU; scaling structure
checked unconditionally on CPU.
"""

from __future__ import annotations

import pytest
import torch

from scripts.analyze_sae_forward_memory import (
    predict_fwd_transient_mb,
    predict_retained_mb,
)
from scripts.scan_sae_forward_memory import _run_one

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="forward memory scan needs CUDA"
)


def test_retained_scales_linearly_with_hooks_and_batch() -> None:
    one = predict_retained_mb(4096, 65536, 1, 2048, "fp32")
    assert predict_retained_mb(4096, 65536, 4, 2048, "fp32") == pytest.approx(4 * one)
    assert predict_retained_mb(4096, 65536, 1, 4096, "fp32") == pytest.approx(2 * one)


def test_retained_counts_four_bdsae_and_two_bdin() -> None:
    db = 4
    d_in, d_sae, B = 4096, 65536, 2048
    bdsae = B * d_sae * db / 1024**2
    bdin = B * d_in * db / 1024**2
    assert predict_retained_mb(d_in, d_sae, 1, B, "fp32") == pytest.approx(
        4 * bdsae + 2 * bdin
    )


def test_dtype_halves_retained() -> None:
    fp32 = predict_retained_mb(4096, 65536, 2, 2048, "fp32")
    bf16 = predict_retained_mb(4096, 65536, 2, 2048, "bf16")
    assert bf16 == pytest.approx(fp32 / 2)


def test_transient_is_retained_plus_one_bdin() -> None:
    d_in, d_sae, B = 4096, 65536, 2048
    bdin = B * d_in * 4 / 1024**2
    ret = predict_retained_mb(d_in, d_sae, 1, B, "fp32")
    assert predict_fwd_transient_mb(d_in, d_sae, 1, B, "fp32") == pytest.approx(
        ret + bdin
    )


@requires_cuda
@pytest.mark.parametrize("n_hooks", [1, 2])
@pytest.mark.parametrize("batch", [1024, 2048])
def test_measured_retained_matches_closed_form(n_hooks: int, batch: int) -> None:
    # Residual (topk B*k indices/values, biases) is a few MB; allow 1% of the
    # (B, d_sae) block plus a small constant.
    d_in, d_sae, dtype = 2048, 32768, "fp32"
    device = torch.device("cuda:0")
    res = _run_one(
        d_in=d_in,
        d_sae=d_sae,
        n_hooks=n_hooks,
        dtype=dtype,
        batch=batch,
        k=128,
        warmup=2,
        device=device,
    )
    assert res.ok, res.error
    pred = predict_retained_mb(d_in, d_sae, n_hooks, batch, dtype)
    tol = 0.01 * pred + 4
    assert res.retained_mb == pytest.approx(pred, abs=tol)
