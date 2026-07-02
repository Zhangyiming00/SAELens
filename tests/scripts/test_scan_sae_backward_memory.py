"""Tests for scripts/scan_sae_backward_memory + analyze_sae_backward_memory.

The backward phase has one exact relation (grad == M_params) and one clean
closed form only in the P_big-dominant regime; the batch-dominant regime is
schedule-dependent. Tests check the closed forms on CPU and validate the
regime-A form against a real GPU measurement.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.analyze_sae_backward_memory import (
    check_grad_equals_params,
    check_peak_over_end_hook_independent,
)
from scripts.scan_sae_backward_memory import (
    _run_one,
    predict_bwd_transient_mb,
    predict_grad_net_mb,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="backward memory scan needs CUDA"
)


def test_grad_equals_param_count_and_scales_with_hooks() -> None:
    one = predict_grad_net_mb(4096, 65536, 1, "fp32")
    expected = (2 * 4096 * 65536 + 65536 + 4096) * 4 / 1024**2
    assert one == pytest.approx(expected)
    assert predict_grad_net_mb(4096, 65536, 4, "fp32") == pytest.approx(4 * one)


def test_transient_leading_order_is_pbig_plus_two_bdsae() -> None:
    d_in, d_sae, B = 4096, 65536, 2048
    p_big = d_in * d_sae * 4 / 1024**2
    bdsae = B * d_sae * 4 / 1024**2
    assert predict_bwd_transient_mb(d_in, d_sae, B, "fp32") == pytest.approx(
        p_big + 2 * bdsae
    )


def test_analysis_reports_regime_a_hook_independence() -> None:
    # Two hook counts, P_big-dominant shape, identical peak_over_end -> spread 0.
    def row(n_hooks: int, poe: float) -> dict[str, Any]:
        return {
            "ok": True,
            "d_in": 4096,
            "d_sae": 65536,
            "batch": 2048,
            "dtype": "fp32",
            "n_hooks": n_hooks,
            "param_biggest_mb": 1024.0,
            "bdsae_mb": 512.0,
            "peak_over_end_mb": poe,
            "grad_resident_mb": 2090.0,
            "pred_grad_mb": 2048.0,
        }

    rows = [row(1, 2084.0), row(2, 2084.0), row(4, 2084.0)]
    res = check_peak_over_end_hook_independent(rows)
    assert res["regime_A_max_spread_over_hooks_mb"] == pytest.approx(0.0)
    grad = check_grad_equals_params(rows)
    assert 0 < grad["max_excess_mb"] < 100  # only the surviving (B,d_in) + topk


@requires_cuda
def test_measured_grad_equals_params() -> None:
    d_in, d_sae, B = 2048, 16384, 1024
    res = _run_one(
        d_in=d_in,
        d_sae=d_sae,
        n_hooks=1,
        dtype="fp32",
        batch=B,
        k=128,
        warmup=2,
        device=torch.device("cuda:0"),
    )
    assert res.ok, res.error
    # grad_resident = after_bwd - param_floor; equals M_params plus the surviving
    # sae_in / retained (B, d_in) blocks + topk bookkeeping. That excess scales
    # with B*d_in*db (a few such blocks), not with the parameter count.
    excess = res.grad_resident_mb - res.pred_grad_mb
    assert 0 < excess <= 3 * res.bdin_mb + 12


@requires_cuda
def test_measured_regime_a_transient_matches_closed_form() -> None:
    # P_big = 2048*16384*4 = 128MB; bdsae at B=1024 = 64MB -> P_big >= 2*bdsae.
    d_in, d_sae, B = 2048, 16384, 1024
    res = _run_one(
        d_in=d_in,
        d_sae=d_sae,
        n_hooks=1,
        dtype="fp32",
        batch=B,
        k=128,
        warmup=2,
        device=torch.device("cuda:0"),
    )
    assert res.ok, res.error
    assert res.param_biggest_mb >= 2 * res.bdsae_mb  # regime A precondition
    p_big = res.param_biggest_mb
    bdsae = res.bdsae_mb
    assert res.peak_over_end_mb == pytest.approx(p_big + 2 * bdsae, abs=40)
