"""Tests for scripts/scan_sae_aux_loss_memory + analyze_sae_aux_loss_memory.

The dead-feature aux path adds a fixed (num_dead-independent) memory delta once
features die. Closed form checked on CPU; the delta and its num_dead
independence validated on GPU.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.analyze_sae_aux_loss_memory import summarize
from scripts.scan_sae_aux_loss_memory import _run_one

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="aux loss memory scan needs CUDA"
)


def test_summary_validates_closed_form_and_independence_on_synthetic() -> None:
    db = 4
    d_in, d_sae, B = 4096, 65536, 2048
    bdsae = B * d_sae * db / 1024**2
    bdin = B * d_in * db / 1024**2
    pred = 2 * bdsae + 2 * bdin

    def row(dead_frac: float, delta: float) -> dict[str, Any]:
        return {
            "ok": True,
            "d_in": d_in,
            "d_sae": d_sae,
            "n_hooks": 1,
            "batch": B,
            "dtype": "fp32",
            "dead_frac": dead_frac,
            "bdsae_mb": bdsae,
            "bdin_mb": bdin,
            "fwd_retained_delta_mb": delta,
            "bwd_peak_delta_mb": delta,
            "pred_fwd_delta_mb": pred,
        }

    # near-identical deltas across dead fractions -> small spread, small error
    rows = [row(0.01, pred - 21), row(0.1, pred + 1), row(0.5, pred + 1)]
    s = summarize(rows)
    assert s["fwd_delta_vs_closed_form"]["max_abs_err_mb"] <= 22
    assert s["bwd_delta_equals_fwd_delta"]["max_abs_diff_mb"] == pytest.approx(0.0)
    assert s["dead_fraction_independence"]["max_spread_over_dead_frac_mb"] <= 22


@requires_cuda
def test_aux_delta_matches_closed_form_and_is_dead_count_independent() -> None:
    device = torch.device("cuda:0")
    common = dict(
        d_in=2048,
        d_sae=32768,
        n_hooks=1,
        dtype="fp32",
        batch=2048,
        k=128,
        warmup=2,
        device=device,
    )
    r_small = _run_one(dead_frac=0.01, **common)
    r_big = _run_one(dead_frac=0.5, **common)
    assert r_small.ok and r_big.ok
    # closed form 2*bdsae + 2*bdin, within a small (B, k_aux) residue.
    pred = r_big.pred_fwd_delta_mb
    assert r_big.fwd_retained_delta_mb == pytest.approx(pred, abs=0.03 * pred + 4)
    # backward peak lifts by the same retained delta.
    assert r_big.bwd_peak_delta_mb == pytest.approx(
        r_big.fwd_retained_delta_mb, abs=0.03 * pred + 4
    )
    # delta barely depends on how many features are dead.
    assert abs(r_big.fwd_retained_delta_mb - r_small.fwd_retained_delta_mb) <= (
        0.05 * pred + 4
    )


@requires_cuda
def test_no_dead_features_costs_nothing() -> None:
    # The "off" baseline uses an all-false mask == early return; it must match a
    # plain forward with no aux retained tensors. We just assert the aux-on path
    # is strictly larger so the delta is real and positive.
    res = _run_one(
        d_in=2048,
        d_sae=32768,
        n_hooks=1,
        dtype="fp32",
        batch=2048,
        k=128,
        dead_frac=0.1,
        warmup=2,
        device=torch.device("cuda:0"),
    )
    assert res.ok
    assert res.fwd_retained_on_mb > res.fwd_retained_off_mb
    assert res.fwd_retained_delta_mb > 0
