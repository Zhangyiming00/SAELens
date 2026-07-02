"""Tests for scripts/benchmark_sae_optim_phases.

The closed-form predictions are validated against real CUDA-allocator
measurements when a GPU is available (the meat of the feature), and the
scaling-law structure is checked unconditionally on CPU.
"""

from __future__ import annotations

import pytest
import torch

from scripts.benchmark_sae_optim_phases import (
    _build_optimizer,
    _build_saes,
    _measure_isolated_optim,
    _param_refs_mb,
    predict_optim_transient_mb,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer memory tracing needs CUDA"
)


def test_param_refs_match_topk_sae_layout() -> None:
    # W_enc + W_dec + b_enc + b_dec, fp32. d_in=4096 d_sae=65536.
    total_mb, biggest_mb = _param_refs_mb(4096, 65536, n_hooks=1, dtype="fp32")
    expected_total = (2 * 4096 * 65536 + 65536 + 4096) * 4 / 1024**2
    expected_biggest = 4096 * 65536 * 4 / 1024**2
    assert total_mb == pytest.approx(expected_total)
    assert biggest_mb == pytest.approx(expected_biggest)


def test_total_params_scale_with_hooks_but_biggest_does_not() -> None:
    t1, b1 = _param_refs_mb(4096, 65536, 1, "fp32")
    t4, b4 = _param_refs_mb(4096, 65536, 4, "fp32")
    assert t4 == pytest.approx(4 * t1)
    assert b4 == pytest.approx(b1)


def test_fused_predicts_zero_transient_for_all_shapes() -> None:
    for n_hooks in (1, 2, 4):
        for d_sae in (16384, 65536):
            assert (
                predict_optim_transient_mb("fused", 4096, d_sae, n_hooks, "fp32") == 0.0
            )


def test_foreach_scales_linearly_with_hooks() -> None:
    one = predict_optim_transient_mb("foreach", 4096, 65536, 1, "fp32")
    four = predict_optim_transient_mb("foreach", 4096, 65536, 4, "fp32")
    assert four == pytest.approx(4 * one)


def test_for_loop_is_three_biggest_and_hook_independent() -> None:
    _, biggest = _param_refs_mb(4096, 65536, 1, "fp32")
    for n_hooks in (1, 2, 4):
        pred = predict_optim_transient_mb("for_loop", 4096, 65536, n_hooks, "fp32")
        assert pred == pytest.approx(3 * biggest)


def test_foreach_overtakes_for_loop_only_for_multi_hook() -> None:
    # H=1: foreach (2 biggest) < for_loop (3 biggest). H>=2: foreach grows past.
    fe1 = predict_optim_transient_mb("foreach", 4096, 65536, 1, "fp32")
    fl1 = predict_optim_transient_mb("for_loop", 4096, 65536, 1, "fp32")
    assert fe1 < fl1
    fe2 = predict_optim_transient_mb("foreach", 4096, 65536, 2, "fp32")
    fl2 = predict_optim_transient_mb("for_loop", 4096, 65536, 2, "fp32")
    assert fe2 > fl2


def test_dtype_halves_transient() -> None:
    fp32 = predict_optim_transient_mb("foreach", 4096, 65536, 2, "fp32")
    bf16 = predict_optim_transient_mb("foreach", 4096, 65536, 2, "bf16")
    assert bf16 == pytest.approx(fp32 / 2)


@requires_cuda
@pytest.mark.parametrize("optim_mode", ["fused", "foreach", "for_loop"])
@pytest.mark.parametrize("n_hooks", [1, 2])
def test_measured_isolated_transient_matches_closed_form(
    optim_mode: str, n_hooks: int
) -> None:
    # Small d_sae so the test is fast but the param blocks are still well above
    # allocator noise; the closed form must hold to within a few MiB.
    d_in, d_sae, dtype = 1024, 8192, "fp32"
    device = torch.device("cuda:0")
    saes = _build_saes(
        n_hooks=n_hooks, d_in=d_in, d_sae=d_sae, k=64, dtype=dtype, device=device
    )
    optimizer = _build_optimizer(saes, optim_mode)
    try:
        measured = _measure_isolated_optim(saes, optimizer, 0)["iso_transient_mb"]
    finally:
        del saes, optimizer
        torch.cuda.empty_cache()
    predicted = predict_optim_transient_mb(optim_mode, d_in, d_sae, n_hooks, dtype)
    assert measured == pytest.approx(predicted, abs=2.0)
