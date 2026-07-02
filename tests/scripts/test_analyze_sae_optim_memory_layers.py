"""Tests for scripts/analyze_sae_optim_memory_layers.

Validates the layer decomposition identities on synthetic rows (so no GPU is
needed): reserved = allocated + cached_unreleased, driver = reserved + ctx, and
that mode differences at the reserved layer equal allocated diff plus cached
diff. These identities are the core claim of the layers analysis.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.analyze_sae_optim_memory_layers import (
    build_diff_table,
    group_by_config,
    summarize_diffs,
)


def _row(mode: str, alloc: float, cached: float, ctx: float = 332.0) -> dict[str, Any]:
    reserved = alloc + cached
    return {
        "d_in": 4096,
        "d_sae": 65536,
        "n_hooks": 2,
        "dtype": "fp32",
        "optim_mode": mode,
        "batch": 2048,
        "param_total_mb": 4097.0,
        "param_biggest_mb": 1024.0,
        "step_peak_alloc_mb": alloc,
        "step_peak_reserved_mb": reserved,
        "driver_used_mb": reserved + ctx,
        "cached_unreleased_mb": cached,
        "ctx_constant_mb": ctx,
    }


def test_reserved_delta_decomposes_into_alloc_plus_cached() -> None:
    # fused: low alloc, low cache. foreach: same alloc peak (backward-bound) but
    # extra cached temporaries. for_loop: in between.
    rows = [
        _row("fused", alloc=16434, cached=2962),
        _row("foreach", alloc=16434, cached=6034),
        _row("for_loop", alloc=16434, cached=5034),
    ]
    table = build_diff_table(group_by_config(rows))
    summary = summarize_diffs(table)
    assert summary["reserved_delta_decomposition_max_abs_mb"] == pytest.approx(0.0)
    assert summary["driver_diff_equals_reserved_diff_max_abs_mb"] == pytest.approx(0.0)


def test_reserved_diff_can_exceed_allocated_diff_when_backward_bound() -> None:
    # When all modes tie at the allocated layer, the reserved-layer difference is
    # entirely the cached/unreleased gap — exactly the point that allocated-only
    # measurement misses.
    rows = [
        _row("fused", alloc=16434, cached=2962),
        _row("foreach", alloc=16434, cached=6034),
        _row("for_loop", alloc=16434, cached=5034),
    ]
    table = build_diff_table(group_by_config(rows))
    rec = table[0]
    assert rec["allocated_foreach_minus_fused"] == pytest.approx(0.0)
    assert rec["reserved_foreach_minus_fused"] == pytest.approx(6034 - 2962)


def test_allocated_regime_split_counts_optimizer_and_backward_bound() -> None:
    # Config A: optimizer-bound (foreach lifts alloc by full param_total).
    a = [
        _row("fused", alloc=10000, cached=100),
        _row("foreach", alloc=10000 + 4097, cached=100),
        _row("for_loop", alloc=10000 + 3072, cached=100),
    ]
    # Config B: backward-bound (alloc ties across modes).
    b = [
        {**_row("fused", alloc=20000, cached=100), "d_sae": 32768},
        {**_row("foreach", alloc=20000, cached=100), "d_sae": 32768},
        {**_row("for_loop", alloc=20000, cached=100), "d_sae": 32768},
    ]
    # Config B uses smaller params so the predicted transient differs; set refs.
    for r in b:
        r["param_total_mb"] = 2048.0
        r["param_biggest_mb"] = 512.0
    table = build_diff_table(group_by_config(a + b))
    summary = summarize_diffs(table)
    reg = summary["allocated_foreach_minus_fused_regime"]
    assert reg["optimizer_bound_full_transient"] == 1
    assert reg["backward_bound_zero_diff"] == 1
