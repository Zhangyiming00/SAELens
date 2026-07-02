"""Tests for scripts/analyze_vllm_capture_memory + scan helpers (no GPU).

Validates the capture closed form and the layer-position selection logic on
synthetic scan rows, so CI does not require vLLM or a GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.analyze_vllm_capture_memory import summarize
from scripts.scan_vllm_capture_memory import _hook_positions


def test_hook_positions_layouts() -> None:
    assert _hook_positions(24, 1, "late") == [23]
    assert _hook_positions(24, 4, "late") == [20, 21, 22, 23]
    assert _hook_positions(24, 4, "early") == [0, 1, 2, 3]
    assert _hook_positions(24, 2, "mid") == [11, 12]
    # spread covers the full depth roughly evenly
    spread = _hook_positions(24, 4, "spread")
    assert spread[0] == 0 and spread[-1] >= 17 and len(spread) == 4
    # n_hooks >= n_layers -> every layer
    assert _hook_positions(24, 24, "late") == list(range(24))


def _doc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "test",
        "dtype": "bf16",
        "arch_layers": 24,
        "d_model": 896,
        "gpu_memory_utilization": 0.45,
        "stages": {
            "before_load": {"alloc": 0.0, "reserved": 0.0, "driver": 270.0},
            "after_load": {"alloc": 1240.0, "reserved": 1962.0, "driver": 2312.0},
        },
        "results": rows,
    }


def _row(n_hooks: int, batch: int, seq: int, d_model: int = 896) -> dict[str, Any]:
    captured = n_hooks * batch * seq * d_model * 2 / 1024**2  # bf16
    return {
        "n_hooks": n_hooks,
        "layout": "late",
        "hook_type": "hook_resid_post",
        "positions": list(range(24 - n_hooks, 24)),
        "batch": batch,
        "seq": seq,
        "capture_peak_reserved_mb": 1962.0,
        "captured_tensor_mb": captured,
        "driver_minus_reserved_mb": 350.0,
        "depth_peak_layer": 23,
        "capture_peak_alloc_mb": 1382.0,
    }


def test_captured_tensor_matches_BSdc_closed_form() -> None:
    rows = [_row(1, 8, 512), _row(4, 8, 512), _row(24, 8, 512), _row(8, 8, 2048)]
    s = summarize(_doc(rows))
    assert s["captured_tensor_vs_BSdc"]["max_abs_err_mb"] == pytest.approx(0.0, abs=0.1)


def test_reserved_within_load_watermark_when_capture_fits() -> None:
    # all rows share the load reserved -> all within, none pushed
    rows = [_row(1, 8, 512), _row(24, 8, 512)]
    s = summarize(_doc(rows))
    rb = s["reserved_behaviour"]
    assert rb["load_reserved_mb"] == pytest.approx(1962.0)
    assert rb["n_within_load_watermark"] == 2
    assert rb["n_pushed_above"] == 0


def test_reserved_pushed_above_load_when_capture_exceeds() -> None:
    # a row whose capture peak reserved exceeds the load watermark is detected.
    big = _row(24, 8, 2048)
    big["capture_peak_reserved_mb"] = 2722.0  # > load 1962
    rows = [_row(1, 8, 512), big]
    s = summarize(_doc(rows))
    rb = s["reserved_behaviour"]
    assert rb["n_pushed_above"] == 1
    assert rb["max_push_above_load_mb"] == pytest.approx(760.0)


def test_ctx_constant_reported() -> None:
    rows = [_row(1, 8, 512), _row(4, 8, 512)]
    s = summarize(_doc(rows))
    assert s["ctx_constant_mb"]["min"] == pytest.approx(350.0)
    assert s["ctx_constant_mb"]["max"] == pytest.approx(350.0)
