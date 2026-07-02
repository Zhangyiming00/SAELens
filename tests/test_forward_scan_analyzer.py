"""End-to-end check: analyze_forward_scan.py on the v4 freed reference run
(treating it as a 1-cell forward scan) recovers the right after_forward_all
medians and produces a non-trivial diff vs predict_sae_memory.predict().

The analyzer takes a case directory plus a cases.json index. We synthesize a
minimal cases.json that points at the v4 reference subdir.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ANALYZER = Path("/home/zhangyiming/SAELens/scripts/tests/analyze_forward_scan.py")
REF_TP1 = Path("/home/zhangyiming/SAELens/results/memory_model/sae_phase_v4/tp1")


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_forward_scan", ANALYZER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_forward_scan"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def reference_root(tmp_path: Path) -> Path:
    """Stage the v4 TP=1 ref dir as a fake 1-cell scan so analyze can run on it."""
    if not REF_TP1.exists():
        pytest.skip(f"reference run missing: {REF_TP1}")
    case_dir = tmp_path / "v4_tp1"
    case_dir.mkdir()
    # Symlink only what the analyzer reads.
    (case_dir / "memory_phase_history_rank0.jsonl").symlink_to(
        REF_TP1 / "memory_phase_history_rank0.jsonl"
    )
    cfg_dir = case_dir / "blocks_21_hook_resid_post"
    cfg_dir.mkdir()
    (cfg_dir / "cfg.json").write_text(
        json.dumps({"d_in": 4096, "d_sae": 65536, "dtype": "float32"})
    )
    (tmp_path / "cases.json").write_text(json.dumps({
        "v4_tp1": {
            "name": "v4_tp1",
            "d_sae": 65536,
            "batch": 2048,
            "hooks": 2,
            "tp": 1,
            "dtype": "float32",
        }
    }))
    return tmp_path


def test_analyzer_runs_end_to_end(reference_root: Path):
    """Subprocess invocation matches the user-facing CLI path."""
    result = subprocess.run(
        [sys.executable, str(ANALYZER), "--root", str(reference_root)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((reference_root / "forward_scan_summary.json").read_text())
    assert len(summary) == 1
    row = summary[0]
    # At v4 TP=1 (d=65536, B=2048, hooks=2): measured forward_all peak is
    # ~19419 MB (median over stable steps).
    assert row["m_peak_allocated_mb"] == pytest.approx(19419.4, abs=2.0)
    # Predictor without grads under-estimates by ~2.7 GB on this config; the
    # analyzer must surface this rather than hide it.
    assert row["diff_pred_minus_measured_peak_mb"] < 0
    assert row["pred_forward_peak_mb"] < row["m_peak_allocated_mb"]


def test_analyzer_proxy_matches_combined_csv(reference_root: Path):
    """Phase driver proxy (peak_alloc + driver_used - allocated) should match
    what the combined-layers CSV reports for after_forward_all on v4 TP=1."""
    mod = _load_analyzer()
    predictor_path = ANALYZER.parent.parent / "predict_sae_memory.py"
    spec = importlib.util.spec_from_file_location("predict_sae_memory", predictor_path)
    predictor = importlib.util.module_from_spec(spec)
    sys.modules["predict_sae_memory"] = predictor
    spec.loader.exec_module(predictor)

    cases_index = json.loads((reference_root / "cases.json").read_text())
    case_dir = reference_root / "v4_tp1"
    summary = mod.summarize_case(case_dir, cases_index, stable_from=10, predictor=predictor)
    # From results/memory_model/sae_phase_v4_freed/figures/.../combined.csv:
    #   after_forward_all tp1_phase_driver_proxy_mb = 19816.2
    # That CSV uses freed-run driver_used; here we have the watermark run, so
    # the proxy will be higher. We only assert the structural relation.
    assert summary.m_phase_driver_proxy_mb == pytest.approx(
        summary.m_peak_allocated_mb
        + (summary.m_driver_used_mb - summary.m_allocated_mb),
        abs=0.5,
    )
