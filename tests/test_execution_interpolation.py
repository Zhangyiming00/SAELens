"""Interpolation safety and native memory ownership invariants."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sae_lens.autoconfig.allocated_peak_model import predict_native as predict_memory
from sae_lens.autoconfig.allocated_peak_model import state_payload
from sae_lens.autoconfig.execution_time_model import (
    SCHEMA,
    predict_native,
    predict_sparse,
)
from sae_lens.autoconfig.interpolation_model import interpolate


def grid():
    return [dict(name=f"{b}_{f}", config=dict(batch=b, width=f, engine="openai", policy="none"),
                 regime="sparse", metrics=dict(ms=2*b+3*f+b*f, ranks=[b*f]))
            for b, f in itertools.product((2, 6), (10, 20))]


def test_multilinear_weights_and_independent_bilinear_oracle():
    result = interpolate(dict(batch=3, width=14, engine="openai", policy="none"), grid(), ["batch", "width"])
    assert result["metrics"]["ms"] == pytest.approx(2*3+3*14+3*14)
    assert result["metrics"]["ranks"] == pytest.approx([42])
    assert sum(c["weight"] for c in result["corners"]) == pytest.approx(1)
    assert len(result["corners"]) == 4


def test_duplicate_measurements_use_median_and_exact_points_need_no_neighbours():
    rows = [grid()[0], copy.deepcopy(grid()[0]), copy.deepcopy(grid()[0])]
    rows[1]["metrics"]["ms"] += 1000
    result = interpolate(rows[0]["config"], rows, ["batch", "width"])
    assert result["metrics"]["ms"] == rows[0]["metrics"]["ms"]
    assert result["exact"]


@pytest.mark.parametrize("change,match", [({"batch": 7}, "extrapolation"),
                                         ({"engine": "triton"}, "family"),
                                         ({"policy": None}, "family")])
def test_no_extrapolation_no_engine_alias_no_null_alias(change, match):
    cfg = dict(batch=3, width=14, engine="openai", policy="none", **{})
    cfg.update(change)
    with pytest.raises(ValueError, match=match):
        interpolate(cfg, grid(), ["batch", "width"])


def test_missing_corner_and_algorithm_boundary_are_not_silently_scaled():
    cfg = dict(batch=3, width=14, engine="openai", policy="none")
    with pytest.raises(ValueError, match="Missing interpolation corner"):
        interpolate(cfg, grid()[:-1], ["batch", "width"])
    rows = grid()
    rows[-1]["regime"] = "dense"
    with pytest.raises(ValueError, match="Execution path changes"):
        interpolate(cfg, rows, ["batch", "width"])


def native_profile(dp=1, zero=False):
    base = dict(tp=2, dp=dp, pp=1, h=1, batch=4096, ga=1, d_in=4096,
                d_sae=65536, k=128, auxk=0, dead=0, wave="off", live=1,
                zero=zero, overlap="off", backend="sharded_dense",
                execution=dict(main_representation="none", main_compute="none"))
    rows = []
    for b, f in itertools.product((4096, 8192), (32768, 65536)):
        cfg = dict(base, batch=b, d_sae=f)
        memory = [dict(peak_residual_bytes=4*b*f/cfg["tp"]/dp,
                       resident_residual_bytes=512.) for _ in range(cfg["tp"]*dp)]
        rows.append(dict(name=f"{b}_{f}", config=cfg, regime="openai_sparse_disabled_aux",
                         metrics=dict(wall_ms=b*f/1e6, sample_min_ms=b*f/1e6*.9,
                                      sample_max_ms=b*f/1e6*1.1, rank_memory=memory)))
    return dict(base, batch=6144, d_sae=49152), dict(schema=SCHEMA, axes=["batch", "d_sae"], rows=rows)


@pytest.mark.parametrize("dp,zero", [(1, False), (2, False), (2, True)])
def test_structural_states_are_computed_at_target_with_rank_ownership(dp, zero):
    cfg, profile = native_profile(dp, zero)
    result = predict_native(cfg, profile)
    for rank, mem in enumerate(result["ranks"]):
        assert mem["structural_payload"] == state_payload(cfg, rank)
        expected = sum(state_payload(cfg, rank).values()) + 4*6144*49152/2/dp
        assert mem["peak_allocated_bytes"] == expected
    assert predict_memory(cfg, profile)["peak_allocated"] == result["ranks"][0]["peak_allocated_bytes"]


def test_unmeasured_optimizer_aux_state_or_scheduler_requires_new_profile():
    cfg, profile = native_profile()
    for change in (dict(auxk=None), dict(h=3), dict(zero=True), dict(wave="lazy"), dict(k=64)):
        with pytest.raises(ValueError, match="family"):
            predict_native(dict(cfg, **change), profile)


def test_memory_does_not_apply_fp32_payload_to_bfloat16_or_overridden_shapes():
    cfg, profile = native_profile()
    cfg["execution"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="FP32"):
        predict_native(cfg, profile)
    cfg, profile = native_profile()
    cfg["execution"]["d_sae"] = 1
    with pytest.raises(ValueError, match="workload dimensions"):
        predict_native(cfg, profile)


def test_no_interpolation_across_known_topk_width_boundary():
    cfg, profile = native_profile()
    profile["rows"] = [dict(r, config=dict(r["config"], d_sae=r["config"]["d_sae"]//8))
                       for r in profile["rows"]]
    cfg["d_sae"] = 6144
    # Both local widths are <=4096, so this bracket is allowed.
    predict_native(cfg, profile)
    for row in profile["rows"]:
        row["config"]["d_sae"] *= 2
    cfg["d_sae"] = 12288
    with pytest.raises(ValueError, match="TopK algorithm boundary"):
        predict_native(cfg, profile)


@pytest.mark.parametrize("schema", [1, 2, "k_aux_v1"])
def test_retired_phase_profiles_are_not_used_as_current_interpolation(schema):
    cfg, profile = native_profile()
    profile["schema"] = schema
    with pytest.raises(ValueError, match=SCHEMA):
        predict_native(cfg, profile)


def cli_case(tmp_path, sparse=False):
    if sparse:
        cfg = dict(batch=3, width=14, engine="openai", policy="none")
        profile = dict(schema="sparse_parts_interpolation_v1", axes=["batch", "width"], rows=grid())
    else:
        cfg, profile = native_profile()
    source = tmp_path / "runtime.py"
    source.write_text("# measured implementation\n")
    profile["source_sha256"] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    configs = tmp_path / "configs.json"
    configs.write_text(json.dumps(dict(target=cfg)))
    output = tmp_path / "predictions.json"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/profile/simulate_megatron_execution_time.py"),
               "predict", "--profile", str(profile_path), "--configs", str(configs), "--output", str(output)]
    return command, output, source, cfg, profile


@pytest.mark.parametrize("sparse", [False, True])
def test_unified_cli_predicts_both_tables_and_keeps_frozen_outputs(tmp_path, sparse):
    command, output, _, cfg, profile = cli_case(tmp_path, sparse)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = (predict_sparse if sparse else predict_native)(cfg, profile)
    frozen = output.read_bytes()
    assert json.loads(frozen)["predictions"]["target"] == expected
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == frozen


@pytest.mark.parametrize("missing", [False, True])
def test_unified_cli_rejects_changed_or_removed_profile_sources(tmp_path, missing):
    command, output, source, _, _ = cli_case(tmp_path)
    if missing:
        source.unlink()
    else:
        source.write_text("# changed implementation\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert ("no longer exists" if missing else "Source changed") in result.stderr
    assert not output.exists()


def test_elastic_measurement_can_run_without_an_uncalibrated_prediction():
    from scripts.profile.profile_elastic_watermarks import predict

    args = SimpleNamespace(native_profile=None, microbatch=6144, chunk_tokens=2048,
                           prompts=1, context=1024, chunks=256)
    result = predict(args)
    assert result["prediction_status"] == "not_requested"
    assert result["selected"] is None
    assert result["shm_host_gib"] == 24


def test_elastic_predictions_use_current_model_and_require_measured_dp_family(tmp_path):
    from scripts.profile.profile_elastic_watermarks import predict

    base, _ = native_profile()
    configs = {f"dp{dp}": dict(base, tp=1, dp=dp, h=3, d_sae=65536, batch=12288, ga=2) for dp in (2, 3)}
    rows = [dict(name=name, config=cfg, regime="native",
                 metrics=dict(wall_ms=100., sample_min_ms=99., sample_max_ms=101.,
                              rank_memory=[dict(peak_residual_bytes=2048., resident_residual_bytes=1024.)
                                           for _ in range(cfg["dp"])])) for name, cfg in configs.items()]
    profile = dict(schema=SCHEMA, axes=["batch", "d_sae"], rows=rows, source_sha256={})
    path, cfg_path = tmp_path / "native.json", tmp_path / "configs.json"
    path.write_text(json.dumps(profile))
    cfg_path.write_text(json.dumps(configs))
    args = SimpleNamespace(native_profile=str(path), native_configs=str(cfg_path), d_sae=65536,
                           microbatch=6144, ga=2, chunk_tokens=2048, prompts=1, context=1024, chunks=256)
    result = predict(args)
    assert result["prediction_status"] == "calibrated"
    assert [r["tokens_per_second"] for r in result["selected"]["estimates"]] == [122880., 122880.]
    profile["rows"] = rows[:1]
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="Unmeasured execution family"):
        predict(args)
