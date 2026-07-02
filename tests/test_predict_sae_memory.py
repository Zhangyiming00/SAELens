"""Snapshot tests for ``scripts/predict_sae_memory.py`` against the v4
reference TP1/TP2 multi-hook runs in ``results/memory_model/sae_phase_v4/``.

We calibrate against **two independent layers** of GPU memory because they
answer different questions:

  - ``peak_allocated`` — bytes held by live tensors. This is what the
    in-trainer phase records use, and it is what the plot script displays.
  - ``driver_used`` — what nvidia-smi sees: allocator pool + CUDA context
    + cuBLAS/cuDNN workspaces + NCCL bootstrap buffers. **This is the only
    quantity that determines whether a config actually fits on the GPU.**

If only ``peak_allocated`` is calibrated, the predictor can pass tests yet
still be off by ~10 % on real VRAM footprint (TP=1 needs ~2.6 GB more than
``peak_allocated`` suggests on the v4 reference run). The driver-layer
assertions catch that class of bug.

The predictor is allowed to overshoot measurement (better safe than OOM)
but must not undershoot by more than a small slack on either layer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from statistics import median

import pytest

REF_TP1 = Path(
    "/home/zhangyiming/SAELens/results/memory_model/sae_phase_v4/tp1/memory_phase_history_rank0.jsonl"
)
REF_TP2 = Path(
    "/home/zhangyiming/SAELens/results/memory_model/sae_phase_v4/tp2/memory_phase_history_rank0.jsonl"
)
PREDICTOR = Path("/home/zhangyiming/SAELens/scripts/predict_sae_memory.py")


def _load_predictor():
    if "predict_sae_memory" in sys.modules:
        return sys.modules["predict_sae_memory"]
    spec = importlib.util.spec_from_file_location("predict_sae_memory", PREDICTOR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["predict_sae_memory"] = mod
    spec.loader.exec_module(mod)
    return mod


def _measured_per_phase(jsonl_path: Path, key: str) -> dict[str, float]:
    """Median of ``key`` per phase across stable steps."""
    if not jsonl_path.exists():
        pytest.skip(f"reference run missing: {jsonl_path}")
    rows = [json.loads(line) for line in jsonl_path.open() if line.strip()]
    rows = [r for r in rows if r["step"] >= 10]
    by_phase: dict[str, list[float]] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r[key])
    return {ph: median(vals) for ph, vals in by_phase.items()}


def _measured_step_max(jsonl_path: Path, key: str) -> float:
    """Run-wide max of ``key`` across all phases and stable steps."""
    rows = [json.loads(line) for line in jsonl_path.open() if line.strip()]
    rows = [r for r in rows if r["step"] >= 10]
    return max(r[key] for r in rows)


@pytest.mark.parametrize(
    "tp,jsonl",
    [(1, REF_TP1), (2, REF_TP2)],
    ids=["tp1", "tp2"],
)
def test_predict_peak_allocated_per_phase_within_150mb(tp: int, jsonl: Path):
    """Layer 1: per-phase peak_allocated_mb must match within +/- 150 MB,
    with no more than 50 MB undershoot (would cause silent OOM)."""
    mod = _load_predictor()
    measured = _measured_per_phase(jsonl, "peak_allocated_mb")

    p = mod.predict(
        d_in=4096,
        d_sae=65536,
        batch_tokens=2048,
        k=128,
        dtype="float32",
        hooks=2,
        tp=tp,
        dp_size=1,
        dp_mode="ddp",
    )

    pairs = [
        ("after_stats_tail", p.persistent_mb, "persistent"),
        ("after_combined_backward", p.backward_peak_mb, "backward"),
        ("after_optimizer_step", p.optimizer_peak_mb, "optimizer"),
    ]
    tol = 150
    for phase, predicted, label in pairs:
        assert phase in measured, phase
        m = measured[phase]
        diff = predicted - m
        assert -tol <= diff <= tol, (
            f"tp={tp} {label}: predicted {predicted:.1f} vs measured {m:.1f} "
            f"(diff {diff:+.1f} MB, allowed +-{tol})"
        )
        assert diff >= -50, (
            f"tp={tp} {label}: predicted {predicted:.1f} undershoots measured "
            f"{m:.1f} by {abs(diff):.1f} MB; predictor would mislead capacity "
            "planning."
        )


@pytest.mark.parametrize(
    "tp,jsonl",
    [(1, REF_TP1), (2, REF_TP2)],
    ids=["tp1", "tp2"],
)
def test_predict_driver_used_run_max_within_envelope(tp: int, jsonl: Path):
    """Layer 2: run-wide max(driver_used_mb) must be predicted with a
    conservative bias — never undershoot, never overshoot by more than 1.5 GB.

    Driver-used is the quantity that nvidia-smi reports and the only one the
    OS uses to decide whether to OOM-kill the process. A predictor that
    matches peak_allocated but ignores allocator fragmentation + CUDA context
    + NCCL workspaces would silently mislead capacity planning by ~10 %; this
    test pins down both directions.
    """
    mod = _load_predictor()
    measured_driver = _measured_step_max(jsonl, "driver_used_mb")

    p = mod.predict(
        d_in=4096,
        d_sae=65536,
        batch_tokens=2048,
        k=128,
        dtype="float32",
        hooks=2,
        tp=tp,
        dp_size=1,
        dp_mode="ddp",
    )

    diff = p.step_peak_driver_mb - measured_driver
    # Must overshoot (positive diff) — never tell the user a config fits when
    # it doesn't.
    assert diff >= 0, (
        f"tp={tp}: driver_used predicted {p.step_peak_driver_mb:.1f} undershoots "
        f"measured {measured_driver:.1f} by {abs(diff):.1f} MB. The predictor "
        "would tell users a config fits when it actually OOMs."
    )
    # Don't be wildly over-conservative either; users would over-provision.
    assert diff <= 1500, (
        f"tp={tp}: driver_used predicted {p.step_peak_driver_mb:.1f} overshoots "
        f"measured {measured_driver:.1f} by {diff:.1f} MB; tighten the slack."
    )


@pytest.mark.parametrize(
    "tp,jsonl",
    [(1, REF_TP1), (2, REF_TP2)],
    ids=["tp1", "tp2"],
)
def test_predict_layers_are_strictly_ordered(tp: int, jsonl: Path):  # noqa: ARG001 — jsonl kept for parametrize symmetry
    """peak_allocated <= peak_reserved <= driver_used. This is a structural
    invariant in CUDA, and the predictor must respect it."""
    mod = _load_predictor()
    p = mod.predict(
        d_in=4096,
        d_sae=65536,
        batch_tokens=2048,
        k=128,
        dtype="float32",
        hooks=2,
        tp=tp,
    )
    assert p.step_peak_allocated_mb < p.step_peak_reserved_mb, (
        p.step_peak_allocated_mb,
        p.step_peak_reserved_mb,
    )
    assert p.step_peak_reserved_mb < p.step_peak_driver_mb, (
        p.step_peak_reserved_mb,
        p.step_peak_driver_mb,
    )


def test_step_peak_is_max_of_phases():
    mod = _load_predictor()
    p = mod.predict(
        d_in=512,
        d_sae=4096,
        batch_tokens=512,
        k=16,
        dtype="bfloat16",
        hooks=1,
        tp=1,
    )
    assert p.step_peak_mb == max(
        p.forward_peak_mb, p.backward_peak_mb, p.optimizer_peak_mb
    )
    # Backwards-compat alias points at the peak_allocated layer.
    assert p.step_peak_mb == p.step_peak_allocated_mb


def test_tp_halves_param_components():
    mod = _load_predictor()
    p1 = mod.predict(
        d_in=1024, d_sae=8192, batch_tokens=1024, k=32,
        dtype="float32", hooks=1, tp=1,
    )
    p2 = mod.predict(
        d_in=1024, d_sae=8192, batch_tokens=1024, k=32,
        dtype="float32", hooks=1, tp=2,
    )
    # b_dec is replicated, so the ratio is slightly above 0.5.
    ratio = p2.components["params_per_rank"] / p1.components["params_per_rank"]
    assert 0.50 <= ratio <= 0.501, ratio
    # Adam follows params.
    ratio_adam = p2.components["adam_per_rank"] / p1.components["adam_per_rank"]
    assert 0.50 <= ratio_adam <= 0.501, ratio_adam
    # Retained outputs are TP-invariant (allgather of feature_acts).
    assert p1.components["retained_outputs"] == pytest.approx(
        p2.components["retained_outputs"]
    )


def test_nccl_overhead_only_when_tp_above_one():
    """CUDA context slack should grow with TP because each extra TP peer adds
    NCCL bootstrap buffers."""
    mod = _load_predictor()
    p1 = mod.predict(d_in=1024, d_sae=4096, batch_tokens=512, k=16,
                     dtype="float32", hooks=1, tp=1)
    p2 = mod.predict(d_in=1024, d_sae=4096, batch_tokens=512, k=16,
                     dtype="float32", hooks=1, tp=2)
    p4 = mod.predict(d_in=1024, d_sae=4096, batch_tokens=512, k=16,
                     dtype="float32", hooks=1, tp=4)
    assert p1.overhead["nccl_overhead"] == 0
    assert p2.overhead["nccl_overhead"] > 0
    assert p4.overhead["nccl_overhead"] > p2.overhead["nccl_overhead"]
    # CUDA context slack must be monotonic in TP.
    assert (
        p1.overhead["cuda_context_slack"]
        < p2.overhead["cuda_context_slack"]
        < p4.overhead["cuda_context_slack"]
    )


# --- phase decomposition --------------------------------------------------


@pytest.mark.parametrize("tp,jsonl", [(1, REF_TP1), (2, REF_TP2)], ids=["tp1", "tp2"])
def test_predict_driver_proxy_matches_measured_per_phase(tp: int, jsonl: Path):
    """Driver model decomposes as ``max_p (peak_alloc_p + runtime_residual)``
    + cross-phase pool fragmentation. Validate the proxy half against measured
    ``phase_driver_proxy = peak_alloc + (driver_freed - allocated)`` per phase
    on the freed reference run.

    The optimizer phase is the binding phase on v4 (largest peak_allocated +
    same runtime residual). Predicted proxy must overshoot but stay within an
    envelope of the measured proxy.
    """
    mod = _load_predictor()

    freed_jsonl = Path(
        f"/home/zhangyiming/SAELens/results/memory_model/sae_phase_v4_freed/tp{tp}/"
        "memory_phase_history_rank0.jsonl"
    )
    if not freed_jsonl.exists():
        pytest.skip(f"freed reference run missing: {freed_jsonl}")

    measured_alloc = _measured_per_phase(jsonl, "allocated_mb")
    measured_peak = _measured_per_phase(jsonl, "peak_allocated_mb")
    measured_freed = _measured_per_phase(freed_jsonl, "driver_used_mb")

    measured_proxy = {
        ph: measured_peak[ph] + (measured_freed[ph] - measured_alloc[ph])
        for ph in measured_alloc
        if ph in measured_freed
    }
    measured_max_proxy = max(measured_proxy.values())

    p = mod.predict(
        d_in=4096, d_sae=65536, batch_tokens=2048, k=128,
        dtype="float32", hooks=2, tp=tp, dp_size=1, dp_mode="ddp",
    )

    # Predicted max proxy should overshoot measured (so a fits-verdict that
    # uses driver_proxy is conservative) but only by a modest envelope —
    # otherwise we waste VRAM.
    diff = p.step_peak_driver_proxy_mb - measured_max_proxy
    assert diff >= 0, (
        f"tp={tp}: predicted driver_proxy {p.step_peak_driver_proxy_mb:.1f} "
        f"undershoots measured max {measured_max_proxy:.1f} by {abs(diff):.1f} MB"
    )
    assert diff <= 800, (
        f"tp={tp}: predicted driver_proxy {p.step_peak_driver_proxy_mb:.1f} "
        f"overshoots measured max {measured_max_proxy:.1f} by {diff:.1f} MB; "
        "tighten runtime_residual."
    )

    # Optimizer phase is the binding phase on v4 — both at TP=1 and TP=2.
    assert p.binding_phase == "optimizer"


def test_phase_predictions_are_individually_addressable():
    """Each phase exposes its own peak_alloc + runtime_residual so future
    callers can take the max over a different / extended phase set."""
    mod = _load_predictor()
    # Use the v4 reference config — it's representative of real workloads
    # (and that's where forward < backward < optimizer holds, due to Adam
    # foreach being the largest transient).
    p = mod.predict(
        d_in=4096, d_sae=65536, batch_tokens=2048, k=128,
        dtype="float32", hooks=2, tp=1,
    )
    names = [ph.name for ph in p.phases]
    assert names == ["forward", "backward", "optimizer"]
    fwd, bwd, opt = p.phases
    # Optimizer adds Adam foreach transient on top of persistent (params·hooks
    # extra). At v4 reference scale (params per hook ≈ 1 GB) optimizer is
    # the binding phase.
    assert opt.peak_allocated_mb > fwd.peak_allocated_mb
    assert opt.peak_allocated_mb > bwd.peak_allocated_mb
    # Per-phase driver proxy = peak_alloc + runtime_residual.
    for ph in p.phases:
        assert ph.driver_proxy_mb == pytest.approx(
            ph.peak_allocated_mb + ph.runtime_residual_mb
        )
    # Step driver proxy is the max of phase proxies.
    assert p.step_peak_driver_proxy_mb == max(ph.driver_proxy_mb for ph in p.phases)
    # Final driver = proxy + pool fragmentation tax.
    assert p.step_peak_driver_mb == pytest.approx(
        p.step_peak_driver_proxy_mb + p.overhead["pool_fragmentation"]
    )


def test_pool_fragmentation_independent_of_phase_max():
    """Two configs with different optimizer-phase peaks should both incur a
    pool tax that scales with peak_allocated, not a magic constant."""
    mod = _load_predictor()
    small = mod.predict(
        d_in=512, d_sae=4096, batch_tokens=512, k=16,
        dtype="float32", hooks=1, tp=1,
    )
    large = mod.predict(
        d_in=4096, d_sae=65536, batch_tokens=2048, k=128,
        dtype="float32", hooks=2, tp=1,
    )
    # Pool fragmentation grows with peak_allocated; never below the floor.
    assert small.overhead["pool_fragmentation"] >= 800.0
    assert large.overhead["pool_fragmentation"] > small.overhead["pool_fragmentation"]

