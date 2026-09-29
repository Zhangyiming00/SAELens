"""Measured native-update interpolation and structural + residual memory.

Topology, scheduler, optimizer, k/AuxK, dead distribution and decoder policies
are discrete: never infer a new scheduler from a single serial run. Batch and
feature width interpolate inside measured rectangles. Sparse stage tables are
separate diagnostics, not additive substitutes for overlapped update latency.
"""
from __future__ import annotations

try:
    from .allocated_peak_model import dimensions, state_payload
    from .interpolation_model import interpolate
except ImportError:
    from allocated_peak_model import dimensions, state_payload
    from interpolation_model import interpolate

SCHEMA = "native_interpolation_v3"


def validate_config(cfg):
    if any(type(cfg.get(k)) is not int for k in ("tp", "dp", "pp", "h", "batch", "ga", "d_in", "d_sae", "live", "dead")):
        raise ValueError("Dimensions, topology and dead count must be integers")
    dimensions(cfg)
    execution = cfg.get("execution", {})
    if cfg.get("dtype", "float32") != "float32" or execution.get("dtype", "float32") != "float32" or cfg.get("autocast", False):
        raise ValueError("The native structural memory model requires FP32 without autocast")
    if any(k in execution and execution[k] != cfg.get(k)
           for k in ("d_in", "d_sae", "k", "auxk")):
        raise ValueError("Execution overrides must not change workload dimensions")
    if type(cfg.get("k")) is not int or not 0 < cfg["k"] <= cfg["d_sae"]:
        raise ValueError("Invalid main k")
    aux = cfg.get("auxk")
    if aux is not None and (type(aux) is not int or aux < 0):
        raise ValueError("AuxK must be None or a nonnegative integer")
    if not 0 <= cfg["dead"] <= cfg["d_sae"]:
        raise ValueError("Invalid dead count")


def selection_regime(cfg):
    width = cfg["d_sae"] // cfg["tp"]
    protocol = cfg.get("execution", {}).get("topk_candidate_protocol", "auto")
    candidate = cfg["tp"] == 1 or cfg["tp"] * min(cfg["k"], width) * 8 <= width * 4
    # The measured Torch key implementation changes algorithm at this width.
    return ("small" if width <= 4096 else "large",
            "candidates" if protocol != "radix" and candidate else "radix")


def predict_native(cfg, profile, *, timeline=False):
    if profile.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA} profile")
    validate_config(cfg)
    estimate = interpolate(cfg, profile["rows"], profile["axes"])
    # Even when endpoints happen to have matching diagnostics, do not mix the
    # known local TopK algorithm families.
    names = {name for corner in estimate["corners"] for name in corner["measurements"]}
    if any(selection_regime(r["config"]) != selection_regime(cfg)
           for r in profile["rows"] if r["name"] in names):
        raise ValueError("TopK algorithm boundary needs separate calibration")
    m = estimate["metrics"]
    ranks = []
    for rank, transient in enumerate(m["rank_memory"]):
        payload = state_payload(cfg, rank)
        structural = sum(payload.values())
        peak = structural + transient["peak_residual_bytes"]
        resident = structural + transient["resident_residual_bytes"]
        if min(peak, resident) < structural or peak < resident:
            raise ValueError("Invalid calibrated memory envelope")
        ranks.append(dict(rank=rank, structural_payload=payload,
                          peak_allocated_bytes=round(peak), resident_allocated_bytes=round(resident),
                          transient_above_resident_bytes=round(peak-resident)))
    return dict(config=cfg, total_ms=m["wall_ms"], tokens_per_second=cfg["batch"]*1000/m["wall_ms"],
                sample_min_ms=m["sample_min_ms"], sample_max_ms=m["sample_max_ms"],
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in ranks), ranks=ranks,
                corners=estimate["corners"], exact=estimate["exact"], regime=estimate["regime"],
                method="multilinear native-update interpolation; exact state payload + interpolated residual",
                warnings=["Timing samples are dispersion, not confidence bounds",
                          "Allocated bytes exclude CUDA/NCCL outside the Torch allocator and allocator reservation",
                          "No vLLM, SHM transport or unmeasured scheduler/optimizer extrapolation"],
                **({"timeline": [dict(name="interpolated_native_update", resource="service",
                                      pool="native_sae", stream="aggregate", start_ms=0.,
                                      elapsed_ms=m["wall_ms"], work_ms=m["wall_ms"],
                                      metadata=dict(detail="Whole update; no inferred CUDA stream timeline"))]}
                   if timeline else {}))


def predict_sparse(cfg, profile):
    if profile.get("schema") != "sparse_parts_interpolation_v1":
        raise ValueError("Expected sparse_parts_interpolation_v1 profile")
    return interpolate(cfg, profile["rows"], profile["axes"])
