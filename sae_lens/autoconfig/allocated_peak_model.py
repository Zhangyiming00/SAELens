"""Config-only, calibrated allocated estimates for the static audit workload.

Exact FP32 state payloads are separate from calibrated resident/peak envelopes.
No target snapshot, allocation inventory, or measured branch diagnostics enter
prediction. See docs/allocated_prediction_validation.md for the tested domain.
"""
from __future__ import annotations

import math

GIB = 2**30
VERSION = "allocated-static-v1"


def dimensions(cfg, rank=0):
    t, r, p, h = (int(cfg[k]) for k in ("tp", "dp", "pp", "h"))
    d, f, u, a = (int(cfg[k]) for k in ("d_in", "d_sae", "batch", "ga"))
    if min(t, r, p, h, d, f, u, a) < 1 or h < p:
        raise ValueError("Invalid shape/topology")
    if f % t or u % (r * a) or not 0 <= rank < t * r * p:
        raise ValueError("Requires divisible features/update batch and a valid rank")
    placement, dp_rank = (rank // t) % p, rank // (t * p)
    hooks = h // p + int(placement < h % p)
    wave = cfg["wave"]
    if wave not in ("off", "lazy", "bounded") or int(cfg["live"]) < 1:
        raise ValueError("Only off/lazy/bounded are modeled")
    w = (1 if wave == "off" or t == 1 or hooks == 1 else
         hooks if wave == "lazy" else min(hooks, cfg["live"]))
    return dict(t=t, r=r, d=d, f=f, s=f // t, b=u // (r * a),
                a=a, hooks=hooks, dp_rank=dp_rank, wave=w)


def parameter_layout(d, s, dp, zero):
    """Current native registration order reversed; one FP32 parameter group.

    SAE PP is placement: each domain has singleton Megatron PP, so all domains
    use the normal default bucket cap.
    """
    sizes = [d * s, s, d * s, d]  # decoder W, encoder b, encoder W, decoder b
    cursor, bucket_start = 0, 0
    ranges, buckets = [], []
    cap = max(40_000_000, 1_000_000 * dp)
    align = math.lcm(dp, 128)
    for n in sizes:
        if zero:
            cursor = ((cursor + 63) // 64) * 64
        ranges.append((cursor, cursor + n))
        cursor += n
        if cursor - bucket_start >= cap:
            if zero:
                cursor = ((cursor + align - 1) // align) * align
            buckets.append((bucket_start, cursor))
            bucket_start = cursor
    if cursor > bucket_start:
        if zero:
            cursor = ((cursor + align - 1) // align) * align
        buckets.append((bucket_start, cursor))
    owned = []
    for rank in range(dp):
        total = 0
        for lo, hi in buckets:
            if zero:
                step = (hi - lo) // dp
                left, right = lo + rank * step, lo + (rank + 1) * step
            else:
                left, right = lo, hi
            total += sum(max(0, min(end, right) - max(start, left))
                         for start, end in ranges)
        owned.append(total)
    return dict(numel=cursor, owned=owned, ranges=ranges, buckets=buckets)


def state_payload(cfg, rank=0):
    v = dimensions(cfg, rank)
    d, s, r, h = (v[k] for k in ("d", "s", "r", "hooks"))
    zero = bool(cfg["zero"]) and r > 1
    layout = parameter_layout(d, s, r, zero)
    p = 4 * h * (2 * d * s + s + d)
    params = 4 * h * layout["numel"] if zero else p
    grads = 4 * h * layout["numel"] if r > 1 else 0
    # Ordinary torch fused Adam: four FP32 step scalars per hook.
    moments = (8 * h * layout["owned"][v["dp_rank"]] if zero else 2 * p + 16 * h)
    return dict(parameters=params, gradients=grads, adam_state=moments,
                inputs=4 * h * v["b"] * v["a"] * d,
                feature_statistics=8 * h * v["f"])


def native_terms(cfg, rank=0):
    if "execution" in cfg or any(cfg.get(k) is not None for k in (
        "main_representation", "aux_representation", "sae_tp_overlap"
    )):
        raise ValueError("Decoupled execution requires a new allocated calibration; the legacy model cannot predict it")
    v = dimensions(cfg, rank)
    d, s, b, t, h, w = (v[k] for k in ("d", "s", "b", "t", "hooks", "wave"))
    if d != 4096 or cfg.get("dead", 2048) != 2048:
        raise ValueError("Calibration domain requires D=4096, K=128, dead=2048")
    if cfg["backend"] not in ("legacy", "sharded_dense", "sharded_ragged"):
        raise ValueError("Unsupported representation")
    x, z, weight = (4 * b * d / GIB, 4 * b * s / GIB, 4 * d * s / GIB)
    aux = 4 * b * (2048 / t) / GIB
    # Expected balanced winner ownership, not a hard local nnz bound.
    entries = b * 128 / t
    ragged = (20 * entries + 8 * (b + 1)) / GIB
    selection = (16 * b * min(128, s) + 8 * b) / GIB
    pending = (w - 1) * (4 * z + 2 * x + selection)
    full_delta = 4 * b * (v["f"] - s) / GIB
    ga_carry = (h * 4 * (2 * d * s + s + d) / GIB
                if v["r"] == 1 and v["a"] > 1 else 0)
    return dict(**v, x=x, z=z, weight=weight, aux=aux, pending=pending,
                full_delta=full_delta, ragged=ragged, ga_carry=ga_carry,
                resident_features=[1., weight if v["r"] > 1 else 0.],
                compute_features=[x, z, weight, aux])


def _dot(a, b):
    if len(a) != len(b):
        raise ValueError("Calibration coefficient shape mismatch")
    return sum(x * y for x, y in zip(a, b, strict=True))


def predict_native(cfg, calibration, rank=0):
    if calibration.get("schema") == "native_interpolation_v3":
        try:
            from .execution_time_model import predict_native as predict_measured
        except ImportError:
            from execution_time_model import predict_native as predict_measured
        result = predict_measured(cfg, calibration)
        if not 0 <= rank < len(result["ranks"]):
            raise ValueError("Invalid rank")
        memory = result["ranks"][rank]
        return dict(peak_allocated=memory["peak_allocated_bytes"],
                    resident_allocated=memory["resident_allocated_bytes"],
                    structural_payload=memory["structural_payload"],
                    effective_live_hooks=dimensions(cfg, rank)["wave"],
                    corners=result["corners"], warnings=result["warnings"])
    if calibration["version"] != VERSION:
        raise ValueError("Calibration version mismatch")
    v = native_terms(cfg, rank)
    states = state_payload(cfg, rank)
    resident = sum(states.values()) / GIB + _dot(
        v["resident_features"], calibration["native_resident"])
    common = resident + v["pending"] + v["ga_carry"] + _dot(
        v["compute_features"], calibration["native_compute"])
    correction = 0.
    if cfg["backend"] == "legacy":
        correction = _dot([v["full_delta"], (v["wave"] - 1) * v["full_delta"]],
                          calibration["full_extra"])
    elif cfg["backend"] == "sharded_ragged":
        key = cfg["main_compute"] + "/" + cfg["aux_compute"]
        # Remove two dense latent copies per live main, add weight layout and
        # packed values/metadata. The remaining envelope uses branch calibration.
        correction = v["wave"] * (-2 * v["z"] + v["weight"] + 2 * v["ragged"])
        if cfg["aux_compute"] != "compact_dense":
            correction += 16 * v["b"] * (2048 / v["t"]) / GIB
        correction += calibration["ragged_extra"][key] * v["weight"]
    elif cfg["aux"] == "local_dense":
        correction += calibration["aux_local_extra"] * v["z"]
    return dict(peak_allocated=round((common + correction) * GIB),
                resident_allocated=round(resident * GIB), structural_payload=states,
                effective_live_hooks=v["wave"])


def predict_online(cfg, calibration):
    # The online calibration is intentionally restricted to the measured
    # Llama-3.1-8B / SAE TP2DP2, F8192, H3, G4096, context256 workload.
    if (cfg["stp"], cfg["sdp"], cfg["spp"], cfg["h"]) != (2, 2, 1, 3):
        raise ValueError("Online SAE topology outside calibration domain")
    if cfg["global_prompts"] % cfg["vdp"] or cfg["buffers"] < 8:
        raise ValueError("Invalid prompt split or insufficient mixing capacity")
    tp = cfg["vtp"]
    weights = calibration["online_weights"][str(tp)]
    tokens = (cfg["global_prompts"] // cfg["vdp"]) * 256
    kv = (math.ceil(tokens / 16) + 1) * 16 * 32 * (8 // tp) * 256 * 2
    buffer_extra = (cfg["buffers"] - 16) * 256 * 3 * 4096 * 4 / GIB
    capture_extra = (cfg["global_prompts"] - 16) * 256 * 3 * 4096 * 4 / (2 * GIB)
    envelope = _dot([1., buffer_extra, capture_extra], calibration["online_envelope"])
    return dict(peak_allocated=round(weights + kv + envelope * GIB),
                llm_weight_and_buffers=weights, kv_bytes=kv)
