"""Predict per-rank GPU peak memory for multi-hook TopK SAE training **without
actually running the training**.

Three quantities are predicted and reported separately, because they answer
different questions:

  peak_allocated_mb  — bytes held by live tensors at the worst phase (matches
                       ``torch.cuda.max_memory_allocated()``).
  peak_reserved_mb   — PyTorch caching allocator pool size, which includes
                       fragmentation that survives within a step (matches
                       ``torch.cuda.max_memory_reserved()``).
  driver_used_mb     — what nvidia-smi sees: reserved + CUDA context +
                       cuBLAS/cuDNN workspaces + NCCL bootstrap buffers.
                       This is the only quantity that is directly comparable
                       to total GPU VRAM for a fits/overflows verdict.

Driver-peak is built up *per phase* and then maxed:

  phase_driver_proxy_p = peak_alloc_p + runtime_residual
                       (peak_alloc_p is phase-specific; runtime_residual is the
                        phase-internal driver constant — CUDA ctx + NCCL
                        bootstrap — that empty_cache cannot reclaim.)

  step_peak_driver_proxy = max_p phase_driver_proxy_p
  step_peak_driver       = step_peak_driver_proxy + pool_fragmentation
                       (pool_fragmentation is the cross-phase allocator pool
                        watermark tax = peak_reserved − max_p peak_alloc_p.)

This matches the empirical decomposition in
``results/memory_model/sae_phase_v4_freed/figures/`` where
``phase_driver_proxy = peak_alloc + (driver_freed − allocated)`` per phase
and ``driver_watermark = max_p phase_driver_proxy + pool_tax``.

Calibration constants (``_POOL_FRAG_*``, ``_RUNTIME_*``) are fitted from the
v4 TP1/TP2 reference runs and checked by ``tests/test_predict_sae_memory.py``.

Phase-level peaks (per rank, peak_allocated layer):

  forward_peak    = persistent
  backward_peak   = persistent + backward_extra
  optimizer_peak  = persistent + optimizer_extra
  step_peak_alloc = max of the three above
  step_peak_reserved = step_peak_alloc + pool_fragmentation
  step_peak_driver   = max_p (peak_alloc_p + runtime_residual) + pool_fragmentation

Component model (TP shards W_enc + W_dec + b_enc; b_dec replicated):

  persistent       = hooks · (params + grads + 2·params/adam_div)
                   + 2·hooks·B·(2·d_in + 2·d_sae)·dtype  # retained outputs
                   + hooks·B·d_in·dtype                  # scaled batch
                   + slack
  backward_extra   = 2·hooks·B·(d_sae/tp)·dtype + 64 MB
  optimizer_extra  = hooks · params_shard               # Adam foreach default

This omits vLLM and the activation buffer / streaming queue — see
``sae_lens.autoconfig.memory_model.estimate_candidate_memory`` for the
full producer+consumer model. Use this script when activations come from a
cached dir or arrive over the network and only the SAE side matters.

Usage::

    python3 scripts/predict_sae_memory.py \\
        --d-in 4096 --d-sae 65536 --batch 2048 --k 128 \\
        --dtype float32 --hooks 2 --tp 2 --dp-size 1 \\
        --gpu-vram 46068
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

_DTYPE_BYTES = {"float32": 4, "fp32": 4, "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2}
MB = 1024**2

# Cross-phase pool fragmentation tax: peak_reserved - max_p peak_alloc_p,
# observed empirically on the v4 reference runs (fp32, B=2048, d_in=4096,
# hooks=2, d_sae=65536):
#   TP=1: 27146 - 24852 = 2294 MB → 9.2 % of max peak_allocated
#   TP=2: 15740 - 14611 = 1129 MB → 7.7 %
# Modeled as a fraction-of-max-peak-allocated, with a floor for tiny configs.
_POOL_FRAG_FRACTION = 0.095
_POOL_FRAG_FLOOR_MB = 800.0

# Runtime residual = driver_freed_mb - allocated_mb, observed inside a single
# phase (constant within a phase: CUDA ctx + cuBLAS/cuDNN + NCCL bootstrap +
# allocator small-block tax that empty_cache cannot reclaim).
# v4 reference (median over stable phases): TP=1 = 466 MB, TP=2 = 727 MB.
# Decomposed as base + per-extra-NCCL-peer; we bias up slightly so the
# predicted proxy overshoots at both TP=1 and TP=2 (fits-verdict must be
# conservative).
_RUNTIME_BASE_MB = 470.0
_RUNTIME_PER_TP_RANK_NCCL_MB = 280.0

# Forward workspace: intra-forward transients on top of params + adam + the
# one retained set + the new current set.
# Empirical fit on the 12-cell forward_scan_260528 grid (d_sae × batch ×
# hooks × tp): residual = 2.5 * B * (d_sae/tp) * db * hooks / MB + 200 MB.
# Max abs error 273 MB at d_sae=65536, tp=2; mean 81 MB. The 2.5× slope
# captures the simultaneous live state during forward of:
#   1× hidden_pre after allgather, 1× feature_acts (TopK output), and a
#   ~0.5× cuBLAS/TopK kernel workspace on top of those.
_FORWARD_WORKSPACE_COEF = 2.5
_FORWARD_WORKSPACE_SLACK_MB = 200.0


@dataclass
class PhasePrediction:
    """Per-phase peaks for a single SAE rank."""

    name: str
    peak_allocated_mb: float
    runtime_residual_mb: float

    @property
    def driver_proxy_mb(self) -> float:
        """Driver bytes if only this phase happened: peak_alloc + runtime."""
        return self.peak_allocated_mb + self.runtime_residual_mb


@dataclass
class Prediction:
    persistent_mb: float
    forward_peak_mb: float
    backward_peak_mb: float
    optimizer_peak_mb: float

    # Per-phase decomposition (input to max_p).
    phases: list[PhasePrediction] = field(default_factory=list)

    # Layered peaks for capacity planning.
    step_peak_allocated_mb: float = 0.0
    step_peak_reserved_mb: float = 0.0
    step_peak_driver_proxy_mb: float = 0.0  # max_p phase.driver_proxy_mb
    step_peak_driver_mb: float = 0.0        # proxy + pool_fragmentation

    components: dict[str, float] = field(default_factory=dict)
    overhead: dict[str, float] = field(default_factory=dict)

    @property
    def step_peak_mb(self) -> float:
        """Backwards-compat alias: peak_allocated layer."""
        return self.step_peak_allocated_mb

    @property
    def binding_phase(self) -> str:
        """Phase whose driver proxy determines the run-wide driver peak."""
        return max(self.phases, key=lambda ph: ph.driver_proxy_mb).name


def predict(
    *,
    d_in: int,
    d_sae: int,
    batch_tokens: int,
    k: int,  # noqa: ARG001 — kept for API; topk indices are negligible vs activations
    dtype: str,
    hooks: int,
    tp: int,
    dp_size: int = 1,
    dp_mode: str = "ddp",
    backward_slack_mb: float = 64.0,
    trainer_buf_mb: float = 2.0,
) -> Prediction:
    """All sizes in MB."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    db = _DTYPE_BYTES[dtype]
    assert d_sae % tp == 0, f"d_sae={d_sae} must be divisible by tp={tp}"

    # Per-hook persistent components (each on the local rank).
    full_param_elems = 2 * d_in * d_sae + d_sae + d_in
    full_param_mb = full_param_elems * db / MB

    # TP shards W_enc + W_dec + b_enc; b_dec replicated.
    sharded_param_elems_per_hook = (
        2 * d_in * (d_sae // tp)  # W_enc + W_dec
        + (d_sae // tp)           # b_enc
        + d_in                    # b_dec (replicated)
    )
    params_per_hook_mb = sharded_param_elems_per_hook * db / MB
    grads_per_hook_mb = params_per_hook_mb
    # Adam exp_avg + exp_avg_sq, in PARAM dtype (torch.zeros_like(p)).
    # FSDP shards adam over dp_size; DDP/manual replicates.
    adam_divisor = dp_size if dp_mode == "fsdp" else 1
    adam_per_hook_mb = 2 * params_per_hook_mb / adam_divisor

    # The trainer holds two copies of "outputs" simultaneously (retained from
    # previous step + current). Each copy contains, per hook:
    #   sae_in + sae_out + hidden_pre_full + feature_acts_full
    # = B * (2*d_in + 2*d_sae) * dtype.
    # TP-invariant: hidden_pre and feature_acts are full after allgather.
    one_outputs_set_per_hook = batch_tokens * (2 * d_in + 2 * d_sae) * db / MB
    retained_outputs_mb = 2.0 * one_outputs_set_per_hook * hooks

    # Scaled activations for this step (one per hook; raw is freed after scaling).
    scaled_batch_mb = batch_tokens * d_in * db / MB * hooks

    persistent = (
        params_per_hook_mb * hooks
        + grads_per_hook_mb * hooks
        + adam_per_hook_mb * hooks
        + retained_outputs_mb
        + scaled_batch_mb
        + trainer_buf_mb
    )

    # Backward saved tensors: TopK keeps hidden_pre_local for backward (B * d_sae/tp).
    # Empirical fit: 2 of these per hook + small kernel slack.
    backward_extra = 2.0 * batch_tokens * (d_sae // tp) * db / MB * hooks + backward_slack_mb

    # Adam foreach (default) allocates one extra param-shaped buffer per param
    # group during step(). Empirical fit: 1.0 × params_shard, summed over hooks.
    optimizer_extra = 1.0 * params_per_hook_mb * hooks

    # DDP gradient bucket overhead (small, but noticeable at TP=1 + DP>1).
    if dp_mode == "ddp" and dp_size > 1:
        persistent += params_per_hook_mb * hooks * 0.05  # ~5 % overhead
    if dp_mode == "fsdp" and dp_size > 1:
        # FSDP backward needs unsharded params transiently; capped at 2×.
        backward_extra += 2.0 * params_per_hook_mb * dp_size * hooks

    # Forward phase has different live tensors than backward/optimizer.
    # MultiSAETrainer sequence is:
    #   data_fetch → scale_to_device → zero_grad_start → forward_all → ...
    # so at after_forward_all:
    #   - grads = 0 (just zeroed)
    #   - retained_outputs (1 set / hook, from previous step's current)
    #   - current_outputs (this step's, but sae_in shares the scaled_batch
    #     allocation, so the new bytes are sae_out + hidden_pre + feature_acts
    #     = B*(d_in + 2*d_sae)*db per hook)
    #   - intra-forward workspace: TopK kernel scratch + cuBLAS workspace +
    #     allreduce buffer at TP>1, all roughly proportional to B*(d_sae/tp).
    #     Empirically calibrated to 2.5 × B*(d_sae/tp)*db*hooks + 200 MB
    #     across the 12-cell forward_scan grid (max 273 MB error at d65536_tp2).
    one_retained_per_hook_mb = batch_tokens * (2 * d_in + 2 * d_sae) * db / MB
    current_outputs_minus_sae_in_per_hook_mb = (
        batch_tokens * (d_in + 2 * d_sae) * db / MB
    )
    forward_workspace_mb = (
        _FORWARD_WORKSPACE_COEF * batch_tokens * (d_sae // tp) * db * hooks / MB
        + _FORWARD_WORKSPACE_SLACK_MB
    )
    forward_peak = (
        params_per_hook_mb * hooks
        + adam_per_hook_mb * hooks
        + one_retained_per_hook_mb * hooks
        + current_outputs_minus_sae_in_per_hook_mb * hooks
        + scaled_batch_mb
        + trainer_buf_mb
        + forward_workspace_mb
    )
    backward_peak = persistent + backward_extra
    optimizer_peak = persistent + optimizer_extra
    step_peak_alloc = max(forward_peak, backward_peak, optimizer_peak)

    # Per-phase driver proxy. runtime_residual is phase-internal (CUDA ctx +
    # NCCL bootstrap + small-block tax). It is virtually phase-invariant on
    # the reference runs, so we apply the same value to each phase — but the
    # decomposition is what would change if a future phase (e.g. data_fetch
    # with pinned-host transfer) needed a phase-specific residual.
    nccl_residual = _RUNTIME_PER_TP_RANK_NCCL_MB * (tp - 1)
    runtime_residual = _RUNTIME_BASE_MB + nccl_residual
    phases = [
        PhasePrediction("forward",   forward_peak,   runtime_residual),
        PhasePrediction("backward",  backward_peak,  runtime_residual),
        PhasePrediction("optimizer", optimizer_peak, runtime_residual),
    ]
    step_peak_driver_proxy = max(ph.driver_proxy_mb for ph in phases)

    # Cross-phase pool fragmentation: the allocator pool grows to accommodate
    # the worst transient, and never shrinks within a step. Modeled as a flat
    # fraction of the max peak_allocated, with a floor.
    pool_fragmentation = max(
        _POOL_FRAG_FLOOR_MB, _POOL_FRAG_FRACTION * step_peak_alloc
    )
    step_peak_reserved = step_peak_alloc + pool_fragmentation
    step_peak_driver = step_peak_driver_proxy + pool_fragmentation

    return Prediction(
        persistent_mb=persistent,
        forward_peak_mb=forward_peak,
        backward_peak_mb=backward_peak,
        optimizer_peak_mb=optimizer_peak,
        phases=phases,
        step_peak_allocated_mb=step_peak_alloc,
        step_peak_reserved_mb=step_peak_reserved,
        step_peak_driver_proxy_mb=step_peak_driver_proxy,
        step_peak_driver_mb=step_peak_driver,
        components={
            "params_per_rank": params_per_hook_mb * hooks,
            "grads_per_rank": grads_per_hook_mb * hooks,
            "adam_per_rank": adam_per_hook_mb * hooks,
            "retained_outputs": retained_outputs_mb,
            "scaled_batch": scaled_batch_mb,
            "backward_extra": backward_extra,
            "optimizer_extra": optimizer_extra,
            "full_param_per_hook_mb": full_param_mb,
            "forward_current_outputs_minus_sae_in":
                current_outputs_minus_sae_in_per_hook_mb * hooks,
            "forward_workspace": forward_workspace_mb,
        },
        overhead={
            "pool_fragmentation": pool_fragmentation,
            "runtime_residual": runtime_residual,
            "nccl_residual": nccl_residual,
            # Backwards-compat aliases for older callers / tests.
            "fragment_slack": pool_fragmentation,
            "cuda_context_slack": runtime_residual,
            "nccl_overhead": nccl_residual,
        },
    )


def _format_table(p: Prediction, gpu_vram_mb: float | None) -> str:
    lines = [
        "Persistent components (MB):",
        f"  params           {p.components['params_per_rank']:9.1f}",
        f"  grads            {p.components['grads_per_rank']:9.1f}",
        f"  adam             {p.components['adam_per_rank']:9.1f}",
        f"  retained_outputs {p.components['retained_outputs']:9.1f}",
        f"  scaled_batch     {p.components['scaled_batch']:9.1f}",
        "  -----------------------",
        f"  persistent       {p.persistent_mb:9.1f}",
        "",
        "Per-phase peaks (MB):",
        f"  {'phase':<10}  {'peak_alloc':>10}  {'+runtime':>9}  {'driver_proxy':>13}",
    ]
    for ph in p.phases:
        marker = "  ←" if ph.name == p.binding_phase else ""
        lines.append(
            f"  {ph.name:<10}  {ph.peak_allocated_mb:10.1f}  "
            f"{ph.runtime_residual_mb:9.1f}  {ph.driver_proxy_mb:13.1f}{marker}"
        )
    lines += [
        "",
        "Step peak — three layers (MB):",
        f"  peak_allocated   {p.step_peak_allocated_mb:9.1f}    "
        f"max over phases (max_memory_allocated)",
        f"  peak_reserved    {p.step_peak_reserved_mb:9.1f}    "
        f"+ {p.overhead['pool_fragmentation']:.0f} pool fragmentation",
        f"  driver_proxy     {p.step_peak_driver_proxy_mb:9.1f}    "
        f"max_p (peak_alloc + runtime_residual)",
        f"  driver_used      {p.step_peak_driver_mb:9.1f}    "
        f"+ {p.overhead['pool_fragmentation']:.0f} pool tax  ← compare to GPU VRAM",
    ]
    if gpu_vram_mb is not None:
        # Capacity verdict uses driver_used — this is what nvidia-smi sees and
        # what determines whether the run actually fits on the card.
        margin = gpu_vram_mb - p.step_peak_driver_mb
        verdict = "FITS" if margin > 0 else "OVERFLOWS"
        pct = p.step_peak_driver_mb / gpu_vram_mb * 100
        lines.append("")
        lines.append(
            f"  GPU VRAM         {gpu_vram_mb:9.1f}  → {verdict}  "
            f"(driver_used margin {margin:+.0f} MB, {pct:.1f}% used)"
        )
        # Also show the optimistic verdict so users see why they should not
        # rely on peak_allocated alone.
        opt_margin = gpu_vram_mb - p.step_peak_allocated_mb
        if (margin <= 0) != (opt_margin <= 0):
            lines.append(
                "  WARNING          peak_allocated alone would predict "
                f"{'FITS' if opt_margin > 0 else 'OVERFLOWS'}; "
                "ignore that and trust driver_used."
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-in", type=int, required=True)
    ap.add_argument("--d-sae", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True, help="train_batch_size_tokens")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--dtype", choices=list(_DTYPE_BYTES), default="float32")
    ap.add_argument("--hooks", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1, help="sae_tp_size")
    ap.add_argument("--dp-size", type=int, default=1, help="sae_dp_size")
    ap.add_argument("--dp-mode", choices=["ddp", "fsdp", "manual"], default="ddp")
    ap.add_argument("--gpu-vram", type=float, default=None, help="per-rank VRAM in MB; report fits/overflows")
    args = ap.parse_args()

    p = predict(
        d_in=args.d_in,
        d_sae=args.d_sae,
        batch_tokens=args.batch,
        k=args.k,
        dtype=args.dtype,
        hooks=args.hooks,
        tp=args.tp,
        dp_size=args.dp_size,
        dp_mode=args.dp_mode,
    )
    print(
        f"Config: d_in={args.d_in} d_sae={args.d_sae} B={args.batch} k={args.k} "
        f"{args.dtype} hooks={args.hooks} tp={args.tp} dp={args.dp_size}/{args.dp_mode}"
    )
    print()
    print(_format_table(p, args.gpu_vram))


if __name__ == "__main__":
    main()
