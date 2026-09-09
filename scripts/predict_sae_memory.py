"""Predict per-rank GPU peak memory for multi-hook TopK SAE training.

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
                         + foreign_cuda_contexts
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

Current component model (TP shards W_enc + W_dec + b_enc; b_dec replicated):

  B_local          = ceil(global_train_batch / sae_dp)
  train_persistent = local_hooks · (params + grads + 2·params/adam_div)
                   + local_hooks·B_local·(2·d_in + 2·d_sae)·dtype
                   + slack
  backward_extra   = 4·local_hooks·B_local·(d_sae/tp)·dtype + 64 MB
  optimizer_extra  = local_hooks · params_shard          # Adam foreach only

The previous step's ``TrainStepOutput`` is absent from the default model
because the trainer releases it before the next data fetch. The
``--legacy-retain-previous-outputs`` switch exists only to compare historical
profiles made before that lifecycle fix.

``sae_only`` mode models cached/split-role SAE ranks. ``ordinary`` mode also
models the co-located vLLM producer, activation mixing buffer, and PP routing.
The two hook counts are deliberately separate: ``total_hooks`` is captured by
the producer, while ``local_hooks`` is assigned to the fullest SAE PP rank.

Capacity checks include allocator pool fragmentation, the local CUDA/NCCL
runtime, and foreign CUDA contexts separately. They are not tensor memory and
must not be hidden in an unexplained calibration residual.

Usage::

    python3 scripts/predict_sae_memory.py \\
        --mode ordinary --d-in 4096 --d-sae 32768 --batch 4096 \\
        --dtype float32 --total-hooks 6 --local-hooks 2 \\
        --tp 1 --dp-size 1 --vllm-tp 4 --optimizer fused \\
        --gpu-vram 32607
"""

from __future__ import annotations

import argparse
import math
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

# Current combined-forward implementation: all local hooks retain their
# autograd-saved tensors until the one combined backward. The H=6, PP=4,
# B=4096 run measured about 5.1 GiB beyond the output tensors themselves.
_CURRENT_FORWARD_WORKSPACE_COEF = 5.0
_CURRENT_FORWARD_WORKSPACE_SLACK_MB = 128.0
_CURRENT_BACKWARD_WORKSPACE_COEF = 4.0

# Ordinary co-located runs create larger, differently shaped allocations in
# vLLM capture, dtype conversion, routing, and SAE phases. Their allocator
# watermark is materially higher than in SAE-only reference runs.
_ORDINARY_POOL_FRAG_FRACTION = 0.25
_ORDINARY_TORCH_RUNTIME_MB = 1100.0
_FOREIGN_CONTEXT_MB = 500.0
_VLLM_TOTAL_WEIGHT_GIB = 14.9889
_VLLM_INTERMEDIATE_SIZE = 14_336
_VLLM_NUM_LAYERS = 32
_VLLM_NUM_KV_HEADS = 8
_VLLM_HEAD_DIM = 128
_VLLM_BLOCK_SIZE = 16
_VLLM_MAX_BATCHED_TOKENS = 8192


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
    step_peak_driver_mb: float = 0.0  # proxy + pool + foreign contexts

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


def _vllm_resident_mb(tp: int, weight_gib: float) -> float:
    blocks = (
        (_VLLM_MAX_BATCHED_TOKENS + _VLLM_BLOCK_SIZE - 1) // _VLLM_BLOCK_SIZE
        + 1
    )
    kv_heads_per_rank = _VLLM_NUM_KV_HEADS / tp
    kv_mb = (
        _VLLM_NUM_LAYERS
        * blocks
        * 2
        * _VLLM_BLOCK_SIZE
        * kv_heads_per_rank
        * _VLLM_HEAD_DIM
        * 2
        / MB
    )
    return weight_gib * 1024 / tp + kv_mb


def predict(
    *,
    d_in: int,
    d_sae: int,
    batch_tokens: int,
    k: int,  # noqa: ARG001 - kept for API; sparse TopK state is comparatively small
    dtype: str,
    hooks: int | None = None,
    tp: int,
    dp_size: int = 1,
    dp_mode: str = "ddp",
    local_hooks: int | None = None,
    total_hooks: int | None = None,
    mode: str = "sae_only",
    optimizer_impl: str = "foreach",
    retain_previous_outputs: bool = False,
    vllm_tp: int = 1,
    store_batch_size_prompts: int = 8,
    context_size: int = 2048,
    mixing_fraction: float = 0.5,
    vllm_total_weight_gib: float = _VLLM_TOTAL_WEIGHT_GIB,
    vllm_activation_bytes: int = 2,
    foreign_cuda_contexts: int | None = None,
    foreign_context_mb: float = _FOREIGN_CONTEXT_MB,
    backward_slack_mb: float = 64.0,
    trainer_buf_mb: float = 2.0,
) -> Prediction:
    """Return phase peaks in MiB for the fullest SAE PP rank.

    ``hooks`` remains a compatibility alias for ``local_hooks``. Ordinary
    mode requires an explicit ``total_hooks`` whenever PP partitions hooks.
    ``batch_tokens`` is the global SAE batch; activation terms use the fullest
    DP rank's ``ceil(batch_tokens / dp_size)`` rows.
    """
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    if mode not in ("sae_only", "ordinary"):
        raise ValueError("mode must be 'sae_only' or 'ordinary'")
    if optimizer_impl not in ("fused", "foreach", "for_loop"):
        raise ValueError("optimizer_impl must be fused, foreach, or for_loop")
    if dp_mode not in ("ddp", "fsdp", "manual"):
        raise ValueError("dp_mode must be ddp, fsdp, or manual")
    if not 0 <= mixing_fraction <= 1:
        raise ValueError("mixing_fraction must be in [0, 1]")
    if min(d_in, d_sae, batch_tokens, tp, dp_size) < 1:
        raise ValueError("dimensions, batch_tokens, tp, and dp_size must be >= 1")
    if mode == "ordinary" and min(vllm_tp, store_batch_size_prompts, context_size) < 1:
        raise ValueError(
            "ordinary mode requires vllm_tp, store_batch_size_prompts, and "
            "context_size >= 1"
        )

    local_hooks = local_hooks if local_hooks is not None else hooks
    if local_hooks is None:
        local_hooks = 1
    total_hooks = local_hooks if total_hooks is None else total_hooks
    if local_hooks < 1 or total_hooks < local_hooks:
        raise ValueError("require total_hooks >= local_hooks >= 1")
    if d_sae % tp:
        raise ValueError(f"d_sae={d_sae} must be divisible by tp={tp}")

    db = _DTYPE_BYTES[dtype]
    local_batch_tokens = math.ceil(batch_tokens / dp_size)
    full_param_elems = 2 * d_in * d_sae + d_sae + d_in
    full_param_mb = full_param_elems * db / MB
    sharded_param_elems_per_hook = (
        2 * d_in * (d_sae // tp)
        + (d_sae // tp)
        + d_in
    )
    params_per_hook_mb = sharded_param_elems_per_hook * db / MB
    grads_per_hook_mb = params_per_hook_mb
    adam_divisor = dp_size if dp_mode == "fsdp" else 1
    adam_per_hook_mb = 2 * params_per_hook_mb / adam_divisor

    params_mb = params_per_hook_mb * local_hooks
    grads_mb = grads_per_hook_mb * local_hooks
    adam_mb = adam_per_hook_mb * local_hooks
    scaled_batch_mb = local_batch_tokens * d_in * db / MB * local_hooks
    one_output_per_hook_mb = (
        local_batch_tokens * (2 * d_in + 2 * d_sae) * db / MB
    )
    current_outputs_mb = one_output_per_hook_mb * local_hooks
    current_outputs_minus_sae_in_mb = current_outputs_mb - scaled_batch_mb
    activation_unit_mb = (
        local_batch_tokens * (d_sae // tp) * db / MB * local_hooks
    )

    ddp_bucket_mb = 0.0
    if dp_mode == "ddp" and dp_size > 1:
        ddp_bucket_mb = params_mb * 0.05

    if retain_previous_outputs:
        # Compatibility mode for old profiles taken before completed outputs
        # were released ahead of the next data fetch.
        retained_outputs_mb = 2.0 * current_outputs_mb
        persistent = (
            params_mb
            + grads_mb
            + adam_mb
            + retained_outputs_mb
            + scaled_batch_mb
            + trainer_buf_mb
            + ddp_bucket_mb
        )
        forward_workspace_mb = (
            _FORWARD_WORKSPACE_COEF * activation_unit_mb
            + _FORWARD_WORKSPACE_SLACK_MB
        )
        forward_peak = (
            params_mb
            + adam_mb
            + current_outputs_mb
            + current_outputs_minus_sae_in_mb
            + scaled_batch_mb
            + trainer_buf_mb
            + ddp_bucket_mb
            + forward_workspace_mb
        )
        backward_extra = 2.0 * activation_unit_mb + backward_slack_mb
    else:
        retained_outputs_mb = 0.0
        persistent = (
            params_mb
            + grads_mb
            + adam_mb
            + current_outputs_mb
            + trainer_buf_mb
            + ddp_bucket_mb
        )
        forward_workspace_mb = (
            _CURRENT_FORWARD_WORKSPACE_COEF * activation_unit_mb
            + _CURRENT_FORWARD_WORKSPACE_SLACK_MB
        )
        forward_peak = (
            params_mb
            + adam_mb
            + scaled_batch_mb
            + current_outputs_minus_sae_in_mb
            + trainer_buf_mb
            + ddp_bucket_mb
            + forward_workspace_mb
        )
        backward_extra = (
            _CURRENT_BACKWARD_WORKSPACE_COEF * activation_unit_mb
            + backward_slack_mb
        )

    if dp_mode == "fsdp" and dp_size > 1:
        backward_extra += 2.0 * params_per_hook_mb * dp_size * local_hooks

    optimizer_extra = params_mb if optimizer_impl == "foreach" else 0.0
    backward_peak = persistent + backward_extra
    optimizer_peak = persistent + optimizer_extra

    ordinary_resident_mb = 0.0
    vllm_resident_mb = 0.0
    vllm_workspace_mb = 0.0
    producer_transient_mb = 0.0
    producer_capture_mb = 0.0
    producer_raw_capture_mb = 0.0
    endpoint_pack_mb = 0.0
    mixing_buffer_mb = 0.0
    data_fetch_peak = 0.0
    if mode == "ordinary":
        store_tokens = store_batch_size_prompts * context_size
        producer_capture_mb = total_hooks * store_tokens * d_in * db / MB
        producer_raw_capture_mb = (
            total_hooks * store_tokens * d_in * vllm_activation_bytes / MB
        )
        endpoint_pack_mb = local_hooks * store_tokens * d_in * db / MB

        n_batches = max(
            2,
            (local_batch_tokens + context_size - 1) // context_size,
        )
        retained_batches = math.ceil(n_batches * mixing_fraction)
        peak_buffer_batches = retained_batches + 1
        mixing_buffer_mb = (
            local_hooks
            * peak_buffer_batches
            * store_tokens
            * d_in
            * db
            / MB
        )
        vllm_resident_mb = _vllm_resident_mb(vllm_tp, vllm_total_weight_gib)
        effective_tokens = min(store_tokens, _VLLM_MAX_BATCHED_TOKENS)
        coefficient = 2.0 + 3.0 * (_VLLM_INTERMEDIATE_SIZE / d_in) / vllm_tp
        vllm_workspace_mb = (
            coefficient * effective_tokens * d_in * vllm_activation_bytes / MB
        )
        producer_transient_mb = max(
            producer_raw_capture_mb + vllm_workspace_mb,
            producer_raw_capture_mb + producer_capture_mb,
            producer_capture_mb + endpoint_pack_mb,
        )
        ordinary_resident_mb = (
            vllm_resident_mb
            + mixing_buffer_mb
            + _ORDINARY_TORCH_RUNTIME_MB
        )
        data_fetch_peak = (
            params_mb
            + grads_mb
            + adam_mb
            + trainer_buf_mb
            + ddp_bucket_mb
            + ordinary_resident_mb
            + producer_transient_mb
        )
        persistent += ordinary_resident_mb
        forward_peak += ordinary_resident_mb
        backward_peak += ordinary_resident_mb
        optimizer_peak += ordinary_resident_mb

    nccl_peers = max(0, tp - 1) + max(0, dp_size - 1)
    if mode == "ordinary":
        nccl_peers += max(0, vllm_tp - 1)
    nccl_residual = _RUNTIME_PER_TP_RANK_NCCL_MB * nccl_peers
    runtime_residual = _RUNTIME_BASE_MB + nccl_residual
    phases = []
    if mode == "ordinary":
        phases.append(PhasePrediction("data_fetch", data_fetch_peak, runtime_residual))
    phases.extend(
        [
            PhasePrediction("forward", forward_peak, runtime_residual),
            PhasePrediction("backward", backward_peak, runtime_residual),
            PhasePrediction("optimizer", optimizer_peak, runtime_residual),
        ]
    )
    step_peak_alloc = max(ph.peak_allocated_mb for ph in phases)
    step_peak_driver_proxy = max(ph.driver_proxy_mb for ph in phases)

    pool_fraction = (
        _ORDINARY_POOL_FRAG_FRACTION
        if mode == "ordinary"
        else _POOL_FRAG_FRACTION
    )
    pool_fragmentation = max(_POOL_FRAG_FLOOR_MB, pool_fraction * step_peak_alloc)
    step_peak_reserved = step_peak_alloc + pool_fragmentation
    if foreign_cuda_contexts is None:
        foreign_cuda_contexts = max(0, vllm_tp - 1) if mode == "ordinary" else 0
    foreign_contexts_mb = foreign_cuda_contexts * foreign_context_mb
    step_peak_driver = (
        step_peak_driver_proxy
        + pool_fragmentation
        + foreign_contexts_mb
    )

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
            "total_hooks": float(total_hooks),
            "local_hooks": float(local_hooks),
            "global_batch_tokens": float(batch_tokens),
            "max_local_batch_tokens": float(local_batch_tokens),
            "params_per_rank": params_mb,
            "grads_per_rank": grads_mb,
            "adam_per_rank": adam_mb,
            "retained_outputs": retained_outputs_mb,
            "current_outputs": current_outputs_mb,
            "scaled_batch": scaled_batch_mb,
            "backward_extra": backward_extra,
            "optimizer_extra": optimizer_extra,
            "full_param_per_hook_mb": full_param_mb,
            "forward_current_outputs_minus_sae_in": current_outputs_minus_sae_in_mb,
            "forward_workspace": forward_workspace_mb,
            "ordinary_resident": ordinary_resident_mb,
            "vllm_resident": vllm_resident_mb,
            "ordinary_torch_runtime": (
                _ORDINARY_TORCH_RUNTIME_MB if mode == "ordinary" else 0.0
            ),
            "mixing_buffer": mixing_buffer_mb,
            "producer_capture": producer_capture_mb,
            "producer_raw_capture": producer_raw_capture_mb,
            "vllm_workspace": vllm_workspace_mb,
            "producer_transient": producer_transient_mb,
            "endpoint_pack": endpoint_pack_mb,
        },
        overhead={
            "pool_fragmentation": pool_fragmentation,
            "runtime_residual": runtime_residual,
            "nccl_residual": nccl_residual,
            "foreign_contexts": foreign_contexts_mb,
            "fragment_slack": pool_fragmentation,
            "cuda_context_slack": runtime_residual,
            "nccl_overhead": nccl_residual,
        },
    )


def _format_table(
    p: Prediction,
    gpu_vram_mb: float | None,
    allowed_fraction: float = 1.0,
) -> str:
    lines = [
        "Hook scopes:",
        f"  producer total   {p.components['total_hooks']:9.0f}",
        f"  fullest PP rank  {p.components['local_hooks']:9.0f}",
        f"  global batch     {p.components['global_batch_tokens']:9.0f}",
        f"  max local batch  {p.components['max_local_batch_tokens']:9.0f}",
        "",
        "Persistent components (MB):",
        f"  params           {p.components['params_per_rank']:9.1f}",
        f"  grads            {p.components['grads_per_rank']:9.1f}",
        f"  adam             {p.components['adam_per_rank']:9.1f}",
        f"  retained_outputs {p.components['retained_outputs']:9.1f}",
        f"  current_outputs  {p.components['current_outputs']:9.1f}",
        f"  scaled_batch     {p.components['scaled_batch']:9.1f}",
        "  -----------------------",
        f"  persistent       {p.persistent_mb:9.1f}",
    ]
    if p.components["ordinary_resident"] > 0:
        lines += [
            "",
            "Ordinary producer/routing (MB):",
            f"  vLLM resident    {p.components['vllm_resident']:9.1f}",
            f"  torch runtime    {p.components['ordinary_torch_runtime']:9.1f}",
            f"  mixing_buffer    {p.components['mixing_buffer']:9.1f}",
            f"  total capture    {p.components['producer_capture']:9.1f}",
            f"  raw capture      {p.components['producer_raw_capture']:9.1f}",
            f"  vLLM workspace   {p.components['vllm_workspace']:9.1f}",
            f"  endpoint pack    {p.components['endpoint_pack']:9.1f}",
            f"  fetch transient  {p.components['producer_transient']:9.1f}",
        ]
    lines += [
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
        "proxy + pool + foreign contexts",
        "",
        "Non-tensor overhead (MB):",
        f"  allocator pool   {p.overhead['pool_fragmentation']:9.1f}",
        f"  local CUDA/NCCL  {p.overhead['runtime_residual']:9.1f}",
        f"  foreign contexts {p.overhead['foreign_contexts']:9.1f}",
    ]
    if gpu_vram_mb is not None:
        usable_vram_mb = gpu_vram_mb * allowed_fraction
        margin = usable_vram_mb - p.step_peak_driver_mb
        verdict = "FITS" if margin > 0 else "OVERFLOWS"
        pct = p.step_peak_driver_mb / usable_vram_mb * 100
        lines.append("")
        lines.append(
            f"  usable VRAM      {usable_vram_mb:9.1f}  -> {verdict}  "
            f"({allowed_fraction:.0%} of {gpu_vram_mb:.0f} MB, "
            f"margin {margin:+.0f} MB, {pct:.1f}% used)"
        )
        # Also show the optimistic verdict so users see why they should not
        # rely on peak_allocated alone.
        opt_margin = usable_vram_mb - p.step_peak_allocated_mb
        if (margin <= 0) != (opt_margin <= 0):
            lines.append(
                "  WARNING          peak_allocated alone would predict "
                f"{'FITS' if opt_margin > 0 else 'OVERFLOWS'}; "
                "ignore that and trust driver_used."
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--d-in", type=int, required=True)
    ap.add_argument("--d-sae", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True, help="train_batch_size_tokens")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--dtype", choices=list(_DTYPE_BYTES), default="float32")
    ap.add_argument("--hooks", type=int, default=None, help="legacy alias for --local-hooks")
    ap.add_argument("--local-hooks", type=int, default=None)
    ap.add_argument("--total-hooks", type=int, default=None)
    ap.add_argument("--mode", choices=["sae_only", "ordinary"], default="sae_only")
    ap.add_argument("--tp", type=int, default=1, help="sae_tp_size")
    ap.add_argument("--dp-size", type=int, default=1, help="sae_dp_size")
    ap.add_argument("--dp-mode", choices=["ddp", "fsdp", "manual"], default="ddp")
    ap.add_argument(
        "--optimizer",
        choices=["fused", "foreach", "for_loop"],
        default="fused",
    )
    ap.add_argument("--vllm-tp", type=int, default=1)
    ap.add_argument("--store-batch-size-prompts", type=int, default=8)
    ap.add_argument("--context-size", type=int, default=2048)
    ap.add_argument("--mixing-fraction", type=float, default=0.5)
    ap.add_argument("--vllm-activation-bytes", type=int, choices=[2, 4], default=2)
    ap.add_argument("--foreign-cuda-contexts", type=int, default=None)
    ap.add_argument("--foreign-context-mb", type=float, default=_FOREIGN_CONTEXT_MB)
    ap.add_argument("--legacy-retain-previous-outputs", action="store_true")
    ap.add_argument(
        "--gpu-vram",
        type=float,
        default=None,
        help="physical per-rank VRAM in MiB",
    )
    ap.add_argument("--allowed-fraction", type=float, default=0.92)
    args = ap.parse_args()
    if not 0 < args.allowed_fraction <= 1:
        ap.error("--allowed-fraction must be in (0, 1]")

    p = predict(
        d_in=args.d_in,
        d_sae=args.d_sae,
        batch_tokens=args.batch,
        k=args.k,
        dtype=args.dtype,
        hooks=args.hooks,
        local_hooks=args.local_hooks,
        total_hooks=args.total_hooks,
        mode=args.mode,
        tp=args.tp,
        dp_size=args.dp_size,
        dp_mode=args.dp_mode,
        optimizer_impl=args.optimizer,
        retain_previous_outputs=args.legacy_retain_previous_outputs,
        vllm_tp=args.vllm_tp,
        store_batch_size_prompts=args.store_batch_size_prompts,
        context_size=args.context_size,
        mixing_fraction=args.mixing_fraction,
        vllm_activation_bytes=args.vllm_activation_bytes,
        foreign_cuda_contexts=args.foreign_cuda_contexts,
        foreign_context_mb=args.foreign_context_mb,
    )
    print(
        f"Config: d_in={args.d_in} d_sae={args.d_sae} B={args.batch} k={args.k} "
        f"{args.dtype} total_hooks={p.components['total_hooks']:.0f} "
        f"local_hooks={p.components['local_hooks']:.0f} "
        f"sae_tp={args.tp} dp={args.dp_size}/{args.dp_mode} mode={args.mode}"
    )
    print()
    print(_format_table(p, args.gpu_vram, args.allowed_fraction))


if __name__ == "__main__":
    main()
