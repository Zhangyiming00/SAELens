"""Peak-allocated memory simulator for offline-activation SAE training.

Given a single :class:`SAEPhaseMemoryConfig`, this predicts the peak
``memory_allocated`` of each of the three training phases (forward, backward,
optimizer) using the closed-form formulas validated in ``docs/memory_model_sae_*``.

Scope and conventions (per the formula spec this implements):

- Only the **offline cached-activation** training path is modeled.
- The buffer term is ``0`` (offline cache, ``buffer_size = 0``).
- Only the truly-undefined terms are ``0`` by default: ``C_forward`` /
  ``C_backward`` (unmodeled constants), ``r_sum`` and the dead-feature aux count
  ``num_dead_features`` (``F``). Everything the formula assigns an explicit value
  to — including the resident ``P + G + M + V`` weight set — is computed.
- TP shards ``W_enc`` / ``W_dec`` / ``b_enc`` along ``d_sae`` (``b_dec`` is
  replicated). DDP replicates params and splits the batch. FSDP shards params by
  ``dp`` and splits the batch. The resident weight set is phase-aware: forward has
  no gradient yet (``zero_grad`` already ran), so it holds ``P + M + V`` (3 copies),
  while backward/optimizer hold ``P + G + M + V`` (4 copies). This mirrors the FSDP
  per-phase formulas (forward = ``3HW/dp + HW``).
- The forward peak holds the **retained output** (previous hook's
  ``TrainStepOutput``, pinned by the autograd graph) and the **current output**
  simultaneously — both are summed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

INT64_BYTES = 8
# CUDA caching-allocator minimum block; the loss scalars each round up to one.
ALLOC_BLOCK_BYTES = 512
MB = 1024**2

DpMode = Literal["ddp", "fsdp"]
OptimizerImpl = Literal["fused", "foreach", "for_loop"]


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return 4
    if normalized in {"bfloat16", "bf16", "float16", "fp16"}:
        return 2
    raise ValueError(f"Unsupported dtype for memory model: {dtype!r}")


@dataclass(frozen=True)
class SAEPhaseMemoryConfig:
    """Inputs for one offline-activation SAE training memory estimate.

    All parallel degrees describe the SAE side only. ``num_hooks`` (``H``) SAEs
    share one optimizer, matching ``MultiSAETrainer``.
    """

    d_in: int
    d_sae: int
    num_hooks: int = 1
    k: int = 128
    train_batch_size_tokens: int = 2048
    dtype: str = "float32"

    tp_size: int = 1
    dp_size: int = 1
    dp_mode: DpMode = "ddp"
    max_buffer_tokens: int = 0
    gradient_as_bucket_view: bool = False
    use_scale_factor: bool = False
    optimizer_impl: OptimizerImpl = "fused"

    # Undefined / unmodeled extra terms — 0 by default (see module docstring).
    c_forward_bytes: int = 0
    c_backward_bytes: int = 0
    c_optimizer_bytes: int = 0    
    # c_optimizer_bytes_fused: int = 0
    # c_optimizer_bytes_foreach: int = 0
    # c_optimizer_bytes_forloop: int = 0
    r_sum_bytes: int = 0
    num_dead_features: int = 0

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.d_sae <= 0:
            raise ValueError("d_in and d_sae must be positive")
        if self.num_hooks < 1:
            raise ValueError("num_hooks must be >= 1")
        for name, value in (
            ("tp_size", self.tp_size),
            ("dp_size", self.dp_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.d_sae % self.tp_size != 0:
            raise ValueError(
                f"d_sae ({self.d_sae}) must be divisible by tp_size ({self.tp_size})"
            )
        if self.dp_mode not in ("ddp", "fsdp"):
            raise ValueError(f"Unknown dp_mode: {self.dp_mode!r}")
        if self.optimizer_impl not in ("fused", "foreach", "for_loop"):
            raise ValueError(f"Unknown optimizer_impl: {self.optimizer_impl!r}")


@dataclass(frozen=True)
class PhaseMemoryEstimate:
    """Peak-allocated bytes per phase for one config."""

    forward_bytes: float
    backward_bytes: float
    optimizer_bytes: float
    components: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def peak_bytes(self) -> float:
        return max(self.forward_bytes, self.backward_bytes, self.optimizer_bytes)

    def as_mb(self) -> dict[str, float]:
        return {
            "forward": self.forward_bytes / MB,
            "backward": self.backward_bytes / MB,
            "optimizer": self.optimizer_bytes / MB,
            "peak": self.peak_bytes / MB,
        }


def _weight_elems(d_in: int, d_sae: int, tp_size: int) -> int:
    """Elements of one SAE's weights (W), sharded by TP along d_sae.

    W = W_enc + W_dec + b_enc + b_dec. TP shards W_enc/W_dec/b_enc along d_sae;
    b_dec (d_in,) is replicated.
    """
    d_sae_local = d_sae // tp_size
    return 2 * d_in * d_sae_local + d_sae_local + d_in


def _p_big_elems(d_in: int, d_sae: int, tp_size: int) -> int:
    """Largest single parameter block (W_enc/W_dec), sharded along d_sae."""
    return d_in * (d_sae // tp_size)


def estimate_phase_memory(config: SAEPhaseMemoryConfig) -> PhaseMemoryEstimate:
    db = _dtype_bytes(config.dtype)
    H = config.num_hooks
    d_in = config.d_in
    d_sae = config.d_sae
    tp = config.tp_size
    dp = config.dp_size
    k = config.k
    # DDP/FSDP split the global batch across dp ranks; B is the local batch.
    B = config.train_batch_size_tokens // (
        dp if config.dp_mode in ("ddp", "fsdp") else 1
    )

    d_sae_local = d_sae // tp
    W_bytes = _weight_elems(d_in, d_sae, tp) * db  # per-hook weights W (TP-sharded)
    p_big_bytes = _p_big_elems(d_in, d_sae, tp) * db  # d_in·d_sae/tp
    topk_one_hook = B * k * db + B * k * INT64_BYTES  # values (db) + indices (int64)

    # ---- Resident weight set (P/G/M/V), per phase & parallel mode ----
    # manual/DDP: forward = 3HW (no grad yet), backward/optimizer = 4HW.
    # FSDP shards by dp and gathers full params transiently (formulas below).
    HW = H * W_bytes
    ddp_bucket = 0.0
    if config.dp > 1 and config.dp_mode == "ddp" and not config.gradient_as_bucket_view:
        ddp_bucket = HW  # +1 H W grad bucket, removed by gradient_as_bucket_view

    def _weight_mem(phase: str) -> float:
        if config.dp_mode == "fsdp":
            if phase == "forward":
                return 3 * HW / dp + HW
            if phase == "backward":
                return 3 * HW / dp + (H + 1) * W_bytes + (H - 1) * W_bytes / dp
            return 4 * HW / dp  # optimizer
        base = (3 if phase == "forward" else 4) * HW
        return base + ddp_bucket

    # ---- Batch: sae_in / scaled batch (one slice; B is local batch) ----
    if not config.use_scale_factor:
        batch_bytes = H * B * d_in * db
    else: 
        batch_bytes = 2* H * B * d_in * db

    # ---- buffer ----


    # ---- Retained output kept live across the forward→backward boundary ----
    # H·(2·B·d_sae + 2·B·d_in)·db + H·3 alloc-blocks (loss/mse/aux scalars).
    current_output_bytes = (
        H * (2 * B * d_sae + B * d_in) * db + H * 3 * ALLOC_BLOCK_BYTES
    )   
    retained_output_bytes = (
        H * (2 * B * d_sae + 2 * B * d_in) * db + H * 3 * ALLOC_BLOCK_BYTES
    )

    # ---- Forward dead-feature aux term (F = num_dead_features, R_sum) ----
    #   max(B·F·1 + B·F·4, B·F·4 + R_sum);  = R_sum = 0 by default (F = 0).
    F = config.num_dead_features
    forward_aux_bytes = max(B * F * 1 + B * F * 4, B * F * 4 + config.r_sum_bytes)

    # ==================== FORWARD ====================
    # transient: H·(2·B·d_sae/tp + B·d_in)·db + H·topk + aux + C_forward
    fwd_current = H * (2 * B * d_sae_local + B * d_in) * db
    fwd_topk = H * topk_one_hook
    forward_bytes = (
        _weight_mem("forward")
        + batch_bytes
        + current_output_bytes
        + retained_output_bytes
        + fwd_current
        + fwd_topk
        + forward_aux_bytes
        + config.c_forward_bytes
    )

    # ==================== BACKWARD ====================
    # transient: (d_in·d_sae/tp + 2·B·d_sae/tp + B·d_in)·db      [bracket incl. P_big]
    #          + (H-1)·topk + C_backward + T_full_feature_grad(2·B·d_sae·db)
    bwd_bracket = (d_in * d_sae_local + 2 * B * d_sae_local + B * d_in) * db
    bwd_topk_prev = max(0, H - 1) * topk_one_hook
    backward_bytes = (
        _weight_mem("backward")
        + batch_bytes
        + current_output_bytes
        + retained_output_bytes
        + bwd_bracket
        + bwd_topk_prev
        + config.c_backward_bytes
    )

    # ==================== OPTIMIZER ====================
    # transient: fused=0 | foreach=H·W (DDP) / H·W/dp (FSDP) | for_loop=3·d_in·d_sae/tp.
    # Retained output is freed after backward, so it does not appear here.
    if config.optimizer_impl == "fused":
        opt_transient = 0.0
    elif config.optimizer_impl == "foreach":
        opt_transient = HW / dp if config.dp_mode == "fsdp" else HW
    else:  # for_loop
        opt_transient = 3 * W_bytes / dp if config.dp_mode == "fsdp" else 3 * p_big_bytes
    optimizer_bytes = (
        _weight_mem("optimizer") 
        + batch_bytes       
        + current_output_bytes
        + retained_output_bytes
        + opt_transient
        + config.c_optimizer_bytes
    )

    components = {
        "forward": {
            "weights": _weight_mem("forward"),
            "batch": batch_bytes,
            "current_output": current_output_bytes,
            "retained_output": retained_output_bytes,
            "forward__transient": fwd_current,
            "topk": fwd_topk,
            "aux": forward_aux_bytes,
            "c_forward": config.c_forward_bytes,
        },
        "backward": {
            "weights": _weight_mem("backward"),
            "batch": batch_bytes,
            "current_output": current_output_bytes,
            "retained_output": retained_output_bytes,
            "bracket": bwd_bracket,
            "topk_prev_hooks": bwd_topk_prev,
            "c_backward": config.c_backward_bytes,
        },
        "optimizer": {
            "weights": _weight_mem("optimizer"),
            "batch": batch_bytes,
            "current_output": current_output_bytes,
            "retained_output": retained_output_bytes,
            "optim_transient": opt_transient,
            "optimizer": config.c_optimizer_bytes,
        },
    }
    return PhaseMemoryEstimate(
        forward_bytes=forward_bytes,
        backward_bytes=backward_bytes,
        optimizer_bytes=optimizer_bytes,
        components=components,
    )
