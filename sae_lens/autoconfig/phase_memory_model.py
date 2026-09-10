"""Closed-form GPU memory estimates for SAE training phases.

The model deliberately separates live tensors from streaming-buffer peaks.  A
CUDA caching allocator can reserve more memory than is currently live, but that
reserved slack is not part of the tensor model and is reported separately by
the runtime profiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


_DTYPE_BYTES = {
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
}
_OPTIMIZER_IMPLS = {"fused", "foreach", "for_loop"}
_DP_MODES = {"none", "ddp", "fsdp"}


@dataclass(frozen=True)
class SAEPhaseMemoryConfig:
    """Inputs to :func:`estimate_phase_memory`.

    ``train_batch_size_tokens`` is the global batch size for DDP/FSDP and the
    local batch size for the single-process case.  Streaming fields are
    optional; they are inactive by default so existing phase estimates remain
    unchanged.  Set ``streaming_source_rank=True`` for the source/root rank,
    which owns the logical mixers and their GPU buffers.
    """

    d_in: int
    d_sae: int
    num_hooks: int = 1
    k: int = 128
    train_batch_size_tokens: int = 2048
    dtype: str = "fp32"
    optimizer_impl: str = "fused"
    tp_size: int = 1
    dp_mode: str = "none"
    dp_size: int = 1
    gradient_as_bucket_view: bool = True
    num_dead_features: int = 0
    dead_feature_residual_sum_bytes: int = 0

    # Streaming memory is modeled only when one of these fields activates it.
    streaming_enabled: bool = False
    streaming_mix_chunks: int = 0
    streaming_chunk_size_tokens: int = 0
    streaming_mix_fraction: float = 0.5
    streaming_mixing_streams: int = 0
    streaming_buffer_size_tokens: int | None = None
    streaming_prefetch_chunks: int = 0
    streaming_source_rank: bool = False

    def __post_init__(self) -> None:
        if self.d_in < 1 or self.d_sae < 1:
            raise ValueError("d_in and d_sae must be >= 1")
        if self.num_hooks < 1:
            raise ValueError("num_hooks must be >= 1")
        if self.k < 1 or self.k > self.d_sae:
            raise ValueError("k must satisfy 1 <= k <= d_sae")
        if self.train_batch_size_tokens < 1:
            raise ValueError("train_batch_size_tokens must be >= 1")
        if self.tp_size < 1 or self.d_sae % self.tp_size != 0:
            raise ValueError("tp_size must be >= 1 and divide d_sae")
        if self.dp_size < 1:
            raise ValueError("dp_size must be >= 1")

        dtype = self.dtype.lower().replace("-", "")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype!r}")
        object.__setattr__(self, "dtype", dtype)

        optimizer_impl = self.optimizer_impl.lower().replace("-", "_")
        if optimizer_impl == "forloop":
            optimizer_impl = "for_loop"
        if optimizer_impl not in _OPTIMIZER_IMPLS:
            raise ValueError(
                "optimizer_impl must be one of: "
                + ", ".join(sorted(_OPTIMIZER_IMPLS))
            )
        object.__setattr__(self, "optimizer_impl", optimizer_impl)

        dp_mode = self.dp_mode.lower()
        if dp_mode in {"single", "manual"}:
            dp_mode = "none"
        if dp_mode not in _DP_MODES:
            raise ValueError("dp_mode must be one of: none, ddp, fsdp")
        object.__setattr__(self, "dp_mode", dp_mode)
        if self.num_dead_features < 0 or self.num_dead_features > self.d_sae:
            raise ValueError("num_dead_features must be in [0, d_sae]")
        if self.dead_feature_residual_sum_bytes < 0:
            raise ValueError("dead_feature_residual_sum_bytes must be >= 0")

        if self.streaming_mix_chunks < 0:
            raise ValueError("streaming_mix_chunks must be >= 0")
        if self.streaming_chunk_size_tokens < 0:
            raise ValueError("streaming_chunk_size_tokens must be >= 0")
        if self.streaming_mixing_streams < 0:
            raise ValueError("streaming_mixing_streams must be >= 0")
        if self.streaming_prefetch_chunks < 0:
            raise ValueError("streaming_prefetch_chunks must be >= 0")
        if not 0 <= self.streaming_mix_fraction <= 1:
            raise ValueError("streaming_mix_fraction must be in [0, 1]")
        if self.streaming_buffer_size_tokens is not None:
            if self.streaming_buffer_size_tokens < 1:
                raise ValueError("streaming_buffer_size_tokens must be >= 1")
        if self.streaming_mix_chunks > 0 and self.streaming_chunk_size_tokens < 1:
            raise ValueError(
                "streaming_chunk_size_tokens must be >= 1 when mix_chunks > 0"
            )

    @property
    def dtype_bytes(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    @property
    def local_batch_tokens(self) -> int:
        if self.dp_mode in {"ddp", "fsdp"}:
            # Exact-DP streaming permits a one-token remainder on some ranks;
            # ceil keeps the estimate conservative for that case.
            return ceil(self.train_batch_size_tokens / self.dp_size)
        return self.train_batch_size_tokens

    @property
    def d_sae_local(self) -> int:
        return self.d_sae // self.tp_size

    @property
    def weight_bytes_per_hook(self) -> int:
        # W_enc and W_dec are TP-sharded; b_dec is replicated.
        return (
            2 * self.d_in * self.d_sae_local + self.d_sae_local + self.d_in
        ) * self.dtype_bytes

    @property
    def streaming_active(self) -> bool:
        return bool(
            self.streaming_enabled
            or self.streaming_mix_chunks
            or self.streaming_chunk_size_tokens
            or self.streaming_buffer_size_tokens is not None
        )

    @property
    def streaming_stream_count(self) -> int:
        if self.streaming_mixing_streams:
            return self.streaming_mixing_streams
        if self.dp_mode == "ddp":
            return self.dp_size
        return 1

    @property
    def streaming_logical_batch_tokens(self) -> int:
        return ceil(self.train_batch_size_tokens / self.streaming_stream_count)

    @property
    def streaming_capacity_tokens(self) -> int:
        requested = self.streaming_buffer_size_tokens or 0
        mixing_capacity = self.streaming_mix_chunks * self.streaming_chunk_size_tokens
        return max(
            self.streaming_logical_batch_tokens,
            requested,
            mixing_capacity,
        )


@dataclass(frozen=True)
class SAEPhaseMemoryEstimate:
    """Estimated bytes for the three training phases and the detailed terms."""

    forward_bytes: float
    backward_bytes: float
    optimizer_bytes: float
    peak_bytes: float
    components: dict[str, dict[str, float]]

    @property
    def forward_mb(self) -> float:
        return self.forward_bytes / 1024**2

    @property
    def backward_mb(self) -> float:
        return self.backward_bytes / 1024**2

    @property
    def optimizer_mb(self) -> float:
        return self.optimizer_bytes / 1024**2

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1024**2

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation for sweep scripts."""
        return {
            "forward_bytes": self.forward_bytes,
            "backward_bytes": self.backward_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "peak_bytes": self.peak_bytes,
            "components": self.components,
        }


def _weight_terms(cfg: SAEPhaseMemoryConfig) -> tuple[float, float, float]:
    """Return (forward, backward, optimizer) resident weight bytes."""
    hooks = cfg.num_hooks
    weight = cfg.weight_bytes_per_hook
    if cfg.dp_mode == "fsdp":
        # FSDP shards P/M/V, all-gathers one full parameter set for forward,
        # and keeps full parameters/gradients for the currently reduced hook.
        forward = 3 * hooks * weight / cfg.dp_size + hooks * weight
        backward = (
            3 * hooks * weight / cfg.dp_size
            + (hooks + 1) * weight
            + (hooks - 1) * weight / cfg.dp_size
        )
        optimizer = 4 * hooks * weight / cfg.dp_size
    else:
        forward = 3 * hooks * weight
        backward = optimizer = 4 * hooks * weight
    return forward, backward, optimizer


def _streaming_terms(cfg: SAEPhaseMemoryConfig) -> dict[str, float]:
    """Estimate source-rank streaming storage and refill peaks."""
    empty: dict[str, float] = {
        "active": 0,
        "capacity_tokens": 0,
        "stream_count": 0,
        "mixing_storage": 0,
        "prefetch": 0,
        "refill_transient": 0,
        "shuffle_copy": 0,
        "data_provider_buffers": 0,
        "data_fetch_peak": 0,
    }
    if not cfg.streaming_active or not cfg.streaming_source_rank:
        return empty

    element_bytes = cfg.d_in * cfg.dtype_bytes
    hooks = cfg.num_hooks
    streams = cfg.streaming_stream_count
    capacity = cfg.streaming_capacity_tokens
    storage = streams * capacity * hooks * element_bytes

    chunk_tokens = cfg.streaming_chunk_size_tokens
    prefetch = (
        cfg.streaming_prefetch_chunks * chunk_tokens * hooks * element_bytes
        if chunk_tokens
        else 0
    )

    # Each refill first concatenates old and new tensors.  The dict
    # comprehension retains all old hook tensors until the last hook is done,
    # so count one full post-concat buffer as a transient allocation.
    refill_tokens = cfg.streaming_logical_batch_tokens
    refill = hooks * (capacity + refill_tokens) * element_bytes
    has_streaming_shuffle = (
        cfg.streaming_mix_chunks > 0
        or cfg.streaming_buffer_size_tokens is not None
    )
    shuffle_copy = (
        storage if has_streaming_shuffle and cfg.streaming_mix_fraction > 0 else 0
    )
    resident = storage + prefetch
    data_fetch_peak = resident + refill + shuffle_copy
    return {
        "active": 1,
        "capacity_tokens": float(capacity),
        "stream_count": float(streams),
        "mixing_storage": float(storage),
        "prefetch": float(prefetch),
        "refill_transient": float(refill),
        "shuffle_copy": float(shuffle_copy),
        "data_provider_buffers": float(resident),
        "data_fetch_peak": float(data_fetch_peak),
    }


def estimate_phase_memory(cfg: SAEPhaseMemoryConfig) -> SAEPhaseMemoryEstimate:
    """Estimate forward, backward, optimizer, and overall peak memory.

    The forward/backward terms model the dense TopK SAE tensors retained by the
    current multi-hook trainer.  Streaming data-provider terms are persistent
    on the source rank and are included in every phase; their refill/shuffle
    allocations are included only in ``peak_bytes`` because they occur during
    data fetch, before the model forward.
    """
    db = cfg.dtype_bytes
    h = cfg.num_hooks
    b = cfg.local_batch_tokens
    d_in = cfg.d_in
    d_sae = cfg.d_sae_local
    weight_fwd, weight_bwd, weight_opt = _weight_terms(cfg)

    batch = h * b * d_in * db
    retained = h * ((2 * b * d_sae + 2 * b * d_in) * db + 3 * 512)
    current = h * (2 * b * d_sae + b * d_in) * db
    topk = h * (b * cfg.k * db + b * cfg.k * 8)

    if cfg.num_dead_features:
        f = cfg.num_dead_features
        aux = h * max(
            b * f * 1 + b * f * 4,
            b * f * 4 + cfg.dead_feature_residual_sum_bytes,
        )
    else:
        aux = 0

    bracket = h * (
        d_in * d_sae + 2 * b * d_sae + b * d_in
    ) * db
    full_backward = h * 2 * b * d_sae * db
    grad_bucket = (
        h * cfg.weight_bytes_per_hook
        if cfg.dp_mode == "ddp" and not cfg.gradient_as_bucket_view
        else 0
    )

    if cfg.optimizer_impl == "fused":
        optim_transient = 0
    elif cfg.optimizer_impl == "foreach":
        optim_transient = h * cfg.weight_bytes_per_hook
    else:
        optim_transient = 3 * d_in * d_sae * db

    streaming = _streaming_terms(cfg)
    streaming_resident = streaming["data_provider_buffers"]
    forward_components = {
        "weights": weight_fwd,
        "batch": float(batch),
        "retained_output": float(retained),
        "current_output": float(current),
        "topk": float(topk),
        "aux": float(aux),
        "streaming_buffers": streaming_resident,
    }
    backward_components = {
        "weights": weight_bwd,
        "batch": float(batch),
        "retained_output": float(retained),
        "bracket": float(bracket),
        "full_backward": float(full_backward),
        "grad_bucket": float(grad_bucket),
        "streaming_buffers": streaming_resident,
    }
    optimizer_components = {
        "weights": weight_opt,
        "batch": float(batch),
        "optim_transient": float(optim_transient),
        "streaming_buffers": streaming_resident,
    }

    forward = sum(forward_components.values())
    backward = sum(backward_components.values())
    optimizer = sum(optimizer_components.values())
    model_peak = max(forward, backward, optimizer)
    # Data fetch follows the optimizer boundary. The source rank still holds
    # the optimizer-phase resident tensors while refill/shuffle allocates its
    # temporary copies, so add those transients to the optimizer baseline.
    data_fetch_total = (
        optimizer
        + streaming["refill_transient"]
        + streaming["shuffle_copy"]
    )
    peak = max(model_peak, data_fetch_total)

    return SAEPhaseMemoryEstimate(
        forward_bytes=forward,
        backward_bytes=backward,
        optimizer_bytes=optimizer,
        peak_bytes=peak,
        components={
            "forward": forward_components,
            "backward": backward_components,
            "optimizer": optimizer_components,
            "streaming": streaming,
        },
    )


__all__ = ["SAEPhaseMemoryConfig", "SAEPhaseMemoryEstimate", "estimate_phase_memory"]
