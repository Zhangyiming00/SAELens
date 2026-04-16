"""
Thin wrapper over distributed_v2 for streaming_mode v1.

Exposes the subset of distributed primitives needed by the streaming producer/consumer
pipeline, and adds helpers that distributed_v2 doesn't provide:
  - init_distributed_streaming() — enforces sae_dp=1 then delegates to init_distributed_v2
  - get_vllm_dp_size()           — distributed_v2 has _P but no getter
  - get_producer_tp_root()       — convenience: TP root for *this* producer
  - get_consumer_tp_root()       — convenience: TP root for consumer 0 (sae_dp=1 always)
  - is_vllm_tp_root()
  - is_sae_tp_root()
"""

import sae_lens.distributed_v2 as _v2

_vllm_dp_size: int = 0


def init_distributed_streaming(
    vllm_tp: int,
    vllm_dp: int,
    sae_tp: int,
    sae_dp: int = 1,
) -> None:
    """Initialize process groups for streaming_mode v1.

    Enforces sae_dp == 1 (sae_dp > 1 is not supported in v1 — independent
    acquire_up_to() calls diverge at stream tail, causing DDP AllReduce hangs).
    """
    if sae_dp != 1:
        raise ValueError(
            f"streaming_mode v1 requires sae_dp=1, got sae_dp={sae_dp}. "
            "sae_dp > 1 is not supported in v1."
        )
    global _vllm_dp_size
    _vllm_dp_size = vllm_dp
    _v2.init_distributed_v2(
        P=vllm_dp,
        Q=1,
        vllm_tp_size=vllm_tp,
        sae_tp_size=sae_tp,
        batch_size=1,
        disjoint=True,
    )


# ---------------------------------------------------------------------------
# Re-exports from distributed_v2 (no changes needed there)
# ---------------------------------------------------------------------------
is_producer = _v2.is_producer
is_consumer = _v2.is_consumer
get_producer_idx = _v2.get_producer_idx
get_vllm_tp_group = _v2.get_vllm_tp_group
get_sae_tp_group = _v2.get_sae_tp_group
get_sae_tp_rank = _v2.get_sae_tp_rank
get_sae_tp_size = _v2.get_sae_tp_size
get_vllm_tp_size = _v2.get_vllm_tp_size
get_vllm_tp_rank = _v2.get_vllm_tp_rank


# ---------------------------------------------------------------------------
# Additions not in distributed_v2
# ---------------------------------------------------------------------------

def get_vllm_dp_size() -> int:
    """Return the vLLM DP (producer) count set by init_distributed_streaming."""
    return _vllm_dp_size


def get_producer_tp_root() -> int:
    """World rank of the TP root for *this* producer rank."""
    return _v2.get_producer_tp_root(_v2.get_producer_idx())


def get_consumer_tp_root() -> int:
    """World rank of the TP root for consumer 0.

    In v1 sae_dp=1 always, so there is only consumer 0.
    """
    return _v2.get_consumer_tp_root(0)


def is_vllm_tp_root() -> bool:
    """True if this rank is the TP root (vllm_tp_rank == 0) of its producer group."""
    return _v2.get_vllm_tp_rank() == 0


def is_sae_tp_root() -> bool:
    """True if this rank is the TP root (sae_tp_rank == 0) of the consumer group."""
    return _v2.get_sae_tp_rank() == 0
