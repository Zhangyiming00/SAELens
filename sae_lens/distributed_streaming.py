"""
Thin wrapper over distributed_v2 for streaming_mode v1.

Exposes the subset of distributed primitives needed by the streaming producer/consumer
pipeline, and adds helpers that distributed_v2 doesn't provide:
  - init_distributed_streaming() — delegates to init_distributed_v2
  - get_vllm_dp_size()           — distributed_v2 has _P but no getter
  - get_producer_tp_root()       — convenience: TP root for *this* producer
  - get_consumer_tp_root()       — TP root for this physical SAE endpoint
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
    sae_pp_size: int = 1,
    use_gpu_direct: bool = False,
) -> None:
    """Initialize process groups for streaming_mode v1.

    ``sae_dp=0`` is the vLLM-only topology.  ``sae_dp>=1`` is supported on
    the SHM path; SAE-DP roots use equal-cohort allocation in
    SharedActivationBuffer so DDP/FSDP replicas see equal-length inputs.
    """
    if sae_dp < 0:
        raise ValueError(f"sae_dp must be >= 0, got sae_dp={sae_dp}")
    global _vllm_dp_size
    _vllm_dp_size = vllm_dp
    _v2.init_distributed_v2(
        P=vllm_dp,
        Q=sae_dp,
        vllm_tp_size=vllm_tp,
        sae_tp_size=sae_tp,
        sae_pp_size=sae_pp_size,
        batch_size=1,
        disjoint=True,
        use_gpu_direct=use_gpu_direct,
        build_routing_table=False,
    )


# ---------------------------------------------------------------------------
# Re-exports from distributed_v2 (no changes needed there)
# ---------------------------------------------------------------------------
is_producer = _v2.is_producer
is_consumer = _v2.is_consumer
get_producer_idx = _v2.get_producer_idx
get_vllm_tp_group = _v2.get_vllm_tp_group
get_sae_tp_group = _v2.get_sae_tp_group
get_sae_tp_cpu_group = _v2.get_sae_tp_cpu_group
get_sae_tp_rank = _v2.get_sae_tp_rank
get_sae_tp_size = _v2.get_sae_tp_size
get_sae_dp_size = _v2.get_sae_dp_size
get_sae_dp_idx = _v2.get_sae_dp_idx
get_sae_dp_group = _v2.get_sae_dp_group
get_sae_pp_rank = _v2.get_sae_pp_rank
get_sae_pp_size = _v2.get_sae_pp_size
get_sae_endpoint_idx = _v2.get_sae_endpoint_idx
get_sae_pp_root_group = _v2.get_sae_pp_root_group
get_sae_pp_root_global_rank = _v2.get_sae_pp_root_global_rank
get_vllm_tp_size = _v2.get_vllm_tp_size
get_vllm_tp_rank = _v2.get_vllm_tp_rank
get_streaming_nccl_group = _v2.get_streaming_nccl_group
get_gloo_ctrl_group = _v2.get_gloo_ctrl_group
get_pp_coord_group = _v2.get_pp_coord_group


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
    """World rank of the TP root for this physical SAE endpoint."""
    endpoint_idx = _v2.get_sae_endpoint_idx()
    if endpoint_idx < 0:
        return -1
    return _v2.get_consumer_tp_root(endpoint_idx)


def is_vllm_tp_root() -> bool:
    """True if this rank is the TP root (vllm_tp_rank == 0) of its producer group."""
    return _v2.get_vllm_tp_rank() == 0


def is_sae_tp_root() -> bool:
    """True if this rank is the TP root (sae_tp_rank == 0) of the consumer group."""
    return _v2.get_sae_tp_rank() == 0
