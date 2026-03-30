"""Distributed training utilities for prefix-overlap vLLM TP + SAE TP + SAE DP.

Supports three parallelism modes:

1. **1:1 / local overlap**: ``vllm_dp_size=1``, ``sae_dp_size=1``.
   ``replica_size = max(vllm_tp, sae_tp)``, ``world = replica_size``.

2. **m:1 / vLLM DP fan-in**: ``vllm_dp_size>1``, ``sae_dp_size=1``.
   Multiple independent vLLM instances feed one SAE trainer.
   ``world = max(vllm_tp * vllm_dp, sae_tp)``.

3. **m:m / matched DP**: ``vllm_dp_size==sae_dp_size>1``.
   Each vLLM DP replica feeds one SAE DP replica; SAE replicas sync gradients.
   ``world = max(vllm_tp, sae_tp) * sae_dp_size``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

# Module-level state for prefix-overlap training.
_sae_tp_group: dist.ProcessGroup | None = None
_sae_dp_group: dist.ProcessGroup | None = None
_vllm_tp_group: dist.ProcessGroup | None = None
_worker_group: dist.ProcessGroup | None = None
_worker_cpu_group: dist.ProcessGroup | None = None
_vllm_dp_p2p_group: dist.ProcessGroup | None = None  # Gloo group for vLLM DP P2P
_sae_tp_rank: int = 0
_sae_tp_size: int = 1
_sae_dp_rank: int = 0
_sae_dp_size: int = 1
_worker_rank: int = 0
_worker_size: int = 1
_cluster_block: int = 1  # ranks per cluster in fan-in (mn:m) topology
_vllm_tp_rank: int = -1
_vllm_tp_size: int = 1
_vllm_dp_rank: int = 0
_vllm_dp_size: int = 1
_sae_active: bool = True
_vllm_active: bool = True
_layout_mode: str = "local_overlap"


def init_distributed(
    tp_size: int | None = None,
    sae_dp_size: int = 1,
    *,
    sae_tp_size: int | None = None,
    vllm_tp_size: int = 1,
    vllm_dp_size: int = 1,
    shared_tp_size: int | None = None,
) -> None:
    """Initialize process groups for vLLM TP + SAE TP (+ optional DP).

    Supported topologies:

    - 1:1: ``vllm_dp_size = sae_dp_size = 1``
    - m:1: ``vllm_dp_size > 1, sae_dp_size = 1``
    - m:m: ``vllm_dp_size = sae_dp_size > 1``

    Must be called after torch.distributed.init_process_group().
    """
    global _sae_tp_group, _sae_dp_group, _vllm_tp_group, _worker_group, _worker_cpu_group
    global _vllm_dp_p2p_group
    global _sae_tp_rank, _sae_tp_size, _sae_dp_rank, _sae_dp_size
    global _worker_rank, _worker_size, _cluster_block, _vllm_tp_rank, _vllm_tp_size
    global _vllm_dp_rank, _vllm_dp_size
    global _sae_active, _vllm_active, _layout_mode

    if shared_tp_size is None and tp_size is not None and sae_tp_size is None:
        shared_tp_size = tp_size
    if shared_tp_size is not None:
        if vllm_dp_size > 1:
            raise ValueError(
                "shared_tp_size is incompatible with vllm_dp_size > 1"
            )
        if sae_tp_size is not None and sae_tp_size != shared_tp_size:
            raise ValueError(
                f"shared_tp_size={shared_tp_size} conflicts with sae_tp_size={sae_tp_size}"
            )
        if vllm_tp_size not in (1, shared_tp_size):
            raise ValueError(
                f"shared_tp_size={shared_tp_size} conflicts with vllm_tp_size={vllm_tp_size}"
            )
        sae_tp_size = shared_tp_size
        vllm_tp_size = shared_tp_size

    if sae_tp_size is None:
        sae_tp_size = 1

    if sae_dp_size < 1:
        raise ValueError(f"sae_dp_size must be >= 1, got {sae_dp_size}")
    if vllm_dp_size < 1:
        raise ValueError(f"vllm_dp_size must be >= 1, got {vllm_dp_size}")

    if sae_dp_size > 1 and vllm_dp_size == 1:
        raise ValueError(
            "sae_dp_size > 1 requires vllm_dp_size > 1; 1:m is not supported"
        )
    if vllm_dp_size > 1 and sae_dp_size > 1:
        large = max(vllm_dp_size, sae_dp_size)
        small = min(vllm_dp_size, sae_dp_size)
        if large % small != 0:
            raise ValueError(
                "vllm_dp_size and sae_dp_size must be integer multiples of each other; "
                f"got vllm_dp_size={vllm_dp_size}, sae_dp_size={sae_dp_size}"
            )
        if sae_dp_size > vllm_dp_size:
            raise ValueError(
                "m:mn topology (sae_dp_size > vllm_dp_size) is not yet supported"
            )
    fan_in_topology = vllm_dp_size > sae_dp_size
    matched_dp_topology = vllm_dp_size == sae_dp_size and vllm_dp_size > 1

    if fan_in_topology and sae_tp_size > vllm_tp_size * vllm_dp_size:
        raise ValueError(
            f"sae_tp_size={sae_tp_size} > vllm_tp_size*vllm_dp_size="
            f"{vllm_tp_size * vllm_dp_size} is not supported with vllm_dp > sae_dp"
        )

    assert dist.is_initialized(), "Call dist.init_process_group() before init_distributed()"
    world_size = dist.get_world_size()

    if fan_in_topology:
        # mn:m cluster model: vllm_dp = n * sae_dp, each cluster has n vLLM replicas → 1 SAE.
        # n=1, sae_dp=1 is the legacy m:1 case (special case of this model).
        expected_world_size = max(vllm_tp_size * vllm_dp_size, sae_tp_size * sae_dp_size)
    else:
        replica_size = max(vllm_tp_size, sae_tp_size)
        expected_world_size = replica_size * sae_dp_size

    assert world_size == expected_world_size, (
        f"world_size={world_size} != expected={expected_world_size} "
        f"(vllm_tp={vllm_tp_size}, vllm_dp={vllm_dp_size}, "
        f"sae_tp={sae_tp_size}, sae_dp={sae_dp_size})"
    )

    rank = dist.get_rank()
    _sae_tp_size = sae_tp_size
    _sae_dp_size = sae_dp_size
    _vllm_tp_size = vllm_tp_size
    _vllm_dp_size = vllm_dp_size
    _vllm_dp_p2p_group = None

    if fan_in_topology:
        # ---- mn:m fan-in topology (generalisation of m:1) ----
        # n vLLM replicas per cluster, each cluster feeds one SAE replica.
        # n=vllm_dp_size when sae_dp_size==1 (legacy m:1 is a special case).
        _layout_mode = "vllm_dp_fan_in"
        n = vllm_dp_size // sae_dp_size
        block = max(n * vllm_tp_size, sae_tp_size)  # ranks per cluster

        c = rank // block   # cluster index == sae_dp_rank
        w = rank % block    # offset within cluster

        _sae_dp_rank = c
        _sae_active = w < sae_tp_size
        _sae_tp_rank = w if _sae_active else -1
        _vllm_active = w < n * vllm_tp_size
        if _vllm_active:
            _vllm_dp_rank = c * n + w // vllm_tp_size
            _vllm_tp_rank = w % vllm_tp_size
        else:
            _vllm_dp_rank = -1
            _vllm_tp_rank = -1
        _worker_size = vllm_tp_size
        _cluster_block = block
        _worker_rank = _vllm_tp_rank if _vllm_active else -1

        # vLLM TP group for each vLLM DP replica.
        for vdp in range(vllm_dp_size):
            cc = vdp // n
            j = vdp % n
            base = cc * block + j * vllm_tp_size
            ranks = list(range(base, base + vllm_tp_size))
            group = dist.new_group(ranks)
            cpu_group = dist.new_group(ranks, backend="gloo")
            if _vllm_active and _vllm_dp_rank == vdp:
                _vllm_tp_group = group
                _worker_group = group
                _worker_cpu_group = cpu_group

        # SAE TP group for each SAE replica (cluster).
        for cc in range(sae_dp_size):
            base = cc * block
            sae_ranks = list(range(base, base + sae_tp_size))
            group = dist.new_group(sae_ranks)
            if _sae_active and _sae_dp_rank == cc:
                _sae_tp_group = group

        # SAE DP group: same sae_tp_rank across all clusters.
        for sae_tp_r in range(sae_tp_size):
            dp_ranks = [cc * block + sae_tp_r for cc in range(sae_dp_size)]
            group = dist.new_group(dp_ranks)
            if _sae_active and _sae_tp_rank == sae_tp_r:
                _sae_dp_group = group

        # Dedicated Gloo group for vLLM DP P2P activation transfer.
        # Keep this path off NCCL entirely: vLLM's parallel-state setup can
        # leave additional NCCL P2P communicators in an invalid device state.
        all_ranks = list(range(world_size))
        _vllm_dp_p2p_group = dist.new_group(all_ranks, backend="gloo")
    else:
        # ---- Local overlap / matched DP topology ----
        _layout_mode = "matched_dp" if matched_dp_topology else "local_overlap"
        replica_size = max(vllm_tp_size, sae_tp_size)
        _worker_size = replica_size
        _sae_dp_rank = rank // replica_size
        _worker_rank = rank % replica_size
        _sae_active = _worker_rank < sae_tp_size
        _vllm_active = _worker_rank < vllm_tp_size
        _sae_tp_rank = _worker_rank if _sae_active else -1
        _vllm_tp_rank = _worker_rank if _vllm_active else -1
        _vllm_dp_rank = _sae_dp_rank if matched_dp_topology else 0
        _vllm_dp_size = sae_dp_size if matched_dp_topology else 1

        # Active worker group for each replica.
        for replica_r in range(sae_dp_size):
            base = replica_r * replica_size
            ranks = list(range(base, base + replica_size))
            group = dist.new_group(ranks)
            cpu_group = dist.new_group(ranks, backend="gloo")
            if _sae_dp_rank == replica_r:
                _worker_group = group
                _worker_cpu_group = cpu_group

        # SAE TP group: prefix [0, sae_tp_size) inside each replica.
        for replica_r in range(sae_dp_size):
            base = replica_r * replica_size
            ranks = list(range(base, base + sae_tp_size))
            group = dist.new_group(ranks)
            if _sae_dp_rank == replica_r and _sae_active:
                _sae_tp_group = group

        # vLLM TP group: prefix [0, vllm_tp_size) inside each replica.
        for replica_r in range(sae_dp_size):
            base = replica_r * replica_size
            ranks = list(range(base, base + vllm_tp_size))
            group = dist.new_group(ranks)
            if _sae_dp_rank == replica_r and _vllm_active:
                _vllm_tp_group = group

        # SAE DP group: same local SAE rank across replicas.
        for sae_tp_r in range(sae_tp_size):
            ranks = [replica_r * replica_size + sae_tp_r for replica_r in range(sae_dp_size)]
            group = dist.new_group(ranks)
            if _sae_active and _sae_tp_rank == sae_tp_r:
                _sae_dp_group = group


def get_worker_group() -> dist.ProcessGroup | None:
    return _worker_group


def get_worker_cpu_group() -> dist.ProcessGroup | None:
    return _worker_cpu_group


def get_worker_rank() -> int:
    return _worker_rank


def get_worker_size() -> int:
    return _worker_size


def is_worker_active() -> bool:
    return _worker_group is not None


def is_sae_active() -> bool:
    return _sae_active


def is_vllm_active() -> bool:
    return _vllm_active


def get_replica_base_rank() -> int:
    return _sae_dp_rank * _worker_size


def get_vllm_root_rank() -> int:
    """Root rank of this rank's vLLM DP group."""
    if _layout_mode == "vllm_dp_fan_in":
        return _vllm_dp_rank * _vllm_tp_size
    return get_replica_base_rank()


def get_vllm_world_ranks() -> list[int]:
    if _layout_mode == "vllm_dp_fan_in":
        return list(range(_vllm_tp_size * _vllm_dp_size))
    return [
        dp_r * _worker_size + local_rank
        for dp_r in range(_sae_dp_size)
        for local_rank in range(_vllm_tp_size)
    ]


def get_sae_tp_group() -> dist.ProcessGroup | None:
    return _sae_tp_group


def get_tp_group() -> dist.ProcessGroup | None:
    """Backward-compatible alias for the SAE TP group."""
    return get_sae_tp_group()


def get_sae_dp_group() -> dist.ProcessGroup | None:
    return _sae_dp_group


def get_vllm_tp_group() -> dist.ProcessGroup | None:
    return _vllm_tp_group


def get_dp_group() -> dist.ProcessGroup | None:
    """Backward-compatible alias for the SAE DP group."""
    return get_sae_dp_group()


def get_sae_tp_rank() -> int:
    return _sae_tp_rank


def get_tp_rank() -> int:
    """Backward-compatible alias for the SAE TP rank."""
    return get_sae_tp_rank()


def get_sae_tp_size() -> int:
    return _sae_tp_size


def get_tp_size() -> int:
    """Backward-compatible alias for the SAE TP size."""
    return get_sae_tp_size()


def get_sae_dp_rank() -> int:
    return _sae_dp_rank


def get_dp_rank() -> int:
    """Backward-compatible alias for the SAE DP rank."""
    return get_sae_dp_rank()


def get_sae_dp_size() -> int:
    return _sae_dp_size


def get_dp_size() -> int:
    """Backward-compatible alias for the SAE DP size."""
    return get_sae_dp_size()


def get_vllm_tp_rank() -> int:
    return _vllm_tp_rank


def get_vllm_tp_size() -> int:
    return _vllm_tp_size


def get_vllm_dp_rank() -> int:
    return _vllm_dp_rank


def get_vllm_dp_size() -> int:
    return _vllm_dp_size


def is_vllm_dp_root() -> bool:
    """True if this rank is the root (TP rank 0) of its vLLM DP group."""
    return _vllm_active and _vllm_tp_rank == 0


def get_sae_root_rank() -> int:
    """Global rank of the SAE TP root (sae_tp_rank=0) in this rank's cluster.

    In fan-in (mn:m) topology this is the first rank of the cluster (c * block).
    In matched DP / local overlap it is the first rank of the replica.
    """
    if _layout_mode == "vllm_dp_fan_in":
        return _sae_dp_rank * _cluster_block
    return _sae_dp_rank * _worker_size


def get_vllm_dp_p2p_group() -> dist.ProcessGroup | None:
    """Dedicated Gloo group for vLLM DP P2P send/recv."""
    return _vllm_dp_p2p_group


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class _AllGather(torch.autograd.Function):
    """AllGather along the last dimension. Backward slices the local shard's gradient."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        local: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        tp_size = dist.get_world_size(group)
        ctx.tp_rank = dist.get_rank(group)
        ctx.shard_size = local.shape[-1]
        output_tensors = [torch.zeros_like(local) for _ in range(tp_size)]
        dist.all_gather(output_tensors, local.contiguous(), group=group)
        return torch.cat(output_tensors, dim=-1)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_full: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        start = ctx.tp_rank * ctx.shard_size
        grad_local = grad_full[..., start : start + ctx.shard_size].contiguous()
        return grad_local, None


class _AllReduce(torch.autograd.Function):
    """AllReduce (sum). Backward is identity (grad already replicated across ranks)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        out = x.clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return grad, None


def tp_allgather(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """AllGather along last dim with proper gradient support."""
    return _AllGather.apply(local, group)  # type: ignore[return-value]


def tp_allreduce(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """AllReduce (sum) with proper gradient support."""
    return _AllReduce.apply(x, group)  # type: ignore[return-value]


def preinit_vllm_distributed(vllm_world_ranks: list[int], vllm_tp_size: int) -> None:
    """Pre-initialize vLLM parallel state on ALL torchrun ranks.

    Call this on every rank before any rank constructs LLM(), when
    sae_tp_size > vllm_tp_size (split roles).  After this call, LLM()
    construction finds vLLM parallel state already set up and skips its own
    dist.new_group() calls, preventing the deadlock that would otherwise occur
    because non-vLLM ranks never enter LLM() and never call dist.new_group().
    """
    from vllm.distributed.parallel_state import preinit_vllm_parallel_state

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    preinit_vllm_parallel_state(vllm_world_ranks, vllm_tp_size, local_rank)
