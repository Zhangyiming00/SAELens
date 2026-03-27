"""Distributed training utilities for prefix-overlap vLLM TP + SAE TP + DP.

Supports two parallelism modes:

1. **Prefix-overlap (legacy)**: ``vllm_dp_size=1``.
   ``replica_size = max(vllm_tp, sae_tp)``, ``world = replica_size * dp_size``.

2. **vLLM Data Parallel**: ``vllm_dp_size > 1``, ``dp_size == 1``.
   Multiple independent vLLM instances feed one SAE trainer.
   ``world = max(vllm_tp * vllm_dp, sae_tp)``.
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
_dp_rank: int = 0
_dp_size: int = 1
_worker_rank: int = 0
_worker_size: int = 1
_vllm_tp_rank: int = -1
_vllm_tp_size: int = 1
_vllm_dp_rank: int = 0
_vllm_dp_size: int = 1
_sae_active: bool = True
_vllm_active: bool = True


def init_distributed(
    tp_size: int | None = None,
    dp_size: int = 1,
    *,
    sae_tp_size: int | None = None,
    vllm_tp_size: int = 1,
    vllm_dp_size: int = 1,
    shared_tp_size: int | None = None,
) -> None:
    """Initialize process groups for vLLM TP + SAE TP (+ optional vLLM DP).

    When ``vllm_dp_size == 1`` (default), this behaves identically to the
    legacy prefix-overlap layout.

    When ``vllm_dp_size > 1``, multiple independent vLLM TP groups are
    created, each producing activations that are gathered to a single SAE
    trainer (sae_dp fixed at 1).

    Process grid layout (vllm_dp > 1):

    - world_size = max(vllm_tp_size * vllm_dp_size, sae_tp_size)
    - vLLM DP group k owns ranks [k*vllm_tp, (k+1)*vllm_tp)
    - SAE occupies ranks [0, sae_tp_size)
    - Ranks can be both vLLM-active and SAE-active (overlap)

    Must be called after torch.distributed.init_process_group().
    """
    global _sae_tp_group, _sae_dp_group, _vllm_tp_group, _worker_group, _worker_cpu_group
    global _vllm_dp_p2p_group
    global _sae_tp_rank, _sae_tp_size, _dp_rank, _dp_size
    global _worker_rank, _worker_size, _vllm_tp_rank, _vllm_tp_size
    global _vllm_dp_rank, _vllm_dp_size
    global _sae_active, _vllm_active

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

    if vllm_dp_size > 1 and dp_size > 1:
        raise ValueError(
            "vllm_dp_size > 1 and dp_size > 1 simultaneously is not supported"
        )
    if vllm_dp_size > 1 and sae_tp_size > vllm_tp_size * vllm_dp_size:
        raise ValueError(
            f"sae_tp_size={sae_tp_size} > vllm_tp_size*vllm_dp_size="
            f"{vllm_tp_size * vllm_dp_size} is not supported with vllm_dp > 1"
        )

    assert dist.is_initialized(), "Call dist.init_process_group() before init_distributed()"
    world_size = dist.get_world_size()

    if vllm_dp_size > 1:
        # vLLM DP mode: multiple vLLM groups → one SAE.
        expected_world_size = max(vllm_tp_size * vllm_dp_size, sae_tp_size)
    else:
        # Legacy prefix-overlap mode.
        expected_world_size = max(vllm_tp_size, sae_tp_size) * dp_size

    assert world_size == expected_world_size, (
        f"world_size={world_size} != expected={expected_world_size} "
        f"(vllm_tp={vllm_tp_size}, vllm_dp={vllm_dp_size}, "
        f"sae_tp={sae_tp_size}, dp={dp_size})"
    )

    rank = dist.get_rank()
    _sae_tp_size = sae_tp_size
    _dp_size = dp_size
    _vllm_tp_size = vllm_tp_size
    _vllm_dp_size = vllm_dp_size
    _vllm_dp_p2p_group = None

    if vllm_dp_size > 1:
        # ---- vLLM DP mode ----
        total_vllm_ranks = vllm_tp_size * vllm_dp_size
        _vllm_active = rank < total_vllm_ranks
        _sae_active = rank < sae_tp_size
        _dp_rank = 0  # sae_dp fixed at 1

        if _vllm_active:
            _vllm_dp_rank = rank // vllm_tp_size
            _vllm_tp_rank = rank % vllm_tp_size
        else:
            _vllm_dp_rank = -1
            _vllm_tp_rank = -1

        _sae_tp_rank = rank if _sae_active else -1
        _worker_size = vllm_tp_size
        _worker_rank = _vllm_tp_rank if _vllm_active else -1

        # vLLM TP group per DP group.
        for vdp in range(vllm_dp_size):
            base = vdp * vllm_tp_size
            ranks = list(range(base, base + vllm_tp_size))
            group = dist.new_group(ranks)
            cpu_group = dist.new_group(ranks, backend="gloo")
            if _vllm_active and _vllm_dp_rank == vdp:
                _vllm_tp_group = group
                _worker_group = group
                _worker_cpu_group = cpu_group

        # SAE TP group (single, sae_dp=1).
        # dist.new_group must be called by ALL ranks for correctness.
        sae_ranks = list(range(sae_tp_size))
        sae_group = dist.new_group(sae_ranks)
        if _sae_active:
            _sae_tp_group = sae_group

        # Dedicated Gloo group for vLLM DP P2P activation transfer.
        # Keep this path off NCCL entirely: vLLM's parallel-state setup can
        # leave additional NCCL P2P communicators in an invalid device state.
        all_ranks = list(range(world_size))
        _vllm_dp_p2p_group = dist.new_group(all_ranks, backend="gloo")

        # No SAE DP group needed (sae_dp=1).
        _sae_dp_group = None
    else:
        # ---- Legacy prefix-overlap mode (unchanged) ----
        replica_size = max(vllm_tp_size, sae_tp_size)
        _worker_size = replica_size
        _dp_rank = rank // replica_size
        _worker_rank = rank % replica_size
        _sae_active = _worker_rank < sae_tp_size
        _vllm_active = _worker_rank < vllm_tp_size
        _sae_tp_rank = _worker_rank if _sae_active else -1
        _vllm_tp_rank = _worker_rank if _vllm_active else -1
        _vllm_dp_rank = 0
        _vllm_dp_size = 1

        # Active worker group for each DP replica.
        for dp_r in range(dp_size):
            base = dp_r * replica_size
            ranks = list(range(base, base + replica_size))
            group = dist.new_group(ranks)
            cpu_group = dist.new_group(ranks, backend="gloo")
            if _dp_rank == dp_r:
                _worker_group = group
                _worker_cpu_group = cpu_group

        # SAE TP group: prefix [0, sae_tp_size) inside each replica.
        for dp_r in range(dp_size):
            base = dp_r * replica_size
            ranks = list(range(base, base + sae_tp_size))
            group = dist.new_group(ranks)
            if _dp_rank == dp_r and _sae_active:
                _sae_tp_group = group

        # vLLM TP group: prefix [0, vllm_tp_size) inside each replica.
        for dp_r in range(dp_size):
            base = dp_r * replica_size
            ranks = list(range(base, base + vllm_tp_size))
            group = dist.new_group(ranks)
            if _dp_rank == dp_r and _vllm_active:
                _vllm_tp_group = group

        # SAE DP group: same local SAE rank across replicas.
        for sae_tp_r in range(sae_tp_size):
            ranks = [dp_r * replica_size + sae_tp_r for dp_r in range(dp_size)]
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
    return _dp_rank * _worker_size


def get_vllm_root_rank() -> int:
    """Root rank of this rank's vLLM DP group."""
    if _vllm_dp_size > 1:
        return _vllm_dp_rank * _vllm_tp_size
    return get_replica_base_rank()


def get_vllm_world_ranks() -> list[int]:
    if _vllm_dp_size > 1:
        return list(range(_vllm_tp_size * _vllm_dp_size))
    return [
        dp_r * _worker_size + local_rank
        for dp_r in range(_dp_size)
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


def get_dp_rank() -> int:
    return _dp_rank


def get_dp_size() -> int:
    return _dp_size


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
