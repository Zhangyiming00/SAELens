"""Unified shard-routing distributed init for SAELens.

Provides ``init_distributed_v2()`` as a standalone replacement for ``init_distributed()``
when ``use_shard_routing=True``.  Supports arbitrary ``vllm_dp:sae_dp`` ratios without
requiring integer multiples.

Two world layouts are supported via the ``disjoint`` parameter:

Overlapping (default, ``disjoint=False``):
    world_size = max(P * vllm_tp, Q * sae_tp)
    Producer ranks:  [0, P * vllm_tp)
    Consumer ranks:  [0, Q * sae_tp)
    Dual-role ranks: [0, min(P * vllm_tp, Q * sae_tp))
    Producer group p: ranks [p * vllm_tp, (p+1) * vllm_tp).
    Consumer group c: ranks [c * sae_tp,  (c+1) * sae_tp).

Disjoint (``disjoint=True``, used by streaming_mode v1):
    world_size = P * vllm_tp + Q * sae_tp
    Producer ranks:  [0, P * vllm_tp)
    Consumer ranks:  [P * vllm_tp, P * vllm_tp + Q * sae_tp)
    Producer group p: ranks [p * vllm_tp, (p+1) * vllm_tp).
    Consumer group c: ranks [P*vllm_tp + c*sae_tp, P*vllm_tp + (c+1)*sae_tp).
"""

from __future__ import annotations

import torch.distributed as dist

from sae_lens.shard_routing import ShardRoute, compute_routing_table, routes_for_consumer

# ---------------------------------------------------------------------------
# Module-level state (isolated from distributed.py)
# ---------------------------------------------------------------------------

_initialized: bool = False
_P: int = 0  # number of producers (vllm_dp_size)
_Q: int = 0  # number of consumers (sae_dp_size)
_vllm_tp_size: int = 1
_sae_tp_size: int = 1

_is_producer: bool = False
_is_consumer: bool = False
_producer_idx: int = -1  # logical producer index; -1 if not a producer
_consumer_idx: int = -1  # logical consumer index; -1 if not a consumer
_vllm_tp_rank: int = -1  # rank within this rank's vLLM TP group; -1 if not a producer
_sae_tp_rank: int = -1   # rank within this rank's SAE TP group; -1 if not a consumer

# Explicit world-rank maps
_producer_world_ranks: dict[int, list[int]] = {}  # p -> [world ranks in TP group]
_consumer_world_ranks: dict[int, list[int]] = {}  # c -> [world ranks in TP group]
_producer_tp_root: dict[int, int] = {}            # p -> world rank of TP root (vllm_tp_rank=0)
_consumer_tp_root: dict[int, int] = {}            # c -> world rank of TP root (sae_tp_rank=0)

# Process groups
_vllm_tp_group: dist.ProcessGroup | None = None
_sae_tp_group: dist.ProcessGroup | None = None
_sae_dp_group: dist.ProcessGroup | None = None
_consumer_p2p_groups: dict[int, dist.ProcessGroup] = {}  # consumer_idx -> NCCL P2P group

_routing_table: list[ShardRoute] = []


def _reset() -> None:
    """Reset all module state.  Used in tests."""
    global _initialized, _P, _Q, _vllm_tp_size, _sae_tp_size
    global _is_producer, _is_consumer, _producer_idx, _consumer_idx
    global _vllm_tp_rank, _sae_tp_rank
    global _producer_world_ranks, _consumer_world_ranks
    global _producer_tp_root, _consumer_tp_root
    global _vllm_tp_group, _sae_tp_group, _sae_dp_group
    global _consumer_p2p_groups, _routing_table

    _initialized = False
    _P = _Q = 0
    _vllm_tp_size = _sae_tp_size = 1
    _is_producer = _is_consumer = False
    _producer_idx = _consumer_idx = -1
    _vllm_tp_rank = _sae_tp_rank = -1
    _producer_world_ranks = {}
    _consumer_world_ranks = {}
    _producer_tp_root = {}
    _consumer_tp_root = {}
    _vllm_tp_group = None
    _sae_tp_group = None
    _sae_dp_group = None
    _consumer_p2p_groups = {}
    _routing_table = []


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_distributed_v2(
    P: int,
    Q: int,
    vllm_tp_size: int,
    sae_tp_size: int,
    batch_size: int,
    disjoint: bool = False,
) -> None:
    """Initialize all process groups for the unified shard-routing path.

    Must be called after ``dist.init_process_group()``.  Must NOT be combined
    with ``init_distributed()`` — this is a standalone replacement.

    Parameters
    ----------
    P:
        Number of vLLM DP replicas (producers).
    Q:
        Number of SAE DP replicas (consumers).
    vllm_tp_size:
        Tensor-parallel size for each vLLM replica.
    sae_tp_size:
        Tensor-parallel size for each SAE replica.
    batch_size:
        Rows per producer per step (``store_batch_size_prompts * training_context_size``).
        Used to build the routing table.  Must be large enough that every connected
        producer→consumer edge receives at least 1 row.
    disjoint:
        When True, use a disjoint topology where producer and consumer ranks do not
        overlap.  Producer ranks are ``[0, P*vllm_tp_size)`` and consumer ranks are
        ``[P*vllm_tp_size, P*vllm_tp_size + Q*sae_tp_size)``, so
        ``world_size = P*vllm_tp_size + Q*sae_tp_size``.  When False (default), use
        the overlapping topology where ``world_size = max(P*vllm_tp, Q*sae_tp)``.
    """
    global _initialized, _P, _Q, _vllm_tp_size, _sae_tp_size
    global _is_producer, _is_consumer, _producer_idx, _consumer_idx
    global _vllm_tp_rank, _sae_tp_rank
    global _producer_world_ranks, _consumer_world_ranks
    global _producer_tp_root, _consumer_tp_root
    global _vllm_tp_group, _sae_tp_group, _sae_dp_group
    global _consumer_p2p_groups, _routing_table

    assert dist.is_initialized(), "Call dist.init_process_group() before init_distributed_v2()"

    world_size = dist.get_world_size()
    if disjoint:
        expected = P * vllm_tp_size + Q * sae_tp_size
        assert world_size == expected, (
            f"world_size={world_size} != P*vllm_tp + Q*sae_tp={expected} "
            f"(P={P}, vllm_tp={vllm_tp_size}, Q={Q}, sae_tp={sae_tp_size})"
        )
    else:
        expected = max(P * vllm_tp_size, Q * sae_tp_size)
        assert world_size == expected, (
            f"world_size={world_size} != max(P*vllm_tp, Q*sae_tp)={expected} "
            f"(P={P}, vllm_tp={vllm_tp_size}, Q={Q}, sae_tp={sae_tp_size})"
        )

    rank = dist.get_rank()
    _P = P
    _Q = Q
    _vllm_tp_size = vllm_tp_size
    _sae_tp_size = sae_tp_size

    # --- Build explicit rank maps ---
    for p in range(P):
        ranks = list(range(p * vllm_tp_size, (p + 1) * vllm_tp_size))
        _producer_world_ranks[p] = ranks
        _producer_tp_root[p] = ranks[0]

    consumer_offset = P * vllm_tp_size if disjoint else 0
    for c in range(Q):
        base = consumer_offset + c * sae_tp_size
        ranks = list(range(base, base + sae_tp_size))
        _consumer_world_ranks[c] = ranks
        _consumer_tp_root[c] = ranks[0]

    # --- Determine this rank's role by membership ---
    _is_producer = False
    _is_consumer = False
    _producer_idx = -1
    _consumer_idx = -1
    _vllm_tp_rank = -1
    _sae_tp_rank = -1

    for p, ranks in _producer_world_ranks.items():
        if rank in ranks:
            _is_producer = True
            _producer_idx = p
            _vllm_tp_rank = ranks.index(rank)

    for c, ranks in _consumer_world_ranks.items():
        if rank in ranks:
            _is_consumer = True
            _consumer_idx = c
            _sae_tp_rank = ranks.index(rank)

    # --- Create P vLLM TP groups (NCCL) ---
    for p in range(P):
        ranks = _producer_world_ranks[p]
        grp = dist.new_group(ranks, backend="nccl")
        if _is_producer and _producer_idx == p:
            _vllm_tp_group = grp

    # --- Create Q SAE TP groups (NCCL) ---
    for c in range(Q):
        ranks = _consumer_world_ranks[c]
        grp = dist.new_group(ranks, backend="nccl")
        if _is_consumer and _consumer_idx == c:
            _sae_tp_group = grp

    # --- Create SAE DP groups (NCCL): one per sae_tp_rank position ---
    for tp_r in range(sae_tp_size):
        dp_ranks = [_consumer_world_ranks[c][tp_r] for c in range(Q)]
        grp = dist.new_group(dp_ranks, backend="nccl")
        if _is_consumer and _sae_tp_rank == tp_r:
            _sae_dp_group = grp

    # --- Compute routing table ---
    _routing_table = compute_routing_table(P, Q, batch_size)

    # --- Create Q per-consumer NCCL P2P groups ---
    # Uses NCCL for efficient GPU-to-GPU activation transfers.
    # Requires vLLM parallel state to be pre-initialized via preinit_vllm_distributed()
    # to avoid NCCL communicator conflicts.
    for c in range(Q):
        sources = {r.producer_idx for r in _routing_table if r.consumer_idx == c}
        # Deduplicate: producer TP root and consumer TP root may coincide.
        p2p_members = sorted(
            {_consumer_tp_root[c]} | {_producer_tp_root[p] for p in sources}
        )
        grp = dist.new_group(p2p_members, backend="nccl")
        if rank in p2p_members:
            _consumer_p2p_groups[c] = grp

    _initialized = True


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def is_producer() -> bool:
    return _is_producer


def is_consumer() -> bool:
    return _is_consumer


def get_producer_idx() -> int:
    return _producer_idx


def get_consumer_idx() -> int:
    return _consumer_idx


def get_vllm_tp_rank() -> int:
    return _vllm_tp_rank


def get_sae_tp_rank() -> int:
    return _sae_tp_rank


def get_vllm_tp_size() -> int:
    return _vllm_tp_size


def get_sae_tp_size() -> int:
    return _sae_tp_size


def get_sae_dp_size() -> int:
    return _Q


def get_routing_table() -> list[ShardRoute]:
    return _routing_table


def get_vllm_tp_group() -> dist.ProcessGroup | None:
    return _vllm_tp_group


def get_sae_tp_group() -> dist.ProcessGroup | None:
    return _sae_tp_group


def get_sae_dp_group() -> dist.ProcessGroup | None:
    return _sae_dp_group


def get_p2p_group(consumer_idx: int) -> dist.ProcessGroup:
    """Return the NCCL P2P group for the given consumer.

    Raises ``KeyError`` if this rank is not a member of that consumer's P2P group.
    """
    return _consumer_p2p_groups[consumer_idx]


def get_producer_tp_root(p: int) -> int:
    return _producer_tp_root[p]


def get_consumer_tp_root(c: int) -> int:
    return _consumer_tp_root[c]
