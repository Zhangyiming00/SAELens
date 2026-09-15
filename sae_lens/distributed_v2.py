"""Unified shard-routing distributed init for SAELens.

Provides ``init_distributed_v2()`` as a standalone replacement for ``init_distributed()``
when ``use_shard_routing=True``.  Supports arbitrary ``vllm_dp:sae_dp`` ratios without
requiring integer multiples.

Two world layouts are supported via the ``disjoint`` parameter:

Overlapping (default, ``disjoint=False``):
    world_size = max(P * vllm_tp, Q * sae_pp * sae_tp)
    Producer ranks:  [0, P * vllm_tp)
    SAE endpoint ranks:  [0, Q * sae_pp * sae_tp)
    Dual-role ranks: [0, min(P * vllm_tp, Q * sae_pp * sae_tp))
    Producer group p: ranks [p * vllm_tp, (p+1) * vllm_tp).
    SAE endpoint e: ranks [e * sae_tp,  (e+1) * sae_tp).

Disjoint (``disjoint=True``, used by streaming_mode v1):
    world_size = P * vllm_tp + Q * sae_pp * sae_tp
    Producer ranks:  [0, P * vllm_tp)
    SAE endpoint ranks:  [P * vllm_tp, P * vllm_tp + Q * sae_pp * sae_tp)
    Producer group p: ranks [p * vllm_tp, (p+1) * vllm_tp).
    SAE endpoint e: ranks [P*vllm_tp + e*sae_tp, P*vllm_tp + (e+1)*sae_tp).
"""

from __future__ import annotations

import torch.distributed as dist

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain
from sae_lens.shard_routing import ShardRoute, compute_routing_table


def hooks_for_pp_rank(pp_rank: int, pp_size: int, all_hooks: list[str]) -> list[str]:
    """Return the subset of hooks assigned to a given PP stage.

    Distributes hooks as evenly as possible; the first ``len(all_hooks) % pp_size``
    stages each get one extra hook.
    """
    n = len(all_hooks)
    base = n // pp_size
    extra = n % pp_size
    start = pp_rank * base + min(pp_rank, extra)
    count = base + (1 if pp_rank < extra else 0)
    return all_hooks[start : start + count]

# ---------------------------------------------------------------------------
# Module-level state (isolated from distributed.py)
# ---------------------------------------------------------------------------

_sae_runtime: SAERuntime | None = None
_routing_groups: list[dist.ProcessGroup] = []
_initialized: bool = False
_P: int = 0  # number of producers (vllm_dp_size)
_Q: int = 0  # number of routing consumers / SAE DP replicas (sae_dp_size)
_vllm_tp_size: int = 1
_sae_tp_size: int = 1
_sae_pp_size: int = 1
_num_sae_stage_endpoints: int = 0  # sae_dp_size * sae_pp_size

_is_producer: bool = False
_is_consumer: bool = False
_producer_idx: int = -1  # logical producer index; -1 if not a producer
_consumer_idx: int = -1  # routing consumer / SAE DP index; -1 if not a consumer
_sae_endpoint_idx: int = -1  # physical SAE endpoint index d * sae_pp + pp_rank
_vllm_tp_rank: int = -1  # rank within this rank's vLLM TP group; -1 if not a producer
_sae_tp_rank: int = -1   # rank within this rank's SAE TP group; -1 if not a consumer
_sae_pp_rank: int = -1   # PP stage index; -1 if not a consumer
_sae_dp_idx: int = -1    # DP replica index; -1 if not a consumer

# Explicit world-rank maps
_producer_world_ranks: dict[int, list[int]] = {}  # p -> [world ranks in TP group]
_sae_endpoint_world_ranks: dict[int, list[int]] = {}  # e -> [world ranks in TP group]
_producer_tp_root: dict[int, int] = {}            # p -> world rank of TP root (vllm_tp_rank=0)
_sae_endpoint_tp_root: dict[int, int] = {}        # e -> world rank of TP root (sae_tp_rank=0)

# Process groups
_vllm_tp_group: dist.ProcessGroup | None = None
_sae_tp_group: dist.ProcessGroup | None = None
_sae_tp_cpu_group: dist.ProcessGroup | None = None  # same SAE TP members, Gloo, checkpoint/export only
_sae_dp_group: dist.ProcessGroup | None = None
_sae_endpoint_p2p_groups: dict[int, dist.ProcessGroup] = {}  # endpoint_idx -> NCCL P2P group
_sae_dp_replica_group: dist.ProcessGroup | None = None  # all PP*TP ranks of this DP replica
_sae_dp_replica_root: int = -1  # world rank of the DP replica's PP-0 + TP-0
_sae_pp_root_group: dist.ProcessGroup | None = None  # TP-0 of every PP stage in this DP replica
_sae_pp_root_global: int = -1  # world rank of this DP replica's PP-0 + TP-0

# GPU direct streaming groups (created only when use_gpu_direct=True)
_streaming_nccl_groups: list[dist.ProcessGroup] = []  # indexed by pp_stage
_gloo_ctrl_group: dist.ProcessGroup | None = None  # vLLM TP root <-> SAE DP root
_pp_coord_groups: list[dist.ProcessGroup] = []  # indexed by pp_stage; SAE DP root <-> pp_stage TP-0

# Backwards-compatible aliases for code/tests that inspect module state directly.
_consumer_world_ranks: dict[int, list[int]] = _sae_endpoint_world_ranks
_consumer_tp_root: dict[int, int] = _sae_endpoint_tp_root
_consumer_p2p_groups: dict[int, dist.ProcessGroup] = _sae_endpoint_p2p_groups

_routing_table: list[ShardRoute] = []


def _reset() -> None:
    """Reset all module state.  Used in tests."""
    global _initialized, _P, _Q, _vllm_tp_size, _sae_tp_size, _sae_pp_size
    global _num_sae_stage_endpoints
    global _is_producer, _is_consumer, _producer_idx, _consumer_idx, _sae_endpoint_idx
    global _vllm_tp_rank, _sae_tp_rank, _sae_pp_rank, _sae_dp_idx
    global _producer_world_ranks, _sae_endpoint_world_ranks, _consumer_world_ranks
    global _producer_tp_root, _sae_endpoint_tp_root, _consumer_tp_root
    global _vllm_tp_group, _sae_tp_group, _sae_tp_cpu_group, _sae_dp_group
    global _sae_endpoint_p2p_groups, _consumer_p2p_groups, _routing_table
    global _sae_dp_replica_group, _sae_dp_replica_root
    global _sae_pp_root_group, _sae_pp_root_global
    global _streaming_nccl_groups, _gloo_ctrl_group, _pp_coord_groups

    global _sae_runtime, _routing_groups
    if dist.is_initialized():
        for group in reversed(_routing_groups):
            dist.destroy_process_group(group)
    _routing_groups = []
    if _sae_runtime is not None:
        _sae_runtime.close()
    _sae_runtime = None
    _initialized = False
    _P = _Q = _num_sae_stage_endpoints = 0
    _vllm_tp_size = _sae_tp_size = _sae_pp_size = 1
    _is_producer = _is_consumer = False
    _producer_idx = _consumer_idx = _sae_endpoint_idx = -1
    _vllm_tp_rank = _sae_tp_rank = _sae_pp_rank = _sae_dp_idx = -1
    _producer_world_ranks = {}
    _sae_endpoint_world_ranks = {}
    _consumer_world_ranks = _sae_endpoint_world_ranks
    _producer_tp_root = {}
    _sae_endpoint_tp_root = {}
    _consumer_tp_root = _sae_endpoint_tp_root
    _vllm_tp_group = None
    _sae_tp_group = None
    _sae_tp_cpu_group = None
    _sae_dp_group = None
    _sae_endpoint_p2p_groups = {}
    _consumer_p2p_groups = _sae_endpoint_p2p_groups
    _routing_table = []
    _sae_dp_replica_group = None
    _sae_dp_replica_root = -1
    _sae_pp_root_group = None
    _sae_pp_root_global = -1
    _streaming_nccl_groups = []
    _gloo_ctrl_group = None
    _pp_coord_groups = []


def _create_routing_group(ranks, backend):
    if backend == "nccl" and dist.get_backend() == "gloo":
        backend = "gloo"
    group = dist.new_group(ranks, backend=backend)
    if dist.get_rank() in ranks:
        _routing_groups.append(group)
    return group


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
    sae_pp_size: int = 1,
    use_gpu_direct: bool = False,
    build_routing_table: bool = True,
    runtime: SAERuntime | None = None,
    hook_names: tuple[str, ...] = (),
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
    sae_pp_size:
        Number of SAE pipeline stages.  PP stages are physical training endpoints
        for different hook subsets; they are not extra routing consumers.
    batch_size:
        Rows per producer per step (``store_batch_size_prompts * training_context_size``).
        Used to build the routing table.  Must be large enough that every connected
        producer→consumer edge receives at least 1 row.
    disjoint:
        When True, use a disjoint topology where producer and consumer ranks do not
        overlap.  Producer ranks are ``[0, P*vllm_tp_size)`` and SAE endpoint ranks are
        ``[P*vllm_tp_size, P*vllm_tp_size + Q*sae_pp_size*sae_tp_size)``, so
        ``world_size = P*vllm_tp_size + Q*sae_pp_size*sae_tp_size``.  When False
        (default), use the overlapping topology where
        ``world_size = max(P*vllm_tp, Q*sae_pp*sae_tp)``.
    use_gpu_direct:
        When True, create GPU direct streaming process groups (streaming_nccl_groups,
        gloo_ctrl_group, pp_coord_groups). Only applies when both P > 0 and Q > 0.
    build_routing_table:
        When False, skip shard-routing table and P2P route construction. SHM
        streaming uses its shared buffer protocol instead of shard routing.
    """
    global _initialized, _P, _Q, _vllm_tp_size, _sae_tp_size, _sae_pp_size
    global _num_sae_stage_endpoints
    global _is_producer, _is_consumer, _producer_idx, _consumer_idx, _sae_endpoint_idx
    global _vllm_tp_rank, _sae_tp_rank, _sae_pp_rank, _sae_dp_idx
    global _producer_world_ranks, _sae_endpoint_world_ranks, _consumer_world_ranks
    global _producer_tp_root, _sae_endpoint_tp_root, _consumer_tp_root
    global _vllm_tp_group, _sae_tp_group, _sae_tp_cpu_group, _sae_dp_group
    global _sae_endpoint_p2p_groups, _consumer_p2p_groups, _routing_table
    global _sae_dp_replica_group, _sae_dp_replica_root
    global _sae_pp_root_group, _sae_pp_root_global
    global _streaming_nccl_groups, _gloo_ctrl_group, _pp_coord_groups

    assert dist.is_initialized(), "Call dist.init_process_group() before init_distributed_v2()"

    num_sae_stage_endpoints = Q * sae_pp_size
    world_size = dist.get_world_size()
    if runtime is not None:
        if P * vllm_tp_size > world_size:
            raise ValueError("Producer ranks are outside the routing world")
        if disjoint and any(r < P * vllm_tp_size for d in runtime.domains for r in d.ranks):
            raise ValueError("Disjoint routing cannot overlap producer and SAE ranks")
    elif disjoint:
        expected = P * vllm_tp_size + num_sae_stage_endpoints * sae_tp_size
        assert world_size == expected, (
            f"world_size={world_size} != P*vllm_tp + sae_endpoints*sae_tp={expected} "
            f"(P={P}, vllm_tp={vllm_tp_size}, Q={Q}, sae_pp={sae_pp_size}, sae_tp={sae_tp_size})"
        )
    else:
        expected = max(P * vllm_tp_size, num_sae_stage_endpoints * sae_tp_size)
        assert world_size == expected, (
            f"world_size={world_size} != max(P*vllm_tp, sae_endpoints*sae_tp)={expected} "
            f"(P={P}, vllm_tp={vllm_tp_size}, Q={Q}, sae_pp={sae_pp_size}, sae_tp={sae_tp_size})"
        )

    rank = dist.get_rank()
    _P = P
    _Q = Q
    _vllm_tp_size = vllm_tp_size
    _sae_tp_size = sae_tp_size
    _sae_pp_size = sae_pp_size
    _num_sae_stage_endpoints = num_sae_stage_endpoints

    # --- Build explicit rank maps ---
    for p in range(P):
        ranks = list(range(p * vllm_tp_size, (p + 1) * vllm_tp_size))
        _producer_world_ranks[p] = ranks
        _producer_tp_root[p] = ranks[0]

    # Runtime owns training-domain initialization. Routing consumes actual
    # Megatron membership reports; it never derives SAE TP/DP members.
    global _sae_runtime
    _sae_runtime = runtime or SAERuntime.from_layout(
        dp_size=Q, tp_size=sae_tp_size, placement_size=sae_pp_size,
        offset=P * vllm_tp_size if disjoint else 0, hooks=hook_names,
        backend="gloo" if dist.get_backend() == "gloo" else "nccl",
    )
    if len(_sae_runtime.endpoints) != num_sae_stage_endpoints:
        raise ValueError("SAE runtime endpoint count differs from routing configuration")
    for endpoint in _sae_runtime.endpoints:
        _sae_endpoint_world_ranks[endpoint.index] = list(endpoint.tp_ranks)
        _sae_endpoint_tp_root[endpoint.index] = endpoint.receive_rank

    # --- Determine this rank's role by membership ---
    _is_producer = False
    _is_consumer = False
    _producer_idx = -1
    _consumer_idx = -1
    _sae_endpoint_idx = -1
    _vllm_tp_rank = -1
    _sae_tp_rank = -1
    _sae_pp_rank = -1
    _sae_dp_idx = -1

    for p, ranks in _producer_world_ranks.items():
        if rank in ranks:
            _is_producer = True
            _producer_idx = p
            _vllm_tp_rank = ranks.index(rank)

    if _sae_runtime.local is not None:
        context = _sae_runtime.require_local()
        endpoint = next(e for e in _sae_runtime.endpoints if rank in e.tp_ranks)
        _is_consumer = True
        _consumer_idx = _sae_dp_idx = context.dp_rank
        _sae_endpoint_idx = endpoint.index
        _sae_tp_rank = context.tp_rank
        _sae_pp_rank = endpoint.placement_index
        _sae_tp_group = context.tp_group
        _sae_tp_cpu_group = context.tp_cpu_group
        _sae_dp_group = context.dp_group

    # --- Create P vLLM TP groups (NCCL) ---
    for p in range(P):
        ranks = _producer_world_ranks[p]
        grp = _create_routing_group(ranks, backend="nccl")
        if _is_producer and _producer_idx == p:
            _vllm_tp_group = grp

    # --- Create per-DP-replica groups: all PP*TP ranks of one DP replica ---
    # Kept for existing callers that need the full replica group.
    if Q > 0:
        for d in range(Q):
            members = [r for e in _sae_runtime.endpoints if e.replica_index == d
                       for r in e.tp_ranks]
            grp = _create_routing_group(members, backend="nccl")
            if _is_consumer and _sae_dp_idx == d:
                _sae_dp_replica_group = grp
                _sae_dp_replica_root = members[0]

    # --- PP-root groups: TP-0 of every PP stage in one DP replica ---
    # Chunk-index broadcasts must not include TP followers: followers take a
    # different provider path and only participate in their SAE-TP broadcast.
    if Q > 0 and sae_pp_size > 1:
        for d in range(Q):
            members = [e.receive_rank for e in _sae_runtime.endpoints if e.replica_index == d]
            grp = _create_routing_group(members, backend="nccl")
            if _is_consumer and _sae_dp_idx == d and _sae_tp_rank == 0:
                _sae_pp_root_group = grp
                _sae_pp_root_global = members[0]

    # --- Compute routing table: partition rows across DP replicas only ---
    # SHM streaming has its own shared-buffer allocation protocol; it still
    # needs the process groups above for role/TP coordination, but no row routes.
    _routing_table = (
        compute_routing_table(P, Q, batch_size)
        if build_routing_table and P > 0 and Q > 0
        else []
    )

    # --- Create one P2P group per physical SAE endpoint ---
    # Each endpoint (d*sae_pp+s) gets its TP root and all producer TP roots
    # connected to its DP consumer. PP stages in the same DP replica share routes.
    if build_routing_table:
        for endpoint_idx in range(num_sae_stage_endpoints):
            d = get_endpoint(endpoint_idx).replica_index
            sources = {r.producer_idx for r in _routing_table if r.consumer_idx == d}
            p2p_members = sorted(
                {_sae_endpoint_tp_root[endpoint_idx]}
                | {_producer_tp_root[p] for p in sources}
            )
            grp = _create_routing_group(p2p_members, backend="nccl")
            if rank in p2p_members:
                _sae_endpoint_p2p_groups[endpoint_idx] = grp

    # --- GPU direct streaming groups (created only when use_gpu_direct=True) ---
    if use_gpu_direct and P > 0 and Q > 0:
        # streaming_nccl_group[pp_stage]: vLLM TP root + all SAE TP ranks for this PP stage
        for pp_stage in range(sae_pp_size):
            members = [_producer_tp_root[0]]  # vLLM TP root (only producer 0 in MVP)
            for d in range(Q):
                for tp_r in range(sae_tp_size):
                    endpoint_idx = d * sae_pp_size + pp_stage
                    members.append(_sae_endpoint_world_ranks[endpoint_idx][tp_r])
            grp = _create_routing_group(members, backend="nccl")
            _streaming_nccl_groups.append(grp)
            if rank in members:
                # Store for later access by pp_stage
                pass

        # gloo_ctrl_group: vLLM TP root <-> SAE DP root (CPU tensors)
        # Only create if both ranks exist in this process group
        sae_dp_root = _sae_endpoint_world_ranks[0][0]  # First SAE endpoint's TP root
        gloo_members = [_producer_tp_root[0], sae_dp_root]
        if all(m < world_size for m in gloo_members):
            gloo_grp = _create_routing_group(gloo_members, backend="gloo")
            if rank in gloo_members:
                _gloo_ctrl_group = gloo_grp

        # pp_coord_group[pp_stage]: SAE DP root (PP-0 TP-0) <-> pp_stage TP-0
        for pp_stage in range(sae_pp_size):
            if pp_stage == 0:
                # Single member (DP root = PP-0 TP-0), skip
                _pp_coord_groups.append(None)
            else:
                # DP root + pp_stage TP-0 of each DP replica
                members = [sae_dp_root]
                for d in range(Q):
                    endpoint_idx = d * sae_pp_size + pp_stage
                    members.append(_sae_endpoint_tp_root[endpoint_idx])
                if all(m < world_size for m in members):
                    grp = _create_routing_group(members, backend="nccl")
                    _pp_coord_groups.append(grp)
                    if rank in members:
                        pass
                else:
                    _pp_coord_groups.append(None)

    _consumer_world_ranks = _sae_endpoint_world_ranks
    _consumer_tp_root = _sae_endpoint_tp_root
    _consumer_p2p_groups = _sae_endpoint_p2p_groups

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
    """Return this rank's SAE DP consumer index used by shard routing."""
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


def get_sae_pp_rank() -> int:
    return _sae_pp_rank


def get_sae_pp_size() -> int:
    return _sae_pp_size


def get_sae_dp_idx() -> int:
    """Return this rank's SAE DP replica index used by shard routing."""
    return _sae_dp_idx


def get_sae_endpoint_idx() -> int:
    """Return this rank's physical SAE endpoint index ``dp_idx * pp_size + pp_rank``."""
    return _sae_endpoint_idx


def get_num_sae_stage_endpoints() -> int:
    """Return the number of physical SAE training endpoints (DP replicas x PP stages)."""
    return _num_sae_stage_endpoints


def get_q_total() -> int:
    """Deprecated alias for ``get_num_sae_stage_endpoints()``.

    This is not the routing consumer count.  Routing consumers are SAE DP replicas
    and are returned by ``get_sae_dp_size()``.
    """
    return _num_sae_stage_endpoints


def get_routing_table() -> list[ShardRoute]:
    return _routing_table


def get_vllm_tp_group() -> dist.ProcessGroup | None:
    return _vllm_tp_group


def get_sae_tp_group() -> dist.ProcessGroup | None:
    return _sae_tp_group


def get_sae_tp_cpu_group() -> dist.ProcessGroup | None:
    """Gloo process group with exactly the same members as this SAE TP endpoint."""
    return _sae_tp_cpu_group


def get_sae_dp_group() -> dist.ProcessGroup | None:
    return _sae_dp_group


def get_sae_dp_replica_group() -> dist.ProcessGroup | None:
    """Process group spanning all PP*TP ranks of this rank's DP replica.

    Returns None if this rank is not a consumer or the run is not initialized.
    The group is created even when ``sae_pp_size == 1`` (members = the TP group
    of the single endpoint), which lets callers use it unconditionally.
    """
    return _sae_dp_replica_group


def get_sae_dp_replica_root_global_rank() -> int:
    """World rank of the DP replica's root (PP-0 + TP-0).

    Returns -1 if this rank is not a consumer.
    """
    return _sae_dp_replica_root


def get_sae_pp_root_group() -> dist.ProcessGroup | None:
    """TP-0-only group spanning PP stages inside this SAE-DP replica."""
    return _sae_pp_root_group


def get_sae_pp_root_global_rank() -> int:
    """World rank of PP-0 + TP-0 for the TP-0-only PP coordination group."""
    return _sae_pp_root_global


def get_p2p_group(endpoint_idx: int) -> dist.ProcessGroup:
    """Return the NCCL P2P group for the given SAE endpoint.

    Raises ``KeyError`` if this rank is not a member of that endpoint's P2P group.
    """
    return _sae_endpoint_p2p_groups[endpoint_idx]


def get_producer_tp_root(p: int) -> int:
    return _producer_tp_root[p]


def get_consumer_tp_root(endpoint_idx: int) -> int:
    """Return the SAE TP root for a physical endpoint.

    The function name is kept for compatibility.  When ``sae_pp_size > 1``, pass
    an endpoint index, not a routing consumer index.
    """
    return _sae_endpoint_tp_root[endpoint_idx]


def get_streaming_nccl_group(pp_stage: int) -> dist.ProcessGroup:
    """Return the NCCL group for GPU direct streaming at a given PP stage.

    Members: vLLM TP root + all SAE TP ranks for this PP stage.
    Raises IndexError if pp_stage is out of range or groups not initialized.
    """
    return _streaming_nccl_groups[pp_stage]


def get_gloo_ctrl_group() -> dist.ProcessGroup | None:
    """Return the Gloo control group for GPU direct streaming.

    Members: vLLM TP root <-> SAE DP root (CPU tensors).
    Returns None if GPU direct groups not initialized.
    """
    return _gloo_ctrl_group


def get_pp_coord_group(pp_stage: int) -> dist.ProcessGroup | None:
    """Return the NCCL coordination group for a given PP stage.

    Members: SAE DP root (PP-0 TP-0) <-> pp_stage TP-0 of each DP replica.
    For pp_stage=0, returns None (single member, no-op).
    Raises IndexError if pp_stage is out of range or groups not initialized.
    """
    return _pp_coord_groups[pp_stage]


def get_sae_runtime() -> SAERuntime | None:
    return _sae_runtime


def producer_helper_ranks() -> tuple[int, ...]:
    """Ranks producing activations without a local SAE training unit."""
    assert _sae_runtime is not None
    consumers = {r for domain in _sae_runtime.domains for r in domain.ranks}
    return tuple(sorted(r for members in _producer_world_ranks.values() for r in members if r not in consumers))


def control_producer_helpers(command: str, checkpoint_path: str | None = None) -> None:
    """Synchronous static control; the receiver map still comes from runtime."""
    runtime = _sae_runtime
    if runtime is None or not runtime.endpoints:
        return
    if dist.get_rank() != runtime.endpoints[0].receive_rank:
        return
    for rank in producer_helper_ranks():
        monitor = getattr(runtime, "failure_monitor", None)
        if monitor is not None:
            monitor.send_command(rank, command, checkpoint_path)
            continue
        dist.send_object_list([(command, checkpoint_path)], dst=rank, group=runtime.control_group)
        if command == "checkpoint":
            reply = [None]
            dist.recv_object_list(reply, src=rank, group=runtime.control_group)
            if reply[0] != "saved":
                raise RuntimeError(f"Producer rank {rank} did not finish its checkpoint")


def wait_producer_control():
    runtime = _sae_runtime
    assert runtime is not None
    monitor = getattr(runtime, "failure_monitor", None)
    if monitor is not None:
        return monitor.receive_command()
    command = [None]
    dist.recv_object_list(command, src=runtime.endpoints[0].receive_rank, group=runtime.control_group)
    return command[0]


def acknowledge_producer_checkpoint():
    assert _sae_runtime is not None
    monitor = getattr(_sae_runtime, "failure_monitor", None)
    if monitor is not None:
        monitor.acknowledge_checkpoint()
        return
    dist.send_object_list(["saved"], dst=_sae_runtime.endpoints[0].receive_rank, group=_sae_runtime.control_group)


def get_endpoint(endpoint_idx: int):
    if _sae_runtime is None:
        raise RuntimeError("SAE routing has not been initialized")
    return _sae_runtime.endpoints[endpoint_idx]


def initialize_sae_routing(
    *, P: int, Q: int, vllm_tp_size: int, sae_tp_size: int,
    batch_size: int, hook_names: tuple[str, ...], sae_pp_size: int = 1,
    disjoint: bool = False, training_domains: tuple[SAETrainingDomain, ...] | None = None,
) -> SAERuntime:
    """Static routing entry: initialize Megatron domains and register receiver metadata.

    All routing-world ranks call this once, including producer-only ranks.
    Activation slicing/filtering/buffering remain in ActivationsStore.
    """
    if _initialized:
        raise RuntimeError("SAE routing is already initialized; close the previous runtime first")
    if len(set(hook_names)) != len(hook_names) or (Q > 0 and not hook_names):
        raise ValueError("Static SAE routing requires distinct, nonempty hook names")
    runtime = None
    if training_domains is not None:
        assigned = tuple(h for d in training_domains for h in d.hooks)
        if set(assigned) != set(hook_names) or len(assigned) != len(hook_names):
            raise ValueError("Training domains must assign every hook exactly once")
        if len(training_domains) != sae_pp_size:
            raise ValueError("Placement count differs from training domains")
        if any(d.tp_size != sae_tp_size or len(d.ranks) != Q * sae_tp_size for d in training_domains):
            raise ValueError("Training domains differ from configured TP/DP sizes")
        runtime = SAERuntime(training_domains, backend="gloo" if dist.get_backend() == "gloo" else "nccl")
    try:
        init_distributed_v2(
            P=P, Q=Q, vllm_tp_size=vllm_tp_size, sae_tp_size=sae_tp_size,
            batch_size=batch_size, sae_pp_size=sae_pp_size, disjoint=disjoint,
            runtime=runtime, hook_names=hook_names,
        )
    except BaseException:
        _reset()
        if runtime is not None:
            runtime.close()
        raise
    assert _sae_runtime is not None
    return _sae_runtime


def close_sae_routing() -> None:
    """Close transport and SAE groups collectively, preserving the default world."""
    _reset()
