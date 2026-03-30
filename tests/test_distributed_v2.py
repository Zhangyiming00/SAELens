"""Tests for sae_lens.distributed_v2.init_distributed_v2.

All tests mock torch.distributed so no GPU or NCCL is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

import sae_lens.distributed_v2 as v2_mod
from sae_lens.distributed_v2 import init_distributed_v2
from sae_lens.shard_routing import compute_routing_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state():
    v2_mod._reset()
    yield
    v2_mod._reset()


def _run_init(
    rank: int,
    world_size: int,
    P: int,
    Q: int,
    vllm_tp: int = 1,
    sae_tp: int = 1,
    batch_size: int = 60,
) -> list:
    """Call init_distributed_v2 for the given rank, return list of new_group call args."""
    mock_group = MagicMock()
    new_group_calls: list = []

    def _new_group(ranks, backend="nccl"):
        new_group_calls.append((sorted(ranks), backend))
        return mock_group

    with (
        patch("sae_lens.distributed_v2.dist.is_initialized", return_value=True),
        patch("sae_lens.distributed_v2.dist.get_world_size", return_value=world_size),
        patch("sae_lens.distributed_v2.dist.get_rank", return_value=rank),
        patch("sae_lens.distributed_v2.dist.new_group", side_effect=_new_group),
    ):
        init_distributed_v2(
            P=P, Q=Q,
            vllm_tp_size=vllm_tp,
            sae_tp_size=sae_tp,
            batch_size=batch_size,
        )
    return new_group_calls


# ---------------------------------------------------------------------------
# World-size validation
# ---------------------------------------------------------------------------


def test_init_v2_world_size_formula_fan_in() -> None:
    """P=3, Q=1, vllm_tp=sae_tp=1 → world=max(3,1)=3."""
    calls = _run_init(0, 3, P=3, Q=1)
    assert v2_mod._P == 3
    assert v2_mod._Q == 1


def test_init_v2_world_size_formula_fan_out() -> None:
    """P=2, Q=3, vllm_tp=sae_tp=1 → world=max(2,3)=3."""
    calls = _run_init(0, 3, P=2, Q=3)
    assert v2_mod._P == 2
    assert v2_mod._Q == 3


def test_init_v2_world_size_formula_matched() -> None:
    """P=2, Q=2, vllm_tp=sae_tp=2 → world=max(4,4)=4."""
    calls = _run_init(0, 4, P=2, Q=2, vllm_tp=2, sae_tp=2)
    assert v2_mod._P == 2
    assert v2_mod._Q == 2


def test_init_v2_wrong_world_size_raises() -> None:
    with pytest.raises(AssertionError):
        _run_init(0, 5, P=2, Q=3)  # expected world=3


def test_init_v2_no_divisibility_check() -> None:
    """5:3 should succeed without divisibility constraint."""
    _run_init(0, 5, P=5, Q=3)  # world = max(5,3) = 5


# ---------------------------------------------------------------------------
# Producer / consumer role assignment
# ---------------------------------------------------------------------------


def test_init_v2_producer_only() -> None:
    """P=2, Q=3; rank=1 is in producers [0,2) but not consumers [0,3). Wait, 1 < 3 so dual-role."""
    # rank=1: producer rank (< 2*1=2) AND consumer rank (< 3*1=3) → dual-role
    _run_init(1, 3, P=2, Q=3)
    assert v2_mod._is_producer
    assert v2_mod._is_consumer  # dual-role
    assert v2_mod._producer_idx == 1
    assert v2_mod._consumer_idx == 1


def test_init_v2_consumer_only() -> None:
    """P=2, Q=3; rank=2 is NOT in producers [0,2) but IS in consumers [0,3)."""
    _run_init(2, 3, P=2, Q=3)
    assert not v2_mod._is_producer
    assert v2_mod._is_consumer
    assert v2_mod._consumer_idx == 2


def test_init_v2_dual_role_rank() -> None:
    """P=3, Q=3; rank=0 is in both producer group 0 and consumer group 0."""
    _run_init(0, 3, P=3, Q=3)
    assert v2_mod._is_producer
    assert v2_mod._is_consumer
    assert v2_mod._producer_idx == 0
    assert v2_mod._consumer_idx == 0


def test_init_v2_producer_only_large_producer_block() -> None:
    """P=3, Q=1, vllm_tp=1, sae_tp=1; rank=2 is producer-only (not in [0,1))."""
    _run_init(2, 3, P=3, Q=1)
    assert v2_mod._is_producer
    assert not v2_mod._is_consumer
    assert v2_mod._producer_idx == 2


# ---------------------------------------------------------------------------
# Group creation count
# ---------------------------------------------------------------------------


def test_init_v2_new_group_total_count_3_1() -> None:
    """P=3, Q=1: P + Q + sae_tp_size + Q = 3+1+1+1 = 6 new_group calls."""
    calls = _run_init(0, 3, P=3, Q=1)
    assert len(calls) == 6


def test_init_v2_new_group_total_count_2_3() -> None:
    """P=2, Q=3: P + Q + sae_tp_size + Q = 2+3+1+3 = 9 new_group calls."""
    calls = _run_init(0, 3, P=2, Q=3)
    assert len(calls) == 9


def test_init_v2_new_group_total_count_with_tp() -> None:
    """P=2, Q=2, vllm_tp=2, sae_tp=2: 2+2+2+2 = 8 new_group calls."""
    calls = _run_init(0, 4, P=2, Q=2, vllm_tp=2, sae_tp=2)
    assert len(calls) == 8


# ---------------------------------------------------------------------------
# Gloo groups are per-consumer and deduplicated
# ---------------------------------------------------------------------------


def test_init_v2_gloo_groups_count_equals_Q() -> None:
    """Exactly Q Gloo groups are created."""
    calls = _run_init(0, 3, P=2, Q=3)
    gloo_calls = [c for c in calls if c[1] == "gloo"]
    assert len(gloo_calls) == 3  # Q = 3


def test_init_v2_gloo_groups_deduplicated_no_duplicate_rank() -> None:
    """No rank appears twice in any P2P group."""
    calls = _run_init(0, 3, P=2, Q=3)
    gloo_calls = [c for c in calls if c[1] == "gloo"]
    for ranks, _ in gloo_calls:
        assert len(ranks) == len(set(ranks)), f"Duplicate ranks in gloo group: {ranks}"


def test_init_v2_p2p_membership_2_3() -> None:
    """P=2, Q=3: verify Gloo group membership is correct for each consumer."""
    calls = _run_init(0, 3, P=2, Q=3)
    gloo_calls = [c for c in calls if c[1] == "gloo"]
    # P=2, Q=3, vllm_tp=1, sae_tp=1
    # producer tp roots: p0→rank0, p1→rank1
    # consumer tp roots: c0→rank0, c1→rank1, c2→rank2
    # Routing (batch_size=60): p0→c0,c1; p1→c1,c2
    # Group for c0: sources={p0}, members={c0_root=0, p0_root=0} → deduplicated → {0}
    # Group for c1: sources={p0,p1}, members={c1_root=1, p0_root=0, p1_root=1} → dedup → {0,1}
    # Group for c2: sources={p1}, members={c2_root=2, p1_root=1} → {1,2}
    group_members = sorted([sorted(c[0]) for c in gloo_calls])
    assert [0] in group_members
    assert [0, 1] in group_members
    assert [1, 2] in group_members


def test_init_v2_sae_dp_group_spans_all_consumers() -> None:
    """SAE DP groups contain one rank per consumer."""
    calls = _run_init(0, 3, P=2, Q=3)
    # SAE DP groups are NCCL groups with sae_tp_size members (1 here), Q groups total
    # actually sae_tp_size=1, so 1 SAE DP group with 3 members (one per consumer)
    nccl_calls = [c for c in calls if c[1] == "nccl"]
    # P vLLM TP groups + Q SAE TP groups + sae_tp_size DP groups = 2+3+1 = 6 nccl groups
    # DP group should contain consumer tp roots: [0, 1, 2]
    dp_group_candidates = [c for c in nccl_calls if sorted(c[0]) == [0, 1, 2]]
    assert len(dp_group_candidates) == 1


def test_init_v2_producer_only_rank_not_in_sae_tp_group() -> None:
    """A producer-only rank (rank 2 in P=3, Q=1) is not a consumer, so no sae_tp_group stored."""
    _run_init(2, 3, P=3, Q=1)
    assert v2_mod._sae_tp_group is None
    assert v2_mod._sae_dp_group is None


# ---------------------------------------------------------------------------
# Routing table is computed
# ---------------------------------------------------------------------------


def test_init_v2_routing_table_set() -> None:
    _run_init(0, 3, P=2, Q=3, batch_size=60)
    expected = compute_routing_table(2, 3, 60)
    assert v2_mod._routing_table == expected


# ---------------------------------------------------------------------------
# TP-mismatch cases
# ---------------------------------------------------------------------------


def test_init_v2_vllm_tp2_sae_tp1_rank0() -> None:
    """P=2, Q=3, vllm_tp=2, sae_tp=1; world=max(4,3)=4; rank=0 is producer TP root."""
    _run_init(0, 4, P=2, Q=3, vllm_tp=2, sae_tp=1)
    assert v2_mod._is_producer
    assert v2_mod._producer_idx == 0
    assert v2_mod._vllm_tp_rank == 0  # TP root
    # rank 0 < Q*sae_tp=3 → also consumer
    assert v2_mod._is_consumer
    assert v2_mod._consumer_idx == 0


def test_init_v2_vllm_tp2_sae_tp1_rank1() -> None:
    """rank=1 is producer TP non-root (vllm_tp_rank=1) and consumer c1."""
    _run_init(1, 4, P=2, Q=3, vllm_tp=2, sae_tp=1)
    assert v2_mod._is_producer
    assert v2_mod._vllm_tp_rank == 1  # non-root
    assert v2_mod._producer_idx == 0  # same TP group as rank 0
    assert v2_mod._is_consumer
    assert v2_mod._consumer_idx == 1


def test_init_v2_vllm_tp2_sae_tp1_rank3() -> None:
    """rank=3 is producer-only (P*vllm_tp=4 > rank=3, consumer boundary=3)."""
    _run_init(3, 4, P=2, Q=3, vllm_tp=2, sae_tp=1)
    assert v2_mod._is_producer
    assert not v2_mod._is_consumer  # rank 3 >= Q*sae_tp=3


def test_init_v2_vllm_tp1_sae_tp2() -> None:
    """P=2, Q=3, vllm_tp=1, sae_tp=2; world=max(2,6)=6; rank=0 is dual-role."""
    _run_init(0, 6, P=2, Q=3, vllm_tp=1, sae_tp=2)
    assert v2_mod._is_producer
    assert v2_mod._is_consumer
    assert v2_mod._producer_idx == 0
    assert v2_mod._consumer_idx == 0
    assert v2_mod._sae_tp_rank == 0  # consumer TP root


def test_init_v2_vllm_tp1_sae_tp2_consumer_follower() -> None:
    """rank=1: P=2 vllm_tp=1 → producer p1; Q=3 sae_tp=2 → consumer c0 follower (sae_tp_rank=1)."""
    # Producer ranks: [0, P*vllm_tp) = [0, 2); rank 1 is p1, vllm_tp_rank=0
    # Consumer ranks: [0, Q*sae_tp) = [0, 6); rank 1 is in consumer group c0=[0,1], sae_tp_rank=1
    _run_init(1, 6, P=2, Q=3, vllm_tp=1, sae_tp=2)
    assert v2_mod._is_producer  # rank 1 < 2*1=2
    assert v2_mod._producer_idx == 1
    assert v2_mod._vllm_tp_rank == 0  # only rank in its TP group
    assert v2_mod._is_consumer  # rank 1 < 3*2=6
    assert v2_mod._consumer_idx == 0  # consumer group c0 = [0, 1]
    assert v2_mod._sae_tp_rank == 1  # follower
