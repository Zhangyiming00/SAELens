"""Tests for sae_lens.distributed: all 4 (sae_tp, vllm_tp) combinations.

All tests mock torch.distributed so no GPU or NCCL is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import sae_lens.distributed as distributed_mod
from sae_lens.distributed import init_distributed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_distributed_state() -> None:
    distributed_mod._sae_tp_group = None
    distributed_mod._sae_dp_group = None
    distributed_mod._vllm_tp_group = None
    distributed_mod._worker_group = None
    distributed_mod._worker_cpu_group = None
    distributed_mod._vllm_dp_p2p_group = None
    distributed_mod._sae_tp_rank = 0
    distributed_mod._sae_tp_size = 1
    distributed_mod._sae_dp_rank = 0
    distributed_mod._sae_dp_size = 1
    distributed_mod._worker_rank = 0
    distributed_mod._worker_size = 1
    distributed_mod._cluster_block = 1
    distributed_mod._vllm_tp_rank = -1
    distributed_mod._vllm_tp_size = 1
    distributed_mod._vllm_dp_rank = 0
    distributed_mod._vllm_dp_size = 1
    distributed_mod._sae_active = True
    distributed_mod._vllm_active = True
    distributed_mod._layout_mode = "local_overlap"


@pytest.fixture(autouse=True)
def reset_state():
    _reset_distributed_state()
    yield
    _reset_distributed_state()


def _run_init(rank: int, world_size: int, sae_tp: int, vllm_tp: int) -> None:
    """Call init_distributed() simulating the given rank, with mocked dist."""
    mock_group = MagicMock()
    with (
        patch("sae_lens.distributed.dist.is_initialized", return_value=True),
        patch("sae_lens.distributed.dist.get_world_size", return_value=world_size),
        patch("sae_lens.distributed.dist.get_rank", return_value=rank),
        patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
    ):
        init_distributed(sae_tp_size=sae_tp, vllm_tp_size=vllm_tp)


# ---------------------------------------------------------------------------
# Config 1: sae_tp=1, vllm_tp=1  (trivial single-rank case)
# ---------------------------------------------------------------------------


def test_config1_rank0_both_active():
    _run_init(rank=0, world_size=1, sae_tp=1, vllm_tp=1)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_tp_rank == 0
    assert distributed_mod._vllm_tp_rank == 0
    assert distributed_mod._sae_dp_rank == 0
    assert distributed_mod._worker_rank == 0


def test_config1_no_split_roles():
    _run_init(rank=0, world_size=1, sae_tp=1, vllm_tp=1)
    assert distributed_mod._vllm_tp_size == distributed_mod._sae_tp_size


def test_config1_both_groups_set():
    _run_init(rank=0, world_size=1, sae_tp=1, vllm_tp=1)
    assert distributed_mod._sae_tp_group is not None
    assert distributed_mod._vllm_tp_group is not None


# ---------------------------------------------------------------------------
# Config 2: sae_tp=2, vllm_tp=2  (shared TP, no split roles)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 1])
def test_config2_both_ranks_fully_active(rank: int):
    _run_init(rank=rank, world_size=2, sae_tp=2, vllm_tp=2)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_tp_rank == rank
    assert distributed_mod._vllm_tp_rank == rank


@pytest.mark.parametrize("rank", [0, 1])
def test_config2_both_groups_set(rank: int):
    _run_init(rank=rank, world_size=2, sae_tp=2, vllm_tp=2)
    assert distributed_mod._sae_tp_group is not None
    assert distributed_mod._vllm_tp_group is not None


def test_config2_no_split_roles():
    _run_init(rank=0, world_size=2, sae_tp=2, vllm_tp=2)
    assert distributed_mod._vllm_tp_size == distributed_mod._sae_tp_size


# ---------------------------------------------------------------------------
# Config 3: sae_tp=1, vllm_tp=2  (rank 1 is vLLM-only / helper)
# ---------------------------------------------------------------------------


def test_config3_rank0_both_active():
    _run_init(rank=0, world_size=2, sae_tp=1, vllm_tp=2)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_tp_rank == 0
    assert distributed_mod._vllm_tp_rank == 0


def test_config3_rank1_vllm_only():
    _run_init(rank=1, world_size=2, sae_tp=1, vllm_tp=2)
    assert distributed_mod._sae_active is False
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_tp_rank == -1
    assert distributed_mod._vllm_tp_rank == 1


def test_config3_rank1_has_no_sae_group():
    _run_init(rank=1, world_size=2, sae_tp=1, vllm_tp=2)
    # Rank 1 is not SAE-active, so it must not own an SAE TP group.
    assert distributed_mod._sae_tp_group is None


def test_config3_split_roles():
    _run_init(rank=0, world_size=2, sae_tp=1, vllm_tp=2)
    assert distributed_mod._vllm_tp_size != distributed_mod._sae_tp_size


# ---------------------------------------------------------------------------
# Config 4: sae_tp=2, vllm_tp=1  (rank 1 is SAE-only; deadlock-risk case)
# ---------------------------------------------------------------------------


def test_config4_rank0_both_active():
    _run_init(rank=0, world_size=2, sae_tp=2, vllm_tp=1)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_tp_rank == 0
    assert distributed_mod._vllm_tp_rank == 0


def test_config4_rank1_sae_only():
    _run_init(rank=1, world_size=2, sae_tp=2, vllm_tp=1)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is False
    assert distributed_mod._sae_tp_rank == 1
    assert distributed_mod._vllm_tp_rank == -1


def test_config4_rank1_has_no_vllm_group():
    _run_init(rank=1, world_size=2, sae_tp=2, vllm_tp=1)
    # Rank 1 is not vLLM-active, so it must not own a vLLM TP group.
    assert distributed_mod._vllm_tp_group is None


def test_config4_split_roles():
    _run_init(rank=0, world_size=2, sae_tp=2, vllm_tp=1)
    assert distributed_mod._vllm_tp_size != distributed_mod._sae_tp_size


# ---------------------------------------------------------------------------
# preinit_vllm_parallel_state: ALL ranks participate in exactly 12
# dist.new_group() calls (6 groups × NCCL + Gloo), regardless of membership.
# ---------------------------------------------------------------------------


def _run_preinit(
    rank: int, vllm_world_ranks: list[int], vllm_tp_size: int
) -> list[tuple[tuple[int, ...], str | None]]:
    """Call preinit_vllm_parallel_state() and return (ranks, backend) for each call."""
    import vllm.distributed.parallel_state as ps

    new_group_calls: list[tuple[tuple[int, ...], str | None]] = []

    def fake_new_group(ranks, backend=None, **kwargs):
        new_group_calls.append((tuple(ranks), backend))
        return MagicMock()

    saved = (ps._WORLD, ps._TP, ps._DCP, ps._PCP, ps._PP, ps._DP)
    ps._WORLD = ps._TP = ps._DCP = ps._PCP = ps._PP = ps._DP = None
    try:
        with (
            patch("torch.distributed.new_group", side_effect=fake_new_group),
            patch("torch.distributed.get_rank", return_value=rank),
            patch("torch.distributed.is_initialized", return_value=True),
        ):
            ps.preinit_vllm_parallel_state(vllm_world_ranks, vllm_tp_size, local_rank=rank)
    finally:
        ps._WORLD, ps._TP, ps._DCP, ps._PCP, ps._PP, ps._DP = saved

    return new_group_calls


def test_preinit_member_rank_makes_12_new_group_calls():
    calls = _run_preinit(rank=0, vllm_world_ranks=[0], vllm_tp_size=1)
    assert len(calls) == 12


def test_preinit_nonmember_rank_also_makes_12_new_group_calls():
    # Rank 1 is not in vllm_world_ranks=[0] but must still participate.
    calls = _run_preinit(rank=1, vllm_world_ranks=[0], vllm_tp_size=1)
    assert len(calls) == 12


def test_preinit_member_gets_valid_coordinator():
    import vllm.distributed.parallel_state as ps

    saved = (ps._WORLD, ps._TP, ps._DCP, ps._PCP, ps._PP, ps._DP)
    ps._WORLD = ps._TP = ps._DCP = ps._PCP = ps._PP = ps._DP = None
    try:
        with (
            patch("torch.distributed.new_group", return_value=MagicMock()),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.is_initialized", return_value=True),
        ):
            ps.preinit_vllm_parallel_state([0], vllm_tp_size=1, local_rank=0)
        # Member rank 0 should have a real coordinator (world_size == 1).
        assert ps._WORLD is not None
        assert ps._WORLD.world_size == 1
        assert ps._TP.world_size == 1
        assert ps._PP.world_size == 1
    finally:
        ps._WORLD, ps._TP, ps._DCP, ps._PCP, ps._PP, ps._DP = saved


def test_preinit_nonmember_gets_null_coordinator():
    import vllm.distributed.parallel_state as ps

    saved = (ps._WORLD, ps._TP, ps._DCP, ps._PCP, ps._PP, ps._DP)
    ps._WORLD = ps._TP = ps._DCP = ps._PCP = ps._PP = ps._DP = None
    try:
        with (
            patch("torch.distributed.new_group", return_value=MagicMock()),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.is_initialized", return_value=True),
        ):
            ps.preinit_vllm_parallel_state([0], vllm_tp_size=1, local_rank=1)
        # Non-member rank 1 gets null coordinators (world_size == 0).
        assert ps._WORLD is not None
        assert ps._WORLD.world_size == 0
        assert ps._TP.world_size == 0
        assert ps._PP.world_size == 0
    finally:
        ps._WORLD, ps._TP, ps._DCP, ps._PCP, ps._PP, ps._DP = saved


# ---------------------------------------------------------------------------
# vLLM DP helpers
# ---------------------------------------------------------------------------


def _run_init_dp(
    rank: int,
    world_size: int,
    sae_tp: int,
    vllm_tp: int,
    vllm_dp: int,
    sae_dp: int = 1,
) -> None:
    mock_group = MagicMock()
    with (
        patch("sae_lens.distributed.dist.is_initialized", return_value=True),
        patch("sae_lens.distributed.dist.get_world_size", return_value=world_size),
        patch("sae_lens.distributed.dist.get_rank", return_value=rank),
        patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
    ):
        init_distributed(
            sae_tp_size=sae_tp,
            vllm_tp_size=vllm_tp,
            vllm_dp_size=vllm_dp,
            sae_dp_size=sae_dp,
        )


# ---------------------------------------------------------------------------
# Config 5: vllm_tp=2, vllm_dp=3, sae_tp=1 → world_size=6
# ---------------------------------------------------------------------------


def test_vllm_dp_config5_rank0_both_active():
    _run_init_dp(rank=0, world_size=6, sae_tp=1, vllm_tp=2, vllm_dp=3)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_dp_rank == 0
    assert distributed_mod._vllm_tp_rank == 0
    assert distributed_mod._sae_tp_rank == 0


def test_vllm_dp_config5_rank1_vllm_only():
    _run_init_dp(rank=1, world_size=6, sae_tp=1, vllm_tp=2, vllm_dp=3)
    assert distributed_mod._sae_active is False
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_dp_rank == 0
    assert distributed_mod._vllm_tp_rank == 1


def test_vllm_dp_config5_rank2_helper_root():
    _run_init_dp(rank=2, world_size=6, sae_tp=1, vllm_tp=2, vllm_dp=3)
    assert distributed_mod._sae_active is False
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_dp_rank == 1
    assert distributed_mod._vllm_tp_rank == 0


def test_vllm_dp_config5_rank5_helper_nonroot():
    _run_init_dp(rank=5, world_size=6, sae_tp=1, vllm_tp=2, vllm_dp=3)
    assert distributed_mod._sae_active is False
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_dp_rank == 2
    assert distributed_mod._vllm_tp_rank == 1


# ---------------------------------------------------------------------------
# Config 6: vllm_tp=1, vllm_dp=2, sae_tp=2 → world_size=2 (full overlap)
# ---------------------------------------------------------------------------


def test_vllm_dp_config6_rank0_both_active():
    _run_init_dp(rank=0, world_size=2, sae_tp=2, vllm_tp=1, vllm_dp=2)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_dp_rank == 0
    assert distributed_mod._sae_tp_rank == 0


def test_vllm_dp_config6_rank1_both_active():
    _run_init_dp(rank=1, world_size=2, sae_tp=2, vllm_tp=1, vllm_dp=2)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_dp_rank == 1
    assert distributed_mod._sae_tp_rank == 1


# ---------------------------------------------------------------------------
# Config 7: vllm_tp=1, vllm_dp=4, sae_tp=1 → world_size=4 (no TP)
# ---------------------------------------------------------------------------


def test_vllm_dp_config7_rank0():
    _run_init_dp(rank=0, world_size=4, sae_tp=1, vllm_tp=1, vllm_dp=4)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_dp_rank == 0


def test_vllm_dp_config7_rank3():
    _run_init_dp(rank=3, world_size=4, sae_tp=1, vllm_tp=1, vllm_dp=4)
    assert distributed_mod._sae_active is False
    assert distributed_mod._vllm_dp_rank == 3


# ---------------------------------------------------------------------------
# vLLM DP accessor tests
# ---------------------------------------------------------------------------


def test_vllm_dp_accessors():
    _run_init_dp(rank=2, world_size=6, sae_tp=1, vllm_tp=2, vllm_dp=3)
    from sae_lens.distributed import (
        get_vllm_dp_rank,
        get_vllm_dp_size,
        get_vllm_root_rank,
        get_vllm_world_ranks,
        is_vllm_dp_root,
    )

    assert get_vllm_dp_rank() == 1
    assert get_vllm_dp_size() == 3
    assert is_vllm_dp_root() is True
    assert get_vllm_root_rank() == 2
    assert get_vllm_world_ranks() == [0, 1, 2, 3, 4, 5]


def test_vllm_dp_nonroot_is_not_dp_root():
    _run_init_dp(rank=3, world_size=6, sae_tp=1, vllm_tp=2, vllm_dp=3)
    from sae_lens.distributed import is_vllm_dp_root

    assert is_vllm_dp_root() is False


# ---------------------------------------------------------------------------
# vLLM DP backward compatibility (vllm_dp=1 matches legacy)
# ---------------------------------------------------------------------------


def test_vllm_dp1_matches_legacy():
    _run_init_dp(rank=0, world_size=2, sae_tp=1, vllm_tp=2, vllm_dp=1)
    assert distributed_mod._vllm_dp_size == 1
    assert distributed_mod._vllm_dp_rank == 0
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._vllm_tp_rank == 0


# ---------------------------------------------------------------------------
# Matched DP (m:m) topology
# ---------------------------------------------------------------------------


def test_matched_dp_rank0_both_active():
    _run_init_dp(rank=0, world_size=4, sae_tp=1, vllm_tp=2, vllm_dp=2, sae_dp=2)
    assert distributed_mod._layout_mode == "matched_dp"
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_dp_rank == 0
    assert distributed_mod._vllm_dp_rank == 0
    assert distributed_mod._vllm_tp_rank == 0


def test_matched_dp_rank1_vllm_only():
    _run_init_dp(rank=1, world_size=4, sae_tp=1, vllm_tp=2, vllm_dp=2, sae_dp=2)
    assert distributed_mod._sae_active is False
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_dp_rank == 0
    assert distributed_mod._vllm_dp_rank == 0
    assert distributed_mod._vllm_tp_rank == 1


def test_matched_dp_rank2_both_active_second_replica():
    _run_init_dp(rank=2, world_size=4, sae_tp=1, vllm_tp=2, vllm_dp=2, sae_dp=2)
    assert distributed_mod._sae_active is True
    assert distributed_mod._vllm_active is True
    assert distributed_mod._sae_dp_rank == 1
    assert distributed_mod._vllm_dp_rank == 1
    assert distributed_mod._vllm_tp_rank == 0


def test_matched_dp_accessors_use_replica_layout():
    _run_init_dp(rank=2, world_size=4, sae_tp=1, vllm_tp=2, vllm_dp=2, sae_dp=2)
    from sae_lens.distributed import (
        get_sae_dp_rank,
        get_sae_dp_size,
        get_vllm_dp_rank,
        get_vllm_dp_size,
        get_vllm_root_rank,
        get_vllm_world_ranks,
        is_vllm_dp_root,
    )

    assert get_sae_dp_rank() == 1
    assert get_sae_dp_size() == 2
    assert get_vllm_dp_rank() == 1
    assert get_vllm_dp_size() == 2
    assert is_vllm_dp_root() is True
    assert get_vllm_root_rank() == 2
    assert get_vllm_world_ranks() == [0, 1, 2, 3]


def test_matched_dp_accessors_handle_noncontiguous_vllm_world_ranks():
    _run_init_dp(rank=2, world_size=4, sae_tp=2, vllm_tp=1, vllm_dp=2, sae_dp=2)
    from sae_lens.distributed import get_vllm_root_rank, get_vllm_world_ranks

    assert get_vllm_root_rank() == 2
    assert get_vllm_world_ranks() == [0, 2]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_rejects_sae_dp_without_vllm_dp():
    with pytest.raises(ValueError, match="sae_dp_size > 1 requires vllm_dp_size > 1"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=2),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(
                sae_tp_size=1, vllm_tp_size=1, vllm_dp_size=1, sae_dp_size=2
            )


def test_rejects_mismatched_dp_sizes():
    with pytest.raises(ValueError, match="must be integer multiples"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=6),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(
                sae_tp_size=1, vllm_tp_size=1, vllm_dp_size=3, sae_dp_size=2
            )


def test_vllm_dp_rejects_shared_tp_size():
    with pytest.raises(ValueError, match="shared_tp_size is incompatible"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=4),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(shared_tp_size=2, vllm_dp_size=2)


def test_vllm_dp_rejects_sae_tp_too_large():
    with pytest.raises(ValueError, match="sae_tp_size=.*not supported"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=4),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(
                sae_tp_size=3, vllm_tp_size=1, vllm_dp_size=2
            )


# ---------------------------------------------------------------------------
# mn:m topology tests
# ---------------------------------------------------------------------------


def _run_init_mn_m(
    rank: int,
    world_size: int,
    sae_tp: int,
    vllm_tp: int,
    vllm_dp: int,
    sae_dp: int,
) -> None:
    mock_group = MagicMock()
    with (
        patch("sae_lens.distributed.dist.is_initialized", return_value=True),
        patch("sae_lens.distributed.dist.get_world_size", return_value=world_size),
        patch("sae_lens.distributed.dist.get_rank", return_value=rank),
        patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
    ):
        init_distributed(
            sae_tp_size=sae_tp,
            vllm_tp_size=vllm_tp,
            vllm_dp_size=vllm_dp,
            sae_dp_size=sae_dp,
        )


# ---------------------------------------------------------------------------
# Case A: vllm_dp=4, sae_dp=2, vllm_tp=2, sae_tp=1 → world=8
# n=2, block=4
# cluster 0: ranks 0-3, cluster 1: ranks 4-7
# rank 0: vllm_dp=0,tp=0  sae_dp=0,tp=0  both active
# rank 1: vllm_dp=0,tp=1                  vllm only
# rank 2: vllm_dp=1,tp=0                  vllm only (helper root)
# rank 3: vllm_dp=1,tp=1                  vllm only
# rank 4: vllm_dp=2,tp=0  sae_dp=1,tp=0  both active
# rank 7: vllm_dp=3,tp=1                  vllm only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank,exp_sae_active,exp_vllm_active,exp_sae_dp,exp_vllm_dp,exp_sae_tp,exp_vllm_tp",
    [
        (0, True,  True,  0, 0, 0, 0),
        (1, False, True,  0, 0, -1, 1),
        (2, False, True,  0, 1, -1, 0),
        (3, False, True,  0, 1, -1, 1),
        (4, True,  True,  1, 2, 0, 0),
        (5, False, True,  1, 2, -1, 1),
        (6, False, True,  1, 3, -1, 0),
        (7, False, True,  1, 3, -1, 1),
    ],
)
def test_mn_m_case_a_rank_roles(
    rank, exp_sae_active, exp_vllm_active, exp_sae_dp, exp_vllm_dp, exp_sae_tp, exp_vllm_tp
):
    _run_init_mn_m(rank=rank, world_size=8, sae_tp=1, vllm_tp=2, vllm_dp=4, sae_dp=2)
    assert distributed_mod._sae_active is exp_sae_active
    assert distributed_mod._vllm_active is exp_vllm_active
    assert distributed_mod._sae_dp_rank == exp_sae_dp
    assert distributed_mod._vllm_dp_rank == exp_vllm_dp
    assert distributed_mod._sae_tp_rank == exp_sae_tp
    assert distributed_mod._vllm_tp_rank == exp_vllm_tp


def test_mn_m_case_a_layout_mode():
    _run_init_mn_m(rank=0, world_size=8, sae_tp=1, vllm_tp=2, vllm_dp=4, sae_dp=2)
    assert distributed_mod._layout_mode == "vllm_dp_fan_in"


def test_mn_m_case_a_sae_root_rank():
    from sae_lens.distributed import get_sae_root_rank
    # cluster 0 SAE root = rank 0
    _run_init_mn_m(rank=0, world_size=8, sae_tp=1, vllm_tp=2, vllm_dp=4, sae_dp=2)
    assert get_sae_root_rank() == 0
    # cluster 1 SAE root = rank 4
    _run_init_mn_m(rank=4, world_size=8, sae_tp=1, vllm_tp=2, vllm_dp=4, sae_dp=2)
    assert get_sae_root_rank() == 4


def test_mn_m_case_a_sae_dp_group_spans_clusters():
    # SAE DP rank 0 (rank 0) and SAE DP rank 1 (rank 4) should share an sae_dp_group.
    # We verify both ranks get a non-None sae_dp_group (both sae_active with sae_tp_rank=0).
    _run_init_mn_m(rank=0, world_size=8, sae_tp=1, vllm_tp=2, vllm_dp=4, sae_dp=2)
    assert distributed_mod._sae_dp_group is not None
    _run_init_mn_m(rank=4, world_size=8, sae_tp=1, vllm_tp=2, vllm_dp=4, sae_dp=2)
    assert distributed_mod._sae_dp_group is not None


# ---------------------------------------------------------------------------
# Case B: vllm_dp=4, sae_dp=2, vllm_tp=1, sae_tp=4 → world=8
# n=2, block=max(2,4)=4
# rank 0: w=0, both active, vllm_dp=0, sae_tp=0
# rank 1: w=1, BOTH active (vllm+sae), vllm_dp=1, sae_tp=1
# rank 2: w=2, sae only, sae_tp=2
# rank 3: w=3, sae only, sae_tp=3
# rank 4: both active, sae_dp=1, vllm_dp=2, sae_tp=0
# rank 5: BOTH active, sae_dp=1, vllm_dp=3, sae_tp=1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank,exp_sae_active,exp_vllm_active,exp_sae_dp,exp_vllm_dp,exp_sae_tp,exp_vllm_tp",
    [
        (0, True,  True,  0, 0, 0,  0),
        (1, True,  True,  0, 1, 1,  0),   # simultaneously vllm+sae active
        (2, True,  False, 0, -1, 2, -1),
        (3, True,  False, 0, -1, 3, -1),
        (4, True,  True,  1, 2, 0,  0),
        (5, True,  True,  1, 3, 1,  0),   # simultaneously vllm+sae active
        (6, True,  False, 1, -1, 2, -1),
        (7, True,  False, 1, -1, 3, -1),
    ],
)
def test_mn_m_case_b_rank_roles(
    rank, exp_sae_active, exp_vllm_active, exp_sae_dp, exp_vllm_dp, exp_sae_tp, exp_vllm_tp
):
    _run_init_mn_m(rank=rank, world_size=8, sae_tp=4, vllm_tp=1, vllm_dp=4, sae_dp=2)
    assert distributed_mod._sae_active is exp_sae_active
    assert distributed_mod._vllm_active is exp_vllm_active
    assert distributed_mod._sae_dp_rank == exp_sae_dp
    assert distributed_mod._vllm_dp_rank == exp_vllm_dp
    assert distributed_mod._sae_tp_rank == exp_sae_tp
    assert distributed_mod._vllm_tp_rank == exp_vllm_tp


def test_mn_m_case_b_sae_root_rank():
    from sae_lens.distributed import get_sae_root_rank
    _run_init_mn_m(rank=2, world_size=8, sae_tp=4, vllm_tp=1, vllm_dp=4, sae_dp=2)
    assert get_sae_root_rank() == 0  # rank 2 is in cluster 0, SAE root = 0
    _run_init_mn_m(rank=6, world_size=8, sae_tp=4, vllm_tp=1, vllm_dp=4, sae_dp=2)
    assert get_sae_root_rank() == 4  # rank 6 is in cluster 1, SAE root = 4


# ---------------------------------------------------------------------------
# Case C: vllm_dp=6, sae_dp=2, vllm_tp=1, sae_tp=1 → world=6 (n=3)
# block=3; cluster 0: ranks 0,1,2; cluster 1: ranks 3,4,5
# rank 0: sae_dp=0, vllm_dp=0 (cluster SAE root)
# rank 1: vllm only, helper root, vllm_dp=1
# rank 2: vllm only, helper root, vllm_dp=2
# rank 3: sae_dp=1, vllm_dp=3
# rank 4: vllm only, vllm_dp=4
# rank 5: vllm only, vllm_dp=5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank,exp_sae_active,exp_vllm_active,exp_sae_dp,exp_vllm_dp",
    [
        (0, True,  True,  0, 0),
        (1, False, True,  0, 1),
        (2, False, True,  0, 2),
        (3, True,  True,  1, 3),
        (4, False, True,  1, 4),
        (5, False, True,  1, 5),
    ],
)
def test_mn_m_case_c_rank_roles(rank, exp_sae_active, exp_vllm_active, exp_sae_dp, exp_vllm_dp):
    _run_init_mn_m(rank=rank, world_size=6, sae_tp=1, vllm_tp=1, vllm_dp=6, sae_dp=2)
    assert distributed_mod._sae_active is exp_sae_active
    assert distributed_mod._vllm_active is exp_vllm_active
    assert distributed_mod._sae_dp_rank == exp_sae_dp
    assert distributed_mod._vllm_dp_rank == exp_vllm_dp


def test_mn_m_case_c_sae_dp_group():
    # rank 0 (sae_dp=0) and rank 3 (sae_dp=1) both sae_active with sae_tp_rank=0
    # → both should have a non-None sae_dp_group
    _run_init_mn_m(rank=0, world_size=6, sae_tp=1, vllm_tp=1, vllm_dp=6, sae_dp=2)
    assert distributed_mod._sae_dp_group is not None
    _run_init_mn_m(rank=3, world_size=6, sae_tp=1, vllm_tp=1, vllm_dp=6, sae_dp=2)
    assert distributed_mod._sae_dp_group is not None


def test_mn_m_case_c_sae_root_rank():
    from sae_lens.distributed import get_sae_root_rank
    # All ranks in cluster 0 should return SAE root = 0
    for r in [0, 1, 2]:
        _run_init_mn_m(rank=r, world_size=6, sae_tp=1, vllm_tp=1, vllm_dp=6, sae_dp=2)
        assert get_sae_root_rank() == 0, f"rank {r} failed"
    # All ranks in cluster 1 should return SAE root = 3
    for r in [3, 4, 5]:
        _run_init_mn_m(rank=r, world_size=6, sae_tp=1, vllm_tp=1, vllm_dp=6, sae_dp=2)
        assert get_sae_root_rank() == 3, f"rank {r} failed"


# ---------------------------------------------------------------------------
# mn:m validation errors
# ---------------------------------------------------------------------------


def test_mn_m_rejects_non_multiple_dp_sizes():
    with pytest.raises(ValueError, match="integer multiples"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=6),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(sae_tp_size=1, vllm_tp_size=1, vllm_dp_size=4, sae_dp_size=3)


def test_mn_m_rejects_m_mn_topology():
    with pytest.raises(ValueError, match="m:mn topology.*not yet supported"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=4),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(sae_tp_size=1, vllm_tp_size=1, vllm_dp_size=2, sae_dp_size=4)
