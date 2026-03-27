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
    distributed_mod._sae_tp_rank = 0
    distributed_mod._sae_tp_size = 1
    distributed_mod._dp_rank = 0
    distributed_mod._dp_size = 1
    distributed_mod._worker_rank = 0
    distributed_mod._worker_size = 1
    distributed_mod._vllm_tp_rank = -1
    distributed_mod._vllm_tp_size = 1
    distributed_mod._vllm_dp_rank = 0
    distributed_mod._vllm_dp_size = 1
    distributed_mod._sae_active = True
    distributed_mod._vllm_active = True


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
    assert distributed_mod._dp_rank == 0
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
# vLLM DP validation errors
# ---------------------------------------------------------------------------


def test_vllm_dp_rejects_dp_size_gt1():
    with pytest.raises(ValueError, match="vllm_dp_size > 1 and dp_size > 1"):
        mock_group = MagicMock()
        with (
            patch("sae_lens.distributed.dist.is_initialized", return_value=True),
            patch("sae_lens.distributed.dist.get_world_size", return_value=4),
            patch("sae_lens.distributed.dist.get_rank", return_value=0),
            patch("sae_lens.distributed.dist.new_group", return_value=mock_group),
        ):
            init_distributed(
                sae_tp_size=1, vllm_tp_size=1, vllm_dp_size=2, dp_size=2
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
