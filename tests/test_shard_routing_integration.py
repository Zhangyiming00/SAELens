"""Legacy CPU/Gloo integration tests for the old shard-routing P2P protocol.

Runtime shard routing is now NCCL-only and is validated with GPU torchrun smoke
tests instead of CPU pytest workers.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.distributed_v2 import (
    _reset,
    get_consumer_tp_root,
    get_p2p_group,
    get_producer_tp_root,
    get_routing_table,
    init_distributed_v2,
    is_consumer,
    is_producer,
    get_consumer_idx,
    get_producer_idx,
    get_sae_tp_rank,
    get_vllm_tp_rank,
)
from sae_lens.shard_routing import routes_for_consumer, routes_for_producer

pytestmark = pytest.mark.skip(
    reason="Shard-routing P2P is NCCL-only; use GPU torchrun smoke tests."
)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    P: int,
    Q: int,
    vllm_tp: int,
    sae_tp: int,
    batch_size: int,
    d_in: int,
    result_list,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29601"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    init_distributed_v2(
        P=P, Q=Q,
        vllm_tp_size=vllm_tp,
        sae_tp_size=sae_tp,
        batch_size=batch_size,
    )

    routing = get_routing_table()

    # Determine role flags for this rank
    i_am_producer = is_producer()
    i_am_consumer = is_consumer()
    i_am_producer_root = i_am_producer and get_vllm_tp_rank() == 0
    i_am_consumer_root = i_am_consumer and get_sae_tp_rank() == 0

    local_slices: dict = {}   # producer_idx -> tensor (for local same-rank routes)
    outgoing: dict = {}       # consumer_idx -> (route, slice tensor)

    # === PHASE 1: Consumer roots pre-allocate payload buffers and post irecvs ===
    # Buffer shape is known from routing table (n_rows) and d_in — no metadata needed.
    payload_work = []
    if i_am_consumer_root:
        c = get_consumer_idx()
        c_routes = routes_for_consumer(routing, c)
        for route in c_routes:
            pp = route.producer_idx
            if get_producer_tp_root(pp) == get_consumer_tp_root(c):
                payload_work.append((route, None, None))  # local: filled after Phase 2
            else:
                n_rows = route.row_end - route.row_start
                act_buf = torch.zeros(n_rows, d_in)
                grp = get_p2p_group(c)
                src = get_producer_tp_root(pp)
                w = dist.irecv(act_buf, src=src, group=grp)
                payload_work.append((route, act_buf, w))

    # === PHASE 2: Producer roots generate batch; local slices stay on device ===
    if i_am_producer_root:
        p = get_producer_idx()
        # Create a deterministic batch: row i has value p*1000 + i
        batch = torch.zeros(batch_size, d_in)
        for i in range(batch_size):
            batch[i] = p * 1000 + i

        my_routes = routes_for_producer(routing, p)
        for route in my_routes:
            cc = route.consumer_idx
            sl = batch[route.row_start:route.row_end].clone().contiguous()
            outgoing[cc] = (route, sl)
            if get_producer_tp_root(p) == get_consumer_tp_root(cc):
                local_slices[route.producer_idx] = sl  # key: producer_idx
                # No metadata send — consumer pre-allocates from routing table.

    # Fill local payload_work slots from local_slices (dual-role ranks only).
    if i_am_consumer_root:
        c = get_consumer_idx()
        payload_work = [
            (route, local_slices[route.producer_idx], None)
            if buf is None
            else (route, buf, w)
            for route, buf, w in payload_work
        ]

    # === PHASE 3: Producer roots send cached slice payloads (non-blocking) ===
    if i_am_producer_root:
        p = get_producer_idx()
        my_routes = routes_for_producer(routing, p)
        work_handles = []
        for route in my_routes:
            cc = route.consumer_idx
            if get_producer_tp_root(p) != get_consumer_tp_root(cc):
                _, sl = outgoing[cc]
                grp = get_p2p_group(cc)
                dst = get_consumer_tp_root(cc)
                w = dist.isend(sl, dst=dst, group=grp)
                work_handles.append(w)
        for w in work_handles:
            w.wait()

    # === PHASE 4: Consumer roots wait payloads, assemble in route order ===
    if i_am_consumer_root:
        c = get_consumer_idx()
        c_routes = routes_for_consumer(routing, c)
        buf_map = {}
        for route, buf, w in payload_work:
            if w is not None:
                w.wait()
            buf_map[route.producer_idx] = buf

        assembled = torch.cat(
            [buf_map[r.producer_idx] for r in c_routes], dim=0
        )
        result_list[rank] = (c, assembled)

    # Ranks that are neither producer root nor consumer root
    # (e.g. producer TP non-roots that are not consumers) just exit cleanly.

    dist.destroy_process_group()
    _reset()


def _spawn(P, Q, vllm_tp=1, sae_tp=1, batch_size=12, d_in=4):
    world_size = max(P * vllm_tp, Q * sae_tp)
    manager = mp.Manager()
    result_list = manager.list([None] * world_size)
    mp.spawn(
        _worker,
        args=(world_size, P, Q, vllm_tp, sae_tp, batch_size, d_in, result_list),
        nprocs=world_size,
        join=True,
    )
    # Collect consumer results
    consumer_results = {}
    for item in result_list:
        if item is not None:
            c, assembled = item
            consumer_results[c] = assembled
    return consumer_results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_2producer_3consumer_no_deadlock() -> None:
    """Test completes without hanging."""
    results = _spawn(P=2, Q=3, batch_size=12)
    assert len(results) == 3


@pytest.mark.slow
def test_2producer_3consumer_no_duplicate_rows() -> None:
    """All received row IDs are globally unique (no batch was duplicated)."""
    results = _spawn(P=2, Q=3, batch_size=12, d_in=4)
    all_tensors = [t for t in results.values()]
    all_rows = torch.cat(all_tensors, dim=0)
    # Row identifier: each row i in producer p has value p*1000 + i in column 0
    row_ids = all_rows[:, 0].tolist()
    assert len(row_ids) == len(set(row_ids)), f"Duplicate rows: {sorted(row_ids)}"


@pytest.mark.slow
def test_2producer_3consumer_token_conservation() -> None:
    """Total rows received across all consumers == P * batch_size."""
    P, B = 2, 12
    results = _spawn(P=P, Q=3, batch_size=B)
    total = sum(t.shape[0] for t in results.values())
    assert total == P * B


@pytest.mark.slow
def test_2producer_3consumer_per_consumer_count() -> None:
    """Each consumer receives the exact number of rows defined by the routing table."""
    from sae_lens.shard_routing import compute_routing_table
    P, Q, B = 2, 3, 12
    routes = compute_routing_table(P, Q, B)
    results = _spawn(P=P, Q=Q, batch_size=B)
    for c in range(Q):
        expected = sum(r.row_end - r.row_start for r in routes if r.consumer_idx == c)
        actual = results[c].shape[0]
        assert actual == expected, f"Consumer {c}: expected {expected} rows, got {actual}"


@pytest.mark.slow
def test_2producer_3consumer_one_batch_per_step() -> None:
    """Each consumer receives exactly one assembled tensor (not one per route)."""
    results = _spawn(P=2, Q=3, batch_size=12)
    # The worker stores exactly one assembled tensor per consumer rank
    assert all(isinstance(v, torch.Tensor) for v in results.values())
    assert len(results) == 3


@pytest.mark.slow
def test_3producer_3consumer_dual_role() -> None:
    """Matched P=Q=3; all ranks are dual-role; no P2P self-send; slices in route order."""
    results = _spawn(P=3, Q=3, batch_size=9, d_in=4)
    assert len(results) == 3
    # No duplicate rows
    all_rows = torch.cat(list(results.values()), dim=0)
    row_ids = all_rows[:, 0].tolist()
    assert len(row_ids) == len(set(row_ids))
    # Total conservation
    assert all_rows.shape[0] == 3 * 9


@pytest.mark.slow
def test_2producer_3consumer_vllm_tp2_sae_tp1() -> None:
    """vllm_tp=2, sae_tp=1; world=max(4,3)=4; producer TP non-roots do not send."""
    results = _spawn(P=2, Q=3, vllm_tp=2, sae_tp=1, batch_size=12)
    assert len(results) == 3
    # Token conservation
    all_rows = torch.cat(list(results.values()), dim=0)
    assert all_rows.shape[0] == 2 * 12


@pytest.mark.slow
def test_2producer_3consumer_vllm_tp1_sae_tp2() -> None:
    """vllm_tp=1, sae_tp=2; world=max(2,6)=6; consumer TP followers receive broadcasts."""
    # In this test the worker only stores results from sae_tp_rank==0 roots;
    # followers participate in broadcast but don't store. Just verify no deadlock + conservation.
    results = _spawn(P=2, Q=3, vllm_tp=1, sae_tp=2, batch_size=12)
    # Consumer roots (sae_tp_rank==0) store results → should have 3 entries
    assert len(results) == 3
    all_rows = torch.cat(list(results.values()), dim=0)
    assert all_rows.shape[0] == 2 * 12


def _worker_device(
    rank: int,
    world_size: int,
    P: int,
    Q: int,
    batch_size: int,
    d_in: int,
    result_list,
) -> None:
    """Worker that records tensor device of local slices and assembled output."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29604"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    init_distributed_v2(P=P, Q=Q, vllm_tp_size=1, sae_tp_size=1, batch_size=batch_size)

    routing = get_routing_table()
    i_am_producer_root = is_producer() and get_vllm_tp_rank() == 0
    i_am_consumer_root = is_consumer() and get_sae_tp_rank() == 0

    local_slices: dict = {}
    outgoing: dict = {}

    payload_work = []
    if i_am_consumer_root:
        c = get_consumer_idx()
        for route in routes_for_consumer(routing, c):
            pp = route.producer_idx
            if get_producer_tp_root(pp) == get_consumer_tp_root(c):
                payload_work.append((route, None, None))
            else:
                n_rows = route.row_end - route.row_start
                act_buf = torch.zeros(n_rows, d_in)
                w = dist.irecv(act_buf, src=get_producer_tp_root(pp), group=get_p2p_group(c))
                payload_work.append((route, act_buf, w))

    if i_am_producer_root:
        p = get_producer_idx()
        batch = torch.zeros(batch_size, d_in)
        for route in routes_for_producer(routing, p):
            cc = route.consumer_idx
            sl = batch[route.row_start:route.row_end].clone().contiguous()
            outgoing[cc] = (route, sl)
            if get_producer_tp_root(p) == get_consumer_tp_root(cc):
                local_slices[route.producer_idx] = sl

    if i_am_consumer_root:
        payload_work = [
            (route, local_slices[route.producer_idx], None)
            if buf is None
            else (route, buf, w)
            for route, buf, w in payload_work
        ]

    if i_am_producer_root:
        p = get_producer_idx()
        work_handles = []
        for route in routes_for_producer(routing, p):
            cc = route.consumer_idx
            if get_producer_tp_root(p) != get_consumer_tp_root(cc):
                _, sl = outgoing[cc]
                w = dist.isend(sl, dst=get_consumer_tp_root(cc), group=get_p2p_group(cc))
                work_handles.append(w)
        for w in work_handles:
            w.wait()

    if i_am_consumer_root:
        c = get_consumer_idx()
        buf_map = {}
        for route, buf, w in payload_work:
            if w is not None:
                w.wait()
            buf_map[route.producer_idx] = buf
        # Record the device of each slice in buf_map
        devices = {p: str(t.device) for p, t in buf_map.items()}
        result_list[rank] = (c, devices)

    dist.destroy_process_group()
    _reset()


@pytest.mark.slow
def test_local_routes_preserve_device_matched_dp() -> None:
    """For a fully-local topology (P=Q, all routes same-rank), assembled slices
    must be on the same device as the input batch — no CPU bounce for local routes."""
    P, Q, B, d = 3, 3, 9, 4
    world_size = max(P, Q)
    manager = mp.Manager()
    result_list = manager.list([None] * world_size)
    mp.spawn(
        _worker_device,
        args=(world_size, P, Q, B, d, result_list),
        nprocs=world_size,
        join=True,
    )
    results = [r for r in result_list if r is not None]
    assert len(results) == Q
    for _c, devices in results:
        # All local slices must be on the same device as the batch (cpu in this test).
        # The key property: device is "cpu" (unchanged from the zero-filled batch),
        # NOT something that would appear after an unnecessary to("cpu") call.
        for p, dev in devices.items():
            assert dev == "cpu", (
                f"consumer slice from producer {p} is on device '{dev}' "
                f"(expected 'cpu' — device of the input batch)"
            )
