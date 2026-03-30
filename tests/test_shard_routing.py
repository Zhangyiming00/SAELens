"""Unit tests for sae_lens.shard_routing.compute_routing_table."""

import pytest

from sae_lens.shard_routing import (
    ShardRoute,
    compute_routing_table,
    routes_for_consumer,
    routes_for_producer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_partition_invariants(routes: list[ShardRoute], P: int, Q: int, batch_size: int) -> None:
    """Assert that the routing table forms a valid partition for every producer."""
    assert {r.producer_idx for r in routes} == set(range(P))
    assert {r.consumer_idx for r in routes} == set(range(Q))

    for p in range(P):
        p_routes = routes_for_producer(routes, p)
        # Rows sum to batch_size
        total = sum(r.row_end - r.row_start for r in p_routes)
        assert total == batch_size, f"Producer {p}: row sum {total} != {batch_size}"
        # Intervals are non-overlapping and contiguous
        sorted_routes = sorted(p_routes, key=lambda r: r.row_start)
        cursor = 0
        for r in sorted_routes:
            assert r.row_start == cursor, (
                f"Producer {p}: gap/overlap at row {cursor}, route starts at {r.row_start}"
            )
            assert r.row_end > r.row_start, f"Producer {p}: empty route to consumer {r.consumer_idx}"
            cursor = r.row_end
        assert cursor == batch_size


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

def test_routing_partition_1_1() -> None:
    routes = compute_routing_table(1, 1, 10)
    assert routes == [ShardRoute(0, 0, 0, 10)]
    _check_partition_invariants(routes, 1, 1, 10)


def test_routing_partition_3_1() -> None:
    routes = compute_routing_table(3, 1, 12)
    # All producers → consumer 0, each with all rows
    assert all(r.consumer_idx == 0 for r in routes)
    assert all(r.row_start == 0 and r.row_end == 12 for r in routes)
    _check_partition_invariants(routes, 3, 1, 12)


def test_routing_partition_1_3() -> None:
    routes = compute_routing_table(1, 3, 12)
    # Single producer splits rows among 3 consumers
    assert len(routes) == 3
    assert all(r.producer_idx == 0 for r in routes)
    _check_partition_invariants(routes, 1, 3, 12)


def test_routing_partition_3_3() -> None:
    routes = compute_routing_table(3, 3, 10)
    # Matched: producer i → consumer i only
    for r in routes:
        assert r.producer_idx == r.consumer_idx
    _check_partition_invariants(routes, 3, 3, 10)


def test_routing_partition_6_2() -> None:
    routes = compute_routing_table(6, 2, 12)
    # Exact fan-in: 3 producers per consumer
    c0_producers = {r.producer_idx for r in routes if r.consumer_idx == 0}
    c1_producers = {r.producer_idx for r in routes if r.consumer_idx == 1}
    assert c0_producers == {0, 1, 2}
    assert c1_producers == {3, 4, 5}
    _check_partition_invariants(routes, 6, 2, 12)


def test_routing_partition_2_3() -> None:
    routes = compute_routing_table(2, 3, 12)
    # Fan-out: p0 → {c0, c1}, p1 → {c1, c2}
    p0_consumers = {r.consumer_idx for r in routes_for_producer(routes, 0)}
    p1_consumers = {r.consumer_idx for r in routes_for_producer(routes, 1)}
    assert p0_consumers == {0, 1}
    assert p1_consumers == {1, 2}
    _check_partition_invariants(routes, 2, 3, 12)


def test_routing_partition_5_3() -> None:
    routes = compute_routing_table(5, 3, 15)
    _check_partition_invariants(routes, 5, 3, 15)


def test_routing_all_producers_covered() -> None:
    for P, Q in [(4, 3), (3, 4), (7, 5), (5, 7)]:
        routes = compute_routing_table(P, Q, 100)
        assert {r.producer_idx for r in routes} == set(range(P))


def test_routing_all_consumers_covered() -> None:
    for P, Q in [(4, 3), (3, 4), (7, 5), (5, 7)]:
        routes = compute_routing_table(P, Q, 100)
        assert {r.consumer_idx for r in routes} == set(range(Q))


def test_routing_no_duplicate_rows() -> None:
    """Row intervals per producer are non-overlapping and cover [0, batch_size)."""
    for P, Q in [(2, 3), (5, 3), (3, 5), (7, 4)]:
        routes = compute_routing_table(P, Q, 120)
        _check_partition_invariants(routes, P, Q, 120)


# ---------------------------------------------------------------------------
# Zero-row error
# ---------------------------------------------------------------------------

def test_routing_zero_row_error_2_3_tiny_batch() -> None:
    # batch_size=1 with P=2, Q=3: some edges would get 0 rows
    with pytest.raises(ValueError, match="too small"):
        compute_routing_table(2, 3, 1)


def test_routing_zero_row_error_large_ratio() -> None:
    # P=1, Q=100, batch_size=10: only 10 rows for 100 consumers → some get 0
    with pytest.raises(ValueError, match="too small"):
        compute_routing_table(1, 100, 10)


def test_routing_zero_row_no_error_when_feasible() -> None:
    # Should succeed without error
    routes = compute_routing_table(1, 100, 200)
    _check_partition_invariants(routes, 1, 100, 200)


# ---------------------------------------------------------------------------
# routes_for_producer / routes_for_consumer
# ---------------------------------------------------------------------------

def test_routes_for_producer() -> None:
    routes = compute_routing_table(2, 3, 12)
    p0 = routes_for_producer(routes, 0)
    assert all(r.producer_idx == 0 for r in p0)
    assert len(p0) > 0


def test_routes_for_consumer() -> None:
    routes = compute_routing_table(2, 3, 12)
    c1 = routes_for_consumer(routes, 1)
    assert all(r.consumer_idx == 1 for r in c1)
    # Consumer 1 should receive from both producers in 2:3 case
    assert {r.producer_idx for r in c1} == {0, 1}


# ---------------------------------------------------------------------------
# Row total conservation: sum across all consumers == P * batch_size
# ---------------------------------------------------------------------------

def test_routing_total_row_conservation() -> None:
    for P, Q in [(3, 1), (1, 3), (3, 3), (2, 3), (5, 3)]:
        batch_size = 60
        routes = compute_routing_table(P, Q, batch_size)
        total = sum(r.row_end - r.row_start for r in routes)
        assert total == P * batch_size, f"P={P}, Q={Q}: total {total} != {P * batch_size}"
