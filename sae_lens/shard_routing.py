"""Shard-routing table for unified P-producer / Q-consumer distributed training.

Computes a deterministic assignment of which rows of each producer's raw activation
batch belong to which consumer, using interval-overlap arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardRoute:
    """A directed edge from producer ``p`` to consumer ``c`` with row-slice metadata.

    Row ownership is defined on the pre-filter raw activation tensor before any
    consumer-side filtering. The routing table assumes a fixed raw batch row count
    per producer step (``store_batch_size_prompts * training_context_size``).

    The slice ``[row_start, row_end)`` is an exclusive interval into producer ``p``'s
    local batch of ``batch_size`` rows. Partitions across all consumers of ``p`` are
    non-overlapping and together cover ``[0, batch_size)``.
    """

    producer_idx: int  # p in [0, P)
    consumer_idx: int  # c in [0, Q)
    row_start: int  # inclusive start row index in producer p's batch
    row_end: int  # exclusive end row index


def compute_routing_table(P: int, Q: int, batch_size: int) -> list[ShardRoute]:
    """Build the routing table partitioning each producer's raw batch rows among consumers.

    Producer ``p`` owns the interval ``[p/P, (p+1)/P)``.
    Consumer ``c`` owns the interval ``[c/Q, (c+1)/Q)``.
    In integer units of ``1/(P*Q)``: producer span is ``[p*Q, (p+1)*Q)``
    and consumer span is ``[c*P, (c+1)*P)``.  They are connected when their spans overlap.

    Each producer's ``batch_size`` rows are partitioned non-overlappingly among its connected
    consumers, proportional to the overlap length.  The last edge of each producer absorbs any
    integer remainder so the partition covers ``[0, batch_size)`` exactly.

    Raises ``ValueError`` immediately when any connected edge (positive overlap) would
    receive zero rows — the caller must increase ``batch_size`` or reduce P/Q.

    Returns a list of ``ShardRoute`` objects sorted by ``(producer_idx, consumer_idx)``.
    """
    if P < 1 or Q < 1:
        raise ValueError(f"P and Q must be >= 1, got P={P}, Q={Q}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    routes: list[ShardRoute] = []

    for p in range(P):
        p_lo = p * Q
        p_hi = (p + 1) * Q  # producer interval in P*Q units: [p*Q, (p+1)*Q)

        # Collect connected edges (consumer, overlap_length) in consumer order.
        edges: list[tuple[int, int]] = []
        for c in range(Q):
            c_lo = c * P
            c_hi = (c + 1) * P  # consumer interval in P*Q units: [c*P, (c+1)*P)
            ov = max(0, min(p_hi, c_hi) - max(p_lo, c_lo))
            if ov > 0:
                edges.append((c, ov))

        cursor = 0
        for i, (c, ov_len) in enumerate(edges):
            if i < len(edges) - 1:
                n_rows = (ov_len * batch_size) // Q
            else:
                n_rows = batch_size - cursor  # last edge absorbs remainder

            if n_rows == 0:
                raise ValueError(
                    f"batch_size={batch_size} too small: edge (p={p}, c={c}) has positive "
                    f"overlap (ov_len={ov_len}) but would receive 0 rows. "
                    f"Increase batch_size or reduce P/Q."
                )

            routes.append(ShardRoute(p, c, cursor, cursor + n_rows))
            cursor += n_rows

        assert cursor == batch_size, (
            f"Internal error: producer {p} assigned {cursor} rows, expected {batch_size}"
        )

    return routes


def routes_for_producer(routes: list[ShardRoute], p: int) -> list[ShardRoute]:
    """All routes originating from producer ``p``, in ascending ``consumer_idx`` order."""
    return [r for r in routes if r.producer_idx == p]


def routes_for_consumer(routes: list[ShardRoute], c: int) -> list[ShardRoute]:
    """All routes targeting consumer ``c``, in ascending ``producer_idx`` order."""
    return [r for r in routes if r.consumer_idx == c]
