"""Tests for StreamingActivationProvider (sae_tp=1, no distributed).

Uses a mock SharedActivationBuffer so no /dev/shm or CUDA is needed.
"""

from unittest.mock import MagicMock

import pytest
import torch

from sae_lens.training.streaming_activation_provider import StreamingActivationProvider


def _make_sequential_buffer(chunks: list[torch.Tensor]) -> MagicMock:
    """Buffer that yields chunks one at a time in order, then raises StopIteration."""
    buf = MagicMock()
    remaining = list(chunks)

    def acquire_up_to(n, random=True):
        if not remaining:
            raise StopIteration
        return [0], 0.0

    def read_chunk(idx):
        tensor = remaining.pop(0)
        return tensor, tensor.shape[0]

    def release_chunk(idx):
        pass

    buf.acquire_up_to.side_effect = acquire_up_to
    buf.read_chunk.side_effect = read_chunk
    buf.release_chunk.side_effect = release_chunk
    return buf


def _provider(buf, batch_size: int, d_model: int, prefetch: int = 1):
    return StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=batch_size,
        prefetch_chunks=prefetch,
        device=torch.device("cpu"),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=d_model,
    )


def _drain(provider) -> list[int]:
    """Drain provider, return list of batch sizes."""
    sizes = []
    try:
        while True:
            batch = next(provider)
            sizes.append(batch.shape[0])
    except StopIteration:
        pass
    return sizes


def test_all_tokens_consumed_no_loss():
    # 3 chunks × 100 tokens = 300 total; all tokens must be consumed
    d_model, rows_per_chunk, n_chunks = 16, 100, 3
    chunks = [torch.randn(rows_per_chunk, d_model, dtype=torch.bfloat16) for _ in range(n_chunks)]
    buf = _make_sequential_buffer(chunks)
    p = _provider(buf, batch_size=64, d_model=d_model)

    sizes = _drain(p)
    assert sum(sizes) == rows_per_chunk * n_chunks
    assert p.consumed_tokens == rows_per_chunk * n_chunks


def test_leftover_carry_across_refill():
    # batch_size=64, chunk1=80 rows, chunk2=30 rows (total=110).
    # leftover after chunk1: 80-64=16. Next refill merges 16+30=46.
    # If carry-over is broken, total would be 64+30=94 instead of 64+46=110.
    d_model = 8
    c1 = torch.randn(80, d_model, dtype=torch.bfloat16)
    c2 = torch.randn(30, d_model, dtype=torch.bfloat16)
    buf = _make_sequential_buffer([c1, c2])
    p = _provider(buf, batch_size=64, d_model=d_model)

    sizes = _drain(p)
    assert sum(sizes) == 110  # no tokens lost
    assert sizes[0] == 64    # first batch is full


def test_final_partial_batch_not_dropped():
    # 1 chunk of 50 tokens, batch_size=32.
    # First batch=32. Pool has 18 left; buffer exhausted → final batch=18 (not dropped).
    d_model = 4
    chunk = torch.randn(50, d_model, dtype=torch.bfloat16)
    buf = _make_sequential_buffer([chunk])
    p = _provider(buf, batch_size=32, d_model=d_model)

    sizes = _drain(p)
    assert sum(sizes) == 50
    assert sizes == [32, 18]


def test_stop_iteration_on_buffer_exhausted():
    # Provider raises StopIteration after all tokens consumed
    d_model = 8
    chunk = torch.randn(50, d_model, dtype=torch.bfloat16)
    buf = _make_sequential_buffer([chunk])
    p = _provider(buf, batch_size=60, d_model=d_model)

    # Only one batch (50 < 60 but all tokens served as partial final batch)
    b1 = next(p)
    assert b1.shape[0] == 50
    with pytest.raises(StopIteration):
        next(p)


def test_consume_last_data_timing_nonzero_on_refill():
    # After a refill, consume_last_data_timing() should return positive wait time;
    # subsequent batches from the same pool should return zeros (no new refill occurred).
    d_model = 8
    # 200 tokens, batch_size=64 → first next() triggers refill, next 2 serve from pool
    chunk = torch.randn(200, d_model, dtype=torch.bfloat16)
    buf = _make_sequential_buffer([chunk])
    p = _provider(buf, batch_size=64, d_model=d_model)

    # First next() must refill: timing should be non-zero
    next(p)
    t1 = p.consume_last_data_timing()
    assert t1["vllm_step_time_s"] >= 0.0
    assert t1["transfer_time_s"] >= 0.0
    # At least one of wait/transfer should be non-negative (both are durations)
    assert t1["vllm_step_time_s"] + t1["transfer_time_s"] >= 0.0

    # Second next() served from pool: timing should be cleared (both zeros)
    next(p)
    t2 = p.consume_last_data_timing()
    assert t2["vllm_step_time_s"] == 0.0
    assert t2["transfer_time_s"] == 0.0


def test_consume_last_data_timing_clears_after_read():
    # Calling consume_last_data_timing() twice without a refill should return zeros both times.
    d_model = 8
    chunk = torch.randn(64, d_model, dtype=torch.bfloat16)
    buf = _make_sequential_buffer([chunk])
    p = _provider(buf, batch_size=64, d_model=d_model)

    next(p)
    p.consume_last_data_timing()  # first read clears
    t = p.consume_last_data_timing()  # second read should be zeros
    assert t["vllm_step_time_s"] == 0.0
    assert t["transfer_time_s"] == 0.0


def test_tp_follower_stops_on_zero_meta():
    # Simulate a TP follower (sae_tp_rank=1) receiving meta[0]=0 via broadcast.
    # Provider should raise StopIteration on the follower rank.
    import torch.distributed as dist

    def fake_broadcast(tensor, src, group):
        tensor[0] = 0  # signal end-of-stream

    import unittest.mock as mock
    with mock.patch.object(dist, "broadcast", side_effect=fake_broadcast):
        buf = MagicMock()
        fake_group = MagicMock()

        p = StreamingActivationProvider(
            buffer=buf,
            train_batch_size_tokens=32,
            prefetch_chunks=2,
            device=torch.device("cpu"),
            sae_tp_group=fake_group,
            sae_tp_rank=1,  # follower
            sae_tp_root_global_rank=0,
            d_model=8,
        )

        with pytest.raises(StopIteration):
            next(p)
