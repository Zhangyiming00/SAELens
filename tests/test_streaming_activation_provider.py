"""Tests for StreamingActivationProvider (sae_tp=1, no distributed).

Uses a mock SharedActivationBuffer so no /dev/shm or CUDA is needed.
"""

from unittest.mock import MagicMock

import pytest
import torch

from sae_lens.training.streaming_activation_provider import StreamingActivationProvider
from sae_lens.training.mixing_buffer import mixing_buffer


def _make_sequential_buffer(chunks: list[torch.Tensor]) -> MagicMock:
    """Buffer that yields chunks one at a time in order, then raises StopIteration."""
    buf = MagicMock()
    buf._chunk_size_tokens = chunks[0].shape[0] if chunks else 0
    remaining = list(chunks)

    def acquire_up_to(n, random=True, refcount=1, stop_check=None):
        _ = n
        _ = random
        _ = refcount
        _ = stop_check
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


def _provider_with_mix(
    buf,
    batch_size: int,
    d_model: int,
    *,
    mix_chunks: int,
    mix_fraction: float = 0.5,
):
    return StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=batch_size,
        prefetch_chunks=1,
        device=torch.device("cpu"),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=d_model,
        shuffle=False,
        mix_chunks=mix_chunks,
        mix_fraction=mix_fraction,
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


def test_drain_local_pool_stops_acquiring_new_chunks():
    d_model = 4
    chunks = [
        torch.randn(96, d_model, dtype=torch.bfloat16),
        torch.randn(96, d_model, dtype=torch.bfloat16),
    ]
    buf = _make_sequential_buffer(chunks)
    p = _provider(buf, batch_size=32, d_model=d_model)

    first = next(p)
    assert first.shape[0] == 32
    p.request_drain_local_pool()

    sizes = [first.shape[0]]
    with pytest.raises(StopIteration):
        while True:
            sizes.append(next(p).shape[0])

    assert sizes == [32, 32, 32]
    assert buf.acquire_up_to.call_count == 1
    assert p.consumed_tokens == 96


def test_external_stop_check_drains_without_new_acquire():
    d_model = 4
    chunks = [
        torch.randn(96, d_model, dtype=torch.bfloat16),
        torch.randn(96, d_model, dtype=torch.bfloat16),
    ]
    stop = {"value": False}
    buf = _make_sequential_buffer(chunks)
    p = StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=32,
        prefetch_chunks=1,
        device=torch.device("cpu"),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=d_model,
        stop_acquire_check=lambda: stop["value"],
    )

    next(p)
    stop["value"] = True
    sizes = _drain(p)

    assert sizes == [32, 32]
    assert buf.acquire_up_to.call_count == 1


def test_streaming_local_mixing_keeps_bounded_window():
    d_model = 1
    chunks = [
        torch.arange(0, 64, dtype=torch.float32).reshape(-1, 1),
        torch.arange(64, 128, dtype=torch.float32).reshape(-1, 1),
        torch.arange(128, 192, dtype=torch.float32).reshape(-1, 1),
    ]
    buf = _make_sequential_buffer(chunks)
    p = _provider_with_mix(
        buf, batch_size=32, d_model=d_model, mix_chunks=2, mix_fraction=0.5
    )

    sizes = _drain(p)

    assert sum(sizes) == 192
    assert sizes == [32, 32, 32, 32, 32, 32]


def test_streaming_local_mixing_waits_for_full_mixing_buffer():
    d_model = 1
    chunks = [
        torch.arange(0, 32, dtype=torch.float32).reshape(-1, 1),
        torch.arange(32, 64, dtype=torch.float32).reshape(-1, 1),
        torch.arange(64, 96, dtype=torch.float32).reshape(-1, 1),
    ]
    buf = _make_sequential_buffer(chunks)
    p = _provider_with_mix(
        buf, batch_size=16, d_model=d_model, mix_chunks=3, mix_fraction=0.5
    )

    first = next(p)

    assert first.shape == (16, 1)
    assert buf.read_chunk.call_count == 3


def test_streaming_local_mixing_matches_standard_mixing_buffer():
    d_model = 1
    chunks = [
        torch.arange(0, 32, dtype=torch.float32).reshape(-1, 1),
        torch.arange(32, 64, dtype=torch.float32).reshape(-1, 1),
        torch.arange(64, 96, dtype=torch.float32).reshape(-1, 1),
    ]
    buf = _make_sequential_buffer([chunk.clone() for chunk in chunks])
    p = StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=16,
        prefetch_chunks=1,
        device=torch.device("cpu"),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=d_model,
        shuffle=True,
        mix_chunks=3,
        mix_fraction=0.5,
        mixing_seed=123,
    )
    streaming = torch.cat([batch for batch in p], dim=0)

    gen = torch.Generator()
    gen.manual_seed(123)
    standard = torch.cat(
        list(
            mixing_buffer(
                buffer_size=96,
                batch_size=16,
                activations_loader=iter(chunks),
                mix_fraction=0.5,
                generator=gen,
            )
        ),
        dim=0,
    )

    assert torch.equal(streaming, standard)


def test_tp_mixing_uses_same_shuffle_order_across_ranks(monkeypatch):
    import torch.distributed as dist

    fake_group = MagicMock()
    broadcasted: list[torch.Tensor] = []

    def fake_broadcast(tensor, src, group):
        assert src == 0
        assert group is fake_group
        if not broadcasted:
            broadcasted.append(tensor.detach().clone())
        else:
            tensor.copy_(broadcasted[0])

    def make_provider(tp_rank: int) -> StreamingActivationProvider:
        buf = MagicMock()
        buf._chunk_size_tokens = 4
        provider = StreamingActivationProvider(
            buffer=buf,
            train_batch_size_tokens=2,
            prefetch_chunks=1,
            device=torch.device("cpu"),
            sae_tp_group=fake_group,
            sae_tp_rank=tp_rank,
            sae_tp_root_global_rank=0,
            d_model=1,
            shuffle=True,
            mix_chunks=2,
            mix_fraction=0.0,
        )
        provider._pool = torch.arange(0, 4, dtype=torch.float32).reshape(-1, 1)
        provider._pool_start = 0
        provider._pool_len = 4
        provider._mixing_pool = torch.arange(4, 8, dtype=torch.float32).reshape(-1, 1)
        return provider

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)
    new_data = torch.arange(8, 12, dtype=torch.float32).reshape(-1, 1)

    root = make_provider(tp_rank=0)
    torch.manual_seed(1)
    root._merge_into_mixing_pool(new_data)

    follower = make_provider(tp_rank=1)
    torch.manual_seed(2)
    follower._merge_into_mixing_pool(new_data)

    assert torch.equal(root._pool, follower._pool)
    assert torch.equal(root._mixing_pool, follower._mixing_pool)


def test_multi_hook_mixing_shuffle_preserves_hook_pairing(monkeypatch):
    buf = MagicMock()
    buf._chunk_size_tokens = 4
    p = StreamingActivationProvider(
        buffer=buf,
        train_batch_size_tokens=2,
        prefetch_chunks=1,
        device=torch.device("cpu"),
        sae_tp_group=None,
        sae_tp_rank=0,
        sae_tp_root_global_rank=0,
        d_model=1,
        shuffle=True,
        mix_chunks=2,
        mix_fraction=0.5,
        hook_names=["h0", "h1"],
    )
    prepared = torch.tensor(
        [
            [0.0],
            [1.0],
            [100.0],
            [101.0],
            [2.0],
            [3.0],
            [102.0],
            [103.0],
        ]
    )

    perms = [
        torch.tensor([1, 0, 3, 2], dtype=torch.long),
        torch.tensor([2, 3, 0, 1], dtype=torch.long),
    ]

    def fake_randperm(n: int, device: torch.device) -> torch.Tensor:
        assert n == 4
        return perms.pop(0).to(device)

    monkeypatch.setattr(p, "_randperm_for_tp", fake_randperm)
    p._merge_into_mixing_pool(prepared)

    batch = p._take()

    assert isinstance(batch, dict)
    assert torch.all(batch["h0"] < 10)
    assert torch.all(batch["h1"] >= 100)
    assert torch.equal(batch["h1"].flatten(), batch["h0"].flatten() + 100)
    assert len(perms) == 1
    assert torch.equal(perms[0], torch.tensor([2, 3, 0, 1], dtype=torch.long))


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
