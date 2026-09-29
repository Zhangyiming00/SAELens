"""Bounded CPU reproductions of the threaded SHM consumer protocol."""

import threading
import time
from concurrent.futures import Future

import pytest
import torch

from sae_lens.training.async_streaming_activation_provider import (
    AsyncStreamingActivationProvider,
)
from sae_lens.training.shared_activation_buffer import SharedActivationBuffer


@pytest.mark.parametrize(
    "batch,chunk,mix", [(66, 64, 2), (33, 64, 2), (22, 64, 2), (65, 64, 0), (64, 64, 2)]
)
def test_unaligned_chunks_drain_without_loss(tmp_path, batch, chunk, mix):
    fast = _drain(tmp_path, batch, chunk, mix, delay=0)
    slow = _drain(tmp_path, batch, chunk, mix, delay=0.002)
    assert torch.equal(fast, slow), "Reader timing changed token order"


def _drain(tmp_path, batch, chunk, mix, delay):
    count = 13
    buffer = SharedActivationBuffer(
        name="unaligned",
        num_chunks=count,
        chunk_size_tokens=chunk * 2,
        d_model=1,
        num_producers=1,
        target_chunks=count,
        create=True,
        dtype=torch.float32,
        base_dir=str(tmp_path),
    )
    for i in range(count):
        slot, _ = buffer.allocate_write_chunk()
        values = torch.arange(i * chunk, (i + 1) * chunk, dtype=torch.float32)[:, None]
        buffer.write_chunk(slot, torch.cat([values, values + 10000]), chunk * 2)
        buffer.mark_ready(slot)
    buffer.signal_done()
    original_read = buffer.read_chunk

    def read_chunk(index):
        time.sleep(delay)
        return original_read(index)

    buffer.read_chunk = read_chunk
    provider = AsyncStreamingActivationProvider(
        buffer=buffer,
        train_batch_size_tokens=batch,
        prefetch_chunks=2,
        device="cpu",
        d_model=1,
        dtype=torch.float32,
        hook_names=["h0", "h1"],
        mix_chunks=mix,
        shuffle=True,
        random_chunks=False,
    )
    result = Future()

    def consume():
        try:
            result.set_result([{h: x.clone() for h, x in b.items()} for b in provider])
        except BaseException as exc:
            result.set_exception(exc)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        batches = result.result(timeout=5)
        first = torch.cat([b["h0"] for b in batches])[:, 0]
        second = torch.cat([b["h1"] for b in batches])[:, 0]
        assert torch.equal(
            first.sort().values, torch.arange(count * chunk, dtype=torch.float32)
        )
        assert torch.equal(second, first + 10000)
        assert all(b["h0"].shape[0] == batch for b in batches[:-1])
        return first
    finally:
        if worker.is_alive():
            provider._set_error(RuntimeError("test deadline exceeded"))
        provider.close()
        worker.join(timeout=1)
        buffer.close()


def test_empty_eof_stops_prefetch_thread(tmp_path):
    buffer = SharedActivationBuffer(
        name="empty",
        num_chunks=2,
        chunk_size_tokens=64,
        d_model=1,
        num_producers=1,
        target_chunks=1,
        create=True,
        dtype=torch.float32,
        base_dir=str(tmp_path),
    )
    buffer.signal_done()
    provider = AsyncStreamingActivationProvider(
        buffer=buffer,
        train_batch_size_tokens=66,
        prefetch_chunks=2,
        device="cpu",
        d_model=1,
        dtype=torch.float32,
        mix_chunks=2,
    )
    try:
        provider.start()
        provider._prefetch_thread.join(timeout=2)
        assert not provider._prefetch_thread.is_alive()
        with pytest.raises(StopIteration):
            next(provider)
    finally:
        provider.close()
        buffer.close()
