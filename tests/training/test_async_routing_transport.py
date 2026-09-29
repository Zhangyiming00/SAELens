"""Transport ordering, multicast backpressure and peer failure without CUDA."""
import errno
import threading
import time

import pytest
import torch

from sae_lens.training.async_routing_transport import AsyncRoutingTransport, RoutingEdge


def make(tmp_path, rank, edges, **kwargs):
    return AsyncRoutingTransport(directory=tmp_path, rank=rank, edges=edges,
                                 device="cpu", dtype=torch.bfloat16, width=3,
                                 slots=1, timeout=5, **kwargs)


def test_ring_multicast_preserves_order_and_bounds_prefetch(tmp_path):
    edge = RoutingEdge(0, 0, 0, 0, (1, 2), 4, ("a", "b"))
    transports = [make(tmp_path, rank, [edge]) for rank in range(3)]
    sent = []
    failures = []

    def send():
        try:
            for step in range(12):
                transports[0].senders[0, 0].submit(torch.full((8, 3), step, dtype=torch.bfloat16))
                sent.append(step)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=send)
    thread.start()
    try:
        time.sleep(0.1)
        assert len(sent) < 12, "bounded ring must backpressure an unconsumed reader"
        for step in range(12):
            for rank in (1, 2):
                value = transports[rank].receivers[0, 0].receive()
                torch.testing.assert_close(value, torch.full_like(value, step), rtol=0, atol=0)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not failures
        assert transports[0].stats["sent_bytes"] == 12 * 8 * 3 * 2
        assert transports[0].stats["received_bytes"] == 0
    finally:
        for transport in transports:
            transport.close()
        thread.join(timeout=5)
    assert not list(tmp_path.iterdir())
    assert not any(t.is_alive() for p in transports for t in p.threads)


def test_local_only_edge_creates_no_shm_or_workers(tmp_path):
    edge = RoutingEdge(0, 0, 0, 0, (), 4, ("a",))
    transport = make(tmp_path, 0, [edge])
    try:
        assert not transport.senders and not transport.receivers
        assert not transport.threads
        assert not list(tmp_path.iterdir())
    finally:
        transport.close()


def test_remote_failure_unblocks_receive(tmp_path):
    edge = RoutingEdge(0, 0, 0, 0, (1,), 4, ("a",))
    sender = make(tmp_path, 0, [edge])
    receiver = make(tmp_path, 1, [edge])
    try:
        sender.fail(ValueError("producer failed before publication"))
        with pytest.raises(RuntimeError, match="routing.*fail|routing worker failed"):
            receiver.receivers[0, 0].receive()
    finally:
        receiver.close()
        sender.close()


def test_payload_configuration_mismatch_is_reported(tmp_path):
    edge = RoutingEdge(0, 0, 0, 0, (1,), 4, ("a",))
    sender = make(tmp_path, 0, [edge])
    receiver = AsyncRoutingTransport(directory=tmp_path, rank=1, edges=[edge],
                                    device="cpu", dtype=torch.float16, width=3,
                                    slots=1, timeout=5)
    try:
        with pytest.raises(RuntimeError, match="routing.*fail|routing worker failed"):
            receiver.receivers[0, 0].receive()
    finally:
        receiver.close()
        sender.close()


def test_shm_capacity_failure_cleans_partial_ring(tmp_path, monkeypatch):
    from sae_lens.training import async_routing_transport as module

    def exhausted(*_args):
        raise OSError(errno.ENOSPC, "test SHM capacity exhausted")

    monkeypatch.setattr(module.os, "posix_fallocate", exhausted)
    edge = RoutingEdge(0, 0, 0, 0, (1,), 4, ("a",))
    with pytest.raises(OSError, match="capacity exhausted"):
        make(tmp_path, 0, [edge])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["error_rank0"]
