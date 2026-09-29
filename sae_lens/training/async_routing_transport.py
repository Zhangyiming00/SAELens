"""Bounded, ordered SHM transport for static routing (no worker collectives).

Each producer/SAE endpoint edge has one ring, shared by its remote TP readers.
The source rank is excluded: its slice stays on the GPU. Writers stage D2H on
a separate CUDA stream; readers copy SHM into pinned memory and prefetch H2D.
Sequence numbers and reader ACKs prevent overwrite, duplication and reordering.
Generation, filtering, mixing and training remain on their existing threads.
"""
from __future__ import annotations

import fcntl
import json
import mmap
import os
import queue
import struct
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import torch


class _Stopped(Exception):
    pass


@dataclass(frozen=True)
class RoutingEdge:
    producer: int
    consumer: int
    endpoint: int
    source: int
    readers: tuple[int, ...]
    rows: int
    hooks: tuple[str, ...]


class _Ring:
    def __init__(self, transport, edge, *, writer):
        self.transport, self.edge = transport, edge
        self.writer = writer
        self.path = transport.directory / f"p{edge.producer}_e{edge.endpoint}"
        self.shape = (edge.rows * len(edge.hooks), transport.width)
        self.header_bytes = 16 * transport.slots  # sequence, ACK count per slot
        self.payload_bytes = self.shape[0] * self.shape[1] * torch.empty((), dtype=transport.dtype).element_size()
        self.size = self.header_bytes + transport.slots * self.payload_bytes
        self.fd = None
        self.mm = None
        self.data = None
        self.owned = False
        transport.rings.append(self)
        self.spec = dict(shape=self.shape, slots=transport.slots, dtype=str(transport.dtype), hooks=edge.hooks)
        if writer:
            # Reserve tmpfs pages before mmap writes: ENOSPC must be an exception,
            # not a process-killing SIGBUS partway through publishing a batch.
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            self.owned = True
            try:
                os.posix_fallocate(fd, 0, self.size)
                for slot in range(transport.slots):
                    os.pwrite(fd, struct.pack("qq", -1, len(edge.readers)), 16 * slot)
            except BaseException:
                self.path.unlink(missing_ok=True)
                raise
            finally:
                os.close(fd)
            temporary = self.path.with_suffix(".creating")
            temporary.write_text(json.dumps(self.spec))
            temporary.replace(self.path.with_suffix(".ready"))

    def open(self):
        self.transport.wait_until(lambda: self.path.with_suffix(".ready").exists())
        if json.loads(self.path.with_suffix(".ready").read_text()) != json.loads(json.dumps(self.spec)):
            raise ValueError(f"SHM routing payload configuration mismatch: {self.path}")
        self.fd = os.open(self.path, os.O_RDWR)
        if os.fstat(self.fd).st_size != self.size:
            raise ValueError(f"SHM routing payload configuration mismatch: {self.path}")
        self.mm = mmap.mmap(self.fd, self.size)
        self.data = torch.frombuffer(self.mm, dtype=self.transport.dtype,
                                     offset=self.header_bytes).view(self.transport.slots, *self.shape)

    @contextmanager
    def locked(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def state(self, slot):
        return struct.unpack_from("qq", self.mm, slot * 16)

    def writable(self, slot):
        with self.locked():
            return self.state(slot)[1] == len(self.edge.readers)

    def readable(self, slot, sequence):
        with self.locked():
            observed, _ = self.state(slot)
            if observed > sequence:
                raise RuntimeError("SHM routing sequence overwritten before consumption")
            return observed == sequence

    def publish(self, sequence, host):
        slot = sequence % self.transport.slots
        self.transport.wait_until(lambda: self.writable(slot))
        self.data[slot].copy_(host)
        with self.locked():
            struct.pack_into("qq", self.mm, slot * 16, sequence, 0)

    def read(self, sequence, host):
        slot = sequence % self.transport.slots
        self.transport.wait_until(lambda: self.readable(slot, sequence))
        # The writer cannot recycle this slot until every reader ACKs. Copying
        # outside the lock allows TP readers to copy concurrently.
        host.copy_(self.data[slot])
        with self.locked():
            observed, acknowledgements = self.state(slot)
            if observed != sequence or acknowledgements >= len(self.edge.readers):
                raise RuntimeError("Invalid SHM routing acknowledgement")
            struct.pack_into("qq", self.mm, slot * 16, sequence, acknowledgements + 1)

    def close(self):
        self.data = None
        if self.mm is not None:
            self.mm.close()
        if self.fd is not None:
            os.close(self.fd)
        if self.owned:
            self.path.with_suffix(".creating").unlink(missing_ok=True)
            self.path.with_suffix(".ready").unlink(missing_ok=True)
            self.path.unlink(missing_ok=True)


class _Sender:
    def __init__(self, transport, edge):
        self.transport = transport
        self.ring = _Ring(transport, edge, writer=True)
        self.hosts = [transport.host(self.ring.shape) for _ in range(transport.slots)]
        self.free = queue.Queue()
        for i in range(transport.slots):
            self.free.put(i)
        self.sources = [None] * transport.slots
        self.jobs = queue.Queue()
        self.stream = torch.cuda.Stream(device=transport.device) if transport.cuda else None
        self.events = [torch.cuda.Event() for _ in range(transport.slots)] if transport.cuda else []
        self.sequence = 0
        self.thread = transport.start(self.run, f"routing-send-{edge.producer}-{edge.endpoint}")

    def submit(self, payload):
        if tuple(payload.shape) != self.ring.shape or payload.dtype != self.transport.dtype:
            raise ValueError("SHM routing requires the configured payload shape and dtype")
        slot = self.transport.get(self.free)
        # Reclaim CUDA objects on the issuing thread. A background destructor
        # may need the CUDA driver lock held by a training scalar read waiting
        # for peers that have not yet received their activations.
        self.sources[slot] = payload
        event = None
        if self.transport.cuda:
            if payload.device != self.transport.device:
                raise ValueError("Routing payload must be on its producer device")
            event = self.events[slot]
            current = torch.cuda.current_stream(self.transport.device)
            with torch.cuda.stream(self.stream):
                self.stream.wait_stream(current)
                self.hosts[slot].copy_(payload, non_blocking=True)
                event.record(self.stream)
                # The source is strongly retained through D2H completion.
                # Avoid allocator event insertion from a background destructor.
        else:
            self.hosts[slot].copy_(payload)
        # Retain the source until D2H completes, including externally owned data.
        self.jobs.put((self.sequence, slot))
        self.sequence += 1
        self.transport.stats["sent_bytes"] += self.ring.payload_bytes

    def run(self):
        self.ring.open()
        while True:
            sequence, slot = self.transport.get(self.jobs)
            if self.transport.cuda:
                self.events[slot].synchronize()
            self.ring.publish(sequence, self.hosts[slot])
            self.free.put(slot)


class _Receiver:
    def __init__(self, transport, edge):
        self.transport = transport
        self.ring = _Ring(transport, edge, writer=False)
        self.ready = queue.Queue()
        self.capacity = queue.Queue()
        for _ in range(transport.slots):
            self.capacity.put(None)
        self.thread = transport.start(self.run, f"routing-recv-{edge.producer}-{edge.endpoint}")
        self.sequence = 0

    def run(self):
        self.ring.open()
        host = self.transport.host(self.ring.shape)
        stream = torch.cuda.Stream(device=self.transport.device) if self.transport.cuda else None
        done = torch.cuda.Event() if self.transport.cuda else None
        sequence = 0
        while True:
            self.transport.get(self.capacity)
            self.ring.read(sequence, host)
            if self.transport.cuda:
                with torch.cuda.stream(stream):
                    payload = host.to(self.transport.device, non_blocking=True)
                    done.record(stream)
                # Only this worker waits; the trainer never performs H2D or a
                # device-wide synchronize. Pinned storage is safe to reuse.
                done.synchronize()
            else:
                payload = host.clone()
            self.ready.put((sequence, payload))
            sequence += 1

    def receive(self):
        sequence, payload = self.transport.get(self.ready)
        if sequence != self.sequence:
            raise RuntimeError("Out-of-order SHM routing payload")
        self.sequence += 1
        if self.transport.cuda:
            payload.record_stream(torch.cuda.current_stream(self.transport.device))
        self.capacity.put(None)
        self.transport.stats["received_bytes"] += self.ring.payload_bytes
        return payload


class AsyncRoutingTransport:
    def __init__(self, *, directory, edges, rank, device, dtype, width, slots=2,
                 timeout=300.0, check_failure=None):
        if type(slots) is not int or slots < 1:
            raise ValueError("routing_shm_slots must be a positive integer")
        self.directory = Path(directory)
        self.rank, self.device, self.dtype = rank, torch.device(device), dtype
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.width, self.slots, self.timeout = width, slots, timeout
        self.cuda = self.device.type == "cuda"
        self.check_failure = check_failure
        self.stop = threading.Event()
        self.error = None
        self.threads = []
        self.rings = []
        self.senders, self.receivers = {}, {}
        self.stats = dict(sent_bytes=0, received_bytes=0, local_routes=0)
        self.closed = False
        try:
            # Create all owned files before waiting for any remote writer.
            for edge in edges:
                key = edge.producer, edge.endpoint
                if edge.source == rank and edge.readers:
                    self.senders[key] = _Sender(self, edge)
            for edge in edges:
                if rank in edge.readers:
                    self.receivers[edge.producer, edge.endpoint] = _Receiver(self, edge)
        except BaseException as exc:
            self.fail(exc)
            self.close()
            raise

    def host(self, shape):
        return torch.empty(shape, dtype=self.dtype, device="cpu", pin_memory=self.cuda)

    def fail(self, exc):
        self.error = exc
        with suppress(OSError):
            (self.directory / f"error_rank{self.rank}").write_text(repr(exc))

    def check(self):
        if self.stop.is_set():
            raise _Stopped()
        if self.error is not None:
            raise RuntimeError("Asynchronous routing worker failed") from self.error
        errors = list(self.directory.glob("error_rank*"))
        if errors:
            raise RuntimeError(f"Remote SHM routing failure: {errors[0].read_text()}")

    def wait_until(self, predicate):
        deadline = time.monotonic() + self.timeout
        while True:
            self.check()
            if predicate():
                return
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for SHM routing peer")
            self.stop.wait(0.001)

    def get(self, q):
        deadline = time.monotonic() + self.timeout
        while True:
            self.check()
            # The failure monitor may perform control operations; keep it on
            # the caller thread, never on data ingress/egress workers.
            if threading.current_thread() is threading.main_thread() and self.check_failure:
                self.check_failure()
            try:
                return q.get(timeout=0.02)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise TimeoutError("Timed out waiting for asynchronous routing")

    def start(self, fn, name):
        def run():
            try:
                if self.cuda:
                    torch.cuda.set_device(self.device)
                fn()
            except _Stopped:
                pass
            except BaseException as exc:
                self.fail(exc)
        thread = threading.Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("SHM routing worker did not stop; mappings retained")
        # Finish outstanding D2H before releasing pinned staging/source buffers.
        for sender in self.senders.values():
            if sender.stream is not None:
                sender.stream.synchronize()
        for ring in self.rings:
            ring.close()
        self.senders.clear()
        self.receivers.clear()


def exchange(store, outgoing):
    """Submit outgoing slices and return this rank's ordered incoming slices."""
    import sae_lens.distributed_v2 as routing
    runtime = routing.get_sae_runtime()
    transport = getattr(runtime, "routing_transport_instance", None)
    if transport is None:
        edges = []
        for endpoint in runtime.endpoints:
            for route in routing.get_routing_table():
                if route.consumer_idx != endpoint.replica_index:
                    continue
                source = routing.get_producer_tp_root(route.producer_idx)
                edges.append(RoutingEdge(
                    route.producer_idx, route.consumer_idx, endpoint.index, source,
                    tuple(r for r in endpoint.tp_ranks if r != source),
                    route.row_end - route.row_start,
                    tuple(store._v2_endpoint_hook_names(endpoint.index)),
                ))
        monitor = getattr(runtime, "failure_monitor", None)
        transport = AsyncRoutingTransport(
            directory=runtime.routing_shm_directory, edges=edges,
            rank=torch.distributed.get_rank(), device=store.device, dtype=store.dtype,
            width=store._v2_payload_width(), slots=runtime.routing_shm_slots,
            check_failure=monitor.check if monitor is not None else None,
        )
        runtime.routing_transport_instance = transport
    for sender in transport.senders.values():
        edge = sender.ring.edge
        sender.submit(store._pack_v2_payload(outgoing[edge.consumer], list(edge.hooks)))
    incoming = {}
    if routing.is_consumer():
        endpoint = routing.get_sae_endpoint_idx()
        consumer = routing.get_consumer_idx()
        for route in routing.get_routing_table():
            if route.consumer_idx != consumer:
                continue
            producer = route.producer_idx
            if routing.get_producer_tp_root(producer) == transport.rank:
                incoming[producer] = outgoing[consumer]
                transport.stats["local_routes"] += 1
            else:
                receiver = transport.receivers[producer, endpoint]
                incoming[producer] = store._unpack_v2_payload(
                    receiver.receive(), n_rows=receiver.ring.edge.rows,
                    hook_names=list(receiver.ring.edge.hooks), is_multi_hook=store.is_multi_hook,
                )
    return incoming
