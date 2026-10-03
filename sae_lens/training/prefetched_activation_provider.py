"""Bounded prefetch of a local logical source, before exact DP collectives.

The source must not issue distributed collectives. Its mixing state stays on
the permanent source rank across DP changes; only the caller does DP dispatch.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sae_lens.training.mixing_buffer import ActivationBatch


@dataclass
class _Ready:
    batch: ActivationBatch
    event: Any
    timing: dict[str, float]


_EOF = object()


class PrefetchedActivationProvider:
    def __init__(self, source, *, device, capacity: int, stop_acquire: Callable[[], None]):
        if capacity < 1:
            raise ValueError("prefetch capacity must be positive")
        self._source = source
        self._device = torch.device(device)
        self._stop_acquire = stop_acquire
        self._stop = threading.Event()
        self._drain = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._error: BaseException | None = None
        self._finished = False
        self._closed = False
        self._timing: dict[str, float] = {}
        self._initial_event = None
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                self._initial_event = torch.cuda.Event()
                self._initial_event.record(torch.cuda.current_stream(self._device))
        self._thread = threading.Thread(target=self._run, name="exact-source-prefetch", daemon=True)
        self._thread.start()

    def __iter__(self):
        return self

    def __next__(self) -> ActivationBatch:
        if self._finished or self._closed:
            raise StopIteration
        while True:
            if self._stop.is_set():
                raise StopIteration
            if self._error is not None:
                raise RuntimeError("exact source prefetch failed") from self._error
            try:
                item = self._queue.get(timeout=0.1)
                break
            except queue.Empty:
                if not self._thread.is_alive():
                    if self._error is not None:
                        raise RuntimeError("exact source prefetch failed") from self._error
                    if not self._queue.empty():
                        continue
                    raise RuntimeError("exact source prefetch exited without EOF")
        if item is _EOF:
            self._finished = True
            raise StopIteration
        assert isinstance(item, _Ready)
        if item.event is not None:
            stream = torch.cuda.current_stream(self._device)
            stream.wait_event(item.event)
            tensors = item.batch.values() if isinstance(item.batch, dict) else [item.batch]
            for tensor in tensors:
                tensor.record_stream(stream)
        self._timing = item.timing
        return item.batch

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self, stream=None):
        while not self._stop.is_set():
            if self._drain.is_set():
                self._source.request_drain_local_pool()
                self._drain.clear()
            try:
                batch = next(self._source)
            except StopIteration:
                self._put(_EOF)
                return
            consume = getattr(self._source, "consume_last_data_timing", None)
            timing = consume() if callable(consume) else {}
            event = None
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
            if not self._put(_Ready(batch, event, timing)):
                return

    def _run(self):
        try:
            with torch.no_grad():
                if self._device.type == "cuda":
                    with torch.cuda.device(self._device):
                        stream = torch.cuda.Stream(device=self._device)
                        stream.wait_event(self._initial_event)
                        try:
                            with torch.cuda.stream(stream):
                                self._produce(stream)
                        finally:
                            stream.synchronize()
                else:
                    self._produce()
        except BaseException as exc:
            self._error = exc

    def request_drain_local_pool(self):
        # The worker alone mutates the generator and its mixing state.
        self._stop_acquire()
        self._drain.set()

    def consume_last_data_timing(self):
        result, self._timing = self._timing, {}
        return result

    def close(self):
        if self._closed:
            return
        self._stop_acquire()
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            # Callers must not unmap SHM while the worker still owns a read.
            raise RuntimeError("exact source prefetch did not stop; SHM must remain mapped")
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def build_prefetched_logical_source(*, raw_kwargs, logical_kwargs, capacity, tp_size, pp_size):
    """One local source only: never put TP/PP/NCCL operations in a worker."""
    from sae_lens.training.logical_streaming_mixer import LogicalStreamingMixingProvider
    from sae_lens.training.streaming_activation_provider import StreamingActivationProvider

    if capacity and (tp_size != 1 or pp_size != 1):
        raise ValueError("exact source prefetch currently requires SAE TP1/PP1")
    kwargs = dict(raw_kwargs)
    stop = threading.Event()
    previous_stop = kwargs.get("stop_acquire_check")
    if capacity:
        if kwargs.get("sae_dp_size", 1) != 1:
            raise ValueError("prefetched source must own one logical DP stream")
        if kwargs.get("sae_tp_group") is not None or kwargs.get("pp_root_group") is not None:
            raise ValueError("prefetched logical source cannot contain collective groups")
        kwargs["stop_acquire_check"] = lambda: stop.is_set() or bool(previous_stop and previous_stop())
        kwargs["pin_memory_transfer"] = True
    raw = StreamingActivationProvider(**kwargs)
    logical = LogicalStreamingMixingProvider(source=raw, **logical_kwargs)
    if not capacity:
        return logical
    return PrefetchedActivationProvider(logical, device=kwargs["device"], capacity=capacity, stop_acquire=stop.set)
