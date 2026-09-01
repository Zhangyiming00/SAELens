from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


class AsyncJsonlWriter:
    """Tiny CPU-only JSONL writer so logging never blocks the producer hot path."""

    def __init__(self, path: Path | None, *, t0: float | None = None):
        self._path = path
        self._t0 = t0
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            self._thread = threading.Thread(
                target=self._run,
                name=f"jsonl-writer:{path.name}",
                daemon=True,
            )
            self._thread.start()

    def log(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        self._raise_if_error()
        payload = dict(record)
        if self._t0 is not None and "elapsed_s" not in payload:
            payload["elapsed_s"] = time.time() - self._t0
        self._q.put(payload)

    def close(self) -> None:
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join()
        self._thread = None
        self._raise_if_error()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("async JSONL writer failed") from self._error

    def _run(self) -> None:
        assert self._path is not None
        try:
            with self._path.open("a", buffering=1024 * 1024) as f:
                while True:
                    item = self._q.get()
                    try:
                        if item is None:
                            return
                        json.dump(item, f)
                        f.write("\n")
                    finally:
                        self._q.task_done()
        except BaseException as exc:
            self._error = exc


@dataclass
class _Slot:
    host: torch.Tensor
    d2h_start: torch.cuda.Event
    d2h_done: torch.cuda.Event


@dataclass
class _WriteJob:
    slot_idx: int
    chunk_idx: int
    seq_no: int
    step: int
    valid_rows: int
    valid_tokens_per_hook: int
    inference_time_s: float
    staging_wait_s: float
    submitted_at: float


class AsyncVLLMShmWriter:
    """Overlap vLLM compute with GPU->pinned-host copy and SHM publication.

    The producer thread:
      1. produces a packed GPU activation chunk,
      2. enqueues a non-blocking D2H into one of a few pinned host slots,
      3. immediately continues with the next vLLM iteration.

    The background writer waits for the CUDA event, copies pinned host memory
    into SharedActivationBuffer, then publishes READY.

    No CUDA synchronize occurs on the producer thread.
    """

    def __init__(
        self,
        *,
        buffer: Any,
        device: torch.device,
        dtype: torch.dtype,
        max_rows: int,
        d_model: int,
        producer_id: int,
        staging_slots: int = 2,
        on_written: Callable[[dict[str, Any]], None] | None = None,
    ):
        if staging_slots < 2:
            raise ValueError("staging_slots must be >= 2")
        if device.type != "cuda":
            raise ValueError("AsyncVLLMShmWriter requires a CUDA producer device")

        self._buffer = buffer
        self._device = device
        self._producer_id = int(producer_id)
        self._on_written = on_written

        # Pinned host storage is deliberately fixed-size and reused.  This removes
        # the per-chunk CPU cat/allocation path from get_streaming_activations().
        self._slots = [
            _Slot(
                host=torch.empty(
                    (max_rows, d_model),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                ),
                d2h_start=torch.cuda.Event(enable_timing=True),
                d2h_done=torch.cuda.Event(enable_timing=True),
            )
            for _ in range(staging_slots)
        ]
        self._free: queue.Queue[int] = queue.Queue()
        for i in range(staging_slots):
            self._free.put(i)

        self._jobs: queue.Queue[_WriteJob | None] = queue.Queue()
        self._copy_stream = torch.cuda.Stream(device=device)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-shm-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        chunk_idx: int,
        seq_no: int,
        step: int,
        activations_gpu: torch.Tensor,
        valid_rows: int,
        valid_tokens_per_hook: int,
        inference_time_s: float,
    ) -> float:
        """Enqueue one packed activation chunk.

        Returns CPU-side wait for a reusable staging slot.  This is the only
        producer-visible backpressure introduced by the local staging ring.
        """
        self._raise_if_error()
        if self._closed:
            raise RuntimeError("submit() called after AsyncVLLMShmWriter.close()")
        if valid_rows <= 0:
            raise ValueError("valid_rows must be > 0")
        if activations_gpu.device.type != "cuda":
            raise ValueError("activations_gpu must be CUDA")
        if activations_gpu.ndim != 2:
            raise ValueError(
                f"expected 2D packed activations, got shape={tuple(activations_gpu.shape)}"
            )
        if valid_rows > activations_gpu.shape[0]:
            raise ValueError(
                f"valid_rows={valid_rows} exceeds source rows={activations_gpu.shape[0]}"
            )

        wait_t0 = time.perf_counter()
        slot_idx = self._free.get()
        staging_wait_s = time.perf_counter() - wait_t0
        self._raise_if_error()

        slot = self._slots[slot_idx]
        src = activations_gpu[:valid_rows]
        if src.dtype != slot.host.dtype:
            # Keep SHM dtype identical to the configured activation dtype.
            src = src.to(dtype=slot.host.dtype)

        producer_stream = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(producer_stream)
            slot.d2h_start.record(self._copy_stream)
            slot.host[:valid_rows].copy_(src, non_blocking=True)
            slot.d2h_done.record(self._copy_stream)
            # The caching allocator must not recycle src storage before the copy
            # stream has consumed it.
            src.record_stream(self._copy_stream)

        self._jobs.put(
            _WriteJob(
                slot_idx=slot_idx,
                chunk_idx=int(chunk_idx),
                seq_no=int(seq_no),
                step=int(step),
                valid_rows=int(valid_rows),
                valid_tokens_per_hook=int(valid_tokens_per_hook),
                inference_time_s=float(inference_time_s),
                staging_wait_s=float(staging_wait_s),
                submitted_at=time.time(),
            )
        )
        return staging_wait_s

    def drain(self) -> None:
        """Wait for all already-submitted chunks to become READY in SHM."""
        self._jobs.join()
        self._raise_if_error()

    def close(self) -> None:
        if self._closed:
            return
        self.drain()
        self._closed = True
        self._jobs.put(None)
        self._thread.join()
        self._raise_if_error()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("async vLLM SHM writer failed") from self._error

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self._device)
            while True:
                job = self._jobs.get()
                try:
                    if job is None:
                        return
                    slot = self._slots[job.slot_idx]

                    # Synchronization is confined to this CPU background thread.
                    slot.d2h_done.synchronize()
                    d2h_time_s = (
                        slot.d2h_start.elapsed_time(slot.d2h_done) / 1000.0
                    )

                    write_t0 = time.perf_counter()
                    self._buffer.write_chunk(
                        job.chunk_idx,
                        slot.host[: job.valid_rows],
                        job.valid_rows,
                        producer_id=self._producer_id,
                    )
                    self._buffer.mark_ready(job.chunk_idx)
                    shm_write_time_s = time.perf_counter() - write_t0

                    if self._on_written is not None:
                        self._on_written(
                            {
                                "event": "chunk_written",
                                "chunk_idx": job.chunk_idx,
                                "seq_no": job.seq_no,
                                "step": job.step,
                                # Backward compatible: valid_tokens is physical
                                # packed rows across all hooks.
                                "valid_tokens": job.valid_rows,
                                "valid_rows_total": job.valid_rows,
                                "valid_tokens_per_hook": job.valid_tokens_per_hook,
                                "inference_time_s": job.inference_time_s,
                                "d2h_time_s": d2h_time_s,
                                "write_time_s": shm_write_time_s,
                                "staging_wait_s": job.staging_wait_s,
                                "async_writer": True,
                            }
                        )
                finally:
                    if job is not None:
                        self._free.put(job.slot_idx)
                    self._jobs.task_done()
        except BaseException as exc:
            self._error = exc
            # Mark queued jobs complete so a producer blocked in drain() can
            # observe the original writer failure instead of hanging forever.
            while True:
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._jobs.task_done()
            # Unblock producer if it is waiting for a slot.
            for i in range(len(self._slots)):
                try:
                    self._free.put_nowait(i)
                except queue.Full:
                    pass
