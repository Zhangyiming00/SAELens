"""Asynchronous /dev/shm activation provider for streaming SAE training.

The transport remains SharedActivationBuffer.  The important change is that the
trainer thread never performs SHM acquire/read/mixing/H2D work.  Two CPU-side
workers keep a fixed rolling mixing reservoir and a small GPU-ready batch queue
filled in the background.

Distributed semantics
---------------------
* One root per SAE-DP replica claims chunks from SharedActivationBuffer.
* Chunk metadata is fanned out through a tiny CPU-only mmap mailbox in /dev/shm.
  No background torch.distributed collective is issued.
* Every TP rank reads the same SHM chunks directly.  There is no activation
  broadcast on the SAE TP NCCL communicator, so background data ingress cannot
  reorder/compete with the trainer's TP collectives.
* Every PP stage copies only its assigned hook slices.  After copying, every
  PP*TP rank ACKs the mailbox; the replica root releases the chunk refcount only
  after all ACKs arrive.

Mixing semantics
----------------
The reservoir is page based.  At the start of each mixing generation, pages are
shuffled by index only (activation tensors are never globally permuted).  The
trainer consumes roughly ``1 - mix_fraction`` of the pages while the ingress
thread refills those page slots.  A new generation starts once the consumed
slots have been replenished.  This is the asynchronous analogue of the common
"consume half / refill half / reshuffle" SAE mixing buffer.
"""

from __future__ import annotations

import fcntl
import json
import math
import mmap
import os
import queue
import struct
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


_LOG_STOP = object()
_READY_EOF = object()


class AsyncJsonlLogger:
    """Buffered JSONL writer whose file IO never runs on the hot path."""

    def __init__(self, path: Path | None, *, truncate: bool = True) -> None:
        self._path = path
        self._q: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._closed = False
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            path.write_text("")
        self._thread = threading.Thread(
            target=self._run,
            name=f"jsonl-writer-{path.name}",
            daemon=True,
        )
        self._thread.start()

    def log(self, record: dict[str, Any]) -> None:
        if self._path is None or self._closed:
            return
        # Make a shallow copy so callers may safely reuse/mutate their dict.
        self._q.put(dict(record))

    def close(self) -> None:
        if self._path is None or self._closed:
            return
        self._closed = True
        self._q.put(_LOG_STOP)
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        assert self._path is not None
        with self._path.open("a", buffering=1024 * 1024) as f:
            while True:
                item = self._q.get()
                if item is _LOG_STOP:
                    break
                json.dump(item, f, separators=(",", ":"))
                f.write("\n")
            f.flush()



class _ReplicaMailbox:
    """Tiny CPU-only mmap mailbox for one SAE-DP replica.

    One writer (PP0/TP0) publishes chunk ids + valid-row metadata.  Every PP*TP
    rank ACKs after copying its private hook slice into its local reservoir.  The
    file lock protects only a few int64 values and never touches CUDA/NCCL.
    """

    _HEADER_QWORDS = 4  # generation, count, ack_count, closed

    def __init__(
        self,
        *,
        path: Path,
        max_chunks: int,
        is_root: bool,
        replica_size: int,
    ) -> None:
        self.path = path
        self.max_chunks = int(max_chunks)
        self.replica_size = int(replica_size)
        self._size = 8 * (self._HEADER_QWORDS + 2 * self.max_chunks)
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_root:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.ftruncate(fd, self._size)
            finally:
                os.close(fd)
        else:
            deadline = time.monotonic() + 60.0
            while True:
                try:
                    if path.exists() and path.stat().st_size == self._size:
                        break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"timed out waiting for async SHM mailbox {path}")
                time.sleep(0.002)
        self._fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, self._size)
        self._is_root = bool(is_root)
        if is_root:
            with self._locked(exclusive=True):
                self._write_qword(0, 0)
                self._write_qword(1, 0)
                self._write_qword(2, 0)
                self._write_qword(3, 0)
                self._mm.flush()

    class _Lock:
        def __init__(self, fd: int, exclusive: bool) -> None:
            self.fd = fd
            self.exclusive = exclusive
        def __enter__(self) -> None:
            fcntl.flock(self.fd, fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _locked(self, *, exclusive: bool) -> "_ReplicaMailbox._Lock":
        return _ReplicaMailbox._Lock(self._fd, exclusive)

    def _read_qword(self, index: int) -> int:
        return int(struct.unpack_from("q", self._mm, 8 * index)[0])

    def _write_qword(self, index: int, value: int) -> None:
        struct.pack_into("q", self._mm, 8 * index, int(value))

    def publish(self, indices: list[int] | None, valid_rows: list[int]) -> int:
        if not self._is_root:
            raise RuntimeError("only the replica root may publish async SHM metadata")
        if indices is not None and len(indices) > self.max_chunks:
            raise ValueError("mailbox publish exceeds max_chunks")
        with self._locked(exclusive=True):
            generation = self._read_qword(0) + 1
            count = -1 if indices is None else len(indices)
            self._write_qword(1, count)
            self._write_qword(2, 0)
            if indices is not None:
                if len(valid_rows) != len(indices):
                    raise ValueError("indices/valid_rows length mismatch")
                base = self._HEADER_QWORDS
                for pos in range(self.max_chunks):
                    self._write_qword(base + pos, indices[pos] if pos < count else -1)
                    self._write_qword(base + self.max_chunks + pos, valid_rows[pos] if pos < count else 0)
            # generation is written last: readers never observe a partially updated payload.
            self._write_qword(0, generation)
            self._mm.flush()
            return generation

    def wait_next(
        self,
        last_generation: int,
        *,
        stop: threading.Event,
    ) -> tuple[int, list[int] | None, list[int]]:
        while not stop.is_set():
            with self._locked(exclusive=False):
                generation = self._read_qword(0)
                if generation > last_generation:
                    count = self._read_qword(1)
                    if count < 0:
                        return generation, None, []
                    base = self._HEADER_QWORDS
                    indices = [self._read_qword(base + i) for i in range(count)]
                    valid = [self._read_qword(base + self.max_chunks + i) for i in range(count)]
                    return generation, indices, valid
            time.sleep(0.0005)
        return last_generation, None, []

    def ack(self, generation: int) -> None:
        with self._locked(exclusive=True):
            if self._read_qword(0) != generation:
                raise RuntimeError("async SHM mailbox generation advanced before ACK")
            self._write_qword(2, self._read_qword(2) + 1)

    def wait_all_acks(self, generation: int, *, stop: threading.Event) -> None:
        while not stop.is_set():
            with self._locked(exclusive=False):
                current = self._read_qword(0)
                ack_count = self._read_qword(2)
                if current != generation:
                    raise RuntimeError("async SHM mailbox generation advanced before all ACKs")
                if ack_count >= self.replica_size:
                    return
            time.sleep(0.0005)
        raise RuntimeError("async SHM mailbox stopped before all replica ACKs")

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            os.close(self._fd)

    def unlink(self) -> None:
        if not self._is_root:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


@dataclass
class _BatchSlot:
    pinned_by_hook: dict[str, torch.Tensor]
    device_by_hook: dict[str, torch.Tensor]
    ready_event: torch.cuda.Event | None
    consume_done_event: torch.cuda.Event | None
    valid_rows: int = 0
    has_ready_event: bool = False
    has_consume_event: bool = False


def _choose_page_tokens(batch_tokens: int, chunk_tokens_per_hook: int) -> int:
    """Choose a small contiguous page that divides both batch and chunk sizes."""
    upper = min(256, batch_tokens, chunk_tokens_per_hook)
    for candidate in (256, 128, 64, 32, 16, 8, 4, 2, 1):
        if candidate <= upper and batch_tokens % candidate == 0 and chunk_tokens_per_hook % candidate == 0:
            return candidate
    return 1


class AsyncStreamingActivationProvider:
    """Drop-in asynchronous replacement for ``StreamingActivationProvider``.

    The constructor intentionally accepts the old provider's distributed
    arguments as well as the newer SAE-DP/PP aliases.  This keeps the runner diff
    small and makes the provider usable across the current streaming topology
    variants.
    """

    def __init__(
        self,
        *,
        buffer: Any,
        train_batch_size_tokens: int,
        prefetch_chunks: int,
        device: torch.device | str,
        sae_tp_group: dist.ProcessGroup | None = None,  # compatibility only
        sae_tp_rank: int = 0,
        sae_tp_root_global_rank: int | None = None,  # compatibility only
        d_model: int,
        dtype: torch.dtype = torch.bfloat16,
        shm_log_path: Path | None = None,
        shuffle: bool = True,
        random_chunks: bool = True,
        mix_chunks: int = 8,
        mix_fraction: float = 0.5,
        mixing_seed: int = 42,
        mixing_shard_index: int = 0,
        stop_acquire_check: Callable[[], bool] | None = None,
        buffer_monitor_path: Path | None = None,
        hook_names: list[str] | None = None,
        force_multi_hook: bool = False,
        select_hook_names: list[str] | None = None,
        pp_root_group: dist.ProcessGroup | None = None,  # compatibility only
        pp_root_global_rank: int | None = None,  # compatibility only
        sae_dp_size: int = 1,
        sae_dp_idx: int = 0,
        sae_pp_size: int = 1,
        pp_rank: int = 0,
        dp_replica_group: dist.ProcessGroup | None = None,  # compatibility only
        dp_replica_root_global_rank: int | None = None,
        # New async coordination. Metadata uses a tiny /dev/shm mmap mailbox.
        replica_root_global_rank: int | None = None,
        coord_name: str | None = None,
        gpu_prefetch_batches: int = 4,
    ) -> None:
        del pp_root_group, pp_root_global_rank, dp_replica_group

        if train_batch_size_tokens <= 0:
            raise ValueError("train_batch_size_tokens must be > 0")
        if prefetch_chunks <= 0:
            raise ValueError("prefetch_chunks must be > 0")
        if not 0.0 <= mix_fraction < 1.0:
            raise ValueError("mix_fraction must be in [0, 1)")
        if gpu_prefetch_batches < 2:
            raise ValueError("gpu_prefetch_batches must be >= 2")

        self._buffer = buffer
        self._batch_size = int(train_batch_size_tokens)
        self._prefetch_chunks = int(prefetch_chunks)
        self._device = torch.device(device)
        self._dtype = dtype
        self._d_model = int(d_model)
        self._shuffle = bool(shuffle)
        self._random_chunks = bool(random_chunks)
        self._mix_fraction = float(mix_fraction)
        self._mix_chunks = int(mix_chunks)
        self._stop_acquire_check = stop_acquire_check
        self._sae_tp_rank = int(sae_tp_rank)
        self._sae_dp_size = int(sae_dp_size)
        self._sae_dp_idx = int(sae_dp_idx)
        self._sae_pp_size = int(sae_pp_size)
        self._pp_rank = int(pp_rank)
        self._replica_root_global = (
            replica_root_global_rank
            if replica_root_global_rank is not None
            else dp_replica_root_global_rank
        )
        if self._replica_root_global is None:
            self._replica_root_global = sae_tp_root_global_rank
        if sae_tp_group is not None and dist.is_initialized():
            self._sae_tp_size = int(dist.get_world_size(sae_tp_group))
        else:
            self._sae_tp_size = 1
        self._replica_size = self._sae_tp_size * self._sae_pp_size
        buffer_coord_name = getattr(buffer, "_name", None) or getattr(buffer, "name", None)
        self._coord_name = str(coord_name or buffer_coord_name or "")
        if self._replica_size > 1 and not self._coord_name:
            raise RuntimeError(
                "async SHM TP/PP coordination needs a stable buffer name; pass coord_name "
                "or expose SharedActivationBuffer._name"
            )

        self._hook_names = list(hook_names) if hook_names else None
        self._num_hooks = len(self._hook_names) if self._hook_names else 1
        self._is_multi_hook = bool(force_multi_hook or self._num_hooks > 1)
        if select_hook_names is None:
            self._selected_hooks = list(self._hook_names) if self._hook_names else ["__single__"]
        else:
            self._selected_hooks = list(select_hook_names)
        if not self._selected_hooks:
            raise ValueError("select_hook_names must contain at least one hook")
        if self._hook_names is not None:
            unknown = [h for h in self._selected_hooks if h not in self._hook_names]
            if unknown:
                raise ValueError(f"select_hook_names contains unknown hooks: {unknown}")
            self._hook_to_global = {name: i for i, name in enumerate(self._hook_names)}
        else:
            self._hook_to_global = {"__single__": 0}

        total_chunk_rows = int(getattr(buffer, "_chunk_size_tokens"))
        if total_chunk_rows % self._num_hooks != 0:
            raise ValueError(
                "SharedActivationBuffer chunk rows must be divisible by num_hooks: "
                f"{total_chunk_rows} % {self._num_hooks} != 0"
            )
        self._chunk_tokens_per_hook = total_chunk_rows // self._num_hooks
        self._page_tokens = _choose_page_tokens(self._batch_size, self._chunk_tokens_per_hook)
        self._pages_per_batch = self._batch_size // self._page_tokens
        self._pages_per_chunk = self._chunk_tokens_per_hook // self._page_tokens

        if self._mix_chunks > 0:
            mix_tokens = self._mix_chunks * self._chunk_tokens_per_hook
        else:
            mix_tokens = max(
                2 * self._batch_size,
                self._prefetch_chunks * self._chunk_tokens_per_hook,
            )
        mix_tokens = max(mix_tokens, self._batch_size, self._chunk_tokens_per_hook)
        self._capacity_pages = max(1, math.ceil(mix_tokens / self._page_tokens))
        # Keep the page reservoir aligned to full training batches.
        self._capacity_pages = max(
            self._pages_per_batch,
            (self._capacity_pages // self._pages_per_batch) * self._pages_per_batch,
        )
        self._capacity_tokens = self._capacity_pages * self._page_tokens

        desired_replace_pages = math.ceil(self._capacity_pages * (1.0 - self._mix_fraction))
        desired_replace_pages = max(
            desired_replace_pages,
            self._pages_per_batch,
            self._pages_per_chunk,
        )
        desired_replace_pages = math.ceil(desired_replace_pages / self._pages_per_batch) * self._pages_per_batch
        self._replace_pages_per_round = min(self._capacity_pages, desired_replace_pages)

        self._rng = random.Random(int(mixing_seed) + 1000003 * int(mixing_shard_index))
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._drain_requested = threading.Event()
        self._eof = False
        self._error: BaseException | None = None
        self._started = False
        self._closed = False
        self._consumed_tokens = 0
        self._consume_step = 0
        self._refill_count = 0
        self._last_served_slot: int | None = None
        self._last_data_timing: dict[str, float] = {
            "vllm_step_time_s": 0.0,
            "transfer_time_s": 0.0,
        }

        # Fixed CPU reservoir.  It is deliberately pageable: keeping hundreds of
        # MiB pinned per TP rank is expensive.  Only small per-batch staging is pinned.
        self._pool_by_hook: dict[str, torch.Tensor] = {
            h: torch.empty(
                (self._capacity_pages, self._page_tokens, self._d_model),
                dtype=self._dtype,
                device="cpu",
            )
            for h in self._selected_hooks
        }
        # Compatibility alias for existing memory instrumentation.
        self._mixing_pool = self._pool_by_hook
        self._page_valid = [0] * self._capacity_pages
        self._page_state = [0] * self._capacity_pages  # 0=EMPTY, 1=FILLING, 2=READY
        self._free_pages: deque[int] = deque(range(self._capacity_pages))

        pin = self._device.type == "cuda" and torch.cuda.is_available()
        self._slots: list[_BatchSlot] = []
        for _ in range(int(gpu_prefetch_batches)):
            pinned = {
                h: torch.empty(
                    (self._batch_size, self._d_model),
                    dtype=self._dtype,
                    device="cpu",
                    pin_memory=pin,
                )
                for h in self._selected_hooks
            }
            dev = {
                h: torch.empty(
                    (self._batch_size, self._d_model),
                    dtype=self._dtype,
                    device=self._device,
                )
                for h in self._selected_hooks
            }
            if self._device.type == "cuda":
                ready_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
            else:
                ready_event = None
                done_event = None
            self._slots.append(_BatchSlot(pinned, dev, ready_event, done_event))

        # Compatibility name used by SAETrainer's memory accounting.
        self._chunk_buffer = [slot.device_by_hook for slot in self._slots]
        self._free_gpu_slots: queue.Queue[int] = queue.Queue()
        self._ready_gpu_slots: queue.Queue[int | object] = queue.Queue()
        for i in range(len(self._slots)):
            self._free_gpu_slots.put(i)

        self._copy_stream = (
            torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        )
        self._ingress_thread: threading.Thread | None = None
        self._prefetch_thread: threading.Thread | None = None

        self._t0 = time.time()
        self._shm_logger = AsyncJsonlLogger(shm_log_path)
        self._monitor_logger = AsyncJsonlLogger(buffer_monitor_path)

        if self._replica_root_global is None:
            if dist.is_initialized():
                self._replica_root_global = dist.get_rank()
            else:
                self._replica_root_global = 0
        self._mailbox: _ReplicaMailbox | None = None
        self._mailbox_generation = 0
        if self._replica_size > 1:
            mailbox_path = Path("/dev/shm") / (
                f"{self._coord_name}_async_d{self._sae_dp_idx}.meta"
            )
            self._mailbox = _ReplicaMailbox(
                path=mailbox_path,
                max_chunks=self._prefetch_chunks,
                is_root=self._is_replica_root(),
                replica_size=self._replica_size,
            )

    def __iter__(self) -> "AsyncStreamingActivationProvider":
        self.start()
        return self

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._ingress_thread = threading.Thread(
            target=self._ingress_main,
            name=f"shm-ingress-d{self._sae_dp_idx}-pp{self._pp_rank}-tp{self._sae_tp_rank}",
            daemon=True,
        )
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_main,
            name=f"sae-prefetch-d{self._sae_dp_idx}-pp{self._pp_rank}-tp{self._sae_tp_rank}",
            daemon=True,
        )
        self._ingress_thread.start()
        self._prefetch_thread.start()
        self._log({
            "event": "async_provider_started",
            "mix_capacity_tokens": self._capacity_tokens,
            "page_tokens": self._page_tokens,
            "replace_pages_per_round": self._replace_pages_per_round,
            "gpu_prefetch_batches": len(self._slots),
            "selected_hooks": self._selected_hooks,
        })

    def __next__(self) -> torch.Tensor | dict[str, torch.Tensor]:
        self.start()
        self._retire_last_served_slot()
        t_wait = time.perf_counter()
        while True:
            self._raise_if_error()
            try:
                item = self._ready_gpu_slots.get(timeout=0.1)
                break
            except queue.Empty:
                if self._eof and self._ready_gpu_slots.empty():
                    self._raise_if_error()
                    raise StopIteration
        wait_s = time.perf_counter() - t_wait
        # Only exposed trainer wait belongs on the synchronous step critical path.
        # SHM copy/mixing/H2D are background work and are logged separately.
        self._last_data_timing = {
            "vllm_step_time_s": 0.0,
            "transfer_time_s": wait_s,
        }
        if item is _READY_EOF:
            self._raise_if_error()
            raise StopIteration
        slot_idx = int(item)
        slot = self._slots[slot_idx]
        if self._device.type == "cuda":
            assert slot.ready_event is not None
            torch.cuda.current_stream(self._device).wait_event(slot.ready_event)
        self._last_served_slot = slot_idx
        self._consume_step += 1
        self._consumed_tokens += slot.valid_rows
        self._log({
            "event": "consume",
            "step": self._consume_step,
            "valid_tokens": slot.valid_rows,
            "consumed_tokens": self._consumed_tokens,
            "trainer_data_wait_s": wait_s,
        })
        if self._is_multi_hook:
            return {h: slot.device_by_hook[h][: slot.valid_rows] for h in self._selected_hooks}
        return slot.device_by_hook[self._selected_hooks[0]][: slot.valid_rows]

    def next_batch(self) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.__next__()

    @property
    def consumed_tokens(self) -> int:
        return self._consumed_tokens

    def consume_last_data_timing(self) -> dict[str, float]:
        """Return only *exposed* data wait for trainer timing.

        Background SHM/mixing/H2D work is intentionally excluded from per-step
        ``transfer_time_s`` because it overlaps SAE compute.  Detailed background
        service timing remains available in ``shm_log_sae*.jsonl``.
        """
        timing = dict(self._last_data_timing)
        self._last_data_timing = {
            "vllm_step_time_s": 0.0,
            "transfer_time_s": 0.0,
        }
        return timing

    def request_drain_local_pool(self) -> None:
        # Stop acquiring new SHM chunks, but keep prefetch/trainer draining data
        # that is already resident in the local reservoir / GPU-ready queue.
        self._drain_requested.set()
        with self._cv:
            self._cv.notify_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._retire_last_served_slot()
        with self._cv:
            self._cv.notify_all()
        if self._ingress_thread is not None:
            self._ingress_thread.join(timeout=10.0)
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=10.0)
        self._shm_logger.close()
        self._monitor_logger.close()
        if self._mailbox is not None:
            self._mailbox.close()
            if self._is_replica_root():
                self._mailbox.unlink()

    # ------------------------------------------------------------------
    # Coordination / SHM ingress
    # ------------------------------------------------------------------

    def _is_replica_root(self) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == int(self._replica_root_global)

    def _combined_stop_check(self) -> bool:
        if self._stop.is_set() or self._drain_requested.is_set():
            return True
        return bool(self._stop_acquire_check is not None and self._stop_acquire_check())

    def _wait_for_chunk_space(self) -> int:
        """Return how many chunks fit in currently EMPTY reservoir pages."""
        while not self._stop.is_set():
            with self._cv:
                free_pages = len(self._free_pages)
                n = min(self._prefetch_chunks, free_pages // self._pages_per_chunk)
                if n > 0:
                    return n
                self._cv.wait(timeout=0.02)
        return 0

    def _root_acquire(self, n: int) -> tuple[list[int] | None, list[int], list[tuple[int, torch.Tensor, int]]]:
        t0 = time.perf_counter()
        try:
            if self._sae_dp_size > 1:
                acquire_dp = getattr(self._buffer, "acquire_dp_up_to", None)
                if acquire_dp is None:
                    raise RuntimeError(
                        "sae_dp_size>1 requires SharedActivationBuffer.acquire_dp_up_to; "
                        "apply the streaming SAE-DP buffer patch first"
                    )
                result = acquire_dp(
                    n,
                    dp_idx=self._sae_dp_idx,
                    dp_size=self._sae_dp_size,
                    random=self._random_chunks,
                    refcount=self._sae_pp_size,
                    stop_check=self._combined_stop_check,
                )
            else:
                result = self._buffer.acquire_up_to(
                    n,
                    random=self._random_chunks,
                    refcount=self._sae_pp_size,
                    stop_check=self._combined_stop_check,
                )
            # Older SharedActivationBuffer versions returned only the index list;
            # newer versions also return the CPU-side acquire wait.  Support both.
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], (int, float))
            ):
                indices, wait_s = result
            else:
                indices, wait_s = result, time.perf_counter() - t0
        except StopIteration:
            return None, [], []
        acquire_wall = time.perf_counter() - t0
        payloads: list[tuple[int, torch.Tensor, int]] = []
        valid_rows: list[int] = []
        for idx in indices:
            data, valid_total = self._buffer.read_chunk(idx)
            payloads.append((int(idx), data, int(valid_total)))
            valid_rows.append(int(valid_total))
        self._log({
            "event": "acquire_complete",
            "chunks": len(indices),
            "indices": [int(i) for i in indices],
            "wait_time_s": float(wait_s),
            "acquire_wall_s": acquire_wall,
        })
        return [int(i) for i in indices], valid_rows, payloads

    def _exchange_acquire_meta(
        self, indices: list[int] | None, valid_rows: list[int]
    ) -> tuple[int, list[int] | None, list[int]]:
        if self._mailbox is None:
            self._mailbox_generation += 1
            return self._mailbox_generation, indices, valid_rows
        if self._is_replica_root():
            generation = self._mailbox.publish(indices, valid_rows)
            self._mailbox_generation = generation
            return generation, indices, valid_rows
        generation, indices, valid_rows = self._mailbox.wait_next(
            self._mailbox_generation, stop=self._stop
        )
        self._mailbox_generation = generation
        return generation, indices, valid_rows

    def _ingress_main(self) -> None:
        try:
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            while not self._stop.is_set():
                n = self._wait_for_chunk_space()
                if n <= 0:
                    break

                root_payloads: dict[int, tuple[torch.Tensor, int]] = {}
                indices: list[int] | None = []
                valid_rows: list[int] = []
                if self._is_replica_root():
                    indices, valid_rows, payloads = self._root_acquire(n)
                    root_payloads = {idx: (data, valid) for idx, data, valid in payloads}
                generation, indices, valid_rows = self._exchange_acquire_meta(indices, valid_rows)
                if indices is None:
                    with self._cv:
                        self._eof = True
                        self._cv.notify_all()
                    break
                if not indices:
                    continue

                t_copy = time.perf_counter()
                for idx, valid_total in zip(indices, valid_rows):
                    if self._is_replica_root() and idx in root_payloads:
                        data, actual_valid = root_payloads[idx]
                        valid_total = actual_valid
                    else:
                        data, actual_valid = self._buffer.read_chunk(idx)
                        if int(actual_valid) != int(valid_total):
                            raise RuntimeError(
                                f"chunk {idx} valid-row mismatch: meta={valid_total} local={actual_valid}"
                            )
                    self._copy_chunk_into_free_pages(data, int(valid_total))

                # ACK is CPU-only. The replica root releases all PP refcount
                # shares after every PP*TP reader has copied its private slice.
                if self._mailbox is not None:
                    self._mailbox.ack(generation)
                    if self._is_replica_root():
                        self._mailbox.wait_all_acks(generation, stop=self._stop)
                if self._is_replica_root():
                    for idx in indices:
                        for _ in range(self._sae_pp_size):
                            self._buffer.release_chunk(idx)

                copy_s = time.perf_counter() - t_copy
                self._refill_count += 1
                with self._cv:
                    ready_rows = sum(self._page_valid[i] for i, s in enumerate(self._page_state) if s == 2)
                    self._cv.notify_all()
                self._log({
                    "event": "refill_complete",
                    "refill": self._refill_count,
                    "chunks": len(indices),
                    "new_tokens": sum(v // self._num_hooks for v in valid_rows),
                    "shm_copy_time_s": copy_s,
                    "ready_tokens": ready_rows,
                })
                if self._refill_count % 16 == 0:
                    try:
                        state = self._buffer.queue_counts()
                    except Exception:
                        state = {}
                    self._monitor_logger.log({
                        "elapsed_s": time.time() - self._t0,
                        "event": "buffer_sample",
                        "refill": self._refill_count,
                        "ready_tokens": ready_rows,
                        "buffer_state": state,
                    })
        except BaseException as exc:
            self._set_error(exc)
        finally:
            with self._cv:
                if self._combined_stop_check() and not self._eof:
                    # Quiesce: no more ingress.  Prefetch drains what is already local.
                    self._eof = True
                self._cv.notify_all()

    def _copy_chunk_into_free_pages(self, data: torch.Tensor, valid_total: int) -> None:
        if valid_total < 0 or valid_total > data.shape[0]:
            raise ValueError(f"invalid valid_total={valid_total} for data rows={data.shape[0]}")
        if valid_total % self._num_hooks != 0:
            raise ValueError(
                f"valid_total={valid_total} is not divisible by num_hooks={self._num_hooks}"
            )
        valid_per_hook = valid_total // self._num_hooks
        pages_needed = math.ceil(valid_per_hook / self._page_tokens)
        with self._cv:
            while len(self._free_pages) < pages_needed and not self._stop.is_set():
                self._cv.wait(timeout=0.02)
            if self._stop.is_set():
                return
            page_ids = [self._free_pages.popleft() for _ in range(pages_needed)]
            for p in page_ids:
                if self._page_state[p] != 0:
                    raise RuntimeError(f"page {p} is not EMPTY")
                self._page_state[p] = 1

        for hook in self._selected_hooks:
            global_hook_idx = self._hook_to_global[hook]
            start = global_hook_idx * valid_per_hook
            end = start + valid_per_hook
            src = data[start:end]
            cursor = 0
            for p in page_ids:
                n = min(self._page_tokens, valid_per_hook - cursor)
                if n <= 0:
                    break
                self._pool_by_hook[hook][p, :n].copy_(src[cursor : cursor + n])
                cursor += n

        with self._cv:
            cursor = 0
            for p in page_ids:
                n = min(self._page_tokens, valid_per_hook - cursor)
                self._page_valid[p] = n
                self._page_state[p] = 2
                cursor += n
            self._cv.notify_all()

    # ------------------------------------------------------------------
    # Reservoir -> pinned batch -> async H2D
    # ------------------------------------------------------------------

    def _ready_page_ids(self) -> list[int]:
        return [i for i, s in enumerate(self._page_state) if s == 2 and self._page_valid[i] > 0]

    def _wait_for_round_pages(self, *, initial: bool) -> list[int]:
        while True:
            self._raise_if_error()
            with self._cv:
                ready = self._ready_page_ids()
                if initial:
                    full = len(ready) == self._capacity_pages
                    enough_at_eof = self._eof and bool(ready)
                    if full or enough_at_eof:
                        return ready
                else:
                    # Before EOF, preserve rolling-buffer semantics: the consumed
                    # slots must be replenished before reshuffling the next round.
                    if self._eof:
                        if ready:
                            return ready
                        return []
                    if len(ready) == self._capacity_pages:
                        return ready
                self._cv.wait(timeout=0.02)

    def _prefetch_main(self) -> None:
        try:
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            initial = True
            while not self._stop.is_set() or not self._eof:
                ready_pages = self._wait_for_round_pages(initial=initial)
                initial = False
                if not ready_pages:
                    break
                if self._shuffle:
                    self._rng.shuffle(ready_pages)

                if self._eof:
                    round_pages = len(ready_pages)
                else:
                    round_pages = min(self._replace_pages_per_round, len(ready_pages))
                    round_pages = max(
                        self._pages_per_batch,
                        (round_pages // self._pages_per_batch) * self._pages_per_batch,
                    )
                order = ready_pages[:round_pages]
                cursor = 0
                while cursor < len(order):
                    self._raise_if_error()
                    pages: list[int] = []
                    rows = 0
                    while cursor < len(order) and rows < self._batch_size:
                        p = order[cursor]
                        cursor += 1
                        if self._page_state[p] != 2:
                            raise RuntimeError(f"page {p} lost READY state during round")
                        pages.append(p)
                        rows += self._page_valid[p]
                    if rows == 0:
                        continue
                    # Steady-state pages divide B exactly.  A short batch is allowed
                    # only while draining EOF/quiesce tail.
                    if rows < self._batch_size and not self._eof:
                        raise RuntimeError(
                            f"non-EOF short async batch: rows={rows}, batch={self._batch_size}"
                        )
                    self._stage_pages_to_gpu(pages, min(rows, self._batch_size))

                if self._eof:
                    # Any pages not in this round are consumed in the next drain
                    # iteration; once no READY pages remain we emit EOF.
                    with self._cv:
                        if not self._ready_page_ids():
                            break
            self._ready_gpu_slots.put(_READY_EOF)
        except BaseException as exc:
            self._set_error(exc)
            self._ready_gpu_slots.put(_READY_EOF)

    def _stage_pages_to_gpu(self, pages: list[int], valid_rows: int) -> None:
        while not self._stop.is_set() or not self._eof:
            try:
                slot_idx = self._free_gpu_slots.get(timeout=0.05)
                break
            except queue.Empty:
                self._raise_if_error()
        else:
            return
        slot = self._slots[slot_idx]

        # A reused pinned staging buffer must not be overwritten while its prior
        # non-blocking H2D is still reading it.  Synchronizing this event blocks
        # only the background prefetch thread, never the trainer/main thread.
        if self._device.type == "cuda" and slot.has_ready_event:
            assert slot.ready_event is not None
            slot.ready_event.synchronize()

        # Copy a handful of small contiguous pages into the fixed pinned batch
        # buffer.  This replaces torch.cat + full-reservoir tensor permutation.
        for hook in self._selected_hooks:
            dst = slot.pinned_by_hook[hook]
            out = 0
            for p in pages:
                n = min(self._page_valid[p], self._batch_size - out)
                if n <= 0:
                    break
                dst[out : out + n].copy_(self._pool_by_hook[hook][p, :n])
                out += n
            if out != valid_rows:
                raise RuntimeError(
                    f"staging row mismatch for {hook}: copied={out} expected={valid_rows}"
                )

        # The page payload is now safely copied into per-GPU-slot staging; SHM
        # ingress may immediately recycle these reservoir pages while SAE trains.
        with self._cv:
            for p in pages:
                self._page_valid[p] = 0
                self._page_state[p] = 0
                self._free_pages.append(p)
            self._cv.notify_all()

        t_h2d = time.perf_counter()
        if self._device.type == "cuda":
            assert self._copy_stream is not None
            assert slot.ready_event is not None
            assert slot.consume_done_event is not None
            with torch.cuda.stream(self._copy_stream):
                # If this GPU slot was used by a previous SAE step, order the new
                # H2D after that step without synchronizing the trainer CPU.
                if slot.has_consume_event:
                    self._copy_stream.wait_event(slot.consume_done_event)
                for hook in self._selected_hooks:
                    slot.device_by_hook[hook][:valid_rows].copy_(
                        slot.pinned_by_hook[hook][:valid_rows],
                        non_blocking=True,
                    )
                slot.ready_event.record(self._copy_stream)
                slot.has_ready_event = True
        else:
            for hook in self._selected_hooks:
                slot.device_by_hook[hook][:valid_rows].copy_(slot.pinned_by_hook[hook][:valid_rows])
        slot.valid_rows = valid_rows
        self._ready_gpu_slots.put(slot_idx)
        self._log({
            "event": "batch_prefetched",
            "valid_tokens": valid_rows,
            "h2d_submit_wall_s": time.perf_counter() - t_h2d,
            "pages": len(pages),
        })

    def _retire_last_served_slot(self) -> None:
        slot_idx = self._last_served_slot
        if slot_idx is None:
            return
        slot = self._slots[slot_idx]
        if self._device.type == "cuda":
            assert slot.consume_done_event is not None
            slot.consume_done_event.record(torch.cuda.current_stream(self._device))
            slot.has_consume_event = True
        self._free_gpu_slots.put(slot_idx)
        self._last_served_slot = None

    # ------------------------------------------------------------------
    # Error/log helpers
    # ------------------------------------------------------------------

    def _set_error(self, exc: BaseException) -> None:
        self._error = exc
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._log({"event": "async_provider_error", "error": repr(exc)})

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("async streaming activation provider failed") from self._error

    def _log(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("elapsed_s", time.time() - self._t0)
        record.setdefault("dp_idx", self._sae_dp_idx)
        record.setdefault("pp_rank", self._pp_rank)
        record.setdefault("tp_rank", self._sae_tp_rank)
        self._shm_logger.log(record)
