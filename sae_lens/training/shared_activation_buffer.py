"""
Shared-memory activation buffer for streaming_mode v1.

Layout (5 files in base_dir, default /dev/shm):
  {name}_state.bin   — int8 memmap  (num_chunks,)
  {name}_meta.bin    — int32 memmap (num_chunks, 4)
                         [0] valid_tokens  [1] producer_id  [2] seq_no  [3] reserved
  {name}_header.bin  — int32 memmap (8,)
                         [0] num_producers
                         [1] done_count       (each producer increments on finish)
                         [2] target_chunks    (global chunk budget, set at create)
                         [3] next_claim_seq   (monotonically incremented under lock)
                         [4] dtype_code       (0=bfloat16, 1=float32)
                         [5-7] reserved
  {name}_data.bin    — dtype-dependent memmap (num_chunks, chunk_size_tokens, d_model)
                         bfloat16: uint16 memmap, bfloat16 stored as raw uint16 bit-pattern
                         float32:  float32 memmap
  {name}.lock        — empty file used for fcntl.flock

v1 dtype: bfloat16 (dtype_code=0) or float32 (dtype_code=1).
"""

import fcntl
import random as _rng
import shutil
import time
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import torch


class ChunkState(IntEnum):
    FREE = 0
    WRITING = 1
    READY = 2
    CONSUMING = 3


class SharedActivationBuffer:
    """Shared-memory pool for passing activation chunks from vLLM producers to SAE consumers.

    Producer side:
      - allocate_write_chunk()  → (chunk_idx, seq_no) or None when quota done
      - write_chunk()           → write activations into slot (cast to buffer dtype)
      - mark_ready()            → mark slot as consumable
      - abort_write_chunk()     → return slot to FREE on dataset EOF
      - signal_done()           → signal this producer has finished

    Consumer side:
      - acquire_up_to(n)        → claim up to n READY slots (returns 1..n); StopIteration on EOF
      - read_chunk(i)           → read activations back in buffer dtype
      - release_chunk(i)        → return slot to FREE

    Locking: fcntl.flock exclusive lock for all state-machine transitions.
    Data reads (read_chunk) do NOT need the lock — slots are exclusively CONSUMING at that point.
    """

    def __init__(
        self,
        name: str,
        num_chunks: int,
        chunk_size_tokens: int,
        d_model: int,
        num_producers: int,
        target_chunks: int = 0,
        create: bool = False,
        base_dir: str = "/dev/shm",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._name = name
        self._num_chunks = num_chunks
        self._chunk_size_tokens = chunk_size_tokens
        self._d_model = d_model
        self._base = Path(base_dir)

        state_path = self._base / f"{name}_state.bin"
        meta_path = self._base / f"{name}_meta.bin"
        header_path = self._base / f"{name}_header.bin"
        data_path = self._base / f"{name}_data.bin"
        lock_path = self._base / f"{name}.lock"

        _DTYPE_TO_CODE = {torch.bfloat16: 0, torch.float32: 1}
        _CODE_TO_NP = {0: np.uint16, 1: np.float32}
        _CODE_TO_TORCH = {0: torch.bfloat16, 1: torch.float32}

        if create:
            if dtype not in _DTYPE_TO_CODE:
                raise ValueError(f"SharedActivationBuffer only supports bfloat16 or float32, got {dtype}")
            dtype_code = _DTYPE_TO_CODE[dtype]
            np_dtype = _CODE_TO_NP[dtype_code]
            elem_bytes = 2 if dtype_code == 0 else 4

            if state_path.exists():
                import logging
                logging.getLogger("saelens.streaming").warning(
                    "SharedActivationBuffer: stale files found for %s, overwriting.", name
                )
            # Check disk space
            data_bytes = num_chunks * chunk_size_tokens * d_model * elem_bytes
            free = shutil.disk_usage(str(self._base)).free
            if free < data_bytes + 1024 * 1024:  # 1 MB headroom
                raise RuntimeError(
                    f"Insufficient space in {base_dir}: need {data_bytes} bytes, "
                    f"have {free} bytes free."
                )
            # Create and zero-initialise all files
            self._create_file(state_path, num_chunks, dtype=np.int8)
            self._create_file(meta_path, num_chunks * 4, dtype=np.int32)
            hdr = self._create_file(header_path, 8, dtype=np.int32)
            hdr[0] = num_producers
            hdr[1] = 0
            hdr[2] = target_chunks
            hdr[3] = 0
            hdr[4] = dtype_code
            hdr.flush()
            self._create_file(data_path, num_chunks * chunk_size_tokens * d_model, dtype=np_dtype)
            lock_path.touch(exist_ok=True)

        # Open memmaps — read dtype_code from header to support attach (create=False)
        mode = "r+"
        self._state: np.ndarray = np.memmap(
            str(state_path), dtype=np.int8, mode=mode, shape=(num_chunks,)
        )
        self._meta: np.ndarray = np.memmap(
            str(meta_path), dtype=np.int32, mode=mode, shape=(num_chunks, 4)
        )
        self._header: np.ndarray = np.memmap(
            str(header_path), dtype=np.int32, mode=mode, shape=(8,)
        )
        dtype_code = int(self._header[4])
        np_dtype = _CODE_TO_NP.get(dtype_code, np.uint16)
        self._dtype: torch.dtype = _CODE_TO_TORCH.get(dtype_code, torch.bfloat16)
        self._dtype_code: int = dtype_code
        self._data: np.ndarray = np.memmap(
            str(data_path),
            dtype=np_dtype,
            mode=mode,
            shape=(num_chunks, chunk_size_tokens, d_model),
        )
        self._lock_fd = open(str(lock_path), "rb")
        self._backoff_count = 0

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def allocate_write_chunk(
        self,
        stop_check: "Callable[[], bool] | None" = None,
    ) -> tuple[int, int] | None:
        """Claim a FREE slot and increment the global sequence counter.

        Returns (chunk_idx, seq_no) if quota remains, or None if
        next_claim_seq >= target_chunks (global budget exhausted) or
        stop_check() returns True.

        Blocks with backoff until a FREE slot is available.
        """
        while True:
            with self._locked():
                if int(self._header[3]) >= int(self._header[2]):
                    return None  # global quota exhausted
                # Find a FREE slot
                free_candidates = np.where(self._state == ChunkState.FREE)[0]
                if len(free_candidates) > 0:
                    free_idx = int(free_candidates[0])
                    seq = int(self._header[3])
                    self._header[3] += 1
                    self._state[free_idx] = ChunkState.WRITING
                    self._meta[free_idx, 2] = seq
                    self._meta.flush()
                    self._header.flush()
                    self._state.flush()
                    self._backoff_count = 0
                    return free_idx, seq
            if stop_check is not None and stop_check():
                return None
            time.sleep(self._backoff())

    def abort_write_chunk(self, chunk_idx: int) -> None:
        """Return a WRITING slot to FREE (producer EOF before data was written)."""
        with self._locked():
            assert int(self._state[chunk_idx]) == ChunkState.WRITING, (
                f"abort_write_chunk called on slot {chunk_idx} in state "
                f"{ChunkState(self._state[chunk_idx]).name}"
            )
            self._state[chunk_idx] = ChunkState.FREE
            self._state.flush()

    def write_chunk(
        self,
        chunk_idx: int,
        activations: torch.Tensor,
        valid_tokens: int,
        producer_id: int = 0,
    ) -> None:
        """Write activations (CPU tensor) into the data memmap.

        activations will be cast to the buffer's dtype (bfloat16 or float32).
        valid_tokens rows are written; remaining rows are zero-filled.
        """
        assert int(self._state[chunk_idx]) == ChunkState.WRITING
        if activations.dtype != self._dtype:
            activations = activations.to(self._dtype)
        if activations.device.type != "cpu":
            activations = activations.cpu()

        rows = min(int(valid_tokens), self._chunk_size_tokens)
        if self._dtype_code == 0:
            # bfloat16: view as int16 → uint16 for memmap storage
            arr = activations[:rows].contiguous().view(torch.int16).numpy().view(np.uint16)
        else:
            # float32: store directly
            arr = activations[:rows].contiguous().numpy()
        self._data[chunk_idx, :rows, :] = arr
        if rows < self._chunk_size_tokens:
            self._data[chunk_idx, rows:, :] = 0
        self._meta[chunk_idx, 0] = rows
        self._meta[chunk_idx, 1] = producer_id
        self._data.flush()
        self._meta.flush()

    def mark_ready(self, chunk_idx: int) -> None:
        """Transition WRITING → READY under lock."""
        with self._locked():
            assert int(self._state[chunk_idx]) == ChunkState.WRITING
            self._state[chunk_idx] = ChunkState.READY
            self._state.flush()

    def signal_done(self) -> None:
        """Increment done_count to indicate this producer has finished."""
        with self._locked():
            self._header[1] += 1
            self._header.flush()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def acquire_up_to(self, n: int, random: bool = True) -> tuple[list[int], float]:
        """Claim up to n READY slots as CONSUMING.

        Returns (indices, wait_s) where wait_s is the total time spent sleeping
        waiting for READY chunks to appear (0.0 if chunks were available immediately).

        Blocks only when ready == 0 and not all producers done.
        Raises StopIteration when ready == 0 and all producers done.

        Args:
            n: Maximum number of slots to claim.
            random: If True, shuffle the ready list before claiming (default True).
        """
        total_sleep: float = 0.0
        while True:
            with self._locked():
                ready = [
                    int(i)
                    for i in range(self._num_chunks)
                    if int(self._state[i]) == ChunkState.READY
                ]
                if ready:
                    if random:
                        _rng.shuffle(ready)
                    claim = ready[:n]
                    for i in claim:
                        self._state[i] = ChunkState.CONSUMING
                    self._state.flush()
                    self._backoff_count = 0
                    return claim, total_sleep
                if int(self._header[1]) >= int(self._header[0]):  # done_count >= num_producers
                    raise StopIteration
            sleep_s = self._backoff()
            total_sleep += sleep_s
            time.sleep(sleep_s)

    def read_chunk(self, chunk_idx: int) -> tuple[torch.Tensor, int]:
        """Read a CONSUMING slot. Returns (activations, valid_tokens).

        activations has shape (valid_tokens, d_model) with the buffer's dtype.
        No lock needed — the slot is exclusively CONSUMING for this caller.
        """
        valid_tokens = int(self._meta[chunk_idx, 0])
        raw = np.array(self._data[chunk_idx, :valid_tokens, :])  # (valid, d_model)
        if self._dtype_code == 0:
            # bfloat16: reinterpret uint16 → int16 → bfloat16
            tensor = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
        else:
            # float32: direct conversion
            tensor = torch.from_numpy(raw)
        return tensor, valid_tokens

    def release_chunk(self, chunk_idx: int) -> None:
        """Transition CONSUMING → FREE under lock."""
        with self._locked():
            assert int(self._state[chunk_idx]) == ChunkState.CONSUMING
            self._state[chunk_idx] = ChunkState.FREE
            self._state.flush()

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def queue_counts(self) -> dict[str, int]:
        """Snapshot of slot counts by state (no lock — approximate)."""
        counts = {s.name.lower(): 0 for s in ChunkState}
        for i in range(self._num_chunks):
            counts[ChunkState(int(self._state[i])).name.lower()] += 1
        return counts

    def snapshot(self) -> dict:
        """Snapshot of slot states: counts by state and list of READY indices (no lock — approximate)."""
        counts = {s.name.lower(): 0 for s in ChunkState}
        ready_indices: list[int] = []
        for i in range(self._num_chunks):
            state = ChunkState(int(self._state[i]))
            counts[state.name.lower()] += 1
            if state == ChunkState.READY:
                ready_indices.append(i)
        return {"counts": counts, "ready_indices": ready_indices}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush all memmaps and close the lock file descriptor."""
        for arr in (self._state, self._meta, self._header, self._data):
            arr.flush()
        self._lock_fd.close()

    def reset_for_restart(self, new_num_producers: int) -> None:
        """Reset buffer state for a new producer group after quiesce.

        Resets any WRITING chunks to FREE (abandoned mid-write), updates
        num_producers to the new vllm_dp value, and zeroes done_count so the
        new producer group can signal completion independently.

        Called by the supervisor before relaunching workers. No workers should
        be alive when this is called.
        """
        with self._locked():
            for i in range(self._num_chunks):
                if int(self._state[i]) == ChunkState.WRITING:
                    self._state[i] = ChunkState.FREE
            self._state.flush()
            self._header[0] = new_num_producers
            self._header[1] = 0
            self._header.flush()

    # NOTE: No destroy() method in v1. Producers only call close().
    # Stale /dev/shm files are detected on the next create (logged + overwritten).

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _locked(self) -> Generator[None, None, None]:
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _backoff(self) -> float:
        """Exponential backoff: starts at 1ms, caps at 10ms."""
        self._backoff_count = min(self._backoff_count + 1, 10)
        return self._backoff_count * 0.001

    @staticmethod
    def _create_file(path: Path, n_elements: int, dtype: type) -> np.ndarray:
        """Create and zero-initialise a flat memmap file."""
        arr: np.ndarray = np.memmap(str(path), dtype=dtype, mode="w+", shape=(n_elements,))
        arr[:] = 0
        arr.flush()
        return arr
