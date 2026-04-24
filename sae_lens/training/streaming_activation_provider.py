"""
Consumer-side DataProvider for streaming_mode v1.

StreamingActivationProvider implements Iterator[torch.Tensor] (the DataProvider
protocol expected by SAETrainer). It:
  1. Acquires activation chunks from SharedActivationBuffer (any READY chunks
     immediately, up to prefetch_chunks at a time).
  2. Carries over leftover rows from the previous pool to avoid dropping data.
  3. Shuffles each newly acquired batch before merging.
  4. Yields batches of exactly train_batch_size_tokens (final batch may be smaller).
  5. For sae_tp > 1, TP root broadcasts new data to followers via NCCL (CUDA tensor).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist

from sae_lens.training.shared_activation_buffer import SharedActivationBuffer


class StreamingActivationProvider:
    """DataProvider that reads from a SharedActivationBuffer and feeds SAETrainer.

    Args:
        buffer: SharedActivationBuffer instance (consumer-side, already attached).
        train_batch_size_tokens: tokens per yielded batch.
        prefetch_chunks: max chunks to acquire per refill call.
        device: target device for tensors fed to SAETrainer.
        sae_tp_group: NCCL process group for SAE TP, or None if sae_tp == 1.
        sae_tp_rank: rank within sae_tp_group (0 = root that reads from buffer).
        sae_tp_root_global_rank: world rank of the TP root (needed as dist.broadcast src).
        d_model: activation feature dimension (for pre-allocating follower buffers).
    """

    def __init__(
        self,
        buffer: SharedActivationBuffer,
        train_batch_size_tokens: int,
        prefetch_chunks: int,
        device: torch.device,
        sae_tp_group: dist.ProcessGroup | None,
        sae_tp_rank: int,
        sae_tp_root_global_rank: int,
        d_model: int,
        dtype: torch.dtype = torch.float32,
        shm_log_path: Path | None = None,
        shuffle: bool = True,
        random_chunks: bool = True,
        buffer_monitor_path: Path | None = None,
        hook_names: list[str] | None = None,
    ) -> None:
        self._buffer = buffer
        self._batch_size = train_batch_size_tokens
        self._prefetch_chunks = prefetch_chunks
        self._device = device
        self._dtype = dtype
        self._sae_tp_group = sae_tp_group
        self._sae_tp_rank = sae_tp_rank
        self._tp_root_global = sae_tp_root_global_rank
        self._d_model = d_model

        self._shuffle = shuffle
        self._random_chunks = random_chunks

        self._hook_names = hook_names
        self._num_hooks = len(hook_names) if hook_names else 1
        self._is_multi_hook = self._num_hooks > 1
        self._tokens_per_hook = train_batch_size_tokens

        self._pool: torch.Tensor | None = None
        self._pool_start: int = 0
        self._pool_len: int = 0
        self._consumed_tokens: int = 0
        self._consume_step: int = 0

        # Per-refill timing, consumed by SAETrainer via consume_last_data_timing()
        self._last_wait_time_s: float = 0.0
        self._last_transfer_time_s: float = 0.0

        self._t_ready: float = time.time()

        # Shared memory management log (TP root only)
        self._shm_log_path = shm_log_path
        if shm_log_path is not None and sae_tp_rank == 0:
            shm_log_path.parent.mkdir(parents=True, exist_ok=True)
            shm_log_path.write_text("")

        # Buffer monitor log: pre-grab snapshot at every refill (TP root only)
        self._buffer_monitor_path = buffer_monitor_path
        self._buffer_monitor_step: int = 0
        if buffer_monitor_path is not None and sae_tp_rank == 0:
            buffer_monitor_path.parent.mkdir(parents=True, exist_ok=True)
            buffer_monitor_path.write_text("")

    def __iter__(self) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> torch.Tensor | dict[str, torch.Tensor]:
        take_size = self._batch_size * self._num_hooks
        # Serve from pool if a full batch is available
        if self._pool is not None and self._pool_start + take_size <= self._pool_len:
            return self._take()
        # Otherwise refill; on StopIteration, serve any remaining partial batch first
        try:
            self._refill()
        except StopIteration:
            if self._pool is not None and self._pool_start < self._pool_len:
                return self._take()
            raise
        if self._pool_len - self._pool_start == 0:
            raise StopIteration
        return self._take()

    @property
    def consumed_tokens(self) -> int:
        return self._consumed_tokens

    def consume_last_data_timing(self) -> dict[str, float]:
        """Return and clear per-refill timing for SAETrainer integration."""
        result = {
            "vllm_step_time_s": self._last_wait_time_s,
            "transfer_time_s": self._last_transfer_time_s,
        }
        self._last_wait_time_s = 0.0
        self._last_transfer_time_s = 0.0
        return result

    def _shm_log(self, record: dict) -> None:
        """Append a JSON record to the shm log (TP root only, no-op if disabled)."""
        if self._shm_log_path is None or self._sae_tp_rank != 0:
            return
        record["elapsed_s"] = time.time() - self._t_ready
        with open(self._shm_log_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def _write_buffer_monitor(self) -> None:
        """Write a pre-grab buffer snapshot to buffer_monitor.jsonl (TP root only)."""
        if self._buffer_monitor_path is None or self._sae_tp_rank != 0:
            return
        snap = self._buffer.snapshot()
        self._buffer_monitor_step += 1
        record = {
            "refill_step": self._buffer_monitor_step,
            "elapsed_s": time.time() - self._t_ready,
            "ready_count": snap["counts"]["ready"],
            "ready_indices": snap["ready_indices"],
            "counts": snap["counts"],
        }
        with open(self._buffer_monitor_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Acquire chunks from the buffer, merge with leftover, shuffle.

        TP root reads from the buffer and broadcasts to followers via NCCL CUDA tensors.
        Followers receive via broadcast and raise StopIteration on meta[0]==0.
        """
        if self._sae_tp_rank == 0:
            leftover = self._pool_len - self._pool_start if self._pool is not None else 0
            self._shm_log({
                "event": "refill_start",
                "pool_tokens_remaining": leftover,
                "consumed_tokens": self._consumed_tokens,
                "prefetch_chunks": self._prefetch_chunks,
            })

            self._write_buffer_monitor()
            t0 = time.perf_counter()
            try:
                indices, vllm_wait_s = self._buffer.acquire_up_to(
                    self._prefetch_chunks, random=self._random_chunks
                )
            except StopIteration:
                self._shm_log({"event": "refill_exhausted", "wait_time_s": time.perf_counter() - t0})
                # Signal end-of-stream to TP followers
                if self._sae_tp_group is not None:
                    meta = torch.zeros(1, dtype=torch.int64, device=self._device)
                    dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
                raise
            t_after_acquire = time.perf_counter()

            self._shm_log({
                "event": "refill_acquired",
                "chunk_indices": indices,
                "n_chunks": len(indices),
                "wait_time_s": vllm_wait_s,
                "buffer_state": self._buffer.queue_counts(),
            })

            # Collect activations from all acquired chunks
            acts_list: list[torch.Tensor] = []
            for i in indices:
                tensor, valid = self._buffer.read_chunk(i)
                acts_list.append(tensor[:valid])
                self._buffer.release_chunk(i)
            new_data = torch.cat(acts_list, dim=0)  # (total_new_tokens, d_model)

            if self._sae_tp_group is not None:
                # Broadcast shape then data to followers (CUDA tensors for NCCL)
                meta = torch.tensor(
                    [new_data.shape[0]], dtype=torch.int64, device=self._device
                )
                dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
                new_data = new_data.to(device=self._device, dtype=self._dtype)
                dist.broadcast(new_data.contiguous(), src=self._tp_root_global, group=self._sae_tp_group)
            else:
                new_data = new_data.to(device=self._device, dtype=self._dtype)

            self._last_wait_time_s = vllm_wait_s
            self._last_transfer_time_s = time.perf_counter() - t_after_acquire

            self._shm_log({
                "event": "refill_complete",
                "new_tokens": new_data.shape[0],
                "pool_tokens_after": leftover + new_data.shape[0],
                "wait_time_s": self._last_wait_time_s,
                "transfer_time_s": self._last_transfer_time_s,
            })

        else:
            # TP follower: receive from root
            t0 = time.perf_counter()
            meta = torch.zeros(1, dtype=torch.int64, device=self._device)
            dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
            num_rows = int(meta[0])
            if num_rows == 0:
                raise StopIteration
            new_data = torch.empty(
                num_rows, self._d_model, dtype=self._dtype, device=self._device
            )
            dist.broadcast(new_data, src=self._tp_root_global, group=self._sae_tp_group)
            self._last_wait_time_s = 0.0
            self._last_transfer_time_s = time.perf_counter() - t0

        # Shuffle new data
        if self._is_multi_hook:
            new_data = self._reinterleave_hooks(new_data)
        elif self._shuffle:
            perm = torch.randperm(new_data.shape[0], device=self._device)
            new_data = new_data[perm]

        # Merge leftover from previous pool (carry-over avoids token loss)
        if self._pool is not None and self._pool_start < self._pool_len:
            leftover = self._pool[self._pool_start:]
            self._pool = torch.cat([leftover, new_data], dim=0)
        else:
            self._pool = new_data
        self._pool_start = 0
        self._pool_len = self._pool.shape[0]

    def _take(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return next batch from the pool, updating pool_start and consumed_tokens.

        For multi-hook, the pool stores interleaved hooks:
        [hook0_batch, hook1_batch, hook0_batch, hook1_batch, ...].
        We take batch_size * num_hooks rows and split into a dict.
        """
        take_size = self._batch_size * self._num_hooks
        end = min(self._pool_start + take_size, self._pool_len)
        batch = self._pool[self._pool_start:end]
        self._pool_start = end
        self._consumed_tokens += batch.shape[0] // self._num_hooks
        self._consume_step += 1
        self._shm_log({
            "event": "consume",
            "step": self._consume_step,
            "batch_tokens": batch.shape[0] // self._num_hooks,
            "pool_tokens_remaining": self._pool_len - self._pool_start,
            "cumulative_tokens": self._consumed_tokens,
        })
        if self._is_multi_hook:
            assert self._hook_names is not None
            tph = batch.shape[0] // self._num_hooks
            return {
                name: batch[i * tph : (i + 1) * tph]
                for i, name in enumerate(self._hook_names)
            }
        return batch

    def _reinterleave_hooks(self, data: torch.Tensor) -> torch.Tensor:
        """Re-interleave multi-hook chunk data into batch-sized blocks.

        Input layout (from buffer chunks):
          [chunk0_hook0, chunk0_hook1, chunk1_hook0, chunk1_hook1, ...]
          where each block is variable-sized but total rows per hook are equal.

        Output layout (for _take()):
          [hook0_batch, hook1_batch, hook0_batch, hook1_batch, ...]
          where each block is batch_size rows.

        Each hook's data is shuffled independently if shuffle is enabled.
        """
        nh = self._num_hooks
        total = data.shape[0]
        tokens_per_hook = total // nh

        # Split concatenated chunks into per-hook streams.
        # Chunks are laid out as [h0, h1, h0, h1, ...] with each h_i block
        # being chunk_size_tokens rows.  Reshape to (num_chunks, nh, chunk_tokens, d)
        # then transpose to (nh, num_chunks * chunk_tokens, d).
        per_hook: list[torch.Tensor] = []
        # Robust path: iterate chunk-sized blocks and gather per-hook slices.
        chunk_rows = self._buffer._chunk_size_tokens if hasattr(self._buffer, '_chunk_size_tokens') else total
        # Each chunk in the buffer has nh * per_hook_chunk rows.
        per_hook_chunk = chunk_rows // nh
        hook_parts: list[list[torch.Tensor]] = [[] for _ in range(nh)]
        pos = 0
        while pos < total:
            for h in range(nh):
                end = min(pos + per_hook_chunk, total)
                hook_parts[h].append(data[pos:end])
                pos = end
        per_hook = [torch.cat(parts, dim=0) for parts in hook_parts]

        if self._shuffle:
            for h in range(nh):
                perm = torch.randperm(per_hook[h].shape[0], device=self._device)
                per_hook[h] = per_hook[h][perm]

        # Interleave in batch_size blocks: [h0_batch, h1_batch, h0_batch, ...]
        bs = self._batch_size
        result_parts: list[torch.Tensor] = []
        n_batches = tokens_per_hook // bs
        for b in range(n_batches):
            for h in range(nh):
                result_parts.append(per_hook[h][b * bs : (b + 1) * bs])
        # Remainder (partial batch)
        rem = tokens_per_hook - n_batches * bs
        if rem > 0:
            for h in range(nh):
                result_parts.append(per_hook[h][n_batches * bs : n_batches * bs + rem])
        return torch.cat(result_parts, dim=0)
