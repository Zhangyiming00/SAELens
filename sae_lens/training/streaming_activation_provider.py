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
  6. For sae_pp_size > 1, the DP-replica root (PP-0 + TP-0) acquires chunks and
     broadcasts the chunk-index list (NOT the data) to its sibling PP stages via
     ``dp_replica_group``. Sibling PP stages then read the same chunks directly
     from the shared mmap and slice out only the rows for their own hooks. Every
     PP stage's TP-root calls ``release_chunk`` once; the buffer's refcount
     ensures the chunk only returns to FREE after all siblings have released.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

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
        hook_names: full list of hook names carried in each buffer chunk (for parsing).
            All providers in one DP replica MUST agree on this list.
        select_hook_names: optional subset of ``hook_names`` to return from ``_take``.
            Used by PP stages that only train on a subset of hooks. ``None`` returns
            all hooks (single-PP behaviour).
        dp_replica_group: NCCL process group spanning all PP*TP ranks of this DP
            replica. Used by PP-0 + TP-0 to broadcast claimed chunk indices to
            sibling PP stages. ``None`` falls back to single-PP behaviour.
        dp_replica_root_global_rank: world rank of PP-0 + TP-0 in this DP replica.
        sae_pp_size: number of PP stages within this DP replica. Used as the buffer
            refcount on acquire. Defaults to 1 (single-PP).
        pp_rank: this rank's PP stage index. 0 == DP root.
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
        mix_chunks: int = 0,
        mix_fraction: float = 0.5,
        mixing_seed: int | None = None,
        mixing_shard_index: int = 0,
        stop_acquire_check: Callable[[], bool] | None = None,
        buffer_monitor_path: Path | None = None,
        hook_names: list[str] | None = None,
        select_hook_names: list[str] | None = None,
        dp_replica_group: dist.ProcessGroup | None = None,
        dp_replica_root_global_rank: int | None = None,
        sae_pp_size: int = 1,
        pp_rank: int = 0,
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
        if mix_chunks < 0:
            raise ValueError("mix_chunks must be >= 0")
        if not 0 <= mix_fraction <= 1:
            raise ValueError("mix_fraction must be in [0, 1]")
        self._mix_chunks = mix_chunks
        self._mix_fraction = mix_fraction
        self._mixing_seed = mixing_seed
        self._mixing_shard_index = mixing_shard_index
        self._mixing_generators: dict[str, torch.Generator] = {}
        self._stop_acquire_check = stop_acquire_check
        self._drain_local_pool = False
        self._mixing_pool: torch.Tensor | None = None

        self._hook_names = hook_names
        self._num_hooks = len(hook_names) if hook_names else 1
        self._is_multi_hook = self._num_hooks >= 1
        self._tokens_per_hook = train_batch_size_tokens

        # PP-stage hook subset (None == train all hooks the buffer carries)
        if select_hook_names is not None and hook_names is not None:
            for name in select_hook_names:
                if name not in hook_names:
                    raise ValueError(
                        f"select_hook_names contains {name!r} not present in "
                        f"hook_names={hook_names}"
                    )
        self._select_hook_names = select_hook_names

        # Per-DP-replica coordination for sae_pp_size > 1
        self._dp_replica_group = dp_replica_group
        self._dp_replica_root_global = dp_replica_root_global_rank
        self._sae_pp_size = sae_pp_size
        self._pp_rank = pp_rank
        # PP-0 + TP-0 is the DP replica root that calls acquire_up_to.
        self._is_dp_root = pp_rank == 0 and sae_tp_rank == 0
        if sae_pp_size > 1 and dp_replica_group is None:
            raise ValueError(
                "sae_pp_size>1 requires dp_replica_group to coordinate chunk "
                "claiming across PP stages"
            )

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

        self._debug_provider_path: Path | None = None
        self._debug_provider_max_steps = int(
            os.environ.get("SAELENS_DEBUG_PROVIDER_MAX_STEPS", "20")
        )
        if (
            os.environ.get("SAELENS_DEBUG_STREAMING_PROVIDER") == "1"
            and buffer_monitor_path is not None
        ):
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self._debug_provider_path = (
                buffer_monitor_path.parent
                / f"debug_provider_rank{rank}_pid{os.getpid()}.jsonl"
            )
            self._debug_provider_path.write_text("")

    def __iter__(self) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> torch.Tensor | dict[str, torch.Tensor]:
        take_size = self._batch_size * self._num_hooks
        # Serve from pool if a full batch is available
        if self._pool is not None and self._pool_start + take_size <= self._pool_len:
            return self._take()
        self._activate_drain_if_requested()
        if self._drain_local_pool:
            self._make_all_local_pool_servable()
            if self._pool is not None and self._pool_start < self._pool_len:
                return self._take()
            raise StopIteration
        # Otherwise refill until local mixing has enough rows to serve or the
        # upstream stream ends. With mix_chunks>0 this mirrors mixing_buffer:
        # rows stay in storage until the configured buffer capacity is reached.
        while True:
            try:
                self._refill()
            except StopIteration:
                if self._pool is not None and self._pool_start < self._pool_len:
                    return self._take()
                raise
            if self._pool is not None and self._pool_start < self._pool_len:
                break
        return self._take()

    @property
    def consumed_tokens(self) -> int:
        return self._consumed_tokens

    def request_drain_local_pool(self) -> None:
        """Stop acquiring new chunks and yield only already-prefetched data."""
        self._drain_local_pool = True
        self._make_all_local_pool_servable()

    def _activate_drain_if_requested(self) -> None:
        if self._drain_local_pool:
            return
        if self._external_stop_requested():
            self.request_drain_local_pool()

    def _external_stop_requested(self) -> bool:
        return (
            self._stop_acquire_check is not None
            and bool(self._stop_acquire_check())
        )

    def _should_stop_acquiring(self) -> bool:
        return self._drain_local_pool or self._external_stop_requested()

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

        Three roles:
        - DP-replica root (PP-0 + TP-0): acquires from the buffer, broadcasts
          chunk indices to PP siblings via ``dp_replica_group``, then reads its
          own copy of those chunks and broadcasts the resulting tensor to its
          TP followers.
        - PP-sibling TP root (PP-rank>0, TP-rank=0): receives chunk indices from
          the DP root, reads the same chunks directly from /dev/shm, slices its
          own hook subset, then broadcasts to its TP followers. Calls
          ``release_chunk`` once per chunk (refcount decrement).
        - TP follower (TP-rank>0): receives the prepared tensor via NCCL TP
          broadcast and parses it. No buffer interaction.
        """
        if self._is_dp_root:
            self._refill_dp_root()
        elif self._sae_tp_rank == 0:
            # PP-sibling TP-root: read same chunks from shm based on broadcast indices.
            self._refill_pp_sibling_tp_root()
        else:
            self._refill_tp_follower()

    def _refill_dp_root(self) -> None:
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
                self._prefetch_chunks,
                random=self._random_chunks,
                refcount=self._sae_pp_size,
                stop_check=self._should_stop_acquiring,
            )
        except StopIteration:
            self._make_all_local_pool_servable()
            self._shm_log({"event": "refill_exhausted", "wait_time_s": time.perf_counter() - t0})
            # Signal end-of-stream to PP siblings (if any) and TP followers.
            self._broadcast_indices_to_pp_siblings([], end_of_stream=True)
            if self._sae_tp_group is not None:
                meta = torch.zeros(1, dtype=torch.int64, device=self._device)
                dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
            raise
        t_after_acquire = time.perf_counter()

        # Read each chunk once to discover valid_rows; collect and release.
        # PP siblings (if any) receive (indices, valid_rows) and read independently
        # from their own mmap, so we share valid_rows over the network.
        acts_list: list[torch.Tensor] = []
        valid_rows_by_chunk: list[int] = []
        for i in indices:
            tensor, valid = self._buffer.read_chunk(i)
            valid_rows_by_chunk.append(int(valid))
            acts_list.append(tensor[:valid])
            self._buffer.release_chunk(i)

        self._broadcast_indices_to_pp_siblings(
            indices, end_of_stream=False, valid_rows=valid_rows_by_chunk
        )

        self._shm_log({
            "event": "refill_acquired",
            "chunk_indices": indices,
            "n_chunks": len(indices),
            "wait_time_s": vllm_wait_s,
            "buffer_state": self._buffer.queue_counts(),
        })

        if not acts_list:
            new_data = torch.empty(0, self._d_model, dtype=self._dtype, device=self._device)
        else:
            new_data = torch.cat(acts_list, dim=0)
            new_data = new_data.to(device=self._device, dtype=self._dtype)
            new_data = self._prepare_new_data(new_data, valid_rows_by_chunk)
        self._tp_broadcast_new_data(new_data)

        self._last_wait_time_s = vllm_wait_s
        self._last_transfer_time_s = time.perf_counter() - t_after_acquire

        self._shm_log({
            "event": "refill_complete",
            "new_tokens": new_data.shape[0],
            "pool_tokens_after": leftover + new_data.shape[0],
            "wait_time_s": self._last_wait_time_s,
            "transfer_time_s": self._last_transfer_time_s,
        })
        self._merge_into_pool(new_data)

    def _refill_pp_sibling_tp_root(self) -> None:
        t0 = time.perf_counter()
        indices, valid_rows = self._receive_indices_from_dp_root()
        if indices is None:
            self._make_all_local_pool_servable()
            # End-of-stream — also propagate to our TP followers, then stop.
            if self._sae_tp_group is not None:
                meta = torch.zeros(1, dtype=torch.int64, device=self._device)
                dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
            raise StopIteration

        new_data = self._read_and_prepare_chunks(indices, valid_rows)
        self._tp_broadcast_new_data(new_data)
        self._last_wait_time_s = 0.0
        self._last_transfer_time_s = time.perf_counter() - t0
        self._merge_into_pool(new_data)

    def _refill_tp_follower(self) -> None:
        t0 = time.perf_counter()
        meta = torch.zeros(1, dtype=torch.int64, device=self._device)
        assert self._sae_tp_group is not None
        dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
        num_rows = int(meta[0])
        if num_rows == 0:
            self._make_all_local_pool_servable()
            raise StopIteration
        new_data = torch.empty(
            num_rows, self._d_model, dtype=self._dtype, device=self._device
        )
        dist.broadcast(new_data, src=self._tp_root_global, group=self._sae_tp_group)
        self._last_wait_time_s = 0.0
        self._last_transfer_time_s = time.perf_counter() - t0
        self._merge_into_pool(new_data)

    def _read_and_prepare_chunks(
        self, indices: list[int], valid_rows_by_chunk: list[int]
    ) -> torch.Tensor:
        """Read the given chunks from shm, release each one, and reshape."""
        acts_list: list[torch.Tensor] = []
        for idx, valid in zip(indices, valid_rows_by_chunk):
            tensor, _ = self._buffer.read_chunk(idx)
            acts_list.append(tensor[:valid])
            self._buffer.release_chunk(idx)
        if not acts_list:
            return torch.empty(0, self._d_model, dtype=self._dtype, device=self._device)
        new_data = torch.cat(acts_list, dim=0)
        new_data = new_data.to(device=self._device, dtype=self._dtype)
        return self._prepare_new_data(new_data, valid_rows_by_chunk)

    def _tp_broadcast_new_data(self, new_data: torch.Tensor) -> None:
        if self._sae_tp_group is None:
            return
        meta = torch.tensor(
            [new_data.shape[0]], dtype=torch.int64, device=self._device
        )
        dist.broadcast(meta, src=self._tp_root_global, group=self._sae_tp_group)
        dist.broadcast(new_data.contiguous(), src=self._tp_root_global, group=self._sae_tp_group)

    def _broadcast_indices_to_pp_siblings(
        self,
        indices: list[int],
        *,
        end_of_stream: bool,
        valid_rows: list[int] | None = None,
    ) -> None:
        """DP root → PP siblings: send (count, [indices..., valid_rows...]).

        When ``end_of_stream`` is True, count is encoded as -1.
        No-op when ``sae_pp_size == 1`` or ``dp_replica_group`` is None.
        """
        if self._sae_pp_size <= 1 or self._dp_replica_group is None:
            return
        assert self._dp_replica_root_global is not None
        if end_of_stream:
            meta = torch.tensor([-1], dtype=torch.int64, device=self._device)
            dist.broadcast(meta, src=self._dp_replica_root_global, group=self._dp_replica_group)
            return
        n = len(indices)
        meta = torch.tensor([n], dtype=torch.int64, device=self._device)
        dist.broadcast(meta, src=self._dp_replica_root_global, group=self._dp_replica_group)
        if n == 0:
            return
        assert valid_rows is not None and len(valid_rows) == n
        payload = torch.tensor(indices + valid_rows, dtype=torch.int64, device=self._device)
        dist.broadcast(payload, src=self._dp_replica_root_global, group=self._dp_replica_group)

    def _receive_indices_from_dp_root(self) -> tuple[list[int] | None, list[int]]:
        """PP sibling: receive (indices, valid_rows) from the DP root.

        Returns (None, []) on end-of-stream, ([], []) on empty refill (impossible
        in the current design but handled for safety), otherwise (indices, valid_rows).
        """
        assert self._dp_replica_group is not None
        assert self._dp_replica_root_global is not None
        meta = torch.zeros(1, dtype=torch.int64, device=self._device)
        dist.broadcast(meta, src=self._dp_replica_root_global, group=self._dp_replica_group)
        n = int(meta[0])
        if n < 0:
            return None, []
        if n == 0:
            return [], []
        payload = torch.zeros(2 * n, dtype=torch.int64, device=self._device)
        dist.broadcast(payload, src=self._dp_replica_root_global, group=self._dp_replica_group)
        flat = payload.tolist()
        return flat[:n], flat[n:]

    def _merge_into_pool(self, new_data: torch.Tensor) -> None:
        if self._external_stop_requested():
            self.request_drain_local_pool()
        if self._mix_chunks > 0 and not self._drain_local_pool:
            self._merge_into_mixing_pool(new_data)
            return
        if self._pool is not None and self._pool_start < self._pool_len:
            leftover = self._pool[self._pool_start:]
            self._pool = torch.cat([leftover, new_data], dim=0)
        else:
            self._pool = new_data
        self._pool_start = 0
        self._pool_len = self._pool.shape[0]

    def _merge_into_mixing_pool(self, new_data: torch.Tensor) -> None:
        parts = []
        if self._pool is not None and self._pool_start < self._pool_len:
            parts.append(self._pool[self._pool_start:])
        if self._mixing_pool is not None and self._mixing_pool.shape[0] > 0:
            parts.append(self._mixing_pool)
        if new_data.shape[0] > 0:
            parts.append(new_data)
        if not parts:
            self._pool = new_data
            self._pool_start = 0
            self._pool_len = 0
            return
        combined = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        take_size = self._batch_size * self._num_hooks
        chunk_rows = int(getattr(self._buffer, "_chunk_size_tokens", take_size))
        capacity = max(take_size, self._mix_chunks * max(chunk_rows, 1))
        if combined.shape[0] < capacity:
            self._pool = combined[:0]
            self._pool_start = 0
            self._pool_len = 0
            self._mixing_pool = combined
            return

        if self._shuffle and self._mix_fraction > 0 and combined.shape[0] > 1:
            if self._is_multi_hook:
                combined = self._shuffle_prepared_multi_hook_data(combined)
            else:
                perm = self._randperm_for_tp(combined.shape[0], combined.device)
                combined = combined[perm]

        keep_for_mixing = int(capacity * self._mix_fraction)
        num_to_serve = combined.shape[0] - keep_for_mixing
        num_serving_batches = max(1, num_to_serve // take_size)
        serving_cutoff = num_serving_batches * take_size

        self._pool = combined[:serving_cutoff]
        self._pool_start = 0
        self._pool_len = self._pool.shape[0]
        self._mixing_pool = combined[serving_cutoff:]

    def _shuffle_prepared_multi_hook_data(self, data: torch.Tensor) -> torch.Tensor:
        """Shuffle multi-hook prepared data without mixing rows between hooks."""
        nh = self._num_hooks
        total = data.shape[0]
        if total % nh != 0:
            raise ValueError(
                f"Multi-hook prepared rows must be divisible by num_hooks: "
                f"rows={total}, num_hooks={nh}"
            )

        per_hook_parts: list[list[torch.Tensor]] = [[] for _ in range(nh)]
        take_size = self._batch_size * nh
        pos = 0
        while pos < total:
            group_rows = min(take_size, total - pos)
            if group_rows % nh != 0:
                raise ValueError(
                    f"Multi-hook prepared group rows must be divisible by num_hooks: "
                    f"group_rows={group_rows}, num_hooks={nh}"
                )
            rows_per_hook = group_rows // nh
            for h in range(nh):
                start = pos + h * rows_per_hook
                per_hook_parts[h].append(data[start : start + rows_per_hook])
            pos += group_rows

        per_hook = [torch.cat(parts, dim=0) for parts in per_hook_parts]
        if not per_hook or per_hook[0].shape[0] == 0:
            return data[:0]
        perm = self._randperm_for_tp(per_hook[0].shape[0], data.device)
        for h in range(nh):
            per_hook[h] = per_hook[h][perm]

        result_parts: list[torch.Tensor] = []
        tokens_per_hook = per_hook[0].shape[0]
        n_batches = tokens_per_hook // self._batch_size
        for b in range(n_batches):
            for h in range(nh):
                result_parts.append(
                    per_hook[h][b * self._batch_size : (b + 1) * self._batch_size]
                )
        rem = tokens_per_hook - n_batches * self._batch_size
        if rem > 0:
            for h in range(nh):
                result_parts.append(
                    per_hook[h][n_batches * self._batch_size : tokens_per_hook]
                )
        return torch.cat(result_parts, dim=0)

    def _randperm_for_tp(self, n: int, device: torch.device) -> torch.Tensor:
        """Return one permutation shared by every SAE TP rank."""
        generator = self._generator_for_device(device)
        if self._sae_tp_group is None:
            return torch.randperm(n, device=device, generator=generator)
        perm = torch.empty(n, dtype=torch.long, device=device)
        if self._sae_tp_rank == 0:
            perm.copy_(torch.randperm(n, device=device, generator=generator))
        dist.broadcast(perm, src=self._tp_root_global, group=self._sae_tp_group)
        return perm

    def _generator_for_device(self, device: torch.device) -> torch.Generator | None:
        if self._mixing_seed is None:
            return None
        device_key = str(device)
        generator = self._mixing_generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self._mixing_seed) + int(self._mixing_shard_index))
            self._mixing_generators[device_key] = generator
        return generator

    def _make_all_local_pool_servable(self) -> None:
        parts = []
        if self._pool is not None and self._pool_start < self._pool_len:
            parts.append(self._pool[self._pool_start:])
        if self._mixing_pool is not None and self._mixing_pool.shape[0] > 0:
            parts.append(self._mixing_pool)
        if not parts:
            self._pool = None
            self._pool_start = 0
            self._pool_len = 0
            self._mixing_pool = None
            return
        self._pool = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        self._pool_start = 0
        self._pool_len = self._pool.shape[0]
        self._mixing_pool = None

    def _debug_log_take(self, batch_by_hook: dict[str, torch.Tensor]) -> None:
        if (
            self._debug_provider_path is None
            or self._consume_step > self._debug_provider_max_steps
        ):
            return
        hooks: dict[str, dict[str, float | int]] = {}
        for name, acts in batch_by_hook.items():
            acts_f = acts.detach().float()
            hooks[name] = {
                "n": int(acts.shape[0]),
                "mean": float(acts_f.mean().cpu().item()),
                "std": float(acts_f.std(unbiased=False).cpu().item()),
                "abs_mean": float(acts_f.abs().mean().cpu().item()),
            }
        record = {
            "event": "take",
            "step": self._consume_step,
            "tp_rank": self._sae_tp_rank,
            "pp_rank": self._pp_rank,
            "batch_tokens": next(iter(hooks.values()))["n"] if hooks else 0,
            "hooks": hooks,
        }
        with open(self._debug_provider_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def _take(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return next batch from the pool, updating pool_start and consumed_tokens.

        For multi-hook, the pool stores interleaved hooks:
        [hook0_batch, hook1_batch, hook0_batch, hook1_batch, ...].
        We take batch_size * num_hooks rows and split into a dict, optionally
        filtered to ``select_hook_names`` so PP stages only see their hooks.
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
            full = {
                name: batch[i * tph : (i + 1) * tph]
                for i, name in enumerate(self._hook_names)
            }
            if self._select_hook_names is None:
                self._debug_log_take(full)
                return full
            selected = {name: full[name] for name in self._select_hook_names}
            self._debug_log_take(selected)
            return selected
        return batch

    def _prepare_new_data(
        self,
        data: torch.Tensor,
        valid_rows_by_chunk: list[int] | None = None,
    ) -> torch.Tensor:
        if self._is_multi_hook:
            return self._reinterleave_hooks(data, valid_rows_by_chunk)
        if self._shuffle and self._mix_chunks <= 0:
            perm = self._randperm_for_tp(data.shape[0], self._device)
            return data[perm]
        return data

    def _reinterleave_hooks(
        self,
        data: torch.Tensor,
        valid_rows_by_chunk: list[int] | None = None,
    ) -> torch.Tensor:
        """Re-interleave multi-hook chunk data into batch-sized blocks.

        Input layout (from buffer chunks):
          [chunk0_hook0, chunk0_hook1, chunk1_hook0, chunk1_hook1, ...]
          where each block is variable-sized but total rows per hook are equal.

        Output layout (for _take()):
          [hook0_batch, hook1_batch, hook0_batch, hook1_batch, ...]
          where each block is batch_size rows.

        When local mixing is disabled, the same token permutation is applied to
        every hook. With local mixing enabled, shuffling is deferred until the
        mixing buffer reaches capacity, matching ordinary mixing_buffer.
        """
        nh = self._num_hooks
        total = data.shape[0]
        if total % nh != 0:
            raise ValueError(
                f"Multi-hook streaming rows must be divisible by num_hooks: "
                f"rows={total}, num_hooks={nh}"
            )
        tokens_per_hook = total // nh

        # Split concatenated chunks into per-hook streams.
        # Chunks are laid out as [h0, h1, h0, h1, ...]. Each chunk can be
        # partial, so use the actual valid row count rather than assuming every
        # chunk contains a full per-hook block.
        chunk_rows = self._buffer._chunk_size_tokens if hasattr(self._buffer, '_chunk_size_tokens') else total
        if chunk_rows % nh != 0:
            raise ValueError(
                f"Multi-hook streaming chunk rows must be divisible by num_hooks: "
                f"chunk_rows={chunk_rows}, num_hooks={nh}"
            )
        if valid_rows_by_chunk is None:
            valid_rows_by_chunk = []
            remaining = total
            while remaining > 0:
                rows = min(chunk_rows, remaining)
                valid_rows_by_chunk.append(rows)
                remaining -= rows

        hook_parts: list[list[torch.Tensor]] = [[] for _ in range(nh)]
        pos = 0
        for valid_rows in valid_rows_by_chunk:
            if valid_rows % nh != 0:
                raise ValueError(
                    f"Multi-hook streaming valid rows must be divisible by num_hooks: "
                    f"valid_rows={valid_rows}, num_hooks={nh}"
                )
            per_hook_rows = valid_rows // nh
            for h in range(nh):
                end = pos + per_hook_rows
                hook_parts[h].append(data[pos:end])
                pos = end
        if pos != total:
            raise ValueError(
                f"Multi-hook streaming chunk metadata mismatch: consumed {pos} rows "
                f"from {total} rows"
            )
        per_hook = [torch.cat(parts, dim=0) for parts in hook_parts]

        if self._shuffle and self._mix_chunks <= 0:
            perm = self._randperm_for_tp(per_hook[0].shape[0], self._device)
            for h in range(nh):
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
