"""Distribute one topology-independent activation batch across SAE-DP ranks."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist

from sae_lens.training.dp_batch import balanced_token_counts


Batch = torch.Tensor | dict[str, torch.Tensor]


class ExactDataParallelBatchProvider(Iterator[Batch]):
    """Keep one logical source/mixing stream and shard its output by token count.

    One designated rank of ``dp_group`` advances ``source``. Every rank in the
    group then receives its exact contiguous slice. All TP/PP coordinates
    instantiate an equivalent provider, so hook alignment is retained without
    changing the upstream mixing algorithm.
    """

    def __init__(
        self,
        *,
        source: Iterator[Batch] | None,
        dp_group: dist.ProcessGroup,
        dp_idx: int,
        dp_size: int,
        device: torch.device,
        dtype: torch.dtype,
        d_model: int,
        hook_names: list[str] | None = None,
        global_training_tokens: int | None = None,
        initial_global_tokens: int = 0,
        initial_step: int = 0,
        current_epoch: int = 0,
        reconfigure_poll: Callable[[], tuple[int, int] | None] | None = None,
        source_dp_idx: int = 0,
    ) -> None:
        if dp_size < 1:
            raise ValueError("ExactDataParallelBatchProvider requires dp_size >= 1")
        if not 0 <= dp_idx < dp_size:
            raise ValueError(f"dp_idx={dp_idx} out of range for dp_size={dp_size}")
        if not 0 <= source_dp_idx < dp_size:
            raise ValueError(
                f"source_dp_idx={source_dp_idx} out of range for dp_size={dp_size}"
            )
        if dp_idx == source_dp_idx and source is None:
            raise ValueError("the source DP rank requires the logical source provider")
        self._source = source
        self._dp_group = dp_group
        self._dp_idx = int(dp_idx)
        self._dp_size = int(dp_size)
        self._source_dp_idx = int(source_dp_idx)
        self._device = torch.device(device)
        self._dtype = dtype
        self._d_model = int(d_model)
        self._hook_names = list(hook_names) if hook_names is not None else None
        self._source_global_rank = dist.get_global_rank(
            dp_group, self._source_dp_idx
        )
        if global_training_tokens is not None and global_training_tokens < 0:
            raise ValueError("global_training_tokens must be nonnegative")
        if initial_global_tokens < 0 or initial_step < 0 or current_epoch < 0:
            raise ValueError("initial progress and epoch must be nonnegative")
        self._global_training_tokens = global_training_tokens
        self._global_tokens_consumed = int(initial_global_tokens)
        self._step_index = int(initial_step)
        self._current_epoch = int(current_epoch)
        self._reconfigure_poll = reconfigure_poll
        self._pending_reconfigure: tuple[int, int] | None = None

    def __iter__(self) -> "ExactDataParallelBatchProvider":
        return self

    def __next__(self) -> Batch:
        batch: Batch | None = None
        status = 1
        if self._dp_idx == self._source_dp_idx:
            assert self._source is not None
            remaining = (
                None
                if self._global_training_tokens is None
                else self._global_training_tokens - self._global_tokens_consumed
            )
            if remaining is not None and remaining <= 0:
                status = 0
            else:
                try:
                    batch = next(self._source)
                    if remaining is not None and self._batch_tokens(batch) > remaining:
                        batch = self._slice_batch(batch, 0, remaining)
                except StopIteration:
                    status = 0

        meta = torch.tensor(
            [status, self._batch_tokens(batch)],
            dtype=torch.int64,
            device=self._device,
        )
        dist.broadcast(meta, src=self._source_global_rank, group=self._dp_group)
        if int(meta[0]) == 0:
            raise StopIteration

        global_tokens = int(meta[1])
        partition_step = self._step_index if self.tracks_global_progress else 0
        counts = balanced_token_counts(global_tokens, self._dp_size, partition_step)
        start = sum(counts[: self._dp_idx])
        stop = start + counts[self._dp_idx]

        if self._hook_names is None:
            source_tensor = batch if isinstance(batch, torch.Tensor) else None
            result = self._scatter_tensor(source_tensor, counts, start, stop)
            self._advance(global_tokens)
            return result

        if self._dp_idx == self._source_dp_idx:
            if not isinstance(batch, dict):
                raise TypeError("logical source must yield a dict for multi-hook training")
            missing = [name for name in self._hook_names if name not in batch]
            if missing:
                raise KeyError(f"logical source batch is missing hooks: {missing}")
        result = {
            name: self._scatter_tensor(
                batch[name] if isinstance(batch, dict) else None,
                counts,
                start,
                stop,
            )
            for name in self._hook_names
        }
        self._advance(global_tokens)
        return result

    @property
    def tracks_global_progress(self) -> bool:
        return self._global_training_tokens is not None

    @property
    def global_tokens_consumed(self) -> int:
        return self._global_tokens_consumed

    @property
    def global_training_tokens(self) -> int | None:
        return self._global_training_tokens

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def pending_reconfigure(self) -> tuple[int, int] | None:
        return self._pending_reconfigure

    def restore_progress(self, *, global_tokens: int, step: int, epoch: int) -> None:
        if global_tokens < 0 or step < 0 or epoch < 0:
            raise ValueError("restored exact-provider progress must be nonnegative")
        self._global_tokens_consumed = int(global_tokens)
        self._step_index = int(step)
        self._current_epoch = int(epoch)
        self._pending_reconfigure = None

    def poll_reconfigure(self) -> bool:
        """Collectively observe a manual request before fetching the next batch."""
        if self._reconfigure_poll is None:
            return False
        request = (
            self._reconfigure_poll()
            if self._dp_idx == self._source_dp_idx
            else None
        )
        meta = torch.tensor(
            [-1, -1] if request is None else list(request),
            dtype=torch.int64,
            device=self._device,
        )
        dist.broadcast(meta, src=self._source_global_rank, group=self._dp_group)
        epoch = int(meta[0])
        if epoch <= self._current_epoch:
            return False
        self._pending_reconfigure = (epoch, int(meta[1]))
        return True

    def _advance(self, global_tokens: int) -> None:
        self._global_tokens_consumed += global_tokens
        self._step_index += 1

    @staticmethod
    def _slice_batch(batch: Batch, start: int, stop: int) -> Batch:
        if isinstance(batch, torch.Tensor):
            return batch[start:stop]
        return {name: value[start:stop] for name, value in batch.items()}

    def _batch_tokens(self, batch: Batch | None) -> int:
        if batch is None:
            return 0
        if isinstance(batch, torch.Tensor):
            return int(batch.shape[0])
        if not batch:
            raise ValueError("logical source returned an empty hook dictionary")
        sizes = {int(value.shape[0]) for value in batch.values()}
        if len(sizes) != 1:
            raise ValueError(f"logical source hook sizes diverged: {sorted(sizes)}")
        return sizes.pop()

    def _scatter_tensor(
        self,
        source_tensor: torch.Tensor | None,
        counts: tuple[int, ...],
        local_start: int,
        local_stop: int,
    ) -> torch.Tensor:
        result: torch.Tensor | None = None
        cursor = 0
        for destination, count in enumerate(counts):
            if self._dp_idx == self._source_dp_idx:
                assert source_tensor is not None
                payload = source_tensor[cursor : cursor + count].contiguous()
            else:
                payload = torch.empty(
                    count,
                    self._d_model,
                    dtype=self._dtype,
                    device=self._device,
                )
            dist.broadcast(payload, src=self._source_global_rank, group=self._dp_group)
            if destination == self._dp_idx:
                result = payload
            cursor += count
        assert result is not None
        if result.shape[0] != local_stop - local_start:
            raise RuntimeError("exact DP scatter produced an invalid local token count")
        return result

    def request_drain_local_pool(self) -> None:
        if self._dp_idx == self._source_dp_idx and self._source is not None:
            request = getattr(self._source, "request_drain_local_pool", None)
            if callable(request):
                request()

    def consume_last_data_timing(self) -> dict[str, float]:
        if self._dp_idx == self._source_dp_idx and self._source is not None:
            consume = getattr(self._source, "consume_last_data_timing", None)
            if callable(consume):
                return consume()
        return {"vllm_step_time_s": 0.0, "transfer_time_s": 0.0}
