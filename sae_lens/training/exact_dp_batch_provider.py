"""Distribute one topology-independent activation batch across SAE-DP ranks."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.distributed as dist

from sae_lens.training.dp_batch import balanced_token_counts


Batch = torch.Tensor | dict[str, torch.Tensor]


class ExactDataParallelBatchProvider(Iterator[Batch]):
    """Keep one logical source/mixing stream and shard its output by token count.

    Rank zero of ``dp_group`` advances ``source``. Every rank in the group then
    receives its exact contiguous slice. All TP/PP coordinates instantiate an
    equivalent provider, so hook alignment is retained without changing the
    upstream mixing algorithm.
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
    ) -> None:
        if dp_size < 2:
            raise ValueError("ExactDataParallelBatchProvider requires dp_size >= 2")
        if not 0 <= dp_idx < dp_size:
            raise ValueError(f"dp_idx={dp_idx} out of range for dp_size={dp_size}")
        if dp_idx == 0 and source is None:
            raise ValueError("DP rank zero requires the logical source provider")
        self._source = source
        self._dp_group = dp_group
        self._dp_idx = int(dp_idx)
        self._dp_size = int(dp_size)
        self._device = torch.device(device)
        self._dtype = dtype
        self._d_model = int(d_model)
        self._hook_names = list(hook_names) if hook_names is not None else None
        self._source_global_rank = dist.get_global_rank(dp_group, 0)

    def __iter__(self) -> "ExactDataParallelBatchProvider":
        return self

    def __next__(self) -> Batch:
        batch: Batch | None = None
        status = 1
        if self._dp_idx == 0:
            assert self._source is not None
            try:
                batch = next(self._source)
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
        counts = balanced_token_counts(global_tokens, self._dp_size)
        start = sum(counts[: self._dp_idx])
        stop = start + counts[self._dp_idx]

        if self._hook_names is None:
            source_tensor = batch if isinstance(batch, torch.Tensor) else None
            return self._scatter_tensor(source_tensor, counts, start, stop)

        if self._dp_idx == 0:
            if not isinstance(batch, dict):
                raise TypeError("logical source must yield a dict for multi-hook training")
            missing = [name for name in self._hook_names if name not in batch]
            if missing:
                raise KeyError(f"logical source batch is missing hooks: {missing}")
        return {
            name: self._scatter_tensor(
                batch[name] if isinstance(batch, dict) else None,
                counts,
                start,
                stop,
            )
            for name in self._hook_names
        }

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
            if self._dp_idx == 0:
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
        if self._dp_idx == 0 and self._source is not None:
            request = getattr(self._source, "request_drain_local_pool", None)
            if callable(request):
                request()

    def consume_last_data_timing(self) -> dict[str, float]:
        if self._dp_idx == 0 and self._source is not None:
            consume = getattr(self._source, "consume_last_data_timing", None)
            if callable(consume):
                return consume()
        return {"vllm_step_time_s": 0.0, "transfer_time_s": 0.0}
