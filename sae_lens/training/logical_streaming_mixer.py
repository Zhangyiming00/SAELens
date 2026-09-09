"""Topology-independent logical mixing streams for exact SHM streaming."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import torch

from sae_lens.training.dp_batch import balanced_token_counts
from sae_lens.training.mixing_buffer import ActivationBatch, mixing_buffer


def _batch_tokens(batch: ActivationBatch) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    if not batch:
        raise ValueError("activation batch must contain at least one hook")
    sizes = {int(value.shape[0]) for value in batch.values()}
    if len(sizes) != 1:
        raise ValueError(f"activation hook sizes diverged: {sorted(sizes)}")
    return sizes.pop()


def _slice_batch(batch: ActivationBatch, start: int, stop: int) -> ActivationBatch:
    if isinstance(batch, torch.Tensor):
        return batch[start:stop]
    return {name: value[start:stop] for name, value in batch.items()}


def _cat_batches(batches: list[ActivationBatch]) -> ActivationBatch:
    first = batches[0]
    if isinstance(first, torch.Tensor):
        return torch.cat([batch for batch in batches if isinstance(batch, torch.Tensor)])
    keys = list(first)
    return {
        name: torch.cat(
            [batch[name] for batch in batches if isinstance(batch, dict)], dim=0
        )
        for name in keys
    }


class _LogicalStreamLoader(Iterator[ActivationBatch]):
    def __init__(self, owner: "LogicalStreamingMixingProvider", stream_idx: int) -> None:
        self._owner = owner
        self._stream_idx = stream_idx

    def __iter__(self) -> "_LogicalStreamLoader":
        return self

    def __next__(self) -> ActivationBatch:
        queue = self._owner._queues[self._stream_idx]
        while not queue:
            self._owner._route_next_source_batch()
        return queue.popleft()


class LogicalStreamingMixingProvider(Iterator[ActivationBatch]):
    """Mix fixed logical token streams, then reconstruct each global batch.

    The source is split according to repeating global-batch ownership intervals.
    Consequently each logical stream receives its exact local count every step,
    independent of raw SHM chunk boundaries. Stream-local buffers keep separate
    capacity, retention state, and RNG state when physical SAE-DP changes later.
    """

    def __init__(
        self,
        *,
        source: Iterator[ActivationBatch],
        global_batch_size: int,
        stream_count: int,
        buffer_size_per_stream: int,
        mix_fraction: float,
        shuffle: bool,
        seed: int,
    ) -> None:
        if global_batch_size < stream_count:
            raise ValueError(
                "global_batch_size must be >= stream_count for logical mixing"
            )
        if buffer_size_per_stream < 1:
            raise ValueError("buffer_size_per_stream must be >= 1")
        self._source = source
        self._global_batch_size = int(global_batch_size)
        self._stream_counts = balanced_token_counts(global_batch_size, stream_count)
        if buffer_size_per_stream < max(self._stream_counts):
            raise ValueError(
                "buffer_size_per_stream must fit the largest logical local batch"
            )
        self._boundaries: tuple[int, ...] = tuple(
            sum(self._stream_counts[: idx + 1]) for idx in range(stream_count)
        )
        self._queues: list[deque[ActivationBatch]] = [
            deque() for _ in range(stream_count)
        ]
        self._global_cursor = 0
        self._mixers: list[Iterator[ActivationBatch]] = []
        for stream_idx, batch_size in enumerate(self._stream_counts):
            generator = torch.Generator()
            generator.manual_seed(int(seed) + stream_idx)
            self._mixers.append(
                mixing_buffer(
                    buffer_size=buffer_size_per_stream,
                    batch_size=batch_size,
                    activations_loader=_LogicalStreamLoader(self, stream_idx),
                    mix_fraction=mix_fraction,
                    generator=generator,
                    shuffle=shuffle,
                )
            )

    @property
    def stream_counts(self) -> tuple[int, ...]:
        return self._stream_counts

    def __iter__(self) -> "LogicalStreamingMixingProvider":
        return self

    def __next__(self) -> ActivationBatch:
        batches = [next(mixer) for mixer in self._mixers]
        result = _cat_batches(batches)
        if _batch_tokens(result) != self._global_batch_size:
            raise RuntimeError("logical streams did not reconstruct one global batch")
        return result

    def _route_next_source_batch(self) -> None:
        batch = next(self._source)
        batch_tokens = _batch_tokens(batch)
        source_cursor = 0
        while source_cursor < batch_tokens:
            step_cursor = self._global_cursor % self._global_batch_size
            stream_idx = next(
                idx for idx, boundary in enumerate(self._boundaries) if step_cursor < boundary
            )
            stream_start = 0 if stream_idx == 0 else self._boundaries[stream_idx - 1]
            available = self._stream_counts[stream_idx] - (step_cursor - stream_start)
            take = min(available, batch_tokens - source_cursor)
            self._queues[stream_idx].append(
                _slice_batch(batch, source_cursor, source_cursor + take)
            )
            source_cursor += take
            self._global_cursor += take

    def request_drain_local_pool(self) -> None:
        request = getattr(self._source, "request_drain_local_pool", None)
        if callable(request):
            request()

    def consume_last_data_timing(self) -> dict[str, float]:
        consume = getattr(self._source, "consume_last_data_timing", None)
        if callable(consume):
            return consume()
        return {"vllm_step_time_s": 0.0, "transfer_time_s": 0.0}
