"""GPU direct streaming activation provider for SAE training.

Provides activations received via NCCL from vLLM producer, maintaining per-hook
pools to handle variable chunk sizes and training batch sizes. Supports optional
shuffling and cross-chunk mixing.
"""

from __future__ import annotations

import torch


class GpuStreamingActivationProvider:
    """Provides activations from GPU direct NCCL streaming.

    Maintains per-hook pools to handle cases where valid_tokens_per_hook
    differs from train_batch_size_tokens. Supports shuffling and cross-chunk mixing.
    """

    def __init__(
        self,
        pp_hook_names: list[str],
        is_multi_sae: bool,
        train_batch_size_tokens: int,
        d_model: int,
        shuffle: bool = False,
        mix_chunks: int = 0,
        buffer_size: int | None = None,
        mix_fraction: float = 0.5,
        mixing_seed: int | None = None,
        mixing_shard_index: int = 0,
        device: torch.device | str = "cuda",
    ):
        self._pp_hook_names = pp_hook_names
        self._is_multi_sae = is_multi_sae
        self._train_batch_size_tokens = train_batch_size_tokens
        self._d_model = d_model
        self._shuffle = shuffle
        self._mix_chunks = mix_chunks
        if buffer_size is not None and buffer_size < train_batch_size_tokens:
            raise ValueError("buffer_size must be >= train_batch_size_tokens")
        if not 0 <= mix_fraction <= 1:
            raise ValueError("mix_fraction must be in [0, 1]")
        self._buffer_size = buffer_size
        self._mix_fraction = mix_fraction
        self._mixing_seed = mixing_seed
        self._mixing_shard_index = mixing_shard_index
        self._mixing_generators: dict[str, torch.Generator] = {}
        self._device = torch.device(device) if isinstance(device, str) else device
        self._pool_by_hook: dict[str, torch.Tensor] = {
            h: torch.empty(0, d_model, device=self._device) for h in pp_hook_names
        }
        self._serving_by_hook: dict[str, torch.Tensor] = {
            h: torch.empty(0, d_model, device=self._device) for h in pp_hook_names
        }
        self._chunk_buffer: list[dict[str, torch.Tensor]] = []
        self._eof_received = False
        self._last_tokens_per_hook: int | None = None

    def _min_pool_tokens(self) -> int:
        """Return minimum tokens across all hook pools."""
        pools = self._serving_by_hook if self._use_standard_mixing else self._pool_by_hook
        if not pools:
            return 0
        return min(t.shape[0] for t in pools.values())

    def serving_tokens(self) -> int:
        """Tokens per hook ready for training."""
        if not self._use_standard_mixing:
            return self._min_pool_tokens()
        if not self._serving_by_hook:
            return 0
        return min(t.shape[0] for t in self._serving_by_hook.values())

    def storage_tokens(self) -> int:
        """Tokens per hook retained in the mixing storage pool."""
        if not self._use_standard_mixing or not self._pool_by_hook:
            return 0
        return min(t.shape[0] for t in self._pool_by_hook.values())

    def _serving_window_tokens(self) -> int:
        if self._buffer_size is None:
            return self._train_batch_size_tokens
        keep_for_mixing = int(self._buffer_size * self._mix_fraction)
        return max(1, self._buffer_size - keep_for_mixing)

    def _receiver_token_budget(self) -> int:
        if self._buffer_size is None:
            return self._train_batch_size_tokens
        return self._buffer_size + self._serving_window_tokens()

    def prefill_target_tokens(self, requested_target_tokens: int) -> int:
        """Compute effective prefill target clamped to one serving window."""
        if requested_target_tokens <= 0:
            return 0
        if not self._use_standard_mixing:
            return requested_target_tokens
        return min(requested_target_tokens, self._serving_window_tokens())

    def prefill_satisfied(self, target_tokens: int) -> bool:
        """Return true once post-mixing serving tokens can satisfy prefill."""
        if target_tokens <= 0:
            return True
        return self.serving_tokens() >= target_tokens

    @property
    def available_tokens(self) -> int:
        """Tokens per hook available across serving and storage pools."""
        return self._min_pool_tokens() + (
            self._min_storage_tokens() if self._use_standard_mixing else 0
        )

    def needs_refill(self) -> bool:
        """True when caller should request more chunks to reach high watermark."""
        if self._eof_received:
            return False
        if self._buffer_size is None:
            return self._min_pool_tokens() < self._train_batch_size_tokens
        return self.available_tokens < self._buffer_size

    def receiver_needs_refill(self) -> bool:
        """True when the receiver should request more producer data."""
        if self._eof_received:
            return False
        if not self._use_standard_mixing:
            return self._min_pool_tokens() < self._train_batch_size_tokens
        return (
            self.storage_tokens() < self._buffer_size
            and self.available_tokens < self._receiver_token_budget()
        )

    @property
    def _use_standard_mixing(self) -> bool:
        return self._buffer_size is not None

    def _refill(self, recv_buf: torch.Tensor) -> None:
        """Split recv_buf by hook and append to per-hook pools.

        recv_buf layout: [hook0_tokens..., hook1_tokens..., ...]
        """
        tokens_per_hook = recv_buf.shape[0] // len(self._pp_hook_names)
        self._last_tokens_per_hook = tokens_per_hook
        chunk_dict = {}
        for i, hook in enumerate(self._pp_hook_names):
            part = recv_buf[i * tokens_per_hook : (i + 1) * tokens_per_hook]
            chunk_dict[hook] = part
        self._chunk_buffer.append(chunk_dict)

        # Legacy local chunk mixing; GPU direct runner now uses buffer_size /
        # mix_fraction to match ordinary activations_store.mixing_buffer.
        if (
            not self._use_standard_mixing
            and self._mix_chunks > 0
            and len(self._chunk_buffer) >= self._mix_chunks
        ):
            self._apply_mixing()

        if not self._use_standard_mixing and self._shuffle:
            self._shuffle_chunk_dict(chunk_dict)

        # Add to storage pools
        for hook, chunk in chunk_dict.items():
            self._pool_by_hook[hook] = torch.cat(
                [self._pool_by_hook[hook], chunk.to(self._device)], dim=0
            )
        if self._use_standard_mixing:
            self._refresh_standard_mixing()

    def _shuffle_chunk_dict(self, chunk_dict: dict[str, torch.Tensor]) -> None:
        """Apply one token permutation to every hook in a received chunk."""
        if not chunk_dict:
            return
        first = next(iter(chunk_dict.values()))
        if first.shape[0] <= 1:
            return
        perm = torch.randperm(first.shape[0], device=first.device)
        for hook in self._pp_hook_names:
            chunk_dict[hook] = chunk_dict[hook][perm]

    def _refresh_standard_mixing(self) -> None:
        """Mirror training.mixing_buffer using per-hook storage pools."""
        assert self._buffer_size is not None
        storage_len = self._min_storage_tokens()
        if storage_len < self._buffer_size:
            return

        if self._shuffle and self._mix_fraction > 0 and storage_len > 1:
            perm = self._randperm(storage_len, self._device)
            for hook in self._pp_hook_names:
                self._pool_by_hook[hook] = self._pool_by_hook[hook][perm]

        keep_for_mixing = int(self._buffer_size * self._mix_fraction)
        num_to_serve = storage_len - keep_for_mixing
        num_serving_batches = max(1, num_to_serve // self._train_batch_size_tokens)
        serving_cutoff = num_serving_batches * self._train_batch_size_tokens

        for hook in self._pp_hook_names:
            serving = self._pool_by_hook[hook][:serving_cutoff]
            self._serving_by_hook[hook] = torch.cat(
                [self._serving_by_hook[hook], serving], dim=0
            )
            self._pool_by_hook[hook] = self._pool_by_hook[hook][serving_cutoff:]

    def _min_storage_tokens(self) -> int:
        if not self._pool_by_hook:
            return 0
        return min(t.shape[0] for t in self._pool_by_hook.values())

    def _randperm(self, n: int, device: torch.device) -> torch.Tensor:
        generator = self._generator_for_device(device)
        return torch.randperm(n, device=device, generator=generator)

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

    def _apply_mixing(self) -> None:
        """Mix tokens from recent chunks according to mix_fraction."""
        if len(self._chunk_buffer) < self._mix_chunks:
            return

        chunks_to_mix = self._chunk_buffer[-self._mix_chunks :]

        for hook in self._pp_hook_names:
            tokens_to_mix = []
            for chunk in chunks_to_mix:
                if chunk[hook].shape[0] > 0:
                    tokens_to_mix.append(chunk[hook])

            if len(tokens_to_mix) > 1:
                mixed = torch.cat(tokens_to_mix, dim=0)
                if self._shuffle:
                    indices = torch.randperm(mixed.shape[0], device=mixed.device)
                    mixed = mixed[indices]

                tokens_per_chunk = mixed.shape[0] // len(chunks_to_mix)
                for i, chunk in enumerate(chunks_to_mix):
                    start = i * tokens_per_chunk
                    end = (
                        start + tokens_per_chunk
                        if i < len(chunks_to_mix) - 1
                        else mixed.shape[0]
                    )
                    chunk[hook] = mixed[start:end]

    def receive_chunk(self, recv_buf: torch.Tensor) -> None:
        """Receive a chunk from NCCL and add to pools."""
        self._refill(recv_buf)

    def desired_refill_chunks(self) -> int:
        """Return chunks needed to bring the mixing buffer back to high watermark."""
        if self._eof_received or self._last_tokens_per_hook is None:
            return 1
        if self._buffer_size is None:
            needed = self._train_batch_size_tokens - self.available_tokens
        else:
            needed = self._buffer_size - self.available_tokens
        if needed <= 0:
            return 1
        return max(1, (needed + self._last_tokens_per_hook - 1) // self._last_tokens_per_hook)

    def desired_receiver_refill_chunks(self) -> int:
        """Return chunks needed for receiver-side overlap without exceeding budget."""
        if self._eof_received or self._last_tokens_per_hook is None:
            return 1
        if not self._use_standard_mixing:
            needed = self._train_batch_size_tokens - self._min_pool_tokens()
        else:
            storage_gap = self._buffer_size - self.storage_tokens()
            budget_gap = self._receiver_token_budget() - self.available_tokens
            needed = min(storage_gap, budget_gap)
        if needed <= 0:
            return 1
        return max(1, (needed + self._last_tokens_per_hook - 1) // self._last_tokens_per_hook)

    def mark_eof(self) -> None:
        """Mark that EOF has been received; drain pool after this."""
        self._eof_received = True
        if self._use_standard_mixing:
            for hook in self._pp_hook_names:
                if self._pool_by_hook[hook].shape[0] > 0:
                    self._serving_by_hook[hook] = torch.cat(
                        [self._serving_by_hook[hook], self._pool_by_hook[hook]], dim=0
                    )
                    self._pool_by_hook[hook] = self._pool_by_hook[hook][:0]

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor] | torch.Tensor:
        """Return next batch of train_batch_size_tokens per hook.

        Partial final batch (fewer tokens than train_batch_size_tokens) is
        returned as-is without padding.
        """
        if (
            self._min_pool_tokens() < self._train_batch_size_tokens
            and not self._eof_received
        ):
            raise StopIteration("No more data and EOF not received")
        if self._min_pool_tokens() == 0:
            raise StopIteration
        return self._take()

    def _take(self) -> dict[str, torch.Tensor] | torch.Tensor:
        """Slice train_batch_size_tokens per hook from pool front."""
        take = min(self._train_batch_size_tokens, self._min_pool_tokens())
        source = self._serving_by_hook if self._use_standard_mixing else self._pool_by_hook
        batch_by_hook = {}
        for hook, pool in source.items():
            batch_by_hook[hook] = pool[:take]
            source[hook] = pool[take:]

        if self._is_multi_sae or len(self._pp_hook_names) > 1:
            return batch_by_hook
        return next(iter(batch_by_hook.values()))
