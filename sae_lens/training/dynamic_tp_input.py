"""Lossless SHM refill for a changing TP group.

Only the permanent input owner uses this loader. All collectives remain on the
training thread, after membership commits. The raw cache retains the SHM dtype
and belongs to DynamicTPSession, including during its usual cache migration.
"""

from __future__ import annotations

import torch

from sae_lens.training.shared_activation_buffer import SharedActivationBuffer


class DynamicTPInputLoader:
    def __init__(
        self,
        buffer: SharedActivationBuffer,
        *,
        capacity_chunks: int,
        device: torch.device | str,
        seed: int = 42,
    ) -> None:
        if capacity_chunks < 1:
            raise ValueError("capacity_chunks must be positive")
        self.buffer = buffer
        self.capacity_chunks = capacity_chunks
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(seed)
        shape = (capacity_chunks * buffer._chunk_size_tokens, buffer._d_model)
        self.host = torch.empty(
            shape, dtype=buffer._dtype, pin_memory=self.device.type == "cuda"
        )
        self.upload = torch.empty(shape, dtype=buffer._dtype, device=self.device)
        self.copy_done = torch.cuda.Event() if self.device.type == "cuda" else None
        self.copy_pending = False
        self.closed = False

    def load_raw(
        self, count: int, *, random_chunks: bool = True
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Return an independent, shuffled cache in the SHM storage dtype."""
        if self.closed:
            raise RuntimeError("input loader is closed")
        if not 1 <= count <= self.capacity_chunks:
            raise ValueError("refill exceeds input capacity")
        # Host memory must not be overwritten while a previous DMA reads it.
        if self.copy_pending:
            assert self.copy_done is not None
            self.copy_done.synchronize()
            self.copy_pending = False
        slots, _ = self.buffer.acquire_up_to(count, random=random_chunks)
        sequences = []
        rows = 0
        try:
            if len(slots) != count:
                raise RuntimeError("refill count no longer matches READY slots")
            for slot in slots:
                valid = int(self.buffer._meta[slot, 0])
                if valid != self.buffer._chunk_size_tokens:
                    raise ValueError("dynamic TP refill requires complete chunks")
                sequences.append(int(self.buffer._meta[slot, 2]))
                self.buffer.copy_chunk_into(slot, self.host[rows : rows + valid])
                rows += valid
        finally:
            for slot in slots:
                self.buffer.release_chunk(slot)

        self.upload[:rows].copy_(self.host[:rows], non_blocking=True)
        if self.copy_done is not None:
            self.copy_done.record(torch.cuda.current_stream(self.device))
            self.copy_pending = True
        # Keep the original CPU RNG sequence, but gather the large tensor on GPU.
        order = torch.randperm(rows, generator=self.generator)
        data = self.upload[:rows][order.to(self.device)]
        return data, sequences, order

    def load(
        self, count: int, *, scale: float | None, random_chunks: bool = True
    ) -> tuple[torch.Tensor, list[int], torch.Tensor, float]:
        """Compatibility API for callers explicitly requesting a scaled FP32 pool."""
        data, sequences, order = self.load_raw(count, random_chunks=random_chunks)
        data = data.float()
        if scale is None:
            scale = float(self.buffer._d_model**0.5 / data.norm(dim=-1).mean())
        data.mul_(scale)
        return data, sequences, order, scale

    def close(self) -> None:
        if self.closed:
            return
        if self.copy_pending:
            assert self.copy_done is not None
            self.copy_done.synchronize()
            self.copy_pending = False
        self.closed = True
        del self.host, self.upload
