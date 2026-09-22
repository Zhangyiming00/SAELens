"""Exact, memory-bounded TopK over feature shards.

This module deliberately has no SAELens/Megatron import dependencies. ``group=None``
means a *local* operation, never WORLD. Global feature scores/activations are never
assembled. The fast path exchanges only int64 candidate keys. If that workspace
would exceed one latent shard, a fixed-round distributed radix selection exchanges
histograms instead. Communication is detached; values are gathered from the local
live autograd tensor after the winning threshold has been determined.

Total order: score descending, global feature index ascending. Scores must be
FP32/BF16/FP16; zeros have identical ranking and NaNs rank above +infinity. This is
not a promise of bitwise agreement with torch.topk's unspecified tie ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist

_MIN_KEY = -(1 << 63)


def _group_size(group: dist.ProcessGroup | None) -> int:
    return 1 if group is None else dist.get_world_size(group)


def _group_rank(group: dist.ProcessGroup | None) -> int:
    return 0 if group is None else dist.get_rank(group)


@torch.no_grad()
def score_keys(scores: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """An int64 lexicographic key; no epsilon is added to a floating-point score.

    Workspace is local [rows, features_per_rank]. The 32 low bits encode an
    inverse global index, so every eligible feature has a unique key.
    """
    if scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Sharded exact TopK supports float16/bfloat16/float32 scores")
    if scores.ndim != 2 or not 0 <= offset < (1 << 32):
        raise ValueError("Expected [rows, local_features] and a uint32 offset")
    if offset + scores.shape[1] > (1 << 32):
        raise ValueError("Global feature indices must fit in uint32")
    f = scores.detach().float().contiguous()
    # Canonicalize signed zero and NaN payloads, preserving infinities. Actual
    # values are NOT sanitized: a selected NaN still propagates through loss.
    f = torch.where(f == 0, 0.0, f)
    bits = f.view(torch.int32).to(torch.int64)
    # Canonical NaN sorts above +inf, independent of sign or payload.
    bits = torch.where(torch.isnan(f), 0x7FFFFFFF, bits)
    ordered = torch.where(bits < 0, ~bits, bits ^ 0x80000000) & 0xFFFFFFFF
    high = (ordered - 0x80000000) << 32
    ids = torch.arange(
        offset, offset + scores.shape[1], device=scores.device, dtype=torch.int64
    )
    return high | (0xFFFFFFFF - ids)


@torch.no_grad()
def _local_candidates(
    scores: torch.Tensor,
    n: int,
    offset: int,
    eligible: torch.Tensor | None,
    key_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound comparison-key scratch independently of token batch size.

    One fused Triton kernel is used when explicitly selected for GPU keys.
    The torch-only fallback uses smaller row tiles to bound its intermediates.
    There is no float-score epsilon perturbation or approximate local capacity.
    """
    rows, width = scores.shape
    out_keys = torch.empty((rows, n), dtype=torch.int64, device=scores.device)
    out_indices = torch.empty_like(out_keys)
    key_kernel = None
    if key_backend == "triton":
        if not scores.is_cuda:
            raise ValueError(
                "Triton comparison keys require CUDA; use key_backend='torch'"
            )
        from sae_lens.sharded_triton import make_score_keys

        key_kernel = make_score_keys
    # GPU key buffer <=32 MiB. Torch fallback has several elementwise temporaries.
    scratch_per_element = 8 if key_kernel else 64
    tile = max(1, (32 * 1024 * 1024) // (width * scratch_per_element))
    for start in range(0, rows, tile):
        block = scores[start : start + tile]
        keys = (
            score_keys(block, offset)
            if key_kernel is None
            else key_kernel(block, offset)
        )
        if eligible is not None:
            keys.masked_fill_(~eligible[None, :], _MIN_KEY)
        values, indices = keys.topk(n, dim=1, sorted=False)
        out_keys[start : start + tile].copy_(values)
        out_indices[start : start + tile].copy_(indices)
        del keys, values, indices
    return out_keys, out_indices


@torch.no_grad()
def _radix_threshold(
    local_keys: torch.Tensor, k: int, group: dist.ProcessGroup | None
) -> torch.Tensor:
    """Find the kth largest unique signed int64 key using bounded histograms.

    Histograms have at most 256 bins, not global feature activations. Values/indices are not
    broadcast. All ranks execute exactly the same number/order of collectives.
    """
    rows = local_keys.shape[0]
    active = torch.ones_like(local_keys, dtype=torch.bool)
    remaining = torch.full((rows,), k, dtype=torch.int64, device=local_keys.device)
    threshold = torch.zeros_like(remaining)
    # Use smaller digits for tiny shards/candidate sets. No rank-dependent loop.
    digit_bits = min(8, max(1, int(math.log2(max(2, local_keys.shape[1])))))
    positions = list(range(0, 64, digit_bits))[::-1]
    for shift in positions:
        bits = min(digit_bits, 64 - shift)
        bins = 1 << bits
        digit = (local_keys >> shift) & (bins - 1)
        if shift + bits == 64:
            digit = digit ^ (bins // 2)  # signed to unsigned lexicographic order
        hist = torch.zeros((rows, bins), dtype=torch.int64, device=local_keys.device)
        hist.scatter_add_(1, digit, active.to(torch.int64))
        if _group_size(group) > 1:
            dist.all_reduce(hist, group=group)
        tail = hist.flip(1).cumsum(1)
        pos = (tail < remaining[:, None]).sum(1)
        bucket = bins - 1 - pos
        before = tail.gather(1, (pos - 1).clamp_min(0)[:, None]).squeeze(1)
        remaining = remaining - torch.where(pos > 0, before, 0)
        active = active & (digit == bucket[:, None])
        raw = (bucket ^ (bins // 2)) if shift + bits == 64 else bucket
        threshold = threshold | (raw << shift)
    return threshold


def _sparse_from_padded(
    indices: torch.Tensor,
    values: torch.Tensor,
    leading_shape: tuple[int, ...],
    width: int,
) -> torch.Tensor:
    rows, count = indices.shape
    flat_row = torch.arange(rows, device=indices.device).repeat_interleave(count)
    coords = []
    stride = rows
    for n in leading_shape:
        stride //= max(n, 1)
        coords.append((flat_row // max(stride, 1)) % max(n, 1))
    coords.append(indices.reshape(-1))
    return torch.sparse_coo_tensor(
        torch.stack(coords),
        values.reshape(-1),
        (*leading_shape, width),
        device=values.device,
    ).coalesce()


@dataclass
class PendingShardedTopK:
    scores: torch.Tensor
    indices: torch.Tensor
    keys: torch.Tensor
    threshold: torch.Tensor
    leading_shape: tuple[int, ...]
    relu: bool
    sparse: bool
    protocol: str
    ready: torch.cuda.Event | None = None

    def wait(self) -> torch.Tensor:
        if self.ready is not None:
            stream = torch.cuda.current_stream(self.scores.device)
            stream.wait_event(self.ready)
            self.threshold.record_stream(stream)
        # Only the local scores participate in autograd. No communication edge
        # has a backward collective and no dense [rows, global_features] exists.
        live = self.scores.reshape(-1, self.scores.shape[-1]).gather(1, self.indices)
        keep = (self.keys >= self.threshold[:, None]) & (self.keys != _MIN_KEY)
        # where, not multiplication, also discards NaN values of ineligible slots.
        values = torch.where(keep, live, 0.0)
        if self.relu:
            values = values.relu()
        width = self.scores.shape[-1]
        if self.sparse:
            return _sparse_from_padded(self.indices, values, self.leading_shape, width)
        dense = self.scores.new_zeros((self.indices.shape[0], width))
        dense.scatter_(1, self.indices, values)
        return dense.reshape(*self.leading_shape, width)


def launch_sharded_topk(
    scores: torch.Tensor,
    k: int,
    group: dist.ProcessGroup | None = None,
    *,
    eligible: torch.Tensor | None = None,
    relu: bool = True,
    sparse: bool = True,
    stream: torch.cuda.Stream | None = None,
    protocol: str = "auto",
    key_backend: str = "torch",
) -> PendingShardedTopK:
    """Submit exact selection for a *uniform-width, feature-sharded* tensor.

    Every TP rank must pass the same leading shape and k, and its own local
    eligibility mask. Callers must obtain k from a global eligible count first.
    Auto selects candidates only when p*min(k, width)*8 <= width*scores.element_size(). No automatic
    fallback to the old full-latent gather is permitted.
    """
    p, rank = _group_size(group), _group_rank(group)
    if key_backend not in ("torch", "triton"):
        raise ValueError("key_backend must be torch or triton")
    if scores.ndim < 2 or scores.shape[-1] <= 0:
        raise ValueError("Expected [..., local_features] with a positive feature width")
    width = scores.shape[-1]
    if scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Sharded exact TopK supports float16/bfloat16/float32 scores")
    if p * width > (1 << 32):
        raise ValueError("Global feature indices must fit in uint32")
    if not 0 < k <= p * width:
        raise ValueError(f"k must be in [1, {p * width}], got {k}")
    if protocol not in ("auto", "candidates", "radix"):
        raise ValueError("protocol must be auto, candidates, or radix")
    nlocal = min(k, width)
    candidate_safe = p * nlocal * 8 <= width * scores.element_size()
    if protocol == "candidates" and not candidate_safe and p > 1:
        raise ValueError(
            "Candidate workspace exceeds one shard; use auto/radix, not full gather"
        )
    chosen = (
        "candidates" if protocol != "radix" and (candidate_safe or p == 1) else "radix"
    )
    leading = tuple(scores.shape[:-1])
    rows = math.prod(leading)
    if eligible is not None:
        if eligible.shape != (width,) or eligible.dtype != torch.bool:
            raise ValueError("eligible must be a local-width boolean feature mask")
        if eligible.device != scores.device:
            raise ValueError("eligible and scores must be on the same device")
    local_keys, indices = _local_candidates(
        scores.reshape(rows, width), nlocal, rank * width, eligible, key_backend
    )

    def select() -> torch.Tensor:
        if rows == 0:
            return torch.empty(0, dtype=torch.int64, device=scores.device)
        if p == 1:
            return local_keys.topk(k, dim=1, sorted=False).values.amin(1)
        if chosen == "radix":
            return _radix_threshold(local_keys, k, group)
        recv = torch.empty((p * rows, nlocal), dtype=torch.int64, device=scores.device)
        dist.all_gather_into_tensor(recv, local_keys.contiguous(), group=group)
        candidates = (
            recv.view(p, rows, nlocal).permute(1, 0, 2).reshape(rows, p * nlocal)
        )
        return candidates.topk(k, dim=1, sorted=False).values.amin(1)

    ready = None
    if stream is not None:
        if not scores.is_cuda:
            raise ValueError("A CUDA stream requires CUDA scores")
        current = torch.cuda.current_stream(scores.device)
        with torch.cuda.stream(stream):
            stream.wait_stream(current)
            local_keys.record_stream(stream)
            with torch.no_grad():
                threshold = select()
            ready = torch.cuda.Event()
            ready.record(stream)
    else:
        with torch.no_grad():
            threshold = select()
    return PendingShardedTopK(
        scores, indices, local_keys, threshold, leading, relu, sparse, chosen, ready
    )


def sharded_topk(
    scores: torch.Tensor, k: int, group: dist.ProcessGroup | None = None, **kwargs
) -> torch.Tensor:
    return launch_sharded_topk(scores, k, group, **kwargs).wait()


@torch.no_grad()
def sharded_firing_counts(
    acts: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    """Replicate only the [global_features] summary, never token-by-feature data."""
    width = acts.shape[-1]
    local = torch.zeros(width, dtype=torch.float32, device=acts.device)
    if acts.is_sparse:
        coo = acts.coalesce()
        local.scatter_add_(0, coo.indices()[-1], (coo.values() != 0).float())
    else:
        local.copy_((acts != 0).reshape(-1, width).float().sum(0))
    p, rank = _group_size(group), _group_rank(group)
    if p == 1:
        return local
    counts = torch.zeros(p * width, dtype=torch.float32, device=acts.device)
    counts.narrow(0, rank * width, width).copy_(local)
    dist.all_reduce(counts, group=group)
    return counts


def feature_counts_from_output(output) -> torch.Tensor:
    """Consume precomputed global summaries without an extra TP collective."""
    summary = getattr(output, "feature_firing_counts", None)
    if summary is not None:
        return summary
    acts = output.feature_acts
    if acts.is_sparse:
        return sharded_firing_counts(acts, None)
    return (acts != 0).reshape(-1, acts.shape[-1]).float().sum(0)


@torch.no_grad()
def gather_feature_summary(
    local: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    """All-gather a one-dimensional feature summary, NEVER an activation batch."""
    if local.ndim != 1:
        raise ValueError("Only one-dimensional feature summaries may be gathered")
    p = _group_size(group)
    if p == 1:
        return local
    output = torch.empty(p * local.numel(), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(output, local.contiguous(), group=group)
    return output
