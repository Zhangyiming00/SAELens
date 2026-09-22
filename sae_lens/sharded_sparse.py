"""Local sparse SAE decoder; never gathers or reconstructs a global latent.

The default backend reuses PyTorch's weighted embedding_bag (dense PARAMETER
weight gradients). The optional Triton backend uses the same three operations
as the OpenAI SAE decoder: sparse-dense, sampled dense-dense, sparse.T-dense.
It is independently implemented for arbitrary decoder strides and ragged nnz.
Encoder autograd remains native and may allocate a *local* dense latent gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flatten_coo(
    acts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not acts.is_sparse or acts.ndim < 2:
        raise TypeError("Expected a COO latent with at least two dimensions")
    acts = acts.coalesce()
    coords = acts.indices()
    rows = math.prod(acts.shape[:-1])
    row = torch.zeros_like(coords[0])
    for axis, width in enumerate(acts.shape[:-1]):
        row = row * width + coords[axis]
    return row, coords[-1], acts.values(), rows


def scale_sparse_features(acts: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Differentiable feature scaling using only nnz values, not [batch, features]."""
    if not acts.is_sparse:
        return acts * scale
    acts = acts.coalesce()
    return torch.sparse_coo_tensor(
        acts.indices(),
        acts.values() * scale[acts.indices()[-1]],
        acts.shape,
        device=acts.device,
        is_coalesced=True,
    )


def sparse_decode(
    acts: torch.Tensor, weight: torch.Tensor, *, backend: str = "torch"
) -> torch.Tensor:
    """Compute local A @ weight.T; weight uses native Megatron [d_in, shard].

    No [batch, k, d_in] embedding expansion is built. The weight may have arbitrary
    strides; torch embedding_bag may make a local weight-layout copy internally.
    No sparse PARAMETER gradient is exposed to DDP/Adam.
    """
    if backend not in ("torch", "triton"):
        raise ValueError("Sparse decoder backend must be torch or triton")
    if acts.shape[-1] != weight.shape[1]:
        raise ValueError("Sparse decoder consumes only this rank's feature shard")
    if acts.device != weight.device:
        raise ValueError("Latents and decoder weights must share a device")
    row, feature, values, rows = flatten_coo(acts)
    device_type = weight.device.type
    dtype = (
        torch.get_autocast_dtype(device_type)
        if torch.is_autocast_enabled(device_type)
        else weight.dtype
    )
    with torch.autocast(device_type=device_type, enabled=False):
        values = values.to(dtype)
        weight = weight.to(dtype)
        counts = torch.zeros(rows, dtype=torch.int64, device=acts.device)
        counts.scatter_add_(0, row, torch.ones_like(row))
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        if backend == "torch":
            # PyTorch 2.10 CUDA embedding_bag lacks BF16 per-sample-weight
            # backward. Preserve BF16 input rounding, then accumulate in FP32.
            # These casts only touch sparse values and the local parameter shard.
            if weight.is_cuda and dtype == torch.bfloat16:
                values = values.float()
                weight = weight.float()
            out = F.embedding_bag(
                feature,
                weight.T,
                offsets=offsets,
                per_sample_weights=values,
                mode="sum",
                include_last_offset=True,
                sparse=False,
            )
        else:
            if not weight.is_cuda:
                raise ValueError(
                    "Triton sparse decoder requires CUDA; use torch on CPU"
                )
            from sae_lens.sharded_triton import triton_sparse_decode

            out = triton_sparse_decode(feature, values, row, offsets, weight)
    return out.to(dtype).reshape(*acts.shape[:-1], weight.shape[0])
