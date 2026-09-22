"""Optional, independently implemented sparse decoder kernels.

Algorithmic references (not vendored code): OpenAI sparse_autoencoder/kernels.py
and the weighted embedding-bag approach used in EleutherAI sparsify. This version
accepts Megatron's actual strides; it never transposes/copies the full weight in
forward, and never expands embeddings to [batch, k, d_in]. Triton is optional.

CUDA validation is required before using this experimental backend in production.
The torch backend in sharded_sparse.py is the numerical reference.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        "Optional sharded kernels require Triton; select --sae-sparse-decoder torch"
    ) from exc


@triton.jit
def _score_keys_kernel(
    S,
    O,
    N: tl.constexpr,
    F: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // F, idx % F
    value = tl.load(S + row * S0 + col * S1, idx < N, other=0).to(tl.float32)
    value = tl.where(value == 0, 0.0, value)
    bits = value.to(tl.int32, bitcast=True).to(tl.int64)
    bits = tl.where(value != value, 0x7FFFFFFF, bits)
    ordered = tl.where(bits < 0, ~bits, bits ^ 0x80000000) & 0xFFFFFFFF
    high = (ordered - 0x80000000) << 32
    key = high | (0xFFFFFFFF - (OFFSET + col.to(tl.int64)))
    tl.store(O + idx, key, idx < N)


def make_score_keys(scores: torch.Tensor, offset: int) -> torch.Tensor:
    if not scores.is_cuda:
        raise ValueError("Triton score keys require CUDA")
    if scores.ndim != 2 or scores.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError("Expected two-dimensional FP32/FP16/BF16 scores")
    if not 0 <= offset < (1 << 32) or offset + scores.shape[1] > (1 << 32):
        raise ValueError("Global feature indices must fit in uint32")
    out = torch.empty(scores.shape, device=scores.device, dtype=torch.int64)
    if scores.numel():
        _score_keys_kernel[(triton.cdiv(scores.numel(), 256),)](
            scores,
            out,
            scores.numel(),
            scores.shape[1],
            *scores.stride(),
            offset,
            256,
        )
    return out


@triton.jit
def _decode_kernel(
    I,
    V,
    P,
    W,
    Y,
    D: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d = (tl.program_id(1) * BD + tl.arange(0, BD)).to(tl.int64)
    lo, hi = tl.load(P + row), tl.load(P + row + 1)
    acc = tl.zeros((BD,), tl.float32)
    for start in range(lo, hi, BK):
        pos = start + tl.arange(0, BK)
        feature = tl.load(I + pos, pos < hi, other=0)
        value = tl.load(V + pos, pos < hi, other=0).to(tl.float32)
        weight = tl.load(
            W + feature[:, None] * W1 + d[None, :] * W0,
            (pos[:, None] < hi) & (d[None, :] < D) & (value[:, None] != 0),
            other=0,
        ).to(tl.float32)
        acc += tl.sum(value[:, None] * weight, axis=0)
    tl.store(Y + row * D + d, acc, d < D)


@triton.jit
def _value_grad_kernel(
    I,
    R,
    W,
    G,
    DV,
    E: tl.constexpr,
    D: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    BE: tl.constexpr,
    BD: tl.constexpr,
):
    edge = tl.program_id(0) * BE + tl.arange(0, BE)
    feature = tl.load(I + edge, edge < E, other=0)
    row = tl.load(R + edge, edge < E, other=0)
    acc = tl.zeros((BE,), tl.float32)
    for start in range(0, D, BD):
        d = (start + tl.arange(0, BD)).to(tl.int64)
        mask = (edge[:, None] < E) & (d[None, :] < D)
        g = tl.load(G + row[:, None] * D + d[None, :], mask, other=0).to(tl.float32)
        w = tl.load(W + feature[:, None] * W1 + d[None, :] * W0, mask, other=0).to(
            tl.float32
        )
        acc += tl.sum(g * w, axis=1)
    tl.store(DV + edge, acc, edge < E)


@triton.jit
def _weight_grad_kernel(
    V, R, PERM, P, G, DW, D: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr
):
    # Group contributions by feature, avoiding atomic writes into dense weights.
    feature = tl.program_id(0).to(tl.int64)
    d = (tl.program_id(1) * BD + tl.arange(0, BD)).to(tl.int64)
    lo, hi = tl.load(P + feature), tl.load(P + feature + 1)
    acc = tl.zeros((BD,), tl.float32)
    for start in range(lo, hi, BE):
        pos = start + tl.arange(0, BE)
        edge = tl.load(PERM + pos, pos < hi, other=0)
        row = tl.load(R + edge, pos < hi, other=0)
        value = tl.load(V + edge, pos < hi, other=0).to(tl.float32)
        g = tl.load(
            G + row[:, None] * D + d[None, :],
            (pos[:, None] < hi) & (d[None, :] < D) & (value[:, None] != 0),
            other=0,
        ).to(tl.float32)
        acc += tl.sum(value[:, None] * g, axis=0)
    # Transposed gradient storage makes writes contiguous; autograd/DDP accepts
    # the returned dense view and applies its normal parameter-layout contract.
    tl.store(DW + feature * D + d, acc, d < D)


class _SparseDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, row, offsets, weight):
        rows, d = offsets.numel() - 1, weight.shape[0]
        out = torch.empty((rows, d), dtype=values.dtype, device=values.device)
        if rows:
            _decode_kernel[(rows, triton.cdiv(d, 64))](
                indices,
                values,
                offsets,
                weight,
                out,
                d,
                *weight.stride(),
                64,
                32,
            )
        ctx.save_for_backward(indices, values, row, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        indices, values, row, weight = ctx.saved_tensors
        grad = grad_output.contiguous()
        d, width = weight.shape
        grad_values = torch.empty_like(values)
        if values.numel():
            _value_grad_kernel[(triton.cdiv(values.numel(), 32),)](
                indices,
                row,
                weight,
                grad,
                grad_values,
                values.numel(),
                d,
                *weight.stride(),
                32,
                64,
            )
        counts = torch.zeros(width, dtype=torch.int64, device=weight.device)
        counts.scatter_add_(0, indices, torch.ones_like(indices))
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        permutation = indices.argsort()
        grad_weight = torch.empty((width, d), dtype=weight.dtype, device=weight.device)
        _weight_grad_kernel[(width, triton.cdiv(d, 64))](
            values,
            row,
            permutation,
            offsets,
            grad,
            grad_weight,
            d,
            64,
            32,
        )
        return None, grad_values, None, None, grad_weight.T


def triton_sparse_decode(indices, values, row, offsets, weight):
    if not weight.is_cuda or weight.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            "Triton decoder requires CUDA float32/float16/bfloat16 weights"
        )
    return _SparseDecode.apply(indices, values, row, offsets, weight)
