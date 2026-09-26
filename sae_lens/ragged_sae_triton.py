# SPDX-License-Identifier: MIT
# Sparse SAE operation decomposition and weighted-vector accumulation adapted
# from OpenAI sparse_autoencoder/kernels.py (Copyright (c) 2023 OpenAI).
# See licenses/OPENAI_SPARSE_AUTOENCODER_MIT.txt and docs/ragged_sparse_provenance.md.
# Ragged CSR, bounded tiles, large-AuxK sampled dvalues, feature-grouped split
# reduction and histogram inversion are new adaptations, NOT unmodified upstream.
"""CUDA-only ragged SAE kernels; all three decoder operations are sparse.

Workspace is O(E+S) metadata plus the local [S,d] output gradient. No E*d
partial-gradient buffer and no dense sampled-gradient fallback for large K.
All multiply/accumulate operations use FP32 (no implicit TF32/tensor cores).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(V, A, I, P, Y, D: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    start = tl.load(P + row)
    stop = tl.load(P + row + 1)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BD,), tl.float32)
    for base in range(start, stop, BK):
        e = base + ks
        j = tl.load(I + e, e < stop, 0)
        a = tl.load(A + e, e < stop, 0).to(tl.float32)
        w = tl.load(V + j[:, None] * D + d[None, :],
                    (e[:, None] < stop) & (d[None, :] < D), 0).to(tl.float32)
        acc += tl.sum(w * a[:, None], axis=0)
    tl.store(Y + row * D + d, acc, d < D)


@triton.jit
def _value_grad(V, I, R, G, DA, E, D: tl.constexpr,
                BE: tl.constexpr, BD: tl.constexpr):
    e = tl.program_id(0) * BE + tl.arange(0, BE)
    j = tl.load(I + e, e < E, 0)
    row = tl.load(R + e, e < E, 0)
    ds = tl.arange(0, BD)
    acc = tl.zeros((BE, BD), tl.float32)
    # Bounded dimension tiles even for d_in > 4096 and AuxK > 512.
    for start in range(tl.cdiv(D, BD)):
        d = start * BD + ds
        mask = (e[:, None] < E) & (d[None, :] < D)
        w = tl.load(V + j[:, None] * D + d[None, :], mask, 0).to(tl.float32)
        g = tl.load(G + row[:, None] * D + d[None, :], mask, 0).to(tl.float32)
        acc += w * g
    tl.store(DA + e, tl.sum(acc, axis=1), e < E)


@triton.jit
def _weight_grad(A, R, ORDER, FP, G, DW, D: tl.constexpr,
                 BK: tl.constexpr, BD: tl.constexpr, SPLIT: tl.constexpr):
    feature = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    part = tl.program_id(2)
    first = tl.load(FP + feature)
    last = tl.load(FP + feature + 1)
    chunk = tl.cdiv(last - first, SPLIT)
    start = first + part * chunk
    stop = tl.minimum(start + chunk, last)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BD,), tl.float32)
    for base in range(start, stop, BK):
        pos = base + ks
        e = tl.load(ORDER + pos, pos < stop, 0)
        row = tl.load(R + e, pos < stop, 0)
        a = tl.load(A + e, pos < stop, 0).to(tl.float32)
        g = tl.load(G + row[:, None] * D + d[None, :],
                    (pos[:, None] < stop) & (d[None, :] < D), 0).to(tl.float32)
        acc += tl.sum(g * a[:, None], axis=0)
    if SPLIT == 1:
        # Every feature/output element is written, including empty features.
        tl.store(DW + feature * D + d, acc, d < D)
    else:
        # No [split,S,D] buffer: at most SPLIT atomics per weight element.
        tl.atomic_add(DW + feature * D + d, acc, d < D, sem='relaxed')


@triton.jit
def _count_features(I, COUNTS, E, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.load(I + e, e < E, 0)
    tl.atomic_add(COUNTS + j, 1, e < E, sem='relaxed')


@triton.jit
def _fill_order(I, CURSOR, ORDER, E, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.load(I + e, e < E, 0)
    pos = tl.atomic_add(CURSOR + j, 1, e < E, sem='relaxed')
    tl.store(ORDER + pos, e, e < E)


def _check(vectors, values, ids):
    if not vectors.is_cuda or not values.is_cuda:
        raise RuntimeError('ragged SAE Triton requires CUDA')
    if not vectors.is_contiguous() or not values.is_contiguous() or not ids.is_contiguous():
        raise ValueError('Ragged kernel input must be contiguous')
    if vectors.dtype not in (torch.float16, torch.bfloat16, torch.float32) or values.dtype != vectors.dtype:
        raise TypeError('Ragged kernel requires matching FP16/BF16/FP32 values/weights')
    if ids.numel() >= 2**31:
        raise ValueError('Ragged kernel currently limits selected entries to int32 index capacity')


def forward(vectors, values, ids, offsets):
    _check(vectors, values, ids)
    b, d = offsets.numel() - 1, vectors.shape[1]
    out = torch.empty((b, d), dtype=values.dtype, device=values.device)
    if b:
        with torch.cuda.device(values.device):
            with torch.cuda.nvtx.range("sae_sparse:decoder_forward"):
                _forward[(b, triton.cdiv(d, 128))](vectors, values, ids, offsets, out,
                                                d, 16, 128, num_warps=4)
    return out


def inverse_index(ids, width, method='sort'):
    if method == 'sort':
        # Same sparse-transpose grouping strategy as the OpenAI implementation,
        # but use a feature CSR boundary table rather than E*d contributions.
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=width)
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        return order, ptr
    if method != 'histogram':
        raise ValueError('Unknown inverse-index method')
    counts = torch.zeros(width, dtype=torch.int32, device=ids.device)
    if ids.numel():
        _count_features[(triton.cdiv(ids.numel(), 256),)](ids, counts, ids.numel(), 256)
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    order = torch.empty_like(ids, dtype=torch.int32)
    cursor = ptr[:-1].clone()
    if ids.numel():
        _fill_order[(triton.cdiv(ids.numel(), 256),)](ids, cursor, order, ids.numel(), 256)
    return order, ptr


def backward(vectors, values, ids, rows, grad, split=1, index_backend='sort'):
    _check(vectors, values, ids)
    grad = grad.contiguous()
    s, d = vectors.shape
    dv = torch.empty_like(values)
    # Mixed precision accumulates weight gradients in FP32, one LOCAL weight.
    dw = (torch.empty((s, d), device=vectors.device, dtype=torch.float32) if split == 1
          else torch.zeros((s, d), device=vectors.device, dtype=torch.float32))
    with torch.cuda.device(values.device):
        with torch.cuda.nvtx.range("sae_sparse:feature_inverse_index"):
            order, ptr = inverse_index(ids, s, index_backend)
        if ids.numel():
            with torch.cuda.nvtx.range("sae_sparse:value_gradient"):
                _value_grad[(triton.cdiv(ids.numel(), 16),)](vectors, ids, rows, grad, dv,
                                                          ids.numel(), d, 16, 128, num_warps=4)
        if s:
            with torch.cuda.nvtx.range("sae_sparse:weight_gradient"):
                _weight_grad[(s, triton.cdiv(d, 128), split)](values, rows, order, ptr,
                                                           grad, dw, d, 16, 128, split, num_warps=4)
    return dw.to(vectors.dtype), dv


def value_backward(vectors, values, ids, rows, grad):
    """V5 independently callable dvalues; same V3 arithmetic, no inverse index."""
    _check(vectors, values, ids)
    grad = grad.contiguous()
    out = torch.empty_like(values)
    if ids.numel():
        with torch.cuda.device(values.device):
            _value_grad[(triton.cdiv(ids.numel(), 16),)](
                vectors, ids, rows, grad, out, ids.numel(), vectors.shape[1],
                16, 128, num_warps=4)
    return out


def weight_backward(vectors, values, ids, rows, grad, split=1, index_backend='sort'):
    """V5 independently callable dweight; does not waste a sampled-gradient call."""
    _check(vectors, values, ids)
    if torch.are_deterministic_algorithms_enabled() and (split != 1 or index_backend != 'sort'):
        raise RuntimeError('Deterministic sparse wgrad requires split=1,index_backend=sort')
    s, d = vectors.shape
    grad = grad.contiguous()
    dw = (torch.empty((s, d), device=vectors.device, dtype=torch.float32) if split == 1
          else torch.zeros((s, d), device=vectors.device, dtype=torch.float32))
    with torch.cuda.device(values.device):
        order, ptr = inverse_index(ids, s, index_backend)
        if s:
            _weight_grad[(s, triton.cdiv(d, 128), split)](
                values, rows, order, ptr, grad, dw, d, 16, 128, split, num_warps=4)
    return dw.to(vectors.dtype)
