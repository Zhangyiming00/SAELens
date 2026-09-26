"""Host adapters to the ACTUAL OpenAI SAE Triton kernels, not new GPU kernels.

Only packing, launch configuration, autograd and local gradient routing live
here. All three arithmetic kernels are in vendor/openai_sae/kernels.py. We do
NOT use the upstream large-K Python wrapper (it contains a dense fallback).

Rows with different lengths are bucketed locally. Long rows are paged at K<=512.
This can perform fewer than 2x real entries in sampled-gradient because a final
page may be padded to a power of two; it never pads every rank to the global K.
Selected zero-valued entries remain differentiable; only padding is discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F


_KERNEL_NAMES = (
    'triton_sparse_dense_matmul_kernel',
    'triton_dense_dense_sparseout_matmul_kernel',
    'triton_sparse_transpose_dense_matmul_kernel',
)


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


@lru_cache(None)
def upstream_identity() -> dict:
    """Content-pinned local snapshot, NOT a claim about an unknown remote SHA."""
    root = Path(__file__).resolve().parent / 'vendor' / 'openai_sae'
    info = json.loads((root / 'SOURCE.json').read_text())
    actual = hashlib.sha256((root / 'kernels.py').read_bytes()).hexdigest()
    if actual != info['bundled_sha256']:
        raise RuntimeError('Vendored OpenAI kernels changed; re-audit provenance before running')
    return {'provider': 'openai/sparse_autoencoder', 'snapshot_sha256': actual,
            'remote_commit': info['remote_commit'], 'kernels': list(_KERNEL_NAMES)}


@lru_cache(None)
def _kernels():
    upstream_identity()
    try:
        from sae_lens.vendor.openai_sae import kernels
    except ImportError as exc:
        raise RuntimeError('OpenAI SAE engine needs Triton; no automatic custom/dense fallback') from exc
    return kernels


def require_upstream(device: torch.device) -> None:
    if device.type != 'cuda':
        raise RuntimeError('openai engine requires CUDA; torch_reference is a separate explicit engine')
    if torch.are_deterministic_algorithms_enabled():
        text = 'Unmodified OpenAI SAE weight-gradient uses atomic_add; bitwise determinism is unsupported'
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(text, RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(text)
    _kernels()


@contextmanager
def _range(label: str, device: torch.device):
    if device.type == 'cuda':
        with torch.cuda.nvtx.range('sae_upstream:openai:' + label):
            yield
    else:  # CPU adapter tests inject kernel spies; never a production fallback.
        yield


@dataclass(frozen=True)
class PageGroup:
    rows: torch.Tensor
    base: int
    k: int


def page_groups(offsets: torch.Tensor, width: int, page_k: int = 512) -> list[PageGroup]:
    """Bounded-K launch groups. May synchronize CUDA to discover local lengths.

    Each true entry appears exactly once. Metadata is O(B+E/page_k), while a
    page has at most page_k<=local_width slots. No global feature array exists.
    """
    if offsets.ndim != 1 or offsets.dtype != torch.int64 or offsets.numel() < 1:
        raise ValueError('Expected int64 CSR row offsets')
    if width < 1 or page_k not in (32, 64, 128, 256, 512):
        raise ValueError('page_k must be 32/64/128/256/512 and local width positive')
    cap = 1 << (min(width, page_k).bit_length() - 1)
    # One structural D2H transfer; no per-bucket nonzero/.item synchronization.
    # This is metadata, not activation/gradient values. Never cache across steps.
    lengths = offsets.diff().detach().cpu().tolist()
    groups = {}
    for row, length in enumerate(lengths):
        for base in range(0, length, cap):
            k = _pow2(min(length-base, cap))
            groups.setdefault((base, k), []).append(row)
    keys = sorted(groups)
    flat = [row for key in keys for row in groups[key]]
    device_rows = torch.tensor(flat, device=offsets.device, dtype=torch.int64)
    result, start = [], 0
    for base, k in keys:
        length = len(groups[(base, k)])
        result.append(PageGroup(device_rows[start:start+length], base, k))
        start += length
    return result


def restore_page_gradient(destination, positions, valid, page_grad):
    """Write fixed-size pages without boolean indexing/nonzero synchronization.

    Real destinations are unique; all padded lanes write zero to a disposable
    sink slot. Duplicate writes only target that sink, never a real derivative.
    destination must have E+1 slots; caller returns the first E.
    """
    sink = destination.numel()-1
    targets = torch.where(valid, positions, sink).reshape(-1)
    source = torch.where(valid, page_grad, 0.0).reshape(-1)
    destination.scatter_(0, targets, source)


def _row_tile(d: int, k: int, workspace_mib: int) -> int:
    if not isinstance(workspace_mib, int) or not 1 <= workspace_mib <= 4096:
        raise ValueError('openai_workspace_mib must be an integer in [1,4096]')
    # Conservative budget for adapter-controlled temporary tensors, NOT allocator
    # peak or the upstream register/sort workspace. No E*d materialization.
    row_bytes = 4 * d * 4 + 80 * k + 64
    size = (workspace_mib << 20) // row_bytes
    if size < 1:
        raise ValueError('Workspace budget does not fit one page row')
    return size


def _page(offsets, ids, values, group, rows):
    start = offsets.index_select(0, rows) + group.base
    stop = offsets.index_select(0, rows + 1)
    positions = start[:, None] + torch.arange(group.k, device=rows.device)
    valid = positions < stop[:, None]
    safe = positions.clamp(max=max(values.numel() - 1, 0))
    page_ids = torch.where(valid, ids[safe], 0).contiguous()
    page_values = torch.where(valid, values[safe], 0.0).contiguous()
    return page_ids, page_values, positions, valid


def _working_vectors(vectors):
    # Exact FP16/BF16 values are promoted to FP32; no TF32 or hidden low-precision
    # arithmetic. Padding prevents relying on undefined masked lanes in upstream.
    d = _pow2(vectors.shape[1])
    value = vectors.float()
    if d != vectors.shape[1]:
        value = F.pad(value, (0, d - vectors.shape[1]))
    return value.contiguous()


def _forward_page(vectors, ids, values):
    """Launch the unmodified upstream FORWARD body directly."""
    n, d = vectors.shape
    b, k = ids.shape
    out = vectors.new_empty((b, d))
    _kernels().triton_sparse_dense_matmul_kernel[(b,)](
        ids, values, vectors, out,
        stride_dn=vectors.stride(0), stride_db=vectors.stride(1),
        A=b, B=d, N=n, K=k, BLOCK_SIZE_K=_pow2(k), BLOCK_SIZE_B=_pow2(d),
    )
    return out


def _value_page(vectors, grad, ids):
    """Direct launch: never invokes upstream K>512 dense-dispatch wrapper."""
    n, d = vectors.shape
    b, k = ids.shape
    if k > 512:
        raise RuntimeError('Internal paging invariant violated; dense fallback is forbidden')
    out = grad.new_empty((b, k))
    _kernels().triton_dense_dense_sparseout_matmul_kernel[(b,)](
        grad, vectors.T, ids, out,
        stride_d1a=grad.stride(0), stride_d1b=grad.stride(1),
        stride_d2b=vectors.stride(1), stride_d2n=vectors.stride(0),
        A=b, B=d, N=n, K=k, BLOCK_SIZE_B=_pow2(d),
        BLOCK_SIZE_N=_pow2(n), BLOCK_SIZE_K=_pow2(k),
    )
    return out


def _coo_multiply(source_rows, output_rows, values, dense, nout, block=128):
    """Unmodified upstream COO sparse-transpose kernel; sorted output rows.

    Tail padding has value zero and repeats the last valid output index. It
    prevents uninitialized masked COO lanes from affecting upstream tl.min.
    No contributor vector is expanded to [E,d]. Final dW remains dense/local.
    """
    out = dense.new_zeros((nout, dense.shape[1]))
    e = values.numel()
    if not e:
        return out
    padded = ((e + block - 1) // block) * block
    coo = torch.empty((2, padded), dtype=torch.int64, device=values.device)
    coo[0, :e].copy_(source_rows)
    coo[1, :e].copy_(output_rows)
    vv = values.new_zeros(padded)
    vv[:e].copy_(values)
    if padded != e:
        coo[0, e:].zero_()
        coo[1, e:].copy_(output_rows[-1].expand(padded - e))
    _kernels().triton_sparse_transpose_dense_matmul_kernel[(padded // block, 1)](
        coo, vv, dense, out,
        stride_da=dense.stride(0), stride_db=dense.stride(1),
        B=dense.shape[1], N=nout, AK=padded,
        BLOCK_SIZE_AK=block, BLOCK_SIZE_B=_pow2(dense.shape[1]),
    )
    return out


class _OpenAISAE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vectors, values, ids, rows, offsets, page_k, workspace_mib, forward_mode):
        require_upstream(vectors.device)
        if vectors.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError('openai engine supports FP32/FP16/BF16, not FP64')
        if forward_mode not in ('bucketed', 'coo'):
            raise ValueError('openai_forward must be bucketed or coo')
        groups = page_groups(offsets, vectors.shape[0], page_k)
        vv, aa = _working_vectors(vectors), values.float().contiguous()
        b, d = offsets.numel() - 1, vv.shape[1]
        _row_tile(d, min(page_k, vectors.shape[0]), workspace_mib)
        with _range('decoder_forward', vectors.device):
            if forward_mode == 'coo':
                # Same original upstream sparse-transpose operator, with roles
                # exchanged: output rows are tokens and source rows are features.
                out = _coo_multiply(ids, rows, aa, vv, b)
            else:
                out = vv.new_zeros((b, d))
                for group in groups:
                    tile = _row_tile(d, group.k, workspace_mib)
                    for begin in range(0, group.rows.numel(), tile):
                        rr = group.rows[begin:begin + tile]
                        ii, av, _, _ = _page(offsets, ids, aa, group, rr)
                        partial = _forward_page(vv, ii, av)
                        out.index_add_(0, rr, partial)
        ctx.groups, ctx.workspace_mib = groups, workspace_mib
        ctx.input_d = vectors.shape[1]
        ctx.vector_dtype, ctx.value_dtype = vectors.dtype, values.dtype
        ctx.save_for_backward(vv, aa, ids, rows, offsets)
        return out[:, :ctx.input_d].contiguous().to(values.dtype)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError('Unmodified upstream GPU kernels support first-order backward only')
        vectors, values, ids, rows, offsets = ctx.saved_tensors
        grad = grad.float()
        if grad.shape[1] != vectors.shape[1]:
            grad = F.pad(grad, (0, vectors.shape[1] - grad.shape[1]))
        grad = grad.contiguous()
        dv = dw = None
        if ctx.needs_input_grad[1]:
            dv_with_sink = values.new_empty(values.numel()+1)
            dv = dv_with_sink[:-1]
            with _range('sampled_value_gradient', vectors.device):
                for group in ctx.groups:
                    tile = _row_tile(vectors.shape[1], group.k, ctx.workspace_mib)
                    for begin in range(0, group.rows.numel(), tile):
                        rr = group.rows[begin:begin + tile]
                        ii, _, positions, valid = _page(offsets, ids, values, group, rr)
                        page_grad = _value_page(vectors, grad.index_select(0, rr), ii)
                        restore_page_gradient(dv_with_sink, positions, valid, page_grad)
            dv = dv.to(ctx.value_dtype)
        if ctx.needs_input_grad[0]:
            with _range('feature_sort', vectors.device):
                order = torch.argsort(ids, stable=True)
                src = rows.index_select(0, order)
                dst = ids.index_select(0, order)
                av = values.index_select(0, order)
            with _range('weight_gradient', vectors.device):
                dw = _coo_multiply(src, dst, av, grad, vectors.shape[0])
            dw = dw[:, :ctx.input_d].contiguous().to(ctx.vector_dtype)
        return dw, dv, None, None, None, None, None, None


def decode_openai(vectors, values, ids, rows, offsets, *, page_k=512,
                  workspace_mib=64, forward_mode='bucketed'):
    """Autograd boundary around original OpenAI arithmetic kernels.

    Callers provide validated LOCAL ragged metadata; no TP group is consulted.
    All reconstructions reduce only in the existing model/runtime layer.
    """
    return _OpenAISAE.apply(vectors, values, ids, rows, offsets,
                           page_k, workspace_mib, forward_mode)
