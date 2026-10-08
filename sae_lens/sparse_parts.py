"""Independent decoder phases reusing V4's actual sparse arithmetic kernels.

No TP/DP calls and no implicit fallback to another provider. CPU reference is
explicit. This module allows sparse forward, dense dvalues, sparse dweight, etc.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _openai_inputs(vectors, values, grad=None):
    from sae_lens.openai_sae_adapter import require_upstream, _working_vectors
    require_upstream(vectors.device)
    vv, aa = _working_vectors(vectors), values.float().contiguous()
    if grad is None:
        return vv, aa
    gg = grad.float()
    if gg.shape[1] != vv.shape[1]:
        gg = F.pad(gg, (0, vv.shape[1]-gg.shape[1]))
    return vv, aa, gg.contiguous()


def sparse_forward(vectors, values, ids, rows, offsets, opts):
    engine = opts['engine']
    if engine == 'openai':
        from sae_lens import openai_sae_adapter as a
        vv, aa = _openai_inputs(vectors, values)
        if opts['forward_mode'] == 'coo':
            out = a._coo_multiply(ids, rows, aa, vv, offsets.numel()-1)
            return out[:, :vectors.shape[1]].to(values.dtype).contiguous(), None
        groups = a.page_groups(offsets, vectors.shape[0], opts['page_k'])
        out = vv.new_zeros((offsets.numel()-1, vv.shape[1]))
        for group in groups:
            tile = a._row_tile(vv.shape[1], group.k, opts['openai_workspace_mib'])
            for begin in range(0, group.rows.numel(), tile):
                rr = group.rows[begin:begin+tile]
                ii, av, _, _ = a._page(offsets, ids, aa, group, rr)
                out.index_add_(0, rr, a._forward_page(vv, ii, av))
        # Removing power-of-two padding leaves a strided view (e.g. d_in=5120
        # padded to 8192). Megatron's reduce mapping reduces input.contiguous()
        # but returns input, so a strided result silently loses the TP sum.
        return out[:, :vectors.shape[1]].to(values.dtype).contiguous(), groups
    if engine == 'triton':
        from sae_lens.ragged_sae_triton import forward
        return forward(vectors, values, ids, offsets), None
    if engine == 'torch_reference':
        from sae_lens.ragged_sae import reference_forward
        return reference_forward(vectors, values, ids, rows, offsets.numel()-1), None
    raise ValueError('Unknown sparse provider ' + engine)


def sparse_dvalues(vectors, values, ids, rows, offsets, grad, opts, groups=None):
    engine = opts['engine']
    if engine == 'openai':
        from sae_lens import openai_sae_adapter as a
        vv, aa, gg = _openai_inputs(vectors, values, grad)
        if groups is None:
            groups = a.page_groups(offsets, vectors.shape[0], opts['page_k'])
        result = gg.new_empty(values.numel()+1)
        for group in groups:
            tile = a._row_tile(vv.shape[1], group.k, opts['openai_workspace_mib'])
            for begin in range(0, group.rows.numel(), tile):
                rr = group.rows[begin:begin+tile]
                ii, _, positions, valid = a._page(offsets, ids, aa, group, rr)
                dv = a._value_page(vv, gg.index_select(0, rr), ii)
                a.restore_page_gradient(result, positions, valid, dv)
        return result[:-1].to(values.dtype)
    if engine == 'triton':
        from sae_lens.ragged_sae_triton import value_backward
        return value_backward(vectors, values, ids, rows, grad)
    if engine == 'torch_reference':
        out = torch.empty_like(values)
        dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
        chunk = max(1, (4 << 20)//max(1, vectors.shape[1]*vectors.element_size()*3))
        for start in range(0, values.numel(), chunk):
            sl = slice(start, start+chunk)
            out[sl] = (grad.index_select(0, rows[sl]).to(dtype)
                       * vectors.index_select(0, ids[sl]).to(dtype)).sum(1).to(values.dtype)
        return out
    raise ValueError('Unknown sparse provider ' + engine)


def sparse_dweight(vectors, values, ids, rows, grad, opts):
    engine = opts['engine']
    if engine == 'openai':
        from sae_lens import openai_sae_adapter as a
        vv, aa, gg = _openai_inputs(vectors, values, grad)
        order = torch.argsort(ids, stable=True)
        result = a._coo_multiply(rows.index_select(0, order), ids.index_select(0, order),
                                 aa.index_select(0, order), gg, vectors.shape[0])
        return result[:, :vectors.shape[1]].contiguous().to(vectors.dtype)
    if engine == 'triton':
        from sae_lens.ragged_sae_triton import weight_backward
        return weight_backward(vectors, values, ids, rows, grad, opts['split'], opts['index_backend'])
    if engine == 'torch_reference':
        dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
        out = torch.zeros(vectors.shape, dtype=dtype, device=vectors.device)
        chunk = max(1, (4 << 20)//max(1, vectors.shape[1]*vectors.element_size()*3))
        for start in range(0, values.numel(), chunk):
            sl = slice(start, start+chunk)
            out.index_add_(0, ids[sl], grad.index_select(0, rows[sl]).to(dtype)*values[sl,None].to(dtype))
        return out.to(vectors.dtype)
    raise ValueError('Unknown sparse provider ' + engine)
