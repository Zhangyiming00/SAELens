"""Packed, local-only SAE decoder: sparse forward, sampled dvalues, sparse dW.

No global latent, no E*d_in workspace, no K-dependent dense fallback. The
explicit torch_reference engine is for numerical tests (chunk-bounded indexed
arithmetic, not a performant GPU substitute). CUDA uses ragged_sae_triton.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _EntryValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, rows, columns):
        ctx.shape = tuple(scores.shape)
        ctx.save_for_backward(rows, columns)
        return scores.reshape(-1, scores.shape[-1])[rows, columns]

    @staticmethod
    def backward(ctx, grad):
        rows, columns = ctx.saved_tensors
        # Unique (row,column) from selector. index_put(accumulate=True) also
        # supports explicitly supplied repeated entries in the standalone API.
        out = grad.new_zeros((math.prod(ctx.shape[:-1]), ctx.shape[-1]))
        out.index_put_((rows, columns), grad, accumulate=True)
        return out.reshape(ctx.shape), None, None


@dataclass(frozen=True)
class RaggedLatents:
    """Row-major selected entries; zeros are valid, padding is NOT an entry.

    feature_ids are ORIGINAL local feature IDs. row_offsets[-1] == values.numel().
    Construction by the selectors guarantees sorted rows, in-range unique IDs
    within each row. No tensor-dependent host validation on every training step.
    """
    row_offsets: torch.Tensor
    row_ids: torch.Tensor
    feature_ids: torch.Tensor
    values: torch.Tensor
    leading_shape: tuple[int, ...]
    width: int
    selection: str = 'topk'

    @property
    def shape(self): return (*self.leading_shape, self.width)
    @property
    def ndim(self): return len(self.shape)
    @property
    def device(self): return self.values.device
    @property
    def dtype(self): return self.values.dtype
    @property
    def is_sparse(self): return True
    @property
    def nnz(self): return self.values.numel()
    @property
    def nrows(self): return math.prod(self.leading_shape)

    def detach(self):
        return replace(self, values=self.values.detach())

    def scale(self, local_scale):
        if local_scale.shape != (self.width,):
            raise ValueError('Feature scale must have LOCAL width')
        return replace(self, values=self.values * local_scale.index_select(0, self.feature_ids))

    def to(self, *args, **kwargs):
        values = self.values.to(*args, **kwargs)
        return replace(self, values=values,
                       row_offsets=self.row_offsets.to(values.device),
                       row_ids=self.row_ids.to(values.device), feature_ids=self.feature_ids.to(values.device))

    def to_dense(self):
        """Explicit LOCAL materialization, used only by diagnostic/eval backends."""
        out = self.values.new_zeros((self.nrows, self.width))
        out.index_put_((self.row_ids, self.feature_ids), self.values, accumulate=True)
        return out.reshape(self.shape)

    def validate(self):
        """Expensive debug validation. May synchronize CUDA. Not used per step."""
        n = self.nnz
        if self.width <= 0 or self.values.ndim != 1:
            raise ValueError('Invalid width/values')
        if self.row_offsets.shape != (self.nrows + 1,) or self.row_ids.shape != (n,) or self.feature_ids.shape != (n,):
            raise ValueError('Invalid packed shapes')
        for t in [self.row_offsets, self.row_ids, self.feature_ids]:
            if t.dtype != torch.int64 or t.device != self.device:
                raise ValueError('Metadata must be device-local int64')
        if int(self.row_offsets[0]) != 0 or int(self.row_offsets[-1]) != n:
            raise ValueError('Invalid CSR endpoints')
        if bool((self.row_offsets[1:] < self.row_offsets[:-1]).any()):
            raise ValueError('Offsets must be monotone')
        if n:
            if int(self.feature_ids.min()) < 0 or int(self.feature_ids.max()) >= self.width:
                raise ValueError('Feature outside owner shard')
            expected = torch.repeat_interleave(torch.arange(self.nrows, device=self.device),
                                              self.row_offsets.diff())
            if not torch.equal(expected, self.row_ids):
                raise ValueError('Entries must be row-major')
        return self


@dataclass(frozen=True)
class SelectedEntries:
    offsets: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    width: int
    selection: str


def represent_latents(acts, kind, group, *, feature_shard=None):
    """Return an actual dense/full tensor or a local RaggedLatents value."""
    from sae_lens.megatron_tp import megatron_tp_allgather

    if kind == "sharded_ragged":
        if not isinstance(acts, RaggedLatents):
            raise TypeError("Ragged representation requires selected entries")
        return acts
    metadata = None
    if isinstance(acts, RaggedLatents):
        metadata = SelectedEntries(
            acts.row_offsets, acts.row_ids, acts.feature_ids, acts.width, acts.selection
        )
        dense = acts.to_dense()
    else:
        dense = acts
    if kind == "full":
        dense = megatron_tp_allgather(dense, group, feature_shard=feature_shard)
    if metadata is not None:
        dense._sae_selected_entries = metadata
    return dense


def local_latent_tensor(acts, kind, rank, width, *, feature_shard=None):
    if kind == "full" and feature_shard is not None:
        return feature_shard.select(acts)
    return acts.narrow(-1, rank * width, width) if kind == "full" else acts


def selected_latents_view(acts, kind, rank, width, *, feature_shard=None):
    """Sparse computation reads dense storage through saved winner indices.

    Do not recover winners with nonzero(): AuxK may select zero-valued entries
    whose derivative must still be computed.
    """

    if isinstance(acts, RaggedLatents):
        return acts
    meta = getattr(acts, "_sae_selected_entries", None)
    if meta is None:
        raise ValueError(
            "Sparse/compact Main decode requires selection metadata from encode(); "
            "use dense computation for arbitrary externally supplied activations"
        )
    local = local_latent_tensor(acts, kind, rank, width, feature_shard=feature_shard)
    values = _EntryValues.apply(local, meta.rows, meta.columns)
    return RaggedLatents(
        meta.offsets,
        meta.rows,
        meta.columns,
        values,
        tuple(local.shape[:-1]),
        width,
        meta.selection,
    )



def _offsets(rows, nrows):
    counts = torch.bincount(rows, minlength=nrows)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0)))


def from_candidates(scores, indices, keep, *, relu=False, selection='topk'):
    """Pack by winner mask, never values!=0. nonzero may synchronize CUDA.

    Candidates and mask may be padded; only true winners become decoder work.
    Packing uses O(B*local_candidate_count) control storage, never global width.
    """
    leading, width = tuple(scores.shape[:-1]), scores.shape[-1]
    rows, slots = indices.shape
    if rows != math.prod(leading) or keep.shape != indices.shape or keep.dtype != torch.bool:
        raise ValueError('Candidate/mask shape mismatch')
    positions = keep.reshape(-1).nonzero(as_tuple=True)[0]
    row_ids = positions // max(1, slots)
    ids = indices.reshape(-1).index_select(0, positions)
    vals = _EntryValues.apply(scores, row_ids, ids)
    if relu: vals = vals.relu()
    return RaggedLatents(_offsets(row_ids, rows), row_ids, ids, vals, leading, width, selection)


def from_dead_columns(scores, columns, keep=None, *, selection='select_all'):
    rows = math.prod(scores.shape[:-1])
    if keep is None:
        # All eligible selected: avoid allocating a B*m bool mask.
        row_ids = torch.arange(rows, device=scores.device).repeat_interleave(columns.numel())
        ids = columns.repeat(rows)
        vals = _EntryValues.apply(scores, row_ids, ids)
        offsets = torch.arange(rows + 1, device=scores.device, dtype=torch.int64) * columns.numel()
        return RaggedLatents(offsets, row_ids, ids, vals,
                             tuple(scores.shape[:-1]), scores.shape[-1], selection)
    return from_candidates(scores, columns.expand(rows, -1), keep, selection=selection)


def launch_ragged_topk(scores, k, group=None, **kwargs):
    """Use the existing exact comparison protocol; change only its result API."""
    from sae_lens.sharded_topk import launch_sharded_topk
    return launch_sharded_topk(scores, k, group, packed=True, **kwargs)


def ragged_auxk(scores, k, eligible, num_eligible, group=None, *, policy='auto',
                protocol='auto', key_backend='torch', complement='auto', tie_policy='stable_id',
                known_columns=None, feature_shard=None):
    from sae_lens.auxk_compact import plan_auxk_dense, _DeadColumnValues, _complement_mask
    from sae_lens.sharded_topk import _group_size, _group_rank
    if eligible.dtype != torch.bool or eligible.shape != (scores.shape[-1],) or eligible.device != scores.device:
        raise ValueError('AuxK requires a LOCAL boolean eligible mask')
    if tie_policy == 'torch_tp1':
        if _group_size(group) != 1:
            raise ValueError('torch_tp1 requires singleton TP')
        if k < num_eligible:
            return launch_ragged_topk(scores, k, group, eligible=eligible, relu=False,
                                     protocol=protocol, key_backend=key_backend,
                                     tie_policy=tie_policy, known_columns=known_columns, feature_shard=feature_shard).wait()
    plan = plan_auxk_dense(num_dead=num_eligible, k=k, shard_width=scores.shape[-1],
                          tp_size=_group_size(group), element_size=scores.element_size(),
                          decoder='auto', complement=complement, selection_policy=policy,
                          protocol=protocol, shard_widths=None if feature_shard is None else feature_shard.widths)
    if plan.selection == 'topk':
        result = launch_ragged_topk(scores, k, group, eligible=eligible, relu=False,
                                   protocol=protocol, key_backend=key_backend,
                                   compact_radix=(policy == 'auto'), known_columns=known_columns, feature_shard=feature_shard).wait()
        return replace(result, selection='topk')
    columns = eligible.nonzero(as_tuple=True)[0] if known_columns is None else known_columns
    if plan.selection == 'select_all':
        return from_dead_columns(scores, columns)
    # Comparison values are detached; autograd values are later gathered only
    # for actual winners from the original local scores.
    with torch.no_grad():
        packed = scores.reshape(-1, scores.shape[-1]).index_select(1, columns)
        ids = columns + _group_rank(group) * scores.shape[-1] if feature_shard is None else feature_shard.ids(scores.device)[columns]
        keep = _complement_mask(packed, ids, plan.exclude_k, group)
    return from_dead_columns(scores, columns, keep, selection=plan.selection)


def reference_forward(vectors, values, feature_ids, row_ids, nrows, chunk_bytes=4 << 20):
    """Explicit sparse indexed arithmetic; chunk workspace <=~chunk_bytes.

    Not torch.sparse.mm/embedding_bag, and never a full [B,S] matrix multiply.
    CPU oracle also supports FP64 gradcheck. CUDA reference is diagnostic only.
    """
    d = vectors.shape[1]
    accum = torch.float64 if vectors.dtype == torch.float64 else torch.float32
    v, a = vectors.to(accum), values.to(accum)
    out = torch.zeros((nrows, d), dtype=accum, device=values.device)
    chunk = max(1, chunk_bytes // max(1, d * torch.empty((), dtype=accum).element_size() * 3))
    for start in range(0, a.numel(), chunk):
        sl = slice(start, start + chunk)
        contribution = v.index_select(0, feature_ids[sl]) * a[sl, None]
        out = out.index_add(0, row_ids[sl], contribution)
    # Empty owner still has a connected gradient path for both inputs.
    return out.to(values.dtype)


def reference_backward(vectors, values, feature_ids, row_ids, grad, chunk_bytes=4 << 20):
    d = vectors.shape[1]
    accum = torch.float64 if vectors.dtype == torch.float64 else torch.float32
    v, a, g = vectors.to(accum), values.to(accum), grad.to(accum)
    dw = torch.zeros_like(v)
    pieces = []
    chunk = max(1, chunk_bytes // max(1, d * v.element_size() * 4))
    for start in range(0, a.numel(), chunk):
        sl = slice(start, start + chunk)
        gg = g.index_select(0, row_ids[sl])
        pieces.append((gg * v.index_select(0, feature_ids[sl])).sum(1))
        dw = dw.index_add(0, feature_ids[sl], gg * a[sl, None])
    dv = torch.cat(pieces) if pieces else values.new_empty(0)
    return dw.to(vectors.dtype), dv.to(values.dtype)


class _RaggedDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vectors, values, ids, rows, offsets, engine, split, index_backend):
        if engine == 'triton':
            from sae_lens.ragged_sae_triton import forward
            out = forward(vectors, values, ids, offsets)
        elif engine == 'torch_reference':
            out = reference_forward(vectors, values, ids, rows, offsets.numel() - 1)
        else:
            raise ValueError('Sparse engine must be triton or torch_reference')
        ctx.engine, ctx.split, ctx.index_backend = engine, split, index_backend
        ctx.save_for_backward(vectors, values, ids, rows, offsets)
        return out

    @staticmethod
    def backward(ctx, grad):
        vectors, values, ids, rows, offsets = ctx.saved_tensors
        if ctx.engine == 'triton':
            if torch.is_grad_enabled():
                raise RuntimeError('Triton sparse decoder supports first-order gradients; use torch_reference for gradgrad')
            from sae_lens.ragged_sae_triton import backward
            dw, dv = backward(vectors, values, ids, rows, grad, ctx.split, ctx.index_backend)
        else:
            dw, dv = reference_backward(vectors, values, ids, rows, grad)
        return dw, dv, None, None, None, None, None, None


def decode_ragged(acts: RaggedLatents, vectors, *, engine='triton', split=1, index_backend='sort',
                  openai_page_k=512, openai_workspace_mib=64, openai_forward='bucketed'):
    if vectors.ndim != 2 or vectors.shape[0] != acts.width or vectors.device != acts.device:
        raise ValueError('Expected local feature-major weight [S,d_in]')
    if not vectors.is_contiguous():
        raise ValueError('Prepare feature-major contiguous weight once per forward')
    if engine not in ('triton', 'torch_reference', 'openai') or split not in (1, 2, 4, 8):
        raise ValueError('Unknown sparse engine or split')
    if index_backend not in ('sort', 'histogram'):
        raise ValueError('Index backend must be sort or histogram')
    if acts.values.ndim != 1 or acts.row_offsets.shape != (acts.nrows + 1,):
        raise ValueError('Invalid packed values or row offsets')
    for t in (acts.row_ids, acts.feature_ids):
        if t.shape != acts.values.shape or t.device != acts.device or t.dtype != torch.int64 or not t.is_contiguous():
            raise ValueError('Packed row/feature metadata must be contiguous local int64')
    if acts.row_offsets.device != acts.device or acts.row_offsets.dtype != torch.int64 or not acts.row_offsets.is_contiguous():
        raise ValueError('Invalid row-offset dtype/device/layout')
    if engine == 'triton':
        if not acts.values.is_cuda:
            raise RuntimeError('Triton engine requires CUDA; CPU reference must be selected explicitly')
        import importlib.util
        if importlib.util.find_spec('triton') is None:
            raise RuntimeError('Install Triton to use the sparse decoder; no dense fallback')
    dtype = (torch.get_autocast_dtype(vectors.device.type)
             if torch.is_autocast_enabled(vectors.device.type) else vectors.dtype)
    if engine == 'triton' and dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('Triton engine requires FP16/BF16/FP32')
    if engine == 'triton' and torch.are_deterministic_algorithms_enabled() and (split != 1 or index_backend != 'sort'):
        raise RuntimeError('Deterministic sparse wgrad requires split=1,index_backend=sort')
    if engine == 'openai':
        if split != 1 or index_backend != 'sort':
            raise ValueError('openai uses its original sorted COO atomic kernel; v3 split/histogram flags do not apply')
        from sae_lens.openai_sae_adapter import decode_openai
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            out = decode_openai(vectors.to(dtype), acts.values.to(dtype), acts.feature_ids,
                                acts.row_ids, acts.row_offsets, page_k=openai_page_k,
                                workspace_mib=openai_workspace_mib, forward_mode=openai_forward)
        return out.reshape(*acts.leading_shape, vectors.shape[1])
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        out = _RaggedDecoder.apply(vectors.to(dtype), acts.values.to(dtype), acts.feature_ids,
                                   acts.row_ids, acts.row_offsets, engine, split, index_backend)
    return out.reshape(*acts.leading_shape, vectors.shape[1])


def dense_ragged_reference(acts: RaggedLatents, vectors):
    """EXPLICIT local-dense ablation, never called by sparse_strict."""
    from sae_lens.adaptive_sae import computation_dtype
    dtype = computation_dtype(vectors)
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        return F.linear(acts.to_dense().to(dtype), vectors.to(dtype).T)


def packed_feature_counts(acts: RaggedLatents, group=None):
    from sae_lens.sharded_topk import gather_feature_summary
    local = torch.zeros(acts.width, dtype=torch.float32, device=acts.device)
    local.scatter_add_(0, acts.feature_ids, (acts.values.detach() != 0).float())
    return gather_feature_summary(local, group)


def packed_diagnostics(acts: RaggedLatents, engine, mode):
    # No .item() on per-row lengths in the training hot path. E is known after
    # packing's nonzero. Row histograms can be collected by the profiler.
    return {'selection': acts.selection, 'engine': engine, 'compute_mode': mode,
            'entries': acts.nnz, 'rows': acts.nrows, 'local_width': acts.width,
            'mean_entries_per_row': acts.nnz / max(acts.nrows, 1),
            'forward': 'sparse' if mode == 'sparse' else 'local_dense',
            'value_gradient': 'sampled' if mode == 'sparse' else 'local_dense',
            'weight_gradient': 'sparse_grouped' if mode == 'sparse' else 'local_dense'}


def compact_ragged_dense(acts: RaggedLatents, vectors):
    """Explicit dense ablation on union of selected local columns.

    This is NEVER called in sparse mode. Used by structural auto only for
    select_all, where selected column set is identical for every token.
    """
    columns, inverse = torch.unique(acts.feature_ids, sorted=True, return_inverse=True)
    values = acts.values.new_zeros((acts.nrows, columns.numel()))
    values.index_put_((acts.row_ids, inverse), acts.values, accumulate=True)
    weights = vectors.index_select(0, columns)
    from sae_lens.adaptive_sae import computation_dtype
    dtype = computation_dtype(vectors)
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        return F.linear(values.to(dtype), weights.to(dtype).T).reshape(*acts.leading_shape, vectors.shape[1])
