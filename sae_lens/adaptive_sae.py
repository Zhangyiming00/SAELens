"""SAE execution policies, local compute planning and mixed decoder autograd.

Selection, TP/DP communication and optimizer ownership stay outside this file.
A plan may differ between TP ranks; it never changes collective order. No global
latent, no approximate local TopK capacity, and no change to the winner mask.
The policy is configurable, not a claim of a hardware-independent optimum.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

MODES = ('inherit', 'sparse', 'local_dense', 'compact_dense', 'auto')
STAGES = ('forward', 'dvalues', 'dweight')


REPRESENTATIONS = ("none", "full", "sharded_dense", "sharded_ragged")
COMPUTATIONS = ("none", "sparse", "dense", "compact", "auto")
DEFAULT_REPRESENTATION = "sharded_ragged"
DEFAULT_COMPUTE_WORKSPACE_MIB = 128
DENSE_COMPUTE_WORKSPACE_MIB = 512
DEFAULT_OPENAI_WORKSPACE_MIB = 256


def representation(value):
    value = "none" if value is None else str(value).lower()
    value = {"shared_dense": "sharded_dense", "shared_ragged": "sharded_ragged"}.get(
        value, value
    )
    if value not in REPRESENTATIONS:
        raise ValueError(f"Unknown latent representation: {value}")
    return value


def computation(value):
    value = "none" if value is None else str(value).lower()
    value = {"inherit": "none", "local_dense": "dense", "compact_dense": "compact"}.get(
        value, value
    )
    if value not in COMPUTATIONS:
        raise ValueError(f"Unknown decoder computation: {value}")
    return value


def enabled(cfg):
    return any(
        getattr(cfg, f"{branch}_{field}", None) is not None
        for branch in ("main", "aux")
        for field in ("representation", "compute")
    ) or any(
        computation(getattr(cfg, f"{branch}_{stage}", "none")) != "none"
        for branch in ("main", "aux")
        for stage in ("forward", "dvalues", "dweight")
    )


def branch_representation(cfg, auxiliary=False):
    value = representation(
        getattr(cfg, ("aux" if auxiliary else "main") + "_representation", None)
    )
    return DEFAULT_REPRESENTATION if value == "none" else value


def branch_compute(cfg, auxiliary=False):
    value = computation(
        getattr(cfg, ("aux" if auxiliary else "main") + "_compute", None)
    )
    return ("compact" if auxiliary else "sparse") if value == "none" else value


def stage_requests(cfg, auxiliary=False):
    branch = "aux" if auxiliary else "main"
    base = branch_compute(cfg, auxiliary)
    result = []
    for stage in ("forward", "dvalues", "dweight"):
        value = computation(getattr(cfg, f"{branch}_{stage}", "none"))
        value = base if value == "none" else value
        result.append(
            {"dense": "local_dense", "compact": "compact_dense"}.get(value, value)
        )
    return tuple(result)


def validate(cfg):
    if not enabled(cfg):
        return
    for aux in (False, True):
        branch_representation(cfg, aux)
        branch_compute(cfg, aux)
        stage_requests(cfg, aux)


def active(cfg, auxiliary=False):
    if enabled(cfg):
        return True
    branch = 'aux' if auxiliary else 'main'
    return any(getattr(cfg, 'v5_' + branch + '_' + suffix, 'inherit') != 'inherit'
               for suffix in ('compute', *STAGES))


def validate_config(cfg):
    validate(cfg)
    for branch in ('main', 'aux'):
        for suffix in ('compute', *STAGES):
            if getattr(cfg, f'v5_{branch}_{suffix}', 'inherit') not in MODES:
                raise ValueError(f'Unknown v5 {branch} {suffix} policy')
        for stage in STAGES:
            threshold = getattr(cfg, f'v5_{branch}_{stage}_threshold', None)
            if threshold is not None and (type(threshold) is not int or threshold < 0):
                raise ValueError('Stage thresholds must be nonnegative integers')
        threshold = getattr(cfg, f'v5_{branch}_threshold', 512)
        if type(threshold) is not int or threshold < 0:
            raise ValueError('Sparse thresholds must be nonnegative integers')
    if getattr(cfg, 'v5_k_metric', 'mean') not in ('mean', 'max'):
        raise ValueError('v5_k_metric must be mean or max')
    for name in ('v5_compact_max_ratio', 'v5_compact_min_density'):
        if not 0 <= getattr(cfg, name, .5 if name.endswith('ratio') else .25) <= 1:
            raise ValueError(name + ' must be in [0,1]')
    for name, default in (('v5_workspace_mib', DEFAULT_COMPUTE_WORKSPACE_MIB),
                          ('ragged_openai_workspace_mib', DEFAULT_OPENAI_WORKSPACE_MIB)):
        value = getattr(cfg, name, default)
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError(name + ' must be an integer in [1,4096] MiB')
    if not enabled(cfg) and (active(cfg) or active(cfg, True)) and cfg.topk_backend != 'sharded_ragged':
        raise ValueError('V5 compute policies require topk_backend=sharded_ragged (also works at TP1)')
    if getattr(cfg, 'topk_tie_policy', 'stable_id') not in ('stable_id', 'torch_tp1'):
        raise ValueError('Unknown TopK tie policy')
    k = getattr(cfg, 'auxk', None)
    if k is not None and (type(k) is not int or k < 0):
        raise ValueError('auxk must be nonnegative or None')


@dataclass(frozen=True)
class Plan:
    forward: str
    dvalues: str
    dweight: str
    branch: str
    local_k: float
    columns: int | None
    reasons: tuple[str, str, str]

    @property
    def needs_sparse(self):
        return 'sparse' in (self.forward, self.dvalues, self.dweight)

    @property
    def needs_compact(self):
        return 'compact_dense' in (self.forward, self.dvalues, self.dweight)

    def diagnostics(self):
        return dict(forward=self.forward, value_gradient=self.dvalues,
                    weight_gradient=self.dweight, compute_mode='v5_mixed',
                    local_k_metric=self.local_k, compact_width=self.columns,
                    dispatch_reasons=dict(zip(STAGES, self.reasons)))


def requests(cfg, auxiliary=False):
    if enabled(cfg):
        return stage_requests(cfg, auxiliary)
    branch = 'aux' if auxiliary else 'main'
    old = getattr(cfg, 'ragged_' + branch + '_compute', 'sparse')
    base = getattr(cfg, 'v5_' + branch + '_compute', 'inherit')
    if base == 'inherit':
        base = old
    return tuple(base if getattr(cfg, f'v5_{branch}_{stage}', 'inherit') == 'inherit'
                 else getattr(cfg, f'v5_{branch}_{stage}') for stage in STAGES)


def choose_plan(cfg, *, auxiliary, rows, width, entries, columns=None,
                max_k=None, select_all=False):
    """Pure host policy. K means REAL local entries, never global K/TP.

    `mean` requires no additional GPU readback after packing; `max` has an
    explicit one-scalar readback at the caller. Thresholds are strict > cuts.
    Compact auto requires a useful column reduction and enough within-compact
    density. Explicit modes do not silently fall back on allocation/JIT errors.
    """
    branch = 'aux' if auxiliary else 'main'
    metric = (entries / max(rows, 1) if getattr(cfg, 'v5_k_metric', 'mean') == 'mean'
              else float(max_k if max_k is not None else 0))
    ratio = getattr(cfg, 'v5_compact_max_ratio', .5)
    density = entries / max(rows * (columns or 0), 1)
    compact_ok = (columns is not None and columns < width and columns <= ratio * width
                  and (select_all or density >= getattr(cfg, 'v5_compact_min_density', .25)))
    modes, reasons = [], []
    for stage, requested in zip(STAGES, requests(cfg, auxiliary)):
        threshold = getattr(cfg, f'v5_{branch}_{stage}_threshold', None)
        if threshold is None:
            threshold = getattr(cfg, f'v5_{branch}_threshold', 512)
        if requested != 'auto':
            mode, reason = requested, 'explicit/inherited'
        elif auxiliary and select_all:
            mode = 'compact_dense' if compact_ok or columns == 0 else 'local_dense'
            reason = 'all eligible selected; regular submatrix'
        elif metric <= threshold:
            mode, reason = 'sparse', f'local K <= {threshold}'
        else:
            mode = 'compact_dense' if compact_ok else 'local_dense'
            reason = f'local K > {threshold}; ' + ('useful compact' if compact_ok else 'dense shard')
        modes.append(mode); reasons.append(reason)
    return Plan(*modes, branch, metric, columns, tuple(reasons))


@contextmanager
def phase(branch, stage):
    with torch.autograd.profiler.record_function(f'sae_v5:{branch}:{stage}'):
        yield


def computation_dtype(vectors):
    dev = vectors.device.type
    return torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else vectors.dtype


def _columns(acts, known_columns=None):
    if known_columns is not None:
        columns = known_columns
        if columns.numel() == acts.width:
            # Sorted unique columns covering the shard are the identity map.
            return columns, acts.feature_ids
        # O(S) integer map, not O(E) sorting. The selected entries are a subset
        # of the supplied eligible set, whose IDs must be sorted and unique.
        lookup = torch.full((acts.width,), -1, dtype=torch.int64, device=acts.device)
        lookup[columns] = torch.arange(columns.numel(), device=acts.device)
        inverse = lookup.index_select(0, acts.feature_ids)
    else:
        columns, inverse = torch.unique(acts.feature_ids, sorted=True, return_inverse=True)
    return columns, inverse


def make_plan(acts, cfg, auxiliary=False, known_columns=None):
    maximum = None
    if getattr(cfg, 'v5_k_metric', 'mean') == 'max':
        maximum = int(acts.row_offsets.diff().max()) if acts.nrows else 0
    count = None if known_columns is None else known_columns.numel()
    plan = choose_plan(cfg, auxiliary=auxiliary, rows=acts.nrows, width=acts.width,
                       entries=acts.nnz, columns=count, max_k=maximum,
                       select_all=acts.selection == 'select_all')
    requested = requests(cfg, auxiliary)
    # Only discover Main's union when explicitly requested or when an auto
    # dense stage might benefit. Small-K sparse never pays torch.unique.
    discover = ('compact_dense' in requested or plan.needs_compact or
                ('auto' in requested and 'local_dense' in (plan.forward, plan.dvalues, plan.dweight)))
    columns = inverse = None
    if discover:
        with phase(plan.branch, 'column_map'):
            columns, inverse = _columns(acts, known_columns)
        plan = choose_plan(cfg, auxiliary=auxiliary, rows=acts.nrows, width=acts.width,
                           entries=acts.nnz, columns=columns.numel(), max_k=maximum,
                           select_all=acts.selection == 'select_all')
    return plan, columns, inverse


def options(cfg):
    return dict(engine=getattr(cfg, 'ragged_decoder_engine', 'triton'),
                split=getattr(cfg, 'ragged_wgrad_split', 1),
                index_backend=getattr(cfg, 'ragged_index_backend', 'sort'),
                page_k=getattr(cfg, 'ragged_openai_page_k', 512),
                openai_workspace_mib=getattr(cfg, 'ragged_openai_workspace_mib', DEFAULT_OPENAI_WORKSPACE_MIB),
                forward_mode=getattr(cfg, 'ragged_openai_forward', 'bucketed'),
                workspace_mib=getattr(cfg, 'v5_workspace_mib', DEFAULT_COMPUTE_WORKSPACE_MIB))


class _HybridDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vectors, values, ids, row_ids, offsets, columns, inverse, plan, opts):
        from sae_lens.sparse_parts import sparse_forward
        nrows, width, d = offsets.numel() - 1, vectors.shape[0], vectors.shape[1]
        ctx.plan, ctx.opts, ctx.nrows = plan, opts, nrows
        ctx.columns, ctx.inverse = columns, inverse
        ctx.save_for_backward(vectors, values, ids, row_ids, offsets)
        ctx.tiles = None
        if any(mode != 'sparse' for mode in (plan.forward, plan.dvalues, plan.dweight)):
            c = width if 'local_dense' in (plan.forward, plan.dvalues, plan.dweight) else columns.numel()
            # Budget for tile activation + dA + output/G and selected metadata.
            row_bytes = max(1, (2*c + 2*d)*max(4, values.element_size()))
            tile = max(1, (opts['workspace_mib'] << 20)//row_bytes)
            boundaries = list(range(0, nrows, tile)) + [nrows]
            # One metadata transfer, never per-tile boolean/nonzero filtering.
            ptr = offsets[torch.tensor(boundaries, device=offsets.device)].detach().cpu().tolist()
            ctx.tiles = [(boundaries[i], boundaries[i+1], ptr[i], ptr[i+1])
                         for i in range(len(boundaries)-1)]
        ctx.groups = None
        with torch.autocast(device_type=vectors.device.type, enabled=False), phase(plan.branch, 'forward:' + plan.forward):
            if plan.forward == 'sparse':
                out, ctx.groups = sparse_forward(vectors, values, ids, row_ids, offsets, opts)
            else:
                out = vectors.new_zeros((nrows, d))
                identity = plan.forward == 'local_dense' or columns.numel() == width
                ww = vectors if identity else vectors.index_select(0, columns)
                jj = ids if identity else inverse
                for lo, hi, a, b in ctx.tiles:
                    dense = _dense_tile(values, jj, row_ids, lo, hi, a, b, ww.shape[0])
                    out[lo:hi] = dense @ ww
        return out

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError('V5 mixed decoder supports first-order training gradients only')
        from sae_lens.sparse_parts import sparse_dvalues, sparse_dweight
        vectors, values, ids, rows, offsets = ctx.saved_tensors
        plan, opts = ctx.plan, ctx.opts
        grad = grad.to(values.dtype).contiguous()
        dv = dw = None
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            if ctx.needs_input_grad[1]:
                with phase(plan.branch, 'dvalues:' + plan.dvalues):
                    if plan.dvalues == 'sparse':
                        dv = sparse_dvalues(vectors, values, ids, rows, offsets, grad, opts, ctx.groups)
                    else:
                        identity = plan.dvalues == 'local_dense' or ctx.columns.numel() == vectors.shape[0]
                        ww = vectors if identity else vectors.index_select(0, ctx.columns)
                        jj = ids if identity else ctx.inverse
                        dv = torch.empty_like(values)
                        for lo, hi, a, b in ctx.tiles:
                            full_grad = grad[lo:hi] @ ww.T
                            # Read ONLY selected positions, including selected zeros.
                            dv[a:b] = full_grad[rows[a:b]-lo, jj[a:b]]
            if ctx.needs_input_grad[0]:
                with phase(plan.branch, 'dweight:' + plan.dweight):
                    if plan.dweight == 'sparse':
                        dw = sparse_dweight(vectors, values, ids, rows, grad, opts)
                    else:
                        c = vectors.shape[0] if plan.dweight == 'local_dense' else ctx.columns.numel()
                        jj = ids if plan.dweight == 'local_dense' else ctx.inverse
                        accum_dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
                        small = torch.zeros((c, vectors.shape[1]), device=vectors.device, dtype=accum_dtype)
                        for lo, hi, a, b in ctx.tiles:
                            dense = _dense_tile(values, jj, rows, lo, hi, a, b, c)
                            if dense.dtype == grad.dtype == accum_dtype:
                                # Accumulate GEMM directly into dW, avoiding a
                                # second [columns,d_in] result for every tile.
                                small.addmm_(dense.T, grad[lo:hi])
                            else:
                                # Preserve FP32 accumulation of low-precision
                                # tile products without upcasting both inputs.
                                small.add_((dense.T @ grad[lo:hi]).to(accum_dtype))
                        if plan.dweight == 'local_dense' or c == vectors.shape[0]:
                            dw = small.to(vectors.dtype)
                        else:
                            dw = torch.zeros_like(vectors)
                            dw.index_copy_(0, ctx.columns, small.to(vectors.dtype))
        return dw, dv, None, None, None, None, None, None, None


def _dense_tile(values, ids, rows, lo, hi, a, b, width):
    out = values.new_zeros((hi-lo, width))
    out.index_put_((rows[a:b]-lo, ids[a:b]), values[a:b], accumulate=True)
    return out


def decode_adaptive(acts, vectors, cfg, *, auxiliary=False, known_columns=None):
    """Return output and scalar-only diagnostics; never stores a cross-step cache."""
    plan, columns, inverse = make_plan(acts, cfg, auxiliary, known_columns)
    dtype = computation_dtype(vectors)
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        out = _HybridDecoder.apply(vectors.to(dtype), acts.values.to(dtype), acts.feature_ids,
                                   acts.row_ids, acts.row_offsets, columns, inverse, plan, options(cfg))
    from sae_lens.ragged_sae import packed_diagnostics
    info = packed_diagnostics(acts, getattr(cfg, 'ragged_decoder_engine', 'triton'), 'v5_mixed')
    info.update(plan.diagnostics())
    info['k_metric'] = getattr(cfg, 'v5_k_metric', 'mean')
    info['workspace_mib'] = getattr(cfg, 'v5_workspace_mib', DEFAULT_COMPUTE_WORKSPACE_MIB)
    return out.reshape(*acts.leading_shape, vectors.shape[1]), info


def try_direct_aux(scores, eligible, num_dead, k_aux, weight, norm, cfg):
    """Exact select-all dense shortcut BEFORE building B*m ragged metadata.

    Returns None if any chosen stage is sparse or not compact. No communication
    here: the model still executes its normal reconstruction reduction.
    """
    if num_dead > k_aux or not active(cfg, True) or getattr(cfg, 'auxk_selection', 'auto') != 'auto':
        return None
    # Explicit sparse remains strict even in the select-all case.
    if 'sparse' in requests(cfg, True):
        return None
    columns = eligible.nonzero(as_tuple=True)[0]
    rows, m = math.prod(scores.shape[:-1]), columns.numel()
    if (rows*m + weight.shape[0]*m)*max(4, weight.element_size()) > (getattr(cfg, 'v5_workspace_mib', DEFAULT_COMPUTE_WORKSPACE_MIB) << 20):
        return None
    plan = choose_plan(cfg, auxiliary=True, rows=rows, width=scores.shape[-1], entries=rows*m,
                       columns=m, max_k=m, select_all=True)
    if (plan.forward, plan.dvalues, plan.dweight) != ('compact_dense',)*3:
        return None
    from sae_lens.auxk_compact import _DeadColumnValues, compact_aux_decode
    with phase('aux', 'direct_compact'):
        values = _DeadColumnValues.apply(scores, columns)
        out = compact_aux_decode(values, columns, weight, norm)
    info = dict(selection='select_all', engine='dense_gemm', entries=rows*m, rows=rows,
                local_width=scores.shape[-1], mean_entries_per_row=float(m),
                direct_compact=True, avoided_ragged_entries=rows*m)
    info.update(plan.diagnostics())
    return out, info
