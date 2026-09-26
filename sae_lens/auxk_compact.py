"""Exact AuxK cliff fixes: dead-column GEMM and bounded complement selection.

Only local feature columns are materialized. There is no global token x feature
activation, embedding_bag, custom optimizer, or communication autograd here.
``group=None`` means LOCAL, never the default world. Callers provide the known
GLOBAL dead count; TP peers must use the same count, k, and configuration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sae_lens.sharded_topk import (
    _group_rank, _group_size, score_keys, sharded_auxk,
)

_MAX_KEY = (1 << 63) - 1
_LOW_BITS = (1 << 32) - 1


@dataclass(frozen=True)
class AuxKDensePlan:
    selection: str
    decoder: str
    k: int
    num_dead: int
    exclude_k: int
    tp_size: int
    shard_width: int


def plan_auxk_dense(*, num_dead: int, k: int, shard_width: int, tp_size: int,
                    element_size: int, decoder: str = "auto",
                    complement: str = "auto", selection_policy: str = "auto",
                    protocol: str = "auto") -> AuxKDensePlan:
    """A host-only plan: no branch depends on rank-local nnz/timing.

    Complement exchange is used only when the excluded budget is smaller than
    k AND its receive payload fits one local score row. q=1 uses a scalar MIN
    per token, rather than candidate all-gather. Explicit radix/legacy disable
    complement to preserve diagnostic controls. Auto packing is conservative:
    at most half of global features are eligible, and no general TopK needed.
    This threshold is a policy, NOT a hardware-optimality claim.
    """
    if decoder not in ("auto", "local_dense", "compact_dense"):
        raise ValueError("auxk_decoder_backend must be auto/local_dense/compact_dense")
    if complement not in ("auto", "off") or selection_policy not in ("auto", "legacy"):
        raise ValueError("Invalid AuxK complement/selection policy")
    if protocol not in ("auto", "candidates", "radix"):
        raise ValueError("Invalid TopK protocol")
    if min(shard_width, tp_size, element_size) < 1:
        raise ValueError("Invalid shard layout or dtype size")
    if not 0 < k <= num_dead <= shard_width * tp_size <= (1 << 32):
        raise ValueError("Invalid global eligible count or AuxK budget")
    q = num_dead - k
    route = "topk"
    if selection_policy == "auto":
        if q == 0:
            route = "select_all"
        elif (complement == "auto" and protocol != "radix" and q < k
              and (q == 1 or tp_size * q * 8 <= shard_width * element_size)):
            route = "complement_min" if q == 1 else "complement_candidates"
    pack = (decoder == "compact_dense" or
            (decoder == "auto" and route != "topk"
             and num_dead * 2 <= shard_width * tp_size))
    return AuxKDensePlan(route, "compact_dense" if pack else "local_dense", k,
                        num_dead, q if route.startswith("complement") else 0,
                        tp_size, shard_width)


class _DeadColumnValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, columns):
        ctx.shape = scores.shape
        ctx.save_for_backward(columns)
        return scores.index_select(-1, columns)

    @staticmethod
    def backward(ctx, grad):
        (columns,) = ctx.saved_tensors
        # Unique dead columns, dense LOCAL dZ for native encoder backward.
        # No full scores/values need to be saved by this operation.
        return grad.new_zeros(ctx.shape).index_add(-1, columns, grad), None


@torch.no_grad()
def _keys_for_columns(values: torch.Tensor, global_columns: torch.Tensor) -> torch.Tensor:
    # Reuse the EXACT score encoding, including NaN, +/-inf and signed-zero
    # policy. Replace only the index low bits; do not negate scores (ties!).
    keys = score_keys(values)
    return (keys & ~_LOW_BITS) | (_LOW_BITS - global_columns[None, :])


def _row_tile(width: int) -> int:
    # Conservative Torch elementwise scratch bound (not allocator-peak claim).
    return max(1, (32 * 1024 * 1024) // (max(width, 1) * 64))


@torch.no_grad()
def _bottom_threshold(values: torch.Tensor, global_columns: torch.Tensor,
                      q: int, group: dist.ProcessGroup | None) -> torch.Tensor:
    """q-th SMALLEST unique global key; columns may be ragged/empty per rank.

    All TP peers send [rows,q] even with zero local dead columns. Only detached
    comparison keys communicate. +MAX padding cannot be selected at threshold:
    q < num_dead, and there is at most one eligible key equal to +MAX.
    """
    rows, width = values.shape
    if rows == 0:
        return torch.empty(0, dtype=torch.int64, device=values.device)
    p = _group_size(group)
    local = torch.full((rows, q), _MAX_KEY, dtype=torch.int64, device=values.device)
    if width:
        take = min(q, width)
        tile = _row_tile(width)
        for start in range(0, rows, tile):
            keys = _keys_for_columns(values[start:start + tile], global_columns)
            candidates = (keys.amin(1, keepdim=True) if q == 1 else
                          keys.topk(take, dim=1, largest=False, sorted=False).values)
            local[start:start + tile, :take].copy_(candidates)
    if q == 1:
        threshold = local[:, 0].contiguous()
        if p > 1:
            dist.all_reduce(threshold, op=dist.ReduceOp.MIN, group=group)
        return threshold
    if p == 1:
        return local.amax(1)
    recv = torch.empty((p * rows, q), dtype=torch.int64, device=values.device)
    dist.all_gather_into_tensor(recv, local.contiguous(), group=group)
    merged = recv.view(p, rows, q).permute(1, 0, 2).reshape(rows, p * q)
    return merged.topk(q, dim=1, largest=False, sorted=False).values.amax(1)


@torch.no_grad()
def _complement_mask(values: torch.Tensor, global_columns: torch.Tensor,
                     q: int, group: dist.ProcessGroup | None) -> torch.Tensor:
    rows, width = values.shape
    threshold = _bottom_threshold(values, global_columns, q, group)
    if q == 1:
        # The threshold key ALREADY contains the unique excluded feature ID.
        # Do not rescan/re-encode all scores merely to identify that one feature.
        excluded_id = _LOW_BITS - (threshold & _LOW_BITS)
        return global_columns[None, :] != excluded_id[:, None]
    keep = torch.empty((rows, width), dtype=torch.bool, device=values.device)
    if width:
        tile = _row_tile(width)
        for start in range(0, rows, tile):
            keys = _keys_for_columns(values[start:start + tile], global_columns)
            # Unique integer ordering: remove exactly q eligible entries globally.
            keep[start:start + tile].copy_(keys > threshold[start:start + tile, None])
    return keep


def prepare_auxk_dense(scores: torch.Tensor, k: int, eligible: torch.Tensor,
                       num_eligible: int, group: dist.ProcessGroup | None = None, *,
                       decoder: str = "auto", complement: str = "auto",
                       selection_policy: str = "auto", protocol: str = "auto",
                       key_backend: str = "torch", tie_policy: str = "stable_id") -> tuple[torch.Tensor, torch.Tensor | None, AuxKDensePlan]:
    """Return local values, optional ORIGINAL local feature columns, and plan.

    columns=None: values already have local shard width (native decoder).
    columns!=None: values are [*,m_local] (ordinary compact dense GEMM).
    nonzero inspects ONLY the one-dimensional dead mask. It may synchronize
    CUDA to learn m_local; no stale mask/weight cache is maintained.
    """
    if scores.ndim < 2 or scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Expected FP32/BF16/FP16 local [...,S] scores")
    if eligible.shape != (scores.shape[-1],) or eligible.dtype != torch.bool or eligible.device != scores.device:
        raise ValueError("Expected a boolean LOCAL dead mask on the score device")
    if key_backend not in ("torch", "triton"):
        raise ValueError("Unknown comparison-key backend")
    if tie_policy == "torch_tp1":
        if _group_size(group) != 1:
            raise ValueError("torch_tp1 requires singleton TP")
        if k < num_eligible:
            selection_policy = "legacy"
    plan = plan_auxk_dense(num_dead=num_eligible, k=k, shard_width=scores.shape[-1],
                          tp_size=_group_size(group), element_size=scores.element_size(),
                          decoder=decoder, complement=complement,
                          selection_policy=selection_policy, protocol=protocol)
    if plan.selection == "topk":
        acts = sharded_auxk(scores, k, eligible, num_eligible, group, sparse=False,
                            policy=selection_policy, protocol=protocol, key_backend=key_backend, tie_policy=tie_policy)
        if plan.decoder == "local_dense":
            return acts, None, plan
        columns = eligible.nonzero(as_tuple=True)[0]
        return _DeadColumnValues.apply(acts, columns), columns, plan
    if plan.selection == "select_all" and plan.decoder == "local_dense":
        return torch.where(eligible, scores, 0.0), None, plan
    columns = eligible.nonzero(as_tuple=True)[0]
    packed = _DeadColumnValues.apply(scores, columns)
    if plan.selection.startswith("complement"):
        flat = packed.reshape(math.prod(scores.shape[:-1]), columns.numel())
        ids = columns + _group_rank(group) * scores.shape[-1]
        # Torch exact keys are the correctness backend for arbitrary column IDs;
        # main/general TopK still respect topk_key_backend. No dynamic global gather.
        keep = _complement_mask(flat.detach(), ids, plan.exclude_k, group)
        packed = torch.where(keep.reshape(packed.shape), packed, 0.0)
    if plan.decoder == "compact_dense":
        return packed, columns, plan
    acts = scores.new_zeros(scores.shape).index_copy(-1, columns, packed)
    return acts, None, plan


def compact_aux_decode(values: torch.Tensor, columns: torch.Tensor,
                       weight: torch.Tensor, norm: torch.Tensor | None = None) -> torch.Tensor:
    """Ordinary dense linear over dead columns, no replicated parameters.

    weight is the existing native [d_in,S] Parameter. index_select returns
    a temporary [d_in,m_local], whose backward scatters into a dense local
    parameter gradient. Empty local masks also execute connected operations;
    caller MUST still perform the TP reconstruction reduction.
    """
    if weight.ndim != 2 or columns.ndim != 1 or columns.dtype != torch.int64:
        raise ValueError("Expected native [d_in,S] weight and 1-D int64 columns")
    if values.shape[-1] != columns.numel() or values.device != weight.device or columns.device != weight.device:
        raise ValueError("Inconsistent compact decoder inputs")
    if norm is not None:
        if norm.shape != (weight.shape[1],):
            raise ValueError("Decoder norm must have LOCAL feature width")
        # Keep the original multiply-by-reciprocal graph; do not algebraically
        # cancel decoder norm or detach it. Selected zero/negative values survive.
        values = values * norm.index_select(0, columns).reciprocal()
    return F.linear(values, weight.index_select(1, columns), bias=None)
