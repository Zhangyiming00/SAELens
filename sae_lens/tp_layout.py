"""Exact feature ownership for unequal and elastic SAE tensor parallelism.

Feature IDs are stable semantic IDs. Storage order belongs to each owner and
may change after a switch. No dummy features or padded parameter tensors exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


def balanced_widths(total: int, size: int) -> tuple[int, ...]:
    if not 1 <= size <= total:
        raise ValueError(
            "TP size must be in [1, d_sae]; empty feature shards are unsupported"
        )
    q, r = divmod(total, size)
    return tuple(q + (i < r) for i in range(size))


def shard_bounds(total: int, size: int, rank: int) -> tuple[int, int]:
    widths = balanced_widths(total, size)
    if not 0 <= rank < size:
        raise ValueError("Invalid TP rank")
    return sum(widths[:rank]), widths[rank]


@dataclass(frozen=True)
class FeatureLayout:
    ranks: tuple[int, ...]
    ids: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if not self.ranks or tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("Ranks must be nonempty, unique and sorted")
        if len(self.ids) != len(self.ranks) or any(not x for x in self.ids):
            raise ValueError("Every member must own a nonempty feature shard")
        flat = [i for part in self.ids for i in part]
        if sorted(flat) != list(range(len(flat))):
            raise ValueError("Feature IDs must form an exact permutation of [0, d_sae)")
        if len(flat) > 1 << 32:
            raise ValueError("Stable TopK IDs must fit uint32")

    @classmethod
    def balanced(cls, total: int, ranks: tuple[int, ...]):
        widths = balanced_widths(total, len(ranks))
        cursor, parts = 0, []
        for width in widths:
            parts.append(tuple(range(cursor, cursor + width)))
            cursor += width
        return cls(ranks, tuple(parts))

    @property
    def widths(self):
        return tuple(map(len, self.ids))

    @property
    def total(self):
        return sum(self.widths)

    def local(self, global_rank: int):
        return LocalFeatures(self, self.ranks.index(global_rank))


class LocalFeatures:
    def __init__(self, layout: FeatureLayout, rank: int):
        self.layout, self.rank = layout, rank
        self.widths = layout.widths
        self.width = self.widths[rank]
        self.total = layout.total
        self.canonical = tuple(i for part in layout.ids for i in part) == tuple(
            range(self.total)
        )
        self.offset = sum(self.widths[:rank])  # storage offset, NOT a feature ID
        self._ids = {}
        part = layout.ids[rank]
        self.contiguous_start = (
            part[0] if part == tuple(range(part[0], part[0] + len(part))) else None
        )

    def ids(self, device):
        device = torch.device(device)
        if device not in self._ids:
            self._ids[device] = torch.tensor(
                self.layout.ids[self.rank], device=device, dtype=torch.long
            )
        return self._ids[device]

    def select(self, full, axis=-1):
        if self.contiguous_start is not None:
            return full.narrow(axis, self.contiguous_start, self.width)
        return full.index_select(axis, self.ids(full.device))

    def counts(self, local, group):
        result = local.new_zeros(self.total)
        result.index_copy_(0, self.ids(local.device), local)
        if len(self.widths) > 1:
            if group is None:
                raise ValueError("A multi-rank feature map requires an explicit group")
            dist.all_reduce(result, group=group)
        return result


@dataclass(frozen=True)
class FeatureMove:
    source: int
    target: int
    start: int
    length: int
    destination: int


@dataclass(frozen=True)
class AdjacentPlan:
    old: FeatureLayout
    new: FeatureLayout
    moves: tuple[FeatureMove, ...]

    @property
    def moved_features(self):
        return sum(x.length for x in self.moves)


def plan_adjacent(old: FeatureLayout, ranks: tuple[int, ...]) -> AdjacentPlan:
    """Tail donation on grow, departing-owner scatter on shrink.

    Assign rounding remainders to the largest surviving shards first. This
    keeps every survivor's target <= old size on grow, >= old size on shrink.
    Thus survivors NEVER exchange features with one another. Balanced inputs
    and outputs differ in width by at most one; arbitrary departing ranks work.
    """
    if tuple(sorted(set(ranks))) != ranks or not ranks:
        raise ValueError("Target ranks must be nonempty, unique and sorted")
    before, after = set(old.ranks), set(ranks)
    if abs(len(before) - len(after)) != 1 or not (before < after or after < before):
        raise ValueError("Only a single member join or leave is supported")
    if max(old.widths) - min(old.widths) > 1:
        raise ValueError("Adjacent minimal migration requires balanced old shards")
    balanced_widths(old.total, len(ranks))
    owned = dict(zip(old.ranks, old.ids))
    q, remainder = divmod(old.total, len(ranks))
    priority = sorted(ranks, key=lambda rank: (-len(owned.get(rank, ())), rank))
    extra = set(priority[:remainder])
    target = {rank: q + (rank in extra) for rank in ranks}
    parts = {rank: list(owned.get(rank, ())) for rank in ranks}
    moves = []
    if len(after) > len(before):
        newcomer = next(iter(after - before))
        for rank in old.ranks:
            cut = target[rank]
            tail = parts[rank][cut:]
            if tail:
                moves.append(
                    FeatureMove(rank, newcomer, cut, len(tail), len(parts[newcomer]))
                )
                parts[newcomer].extend(tail)
                del parts[rank][cut:]
    else:
        departed = next(iter(before - after))
        cursor = 0
        for rank in ranks:
            needed = target[rank] - len(parts[rank])
            if needed:
                moves.append(
                    FeatureMove(departed, rank, cursor, needed, len(parts[rank]))
                )
                parts[rank].extend(owned[departed][cursor : cursor + needed])
                cursor += needed
        if cursor != len(owned[departed]):
            raise RuntimeError("Incomplete departing shard coverage")
    new = FeatureLayout(ranks, tuple(tuple(parts[rank]) for rank in ranks))
    if any(len(parts[r]) != target[r] for r in ranks):
        raise RuntimeError("Incorrect adjacent ownership plan")
    return AdjacentPlan(old, new, tuple(moves))


@torch.no_grad()
def gather_features(local, features: LocalFeatures, group, axis=-1):
    """Canonical gather using exact-sized broadcasts (also works on Gloo).

    Used for legacy full storage and export, never sharded TopK's hot path.
    Receivers allocate the source's real width; no padding is communicated.
    """
    axis %= local.ndim
    shape = list(local.shape)
    shape[axis] = features.total
    full = local.new_empty(shape)
    for index, (owner, ids) in enumerate(
        zip(features.layout.ranks, features.layout.ids)
    ):
        shape[axis] = len(ids)
        part = (
            local.detach().contiguous()
            if index == features.rank
            else local.new_empty(shape)
        )
        if len(features.widths) > 1:
            dist.broadcast(part, src=owner, group=group)
        full.index_copy_(
            axis, torch.tensor(ids, device=local.device, dtype=torch.long), part
        )
    return full


class _GatherFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, features, group):
        ctx.features = features
        return gather_features(local, features, group)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.features.select(gradient).contiguous(), None, None


def gather_features_autograd(local, features, group):
    return _GatherFeatures.apply(local, features, group)
