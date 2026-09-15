"""Explicit Megatron process groups for fixed SAE training domains.

Megatron 0.16.1's global initializer always uses the default world. Instead we
use its RankGenerator, create_group and ProcessGroupCollection APIs explicitly.
The routing world remains intact, including ranks which only produce data.
Hook placement is independent model placement, never pipeline scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch.distributed as dist

from sae_lens.megatron_tp import require_megatron_core


@dataclass(frozen=True)
class SAETrainingDomain:
    name: str
    ranks: tuple[int, ...]
    tp_size: int
    hooks: tuple[str, ...] = ()


@dataclass(frozen=True)
class SAEEndpoint:
    index: int
    domain: str
    placement_index: int
    replica_index: int
    tp_ranks: tuple[int, ...]
    receive_rank: int
    hooks: tuple[str, ...]


@dataclass(frozen=True)
class SAEParallelContext:
    domain: SAETrainingDomain
    groups: Any  # Megatron ProcessGroupCollection; dependency is optional at import.
    tp_cpu_group: dist.ProcessGroup
    tp_rank: int
    dp_rank: int
    tp_ranks: tuple[int, ...]
    dp_ranks: tuple[int, ...]
    receive_rank: int

    @property
    def tp_group(self) -> dist.ProcessGroup:
        return self.groups.tp

    @property
    def dp_group(self) -> dist.ProcessGroup:
        return self.groups.dp


class SAERuntime:
    """Own one fixed topology; all world ranks initialize/close in the same order.

    Co-located hooks share a context. Different placements have disjoint domains.
    Endpoint metadata is exchanged from actual process-group members, including
    to producer-only ranks. ``generation`` reserves topology identity for future
    routing protocols; this runtime does not implement reconfiguration.
    """

    def __init__(
        self,
        domains: tuple[SAETrainingDomain, ...],
        *,
        backend: str = "nccl",
        timeout: timedelta = timedelta(seconds=180),
    ):
        if not dist.is_initialized():
            raise RuntimeError("Initialize the routing world before SAERuntime")
        world_size = dist.get_world_size()
        seen_ranks: set[int] = set()
        seen_hooks: set[str] = set()
        seen_names: set[str] = set()
        dp_sizes = set()
        for domain in domains:
            ranks = domain.ranks
            if not ranks or tuple(sorted(set(ranks))) != ranks:
                raise ValueError(
                    "Training-domain ranks must be nonempty, sorted and unique"
                )
            if min(ranks) < 0 or max(ranks) >= world_size:
                raise ValueError("Training-domain rank is outside the routing world")
            if domain.tp_size < 1 or len(ranks) % domain.tp_size:
                raise ValueError("Training-domain size must be divisible by TP size")
            if seen_ranks.intersection(ranks):
                raise ValueError("Co-located hooks must share one training domain")
            if domain.name in seen_names or seen_hooks.intersection(domain.hooks):
                raise ValueError(
                    "Training-domain names and hook assignments must be unique"
                )
            if len(set(domain.hooks)) != len(domain.hooks):
                raise ValueError("Duplicate hook in training domain")
            seen_ranks.update(ranks)
            seen_hooks.update(domain.hooks)
            seen_names.add(domain.name)
            dp_sizes.add(len(ranks) // domain.tp_size)
        if len(dp_sizes) > 1:
            raise ValueError(
                "Static routing requires the same DP size for every placement"
            )

        require_megatron_core()
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection

        self.domains = domains
        self.generation = 0
        self.local: SAEParallelContext | None = None
        self.endpoints: tuple[SAEEndpoint, ...] = ()
        self._owned_groups: list[dist.ProcessGroup] = []
        self._closed = False
        rank = dist.get_rank()
        cache: dict[tuple[str, tuple[int, ...]], Any] = {}

        def create(ranks: tuple[int, ...], group_backend: str = backend):
            key = (group_backend, ranks)
            if key not in cache:
                group = parallel_state.create_group(
                    ranks=list(ranks),
                    backend=group_backend,
                    timeout=timeout,
                    group_desc="sae_runtime",
                )
                cache[key] = group
                if rank in ranks:
                    self._owned_groups.append(group)
            return cache[key]

        try:
            for domain in domains:
                domain_group = create(domain.ranks)
                generator = parallel_state.RankGenerator(
                    tp=domain.tp_size,
                    ep=1,
                    dp=len(domain.ranks) // domain.tp_size,
                    pp=1,
                    cp=1,
                    order="tp-cp-ep-dp-pp",
                )
                local_groups = {}
                for axis in ("tp", "dp", "pp"):
                    for logical_ranks in generator.get_ranks(axis):
                        members = tuple(domain.ranks[i] for i in logical_ranks)
                        group = create(members)
                        cpu_group = create(members, "gloo") if axis == "tp" else None
                        if rank in members:
                            local_groups[axis] = group
                            if axis == "tp":
                                local_groups["tp_cpu"] = cpu_group
                if rank not in domain.ranks:
                    continue
                tp, dp, singleton = (local_groups[k] for k in ("tp", "dp", "pp"))
                groups = ProcessGroupCollection(
                    tp=tp,
                    dp=dp,
                    pp=singleton,
                    cp=singleton,
                    ep=singleton,
                    mp=tp,
                    tp_cp=tp,
                    dp_cp=dp,
                    expt_tp=tp,
                    expt_dp=dp,
                    tp_ep=tp,
                    tp_ep_pp=tp,
                    intra_dp_cp=dp,
                    intra_expt_dp=dp,
                    tp_dp_cp=domain_group,
                )
                self.local = SAEParallelContext(
                    domain=domain,
                    groups=groups,
                    tp_cpu_group=local_groups["tp_cpu"],
                    tp_rank=dist.get_rank(tp),
                    dp_rank=dist.get_rank(dp),
                    tp_ranks=tuple(dist.get_process_group_ranks(tp)),
                    dp_ranks=tuple(dist.get_process_group_ranks(dp)),
                    receive_rank=dist.get_global_rank(tp, 0),
                )
            # Checkpoint completion spans every hook placement, but never
            # includes producer-only ranks from the routing world.
            training_group = create(tuple(sorted(seen_ranks))) if seen_ranks else None
            self.training_group = training_group if self.local is not None else None
            self.control_group = create(tuple(range(world_size)), "gloo")
            report = None
            if self.local is not None and self.local.tp_rank == 0:
                context = self.local
                placement = domains.index(context.domain)
                report = SAEEndpoint(
                    index=context.dp_rank * len(domains) + placement,
                    domain=context.domain.name,
                    placement_index=placement,
                    replica_index=context.dp_rank,
                    tp_ranks=context.tp_ranks,
                    receive_rank=context.receive_rank,
                    hooks=context.domain.hooks,
                )
            reports: list[Any] = [None] * world_size
            dist.all_gather_object(reports, report, group=self.control_group)
            self.endpoints = tuple(
                sorted((r for r in reports if r is not None), key=lambda r: r.index)
            )
            expected = sum(len(d.ranks) // d.tp_size for d in domains)
            if len(self.endpoints) != expected:
                raise RuntimeError("Megatron endpoint membership report is incomplete")
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_layout(
        cls,
        *,
        dp_size: int,
        tp_size: int,
        placement_size: int = 1,
        offset: int = 0,
        hooks: tuple[str, ...] = (),
        backend: str = "nccl",
    ) -> SAERuntime:
        """Translate placement configuration into domains before Megatron builds groups."""
        if dp_size < 0 or tp_size < 1 or placement_size < 1:
            raise ValueError("Invalid SAE layout dimensions")
        if hooks and len(hooks) < placement_size:
            raise ValueError("Every SAE placement needs at least one hook")
        domains = []
        for placement in range(placement_size) if dp_size else ():
            base, extra = divmod(len(hooks), placement_size)
            start = placement * base + min(placement, extra)
            assigned = hooks[start : start + base + (placement < extra)]
            ranks = tuple(
                offset + (replica * placement_size + placement) * tp_size + t
                for replica in range(dp_size)
                for t in range(tp_size)
            )
            domains.append(
                SAETrainingDomain(f"placement_{placement}", ranks, tp_size, assigned)
            )
        return cls(tuple(domains), backend=backend)

    def require_local(self) -> SAEParallelContext:
        if self._closed:
            raise RuntimeError("SAE runtime is closed")
        if self.local is None:
            raise RuntimeError("This rank is outside the SAE training domains")
        return self.local

    def validate_model(self, model: Any) -> None:
        context = self.require_local()
        if getattr(model, "_tp_group", None) is not context.tp_group:
            raise ValueError("SAE model and runtime must share the same TP group")

    def close(self) -> None:
        """Release only this runtime's groups, retaining the default routing world."""
        if self._closed:
            return
        self._closed = True
        from megatron.core import parallel_state

        # create_group registers handles for Megatron's timeout utilities. Remove
        # our handles on teardown without clearing another model's global state.
        registered = parallel_state._global_process_group_list
        if registered is not None:
            owned = {id(g) for g in self._owned_groups}
            registered[:] = [g for g in registered if id(g) not in owned]
        if dist.is_initialized():
            for group in reversed(self._owned_groups):
                dist.destroy_process_group(group)
        self._owned_groups.clear()
