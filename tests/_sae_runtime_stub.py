"""Runtime contract double for existing routing/role tests; no collective claims."""

from types import SimpleNamespace

import torch.distributed as dist

from sae_lens.sae_runtime import SAEEndpoint, SAETrainingDomain


class RuntimeStub:
    def __init__(self, domains):
        self.domains = domains
        self.endpoints = []
        self.local = None
        rank = dist.get_rank()
        for placement, domain in enumerate(domains):
            replicas = [
                domain.ranks[i : i + domain.tp_size]
                for i in range(0, len(domain.ranks), domain.tp_size)
            ]
            for dp_rank, members in enumerate(replicas):
                self.endpoints.append(
                    SAEEndpoint(
                        dp_rank * len(domains) + placement,
                        domain.name,
                        placement,
                        dp_rank,
                        members,
                        members[0],
                        domain.hooks,
                    )
                )
                if rank in members:
                    tp_rank = members.index(rank)
                    self.local = SimpleNamespace(
                        domain=domain,
                        tp_rank=tp_rank,
                        dp_rank=dp_rank,
                        tp_group=SimpleNamespace(ranks=members),
                        dp_group=SimpleNamespace(
                            ranks=tuple(r[tp_rank] for r in replicas)
                        ),
                        tp_cpu_group=SimpleNamespace(ranks=members),
                    )
        self.endpoints.sort(key=lambda e: e.index)

    @classmethod
    def from_layout(cls, *, dp_size, tp_size, placement_size=1, offset=0, hooks=(), backend="nccl"):
        domains = []
        for stage in range(placement_size) if dp_size else ():
            ranks = tuple(
                offset + (d * placement_size + stage) * tp_size + t
                for d in range(dp_size)
                for t in range(tp_size)
            )
            domains.append(SAETrainingDomain(str(stage), ranks, tp_size, hooks))
        return cls(tuple(domains))

    def require_local(self):
        assert self.local is not None
        return self.local

    def close(self):
        pass
