"""Sample canonical parameters and Adam state across a changing TP layout.

This is a sampled migration check, not a full numerical equivalence oracle.
"""

import torch
import torch.distributed as dist


@torch.no_grad()
def snapshot(session):
    groups = session.groups
    result = {}
    for h, cfg in session.configs.items():
        values = torch.zeros(cfg.d_sae, 11, device=groups.device)
        bias = torch.zeros(cfg.d_in, 3, device=groups.device)
        if groups.rank in groups.active_ranks:
            model = session.state.models[h]
            optimizer = session.state.optimizers[h]
            ids = model.feature_shard.ids(groups.device)
            for i, name in enumerate(
                ("encoder.weight", "encoder.bias", "decoder.weight")
            ):
                p = model.get_parameter(name)
                adam = optimizer.state.get(p, {})
                for j, tensor in enumerate(
                    (p, adam.get("exp_avg"), adam.get("exp_avg_sq"))
                ):
                    # A full-pool GA start can restore a producer before its
                    # first optimizer update; Adam moments are still lazy.
                    if tensor is None:
                        continue
                    sample = (
                        tensor[:, 0] if i == 0 else tensor if i == 1 else tensor[-1, :]
                    )
                    values[ids, i * 3 + j] = sample
            if groups.rank == groups.active_ranks[0]:
                values[:, 9] = session.state.replicated[h + "/since_fired"]
                values[:, 10] = session.state.replicated[h + "/firing_counts"]
                p = model.b_dec
                bias[:, 0].copy_(p)
                adam = optimizer.state.get(p, {})
                for j, name in enumerate(("exp_avg", "exp_avg_sq"), start=1):
                    if name in adam:
                        bias[:, j].copy_(adam[name])
        dist.all_reduce(values, group=groups.transfer)
        dist.all_reduce(bias, group=groups.transfer)
        result[h] = (values, bias)
    groups.synchronize()
    return result
