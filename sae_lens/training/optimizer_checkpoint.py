"""Parameter-group metadata for checkpoints with named, TP-reshardable moments."""

from copy import deepcopy

from sae_lens.training.megatron_optimizer import MEGATRON_GROUP_METADATA


def optimizer_state_for_loading(optimizer, state):
    """Backfill native group identifiers in older flat Adam checkpoints."""
    groups = []
    for live, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
        metadata = {key: live[key] for key in MEGATRON_GROUP_METADATA if key in live}
        groups.append({**metadata, **saved})
    return {**state, "param_groups": groups}


def save_parameter_groups(optimizer, named_parameters):
    if getattr(optimizer, "_sae_distributed_optimizer", False):
        assert len(optimizer.param_groups) == 1
        return [{**deepcopy({k: v for k, v in optimizer.param_groups[0].items() if k != "params"}),
                 "params": [name for name, _ in named_parameters]}]
    names = {id(param): name for name, param in named_parameters}
    groups = []
    for group in optimizer.param_groups:
        members = [names[id(p)] for p in group["params"] if id(p) in names]
        if members:
            groups.append(
                {
                    **deepcopy({k: v for k, v in group.items() if k != "params"}),
                    "params": members,
                }
            )
    return groups


def load_parameter_groups(optimizer, named_parameters, saved_groups):
    if getattr(optimizer, "_sae_distributed_optimizer", False):
        names = {name for name, _ in named_parameters}
        if len(saved_groups) != 1 or set(saved_groups[0]["params"]) != names:
            raise ValueError("Distributed Adam requires the existing single full-model parameter group")
        optimizer.param_groups[0].update(deepcopy({k: v for k, v in saved_groups[0].items() if k != "params"}))
        return
    names = {id(param): name for name, param in named_parameters}
    live = {}
    for group in optimizer.param_groups:
        members = frozenset(names[id(p)] for p in group["params"] if id(p) in names)
        if members:
            live[members] = group
    if set(live) != {frozenset(g["params"]) for g in saved_groups}:
        raise ValueError("Checkpoint optimizer parameter groups do not match the model")
    for saved in saved_groups:
        group = live[frozenset(saved["params"])]
        params = group["params"]
        metadata = {key: group[key] for key in MEGATRON_GROUP_METADATA if key in group}
        # Preserve the live dictionary: UnitOptimizers and the scheduler reference it.
        group.clear()
        group.update(metadata)
        group.update(deepcopy({k: v for k, v in saved.items() if k != "params"}))
        group["params"] = params


def restore_legacy_learning_rates(optimizer, scheduler):
    """Old named checkpoints omitted groups; recover the scheduler's current LR."""
    for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr(), strict=True):
        group["lr"] = lr


def select_scheduler_groups(state, indices):
    """Split the schedulers supported by get_lr_scheduler for old flat checkpoints."""
    state = deepcopy(state)
    for key in ("base_lrs", "_last_lr", "lr_lambdas"):
        if key in state:
            state[key] = [state[key][i] for i in indices]
    if "_schedulers" in state:
        state["_schedulers"] = [
            select_scheduler_groups(s, indices) for s in state["_schedulers"]
        ]
    return state


class UnitLRSchedulers:
    """Advance independent hook schedulers in the same fixed order as optimizers."""

    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self, hooks=None):
        for hook in self.schedulers if hooks is None else hooks:
            self.schedulers[hook].step()

    def get_last_lr(self):
        return [
            lr
            for scheduler in self.schedulers.values()
            for lr in scheduler.get_last_lr()
        ]

    def state_dict(self):
        return {
            "by_hook": {hook: s.state_dict() for hook, s in self.schedulers.items()}
        }

    def load_state_dict(self, state):
        offset = 0
        for hook, scheduler in self.schedulers.items():
            count = len(scheduler.optimizer.param_groups)
            local = (
                state["by_hook"][hook]
                if "by_hook" in state
                else select_scheduler_groups(state, range(offset, offset + count))
            )
            scheduler.load_state_dict(local)
            offset += count
