"""Parameter-group metadata for checkpoints with named, TP-reshardable moments."""

from copy import deepcopy


def save_parameter_groups(optimizer, named_parameters):
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
        # Preserve the live dictionary: UnitOptimizers and the scheduler reference it.
        group.clear()
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
