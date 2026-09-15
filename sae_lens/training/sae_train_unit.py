"""Independent hook ownership with a synchronous optimizer compatibility view."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any

from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from sae_lens.sae_runtime import SAERuntime


@dataclass
class SAETrainUnit:
    hook_name: str
    model: Any
    ddp: Any
    optimizer: Optimizer
    parallel_context: SAERuntime

    def __post_init__(self):
        self.parallel_context.validate_model(self.model)
        context = self.parallel_context.require_local()
        if isinstance(self.ddp, DistributedDataParallel):
            if (
                self.ddp.module is not self.model
                or self.ddp.process_group is not context.dp_group
            ):
                raise ValueError(
                    "SAE unit DDP must wrap its model using the runtime DP group"
                )
        elif self.ddp is not self.model:
            raise ValueError("SAE unit requires its own model or DDP wrapper")
        elif context.dp_group.size() > 1:
            raise ValueError("A multi-replica SAE unit requires its own DDP wrapper")
        model_params = {id(p) for p in self.model.parameters()}
        optimizer_params = [
            id(p) for g in self.optimizer.param_groups for p in g["params"]
        ]
        if set(optimizer_params) != model_params or len(optimizer_params) != len(
            model_params
        ):
            raise ValueError(
                "Each SAE optimizer must own exactly its hook's parameters"
            )

    def forward(self, step_input):
        self.check_failure()
        return self.ddp(step_input)

    def backward(self, loss, scaler):
        self.check_failure()
        scaler.scale(loss).backward()
        monitor = getattr(self.parallel_context, "failure_monitor", None)
        if monitor is not None:
            monitor.complete_backward(
                self.parallel_context.require_local().domain, self.hook_name
            )

    def check_failure(self):
        monitor = getattr(self.parallel_context, "failure_monitor", None)
        if monitor is not None:
            monitor.check()

    def finish_grad_sync(self):
        # PyTorch DDP finishes reduction as backward returns. This explicit
        # boundary is where Megatron DDP's finish_grad_sync will be connected.
        self.check_failure()
        self.model.sync_tensor_parallel_gradients()

    def clip_grad_norm(self, max_norm=1.0):
        self.check_failure()
        return self.model.clip_grad_norm_(max_norm)

    def step(self):
        self.check_failure()
        return self.optimizer.step()


class _UnitStates(MutableMapping):
    """Route checkpoint reads/writes to the owning optimizer without copying state."""

    def __init__(self, units):
        self.owners = {
            p: unit.optimizer
            for unit in units.values()
            for group in unit.optimizer.param_groups
            for p in group["params"]
        }

    def __getitem__(self, parameter):
        return self.owners[parameter].state[parameter]

    def __setitem__(self, parameter, value):
        self.owners[parameter].state[parameter] = value

    def __delitem__(self, parameter):
        del self.owners[parameter].state[parameter]

    def __iter__(self) -> Iterator:
        return (p for p, owner in self.owners.items() if p in owner.state)

    def __len__(self):
        return sum(p in owner.state for p, owner in self.owners.items())

    def __contains__(self, parameter):
        return parameter in self.owners and parameter in self.owners[parameter].state


class UnitOptimizers(Optimizer):
    """Keep existing schedulers/scalers/checkpoint formats over independent Adam instances.

    This object owns no Adam state or update algorithm. Parameter groups and
    state entries refer to the hook optimizers, and step visits units in order.
    """

    def __init__(self, units: dict[str, SAETrainUnit]):
        parameters = [
            p
            for u in units.values()
            for g in u.optimizer.param_groups
            for p in g["params"]
        ]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("SAE training units must not share parameters")
        self.units = units
        groups = [g for u in units.values() for g in u.optimizer.param_groups]
        super().__init__(groups, {})
        self.state = _UnitStates(units)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("SAE units do not support optimizer closures")
        for unit in self.units.values():
            unit.step()

    def zero_grad(self, set_to_none=True):
        for unit in self.units.values():
            unit.optimizer.zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict replaces state/group containers. Transfer
        # those loaded values back to their owners and reinstate the live view.
        groups = self.param_groups
        super().load_state_dict(state_dict)
        loaded_state = self.state
        for live, loaded in zip(groups, self.param_groups, strict=True):
            live.clear()
            live.update(loaded)
        self.param_groups = groups
        self.state = _UnitStates(self.units)
        self.state.clear()
        self.state.update(loaded_state)
