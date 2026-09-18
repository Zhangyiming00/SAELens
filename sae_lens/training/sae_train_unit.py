"""Independent hook ownership with a synchronous optimizer compatibility view."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from sae_lens.profiling import cuda_nvtx_range
from sae_lens.sae_runtime import SAERuntime
from sae_lens.training.megatron_ddp import is_megatron_ddp, supports_early_grad_sync
from sae_lens.training.megatron_optimizer import is_megatron_optimizer
from sae_lens.training.optimizer_checkpoint import optimizer_state_for_loading

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer


class HookPhase(str, Enum):
    ACCUMULATING = "accumulating"
    GRAD_COMM = "gradient_communication"
    GRADS_READY = "gradients_ready"
    UPDATING = "updating"
    PARAMS_ENQUEUED = "parameters_enqueued"
    PARAMS_READY = "parameters_ready"


@dataclass
class SAETrainUnit:
    hook_name: str
    model: Any
    ddp: Any
    optimizer: Optimizer | MegatronOptimizer
    parallel_context: SAERuntime
    _grad_sync_started: bool = field(default=False, init=False)
    _grad_sync_finished: bool = field(default=False, init=False)
    early_grad_sync: bool = field(default=False, init=False)
    phase: HookPhase = field(default=HookPhase.PARAMS_READY, init=False)
    params_ready: Any = field(default=None, init=False)
    update_done: Any = field(default=None, init=False)
    param_gather_pending: bool = field(default=False, init=False)
    update_count: int = field(default=0, init=False)

    def wait_params(self):
        """Order the caller stream after the update; this is not a host wait."""
        if self.param_gather_pending:
            raise RuntimeError("Hook parameters require the deferred native all-gather")
        if self.params_ready is not None:
            torch.cuda.current_stream().wait_event(self.params_ready)

    def params_complete(self):
        """Unlike an enqueued wait, query reports actual GPU completion."""
        complete = not self.param_gather_pending and (
            self.params_ready is None or self.params_ready.query()
        )
        if complete and self.phase == HookPhase.PARAMS_ENQUEUED:
            self.phase = HookPhase.PARAMS_READY
        return complete

    def __post_init__(self):
        self.parallel_context.validate_model(self.model)
        context = self.parallel_context.require_local()
        if is_megatron_ddp(self.ddp):
            if (
                self.ddp.module is not self.model
                or self.ddp.dp_group is not context.dp_group
            ):
                raise ValueError("SAE unit Megatron DDP must use its runtime DP group")
        elif isinstance(self.ddp, DistributedDataParallel):
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
        sharded = getattr(self.optimizer, "_sae_distributed_optimizer", False)
        if sharded:
            if (
                self.optimizer.sae_model is not self.model
                or self.optimizer.model_chunks != [self.ddp]
            ):
                raise ValueError(
                    "Distributed optimizer must own this hook and DDP buffers"
                )
        elif set(optimizer_params) != model_params or len(optimizer_params) != len(
            model_params
        ):
            raise ValueError(
                "Each SAE optimizer must own exactly its hook's parameters"
            )
        self.early_grad_sync = supports_early_grad_sync(self.ddp, self.parallel_context)
        if (
            is_megatron_ddp(self.ddp)
            and context.tp_group.size() > 1
            and context.dp_group.size() > 1
        ):
            # A rank-local environment/version check must not choose different
            # collective orders within one domain. Agree once during setup,
            # before any training window; fall back together if any rank cannot
            # safely overlap communicators. This is not a gradient collective.
            supported = torch.tensor(
                int(self.early_grad_sync), device=next(self.model.parameters()).device
            )
            dist.all_reduce(
                supported, op=dist.ReduceOp.MIN, group=context.groups.tp_dp_cp
            )
            self.early_grad_sync = bool(supported.item())

    def forward(self, step_input):
        self.wait_params()
        self.check_failure()
        return self.ddp(step_input)

    def backward(self, loss, scaler, *, sync_gradients=False):
        if self._grad_sync_started:
            raise RuntimeError(
                "Cannot accumulate gradients after starting hook reduction"
            )
        self.check_failure()
        if sync_gradients:
            if not self.early_grad_sync or not is_megatron_ddp(self.ddp):
                raise RuntimeError(
                    "Native gradient synchronization is not enabled for this hook"
                )
            # This is the last backward allowed to write this window. Megatron
            # owns bucket readiness and may dispatch from its autograd hooks.
            # Reserve ownership before backward: failures may leave a subset
            # of buckets in flight. Never explicitly start those buckets again.
            self._grad_sync_started = True
            self.phase = HookPhase.GRAD_COMM
        scaler.scale(loss).backward()
        monitor = getattr(self.parallel_context, "failure_monitor", None)
        if monitor is not None:
            with cuda_nvtx_range(f"sae:{self.hook_name}:backward_ready_wait"):
                monitor.complete_backward(
                    self.parallel_context.require_local().domain, self.hook_name
                )

    def check_failure(self):
        monitor = getattr(self.parallel_context, "failure_monitor", None)
        if monitor is not None:
            monitor.check()

    def start_grad_sync(self):
        """Explicit fallback launch; a native final backward already owns sync."""
        self.check_failure()
        if is_megatron_ddp(self.ddp):
            if self._grad_sync_started:
                return
            # A later bucket may fail after an earlier one was enqueued. Keep
            # ownership on failure so cleanup cannot clear a live main_grad.
            self._grad_sync_started = True
            self.phase = HookPhase.GRAD_COMM
            self.ddp.start_grad_sync()

    def finish_grad_sync(self):
        """Establish stream dependencies and expose gradients for window update."""
        self.check_failure()
        if is_megatron_ddp(self.ddp):
            if not self._grad_sync_started:
                raise RuntimeError("Start hook gradient reduction before finishing it")
            if self._grad_sync_finished:
                return
            self.ddp.finish_grad_sync()
            if getattr(self.optimizer, "_sae_distributed_optimizer", False):
                self.optimizer.expose_valid_grads()
            else:
                for parameter in self.model.parameters():
                    parameter.grad = parameter.main_grad
        self.model.sync_tensor_parallel_gradients()
        self._grad_sync_finished = True
        self.phase = HookPhase.GRADS_READY

    def zero_grad(self):
        self.wait_params()
        if self._grad_sync_started and not self._grad_sync_finished:
            raise RuntimeError("Cannot clear main_grad while hook reduction is pending")
        self.optimizer.zero_grad(set_to_none=True)
        if is_megatron_ddp(self.ddp):
            self.ddp.zero_grad_buffer()
        self._grad_sync_started = self._grad_sync_finished = False
        self.phase = HookPhase.ACCUMULATING

    def no_sync(self):
        return self.ddp.no_sync() if hasattr(self.ddp, "no_sync") else nullcontext()

    def finish_window(
        self, token_count, *, gradients_are_mean=False, normalization_in_unscale=False
    ):
        """Convert accumulated token-sum gradients to the global token mean."""
        if is_megatron_ddp(self.ddp):
            # Unknown-length short windows and unsupported ordering use an
            # explicit launch. Native final backward skips this launch; its
            # first window is dispatched by Megatron's finish_grad_sync itself.
            self.start_grad_sync()
            self.finish_grad_sync()
        else:
            # CPU/reference wrappers execute every microbatch under no_sync.
            import torch.distributed as dist

            group = self.parallel_context.require_local().dp_group
            for parameter in self.model.parameters():
                if parameter.grad is not None and group.size() > 1:
                    dist.all_reduce(parameter.grad, group=group)
                if parameter.grad is not None and parameter.is_cuda:
                    # Direct autograd grads were allocated on the compute
                    # stream and may be freed by zero_grad on the update stream.
                    parameter.grad.record_stream(torch.cuda.current_stream())
            self.model.sync_tensor_parallel_gradients()
            self._grad_sync_finished = True
            self.phase = HookPhase.GRADS_READY
        if token_count and not (gradients_are_mean or normalization_in_unscale):
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(token_count)

    def clip_grad_norm(self, max_norm=1.0):
        if is_megatron_ddp(self.ddp) and not self._grad_sync_finished:
            raise RuntimeError("Finish hook gradient reduction before clipping")
        self.check_failure()
        if is_megatron_optimizer(self.optimizer):
            return self.optimizer.clip_grad_norm(max_norm)
        return self.model.clip_grad_norm_(max_norm)

    def step(self):
        if is_megatron_ddp(self.ddp) and not self._grad_sync_finished:
            raise RuntimeError("Finish hook gradient reduction before optimizer update")
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
        # Native shard tensors are distinct view objects even if a caller
        # accidentally reuses a full model parameter in two hooks. Audit the
        # full owners as well as the optimizer's local shard objects.
        model_parameters = [p for u in units.values() for p in u.model.parameters()]
        if len({id(p) for p in model_parameters}) != len(model_parameters):
            raise ValueError(
                "SAE training units must not share parameters (full model ownership)"
            )
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
        # Native DistributedOptimizer owns non-leaf FP32 buffer views. This
        # compatibility facade must not re-register them with PyTorch's leaf
        # validator; the native optimizer already owns and validated them.
        super().__init__([{"params": []}], {})
        self.param_groups = groups
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
        super().load_state_dict(optimizer_state_for_loading(self, state_dict))
        loaded_state = self.state
        for live, loaded in zip(groups, self.param_groups, strict=True):
            live.clear()
            live.update(loaded)
        self.param_groups = groups
        self.state = _UnitStates(self.units)
        self.state.clear()
        self.state.update(loaded_state)
