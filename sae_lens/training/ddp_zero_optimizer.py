"""Optional ZeRO-1 optimizer support for replicated DDP training."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Adam, Optimizer


logger = logging.getLogger(__name__)


def build_adam_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    adam_kwargs: dict[str, Any],
    zero_redundancy: bool,
    ddp_enabled: bool,
    dp_group: dist.ProcessGroup | None,
) -> Optimizer:
    """Build Adam, optionally sharding its state and updates across DDP ranks."""
    parameters = list(params)
    if not zero_redundancy:
        return Adam(parameters, **adam_kwargs)
    if not dist.is_available() or not dist.is_initialized() or dp_group is None:
        logger.info(
            "DDP ZeRO optimizer requested without a multi-rank DP group; "
            "using regular Adam."
        )
        return Adam(parameters, **adam_kwargs)
    if not ddp_enabled:
        raise ValueError("ddp_zero_optimizer requires sae_dp_mode='ddp'")
    if dist.get_world_size(dp_group) <= 1:
        logger.info(
            "DDP ZeRO optimizer requested with DP size 1; using regular Adam."
        )
        return Adam(parameters, **adam_kwargs)

    # Import lazily so the default-off path does not initialize PyTorch's
    # experimental distributed optimizer package or emit its warnings.
    from torch.distributed.optim import ZeroRedundancyOptimizer

    optimizer = ZeroRedundancyOptimizer(
        parameters,
        optimizer_class=Adam,
        process_group=dp_group,
        # Existing trainer code controls cross-hook overlap. PyTorch's
        # overlap_with_ddp path requires owning DDP's single comm-hook slot.
        overlap_with_ddp=False,
        **adam_kwargs,
    )
    optimizer._saelens_zero_optimizer = True
    local_numel = sum(
        parameter.numel()
        for group in optimizer.optim.param_groups
        for parameter in group["params"]
    )
    total_numel = sum(parameter.numel() for parameter in parameters)
    logger.info(
        "Enabled DDP ZeRO optimizer on DP rank %d/%d: local Adam ownership "
        "%d/%d parameters (%.1f%%)",
        dist.get_rank(dp_group),
        dist.get_world_size(dp_group),
        local_numel,
        total_numel,
        100.0 * local_numel / max(1, total_numel),
    )
    return optimizer


def is_zero_optimizer(optimizer: Optimizer) -> bool:
    return bool(getattr(optimizer, "_saelens_zero_optimizer", False))


def consolidate_optimizer_state(optimizer: Optimizer, *, to: int) -> None:
    """Gather a ZeRO optimizer's state on one process-group rank."""
    if not is_zero_optimizer(optimizer):
        return
    group = optimizer.process_group  # type: ignore[attr-defined]
    group_rank = dist.get_rank(group)
    device = optimizer._default_device  # type: ignore[attr-defined]
    consolidated: dict[int, dict[str, Any]] = {}
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    for parameter in parameters:
        owns = optimizer_owns_parameter(optimizer, parameter)
        owner = torch.tensor(group_rank if owns else -1, device=device)
        dist.all_reduce(owner, op=dist.ReduceOp.MAX, group=group)
        owner_group_rank = int(owner.item())
        owner_global_rank = dist.get_global_rank(group, owner_group_rank)
        local_state = (
            local_optimizer_parameter_state(optimizer, parameter) if owns else {}
        )
        metadata = [
            (
                key,
                torch.is_tensor(value),
                value.dtype if torch.is_tensor(value) else None,
                tuple(value.shape) if torch.is_tensor(value) else None,
                None if torch.is_tensor(value) else value,
            )
            for key, value in local_state.items()
        ] if owns else None
        payload: list[Any] = [metadata]
        dist.broadcast_object_list(
            payload, src=owner_global_rank, group=group, device=device
        )
        received: dict[str, Any] = {}
        for key, is_tensor, dtype, shape, value in payload[0] or []:
            if not is_tensor:
                received[key] = value
                continue
            tensor = (
                local_state[key].detach().to(device).contiguous()
                if owns
                else torch.empty(shape, dtype=dtype, device=device)
            )
            dist.broadcast(tensor, src=owner_global_rank, group=group)
            if group_rank == to:
                received[key] = tensor.detach().cpu()
        if group_rank == to:
            consolidated[id(parameter)] = received
    setattr(optimizer, "_saelens_consolidated_state_by_parameter", consolidated)


def optimizer_state_by_parameter(
    optimizer: Optimizer,
) -> dict[int, dict[str, Any]]:
    """Return optimizer state keyed by parameter identity.

    ZeRO callers must consolidate to the current rank before calling this.
    """
    if not is_zero_optimizer(optimizer):
        return {
            id(parameter): state
            for parameter, state in optimizer.state.items()
            if isinstance(parameter, torch.Tensor)
        }

    streamed = getattr(optimizer, "_saelens_consolidated_state_by_parameter", None)
    if streamed is not None:
        return streamed
    state_dict = optimizer.state_dict()
    result: dict[int, dict[str, Any]] = {}
    for live_group, saved_group in zip(
        optimizer.param_groups, state_dict["param_groups"]
    ):
        for parameter, index in zip(live_group["params"], saved_group["params"]):
            state = state_dict["state"].get(index)
            if state:
                result[id(parameter)] = state
    return result


def clear_optimizer_state(optimizer: Optimizer) -> None:
    optimizer.state.clear()
    if is_zero_optimizer(optimizer):
        optimizer.optim.state.clear()  # type: ignore[attr-defined]
        optimizer._all_state_dicts.clear()  # type: ignore[attr-defined]
        setattr(optimizer, "_saelens_consolidated_state_by_parameter", {})


def optimizer_owns_parameter(optimizer: Optimizer, parameter: torch.Tensor) -> bool:
    if not is_zero_optimizer(optimizer):
        return True
    return any(
        candidate is parameter
        for group in optimizer.optim.param_groups  # type: ignore[attr-defined]
        for candidate in group["params"]
    )


def local_optimizer_parameter_state(
    optimizer: Optimizer, parameter: torch.Tensor
) -> dict[str, Any]:
    if is_zero_optimizer(optimizer):
        return optimizer.optim.state.get(parameter, {})  # type: ignore[attr-defined]
    return optimizer.state.get(parameter, {})


def clear_optimizer_parameter_state(
    optimizer: Optimizer, parameter: torch.Tensor
) -> None:
    """Release one parameter's live optimizer state without touching others."""
    if is_zero_optimizer(optimizer):
        optimizer.optim.state.pop(parameter, None)  # type: ignore[attr-defined]
        consolidated = getattr(
            optimizer, "_saelens_consolidated_state_by_parameter", None
        )
        if consolidated is not None:
            consolidated.pop(id(parameter), None)
    else:
        optimizer.state.pop(parameter, None)


def set_optimizer_parameter_state(
    optimizer: Optimizer,
    parameter: torch.Tensor,
    state: dict[str, Any],
) -> None:
    """Install state only on the ZeRO rank that owns the parameter."""
    if not optimizer_owns_parameter(optimizer, parameter):
        return
    copied_state = {
        key: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
        for key, value in state.items()
    }
    if is_zero_optimizer(optimizer):
        optimizer.optim.state[parameter] = copied_state  # type: ignore[attr-defined]
        optimizer._all_state_dicts.clear()  # type: ignore[attr-defined]
    else:
        optimizer.state[parameter] = copied_state
