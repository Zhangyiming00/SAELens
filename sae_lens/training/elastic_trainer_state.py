"""Live trainer-state transfer used by streaming SAE-DP hot switches."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def broadcast_optimizer_state(
    *,
    source: torch.optim.Optimizer | None,
    target: torch.optim.Optimizer,
    group: dist.ProcessGroup,
    source_global_rank: int,
    device: torch.device,
) -> None:
    """Populate ``target`` from one live optimizer without a CPU checkpoint."""
    is_source = dist.get_rank() == source_global_rank
    if is_source and source is None:
        raise ValueError("source optimizer is required on source_global_rank")

    source_parameters = _optimizer_parameters(source) if source is not None else []
    target_parameters = _optimizer_parameters(target)
    if is_source and len(source_parameters) != len(target_parameters):
        raise ValueError("source and target optimizer parameter counts differ")

    group_options: list[dict[str, Any]] | None = None
    if is_source:
        assert source is not None
        group_options = [
            {key: value for key, value in param_group.items() if key != "params"}
            for param_group in source.param_groups
        ]
    object_payload: list[Any] = [group_options]
    dist.broadcast_object_list(
        object_payload,
        src=source_global_rank,
        group=group,
        device=device,
    )
    received_options = object_payload[0]
    if len(received_options) != len(target.param_groups):
        raise RuntimeError("optimizer parameter-group counts differ during hot switch")
    for target_group, options in zip(target.param_groups, received_options):
        target_group.update(options)

    for parameter_idx, target_parameter in enumerate(target_parameters):
        source_state = (
            source.state[source_parameters[parameter_idx]]
            if is_source and source is not None
            else None
        )
        metadata = None
        if source_state is not None:
            metadata = []
            for key, value in source_state.items():
                if isinstance(value, torch.Tensor):
                    metadata.append(
                        (key, True, value.dtype, tuple(value.shape), None)
                    )
                else:
                    metadata.append((key, False, None, None, value))
        state_payload: list[Any] = [metadata]
        dist.broadcast_object_list(
            state_payload,
            src=source_global_rank,
            group=group,
            device=device,
        )
        target_state: dict[str, Any] = {}
        for key, is_tensor, dtype, shape, value in state_payload[0]:
            if not is_tensor:
                target_state[key] = value
                continue
            if is_source:
                assert source_state is not None
                tensor = source_state[key].detach().to(device=device).contiguous()
            else:
                tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(
                tensor,
                src=source_global_rank,
                group=group,
            )
            target_state[key] = tensor
        target.state[target_parameter] = target_state


def _broadcast_tensor_dict(
    *,
    source: dict[str, torch.Tensor] | None,
    target: dict[str, torch.Tensor],
    group: dist.ProcessGroup,
    source_global_rank: int,
) -> None:
    is_source = dist.get_rank() == source_global_rank
    for name in target:
        if is_source:
            if source is None or name not in source:
                raise ValueError(f"source trainer state is missing {name!r}")
            tensor = source[name].detach().to(target[name].device).contiguous()
        else:
            tensor = torch.empty_like(target[name])
        dist.broadcast(tensor, src=source_global_rank, group=group)
        target[name] = tensor


def broadcast_multi_sae_trainer_state(
    *,
    source: Any | None,
    target: Any,
    group: dist.ProcessGroup,
    source_global_rank: int,
    device: torch.device,
) -> None:
    """Transfer all mutable MultiSAETrainer state at an optimizer boundary."""
    is_source = dist.get_rank() == source_global_rank
    if is_source and source is None:
        raise ValueError("source trainer is required on source_global_rank")

    scalar_state = None
    if source is not None and is_source:
        scalar_state = {
            "n_training_samples": source.n_training_samples,
            "n_training_steps": source.n_training_steps,
            "lr_scheduler": source.lr_scheduler.state_dict(),
            "grad_scaler": source.grad_scaler.state_dict(),
            "activation_scalers": {
                name: scaler.scaling_factor
                for name, scaler in source.activation_scaler_by_hook.items()
            },
            "n_frac_active_samples": dict(source.n_frac_active_samples_by_hook),
            "pending_sample_count": dict(source._pending_sample_count_by_hook),
            "pending_step_count": dict(source._pending_step_count_by_hook),
            "checkpoint_thresholds": list(source.checkpoint_thresholds),
            "elapsed_s": time.time() - source._t_ready,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
        }
    payload: list[Any] = [scalar_state]
    dist.broadcast_object_list(
        payload,
        src=source_global_rank,
        group=group,
        device=device,
    )
    state = payload[0]
    target.n_training_samples = int(state["n_training_samples"])
    target.n_training_steps = int(state["n_training_steps"])
    target.lr_scheduler.load_state_dict(state["lr_scheduler"])
    target.grad_scaler.load_state_dict(state["grad_scaler"])
    for name, value in state["activation_scalers"].items():
        target.activation_scaler_by_hook[name].scaling_factor = value
    target.n_frac_active_samples_by_hook = dict(state["n_frac_active_samples"])
    target._pending_sample_count_by_hook = dict(state["pending_sample_count"])
    target._pending_step_count_by_hook = dict(state["pending_step_count"])
    target.checkpoint_thresholds = list(state["checkpoint_thresholds"])
    target._t_ready = time.time() - float(state["elapsed_s"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device=device)

    broadcast_optimizer_state(
        source=source.optimizer if is_source else None,
        target=target.optimizer,
        group=group,
        source_global_rank=source_global_rank,
        device=device,
    )
    for hook_name, target_optimizer in target._overlap_optimizer_by_hook.items():
        source_optimizer = (
            source._overlap_optimizer_by_hook[hook_name]
            if is_source and source is not None
            else None
        )
        broadcast_optimizer_state(
            source=source_optimizer,
            target=target_optimizer,
            group=group,
            source_global_rank=source_global_rank,
            device=device,
        )

    _broadcast_tensor_dict(
        source=source.act_freq_scores_by_hook if is_source else None,
        target=target.act_freq_scores_by_hook,
        group=group,
        source_global_rank=source_global_rank,
    )
    _broadcast_tensor_dict(
        source=source.n_forward_passes_since_fired_by_hook if is_source else None,
        target=target.n_forward_passes_since_fired_by_hook,
        group=group,
        source_global_rank=source_global_rank,
    )
    _broadcast_tensor_dict(
        source=source._pending_did_fire_max_by_hook if is_source else None,
        target=target._pending_did_fire_max_by_hook,
        group=group,
        source_global_rank=source_global_rank,
    )
