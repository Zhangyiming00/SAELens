"""Cutover-only state migration for independent native Megatron SAE units.

The old DP group reconstructs Adam moments on its permanent source rank before
any member leaves. GPU snapshots let us retire the old DDP buffers, parameters
and AccumulateGrad hooks before allocating their replacements. Bulk model and
Adam tensors never pass through host memory; Python metadata/RNG stay on host.
No checkpoint files, default-world collectives or migration work enter the training step.
TP and hook placement stay fixed; only DP ownership changes.
"""

from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class ElasticRuntimeState:
    models: dict[str, dict[str, Any]]
    optimizers: dict[str, dict[str, dict[str, Any]]]
    group_options: dict[str, list[dict[str, Any]]]
    scalars: dict[str, Any]
    tensors: dict[str, dict[str, torch.Tensor]]


_TENSORS = (
    "act_freq_scores_by_hook",
    "n_forward_passes_since_fired_by_hook",
    "_pending_did_fire_max_by_hook",
)
_SCALARS = (
    "n_training_samples", "n_training_steps", "n_frac_active_samples_by_hook",
    "_pending_sample_count_by_hook", "_pending_step_count_by_hook",
    "checkpoint_thresholds",
)


def _inner(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def _sharded(optimizer):
    return bool(getattr(optimizer, "_sae_distributed_optimizer", False))


def _snapshot(value):
    return value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)


def _capture_adam(unit, source_rank):
    """Gather named, unpadded moments; empty native shard intersections are valid."""
    optimizer = unit.optimizer
    inner = _inner(optimizer)
    group = unit.parallel_context.require_local().dp_group
    source = dist.get_rank() == source_rank
    result = {}
    if not _sharded(optimizer):
        if source:
            result = {
                name: {key: _snapshot(v) for key, v in inner.state.get(p, {}).items()}
                for name, p in unit.model.named_parameters()
            }
        return result

    local = {}
    for name, (_, shard, start, end) in optimizer.sae_shards.items():
        state = inner.state.get(shard, {})
        if state:
            step = state.get("step", inner.param_groups[0].get("step", 0))
            local[name] = (start, end, int(step), {
                k: v.dtype for k, v in state.items() if k != "step"
            })
    records = [None] * group.size()
    dist.all_gather_object(records, local, group=group)
    members = dist.get_process_group_ranks(group)
    for name, parameter in unit.model.named_parameters():
        pieces = [(rank, record[name]) for rank, record in zip(members, records)
                  if name in record]
        if not pieces:
            if source:
                result[name] = {}
            continue
        steps = {piece[2] for _, piece in pieces}
        cursor = 0
        for _, (start, end, _, _) in sorted(pieces, key=lambda p: p[1][0]):
            if start != cursor:
                raise RuntimeError(f"Incomplete Adam state for {name} at cutover")
            cursor = end
        if cursor != parameter.numel() or len(steps) != 1:
            raise RuntimeError(f"Inconsistent Adam state for {name} at cutover")
        dtypes = pieces[0][1][3]
        if any(piece[3] != dtypes for _, piece in pieces):
            raise RuntimeError(f"Inconsistent Adam moment keys for {name}")
        if source:
            result[name] = {key: torch.empty(parameter.shape, dtype=dtype, device=parameter.device)
                            for key, dtype in dtypes.items()}
            result[name]["step"] = torch.tensor(float(steps.pop()), device=parameter.device)
        # Only the permanent source needs the canonical moments. Transfer each
        # contiguous native shard directly into its destination GPU slice.
        # Broadcasting 4M-element chunks to every old replica incurred hundreds
        # of unnecessary collectives and temporary allocations for a large SAE.
        for owner, (start, end, _, _) in pieces:
            for key in dtypes:
                if source:
                    target = result[name][key].view(-1)[start:end]
                    if owner == source_rank:
                        shard = optimizer.sae_shards[name][1]
                        target.copy_(inner.state[shard][key].view(-1))
                    else:
                        dist.recv(target, src=owner, group=group)
                elif dist.get_rank() == owner:
                    shard = optimizer.sae_shards[name][1]
                    dist.send(inner.state[shard][key].view(-1), dst=source_rank, group=group)
    return result


def capture_runtime_state(trainer, *, source_global_rank: int):
    """All old DP members call at a completed optimizer boundary."""
    if (not trainer.units or getattr(trainer, "_runtime_update_pending", False)
            or getattr(trainer, "_gradient_window_active", False)):
        raise RuntimeError("Elastic migration requires completed native SAE updates")
    for unit in trainer.units.values():
        unit.wait_params()
    device = next(next(iter(trainer.units.values())).model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    source = dist.get_rank() == source_global_rank
    models, optimizers, options = {}, {}, {}
    for hook, unit in trainer.units.items():
        moments = _capture_adam(unit, source_global_rank)
        if source:
            models[hook] = {n: _snapshot(p) for n, p in unit.model.state_dict().items()}
            optimizers[hook] = moments
            options[hook] = [{k: _snapshot(v) for k, v in g.items() if k != "params"}
                             for g in _inner(unit.optimizer).param_groups]
    if not source:
        return None
    scalars = {n: copy.deepcopy(getattr(trainer, n)) for n in _SCALARS}
    scalars.update(
        lr_scheduler=copy.deepcopy(trainer.lr_scheduler.state_dict()),
        grad_scaler=copy.deepcopy(trainer.grad_scaler.state_dict()),
        activation_scalers={h: _snapshot(s.scaling_factor)
                            for h, s in trainer.activation_scaler_by_hook.items()},
        update_counts={h: u.update_count for h, u in trainer.units.items()},
        token_count_remainder=getattr(trainer, "_token_count_remainder", 0),
        # Exact-provider rotation advances per microbatch, not per optimizer
        # update. Keep it independent of n_training_steps when GA > 1.
        provider_step_index=getattr(trainer.data_provider, "step_index", None),
        elapsed_s=time.time() - trainer._t_ready,
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )
    tensors = {n: {h: _snapshot(t) for h, t in getattr(trainer, n).items()} for n in _TENSORS}
    return ElasticRuntimeState(models, optimizers, options, scalars, tensors)


def retire_runtime_trainer(trainer, *, collect=True):
    """Invalidate a stopped trainer and release its entire old native graph.

    Megatron 0.16 does not retain removable backward-hook handles. Rewrapping
    the same Parameters would leave old reducers registered on AccumulateGrad.
    New models at cutover avoid stale hooks and preserve the unmodified native
    DDP implementation used in steady state.
    """
    trainer._stop_device_sampler()
    trainer.__dict__.clear()
    trainer._elastic_retired = True
    if collect:
        gc.collect()


def broadcast_runtime_models(state, models, *, group, source_global_rank):
    source = dist.get_rank() == source_global_rank
    with torch.no_grad():
        for hook, model in models.items():
            extra_state = {}
            for name, parameter in model.state_dict().items():
                if not torch.is_tensor(parameter):
                    payload = [state.models[hook].pop(name) if source else None]
                    dist.broadcast_object_list(payload, src=source_global_rank, group=group,
                                               device=next(model.parameters()).device)
                    extra_state[name] = payload[0]
                    continue
                if source:
                    parameter.copy_(state.models[hook].pop(name))
                dist.broadcast(parameter, src=source_global_rank, group=group)
            if extra_state:
                model.load_state_dict(extra_state, strict=False)


@dataclass
class _TensorMetadata:
    dtype: torch.dtype
    shape: tuple[int, ...]
    host: bool


def _broadcast_metadata(value, *, group, source_global_rank, device):
    """Pickle only structure; transport tensor leaves through the device group.

    Scheduler/scaler options may contain CUDA tensors. Passing them directly to
    broadcast_object_list would silently stage those tensors through the CPU.
    Only originally-host tensors (small RNG states) are returned to the host.
    """
    tensors = []

    def pack(item):
        if torch.is_tensor(item):
            tensors.append(item)
            return _TensorMetadata(item.dtype, tuple(item.shape), item.device.type == "cpu")
        if isinstance(item, dict):
            return {k: pack(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(pack(v) for v in item)
        return item

    source = dist.get_rank() == source_global_rank
    payload = [pack(value) if source else None]
    dist.broadcast_object_list(payload, src=source_global_rank, group=group, device=device)
    tensor_iter = iter(tensors)

    def unpack(item):
        if isinstance(item, _TensorMetadata):
            tensor = (next(tensor_iter).to(device=device).contiguous() if source else
                      torch.empty(item.shape, dtype=item.dtype, device=device))
            dist.broadcast(tensor, src=source_global_rank, group=group)
            return tensor.cpu() if item.host else tensor
        if isinstance(item, dict):
            return {k: unpack(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(unpack(v) for v in item)
        return item

    return unpack(payload[0])


def restore_runtime_state(state, trainer, *, source_global_rank):
    """Broadcast canonical state and install only each new optimizer's shard."""
    context = trainer.runtime.require_local()
    group = context.dp_group
    source = dist.get_rank() == source_global_rank
    device = next(next(iter(trainer.units.values())).model.parameters()).device
    scalars, options = _broadcast_metadata(
        (state.scalars, state.group_options) if source else None,
        group=group, source_global_rank=source_global_rank, device=device,
    )
    for n in _SCALARS:
        setattr(trainer, n, scalars[n])
    trainer.lr_scheduler.load_state_dict(scalars["lr_scheduler"])
    trainer.grad_scaler.load_state_dict(scalars["grad_scaler"])
    trainer._token_count_remainder = scalars["token_count_remainder"]
    trainer._t_ready = time.time() - scalars["elapsed_s"]
    for hook, unit in trainer.units.items():
        unit.update_count = scalars["update_counts"][hook]
        factor = scalars["activation_scalers"][hook]
        trainer.activation_scaler_by_hook[hook].scaling_factor = (
            factor.to(device) if torch.is_tensor(factor) else factor
        )
        inner = _inner(unit.optimizer)
        if len(inner.param_groups) != len(options[hook]):
            raise RuntimeError("Adam group count changed at elastic cutover")
        for target, saved in zip(inner.param_groups, options[hook]):
            # Preserve implementation-specific defaults (torch fused vs TE).
            for key in ("lr", "initial_lr", "betas", "eps", "weight_decay", "amsgrad"):
                if key in saved:
                    target[key] = saved[key]
        restored_steps = set()
        for name, parameter in unit.model.named_parameters():
            saved = state.optimizers[hook].pop(name) if source else None
            meta = [{k: (v.dtype, tuple(v.shape)) if torch.is_tensor(v) else v
                     for k, v in saved.items()} if source else None]
            dist.broadcast_object_list(meta, src=source_global_rank, group=group, device=device)
            received = {}
            for key, desc in meta[0].items():
                if not isinstance(desc, tuple):
                    received[key] = desc
                    continue
                dtype, shape = desc
                tensor = saved[key].to(device) if source else torch.empty(shape, dtype=dtype, device=device)
                dist.broadcast(tensor, src=source_global_rank, group=group)
                received[key] = tensor
            if _sharded(unit.optimizer):
                if "step" in received:
                    restored_steps.add(int(received["step"]))
                    if len(restored_steps) > 1:
                        raise RuntimeError("Native distributed Adam requires a common per-hook step")
                    unit.optimizer.prepare_full_parameter_state({name: received})
                unit.optimizer.load_full_parameter_state(parameter, received)
            elif received:
                # CPU Adam's step lives on CPU; CUDA fused Adam keeps it on device.
                inner.state[parameter] = received
            del received, saved
        for attr in _TENSORS:
            target = getattr(trainer, attr)[hook]
            if source:
                target.copy_(state.tensors[attr].pop(hook))
            dist.broadcast(target, src=source_global_rank, group=group)
    # Model initialization at a cutover must not perturb the training RNG.
    torch.set_rng_state(scalars["torch_rng_state"].cpu())
    if scalars["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(scalars["cuda_rng_state"].cpu(), device)
    return scalars
