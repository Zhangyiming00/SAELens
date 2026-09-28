"""Megatron runtime adapter for the existing cross-hook TP forward schedule."""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from sae_lens import logger
from sae_lens.training.megatron_ddp import is_megatron_ddp
from sae_lens.training.multi_hook_sae import (
    PendingWavefrontOutputs,
    forward_tp_wavefront,
)


def configure_tp_wavefront(trainer, units):
    from sae_lens.config import resolve_tp_overlap
    resolve_tp_overlap(trainer.cfg)
    mode = getattr(trainer.cfg, "multi_sae_distributed_architecture", "legacy_per_hook_wrapper")
    if mode not in ("legacy_per_hook_wrapper", "unified_multi_hook"):
        raise ValueError("Invalid multi_sae_distributed_architecture")
    context = trainer.runtime.require_local()
    device = next(next(iter(units.values())).model.parameters()).device
    reason = None
    if mode != "unified_multi_hook":
        reason = "TP overlap is off"
    elif len(units) < 2:
        reason = "one local hook"
    elif context.tp_group.size() == 1:
        reason = "TP=1"
    elif not all(
        callable(getattr(u.model, "tp_wavefront_supported", None))
        and u.model.tp_wavefront_supported()
        and u.model._tp_group is context.tp_group
        and (u.ddp is u.model or is_megatron_ddp(u.ddp))
        for u in units.values()
    ):
        reason = "requires CUDA Megatron models and native per-hook wrappers"
    # All members must choose the same forward collective sequence, including
    # when one member requests the legacy control or cannot use the fast path.
    supported = torch.tensor(int(reason is None), device=device)
    if context.tp_group.size() * context.dp_group.size() > 1:
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=context.groups.tp_dp_cp)
    enabled = bool(supported.item())
    if not enabled and reason is None:
        reason = "disabled by another rank in this placement"
    schedule = getattr(trainer.cfg, "multi_sae_tp_wavefront_schedule", "bounded")
    window = getattr(trainer.cfg, "multi_sae_tp_wavefront_max_live_hooks", 2)
    if schedule not in ("eager", "lazy", "bounded"):
        raise ValueError("Unknown TP overlap schedule")
    if type(window) is not int or window < 1:
        raise ValueError("TP overlap max-live-hooks must be positive")
    if enabled:
        # Mixed schedules/windows on TP/DP members would change collective order.
        setting = torch.tensor([("eager", "lazy", "bounded").index(schedule), window], device=device)
        low, high = setting.clone(), setting.clone()
        if context.tp_group.size() * context.dp_group.size() > 1:
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=context.groups.tp_dp_cp)
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=context.groups.tp_dp_cp)
        if not torch.equal(low, high):
            raise ValueError("All ranks must use the same TP overlap schedule and window")
        if schedule != "eager":
            # Interleaved updates are only valid for independent SAE parameters.
            seen = set()
            for unit in units.values():
                ids = {id(p) for p in unit.model.parameters()}
                if seen.intersection(ids):
                    raise ValueError("Lazy/bounded TP overlap requires independent hook parameters")
                seen.update(ids)
    trainer._runtime_tp_wavefront_schedule = schedule
    trainer._runtime_tp_wavefront_max_live_hooks = window
    trainer._runtime_tp_wavefront = enabled
    trainer._runtime_tp_wavefront_reason = reason
    trainer._runtime_tp_overlap = enabled
    trainer._runtime_tp_overlap_schedule = schedule if enabled else 'off'
    logger.info(
        "SAE TP overlap requested=%s effective=%s placement=%s hooks=%s "
        "reason=%s; per-hook DDP/optimizer ownership is unchanged",
        schedule if mode == "unified_multi_hook" else "off", enabled, context.domain.name, list(units), reason,
    )
    logger.info("SAE TP overlap schedule=%s max_live_hooks=%s", schedule if enabled else "off",
                window if schedule == "bounded" else "all")


def runtime_wavefront_forward(trainer, units, inputs):
    @contextmanager
    def forward_context(hook):
        unit = units[hook]
        unit.wait_params()
        unit.check_failure()
        # Megatron DDP.forward is a module passthrough. Its grad hooks remain
        # installed and the normal per-unit backward owns no_sync/readiness.
        # Runtime parameter gathers belong to the window scheduler, never a
        # forward pre-hook that a phased forward would bypass.
        if is_megatron_ddp(unit.ddp) and unit.ddp.remove_forward_pre_hook_handles:
            raise RuntimeError("TP wavefront cannot bypass native parameter-gather pre-hooks")
        yield

    schedule = getattr(trainer, "_runtime_tp_wavefront_schedule", "bounded")
    if schedule != "eager":
        return PendingWavefrontOutputs(
            list(units), {h: u.model for h, u in units.items()}, inputs,
            forward_context,
            autocast_context=lambda: trainer.autocast_if_enabled,
            max_live_hooks=(0 if schedule == "lazy" else
                            getattr(trainer, "_runtime_tp_wavefront_max_live_hooks", 2)),
        )
    with trainer.autocast_if_enabled:
        return forward_tp_wavefront(
            list(units), {h: u.model for h, u in units.items()}, inputs,
            forward_context,
        )
