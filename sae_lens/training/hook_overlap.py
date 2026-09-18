"""Fixed-order cross-hook updates, negotiated once per placement."""

import os

import torch
import torch.distributed as dist

from sae_lens import logger
from sae_lens.training.megatron_ddp import is_megatron_ddp


def configure_hook_overlap(trainer, units):
    context = trainer.runtime.require_local()
    mode = getattr(trainer.cfg, "multi_sae_optimizer_overlap", "off")
    if os.environ.get("SAE_DDP_OPT_OVERLAP_V2") == "1":
        mode = "on"
    if mode not in ("off", "on", "non_tp_only"):
        raise ValueError("Unknown multi_sae_optimizer_overlap mode")
    device = next(next(iter(units.values())).model.parameters()).device
    reason = None
    if mode == "off":
        reason = "disabled by configuration"
    elif mode == "non_tp_only" and context.tp_group.size() > 1:
        reason = "non_tp_only with TP > 1"
    elif len(units) < 2:
        reason = "one local hook"
    elif device.type != "cuda" or not all(
        is_megatron_ddp(u.ddp)
        or (u.ddp is u.model and getattr(u.model, "_sae_single_replica_fast_path", False))
        for u in units.values()
    ):
        reason = "requires CUDA Megatron DDP or native single-replica gradients"
    elif context.tp_group.size() * context.dp_group.size() > 1 and not (
        torch.cuda.nccl.version() >= (2, 26, 0)
        and os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") == "1"
    ):
        reason = (
            "requires NCCL >= 2.26 and pre-initialization NCCL_LAUNCH_ORDER_IMPLICIT=1"
        )
    supported = torch.tensor(int(reason is None), device=device)
    if context.tp_group.size() * context.dp_group.size() > 1:
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=context.groups.tp_dp_cp)
    enabled = bool(supported.item())
    if not enabled and reason is None:
        reason = "disabled by another rank in this placement"
    trainer._runtime_optimizer_overlap = enabled
    trainer._runtime_optimizer_overlap_reason = reason
    trainer._runtime_optimizer_stream = (
        torch.cuda.Stream(device=device) if enabled else None
    )
    if enabled:
        # Parameters/buffers outlive every update. record_stream additionally
        # protects allocator reuse if another hook raises before the window's
        # normal join and the trainer is destroyed during failure cleanup.
        for unit in units.values():
            for parameter in unit.model.parameters():
                parameter.record_stream(trainer._runtime_optimizer_stream)
            for buffer in unit.ddp.buffers if is_megatron_ddp(unit.ddp) else ():
                buffer.grad_data.record_stream(trainer._runtime_optimizer_stream)
                if buffer.param_data is not None:
                    buffer.param_data.record_stream(trainer._runtime_optimizer_stream)
    requested_gather = getattr(trainer.cfg, "multi_sae_param_gather_schedule", "eager")
    if requested_gather not in ("eager", "one_hook_lag", "after_backward"):
        raise ValueError("Invalid multi_sae_param_gather_schedule")
    sharded = any(
        getattr(u.optimizer, "_sae_distributed_optimizer", False)
        for u in units.values()
    )
    gather_mode = "eager"
    gather_reason = None
    if not sharded:
        gather_reason = "ordinary optimizer has no parameter gather"
    elif not enabled:
        gather_reason = reason
    elif not getattr(trainer.cfg, "multi_sae_param_gather_overlap", True):
        gather_mode = "deferred"
        gather_reason = "parameter-gather overlap disabled"
    else:
        gather_mode = requested_gather
    if sharded and context.tp_group.size() * context.dp_group.size() > 1:
        code = ("eager", "deferred", "one_hook_lag", "after_backward").index(
            gather_mode
        )
        bounds = torch.tensor([code, -code], device=device)
        dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=context.groups.tp_dp_cp)
        low, negative_high = bounds.tolist()
        if low != -negative_high:
            raise ValueError(
                "Parameter gather schedules differ within the SAE placement"
            )
    trainer._runtime_param_gather_schedule = gather_mode
    trainer._runtime_defer_param_gather = gather_mode != "eager"
    trainer._runtime_param_gather_stream = (
        torch.cuda.Stream(device=device)
        if gather_mode in ("one_hook_lag", "after_backward")
        else None
    )
    if trainer._runtime_param_gather_stream is not None:
        for unit in units.values():
            for parameter in unit.model.parameters():
                parameter.record_stream(trainer._runtime_param_gather_stream)
            for buffer in unit.ddp.buffers:
                if buffer.param_data is not None:
                    buffer.param_data.record_stream(
                        trainer._runtime_param_gather_stream
                    )
    for unit in units.values():
        if (
            getattr(unit.optimizer, "_sae_distributed_optimizer", False)
            and trainer._runtime_defer_param_gather
        ):
            # Native step normally gathers on its caller (optimizer) stream.
            # Delayed schedules transfer sole gather ownership to the scheduler.
            # DDP was constructed without forward pre-hooks; the scheduler is
            # the sole owner of start/wait, so no forward can gather twice.
            unit.ddp.ddp_config.overlap_param_gather = True
            unit.optimizer.config.overlap_param_gather = True
    logger.info(
        "SAE parameter gather requested_schedule=%s effective_schedule=%s reason=%s",
        requested_gather,
        gather_mode,
        gather_reason,
    )
    adam_types = {
        h: type(getattr(u.optimizer, "optimizer", u.optimizer))
        for h, u in units.items()
    }
    logger.info(
        "SAE inner Adam implementations=%s (coupled weight decay)",
        {h: f"{cls.__module__}.{cls.__name__}" for h, cls in adam_types.items()},
    )
    logger.info(
        "SAE clip backend=torch foreach multi-tensor (device norm/coefficient)",
    )
    logger.info(
        "SAE gradient storage=%s",
        {h: "main_grad DDP buffers" if is_megatron_ddp(u.ddp) else "parameter.grad"
         for h, u in units.items()},
    )
    logger.info(
        "SAE optimizer backend=%s requested_overlap=%s effective_overlap=%s "
        "placement=%s hooks=%s reason=%s; collective submission follows fixed hook order",
        {h: type(u.optimizer).__name__ for h, u in units.items()},
        mode,
        enabled,
        context.domain.name,
        list(units),
        reason,
    )
