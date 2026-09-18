"""Per-hook native Megatron updates with SAELens Adam/checkpoint semantics."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch.optim import Optimizer

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    from sae_lens.sae_runtime import SAERuntime


# Required by native FP32Optimizer.load_state_dict's group matching. These
# describe one ordinary SAE group and do not modify its LR or weight decay.
MEGATRON_GROUP_METADATA = dict(
    wd_mult=1.0, lr_mult=1.0, is_expert_parallel=False, is_decoupled_lr=False
)


@lru_cache(maxsize=None)
def _warmup_optimizer_cuda(device: torch.device, adam_type=torch.optim.Adam) -> None:
    """Load update kernels before any training collective can await a peer.

    CUDA lazy module loading may synchronize the context. Loading sqrt/Adam
    for the first time behind an incomplete NCCL operation can then hold the
    context lock needed by the failure monitor's communicator abort. Use only
    disposable tensors/state; no SAE parameter, moment, step or RNG is touched.
    This setup-only synchronization is outside training and profiling windows.
    """
    parameter = torch.nn.Parameter(torch.zeros(64, device=device))
    parameter.grad = torch.zeros_like(parameter)
    torch.ones((), device=device).div_(3)  # AMP normalization coefficient kernel.
    norm = parameter.grad.square().sum().sqrt()
    parameter.grad.mul_((1.0 / (norm + 1e-6)).clamp(max=1.0))
    from sae_lens.training.gpu_clip_optimizer import clip_grads_on_device

    clip_grads_on_device(
        [parameter.grad], [parameter.grad], device=device, group=None, max_norm=1.0
    )
    kwargs = (
        dict(fused=True) if adam_type is torch.optim.Adam else dict(adam_w_mode=False)
    )
    adam_type([parameter], lr=0.0, **kwargs).step()
    torch.cuda.current_stream(device).synchronize()


def is_megatron_optimizer(optimizer) -> bool:
    # Keep the optional Megatron dependency out of ordinary SAELens training.
    if isinstance(optimizer, Optimizer):
        return False
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    return isinstance(optimizer, MegatronOptimizer)


def scheduler_optimizer(optimizer):
    """PyTorch schedulers need the base Optimizer; its groups are native live views."""
    return optimizer.optimizer if is_megatron_optimizer(optimizer) else optimizer


def build_runtime_optimizer(
    model, runtime: SAERuntime, *, adam_kwargs, ddp=None
) -> Optimizer | MegatronOptimizer:
    """Let FP32Optimizer own prepare/clip/step, once per hook and update window.

    Megatron's public FP32Optimizer accepts a base torch optimizer. Preserve
    SAELens' exact Adam groups, epsilon, decay mode, fused kernel and moment
    layout instead of applying language-model factory parameter-group rules.
    Autocast still uses FP32 model parameters and the existing shared scaler.
    """
    parameters = list(model.parameters())
    if parameters[0].device.type != "cuda":
        # Megatron 0.16's optimizer and clipping require CUDA. This is only the
        # CPU layout/reference compatibility path, never the CUDA runtime.
        return torch.optim.Adam(parameters, **adam_kwargs)
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Static Megatron optimization requires FP32 model parameters")
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    from sae_lens.training.gpu_clip_optimizer import GPUClipFP32Optimizer

    if ddp is model and getattr(model, "_sae_single_replica_fast_path", False):
        from sae_lens import logger

        logger.info(
            "Distributed optimizer requested with DP=1: using native FP32Optimizer "
            "and parameter.grad (single-replica fast path; no sharding benefit)"
        )
        ddp = None
    kwargs = {**adam_kwargs, "fused": True}
    kwargs.pop("foreach", None)
    adam_type = torch.optim.Adam
    if ddp is not None:
        # Native DistributedOptimizer validates against its selected Adam class:
        # TE, Apex, then torch. Installing either extension changes that class.
        from megatron.core.optimizer.distrib_optimizer import Adam

        adam_type = Adam
        if adam_type is not torch.optim.Adam:
            kwargs.pop("fused", None)
            kwargs["adam_w_mode"] = False  # Preserve coupled L2 decay, not AdamW.
    _warmup_optimizer_cuda(parameters[0].device, adam_type)
    adam = adam_type(parameters, **kwargs)
    for group in adam.param_groups:
        group.update(MEGATRON_GROUP_METADATA)
    if ddp is not None and not ddp.ddp_config.use_distributed_optimizer:
        raise ValueError(
            "Native DistributedOptimizer requires distributed-optimizer DDP buffers"
        )
    sharded = ddp is not None
    config = OptimizerConfig(
        optimizer="adam",
        lr=adam.defaults["lr"],
        adam_beta1=adam.defaults["betas"][0],
        adam_beta2=adam.defaults["betas"][1],
        adam_eps=adam.defaults["eps"],
        weight_decay=adam.defaults["weight_decay"],
        decoupled_weight_decay=False,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        use_distributed_optimizer=sharded,
    )
    context = runtime.require_local()
    if sharded:
        from sae_lens.training.distributed_optimizer import GPUClipDistributedOptimizer

        optimizer = GPUClipDistributedOptimizer(
            adam,
            config,
            grad_scaler=None,
            init_state_fn=None,
            model_chunks=[ddp],
            per_model_buffers={0: ddp.buffers},
            data_parallel_group=context.dp_group,
            data_parallel_group_gloo=None,
            data_parallel_group_idx=context.tp_rank,
            distributed_optimizer_instance_id=0,
        )
        optimizer.bind_model(model, runtime)
    else:
        optimizer = GPUClipFP32Optimizer(adam, config, init_state_fn=None)
    # Explicit groups, including TP1 singletons: never reduce norms across
    # independent DP copies or fall back to Megatron's global parallel state.
    # Native TP metadata counts replicated b_dec on TP rank zero only.
    optimizer.tp_group = context.tp_group
    optimizer.grad_stats_parallel_group = (
        context.groups.tp_dp_cp if sharded else context.tp_group
    )
    optimizer._sae_single_replica_fast_path = getattr(
        model, "_sae_single_replica_fast_path", False
    )
    return optimizer
