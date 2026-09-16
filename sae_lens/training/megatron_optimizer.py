"""Per-hook native Megatron updates with SAELens Adam/checkpoint semantics."""

from __future__ import annotations

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


def is_megatron_optimizer(optimizer) -> bool:
    # Keep the optional Megatron dependency out of ordinary SAELens training.
    if isinstance(optimizer, Optimizer):
        return False
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    return isinstance(optimizer, MegatronOptimizer)


def scheduler_optimizer(optimizer):
    """PyTorch schedulers need the base Optimizer; its groups are native live views."""
    return optimizer.optimizer if is_megatron_optimizer(optimizer) else optimizer


def build_runtime_optimizer(model, runtime: SAERuntime, *, adam_kwargs) -> Optimizer | MegatronOptimizer:
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

    from megatron.core.optimizer.optimizer import FP32Optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    kwargs = {**adam_kwargs, "fused": True}
    kwargs.pop("foreach", None)
    adam = torch.optim.Adam(parameters, **kwargs)
    for group in adam.param_groups:
        group.update(MEGATRON_GROUP_METADATA)
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
        use_distributed_optimizer=False,
    )
    optimizer = FP32Optimizer(adam, config, init_state_fn=None)
    context = runtime.require_local()
    # Explicit groups, including TP1 singletons: never reduce norms across
    # independent DP copies or fall back to Megatron's global parallel state.
    # Native TP metadata counts replicated b_dec on TP rank zero only.
    optimizer.tp_group = context.tp_group
    optimizer.grad_stats_parallel_group = context.tp_group
    return optimizer
