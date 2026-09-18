"""Megatron FP32 updates with per-hook clipping kept on the GPU."""

import torch
import torch.distributed as dist
from megatron.core.optimizer.optimizer import FP32Optimizer


@torch.no_grad()
def clip_grads_on_device(norm_grads, grads, *, device, group, max_norm):
    """Fused multi-tensor norm/scale over explicitly owned FP32 gradients.

    norm_grads may exclude TP replicas, while grads includes every local
    gradient that must be scaled. Empty DP shards still enter the group norm.
    Only tiny per-tensor norms are stacked; no full-gradient square temporary
    is materialized. The coefficient remains a device tensor for foreach_mul.
    """
    if norm_grads:
        norms = torch._foreach_norm([g.detach().float() for g in norm_grads], 2)
        norm = torch.linalg.vector_norm(torch.stack(norms))
    else:
        norm = torch.zeros((), device=device, dtype=torch.float32)
    if group is not None and group.size() > 1:
        squared = norm.square()
        dist.all_reduce(squared, group=group)
        norm = squared.sqrt()
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    if grads:
        torch._foreach_mul_(grads, coefficient)
    return norm


class GPUClipFP32Optimizer(FP32Optimizer):
    """Inherit native prepare/step; return the pre-clip norm as a CUDA scalar."""

    @torch.no_grad()
    def clip_grad_norm(self, clip_grad: float) -> torch.Tensor:
        parameters = self.get_parameters()
        # Megatron's TP metadata includes each shard and counts replicated
        # b_dec only on TP rank zero. DP replicas must not enter this norm.
        return clip_grads_on_device(
            self.get_main_grads_for_grad_norm(),
            [p.grad for p in parameters if p.grad is not None],
            device=parameters[0].device,
            group=self.get_grad_stats_parallel_group(), max_norm=clip_grad,
        )
