"""Normalize valid-token sums in the shared scaler's existing unscale pass.

This adapter follows torch 2.10 GradScaler.unscale_. It retains its optimizer
stage and found-inf bookkeeping and never changes the window's loss scale.
The runtime still owns cross-rank overflow agreement and one window update.
"""
import torch
from torch.amp.grad_scaler import OptState


@torch.no_grad()
def unscale_with_token_mean(scaler, optimizer, tokens: int):
    if not scaler.is_enabled() or tokens <= 0:
        raise ValueError('Requires an enabled scaler and positive global token count')
    scaler._check_scale_growth_tracker('unscale_')
    state = scaler._per_optimizer_states[id(optimizer)]
    if state['stage'] is OptState.UNSCALED:
        raise RuntimeError('Gradients have already been unscaled for this window')
    if state['stage'] is OptState.STEPPED:
        raise RuntimeError('Cannot unscale after optimizer step')
    # Keep the shared loss-scale tensor unchanged throughout the window.
    inverse = scaler._scale.double().reciprocal().float()
    # Gradients are FP32, including native DistributedOptimizer's valid views.
    # Folding this scalar preserves scaled backward and avoids another scan.
    inverse.div_(tokens)
    found = torch.zeros((), dtype=torch.float32, device=inverse.device)
    state['found_inf_per_device'] = scaler._unscale_grads_(
        optimizer, inverse, found, False
    )
    state['stage'] = OptState.UNSCALED
