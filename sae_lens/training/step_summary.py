"""Detached scalar/statistic results for fit; no latent/reconstruction storage."""
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainStepSummary:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    feature_firing_counts: torch.Tensor | None
    n_tokens: int
    reconstruction_mse: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def summarize_step(output) -> TrainStepSummary:
    mse = output.losses.get("mse_loss")
    if mse is None:
        mse = (output.sae_out - output.sae_in).square().mean()
    # Evaluate closures now rather than retaining a captured output/graph.
    metrics = {}
    for name, value in getattr(output, "metrics", {}).items():
        value = value() if callable(value) else value
        metrics[name] = value.detach() if isinstance(value, torch.Tensor) else value
    counts = output.feature_firing_counts
    return TrainStepSummary(
        output.loss.detach(), {k: v.detach() for k, v in output.losses.items()},
        counts.detach() if counts is not None else None, output.sae_in.shape[0],
        mse.detach(), metrics,
    )
