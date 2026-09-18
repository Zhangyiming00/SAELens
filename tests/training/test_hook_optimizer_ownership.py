"""Native shard view identity must not hide duplicate full-model owners."""

from types import SimpleNamespace

import pytest
import torch

from sae_lens.training.sae_train_unit import UnitOptimizers


def test_distinct_shard_views_cannot_share_full_model_owner():
    model = torch.nn.Linear(2, 2)
    units = {
        hook: SimpleNamespace(
            model=model,
            optimizer=SimpleNamespace(
                param_groups=[
                    {"params": [p.detach().view(-1) for p in model.parameters()]}
                ]
            ),
        )
        for hook in ("h0", "h1")
    }
    assert (
        units["h0"].optimizer.param_groups[0]["params"][0]
        is not units["h1"].optimizer.param_groups[0]["params"][0]
    )
    with pytest.raises(ValueError, match="full model ownership"):
        UnitOptimizers(units)
