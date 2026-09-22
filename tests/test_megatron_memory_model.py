"""Accounting invariants: GA, DP residency, and asynchronous allocator frees."""
from dataclasses import replace

import pytest

from sae_lens.autoconfig.megatron_memory_model import (
    MegatronMemoryConfig,
    estimate_tensor_payloads,
    replay_allocator_window,
)


def test_ga_shrinks_features_but_not_states_or_preloaded_window():
    cfg = MegatronMemoryConfig()
    one = estimate_tensor_payloads(cfg)
    four = estimate_tensor_payloads(replace(cfg, ga=4))
    for term in ("parameters", "adam_moments", "cached_inputs", "persistent_payload"):
        assert one[term] == four[term]
    assert one["full_feature_tensor"] == 4 * four["full_feature_tensor"]


def test_tp_shards_parameters_but_not_full_features_and_dp_keeps_gradients():
    tp = estimate_tensor_payloads(MegatronMemoryConfig(tp=4))
    dp = estimate_tensor_payloads(MegatronMemoryConfig(dp=4))
    assert tp["gradients"] == 0
    assert dp["gradients"] == dp["parameters"]
    assert tp["full_feature_tensor"] == 4 * dp["full_feature_tensor"]
    # Replicated decoder bias is the exception to parameter /TP scaling.
    assert 4 * tp["parameters"] - dp["parameters"] == 3 * 3 * 4096 * 4


@pytest.mark.parametrize("kwargs", [{"ga": 0}, {"dp": 3}, {"tp": 3}, {"d_in": -1}])
def test_reject_invalid_shapes(kwargs):
    with pytest.raises(ValueError):
        MegatronMemoryConfig(**kwargs)


def test_pending_free_is_active_until_completion_and_segment_maps_add_reservation():
    initial = dict(segments=[dict(device=0, total_size=2048, blocks=[
        dict(address=100, size=512, requested_size=4, state="active_allocated", frames=[])
    ])], trace_index=0)
    events = [dict(action="free_requested", addr=100, size=4),
              dict(action="segment_map", addr=10000, size=2048),
              dict(action="alloc", addr=200, size=1024, frames=[]),
              dict(action="free_completed", addr=100, size=4)]
    snapshot = dict(device_traces=[events], segments=[dict(device=0, total_size=4096, blocks=[
        dict(address=200, size=1024, requested_size=1024, state="active_allocated", frames=[])
    ])])
    result = replay_allocator_window(snapshot, initial, 0, [])
    assert result["peak_allocated"] == 1024
    assert result["peak_active"] == 1536
    assert result["pending_free_at_allocated_peak"] == 512
    assert result["peak_reserved"] == 4096
    assert result["final_accounting_error"] == 0


def test_malformed_trace_cannot_silently_reuse_active_storage():
    start = dict(segments=[], trace_index=0)
    snapshot = dict(device_traces=[[
        dict(action="alloc", addr=1, size=1, frames=[]),
        dict(action="alloc", addr=1, size=1, frames=[]),
    ]], segments=[])
    with pytest.raises(ValueError, match="active address"):
        replay_allocator_window(snapshot, start, 0, [])
