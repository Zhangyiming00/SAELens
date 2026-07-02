import math

from scripts.tests.capture_memory_estimator import (
    CaptureMemoryProfile,
    estimate_capture_memory,
    interpolate_tmp_gib,
)


def test_interpolate_tmp_uses_clamped_effective_tokens() -> None:
    profile = CaptureMemoryProfile(
        profiled_mbt=32768,
        persistent_per_token_gib=0.01,
        tmp_by_tokens_gib={512: 1.0, 1024: 2.0, 2048: 4.0},
    )

    assert interpolate_tmp_gib(profile, effective_tokens=256) == 1.0
    assert interpolate_tmp_gib(profile, effective_tokens=1024) == 2.0
    assert interpolate_tmp_gib(profile, effective_tokens=4096) == 4.0


def test_interpolate_tmp_uses_log2_linear_interpolation() -> None:
    profile = CaptureMemoryProfile(
        profiled_mbt=32768,
        persistent_per_token_gib=0.01,
        tmp_by_tokens_gib={512: 1.0, 2048: 5.0},
    )

    assert math.isclose(interpolate_tmp_gib(profile, effective_tokens=1024), 3.0)


def test_estimate_caps_tmp_lookup_tokens_by_requested_mbt() -> None:
    profile = CaptureMemoryProfile(
        profiled_mbt=32768,
        persistent_per_token_gib=0.01,
        tmp_by_tokens_gib={512: 0.5, 1024: 1.0, 2048: 2.0},
    )

    row = estimate_capture_memory(profile, requested_mbt=1024, tokens=2048)

    assert row.effective_tmp_tokens == 1024
    assert row.persistent_gib == 20.48
    assert row.lookup_tmp_gib == 1.0


def test_estimate_promotes_tmp_to_persistent_when_persistent_is_larger() -> None:
    profile = CaptureMemoryProfile(
        profiled_mbt=32768,
        persistent_per_token_gib=0.01,
        tmp_by_tokens_gib={512: 0.5, 1024: 1.0, 2048: 2.0},
    )

    row = estimate_capture_memory(profile, requested_mbt=2048, tokens=2048)

    assert row.lookup_tmp_gib == 2.0
    assert row.tmp_gib == 20.48
    assert row.peak_delta_gib == 40.96
