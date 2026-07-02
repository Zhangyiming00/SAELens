from __future__ import annotations

import torch

from sae_lens.training.gpu_streaming_activation_provider import (
    GpuStreamingActivationProvider,
)


def test_gpu_provider_shuffle_keeps_hook_alignment(monkeypatch):
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda _n, device=None: torch.tensor([3, 1, 2, 0], device=device),
    )
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook_a", "hook_b"],
        is_multi_sae=True,
        train_batch_size_tokens=4,
        d_model=1,
        shuffle=True,
        device="cpu",
    )
    provider.receive_chunk(torch.arange(8, dtype=torch.float32).reshape(8, 1))

    batch = next(provider)

    assert batch["hook_a"].flatten().tolist() == [3, 1, 2, 0]
    assert batch["hook_b"].flatten().tolist() == [7, 5, 6, 4]


def test_gpu_provider_mixing_matches_standard_buffer_config(monkeypatch):
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda _n, device=None, generator=None: torch.tensor(  # noqa: ARG005
            [5, 0, 7, 2, 1, 6, 3, 4], device=device
        ),
    )
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        shuffle=True,
        buffer_size=8,
        mix_fraction=0.5,
        device="cpu",
    )

    provider.receive_chunk(torch.arange(4, dtype=torch.float32).reshape(4, 1))
    try:
        next(provider)
        raise AssertionError("provider should wait for a full mixing buffer")
    except StopIteration:
        pass

    provider.receive_chunk(torch.arange(4, 8, dtype=torch.float32).reshape(4, 1))

    first = next(provider)
    second = next(provider)
    assert first.flatten().tolist() == [5, 0]
    assert second.flatten().tolist() == [7, 2]


def test_gpu_provider_desired_refill_chunks_targets_high_watermark():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        buffer_size=8,
        device="cpu",
    )

    assert provider.desired_refill_chunks() == 1

    provider.receive_chunk(torch.arange(2, dtype=torch.float32).reshape(2, 1))

    assert provider.desired_refill_chunks() == 3


def test_prefill_target_tokens_clamps_to_serving_window():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        buffer_size=8,
        mix_fraction=0.5,
        device="cpu",
    )

    assert provider.prefill_target_tokens(4) == 4
    assert provider.prefill_target_tokens(8) == 4
    assert provider.prefill_target_tokens(2) == 2
    assert provider.prefill_target_tokens(0) == 0


def test_prefill_target_tokens_no_mixing():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        device="cpu",
    )

    assert provider.prefill_target_tokens(10) == 10


def test_prefill_satisfied_requires_serving_tokens():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        buffer_size=8,
        mix_fraction=0.5,
        shuffle=False,
        device="cpu",
    )

    assert not provider.prefill_satisfied(4)

    provider.receive_chunk(torch.arange(8, dtype=torch.float32).reshape(8, 1))

    assert provider.serving_tokens() == 4
    assert provider.storage_tokens() == 4
    assert provider.prefill_satisfied(4)


def test_receiver_refill_targets_storage_but_respects_resident_budget():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        buffer_size=8,
        mix_fraction=0.5,
        shuffle=False,
        device="cpu",
    )

    for start in range(0, 8, 2):
        provider.receive_chunk(torch.arange(start, start + 2, dtype=torch.float32).reshape(2, 1))

    assert provider.serving_tokens() == 4
    assert provider.storage_tokens() == 4
    assert provider.receiver_needs_refill()
    assert provider.desired_receiver_refill_chunks() == 2

    provider.receive_chunk(torch.arange(8, 10, dtype=torch.float32).reshape(2, 1))
    assert provider.receiver_needs_refill()
    assert provider.desired_receiver_refill_chunks() == 1

    provider.receive_chunk(torch.arange(10, 12, dtype=torch.float32).reshape(2, 1))

    assert provider.serving_tokens() == 8
    assert provider.storage_tokens() == 4
    assert not provider.receiver_needs_refill()


def test_receiver_refill_resumes_after_trainer_consumes_budget():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=2,
        d_model=1,
        buffer_size=8,
        mix_fraction=0.5,
        shuffle=False,
        device="cpu",
    )

    for start in range(0, 12, 2):
        provider.receive_chunk(torch.arange(start, start + 2, dtype=torch.float32).reshape(2, 1))

    assert not provider.receiver_needs_refill()

    next(provider)

    assert provider.serving_tokens() == 6
    assert provider.storage_tokens() == 4
    assert provider.receiver_needs_refill()
    assert provider.desired_receiver_refill_chunks() == 1


def test_receiver_needs_refill_no_mixing_mode():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook"],
        is_multi_sae=False,
        train_batch_size_tokens=4,
        d_model=1,
        device="cpu",
    )

    assert provider.receiver_needs_refill()
    provider.receive_chunk(torch.arange(4, dtype=torch.float32).reshape(4, 1))
    assert not provider.receiver_needs_refill()


def test_serving_and_storage_tokens_multi_hook():
    provider = GpuStreamingActivationProvider(
        pp_hook_names=["hook_a", "hook_b"],
        is_multi_sae=True,
        train_batch_size_tokens=2,
        d_model=1,
        buffer_size=8,
        mix_fraction=0.5,
        shuffle=False,
        device="cpu",
    )

    provider.receive_chunk(torch.arange(16, dtype=torch.float32).reshape(16, 1))

    assert provider.serving_tokens() == 4
    assert provider.storage_tokens() == 4
