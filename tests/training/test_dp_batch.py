from unittest.mock import MagicMock, patch

import pytest
import torch

from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.standard_sae import StandardTrainingSAE
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.dp_batch import balanced_token_counts, local_token_budget
from sae_lens.training.exact_dp_batch_provider import ExactDataParallelBatchProvider
from sae_lens.training.logical_streaming_mixer import LogicalStreamingMixingProvider
from sae_lens.training.mixing_buffer import mixing_buffer
from sae_lens.training.sae_trainer import SAETrainer
from tests.helpers import (
    build_batchtopk_runner_cfg,
    build_runner_cfg,
    build_topk_sae_training_cfg,
)


def test_balanced_token_counts_requested_dp3_and_dp5() -> None:
    assert balanced_token_counts(4096, 3) == (1365, 1365, 1366)
    assert balanced_token_counts(4096, 5) == (819, 819, 819, 819, 820)


@pytest.mark.parametrize("tokens", [0, 1, 2, 37, 4096, 8193])
@pytest.mark.parametrize("replicas", range(1, 9))
def test_balanced_token_counts_conserves_tokens(tokens: int, replicas: int) -> None:
    counts = balanced_token_counts(tokens, replicas)
    assert sum(counts) == tokens
    assert max(counts) - min(counts) <= 1


def test_local_token_budget_conserves_full_steps_and_tail() -> None:
    budgets = [local_token_budget(3 * 4096 + 2, 4096, 5, i) for i in range(5)]
    assert sum(budgets) == 3 * 4096 + 2
    assert budgets == [2457, 2457, 2457, 2458, 2461]


@pytest.mark.parametrize("replicas", [3, 5])
def test_exact_mode_matches_legacy_local_sizes_when_divisible(replicas: int) -> None:
    global_batch = 4095
    global_tokens = global_batch * 7
    assert balanced_token_counts(global_batch, replicas) == (
        global_batch // replicas,
    ) * replicas
    assert tuple(
        local_token_budget(global_tokens, global_batch, replicas, idx)
        for idx in range(replicas)
    ) == (global_tokens // replicas,) * replicas


@pytest.mark.parametrize("replicas", [3, 5])
def test_weighted_local_means_match_full_global_gradient(replicas: int) -> None:
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(4096, 4, generator=generator)
    target = torch.randn(4096, 2, generator=generator)
    initial = torch.randn(4, 2, generator=generator)

    full_weight = initial.clone().requires_grad_(True)
    full_loss = (inputs @ full_weight - target).square().mean()
    full_loss.backward()
    assert full_weight.grad is not None

    counts = balanced_token_counts(4096, replicas)
    local_gradients = []
    cursor = 0
    for count in counts:
        local_weight = initial.clone().requires_grad_(True)
        local_loss = (
            inputs[cursor : cursor + count] @ local_weight
            - target[cursor : cursor + count]
        ).square().mean()
        (local_loss * (replicas * count / 4096)).backward()
        assert local_weight.grad is not None
        local_gradients.append(local_weight.grad)
        cursor += count

    ddp_average = torch.stack(local_gradients).mean(0)
    assert torch.allclose(ddp_average, full_weight.grad, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("replicas", [3, 5])
@pytest.mark.parametrize("use_sparse_activations", [False, True])
def test_weighted_topk_gradients_match_full_global_batch(
    replicas: int,
    use_sparse_activations: bool,
) -> None:
    generator = torch.Generator().manual_seed(29)
    cfg = build_topk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        rescale_acts_by_decoder_norm=False,
        use_sparse_activations=use_sparse_activations,
    )
    initial = TopKTrainingSAE(cfg)
    initial_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }
    inputs = torch.randn(4096, cfg.d_in, generator=generator)
    dead_mask = torch.zeros(cfg.d_sae, dtype=torch.bool)
    dead_mask[: cfg.d_sae // 2] = True

    def loss_for(sae: TopKTrainingSAE, batch: torch.Tensor) -> torch.Tensor:
        return sae.training_forward_pass(
            TrainStepInput(
                sae_in=batch,
                dead_neuron_mask=dead_mask,
                coefficients={},
                n_training_steps=0,
                is_logging_step=False,
            )
        ).loss

    full = TopKTrainingSAE(cfg)
    full.load_state_dict(initial_state)
    loss_for(full, inputs).backward()
    full_grads = [parameter.grad.detach().clone() for parameter in full.parameters()]

    local_grads: list[list[torch.Tensor]] = []
    cursor = 0
    for count in balanced_token_counts(4096, replicas):
        local = TopKTrainingSAE(cfg)
        local.load_state_dict(initial_state)
        (loss_for(local, inputs[cursor : cursor + count]) * replicas * count / 4096).backward()
        local_grads.append(
            [parameter.grad.detach().clone() for parameter in local.parameters()]
        )
        cursor += count

    for parameter_idx, full_grad in enumerate(full_grads):
        ddp_gradient = torch.stack(
            [gradients[parameter_idx] for gradients in local_grads]
        ).mean(0)
        assert torch.allclose(ddp_gradient, full_grad, atol=2e-6, rtol=2e-5)


def test_weighted_single_sae_sparsity_uses_global_sample_count() -> None:
    runner_cfg = build_runner_cfg(
        d_in=4,
        d_sae=8,
        training_tokens=2730,
        train_batch_size_tokens=1365,
        feature_sampling_window=1,
    )
    sae = StandardTrainingSAE.from_dict(runner_cfg.get_training_sae_cfg_dict())
    trainer = SAETrainer(
        cfg=runner_cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=iter([]),
        dp_group=MagicMock(),
        token_count_weighted_dp=True,
    )
    trainer.n_frac_active_samples = 1365
    trainer.act_freq_scores.fill_(1)
    trainer._reset_running_sparsity_stats = MagicMock()

    def fake_all_reduce(tensor: torch.Tensor, **_kwargs) -> None:
        if tensor.ndim == 0:
            tensor.fill_(4096)

    with (
        patch("sae_lens.training.sae_trainer.dist.get_world_size", return_value=3),
        patch(
            "sae_lens.training.sae_trainer.dist.all_reduce",
            side_effect=fake_all_reduce,
        ),
    ):
        trainer._train_step(sae=sae, sae_in=torch.randn(1365, 4))

    assert trainer.n_frac_active_samples == 4096 + 1365


def test_exact_mode_rejects_batch_level_topk() -> None:
    cfg = build_batchtopk_runner_cfg(
        training_tokens=4096,
        train_batch_size_tokens=4096,
    )
    cfg.streaming_mode = True
    cfg.streaming_dp_batch_mode = "exact"
    cfg.sae_dp_mode = "ddp"
    with pytest.raises(ValueError, match="batch-level TopK"):
        LanguageModelSAETrainingRunner(
            cfg,
            streaming_mode=True,
            sae_dp_size=3,
        )


def test_logical_streaming_mixer_matches_independent_legacy_mixers() -> None:
    global_batch = 12
    stream_count = 3
    local_batch = global_batch // stream_count
    raw_batches = [
        torch.arange(step * global_batch, (step + 1) * global_batch).reshape(-1, 1)
        for step in range(10)
    ]
    exact = LogicalStreamingMixingProvider(
        source=iter([batch.clone() for batch in raw_batches]),
        global_batch_size=global_batch,
        stream_count=stream_count,
        buffer_size_per_stream=8,
        mix_fraction=0.5,
        shuffle=True,
        seed=23,
    )

    legacy_outputs: list[list[torch.Tensor]] = []
    for stream_idx in range(stream_count):
        generator = torch.Generator().manual_seed(23 + stream_idx)
        local_inputs = [
            batch[stream_idx * local_batch : (stream_idx + 1) * local_batch]
            for batch in raw_batches
        ]
        legacy_outputs.append(
            list(
                mixing_buffer(
                    buffer_size=8,
                    batch_size=local_batch,
                    activations_loader=iter(local_inputs),
                    mix_fraction=0.5,
                    generator=generator,
                    shuffle=True,
                )
            )
        )

    expected = [
        torch.cat([legacy_outputs[stream][step] for stream in range(stream_count)])
        for step in range(len(legacy_outputs[0]))
    ]
    actual = list(exact)
    assert len(actual) == len(expected)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert sorted(torch.cat(actual)[:, 0].tolist()) == list(range(120))


@pytest.mark.parametrize("stream_count", [3, 5])
def test_logical_streaming_mixer_nondivisible_batch_conserves_chunked_source(
    stream_count: int,
) -> None:
    total_tokens = 2 * 4096
    raw_sizes = (997, 3000, 451, 3001, 743)
    assert sum(raw_sizes) == total_tokens
    cursor = 0
    raw_batches = []
    for size in raw_sizes:
        raw_batches.append(torch.arange(cursor, cursor + size).reshape(-1, 1))
        cursor += size

    stream_counts = balanced_token_counts(4096, stream_count)
    provider = LogicalStreamingMixingProvider(
        source=iter(raw_batches),
        global_batch_size=4096,
        stream_count=stream_count,
        buffer_size_per_stream=max(stream_counts),
        mix_fraction=0.5,
        shuffle=True,
        seed=31,
    )
    batches = list(provider)

    assert provider.stream_counts == stream_counts
    assert [batch.shape[0] for batch in batches] == [4096, 4096]
    observed = torch.cat(batches)[:, 0]
    assert sorted(observed.tolist()) == list(range(total_tokens))


def test_exact_provider_distributes_one_global_batch(monkeypatch) -> None:
    group = MagicMock()
    messages: list[torch.Tensor] = []

    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda _group, _rank: 17)

    def record_broadcast(tensor, *, src, group):
        assert src == 17
        messages.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, "broadcast", record_broadcast)
    source_batch = torch.arange(4096, dtype=torch.float32).reshape(-1, 1)
    source = ExactDataParallelBatchProvider(
        source=iter([source_batch]),
        dp_group=group,
        dp_idx=0,
        dp_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        d_model=1,
    )
    rank0 = next(source)
    assert torch.equal(rank0[:, 0], torch.arange(1365, dtype=torch.float32))

    recorded = [message.clone() for message in messages]
    for dp_idx, expected in ((1, source_batch[1365:2730]), (2, source_batch[2730:])):
        cursor = 0

        def replay_broadcast(tensor, *, src, group):
            nonlocal cursor
            tensor.copy_(recorded[cursor])
            cursor += 1

        monkeypatch.setattr(torch.distributed, "broadcast", replay_broadcast)
        replica = ExactDataParallelBatchProvider(
            source=None,
            dp_group=group,
            dp_idx=dp_idx,
            dp_size=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            d_model=1,
        )
        assert torch.equal(next(replica), expected)


def test_exact_provider_propagates_end_of_stream(monkeypatch) -> None:
    group = MagicMock()
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda _group, _rank: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda tensor, **kwargs: None)
    provider = ExactDataParallelBatchProvider(
        source=iter([]),
        dp_group=group,
        dp_idx=0,
        dp_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        d_model=1,
    )
    with pytest.raises(StopIteration):
        next(provider)
