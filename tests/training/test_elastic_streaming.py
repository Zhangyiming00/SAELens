from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from run_sae_runner_gpu import _validate_streaming_world_size, parse_args
from sae_lens.elastic_streaming import (
    ElasticDistributedRuntime,
    ElasticStreamingController,
    ElasticStreamingLayout,
)
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.training.elastic_trainer_state import (
    broadcast_multi_sae_trainer_state,
    broadcast_optimizer_state,
)
from sae_lens.training.exact_dp_batch_provider import ExactDataParallelBatchProvider
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.shared_activation_buffer import SharedActivationBuffer
from scripts.elastic_streaming_control import (
    AutoSwitchPolicy,
    BufferSample,
    SharedBufferMonitor,
    ThroughputRates,
    _wait_for_auto_startup,
    calculate_rates,
)


def test_elastic_streaming_cli_implies_streaming_and_uses_default_control_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_sae_runner_gpu.py", "-es"])

    args = parse_args()

    assert args.elastic_streaming is True
    assert args.streaming_mode is True
    assert args.elastic_streaming_control_path == "/tmp/sae-elastic/control.json"


def test_plain_streaming_does_not_enable_elastic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_sae_runner_gpu.py", "--streaming-mode"],
    )

    args = parse_args()

    assert args.streaming_mode is True
    assert args.elastic_streaming is False
    assert args.elastic_streaming_control_path == "/tmp/sae-elastic/control.json"


def test_elastic_streaming_world_size_has_no_single_process_exception() -> None:
    args = SimpleNamespace(
        elastic_streaming=True,
        vllm_dp_size=2,
        sae_dp_size=2,
        sae_pp_size=1,
    )

    with pytest.raises(ValueError, match=r"elastic_streaming: WORLD_SIZE=1"):
        _validate_streaming_world_size(
            args,
            world_size=1,
            vllm_tp_size=2,
            sae_tp_size=2,
        )

    _validate_streaming_world_size(
        args,
        world_size=8,
        vllm_tp_size=2,
        sae_tp_size=2,
    )


def test_requested_eight_rank_layout_keeps_permanent_roles() -> None:
    layout = ElasticStreamingLayout.from_world_size(
        world_size=8,
        vllm_tp_size=2,
        sae_tp_size=2,
        sae_pp_size=1,
        permanent_vllm_dp=1,
        permanent_sae_dp=2,
    )

    assert layout.elastic_rank_count == 2
    assert layout.elastic_vllm_dp == 1
    assert layout.elastic_sae_dp == 1
    assert layout.permanent_vllm_ranks == (0, 1)
    assert layout.elastic_ranks == (2, 3)
    assert layout.permanent_sae_ranks == (4, 5, 6, 7)
    assert layout.active_sae_ranks(2) == (4, 5, 6, 7)
    assert layout.active_sae_ranks(3) == (4, 5, 6, 7, 2, 3)
    assert [layout.role(rank, 2) for rank in range(8)] == [
        "vllm", "vllm", "vllm", "vllm", "sae", "sae", "sae", "sae"
    ]
    assert [layout.role(rank, 3) for rank in range(8)] == [
        "vllm", "vllm", "sae", "sae", "sae", "sae", "sae", "sae"
    ]


def test_control_request_commit_and_reverse_switch(tmp_path: Path) -> None:
    layout = ElasticStreamingLayout(
        world_size=8,
        vllm_tp_size=2,
        sae_tp_size=2,
        sae_pp_size=1,
        permanent_vllm_dp=1,
        permanent_sae_dp=2,
        elastic_rank_count=2,
    )
    controller = ElasticStreamingController(tmp_path / "elastic.json", layout)
    initial = controller.initialize()
    assert (initial.phase, initial.epoch, initial.active_sae_dp) == ("active", 0, 2)

    requested = controller.request(3)
    assert controller.requested_target(local_epoch=0) == (1, 3)
    assert controller.requested_target(local_epoch=0, require_ready=True) is None
    with pytest.raises(RuntimeError, match="phase"):
        controller.request(2)
    controller.mark_ready(epoch=requested.epoch)
    assert controller.requested_target(local_epoch=0, require_ready=True) == (1, 3)
    committed = controller.commit(epoch=requested.epoch, active_sae_dp=3)
    assert (committed.phase, committed.active_sae_dp) == ("active", 3)

    reverse = controller.request(2)
    assert controller.requested_target(local_epoch=1, require_ready=True) == (2, 2)
    controller.commit(epoch=reverse.epoch, active_sae_dp=2)
    assert controller.read().active_sae_dp == 2


def test_control_publishes_buffer_monitoring_metadata(tmp_path: Path) -> None:
    layout = ElasticStreamingLayout(8, 2, 2, 1, 1, 2, 2)
    controller = ElasticStreamingController(tmp_path / "elastic.json", layout)
    controller.initialize()

    configured = controller.configure_buffer_monitoring(
        buffer_name="test_buffer",
        num_chunks=32,
        chunk_size_tokens=4096,
    )

    assert configured.version == 3
    assert configured.buffer_name == "test_buffer"
    assert configured.buffer_num_chunks == 32
    assert configured.buffer_chunk_size_tokens == 4096


def test_auto_startup_waits_for_buffer_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.json"
    layout = ElasticStreamingLayout(4, 1, 1, 1, 1, 2, 1)
    controller = ElasticStreamingController(control_path, layout)
    controller.initialize()
    buffer = SharedActivationBuffer(
        name="delayed_auto_buffer",
        num_chunks=2,
        chunk_size_tokens=4,
        d_model=2,
        num_producers=2,
        target_chunks=4,
        create=True,
        base_dir=str(tmp_path),
        dtype=torch.float32,
    )
    sleep_calls = 0

    def publish_metadata(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        controller.configure_buffer_monitoring(
            buffer_name="delayed_auto_buffer",
            num_chunks=2,
            chunk_size_tokens=4,
        )

    monkeypatch.setattr(
        "scripts.elastic_streaming_control.time.sleep",
        publish_metadata,
    )
    args = SimpleNamespace(
        control_path=control_path,
        startup_timeout=10.0,
        poll_interval=0.01,
        shm_base_dir=tmp_path,
    )
    try:
        startup = _wait_for_auto_startup(args)
        assert startup is not None
        _, monitor = startup
        assert monitor.chunk_size_tokens == 4
        assert sleep_calls == 1
    finally:
        buffer.destroy()


def test_buffer_flow_rates_match_occupancy_delta() -> None:
    first = BufferSample(
        timestamp=10.0,
        next_claim_seq=100,
        occupied_chunks=3,
        counts={"free": 7, "writing": 0, "ready": 2, "consuming": 1},
    )
    last = BufferSample(
        timestamp=12.0,
        next_claim_seq=108,
        occupied_chunks=5,
        counts={"free": 5, "writing": 1, "ready": 3, "consuming": 1},
    )

    rates = calculate_rates(first, last, chunk_size_tokens=4)

    assert rates is not None
    assert rates.vllm_tokens_per_s == 16.0
    assert rates.sae_tokens_per_s == 12.0
    assert rates.net_tokens_per_s == 4.0
    assert rates.fill_ratio == 0.5


def test_buffer_monitor_observes_existing_state_machine(tmp_path: Path) -> None:
    layout = ElasticStreamingLayout(8, 2, 2, 1, 1, 2, 2)
    controller = ElasticStreamingController(tmp_path / "elastic.json", layout)
    controller.initialize()
    buffer = SharedActivationBuffer(
        name="observed_buffer",
        num_chunks=4,
        chunk_size_tokens=4,
        d_model=2,
        num_producers=1,
        target_chunks=1,
        create=True,
        base_dir=str(tmp_path),
        dtype=torch.float32,
    )
    controller.configure_buffer_monitoring(
        buffer_name="observed_buffer",
        num_chunks=4,
        chunk_size_tokens=4,
    )
    monitor = SharedBufferMonitor(controller.read(), tmp_path)

    try:
        before = monitor.sample(timestamp=0.0)
        index, _ = buffer.allocate_write_chunk()
        buffer.write_chunk(
            index,
            torch.ones(4, 2),
            valid_tokens=4,
        )
        buffer.mark_ready(index)
        produced = monitor.sample(timestamp=1.0)
        claimed, _ = buffer.acquire_up_to(1)
        buffer.release_chunk(claimed[0])
        consumed = monitor.sample(timestamp=2.0)

        assert produced.production_complete
        assert produced.counts["ready"] == 1
        assert consumed.counts["free"] == 4
        assert calculate_rates(
            before, produced, chunk_size_tokens=4
        ) == ThroughputRates(1.0, 0.25, 4.0, 0.0, 4.0)
        assert calculate_rates(
            produced, consumed, chunk_size_tokens=4
        ) == ThroughputRates(1.0, 0.0, 0.0, 4.0, -4.0)
    finally:
        buffer.destroy()


def test_auto_switch_policy_requires_stable_rate_and_watermark() -> None:
    policy = AutoSwitchPolicy(
        low_watermark=0.25,
        high_watermark=0.75,
        stable_samples=3,
        min_rate_ratio=1.05,
        min_rate_gap=100.0,
    )
    growing = ThroughputRates(10.0, 0.8, 1_200.0, 900.0, 300.0)

    assert policy.observe(
        active_sae_dp=2, min_sae_dp=2, max_sae_dp=3, rates=growing
    ) is None
    assert policy.observe(
        active_sae_dp=2, min_sae_dp=2, max_sae_dp=3, rates=growing
    ) is None
    assert policy.observe(
        active_sae_dp=2, min_sae_dp=2, max_sae_dp=3, rates=growing
    ) == 3

    draining = ThroughputRates(10.0, 0.2, 700.0, 1_000.0, -300.0)
    assert policy.observe(
        active_sae_dp=3, min_sae_dp=2, max_sae_dp=3, rates=draining
    ) is None
    assert policy.observe(
        active_sae_dp=3, min_sae_dp=2, max_sae_dp=3, rates=draining
    ) is None
    assert policy.observe(
        active_sae_dp=3, min_sae_dp=2, max_sae_dp=3, rates=draining
    ) == 2


def test_auto_switch_policy_resets_when_watermark_is_not_met() -> None:
    policy = AutoSwitchPolicy(
        low_watermark=0.25,
        high_watermark=0.75,
        stable_samples=2,
        min_rate_ratio=1.05,
        min_rate_gap=0.0,
    )
    growing = ThroughputRates(10.0, 0.8, 1_200.0, 900.0, 300.0)
    below_watermark = ThroughputRates(10.0, 0.5, 1_000.0, 1_000.0, 0.0)

    assert policy.observe(
        active_sae_dp=2, min_sae_dp=2, max_sae_dp=3, rates=growing
    ) is None
    assert policy.observe(
        active_sae_dp=2,
        min_sae_dp=2,
        max_sae_dp=3,
        rates=below_watermark,
    ) is None
    assert policy.growing_samples == 0


def test_auto_switch_policy_allows_rate_tie_at_watermarks() -> None:
    policy = AutoSwitchPolicy(
        low_watermark=0.25,
        high_watermark=0.75,
        stable_samples=2,
        min_rate_ratio=1.05,
        min_rate_gap=100.0,
    )
    full_and_tied = ThroughputRates(10.0, 1.0, 1_000.0, 1_000.0, 0.0)

    assert policy.observe(
        active_sae_dp=2, min_sae_dp=2, max_sae_dp=3, rates=full_and_tied
    ) is None
    assert policy.observe(
        active_sae_dp=2, min_sae_dp=2, max_sae_dp=3, rates=full_and_tied
    ) == 3

    low_and_tied = ThroughputRates(10.0, 0.05, 800.0, 800.0, 0.0)
    assert policy.observe(
        active_sae_dp=3, min_sae_dp=2, max_sae_dp=3, rates=low_and_tied
    ) is None
    assert policy.observe(
        active_sae_dp=3, min_sae_dp=2, max_sae_dp=3, rates=low_and_tied
    ) == 2


def test_auto_switch_policy_does_not_restart_vllm_after_production() -> None:
    policy = AutoSwitchPolicy(
        low_watermark=0.25,
        high_watermark=0.75,
        stable_samples=1,
        min_rate_ratio=1.05,
        min_rate_gap=0.0,
    )
    draining = ThroughputRates(10.0, 0.1, 0.0, 1_000.0, -1_000.0)

    assert policy.observe(
        active_sae_dp=3,
        min_sae_dp=2,
        max_sae_dp=3,
        rates=draining,
        production_complete=True,
    ) is None


def test_finish_is_idempotent_and_rejects_later_switch(tmp_path: Path) -> None:
    layout = ElasticStreamingLayout(8, 2, 2, 1, 1, 2, 2)
    controller = ElasticStreamingController(tmp_path / "elastic.json", layout)
    controller.initialize()

    assert controller.finish(epoch=0).phase == "finished"
    assert controller.finish(epoch=0).phase == "finished"
    with pytest.raises(RuntimeError, match="phase"):
        controller.request(3)


def test_request_wins_atomic_race_with_finish(tmp_path: Path) -> None:
    layout = ElasticStreamingLayout(8, 2, 2, 1, 1, 2, 2)
    controller = ElasticStreamingController(tmp_path / "elastic.json", layout)
    controller.initialize()

    requested = controller.request(3)
    observed = controller.finish(epoch=0)
    assert observed == requested
    assert observed.phase == "switching"


def test_control_rejects_layout_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "elastic.json"
    first = ElasticStreamingLayout(8, 2, 2, 1, 1, 2, 2)
    ElasticStreamingController(path, first).initialize()
    second = ElasticStreamingLayout(10, 2, 2, 1, 1, 3, 2)
    with pytest.raises(ValueError, match="does not match"):
        ElasticStreamingController(path, second).read()


def test_layout_supports_unequal_tp_and_sae_pp() -> None:
    layout = ElasticStreamingLayout.from_world_size(
        world_size=8,
        vllm_tp_size=2,
        sae_tp_size=1,
        sae_pp_size=2,
        permanent_vllm_dp=1,
        permanent_sae_dp=1,
    )

    assert layout.elastic_rank_count == 4
    assert layout.elastic_vllm_dp == 2
    assert layout.elastic_sae_dp == 2
    assert layout.max_vllm_dp == 3
    assert layout.max_sae_dp == 3
    assert layout.permanent_vllm_ranks == (0, 1)
    assert layout.elastic_ranks == (2, 3, 4, 5)
    assert layout.permanent_sae_ranks == (6, 7)
    assert layout.sae_stage_ranks(0, 0) == (6,)
    assert layout.sae_stage_ranks(0, 1) == (7,)
    assert layout.sae_stage_ranks(1, 0) == (2,)
    assert layout.sae_stage_ranks(1, 1) == (3,)
    assert layout.sae_stage_ranks(2, 0) == (4,)
    assert layout.sae_stage_ranks(2, 1) == (5,)
    assert layout.active_sae_ranks(1) == (6, 7)
    assert layout.active_sae_ranks(3) == (6, 7, 2, 3, 4, 5)


def test_layout_requires_complete_groups_for_both_roles() -> None:
    with pytest.raises(ValueError, match="divisible by both"):
        ElasticStreamingLayout.from_world_size(
            world_size=8,
            vllm_tp_size=2,
            sae_tp_size=2,
            sae_pp_size=2,
            permanent_vllm_dp=1,
            permanent_sae_dp=1,
        )


def test_runtime_maps_unequal_tp_pp_and_dp_groups(monkeypatch) -> None:
    layout = ElasticStreamingLayout.from_world_size(
        world_size=8,
        vllm_tp_size=2,
        sae_tp_size=1,
        sae_pp_size=2,
        permanent_vllm_dp=1,
        permanent_sae_dp=1,
    )
    current_rank = [0]

    def new_group(ranks, backend):
        return backend, tuple(sorted(ranks))

    def get_rank(group=None):
        if group is None:
            return current_rank[0]
        return group[1].index(current_rank[0])

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda _group=None: 8)
    monkeypatch.setattr(dist, "get_rank", get_rank)
    monkeypatch.setattr(dist, "new_group", new_group)
    runtime = ElasticDistributedRuntime(layout)

    current_rank[0] = 2
    elastic_pp0 = runtime.context(layout.max_sae_dp)
    assert (elastic_pp0.get_sae_dp_idx(), elastic_pp0.get_sae_pp_rank()) == (0, 0)
    assert elastic_pp0.get_sae_tp_group()[1] == (2,)
    assert elastic_pp0.get_sae_dp_group()[1] == (2, 4, 6)
    assert elastic_pp0.get_sae_pp_root_group()[1] == (2, 3)
    assert elastic_pp0.get_sae_pp_root_global_rank() == 2

    current_rank[0] = 7
    permanent_pp1 = runtime.context(layout.max_sae_dp)
    assert (permanent_pp1.get_sae_dp_idx(), permanent_pp1.get_sae_pp_rank()) == (
        2,
        1,
    )
    assert permanent_pp1.get_sae_tp_group()[1] == (7,)
    assert permanent_pp1.get_sae_dp_group()[1] == (3, 5, 7)
    assert permanent_pp1.get_sae_pp_root_group()[1] == (6, 7)
    assert permanent_pp1.get_sae_pp_root_global_rank() == 6

    current_rank[0] = 0
    assert not runtime.context(layout.max_sae_dp).is_consumer()
    assert runtime.is_permanent_vllm()


def test_joining_elastic_rank_builds_hooks_for_new_pp_context(monkeypatch) -> None:
    runner = object.__new__(LanguageModelSAETrainingRunner)
    runner.sae_pp_size = 2
    runner.sae_tp_size = 1
    runner.hook_names = ["hook.early", "hook.late"]
    runner.device = torch.device("cpu")
    runner._create_training_sae = MagicMock(side_effect=lambda **kwargs: kwargs)
    cfg = SimpleNamespace(
        seed=7,
        multi_sae_seed_mode="same",
        resume_from_checkpoint=None,
    )
    context = SimpleNamespace(
        get_sae_pp_rank=lambda: 1,
        get_sae_tp_group=lambda: None,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    runner._streaming_create_consumer_multi_base(cfg, context)

    assert runner._pp_hook_names == ["hook.late"]
    assert list(runner.base_sae_by_hook) == ["hook.late"]
    assert runner.base_sae_by_hook["hook.late"]["hook_metadata_overrides"] == {
        "hook_name": "hook.late"
    }


def test_buffer_producer_sessions_survive_elastic_leave_and_return(
    tmp_path: Path,
) -> None:
    buffer = SharedActivationBuffer(
        name="elastic_sessions",
        num_chunks=2,
        chunk_size_tokens=4,
        d_model=2,
        num_producers=2,
        target_chunks=4,
        create=True,
        base_dir=str(tmp_path),
        dtype=torch.float32,
    )
    try:
        buffer.signal_done()
        buffer.signal_done()
        assert buffer.snapshot()["finished_producer_sessions"] == 2
        buffer.register_producer()
        snapshot = buffer.snapshot()
        assert snapshot["producer_sessions"] == 3
        assert snapshot["finished_producer_sessions"] == 2

        index, _ = buffer.allocate_write_chunk()
        acts = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        buffer.write_chunk(index, acts, valid_tokens=4, producer_id=1)
        buffer.mark_ready(index)
        claimed, _ = buffer.acquire_up_to(1, random=False)
        observed, valid = buffer.read_chunk(claimed[0])
        assert valid == 4
        assert torch.equal(observed, acts)
        buffer.release_chunk(claimed[0])
    finally:
        buffer.destroy()


def test_elastic_vllm_teardown_preserves_live_iterator_and_activation_tail() -> None:
    runner = object.__new__(LanguageModelSAETrainingRunner)
    runner.device = torch.device("cpu")
    iterator = iter([torch.tensor([11]), torch.tensor([12])])
    store = SimpleNamespace(
        model=object(),
        iterable_sequences=iterator,
        _stream_residual_gpu=torch.tensor([[1.0], [2.0]]),
        _stream_collected_by_hook_gpu={
            "hook": [torch.tensor([[3.0]]), torch.tensor([[4.0]])]
        },
    )
    model = MagicMock()
    runner.activations_store = store
    runner.model = model

    runner._elastic_destroy_vllm()

    assert runner._elastic_activations_store is store
    assert store.model is None
    assert next(store.iterable_sequences).item() == 11
    assert torch.equal(store._stream_residual_gpu, torch.tensor([[1.0], [2.0]]))
    assert [value.item() for value in store._stream_collected_by_hook_gpu["hook"]] == [
        3.0,
        4.0,
    ]
    model.close.assert_called_once_with()


def test_exact_provider_cutover_poll_does_not_consume_next_batch(monkeypatch) -> None:
    group = MagicMock()
    source_batches = iter(
        [torch.arange(12, dtype=torch.float32).reshape(12, 1)]
    )
    monkeypatch.setattr(dist, "get_global_rank", lambda _group, _rank: 0)
    monkeypatch.setattr(dist, "broadcast", lambda _tensor, **_kwargs: None)
    provider = ExactDataParallelBatchProvider(
        source=source_batches,
        dp_group=group,
        dp_idx=0,
        dp_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        d_model=1,
        global_training_tokens=12,
        current_epoch=4,
        reconfigure_poll=lambda: (5, 3),
    )

    assert provider.poll_reconfigure()
    assert provider.pending_reconfigure == (5, 3)
    assert provider.global_tokens_consumed == 0
    assert torch.equal(next(provider)[:, 0], torch.arange(12, dtype=torch.float32))


def test_multi_trainer_cutover_stops_before_fetch_and_skips_final_checkpoint() -> None:
    class FailOnFetch:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("cutover must stop before fetching another batch")

    trainer = object.__new__(MultiSAETrainer)
    trainer.cfg = SimpleNamespace(
        total_training_samples=10,
        save_final_checkpoint=True,
    )
    trainer.n_training_samples = 0
    trainer.data_provider = FailOnFetch()
    trainer.base_sae_by_hook = {}
    trainer.step_window_profiler = None
    trainer._overlap_tp_post = None
    trainer._overlap_ddp_state = None
    trainer._start_device_sampler = MagicMock()
    trainer._stop_device_sampler = MagicMock()
    trainer._sync_deferred_stats_if_needed = MagicMock()
    trainer.save_checkpoint = MagicMock()

    assert trainer.fit(stop_after_step_check=lambda: True) == {}
    assert trainer.last_fit_stopped_for_reconfigure
    trainer.save_checkpoint.assert_not_called()


def test_exact_provider_rotates_remainder_ownership(monkeypatch) -> None:
    group = MagicMock()
    messages: list[torch.Tensor] = []
    monkeypatch.setattr(dist, "get_global_rank", lambda _group, _rank: 0)

    def broadcast(tensor, **_kwargs):
        messages.append(tensor.clone())

    monkeypatch.setattr(dist, "broadcast", broadcast)
    batches = [
        torch.arange(step * 4096, (step + 1) * 4096, dtype=torch.float32).reshape(-1, 1)
        for step in range(2)
    ]
    provider = ExactDataParallelBatchProvider(
        source=iter(batches),
        dp_group=group,
        dp_idx=0,
        dp_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        d_model=1,
        global_training_tokens=8192,
    )

    assert next(provider).shape[0] == 1365
    assert next(provider).shape[0] == 1366
    assert provider.global_tokens_consumed == 8192
    assert provider.step_index == 2
    assert messages


def test_exact_provider_supports_permanent_source_at_nonzero_group_rank(
    monkeypatch,
) -> None:
    group = MagicMock()
    group_global_ranks = (1, 2, 3)
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda _group, group_rank: group_global_ranks[group_rank],
    )
    monkeypatch.setattr(dist, "broadcast", lambda _tensor, **_kwargs: None)
    provider = ExactDataParallelBatchProvider(
        source=iter([torch.arange(12, dtype=torch.float32).reshape(12, 1)]),
        dp_group=group,
        dp_idx=2,
        dp_size=3,
        source_dp_idx=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        d_model=1,
        global_training_tokens=12,
        current_epoch=4,
        reconfigure_poll=lambda: (5, 1),
    )

    assert torch.equal(next(provider)[:, 0], torch.arange(8, 12, dtype=torch.float32))
    assert provider.poll_reconfigure()
    assert provider.pending_reconfigure == (5, 1)


def test_optimizer_state_is_transferred_without_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "broadcast", lambda *_args, **_kwargs: None)
    source_model = torch.nn.Linear(3, 2)
    source = torch.optim.Adam(source_model.parameters(), lr=0.0123)
    source_model(torch.ones(4, 3)).sum().backward()
    source.step()

    target_model = torch.nn.Linear(3, 2)
    target = torch.optim.Adam(target_model.parameters(), lr=1.0)
    broadcast_optimizer_state(
        source=source,
        target=target,
        group=MagicMock(),
        source_global_rank=0,
        device=torch.device("cpu"),
    )

    assert target.param_groups[0]["lr"] == pytest.approx(0.0123)
    for source_parameter, target_parameter in zip(
        source_model.parameters(), target_model.parameters()
    ):
        source_state = source.state[source_parameter]
        target_state = target.state[target_parameter]
        assert source_state.keys() == target_state.keys()
        for name in source_state:
            if isinstance(source_state[name], torch.Tensor):
                assert torch.equal(source_state[name], target_state[name])
            else:
                assert source_state[name] == target_state[name]


@pytest.mark.parametrize(
    ("source_overlap", "target_overlap"),
    [(False, True), (True, False)],
)
def test_trainer_state_transfer_converts_optimizer_topology(
    monkeypatch: pytest.MonkeyPatch,
    source_overlap: bool,
    target_overlap: bool,
) -> None:
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "broadcast", lambda *_args, **_kwargs: None)

    class Stateful:
        def __init__(self) -> None:
            self.loaded: dict[str, object] = {}

        def state_dict(self) -> dict[str, object]:
            return dict(self.loaded)

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.loaded = dict(state)

    def make_trainer(*, overlap: bool, lr: float) -> SimpleNamespace:
        modules = {
            "hook_0": torch.nn.Linear(3, 2),
            "hook_1": torch.nn.Linear(3, 2),
        }
        all_parameters = [
            parameter
            for module in modules.values()
            for parameter in module.parameters()
        ]
        optimizer = torch.optim.Adam(all_parameters, lr=lr)
        overlap_optimizers = (
            {
                name: torch.optim.Adam(module.parameters(), lr=lr)
                for name, module in modules.items()
            }
            if overlap
            else {}
        )
        return SimpleNamespace(
            hook_names=list(modules),
            base_sae_by_hook=modules,
            optimizer=optimizer,
            _overlap_optimizer_by_hook=overlap_optimizers,
            n_training_samples=128,
            n_training_steps=7,
            lr_scheduler=Stateful(),
            grad_scaler=Stateful(),
            activation_scaler_by_hook={
                name: SimpleNamespace(scaling_factor=1.5) for name in modules
            },
            n_frac_active_samples_by_hook={name: 3 for name in modules},
            _pending_sample_count_by_hook={name: 4 for name in modules},
            _pending_step_count_by_hook={name: 5 for name in modules},
            checkpoint_thresholds=[256],
            _t_ready=time.time() - 10.0,
            act_freq_scores_by_hook={
                name: torch.ones(2) for name in modules
            },
            n_forward_passes_since_fired_by_hook={
                name: torch.ones(2, dtype=torch.long) for name in modules
            },
            _pending_did_fire_max_by_hook={
                name: torch.ones(2, dtype=torch.bool) for name in modules
            },
        )

    source = make_trainer(overlap=source_overlap, lr=0.0123)
    target = make_trainer(overlap=target_overlap, lr=1.0)
    for hook_name, module in source.base_sae_by_hook.items():
        optimizer = source._overlap_optimizer_by_hook.get(
            hook_name,
            source.optimizer,
        )
        module(torch.ones(4, 3)).sum().backward()
        if source_overlap:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if not source_overlap:
        source.optimizer.step()
        source.optimizer.zero_grad(set_to_none=True)

    expected_state_by_hook = {}
    for hook_name, module in source.base_sae_by_hook.items():
        optimizer = source._overlap_optimizer_by_hook.get(hook_name, source.optimizer)
        expected_state_by_hook[hook_name] = [
            {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in optimizer.state[parameter].items()
            }
            for parameter in module.parameters()
        ]

    broadcast_multi_sae_trainer_state(
        source=source,
        target=target,
        group=MagicMock(),
        source_global_rank=0,
        device=torch.device("cpu"),
    )

    assert target.optimizer.param_groups[0]["lr"] == pytest.approx(0.0123)
    for hook_name in source.hook_names:
        source_optimizer = source._overlap_optimizer_by_hook.get(
            hook_name,
            source.optimizer,
        )
        target_optimizer = target._overlap_optimizer_by_hook.get(
            hook_name,
            target.optimizer,
        )
        if target_overlap:
            assert target_optimizer.param_groups[0]["lr"] == pytest.approx(0.0123)
        for expected_state, target_parameter in zip(
            expected_state_by_hook[hook_name],
            target.base_sae_by_hook[hook_name].parameters(),
        ):
            target_state = target_optimizer.state[target_parameter]
            assert expected_state.keys() == target_state.keys()
            for name in expected_state:
                if isinstance(expected_state[name], torch.Tensor):
                    assert torch.equal(expected_state[name], target_state[name])
                else:
                    assert expected_state[name] == target_state[name]
    # Elastic transfer consumes the obsolete trainer incrementally so its Adam
    # moments cannot overlap the replacement optimizer's full growth.
    source_optimizers = {source.optimizer, *source._overlap_optimizer_by_hook.values()}
    assert all(not optimizer.state for optimizer in source_optimizers)
