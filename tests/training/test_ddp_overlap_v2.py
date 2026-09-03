from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.training.ddp_overlap_v2 import DDPOptimizerOverlapState
from sae_lens.training.multi_hook_sae import MultiHookSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer


class _FakeWork:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.wait_count = 0

    def wait(self) -> bool:
        self.wait_count += 1
        if self.wait_count == 1:
            self.tensor.mul_(4.0)
        return True


class _FakeBucket:
    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        index: int,
        is_last: bool,
    ) -> None:
        self.tensor = tensor
        self._index = index
        self._is_last = is_last

    def buffer(self) -> torch.Tensor:
        return self.tensor

    def index(self) -> int:
        return self._index

    def is_last(self) -> bool:
        return self._is_last


class _FakeDDP:
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        gradient_as_bucket_view: bool = True,
        **kwargs: object,
    ) -> None:
        self.module = module
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.init_kwargs = kwargs
        self._comm_state: object | None = None
        self._comm_hook: object | None = None
        self.pre_forward_count = 0
        self.post_forward_count = 0

    def register_comm_hook(self, state: object, hook: object) -> None:
        self._comm_state = state
        self._comm_hook = hook

    def emit(self, bucket: _FakeBucket) -> torch.futures.Future[torch.Tensor]:
        assert callable(self._comm_hook)
        return self._comm_hook(self._comm_state, bucket)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True) -> _FakeDDP:
        self.module.train(mode)
        return self

    def _pre_forward(
        self, *args: object, **kwargs: object
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        self.pre_forward_count += 1
        return args, kwargs

    def _post_forward(self, output: object) -> object:
        self.post_forward_count += 1
        return output

    def _inside_ddp_forward(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class _FakeWavefrontSAE(torch.nn.Module):
    def __init__(self, tp_group: object) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self._tp_group = tp_group

    def tp_wavefront_supported(self) -> bool:
        return True

    def tp_wavefront_encode_launch(
        self,
        step_input: TrainStepInput,
    ) -> dict[str, object]:
        return {"input": step_input, "hidden": step_input.sae_in @ self.weight}

    def tp_wavefront_decode_launch(self, state: dict[str, object]) -> None:
        state["decoded"] = state["hidden"]

    def tp_wavefront_finish(self, state: dict[str, object]) -> TrainStepOutput:
        step_input = state["input"]
        output = state["decoded"]
        assert isinstance(step_input, TrainStepInput)
        assert isinstance(output, torch.Tensor)
        loss = output.sum()
        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=output,
            feature_acts=output,
            hidden_pre=output,
            loss=loss,
            losses={"loss": loss},
        )


class _TwoHookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_0 = torch.nn.Linear(2, 2, bias=False)
        self.hook_1 = torch.nn.Linear(2, 2, bias=False)

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "hook_0": self.hook_0(inputs["hook_0"]),
            "hook_1": self.hook_1(inputs["hook_1"]),
        }


def _gloo_overlap_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _TwoHookModel()
        ddp_by_hook = {
            "hook_0": DDP(model.hook_0, gradient_as_bucket_view=True),
            "hook_1": DDP(model.hook_1, gradient_as_bucket_view=True),
        }
        state = DDPOptimizerOverlapState(
            ["hook_0", "hook_1"],
            ddp_by_hook,
            dist.group.WORLD,
        )
        scale = float(rank + 1)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            state.begin_step()
            outputs = {
                "hook_0": ddp_by_hook["hook_0"](torch.ones(2, 2) * scale),
                "hook_1": ddp_by_hook["hook_1"](
                    torch.ones(2, 2) * (scale * 2.0)
                ),
            }
            sum(output.sum() for output in outputs.values()).backward()
            state.end_backward()
            order = state.completion_order(["hook_1", "hook_0"])
            for hook_name in order:
                state.wait_for_hook(hook_name)
            state.finish_step()
        state.close()

        torch.save(
            {
                "hook_0": model.hook_0.weight.grad,
                "hook_1": model.hook_1.weight.grad,
                "order": order,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_overlap_state_launches_each_ready_bucket_without_hook_barrier() -> None:
    hook_0, hook_1 = "hook_0", "hook_1"
    wrappers = {
        hook_0: _FakeDDP(torch.nn.Linear(2, 2, bias=False)),
        hook_1: _FakeDDP(torch.nn.Linear(2, 2, bias=False)),
    }

    works: list[_FakeWork] = []

    def _all_reduce(tensor: torch.Tensor, **_: object) -> _FakeWork:
        work = _FakeWork(tensor)
        works.append(work)
        return work

    with (
        mock.patch(
            "sae_lens.training.ddp_overlap_v2.dist.get_world_size",
            return_value=2,
        ),
        mock.patch(
            "sae_lens.training.ddp_overlap_v2.dist.all_reduce",
            side_effect=_all_reduce,
        ),
        mock.patch("sae_lens.training.ddp_overlap_v2.DDP", _FakeDDP),
    ):
        state = DDPOptimizerOverlapState(
            [hook_0, hook_1],
            wrappers,  # type: ignore[arg-type]
            mock.sentinel.group,
        )
        state.begin_step()
        bucket_1a = _FakeBucket(torch.ones(2), index=0, is_last=False)
        bucket_0 = _FakeBucket(torch.full((2,), 2.0), index=0, is_last=True)
        bucket_1b = _FakeBucket(torch.full((2,), 3.0), index=1, is_last=True)

        future = wrappers[hook_1].emit(bucket_1a)
        assert future.done()
        assert len(works) == 1
        assert all(work.wait_count == 0 for work in works)

        wrappers[hook_0].emit(bucket_0)
        wrappers[hook_1].emit(bucket_1b)
        state.end_backward()
        assert len(works) == 3
        assert state.completion_order([hook_1, hook_0]) == [hook_0, hook_1]

        state.wait_for_hook(hook_1)
        assert works[0].wait_count == 1
        assert works[2].wait_count == 1
        torch.testing.assert_close(bucket_1a.tensor, torch.full((2,), 2.0))
        torch.testing.assert_close(bucket_1b.tensor, torch.full((2,), 6.0))
        assert works[1].wait_count == 0

        state.wait_for_hook(hook_0)
        assert works[1].wait_count == 1
        torch.testing.assert_close(bucket_0.tensor, torch.full((2,), 4.0))

        state.finish_step()
        state.begin_step()
        state.finish_step()


def test_overlap_state_rejects_parameters_shared_by_hooks() -> None:
    shared = torch.nn.Linear(1, 1, bias=False)
    wrappers = {
        "hook_0": _FakeDDP(shared),
        "hook_1": _FakeDDP(shared),
    }
    with (
        mock.patch(
            "sae_lens.training.ddp_overlap_v2.dist.get_world_size",
            return_value=2,
        ),
        mock.patch("sae_lens.training.ddp_overlap_v2.DDP", _FakeDDP),
        pytest.raises(ValueError, match="disjoint parameters"),
    ):
        DDPOptimizerOverlapState(
            ["hook_0", "hook_1"],
            wrappers,  # type: ignore[arg-type]
            mock.sentinel.group,
        )


def test_overlap_state_requires_bucket_view_gradients() -> None:
    wrappers = {
        "hook_0": _FakeDDP(
            torch.nn.Linear(1, 1, bias=False),
            gradient_as_bucket_view=False,
        )
    }
    with (
        mock.patch(
            "sae_lens.training.ddp_overlap_v2.dist.get_world_size",
            return_value=2,
        ),
        mock.patch("sae_lens.training.ddp_overlap_v2.DDP", _FakeDDP),
        pytest.raises(ValueError, match="gradient_as_bucket_view=True"),
    ):
        DDPOptimizerOverlapState(
            ["hook_0"],
            wrappers,  # type: ignore[arg-type]
            mock.sentinel.group,
        )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_ddp_bucket_views_are_the_low_memory_default(
    configured: bool | None,
    expected: bool,
) -> None:
    runner = LanguageModelSAETrainingRunner.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = SimpleNamespace(
        ddp_broadcast_buffers=None,
        ddp_find_unused_parameters=None,
        ddp_gradient_as_bucket_view=configured,
        ddp_static_graph=None,
        ddp_bucket_cap_mb=None,
        ddp_config_strict=False,
    )

    assert runner._resolve_ddp_kwargs()["gradient_as_bucket_view"] is expected


def test_optimizer_overlap_forces_ddp_bucket_views() -> None:
    runner = LanguageModelSAETrainingRunner.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = SimpleNamespace(
        ddp_broadcast_buffers=None,
        ddp_find_unused_parameters=None,
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=None,
        ddp_bucket_cap_mb=None,
        ddp_config_strict=False,
    )

    assert (
        runner._resolve_ddp_kwargs(force_gradient_as_bucket_view=True)[
            "gradient_as_bucket_view"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("mode", "tp_size", "hook_count", "expected"),
    [
        ("on", 1, 3, True),
        ("on", 2, 3, True),
        ("non_tp_only", 1, 3, True),
        ("non_tp_only", 2, 3, False),
        ("on", 1, 1, False),
    ],
)
def test_per_hook_ddp_overlap_wrapper_selection(
    mode: str,
    tp_size: int,
    hook_count: int,
    expected: bool,
) -> None:
    runner = LanguageModelSAETrainingRunner.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = SimpleNamespace(multi_sae_optimizer_overlap=mode)
    runner.sae_dp_size = 2
    runner.sae_tp_size = tp_size

    assert runner._per_hook_ddp_overlap_enabled(hook_count) is expected


def test_per_hook_ddp_wrappers_do_not_change_root_state_dict_layout() -> None:
    raw = {
        "hook_0": torch.nn.Linear(2, 2),
        "hook_1": torch.nn.Linear(2, 2),
    }
    owner = MultiHookSAE(["hook_0", "hook_1"], raw)
    wrappers = {name: _FakeDDP(module) for name, module in raw.items()}

    with mock.patch("sae_lens.training.multi_hook_sae.DDP", _FakeDDP):
        owner.set_ddp_forward_modules(wrappers)  # type: ignore[arg-type]

    assert owner.ddp_forward_modules() == wrappers
    assert all(".module." not in key for key in owner.state_dict())


def test_runner_attaches_one_bucket_view_ddp_per_hook() -> None:
    raw = {
        "hook_0": torch.nn.Linear(2, 2),
        "hook_1": torch.nn.Linear(2, 2),
    }
    owner = MultiHookSAE(["hook_0", "hook_1"], raw)
    runner = LanguageModelSAETrainingRunner.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = SimpleNamespace(
        device="cpu",
        ddp_broadcast_buffers=None,
        ddp_find_unused_parameters=None,
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=None,
        ddp_bucket_cap_mb=32,
        ddp_config_strict=False,
    )

    with (
        mock.patch("torch.nn.parallel.DistributedDataParallel", _FakeDDP),
        mock.patch("sae_lens.training.multi_hook_sae.DDP", _FakeDDP),
    ):
        runner._attach_per_hook_overlap_ddp(owner, mock.sentinel.group)

    wrappers = owner.ddp_forward_modules()
    assert set(wrappers) == set(raw)
    assert all(wrapper.gradient_as_bucket_view for wrapper in wrappers.values())
    assert all(wrapper.module is raw[name] for name, wrapper in wrappers.items())
    assert all(
        wrapper.init_kwargs["bucket_cap_mb"] == 32
        for wrapper in wrappers.values()
    )


def test_tp_wavefront_brackets_each_per_hook_ddp_forward() -> None:
    tp_group = object()
    raw = {
        "hook_0": _FakeWavefrontSAE(tp_group),
        "hook_1": _FakeWavefrontSAE(tp_group),
    }
    owner = MultiHookSAE(["hook_0", "hook_1"], raw)
    wrappers = {name: _FakeDDP(module) for name, module in raw.items()}
    inputs = {
        name: TrainStepInput(
            sae_in=torch.ones(2, 2),
            dead_neuron_mask=None,
            coefficients={},
            n_training_steps=0,
            is_logging_step=False,
        )
        for name in raw
    }

    with (
        mock.patch("sae_lens.training.multi_hook_sae.DDP", _FakeDDP),
        mock.patch(
            "sae_lens.training.multi_hook_sae.TopKTrainingSAE",
            _FakeWavefrontSAE,
        ),
    ):
        owner.set_ddp_forward_modules(wrappers)  # type: ignore[arg-type]
        outputs = owner(inputs)

    sum(output.loss for output in outputs.values()).backward()
    assert all(wrapper.pre_forward_count == 1 for wrapper in wrappers.values())
    assert all(wrapper.post_forward_count == 1 for wrapper in wrappers.values())
    assert all(module.weight.grad is not None for module in raw.values())


def test_memory_phase_sampling_only_activates_on_due_steps() -> None:
    trainer = MultiSAETrainer.__new__(MultiSAETrainer)
    trainer.cfg = SimpleNamespace(save_memory_every_n_steps=512)
    trainer._profile_memory = True
    trainer._memory_phase_records = [{"stale": True}]

    trainer.n_training_steps = 0
    trainer._start_memory_phase_step()
    assert trainer._memory_phase_step_active is False
    assert trainer._memory_phase_records == []

    trainer.n_training_steps = 511
    trainer._start_memory_phase_step()
    assert trainer._memory_phase_step_active is True


def test_unsynchronized_memory_phase_does_not_drain_cuda() -> None:
    trainer = MultiSAETrainer.__new__(MultiSAETrainer)
    trainer.cfg = SimpleNamespace(
        device="cuda",
        record_memory_empty_cache=True,
    )
    trainer._profile_memory = True
    trainer._memory_phase_step_active = True
    trainer._memory_phase_records = []
    trainer._memory_rank = 0
    trainer.n_training_steps = 0
    trainer.n_training_samples = 0

    with (
        mock.patch.object(trainer, "_component_memory_stats_mb", return_value={}),
        mock.patch("torch.cuda.synchronize") as synchronize,
        mock.patch("torch.cuda.empty_cache") as empty_cache,
        mock.patch("torch.cuda.mem_get_info", return_value=(3, 7)),
        mock.patch("torch.cuda.memory_allocated", return_value=0),
        mock.patch("torch.cuda.memory_reserved", return_value=0),
        mock.patch("torch.cuda.max_memory_allocated", return_value=0),
        mock.patch("torch.cuda.max_memory_reserved", return_value=0),
        mock.patch("torch.cuda.reset_peak_memory_stats"),
    ):
        trainer._record_memory_phase(
            "after_combined_backward",
            synchronize=False,
        )

    synchronize.assert_not_called()
    empty_cache.assert_not_called()
    assert trainer._memory_phase_records[0]["phase"] == "after_combined_backward"


def test_per_hook_ddp_overlap_reduces_combined_backward_gradients(
    tmp_path: Path,
) -> None:
    world_size = 2
    init_file = tmp_path / "gloo_init"
    mp.start_processes(
        _gloo_overlap_worker,
        args=(world_size, str(init_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )

    results = [
        torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]
    for result in results:
        torch.testing.assert_close(result["hook_0"], torch.full((2, 2), 3.0))
        torch.testing.assert_close(result["hook_1"], torch.full((2, 2), 6.0))
        assert sorted(result["order"]) == ["hook_0", "hook_1"]
