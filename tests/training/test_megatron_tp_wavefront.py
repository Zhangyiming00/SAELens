"""Real native TP wavefront: outputs, gradients, updates and disk continuation."""

import json
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.sae_runtime import SAERuntime
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.gradient_window import train_runtime_window
from sae_lens.training.multi_hook_sae import MultiHookSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from tests.training.test_hook_optimizer_overlap import (
    NUMERICAL_ERRORS,
    build,
    equal,
    full_adam,
    snapshot,
)


@pytest.mark.parametrize(("mode", "wave", "dp", "producer", "fences"), [
    ("off", True, 2, True, False),
    ("always", False, 1, False, True),
    ("auto", False, 2, True, False),
    ("auto", True, 1, False, False),
    ("auto", True, 2, False, True),
    ("auto", True, 1, True, True),
])
def test_runtime_wavefront_uses_existing_phase_fence(mode, wave, dp, producer, fences):
    trainer = MultiSAETrainer.__new__(MultiSAETrainer)
    trainer._tp_phase_fence_mode = mode
    trainer._tp_phase_event = MagicMock()
    trainer._tp_world_size = lambda: 2
    trainer._dp_world_size = lambda: dp
    trainer._tp_group = lambda: object()
    trainer.dp_group = object()
    trainer._is_ddp = True
    trainer.hook_names = ["a", "b"]
    trainer.multi_sae_distributed_architecture = "legacy_per_hook_wrapper"
    trainer._runtime_tp_wavefront = wave
    trainer.cfg = SimpleNamespace(device="cuda:0", multi_sae_tp_phase_fence_runtime_hazard=producer)
    with patch("torch.cuda.current_stream", return_value=object()):
        trainer._tp_phase_fence_if_needed()
    assert trainer._tp_phase_event.record.call_count == int(fences)
    assert trainer._tp_phase_event.synchronize.call_count == int(fences)


def _models(runtime, records):
    """Compare the phased path with native serial forward, including input grads."""
    device = f"cuda:{dist.get_rank()}"
    for norm, sparse, rescale, bias in [
        ("none", False, True, True), ("none", True, True, True),
        ("none", False, False, False), ("layer_norm", False, True, True),
        ("constant_norm_rescale", False, True, False),
    ]:
        for differentiable in (False, True):
            serial, phased = {}, {}
            for i in range(3):
                cfg = TopKTrainingSAEConfig(
                    d_in=16, d_sae=48, k=4, device=device,
                    normalize_activations=norm, use_sparse_activations=sparse,
                    rescale_acts_by_decoder_norm=rescale, apply_b_dec_to_input=bias,
                )
                torch.manual_seed(781 + i)
                serial[f"h{i}"] = MegatronTopKSAE(cfg, runtime=runtime)
                phased[f"h{i}"] = MegatronTopKSAE(cfg, runtime=runtime)
                phased[f"h{i}"].load_state_dict(serial[f"h{i}"].state_dict())
            owners = [MultiHookSAE(list(serial), serial, enable_tp_wavefront=False),
                      MultiHookSAE(list(phased), phased, enable_tp_wavefront=True)]
            assert owners[1]._can_tp_wavefront()
            batches = []
            outputs = []
            for owner in owners:
                torch.manual_seed(127)
                inputs = {}
                for i, h in enumerate(serial):
                    x = torch.randn(5 + i, 16, device=device, requires_grad=differentiable)
                    mask = torch.arange(48, device=device) % 2 == 0 if norm == "none" else None
                    inputs[h] = TrainStepInput(x, {}, mask, 0, False)
                batches.append(inputs)
                outputs.append(owner(inputs))
                # Combined backward also remains supported by MultiHookSAE.
                sum(o.loss for o in outputs[-1].values()).backward()
            for h in serial:
                for key in ("loss", "sae_out", "hidden_pre", "feature_acts"):
                    a, b = getattr(outputs[0][h], key), getattr(outputs[1][h], key)
                    if a.is_sparse:
                        a, b = a.to_dense(), b.to_dense()
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                for a, b in zip(serial[h].parameters(), phased[h].parameters(), strict=True):
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
                if differentiable:
                    torch.testing.assert_close(batches[0][h].sae_in.grad, batches[1][h].sae_in.grad,
                                               atol=2e-6, rtol=2e-5)
            records.append(dict(kind="model", tp=runtime.local.tp_group.size(), norm=norm,
                                sparse=sparse, rescale=rescale, bias=bias, input_grad=differentiable))


def _matrix(runtime, path, records):
    ctx = runtime.require_local()
    expected_wave = ctx.tp_group.size() > 1 and len(ctx.domain.hooks) > 1
    for sharded, overlap in ((False, "off"), (False, "on"), (True, "off"), (True, "on")):
        for ga, amp in ((1, False), (1, True), (3, False), (3, True)):
            NUMERICAL_ERRORS.clear()
            serial = build(runtime, path, overlap, ga, amp, sharded=sharded)
            wave = build(runtime, path, overlap, ga, amp, sharded=sharded,
                         architecture="unified_multi_hook")
            assert not serial._runtime_tp_wavefront
            assert wave._runtime_tp_wavefront == expected_wave
            assert wave.cfg.multi_sae_distributed_architecture == "unified_multi_hook"
            launches = []
            handles = []
            for h, unit in wave.units.items():
                original = unit.model.tp_wavefront_encode_launch

                def record(step_input, original=original, hook=h):
                    launches.append(hook)
                    return original(step_input)

                unit.model.tp_wavefront_encode_launch = record
            for step in range(6):
                batches = []
                for micro in range(ga if step != 4 else 1):
                    batch = {}
                    for i, h in enumerate(ctx.domain.hooks):
                        n = (ctx.dp_rank + micro + step + i) % 5 + 1
                        if step == 5 or (step == 1 and ctx.dp_rank == 0) or (step == 2 and i == 0):
                            n = 0
                        torch.manual_seed(1700 + step * 37 + micro * 7 + ctx.dp_rank * 3 + i)
                        batch[h] = torch.randn(n, 16, device=f"cuda:{dist.get_rank()}") * (20 if step % 2 else 0.05)
                    batches.append(batch)
                for trainer in (serial, wave):
                    if amp and step == 3 and ctx.tp_rank == ctx.dp_rank == 0:
                        p = next(next(iter(trainer.units.values())).model.parameters())
                        handles.append(p.register_hook(lambda g: torch.full_like(g, float("inf"))))
                    # Unknown-length GA tail must use explicit window-end sync.
                    train_runtime_window(trainer, iter(batches) if step == 4 else batches)
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    trainer.lr_scheduler.step(trainer._last_updated_hooks)
                    trainer.n_training_steps += 1
                equal(snapshot(serial), snapshot(wave))
                if step == 2 and amp:
                    name = f"tp{ctx.tp_group.size()}dp{ctx.dp_group.size()}p{len(runtime.domains)}_s{sharded}_{overlap}_ga{ga}"
                    wave.save_checkpoint(name)
                    restored = build(runtime, path, overlap, ga, amp, sharded=sharded,
                                     architecture="unified_multi_hook")
                    restored.load_trainer_state(path / name)
                    for h in wave.units:
                        equal(wave.units[h].model.state_dict(), restored.units[h].model.state_dict())
                        equal(full_adam(wave.units[h]), full_adam(restored.units[h]))
                        assert wave.units[h].update_count == restored.units[h].update_count
                    equal(wave.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict())
                    equal(wave.grad_scaler.state_dict(), restored.grad_scaler.state_dict())
                    wave = restored
            assert bool(launches) == expected_wave
            records.append(dict(kind="runtime", tp=ctx.tp_group.size(), dp=ctx.dp_group.size(),
                                hooks=len(ctx.domain.hooks), sharded=sharded, optimizer_overlap=overlap,
                                ga=ga, amp=amp, wavefront=expected_wave, errors=dict(NUMERICAL_ERRORS)))
            del serial, wave


def _worker(rank, directory, implicit_order):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=4,
                            init_method=f"file://{directory}/world", timeout=timedelta(seconds=90))
    records = []
    try:
        layouts = [(2, 1, 1, 3), (4, 1, 1, 3), (2, 2, 1, 3), (2, 1, 2, 3), (1, 2, 1, 3)]
        if not implicit_order:
            layouts = [(2, 2, 1, 3)]
        for tp, dp, spp, nh in layouts:
            runtime = SAERuntime.from_layout(tp_size=tp, dp_size=dp, placement_size=spp,
                                             hooks=tuple(f"h{i}" for i in range(nh)))
            failure = None
            try:
                if runtime.local is not None:
                    if dp == spp == 1:
                        _models(runtime, records)
                    _matrix(runtime, Path(directory), records)
            except BaseException:
                failure = traceback.format_exc()
                Path(directory, f"failure_rank{rank}.txt").write_text(failure)
            finally:
                failures = [None] * 4
                dist.all_gather_object(failures, failure, group=runtime.control_group)
                runtime.close()
                if any(failures):
                    raise AssertionError(str(failures))
    finally:
        Path(directory, f"rank{rank}.json").write_text(json.dumps(records, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
@pytest.mark.parametrize("implicit_order", [True, False])
def test_native_tp_wavefront_matrix(tmp_path, monkeypatch, implicit_order):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1" if implicit_order else "0")
    root = Path(os.environ.get("SAE_TP_WAVEFRONT_REPORT_DIR", tmp_path)).resolve()
    directory = root / f"matrix_order{int(implicit_order)}"
    directory.mkdir(parents=True, exist_ok=True)
    ctx = mp.spawn(_worker, args=(str(directory), implicit_order), nprocs=4, join=False)
    deadline = time.monotonic() + 300
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError("TP wavefront validation timed out")
        reports = [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(4)]
        assert all(reports)
        assert all(any(r["kind"] == "runtime" and r["wavefront"] for r in report) for report in reports)
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
        for process in ctx.processes:
            process.join(timeout=5)


def _failure_worker(rank, directory):
    from sae_lens.megatron_tp import _wavefront_stream
    from sae_lens.static_failure import StaticFailureMonitor
    from sae_lens.training.gradient_window import require_window_boundary

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4,
                            init_method=f"file://{directory}/world", timeout=timedelta(seconds=45))
    before = set(dist.distributed_c10d._world.pg_map)
    runtime = SAERuntime.from_layout(tp_size=2, dp_size=1, placement_size=2,
                                     hooks=tuple(f"h{i}" for i in range(6)), backend="nccl")
    trainer = build(runtime, Path(directory), "on", ga=1, architecture="unified_multi_hook")
    # Load native GEMM/mapping kernels before introducing a delayed collective;
    # a first CUDA module load can synchronize away the intended in-flight work.
    train_runtime_window(trainer, [{h: torch.randn(7, 16, device=f"cuda:{rank}") for h in trainer.units}])
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    monitor = StaticFailureMonitor(runtime, set(dist.distributed_c10d._world.pg_map) - before)
    runtime.failure_monitor = monitor
    first, second, _ = (u.model for u in trainer.units.values())
    encode_first, encode_second = first.tp_wavefront_encode_launch, second.tp_wavefront_encode_launch
    pending = None

    def slow_first(step_input):
        nonlocal pending
        with torch.cuda.stream(_wavefront_stream(first.encoder.weight.device)):
            torch.cuda._sleep(300000000)
        state = encode_first(step_input)
        pending = not state.gather.ready.query()
        return state

    def fail_second(step_input):
        if rank == 0:
            assert pending, "The injected TP gather must still be in flight"
            raise RuntimeError("injected TP wavefront failure with gather in flight")
        return encode_second(step_input)

    first.tp_wavefront_encode_launch, second.tp_wavefront_encode_launch = slow_first, fail_second
    started = time.monotonic()
    try:
        train_runtime_window(trainer, [{h: torch.randn(7, 16, device=f"cuda:{rank}") for h in trainer.units}])
        monitor.finish()
        raise AssertionError("Expected peer failure")
    except BaseException as exc:
        monitor.fail(exc)
        with pytest.raises(RuntimeError, match="completed gradient"):
            require_window_boundary(trainer)
    finally:
        monitor.close()
        runtime.close()
    assert "injected TP wavefront failure" in monitor.error, monitor.error
    assert set(dist.distributed_c10d._world.pg_map) == before
    (Path(directory) / f"rank{rank}.json").write_text(json.dumps(dict(
        pending_at_injection=pending, checkpoint_refused=True, owned_groups_released=True,
        elapsed_s=time.monotonic() - started, first_cause=monitor.error,
    ), indent=2))
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_wavefront_failure_with_collective_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    directory = Path(os.environ.get("SAE_TP_WAVEFRONT_REPORT_DIR", tmp_path)).resolve() / "failure"
    directory.mkdir(parents=True, exist_ok=True)
    ctx = mp.spawn(_failure_worker, args=(str(directory),), nprocs=4, join=False)
    deadline = time.monotonic() + 60
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError("TP wavefront failure propagation timed out")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
        for process in ctx.processes:
            process.join(timeout=5)
