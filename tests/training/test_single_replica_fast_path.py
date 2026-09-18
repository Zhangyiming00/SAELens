"""DP1 direct-gradient ownership and migration of real native Adam shards."""

import copy
import json
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.sae_runtime import SAERuntime
from sae_lens.training.gradient_window import train_runtime_window
from sae_lens.training.runtime_checkpoint import _load_single_replica_adam
from tests.training.test_hook_optimizer_overlap import (
    build,
    equal,
    full_adam,
    snapshot,
)


def _worker(rank, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=4,
        init_method=f"file://{directory}/world", timeout=timedelta(seconds=120),
    )
    reports = []
    try:
        for tp, spp in ((1, 1), (2, 1), (4, 1), (2, 2)):
            runtime = SAERuntime.from_layout(
                tp_size=tp, dp_size=1, placement_size=spp, hooks=("h0", "h1", "h2")
            )
            failure = None
            try:
                if runtime.local is None:
                    continue
                path = Path(directory) / f"tp{tp}spp{spp}"
                old = build(runtime, path, "on", ga=1, amp=True, sharded=True, fast_path=False, ga1_norm=False)
                assert all(getattr(u.optimizer, "_sae_distributed_optimizer", False) for u in old.units.values())
                for step in range(2):
                    torch.manual_seed(100 + step)
                    batch = {h: torch.randn(7 + step, 16, device=f"cuda:{rank}") for h in old.units}
                    train_runtime_window(old, [batch])
                    old.lr_scheduler.step(old._last_updated_hooks)
                    old.n_training_steps += 1
                old.save_checkpoint("old_shards")
                new = build(runtime, path, "on", ga=1, amp=True, sharded=True)
                for u in new.units.values():
                    assert u.ddp is u.model
                    assert not getattr(u.optimizer, "_sae_distributed_optimizer", False)
                    assert all(not hasattr(p, "main_grad") for p in u.model.parameters())
                    assert u.model._tp_group is runtime.local.tp_group
                new.load_trainer_state(path / "old_shards")
                for h in new.units:
                    equal(old.units[h].model.state_dict(), new.units[h].model.state_dict())
                    torch.testing.assert_close(
                        full_adam(old.units[h]), full_adam(new.units[h]), atol=0, rtol=0
                    )
                equal(old.lr_scheduler.state_dict(), new.lr_scheduler.state_dict())
                equal(old.grad_scaler.state_dict(), new.grad_scaler.state_dict())
                for step in range(2, 5):
                    torch.manual_seed(100 + step)
                    batch = {h: torch.randn(7 + step, 16, device=f"cuda:{rank}") for h in old.units}
                    for trainer in (old, new):
                        train_runtime_window(trainer, [batch])
                        trainer.lr_scheduler.step(trainer._last_updated_hooks)
                        trainer.n_training_steps += 1
                    equal(snapshot(old), snapshot(new))
                saved = torch.load(
                    path / "old_shards" / f"distributed_adam_rank{rank}.pt",
                    map_location="cpu", weights_only=True,
                )
                first = next(iter(new.units))
                bad = copy.deepcopy(saved[first])
                key = next(iter(bad["signature"]["ranges"]))
                start, end, shape = bad["signature"]["ranges"][key]
                bad["signature"]["ranges"][key] = (start, end - 1, shape)
                with pytest.raises(ValueError, match="complete matching parameters"):
                    _load_single_replica_adam(new.units[first], bad)
                reports.append(dict(tp=tp, spp=spp, native_grad_buffer_bytes=0,
                                    native_shard_moments_restored=True, continued_windows=3,
                                    partial_shard_rejected=True))
            except BaseException:
                failure = traceback.format_exc()
                (Path(directory) / f"failure_rank{rank}.txt").write_text(failure)
            finally:
                failures = [None] * 4
                dist.all_gather_object(failures, failure, group=runtime.control_group)
                runtime.close()
                if any(failures):
                    raise AssertionError(str(failures))
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
def test_direct_gradients_and_dp1_sharded_checkpoint_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    directory = Path(os.environ.get("SAE_FAST_PATH_REPORT_DIR", tmp_path)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    context = mp.spawn(_worker, args=(str(directory),), nprocs=4, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("DP1 migration validation timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def _failure_worker(rank, directory):
    from sae_lens.static_failure import StaticFailureMonitor
    from sae_lens.training.gradient_window import require_window_boundary

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", rank=rank, world_size=4,
        init_method=f"file://{directory}/world", timeout=timedelta(seconds=45),
    )
    before = set(dist.distributed_c10d._world.pg_map)
    runtime = SAERuntime.from_layout(
        tp_size=2, dp_size=1, placement_size=2, hooks=tuple(f"h{i}" for i in range(6)),
        backend="nccl",
    )
    trainer = build(runtime, Path(directory), "on", ga=1, sharded=True)
    assert all(u.ddp is u.model for u in trainer.units.values())
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    monitor = StaticFailureMonitor(runtime, set(dist.distributed_c10d._world.pg_map) - before)
    runtime.failure_monitor = monitor
    first, second, _ = list(trainer.units.values())
    original_step, original_forward = first.optimizer.step, second.forward
    pending = None

    def slow_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        torch.cuda._sleep(300000000)
        return result

    def fail_forward(*args, **kwargs):
        nonlocal pending
        if rank == 0:
            pending = not first.params_complete()
            assert pending
            raise RuntimeError("injected direct-gradient update failure")
        return original_forward(*args, **kwargs)

    first.optimizer.step, second.forward = slow_step, fail_forward
    started = time.monotonic()
    refused = False
    try:
        for _ in range(4):
            train_runtime_window(trainer, [{h: torch.randn(7, 16, device=f"cuda:{rank}") for h in trainer.units}])
        monitor.finish()
        raise AssertionError("Expected peer failure")
    except BaseException as exc:
        monitor.fail(exc)
        with pytest.raises(RuntimeError, match="completed gradient"):
            require_window_boundary(trainer)
        refused = True
    finally:
        monitor.close()
        runtime.close()
    assert "injected direct-gradient update failure" in monitor.error
    assert set(dist.distributed_c10d._world.pg_map) == before
    (Path(directory) / f"rank{rank}.json").write_text(json.dumps(dict(
        pending_at_injection=pending, checkpoint_refused=refused,
        owned_groups_released=True, elapsed_s=time.monotonic() - started,
        first_cause=monitor.error,
    ), indent=2))
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
def test_direct_gradient_failure_with_update_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    root = Path(os.environ.get("SAE_FAST_PATH_REPORT_DIR", tmp_path)).resolve()
    directory = root / "direct_failure"
    directory.mkdir(parents=True, exist_ok=True)
    context = mp.spawn(_failure_worker, args=(str(directory),), nprocs=4, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Direct-gradient failure propagation timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
