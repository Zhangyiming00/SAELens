"""Real distributed placement AMP histories survive disk resume independently."""

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain
from tests.training.test_sae_resume_boundaries import (
    assert_state_equal,
    build_trainer,
    step,
)


def _placement_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=rendezvous)
    runtime = SAERuntime(
        (
            SAETrainingDomain("first", (0, 1), 1, ("h0", "h1")),
            SAETrainingDomain("second", (2, 3), 1, ("h2", "h3")),
        ),
        backend="gloo",
    )
    try:
        hooks = runtime.local.domain.hooks
        original, models = build_trainer(runtime, hooks, Path(output))
        original.grad_scaler = torch.amp.GradScaler(
            "cpu", init_scale=64, growth_interval=2
        )
        batch = torch.linspace(-1, 1, 176).reshape(11, 16)
        for index in range(3):
            handle = None
            if rank >= 2 and index == 1:
                handle = next(iter(models.values())).encoder.weight.register_hook(
                    lambda grad: torch.full_like(grad, float("inf"))
                )
            step(original, batch)
            original.n_training_samples += len(batch)
            if handle is not None:
                handle.remove()
        original.save_checkpoint("cut")
        path = Path(original.cfg.checkpoint_path) / "cut"
        scales = [None] * 4
        dist.all_gather_object(scales, original.grad_scaler.state_dict())
        assert scales[0] == scales[1] and scales[2] == scales[3]
        assert scales[0]["scale"] != scales[2]["scale"]
        restored, restored_models = build_trainer(runtime, hooks, Path(output))
        restored.grad_scaler = torch.amp.GradScaler("cpu")
        restored.load_trainer_state(path)
        assert_state_equal(
            original.grad_scaler.state_dict(), restored.grad_scaler.state_dict()
        )
        assert original.n_training_samples == restored.n_training_samples
        for _ in range(3):
            step(original, batch)
            step(restored, batch)
        for hook in hooks:
            assert_state_equal(
                models[hook].state_dict(), restored_models[hook].state_dict()
            )
        assert_state_equal(
            original.optimizer.state_dict(), restored.optimizer.state_dict()
        )
        assert_state_equal(
            original.grad_scaler.state_dict(), restored.grad_scaler.state_dict()
        )
        # Missing placement state must not silently fall back to placement zero.
        dist.barrier()
        if rank == 2:
            (path / "placement_state_1.pt").unlink()
        dist.barrier()
        if rank >= 2:
            import pytest

            from sae_lens.training.runtime_checkpoint import load_runtime_trainer_state

            with pytest.raises(ValueError, match="placement AMP"):
                load_runtime_trainer_state(restored, path)
    finally:
        runtime.close()
        dist.destroy_process_group()


def test_placement_amp_overflow_history_and_resume(tmp_path):
    mp.spawn(
        _placement_worker,
        args=(f"file://{tmp_path / 'world'}", str(tmp_path)),
        nprocs=4,
    )
