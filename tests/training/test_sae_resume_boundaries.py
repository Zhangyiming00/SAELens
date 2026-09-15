"""Checkpoint and empty-step regressions through the real trainer APIs."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.sae_trainer import SAETrainer
from tests.saes.test_megatron_sae_boundaries import sae  # noqa: F401
from tests.saes.test_megatron_sae_trainers import _trainer_config


def build_trainer(runtime, hooks, path, device="cpu", sparse=False):
    cfg = _trainer_config(device, path, "legacy_per_hook_wrapper")
    cfg.lr_scheduler_name = "cosineannealing"
    cfg.lr_end = 3e-5
    context = runtime.require_local()
    models = {}
    wrapped = {}
    for i, hook in enumerate(hooks):
        torch.manual_seed(791 + i)
        model = MegatronTopKSAE(
            TopKTrainingSAEConfig(
                d_in=16,
                d_sae=32,
                k=4,
                device=device,
                use_sparse_activations=sparse,
            ),
            runtime=runtime,
        )
        models[hook] = model
        wrapped[hook] = (
            DDP(model, process_group=context.dp_group)
            if context.dp_group.size() > 1
            else model
        )
    kwargs = dict(
        cfg=cfg,
        data_provider=MagicMock(),
        dp_group=context.dp_group,
        token_count_weighted_dp=True,
        runtime=runtime,
    )
    if len(hooks) == 1:
        trainer = SAETrainer(sae=wrapped[hooks[0]], base_sae=models[hooks[0]], **kwargs)
    else:
        trainer = MultiSAETrainer(
            hook_names=list(hooks),
            sae_by_hook=wrapped,
            base_sae_by_hook=models,
            save_checkpoint_fn=None,
            sae_dp_mode="ddp",
            **kwargs,
        )
    return trainer, models


def step(trainer, batch):
    if isinstance(trainer, SAETrainer):
        trainer._train_step(trainer.sae, batch)
    else:
        trainer._train_step({h: batch for h in trainer.hook_names}, len(batch))
        if trainer._last_step_had_tokens:
            trainer.lr_scheduler.step()
    trainer.n_training_steps += 1


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("hook_count", [1, 2])
@pytest.mark.parametrize("warmup", [0, 2])
def test_cosine_resume_restores_groups_and_next_update(
    sae,  # noqa: F811
    tmp_path,
    hook_count,
    warmup,
):
    hooks = tuple(f"hook{i}" for i in range(hook_count))
    runtime = object.__new__(SAERuntime)
    runtime._closed = False
    runtime.local = SimpleNamespace(
        tp_group=sae._tp_group,
        dp_group=sae._tp_group,
        domain=SAETrainingDomain("test", (0,), 1, hooks),
        tp_ranks=(0,), dp_ranks=(0,), tp_rank=0, dp_rank=0,
    )
    runtime.domains = (runtime.local.domain,)
    original, models = build_trainer(runtime, hooks, tmp_path)
    # Exercise the real SequentialLR nested state across the warmup boundary.
    from sae_lens.training.optim import get_lr_scheduler
    from sae_lens.training.optimizer_checkpoint import UnitLRSchedulers

    def scheduler(optimizer):
        return get_lr_scheduler(
            "cosineannealing", optimizer, 8, 3e-4, warmup, 0, 3e-5, 1
        )

    if hook_count == 1:
        original.lr_scheduler = scheduler(original.optimizer)
    else:
        original.lr_scheduler = UnitLRSchedulers(
            {h: scheduler(u.optimizer) for h, u in original.units.items()}
        )
    for i, group in enumerate(original.optimizer.param_groups):
        group["betas"] = (0.8 + i * 0.05, 0.99)
        group["eps"] = 1e-7 * (i + 1)
    batch = torch.linspace(-1, 1, 176).reshape(11, 16)
    for _ in range(3):
        step(original, batch)
    path = tmp_path / "resume"
    if hook_count > 1:
        for h in hooks:
            original._save_one_checkpoint_model(path, h)
    original.save_trainer_state(path)
    restored, restored_models = build_trainer(runtime, hooks, tmp_path)
    if hook_count == 1:
        restored.lr_scheduler = scheduler(restored.optimizer)
        restored_models[hooks[0]].load_state_dict(models[hooks[0]].state_dict())
    else:
        restored.lr_scheduler = UnitLRSchedulers(
            {h: scheduler(u.optimizer) for h, u in restored.units.items()}
        )
    restored.load_trainer_state(path)
    assert original.optimizer.param_groups[0]["lr"] != 3e-4
    assert_state_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_state_equal(
        original.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict()
    )
    step(original, batch * 0.7)
    step(restored, batch * 0.7)
    for h in hooks:
        assert_state_equal(models[h].state_dict(), restored_models[h].state_dict())
    assert_state_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())


def _empty_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4, init_method=rendezvous)
    try:
        for tp_size in (1, 2, 4):
            for hook_count in (1, 2):
                hooks = tuple(f"hook{i}" for i in range(hook_count))
                runtime = SAERuntime(
                    (SAETrainingDomain("sae", (0, 1, 2, 3), tp_size, hooks),)
                )
                try:
                    trainer, models = build_trainer(
                        runtime, hooks, Path(output), f"cuda:{rank}", sparse=True
                    )
                    trainer.grad_scaler = torch.amp.GradScaler(
                        "cuda", growth_interval=2
                    )
                    batch = torch.linspace(-1, 1, 176, device=f"cuda:{rank}").reshape(
                        11, 16
                    )
                    step(trainer, batch)
                    before = copy.deepcopy(trainer.optimizer.state_dict())
                    before_scale = copy.deepcopy(trainer.grad_scaler.state_dict())
                    before_lr = copy.deepcopy(trainer.lr_scheduler.state_dict())
                    for _ in range(2):
                        step(trainer, batch[:0])
                        assert_state_equal(before, trainer.optimizer.state_dict())
                        assert_state_equal(
                            before_scale, trainer.grad_scaler.state_dict()
                        )
                        assert_state_equal(before_lr, trainer.lr_scheduler.state_dict())
                        assert all(
                            p.grad is None
                            for m in models.values()
                            for p in m.parameters()
                        )
                    # One empty replica must still enter all statistics and backward collectives.
                    local = (
                        batch[:0]
                        if tp_size < 4 and runtime.local.dp_rank == 0
                        else batch
                    )
                    step(trainer, local)
                    step(trainer, batch)
                finally:
                    runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_sparse_empty_replica_and_scaled_global_empty_steps(tmp_path):
    mp.spawn(
        _empty_worker, args=(f"file://{tmp_path / 'rdzv'}", str(tmp_path)), nprocs=4
    )
