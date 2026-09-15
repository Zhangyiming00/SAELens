"""Real trainer updates and disk resume against the pinned upstream oracle.

CPU TP1 tests use a size-one fake group: no distributed validation is claimed.
NCCL workers use real TP x DDP, unequal token counts, and independent hooks.
"""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.multi_hook_sae import MultiHookSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.sae_trainer import SAETrainer

REFERENCE = Path(__file__).resolve().parents[1] / "native_reference"
TOLERANCE = dict(atol=3e-6, rtol=3e-5)


def _trainer_config(device, checkpoint, architecture):
    cfg = SAETrainerConfig(
        device=device,
        n_checkpoints=0,
        total_training_samples=66,
        train_batch_size_samples=11,
        output_path=None,
        save_mse_every_n_steps=0,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=0,
        record_memory_empty_cache=False,
        record_memory_timeline_step=-1,
        synchronize_timing=False,
        multi_sae_backward_order="forward",
        multi_sae_stats_sync_mode="immediate",
        multi_sae_stats_sync_interval=1,
        lr=3e-4,
        lr_end=3e-4,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        dead_feature_window=100,
        feature_sampling_window=100,
        autocast=False,
        checkpoint_path=str(checkpoint),
        quiesce_checkpoint_path=None,
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
    )
    cfg.multi_sae_distributed_architecture = architecture
    cfg.multi_sae_optimizer_overlap = "off"
    cfg.multi_sae_stats_sync_mode = "immediate"
    return cfg


def _exercise_trainers(
    tp_group, dp_group, device, output_dir, architecture, runtime=None, batch_provider=None
):
    """Run six updates; after update three use the actual trainer disk loaders."""
    inputs = load_file(REFERENCE / "inputs.safetensors")
    golden = load_file(REFERENCE / "golden.safetensors")
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    initial = {
        k.removeprefix("initial."): v
        for k, v in inputs.items()
        if k.startswith("initial.")
    }
    dp_size = dp_group.size() if dp_group is not None else 1
    dp_rank = dp_group.rank() if dp_group is not None else 0
    # Each hook uses a different normalization/bias configuration. This catches
    # mixing hook states and clipping the combined model instead of each SAE.
    cases = {"blocks.0.hook_resid_post": 3}
    if architecture != "single":
        cases["blocks.1.hook_resid_post"] = 0
    if runtime is not None:
        cases = {hook: (3 if hook == "blocks.0.hook_resid_post" else 0)
                 for hook in runtime.require_local().domain.hooks}
    cfg = _trainer_config(device, output_dir, architecture)
    errors = {"parameter": 0.0, "adam": 0.0, "reconstruction": 0.0}

    def build():
        models = {}
        for hook, case in cases.items():
            model_cfg = TopKTrainingSAEConfig.from_dict(manifest["configs"][case])
            model_cfg.device = device
            if runtime is not None:
                from types import SimpleNamespace
                from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

                runner = object.__new__(LanguageModelSAETrainingRunner)
                runner.sae_runtime = runtime
                runner.cfg = SimpleNamespace(get_training_sae_cfg_dict=model_cfg.to_dict)
                models[hook] = runner._create_training_sae(seed=791, tp_group=None, device=device)
                assert models[hook].parallel_context is runtime
            else:
                models[hook] = MegatronTopKSAE(model_cfg, tp_group=tp_group)
            models[hook].import_saelens_state_dict(initial)
        kwargs = {"process_group": dp_group, "gradient_as_bucket_view": True}
        if device.startswith("cuda"):
            kwargs["device_ids"] = [torch.device(device).index]
        if architecture == "single":
            model = next(iter(models.values()))
            wrapped = DDP(model, **kwargs) if dp_size > 1 else model
            trainer = SAETrainer(
                cfg=cfg,
                sae=wrapped,
                base_sae=model,
                data_provider=MagicMock(),
                dp_group=dp_group,
                token_count_weighted_dp=True,
                runtime=runtime,
            )
        else:
            wrapped_models = models
            root = None
            if architecture == "unified_multi_hook":
                root = MultiHookSAE(list(models), models)
                root = DDP(root, **kwargs) if dp_size > 1 else root
            elif dp_size > 1:
                wrapped_models = {h: DDP(m, **kwargs) for h, m in models.items()}
            trainer = MultiSAETrainer(
                hook_names=list(models),
                sae_by_hook=wrapped_models,
                base_sae_by_hook=models,
                data_provider=MagicMock(),
                save_checkpoint_fn=None,
                cfg=cfg,
                dp_group=dp_group,
                token_count_weighted_dp=True,
                sae_dp_mode="ddp",
                multi_hook_sae=root,
                runtime=runtime,
            )
            if runtime is not None:
                assert len(trainer.units) == len(cases)
                assert len({id(u.optimizer) for u in trainer.units.values()}) == len(cases)
                assert all(u.parallel_context is runtime for u in trainer.units.values())
        return trainer, models

    def check(actual, expected, category):
        actual = actual.detach().cpu()
        errors[category] = max(errors[category], (actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, **TOLERANCE)

    trainer, models = build()
    for step, batch in enumerate(inputs["batches"]):
        local_batch = torch.tensor_split(batch, dp_size)[dp_rank].to(device)
        start = sum(
            chunk.shape[0] for chunk in torch.tensor_split(batch, dp_size)[:dp_rank]
        )
        local_slice = slice(start, start + len(local_batch))
        if batch_provider is not None:
            local_batch, local_slice = batch_provider(step)
        for hook in cases:
            mask = inputs["dead_masks"][step].to(device)
            counts = mask.long() * (cfg.dead_feature_window + 1)
            if architecture == "single":
                trainer.n_forward_passes_since_fired = counts
            else:
                trainer.n_forward_passes_since_fired_by_hook[hook] = counts
        if architecture == "single":
            output = trainer._train_step(trainer.sae, local_batch)[0]
            outputs = {next(iter(cases)): output}
        else:
            outputs, _ = trainer._train_step(
                {h: local_batch for h in cases}, len(local_batch)
            )
            trainer.lr_scheduler.step()
        trainer.n_training_steps += 1
        trainer.n_training_samples += len(batch)
        for hook, model in models.items():
            prefix = f"case{cases[hook]}.step{step}."
            if len(local_batch):
                check(
                    outputs[hook].sae_out,
                    golden[prefix + "reconstruction"][local_slice],
                    "reconstruction",
                )
            for name, value in model.export_saelens_state_dict().items():
                check(value, golden[prefix + "parameter." + name], "parameter")
            states = {
                n: copy.deepcopy(trainer.optimizer.state[p])
                for n, p in model.named_parameters()
            }
            assert all(states.values()), (
                "Resume must restore every parameter's Adam state"
            )
            model.process_named_optimizer_state_for_saving(states)
            for name, values in states.items():
                for key, value in values.items():
                    check(value, golden[prefix + f"adam.{name}.{key}"], "adam")
        if step == 2:
            trainer.save_checkpoint("midpoint")
            # Other DP replicas wait until replica zero finishes writing.
            if dist.get_backend() != "fake":
                dist.barrier(group=runtime.training_group if runtime else None)
            checkpoint = Path(output_dir) / "midpoint"
            trainer, models = build()
            if architecture == "single":
                next(iter(models.values())).load_weights_from_checkpoint(checkpoint)
            trainer.load_trainer_state(checkpoint)
            assert trainer.n_training_steps == 3
            assert trainer.n_training_samples == 33
            assert len(trainer.optimizer.state) == sum(
                len(list(m.parameters())) for m in models.values()
            )
    return errors


@pytest.mark.parametrize(
    "architecture", ["single", "legacy_per_hook_wrapper", "unified_multi_hook"]
)
def test_tp1_trainer_resume_against_native_oracle(tmp_path, monkeypatch, architecture):
    pytest.importorskip("megatron.core")
    import torch.testing._internal.distributed.fake_pg  # noqa: F401

    monkeypatch.setenv("SAE_ADAM_IMPL", "forloop")
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        _exercise_trainers(dist.group.WORLD, None, "cpu", tmp_path, architecture)
    finally:
        dist.destroy_process_group()


def _distributed_worker(rank, rendezvous, work_dir, backend, world_size):
    os.environ["SAE_ADAM_IMPL"] = "forloop"
    torch.set_num_threads(1)
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    reports = []
    try:
        for tp_size in (1, 2, 4) if backend == "nccl" else (1,):
            tp_groups = [
                dist.new_group(list(range(i, i + tp_size)))
                for i in range(0, world_size, tp_size)
            ]
            dp_groups = [
                dist.new_group(list(range(i, world_size, tp_size)))
                for i in range(tp_size)
            ]
            tp_group, dp_group = tp_groups[rank // tp_size], dp_groups[rank % tp_size]
            for architecture in (
                "single",
                "legacy_per_hook_wrapper",
                "unified_multi_hook",
            ):
                directory = Path(work_dir) / f"tp{tp_size}_{architecture}"
                errors = _exercise_trainers(
                    tp_group, dp_group, device, directory, architecture
                )
                reports.append(
                    {
                        "tp": tp_size,
                        "dp": world_size // tp_size,
                        "architecture": architecture,
                        "steps": 6,
                        "resume_step": 3,
                        "max_abs_error": errors,
                    }
                )
                dist.barrier()
            for group in tp_groups + dp_groups:
                if group != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(group)
        (Path(work_dir) / f"trainer_rank{rank}.json").write_text(
            json.dumps(reports, indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="TP x DP trainer acceptance requires four CUDA GPUs",
)
def test_tp_ddp_trainers_and_resume(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), "nccl", 4),
        nprocs=4,
        join=True,
    )
