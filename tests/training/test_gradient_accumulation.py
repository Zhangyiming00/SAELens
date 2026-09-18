"""NCCL acceptance against a separately imported, pinned upstream SAELens wheel."""

import copy
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.dp_batch import balanced_token_counts
from sae_lens.training.megatron_ddp import is_megatron_ddp, wrap_runtime_sae
from sae_lens.training.megatron_optimizer import (
    MEGATRON_GROUP_METADATA,
    is_megatron_optimizer,
)
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.sae_train_unit import SAETrainUnit
from sae_lens.training.sae_trainer import SAETrainer
from tests.saes.test_megatron_sae_trainers import _trainer_config

ROOT = Path(__file__).resolve().parents[2]
TOL = dict(atol=4e-6, rtol=4e-5)


class Batches:
    def __init__(self, plan, context, single, device, position=0):
        self.plan, self.context, self.single, self.device = (
            plan,
            context,
            single,
            device,
        )
        self.position = position
        self.seen = []

    def __next__(self):
        i = self.position
        if i == 7:
            raise StopIteration
        self.position += 1
        self.seen.append(i)
        batch = {}
        dp, rank = self.context.dp_group.size(), self.context.dp_rank
        for h in self.context.domain.hooks:
            x = self.plan["batches"][h][i]
            counts = list(balanced_token_counts(len(x), dp, i))
            # Explicitly empty one replica in an otherwise nonempty batch.
            if dp > 1 and i == 1:
                counts[-1] += counts[0]
                counts[0] = 0
            start = sum(counts[:rank])
            batch[h] = x[start : start + counts[rank]].to(self.device)
        return next(iter(batch.values())) if self.single else batch


def native_state(model, tensors):
    tensors = copy.deepcopy(tensors)
    model.process_named_optimizer_state_for_saving(tensors)
    return tensors


def exercise(runtime, plan, golden, path, accumulation, bucket, single):
    ctx = runtime.require_local()
    device = f"cuda:{dist.get_rank()}"
    hooks = ctx.domain.hooks
    path.mkdir(parents=True, exist_ok=True)
    cfg = _trainer_config(device, path, "legacy_per_hook_wrapper")
    cfg.gradient_accumulation_steps = accumulation
    cfg.train_batch_size_samples = balanced_token_counts(11, ctx.dp_group.size())[
        ctx.dp_rank
    ]
    cfg.total_training_samples = cfg.train_batch_size_samples * 7
    cfg.routing_dp_batch_mode = "exact"
    cfg.lr_scheduler_name = "cosineannealing"
    cfg.lr_end = 3e-5
    cfg.dead_feature_window = 0
    errors = dict(grad=0.0, coefficient=0.0, clipped=0.0, parameters=0.0, adam=0.0)
    snapshots = {}
    scaled = accumulation == 3 and bucket >= 1.0

    def comparable_adam_state(optimizer_state):
        # Native group identifiers are extra checkpoint bookkeeping, not
        # numerical hyperparameters or optimizer moments.
        return {**optimizer_state, "param_groups": [
            {k: v for k, v in g.items() if k not in MEGATRON_GROUP_METADATA}
            for g in optimizer_state["param_groups"]
        ]}

    def compare(actual, expected, kind):
        if isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                compare(actual[key], expected[key], kind)
        else:
            torch.testing.assert_close(actual, expected, **TOL)
            if torch.is_tensor(actual) and actual.numel():
                errors[kind] = max(errors[kind], (actual - expected).abs().max().item())

    def build(position=0, *, late_reference=False, legacy_optimizer=False):
        events = []
        audit = {}
        models = {
            h: MegatronTopKSAE(
                TopKTrainingSAEConfig.from_dict(
                    {**plan["configs"][h], "device": device}
                ),
                runtime=runtime,
            )
            for h in hooks
        }
        wrapped = {}
        for h, model in models.items():
            model.import_saelens_state_dict(plan["initial"])
            wrapped[h] = wrap_runtime_sae(
                model, runtime, bucket_cap_mb=bucket, single_replica_fast_path=False
            )
            assert is_megatron_ddp(wrapped[h])
            assert wrapped[h].dp_group is ctx.dp_group
            assert wrapped[h].buffers
        provider = Batches(plan, ctx, single, device, position)
        kwargs = dict(
            cfg=cfg,
            data_provider=provider,
            runtime=runtime,
            dp_group=ctx.dp_group,
            token_count_weighted_dp=True,
        )
        with ExitStack() as stack:
            if legacy_optimizer:
                # Reproduce the previous release's torch Adam + model clip
                # path. No production option can opt out of Megatron on CUDA.
                def old_optimizer(model, _runtime, *, adam_kwargs):
                    kwargs = {**adam_kwargs, "fused": True}
                    kwargs.pop("foreach", None)
                    return torch.optim.Adam(model.parameters(), **kwargs)

                for module in ("sae_trainer", "multi_sae_trainer"):
                    stack.enter_context(patch(
                        f"sae_lens.training.{module}.build_runtime_optimizer", old_optimizer
                    ))
            if single:
                trainer = SAETrainer(
                    sae=wrapped[hooks[0]], base_sae=models[hooks[0]], **kwargs
                )
            else:
                trainer = MultiSAETrainer(
                    hook_names=list(hooks),
                    sae_by_hook=wrapped,
                    base_sae_by_hook=models,
                    save_checkpoint_fn=None,
                    sae_dp_mode="ddp",
                    **kwargs,
                )
        if scaled:
            trainer.grad_scaler = torch.amp.GradScaler(
                "cuda", init_scale=64, growth_interval=2
            )
        assert trainer.global_update_batch_size == 11 * accumulation
        units = {hooks[0]: trainer.unit} if single else trainer.units
        trainer._schedule_audit = audit
        trainer._communication_events = events
        trainer._gradient_collectives = {
            id(b.grad_data): (h, index)
            for h, unit in units.items()
            for index, b in enumerate(b for buf in unit.ddp.buffers for b in buf.buckets)
        }
        for h, unit in units.items():
            assert is_megatron_optimizer(unit.optimizer) != legacy_optimizer
            if not legacy_optimizer:
                assert unit.optimizer.tp_group is ctx.tp_group
                assert unit.optimizer.grad_stats_parallel_group is ctx.tp_group
                assert unit.optimizer.config.clip_grad == 1.0
                for method in ("prepare_grads", "step_with_ready_grads"):
                    original = getattr(unit.optimizer, method)

                    def native_update(*args, original=original, method=method, h=h, unit=unit, **kwargs):
                        assert unit._grad_sync_finished
                        events.append((trainer.n_training_steps, method, h))
                        return original(*args, **kwargs)

                    setattr(unit.optimizer, method, native_update)

                def forbidden_clip(*_args, **_kwargs):
                    raise AssertionError("CUDA updates must use Megatron's clip, not the model's")

                unit.model.clip_grad_norm_ = forbidden_clip
            assert unit.optimizer.param_groups[0]["fused"]
            assert unit.early_grad_sync
            if late_reference:
                unit.early_grad_sync = False
            original_forward = unit.forward
            original_start = unit.ddp.start_grad_sync
            original_backward = unit.backward

            def forward(*args, h=h, original_forward=original_forward, **kwargs):
                events.append((trainer.n_training_steps, "forward", h))
                return original_forward(*args, **kwargs)

            def start(*args, h=h, unit=unit, original_start=original_start, **kwargs):
                # Gradients remain Megatron-owned until the window-end finish.
                assert all(p.grad is None for p in unit.model.parameters())
                events.append((trainer.n_training_steps, "start", h))
                return original_start(*args, **kwargs)

            def backward(*args, h=h, original_backward=original_backward, **kwargs):
                events.append((trainer.n_training_steps, "backward_begin", h, kwargs.get("sync_gradients", False)))
                result = original_backward(*args, **kwargs)
                events.append((trainer.n_training_steps, "backward_end", h))
                return result

            unit.forward = forward
            unit.ddp.start_grad_sync = start
            unit.backward = backward
            clip_owner = unit if legacy_optimizer else unit.optimizer
            original_clip = clip_owner.clip_grad_norm

            def clip(max_norm=1.0, h=h, unit=unit, original_clip=original_clip):
                step = trainer.n_training_steps
                events.append((step, "clip", h))
                expected = golden[accumulation][h]["snapshots"][step]
                gradients = native_state(
                    unit.model,
                    {
                        n: {"g": p.grad.detach().clone()}
                        for n, p in unit.model.named_parameters()
                    },
                )
                compare(
                    {n: v["g"].cpu() for n, v in gradients.items()},
                    expected["grad"],
                    "grad",
                )
                audit[(step, h, "grad")] = {n: v["g"].cpu() for n, v in gradients.items()}
                norm = torch.as_tensor(original_clip(max_norm))
                if not legacy_optimizer:
                    assert norm.is_cuda and norm.ndim == 0
                audit[(step, h, "norm")] = norm.detach().cpu()
                torch.testing.assert_close(norm.cpu(), expected["norm"], **TOL)
                coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
                expected_coefficient = (max_norm / (expected["norm"] + 1e-6)).clamp(max=1.0)
                compare(coefficient.cpu(), expected_coefficient, "coefficient")
                audit[(step, h, "coefficient")] = coefficient.detach().cpu()
                gradients = native_state(
                    unit.model,
                    {
                        n: {"g": p.grad.detach().clone()}
                        for n, p in unit.model.named_parameters()
                    },
                )
                compare(
                    {n: v["g"].cpu() for n, v in gradients.items()},
                    expected["clipped"],
                    "clipped",
                )
                audit[(step, h, "clipped")] = {n: v["g"].cpu() for n, v in gradients.items()}
                return norm

            clip_owner.clip_grad_norm = clip

        def checkpoint():
            step = trainer.n_training_steps - 1
            window_events = [event[1:] for event in events if event[0] == step]
            microbatches = trainer._last_window_microbatches
            native = microbatches == accumulation and not late_reference
            assert [e for e in window_events if e[0] == "forward"] == [
                ("forward", h) for _ in range(microbatches) for h in hooks
            ]
            assert [e for e in window_events if e[0] == "backward_begin"] == [
                ("backward_begin", h, native and mb == microbatches-1)
                for mb in range(microbatches) for h in hooks
            ]
            assert [e for e in window_events if e[0] == "start"] == (
                [] if native else [("start", h) for h in hooks]
            )
            # Count actual gradient collectives, including the first-window
            # fallback in Megatron.finish_grad_sync, not Python start calls.
            expected_reductions = sorted(trainer._gradient_collectives.values())
            assert sorted(e[1:] for e in window_events if e[0] == "reduce") == expected_reductions
            ends = [i for i, e in enumerate(window_events) if e[0] == "backward_end"]
            for h in hooks:
                reductions = [i for i, e in enumerate(window_events) if e[:2] == ("reduce", h)]
                if native and step > position // accumulation:
                    begin = max(i for i, e in enumerate(window_events) if e[:2] == ("backward_begin", h))
                    end = max(i for i, e in enumerate(window_events) if e[:2] == ("backward_end", h))
                    assert all(begin < i < end for i in reductions)
                else:
                    assert all(i > max(ends) for i in reductions)
            # fit() has now advanced scheduler and update counters.
            for h, unit in units.items():
                active = trainer._last_global_tokens_by_hook[h] > 0
                assert [e for e in window_events if e == ("clip", h)] == ([("clip", h)] if active else [])
                if not legacy_optimizer:
                    assert [e for e in window_events if e[0] in ("prepare_grads", "clip", "step_with_ready_grads") and e[1] == h] == ([
                        ("prepare_grads", h), ("clip", h), ("step_with_ready_grads", h)
                    ] if active else [])
                expected = golden[accumulation][h]["snapshots"][step]
                state = unit.model.export_saelens_state_dict()
                compare(
                    {n: t.cpu() for n, t in state.items()},
                    expected["parameters"],
                    "parameters",
                )
                audit[(step, h, "parameters")] = {n: t.cpu() for n, t in state.items()}
                adam = native_state(
                    unit.model,
                    {
                        n: unit.optimizer.state[p]
                        for n, p in unit.model.named_parameters()
                    },
                )
                compare(
                    {n: {k: v.cpu() for k, v in s.items()} for n, s in adam.items()},
                    expected["adam"],
                    "adam",
                )
                audit[(step, h, "adam")] = {
                    n: {k: v.detach().cpu().clone() for k, v in s.items()}
                    for n, s in adam.items()
                }
                assert unit.optimizer.param_groups[0]["lr"] == expected["lr"]
                audit[(step, h, "lr")] = unit.optimizer.param_groups[0]["lr"]
                age = (
                    trainer.n_forward_passes_since_fired
                    if single
                    else trainer.n_forward_passes_since_fired_by_hook[h]
                )
                torch.testing.assert_close(
                    age.cpu(), golden[accumulation][h]["ages"][step], rtol=0, atol=0
                )
                assert all(
                    p.grad is None and not p.main_grad.any()
                    for p in unit.model.parameters()
                )
            snapshots[trainer.n_training_steps] = copy.deepcopy(
                trainer.optimizer.state_dict()
            )
            if trainer.n_training_steps == 1 and position == 0 and not late_reference:
                name = "legacy_window" if legacy_optimizer else "window"
                trainer.save_checkpoint(name)
                torch.save(
                    provider.position,
                    path / name / f"provider_rank{dist.get_rank()}.pt",
                )

        trainer._checkpoint_if_needed = checkpoint
        return trainer, models, provider

    def fit_audited(trainer):
        original_all_reduce = dist.all_reduce

        def all_reduce(tensor, *args, **kwargs):
            key = trainer._gradient_collectives.get(id(tensor))
            if key is not None:
                trainer._communication_events.append((trainer.n_training_steps, "reduce", *key))
            return original_all_reduce(tensor, *args, **kwargs)

        with patch.object(dist, "all_reduce", all_reduce):
            trainer.fit()

    continuous, models, provider = build()
    fit_audited(continuous)
    assert provider.seen == list(range(7))
    assert continuous.n_training_steps == (7 + accumulation - 1) // accumulation
    final = copy.deepcopy(continuous.optimizer.state_dict())
    expected_models = {h: copy.deepcopy(m.state_dict()) for h, m in models.items()}
    final_scheduler = copy.deepcopy(continuous.lr_scheduler.state_dict())
    final_scaler = copy.deepcopy(continuous.grad_scaler.state_dict())
    reference, _, _ = build(late_reference=True)
    fit_audited(reference)
    torch.testing.assert_close(
        continuous._schedule_audit, reference._schedule_audit, rtol=0, atol=0
    )
    torch.testing.assert_close(reference.optimizer.state_dict(), final, rtol=0, atol=0)
    del reference
    position = torch.load(
        path / "window" / f"provider_rank{dist.get_rank()}.pt", weights_only=True
    )
    original_bucket = bucket
    for bucket in (original_bucket, 1.0 if original_bucket < 1 else 0.00025):
        restored, restored_models, restored_provider = build(position)
        restored.load_trainer_state(path / "window")
        if single:
            restored_models[hooks[0]].load_weights_from_checkpoint(path / "window")
        fit_audited(restored)
        # Bucket changes can change NCCL's floating-point summation order.
        tolerance = dict(rtol=0, atol=0) if bucket == original_bucket else TOL
        torch.testing.assert_close(restored.optimizer.state_dict(), final, **tolerance)
        torch.testing.assert_close(
            restored.lr_scheduler.state_dict(), final_scheduler, rtol=0, atol=0
        )
        assert restored.grad_scaler.state_dict() == final_scaler
        for h in hooks:
            torch.testing.assert_close(
                restored_models[h].state_dict(), expected_models[h], **tolerance
            )
        assert restored_provider.seen == list(range(position, 7))
    bucket = original_bucket
    legacy, _, _ = build(legacy_optimizer=True)
    fit_audited(legacy)
    torch.testing.assert_close(legacy._schedule_audit, continuous._schedule_audit, **TOL)
    torch.testing.assert_close(comparable_adam_state(legacy.optimizer.state_dict()), comparable_adam_state(final), **TOL)
    # The legacy writer has the old model clipping and optimizer class. Load
    # its actual on-disk checkpoint into native Megatron, checking state at
    # the boundary exactly and subsequent updates against both references.
    legacy_boundary = copy.deepcopy(snapshots[1])
    migrated, migrated_models, migrated_provider = build(position)
    migrated.load_trainer_state(path / "legacy_window")
    if single:
        migrated_models[hooks[0]].load_weights_from_checkpoint(path / "legacy_window")
    torch.testing.assert_close(comparable_adam_state(migrated.optimizer.state_dict()), comparable_adam_state(legacy_boundary), rtol=0, atol=0)
    fit_audited(migrated)
    torch.testing.assert_close(comparable_adam_state(migrated.optimizer.state_dict()), comparable_adam_state(legacy.optimizer.state_dict()), **TOL)
    torch.testing.assert_close(migrated.optimizer.state_dict(), final, **TOL)
    torch.testing.assert_close(migrated.lr_scheduler.state_dict(), final_scheduler, rtol=0, atol=0)
    assert migrated.grad_scaler.state_dict() == final_scaler
    assert migrated_provider.seen == list(range(position, 7))
    return dict(
        errors=errors,
        ga1_late_schedule_bitwise_equal=True if accumulation == 1 else None,
        native_vs_explicit_sync_bitwise_equal=True,
        native_bucket_ready_collectives_verified=True,
        native_optimizer_clip_and_step_once_verified=True,
        legacy_optimizer_resume="boundary_bitwise_equal; updates_within_tolerance",
        updates=continuous.n_training_steps,
        microbatches=7,
        resume="same_bucket_bitwise_equal; changed_bucket_within_tolerance",
        accumulation=accumulation,
        bucket_mb=original_bucket,
        single=single,
        enabled_scaler=scaled,
    )


def worker(rank, directory, rendezvous=None):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    directory = Path(directory)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=4,
        init_method=f"file://{rendezvous or directory / 'rdzv'}",
        timeout=timedelta(seconds=120),
    )
    try:
        plan = torch.load(directory / "plan.pt", weights_only=True)
        golden = torch.load(directory / "golden.pt", weights_only=True)
        reports = []
        for name, ranks, tp in (
            ("dp1", (0,), 1),
            ("dp3", (0, 1, 2), 1),
            ("tp2dp2", (0, 1, 2, 3), 2),
        ):
            for single in (True, False):
                hooks = ("h0",) if single else ("h0", "h1")
                runtime = SAERuntime((SAETrainingDomain(name, ranks, tp, hooks),))
                try:
                    if runtime.local is not None:
                        for accumulation, bucket in (
                            (1, 1.0),
                            (2, 0.00025),
                            (3, 0.00025),
                            (3, 1.0),
                        ):
                            result = exercise(
                                runtime,
                                plan,
                                golden,
                                directory / f"{name}_{single}_{accumulation}_{bucket}",
                                accumulation,
                                bucket,
                                single,
                            )
                            reports.append(dict(layout=name, **result))
                    dist.barrier()
                finally:
                    runtime.close()
        # If one rank cannot use implicit ordering, the whole TP x DP domain
        # must retain the late schedule. Simulate eligibility without mutating
        # NCCL's environment after communicators have initialized.
        import sae_lens.training.sae_train_unit as unit_module

        original_support = unit_module.supports_early_grad_sync
        unit_module.supports_early_grad_sync = lambda *_: rank != 0
        runtime = SAERuntime((SAETrainingDomain("mixed_support", (0, 1, 2, 3), 2, ("h0",)),))
        try:
            model = MegatronTopKSAE(
                TopKTrainingSAEConfig.from_dict({**plan["configs"]["h0"], "device": f"cuda:{rank}"}),
                runtime=runtime,
            )
            ddp = wrap_runtime_sae(model, runtime)
            unit = SAETrainUnit("h0", model, ddp, torch.optim.Adam(model.parameters(), fused=True), runtime)
            assert not unit.early_grad_sync
            (directory / f"eligibility_rank{rank}.json").write_text(json.dumps(dict(
                local_supported=rank != 0, domain_early_grad_sync=unit.early_grad_sync,
            )))
        finally:
            unit_module.supports_early_grad_sync = original_support
            runtime.close()
        (directory / f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_native_large_batch_and_resume(tmp_path, monkeypatch):
    # Set before any NCCL communicator initialization, including in spawned
    # workers. TP+DP uses NCCL 2.26+'s deterministic multi-communicator ordering.
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    directory = Path(
        os.environ.get("SAE_ACCUMULATION_REPORT_DIR", str(tmp_path))
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    fixture = load_file(ROOT / "tests/native_reference/inputs.safetensors")
    configs = json.loads((ROOT / "tests/native_reference/manifest.json").read_text())[
        "configs"
    ]
    torch.manual_seed(7239)
    plan = dict(
        initial={
            k.removeprefix("initial."): v
            for k, v in fixture.items()
            if k.startswith("initial.")
        },
        configs={"h0": configs[3], "h1": configs[0]},
        batches={
            h: [torch.randn(n, 16) for n in counts]
            for h, counts in {
                "h0": [11, 7, 3, 13, 5, 9, 4],
                "h1": [5, 0, 9, 0, 0, 0, 6],
            }.items()
        },
    )
    torch.save(plan, directory / "plan.pt")
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tests/native_reference/accumulation.py"),
            "--input",
            str(directory / "plan.pt"),
            "--output",
            str(directory / "golden.pt"),
        ],
        check=True,
        timeout=120,
    )
    # A failed run can leave FileStore keys behind in an explicit report
    # directory. Every rerun must rendezvous through a fresh file.
    rendezvous = directory / f"rdzv_{time.time_ns()}"
    context = mp.spawn(worker, args=(str(directory), str(rendezvous)), nprocs=4, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Accumulation acceptance workers did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
