"""Real CUDA fixed-order per-hook scheduling and state equivalence."""

import copy
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.sae_runtime import SAERuntime
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.gradient_window import train_runtime_window
from sae_lens.training.megatron_ddp import wrap_runtime_sae
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from tests.saes.test_megatron_sae_trainers import _trainer_config


def build(runtime, path, overlap, ga=3, amp=False, sharded=False, tiny=False, fast_path=True, ga1_norm=True, architecture="legacy_per_hook_wrapper"):
    ctx = runtime.require_local()
    device = f"cuda:{dist.get_rank()}"
    cfg = _trainer_config(device, path, architecture)
    cfg.gradient_accumulation_steps = ga
    cfg.autocast = amp
    cfg.ddp_zero_optimizer = sharded
    cfg.sae_single_replica_fast_path = fast_path
    cfg.sae_ga1_loss_normalization = ga1_norm
    cfg.multi_sae_param_gather_overlap = (
        os.environ.get("SAE_GATHER_OVERLAP", "1") == "1"
    )
    cfg.multi_sae_param_gather_schedule = os.environ.get("SAE_GATHER_SCHEDULE", "eager")
    cfg.multi_sae_optimizer_overlap = overlap
    models, wrapped = {}, {}
    for h in ctx.domain.hooks:
        torch.manual_seed(731 + int(h[1:]))
        models[h] = MegatronTopKSAE(
            TopKTrainingSAEConfig(
                d_in=2 if tiny else 16,
                d_sae=2 if tiny else 48,
                k=1 if tiny else 4,
                device=device,
                use_sparse_activations=False,
            ),
            runtime=runtime,
        )
        wrapped[h] = wrap_runtime_sae(
            models[h],
            runtime,
            bucket_cap_mb=0.000001 if tiny else 0.001,
            distributed_optimizer=sharded,
            single_replica_fast_path=fast_path,
        )
    trainer = MultiSAETrainer(
        hook_names=list(models),
        sae_by_hook=wrapped,
        base_sae_by_hook=models,
        data_provider=MagicMock(),
        save_checkpoint_fn=None,
        cfg=cfg,
        dp_group=ctx.dp_group,
        token_count_weighted_dp=True,
        sae_dp_mode="ddp",
        runtime=runtime,
    )
    for u in trainer.units.values():
        u.optimizer.param_groups[0].update(
            betas=(0.85, 0.97), eps=1e-7, weight_decay=0.02
        )
    if amp:
        trainer.grad_scaler = torch.amp.GradScaler(
            "cuda", init_scale=64, growth_interval=2
        )
    return trainer


NUMERICAL_ERRORS = {}


def equal(a, b, path=()):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        if a.is_floating_point() and a.numel():
            key = str(path[0]) if path else "direct"
            NUMERICAL_ERRORS[key] = max(
                NUMERICAL_ERRORS.get(key, 0.0), (a - b).abs().max().item()
            )
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k], path + (str(k),))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y, path)
    else:
        assert a == b


def snapshot(t):
    torch.cuda.synchronize()
    return copy.deepcopy(
        dict(
            models={h: u.model.state_dict() for h, u in t.units.items()},
            adam={h: full_adam(u) for h, u in t.units.items()},
            lr=t.lr_scheduler.state_dict(),
            scaler=t.grad_scaler.state_dict(),
            ages=t.n_forward_passes_since_fired_by_hook,
            freq=t.act_freq_scores_by_hook,
            updated=t._last_updated_hooks,
            tokens=t._last_global_tokens_by_hook,
            update_counts={h: u.update_count for h, u in t.units.items()},
        )
    )


def full_adam(unit):
    opt = unit.optimizer
    result = {}
    group = unit.parallel_context.require_local().dp_group
    for name, full in unit.model.named_parameters():
        if not getattr(opt, "_sae_distributed_optimizer", False):
            result[name] = copy.deepcopy(opt.state.get(full, {}))
            continue
        entry = opt.sae_shards.get(name)
        state = opt.state.get(entry[1], {}) if entry else {}
        present = torch.tensor(int(bool(state)), device=full.device)
        if group.size() > 1:
            dist.all_reduce(present, op=dist.ReduceOp.MAX, group=group)
        result[name] = {}
        if present.item():
            for k in ("exp_avg", "exp_avg_sq"):
                value = torch.zeros_like(full).flatten()
                if state:
                    value[entry[2] : entry[3]].copy_(state[k])
                if group.size() > 1:
                    dist.all_reduce(value, group=group)
                result[name][k] = value.reshape(full.shape)
            step = torch.zeros((), device=full.device)
            if state:
                step.fill_(float(state.get("step", opt.param_groups[0].get("step", 0))))
            if group.size() > 1:
                dist.all_reduce(step, op=dist.ReduceOp.MAX, group=group)
            result[name]["step"] = step
    return result


def worker(rank, directory, rendezvous):
    import faulthandler

    trace_stream = open(Path(directory, f"stack_rank{rank}.log"), "w")  # noqa: SIM115 -- remains open for faulthandler until worker exits
    faulthandler.dump_traceback_later(35, repeat=True, file=trace_stream)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=4,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=120),
    )
    records = []
    phase = os.environ.get("SAE_OVERLAP_PHASE", "stage1")
    try:
        for tp, dp, spp in [
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (1, 4, 1),
            (1, 2, 2),
            (2, 1, 2),
        ]:
            for nh in (1, 3, 4):
                if nh < spp:
                    continue
                hooks = tuple(f"h{i}" for i in range(nh))
                runtime = SAERuntime.from_layout(
                    tp_size=tp, dp_size=dp, placement_size=spp, hooks=hooks
                )
                try:
                    if runtime.local is None:
                        continue
                    cases = [(1, True)] if os.environ.get("SAE_OVERLAP_GA1_ONLY") == "1" else [(1, False), (1, True), (3, False), (3, True)]
                    for ga, amp in cases:
                        Path(directory, f"progress_rank{rank}.txt").write_text(
                            f"{tp=} {dp=} {spp=} {nh=} {ga=} {amp=}"
                        )
                        NUMERICAL_ERRORS.clear()
                        off = build(runtime, Path(directory), "off", ga, amp, fast_path=False, ga1_norm=False)
                        on = build(
                            runtime,
                            Path(directory),
                            "off" if phase == "stage2" else "on",
                            ga,
                            amp,
                            sharded=phase != "stage1",
                        )
                        assert on._runtime_optimizer_overlap == (
                            len(on.units) > 1 and phase != "stage2"
                        )
                        assert not off._runtime_mean_loss_fast_path
                        assert on._runtime_mean_loss_fast_path == (ga == 1 and not amp)
                        assert not off._runtime_fused_amp_normalization
                        assert on._runtime_fused_amp_normalization == (ga == 1 and amp)
                        for step in range(6):
                            batches = []
                            for micro in range(ga if step != 4 else 1):
                                batch = {}
                                for h in runtime.local.domain.hooks:
                                    n = (runtime.local.dp_rank + micro + step) % 5 + 1
                                    if (
                                        step == 5
                                        or (step == 1 and runtime.local.dp_rank == 0)
                                        or (step == 2 and h == hooks[-1])
                                    ):
                                        n = 0
                                    torch.manual_seed(
                                        1000
                                        + step * 30
                                        + micro * 5
                                        + runtime.local.dp_rank
                                    )
                                    batch[h] = torch.randn(
                                        n, 16, device=f"cuda:{rank}"
                                    ) * (20 if step % 2 else 0.05)
                                batches.append(batch)
                            for t in (off, on):
                                handle = None
                                if (
                                    amp
                                    and step == 3
                                    and runtime.local.tp_rank
                                    == runtime.local.dp_rank
                                    == 0
                                ):
                                    p = next(
                                        next(iter(t.units.values())).model.parameters()
                                    )
                                    handle = p.register_hook(
                                        lambda g: torch.full_like(
                                            g, float("nan") if nh == 4 else float("inf")
                                        )
                                    )
                                train_runtime_window(t, batches)
                                if handle is not None:
                                    handle.remove()
                                t.lr_scheduler.step(t._last_updated_hooks)
                                t.n_training_steps += 1
                            left, right = snapshot(off), snapshot(on)
                            failure = None
                            try:
                                equal(left, right)
                            except AssertionError:
                                import traceback

                                failure = traceback.format_exc()
                            failures = [None] * len(runtime.local.domain.ranks)
                            dist.all_gather_object(
                                failures, failure, group=runtime.local.groups.tp_dp_cp
                            )
                            if any(failures):
                                raise AssertionError(str(failures))
                            if (
                                os.environ.get("SAE_OVERLAP_RESUME") == "1"
                                and (ga == 3 or os.environ.get("SAE_OVERLAP_GA1_ONLY") == "1")
                                and amp
                                and step in (0, 2)
                            ):
                                # Ordinary -> sharded migration, then native shard resume.
                                source = off if step == 0 else on
                                name = f"{phase}_tp{tp}dp{dp}spp{spp}h{nh}_ga{ga}_step{step}"
                                source.save_checkpoint(name)
                                restored = build(
                                    runtime,
                                    Path(directory),
                                    "off" if phase == "stage2" else "on",
                                    ga,
                                    amp,
                                    sharded=phase != "stage1",
                                )
                                restored.load_trainer_state(Path(directory) / name)
                                for h in on.units:
                                    equal(
                                        on.units[h].model.state_dict(),
                                        restored.units[h].model.state_dict(),
                                    )
                                    equal(
                                        full_adam(on.units[h]),
                                        full_adam(restored.units[h]),
                                    )
                                equal(
                                    on.lr_scheduler.state_dict(),
                                    restored.lr_scheduler.state_dict(),
                                )
                                equal(
                                    on.grad_scaler.state_dict(),
                                    restored.grad_scaler.state_dict(),
                                )
                                on = restored
                        records.append(
                            dict(
                                tp=tp,
                                dp=dp,
                                spp=spp,
                                hooks=nh,
                                ga=ga,
                                amp=amp,
                                pre_backward_normalization=on._runtime_mean_loss_fast_path,
                                normalization_in_unscale=on._runtime_fused_amp_normalization,
                                passed=True,
                                max_absolute_error=dict(NUMERICAL_ERRORS),
                            )
                        )
                        del off, on
                finally:
                    import sys
                    import traceback

                    failure = traceback.format_exc() if sys.exc_info()[0] else None
                    if failure:
                        Path(directory, f"failure_rank{rank}.txt").write_text(failure)
                    failures = [None] * 4
                    dist.all_gather_object(
                        failures, failure, group=runtime.control_group
                    )
                    runtime.close()
                    if any(failures):
                        raise RuntimeError(str(failures))
    finally:
        Path(directory, f"{phase}_rank{rank}.json").write_text(
            json.dumps(records, indent=2)
        )
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_per_hook_overlap_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    directory = Path(os.environ.get("SAE_OVERLAP_REPORT_DIR", tmp_path)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ctx = mp.spawn(
        worker,
        args=(str(directory), str(directory / f"world_{time.time_ns()}")),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + 240
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("overlap matrix timeout")
    finally:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        for p in ctx.processes:
            p.join(timeout=5)


def empty_shard_worker(rank, directory, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=4, init_method=f"file://{rendezvous}"
    )
    runtime = SAERuntime.from_layout(tp_size=1, dp_size=4, hooks=("h0", "h1"))
    try:
        ordinary = build(runtime, Path(directory), "off", 1, True, tiny=True)
        sharded = build(
            runtime, Path(directory), "on", 1, True, sharded=True, tiny=True
        )
        empty = not sharded.units["h0"].optimizer.get_parameters()
        empty_count = torch.tensor(int(empty), device=f"cuda:{rank}")
        dist.all_reduce(empty_count, group=runtime.local.dp_group)
        assert empty_count.item() > 0
        for step in range(3):
            batch = {
                h: torch.ones(rank, 2, device=f"cuda:{rank}") * (step + 1)
                for h in ordinary.units
            }
            for t in (ordinary, sharded):
                handle = None
                if step == 1 and rank == 0:
                    handle = next(t.units["h0"].model.parameters()).register_hook(
                        lambda g: torch.full_like(g, float("inf"))
                    )
                train_runtime_window(t, [batch])
                if handle is not None:
                    handle.remove()
            equal(snapshot(ordinary), snapshot(sharded))
        unit = sharded.units["h0"]
        for buf in unit.ddp.buffers:
            buf.grad_data.fill_(float("nan"))
        unit.optimizer.expose_valid_grads()
        for p in unit.optimizer.get_parameters():
            p.grad.zero_()
        entry = unit.optimizer.sae_shards.get("b_dec")
        if entry is not None:
            full = torch.tensor([3.0, 4.0], device=f"cuda:{rank}")
            entry[1].grad.copy_(full[entry[2] : entry[3]])
        from tests.training.test_gpu_clip_optimizer import RejectScalarHostRead

        with RejectScalarHostRead():
            norm = unit.optimizer.clip_grad_norm(1.0)
        torch.testing.assert_close(norm, torch.tensor(5.0, device=f"cuda:{rank}"))
        Path(directory, f"empty_shard_rank{rank}.json").write_text(
            json.dumps(
                dict(
                    empty=empty,
                    norm=norm.item(),
                    padding_nan_ignored=True,
                    overflow_consistent=True,
                )
            )
        )
    finally:
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_empty_shards_padding_clip_and_amp(tmp_path, monkeypatch):
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    directory = Path(os.environ.get("SAE_OVERLAP_REPORT_DIR", tmp_path)).resolve()
    directory.mkdir(exist_ok=True, parents=True)
    ctx = mp.spawn(
        empty_shard_worker,
        args=(str(directory), str(directory / f"empty_world_{time.time_ns()}")),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + 100
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("empty shard acceptance timeout")
    finally:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        for p in ctx.processes:
            p.join(timeout=5)


def single_resume_worker(rank, directory, rendezvous):
    from sae_lens.training.sae_trainer import SAETrainer

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl", rank=rank, world_size=1, init_method=f"file://{rendezvous}"
    )
    runtime = SAERuntime.from_layout(tp_size=1, dp_size=1, hooks=("h0",))

    def create(sharded):
        cfg = _trainer_config("cuda:0", Path(directory), "legacy_per_hook_wrapper")
        cfg.ddp_zero_optimizer = sharded
        cfg.autocast = True
        torch.manual_seed(731)
        model = MegatronTopKSAE(
            TopKTrainingSAEConfig(d_in=16, d_sae=48, k=4, device="cuda:0"),
            runtime=runtime,
        )
        ddp = wrap_runtime_sae(
            model, runtime, distributed_optimizer=sharded,
            single_replica_fast_path=False,  # Explicit native-DP1 checkpoint control.
        )
        return SAETrainer(
            sae=ddp,
            base_sae=model,
            cfg=cfg,
            data_provider=MagicMock(),
            runtime=runtime,
            dp_group=runtime.local.dp_group,
            token_count_weighted_dp=True,
        )

    try:
        original = create(False)
        batch = [{"h0": torch.randn(5, 16, device="cuda:0")}]
        train_runtime_window(original, batch)
        original.save_checkpoint("single_ordinary")
        restored = create(True)
        path = Path(directory) / "single_ordinary"
        restored._base_sae.load_state_dict(
            restored._base_sae.load_saelens_checkpoint_shard(
                path / "sae_weights.safetensors"
            )
        )
        restored.load_trainer_state(path)
        equal(full_adam(original.unit), full_adam(restored.unit))
        for t in (original, restored):
            train_runtime_window(t, batch)
        equal(original._base_sae.state_dict(), restored._base_sae.state_dict())
        equal(full_adam(original.unit), full_adam(restored.unit))
        restored.save_checkpoint("single_distributed")
        final = create(True)
        path = Path(directory) / "single_distributed"
        final._base_sae.load_state_dict(
            final._base_sae.load_saelens_checkpoint_shard(
                path / "sae_weights.safetensors"
            )
        )
        final.load_trainer_state(path)
        for t in (restored, final):
            train_runtime_window(t, batch)
        equal(restored._base_sae.state_dict(), final._base_sae.state_dict())
        equal(full_adam(restored.unit), full_adam(final.unit))
        shard = path / "distributed_adam_rank0.pt"
        shard.rename(path / "withheld.pt")
        with pytest.raises(ValueError, match="Incomplete distributed Adam checkpoint"):
            final.load_trainer_state(path)
        (path / "withheld.pt").rename(shard)
    finally:
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_single_distributed_resume(tmp_path):
    mp.spawn(
        single_resume_worker,
        args=(str(tmp_path), str(tmp_path / "world")),
        nprocs=1,
        join=True,
    )
