"""Offline current-Megatron memory matrix; never writes legacy memory results.

Run from the repository: .venv/bin/python scripts/profile_megatron_sae_memory.py
Uses cached real activations, current production runtime, and one pickle per rank.
GA means local accumulation steps; effective global batch is held constant.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
LAYOUTS = {"single": (1, 1), "tp2": (2, 1), "tp4": (4, 1),
           "dp2": (1, 2), "dp4": (1, 4), "tp2dp2": (2, 2)}
HOOKS = [f"blocks.{i}.hook_resid_post" for i in (16, 21, 26)]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(trainer, batches):
    """Unique CUDA storages, including aliasing main_grad views only once."""
    import torch
    entries = {}

    def add(tensor, category, name):
        if not torch.is_tensor(tensor) or not tensor.is_cuda:
            return
        storage = tensor.untyped_storage()
        address = storage.data_ptr()
        if address not in entries:
            entries[address] = dict(address=address, bytes=storage.nbytes(), category=category, names=[])
        entries[address]["names"].append(name)

    for hook, unit in trainer.units.items():
        for name, param in unit.model.named_parameters():
            add(param, "parameters", f"{hook}.{name}")
        buffers = unit.ddp.buffers if getattr(unit.ddp, "_sae_megatron_ddp", False) else []
        for index, buf in enumerate(buffers):
            add(buf.grad_data, "gradients", f"{hook}.main_grad_buffer{index}")
        for name, param in unit.model.named_parameters():
            add(param.grad, "gradients", f"{hook}.{name}.grad")
        for index, state in enumerate(unit.optimizer.optimizer.state.values()):
            for name, value in state.items():
                add(value, "adam_moments" if name in ("exp_avg", "exp_avg_sq") else "adam_scalars",
                    f"{hook}.adam{index}.{name}")
    for index, batch in enumerate(batches):
        for hook, value in batch.items():
            add(value, "cached_inputs", f"microbatch{index}.{hook}")
    for name in ("act_freq_scores_by_hook", "n_forward_passes_since_fired_by_hook"):
        for hook, value in getattr(trainer, name).items():
            add(value, "feature_statistics", f"{hook}.{name}")
    return list(entries.values())


def worker(rank, args):
    import math
    from datetime import timedelta
    import torch
    import torch.distributed as dist

    source = args.output / "sources"
    sys.path.insert(0, str(source))
    tp, dp = LAYOUTS[args.layout]
    ga = args.ga
    microbatch = args.batch // (dp * ga)
    directory = args.output / "runs" / f"{args.layout}_ga{ga}"
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", rank=rank, world_size=tp * dp,
                            init_method=f"file://{directory}/world",
                            timeout=timedelta(seconds=180))
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=200000)
    import sae_lens
    assert Path(sae_lens.__file__).resolve().parent == source / "sae_lens"
    from sae_lens.config import SAETrainerConfig, LoggingConfig
    from sae_lens.sae_runtime import SAERuntime
    from sae_lens.static_failure import StaticFailureMonitor
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
    from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
    from sae_lens.training.megatron_ddp import wrap_runtime_sae, is_megatron_ddp
    from sae_lens.training.multi_sae_trainer import MultiSAETrainer
    from sae_lens.training.gradient_window import train_runtime_window

    runtime = SAERuntime.from_layout(tp_size=tp, dp_size=dp, placement_size=1, hooks=tuple(HOOKS))
    monitor = StaticFailureMonitor(runtime, set(runtime._owned_groups))
    runtime.failure_monitor = monitor
    ctx = runtime.require_local()
    total = args.warmup + args.steps
    sampler_stop = threading.Event()
    samples = []
    sampler = None
    try:
        cfg = SAETrainerConfig(
            device=f"cuda:{rank}", n_checkpoints=0,
            total_training_samples=total * microbatch * ga, train_batch_size_samples=microbatch,
            output_path=None, save_mse_every_n_steps=0, save_timing_every_n_steps=0,
            save_memory_every_n_steps=0, record_memory_empty_cache=False,
            record_memory_timeline_step=-1, synchronize_timing=False,
            multi_sae_backward_order="forward", multi_sae_stats_sync_mode="immediate",
            multi_sae_stats_sync_interval=1, lr=3e-4, lr_end=3e-5,
            lr_scheduler_name="constant", lr_warm_up_steps=0, lr_decay_steps=0,
            n_restart_cycles=1, adam_beta1=.9, adam_beta2=.999, dead_feature_window=1000,
            feature_sampling_window=1000, autocast=False, checkpoint_path=str(directory / "unused"),
            quiesce_checkpoint_path=None, save_final_checkpoint=False,
            logger=LoggingConfig(log_to_wandb=False))
        cfg.gradient_accumulation_steps = ga
        cfg.ddp_zero_optimizer = False
        cfg.multi_sae_distributed_architecture = "unified_multi_hook"
        cfg.multi_sae_optimizer_overlap = "on"
        cfg.multi_sae_param_gather_overlap = True
        cfg.sae_single_replica_fast_path = True
        cfg.sae_gradient_accumulation_fusion = True
        cfg.sae_ga1_loss_normalization = True
        cfg.multi_sae_tp_phase_fence = "auto"
        model_cfg = TopKTrainingSAEConfig(d_in=args.d_in, d_sae=args.d_sae, k=args.k,
            device=f"cuda:{rank}", dtype="float32", use_sparse_activations=False,
            normalize_activations="none")
        models, wrapped = {}, {}
        for hook in HOOKS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = MegatronTopKSAE(model_cfg, runtime=runtime)
            models[hook] = model
            wrapped[hook] = wrap_runtime_sae(model, runtime, distributed_optimizer=False,
                single_replica_fast_path=True, gradient_accumulation_fusion=True)
        trainer = MultiSAETrainer(hook_names=HOOKS, sae_by_hook=wrapped, base_sae_by_hook=models,
            data_provider=iter(()), save_checkpoint_fn=None, cfg=cfg, dp_group=ctx.dp_group,
            token_count_weighted_dp=True, sae_dp_mode="ddp", runtime=runtime)
        assert trainer.global_update_batch_size == args.batch
        assert trainer._runtime_tp_wavefront == (tp > 1)
        assert trainer._runtime_optimizer_overlap
        assert trainer._runtime_mean_loss_fast_path == (ga == 1)
        calls, marks = {}, []
        state = {"step": 0}

        def wrap(obj, name, label, verify=None):
            original = getattr(obj, name)
            @functools.wraps(original)
            def measured(*a, **kw):
                calls[label] = calls.get(label, 0) + 1
                result = original(*a, **kw)
                if verify:
                    verify()
                if state["step"] > args.warmup:
                    marks.append(dict(step=state["step"], phase=label,
                        allocated=torch.cuda.memory_allocated(rank),
                        active=torch.cuda.memory_stats(rank)["active_bytes.all.current"],
                        running_peak=torch.cuda.max_memory_allocated(rank)))
                return result
            setattr(obj, name, measured)

        for hook, unit in trainer.units.items():
            direct = not is_megatron_ddp(unit.ddp)
            assert direct == (dp == 1)
            assert unit.model.gradient_accumulation_fusion == (dp > 1)
            assert type(unit.optimizer).__name__ == "GPUClipFP32Optimizer"
            assert all(g["fused"] for g in unit.optimizer.optimizer.param_groups)
            if dp > 1:
                assert unit.early_grad_sync
            def fused_check(u=unit):
                if dp > 1:
                    assert u.model.encoder.weight.grad_added_to_main_grad
                    assert u.model.decoder.weight.grad_added_to_main_grad
            wrap(unit, "forward", f"{hook}:forward")
            wrap(unit, "backward", f"{hook}:backward", fused_check)
            wrap(unit, "finish_grad_sync", f"{hook}:grad_sync")
            wrap(unit.optimizer.optimizer, "step", f"{hook}:adam")
            for name in ("tp_wavefront_encode_launch", "tp_wavefront_decode_launch", "tp_wavefront_finish"):
                wrap(unit.model, name, f"{hook}:{name}")

        order = torch.load(args.cache / "global_row_order.pt", weights_only=True)[:args.batch]
        assert len(order) == args.batch
        # Contiguous global partition: same global token multiset in every topology/GA.
        local_order = order[ctx.dp_rank * args.batch // dp:(ctx.dp_rank + 1) * args.batch // dp]
        batches = [{} for _ in range(ga)]
        for hook in HOOKS:
            cached = torch.load(args.cache / f"{hook}.pt", map_location="cpu", weights_only=True, mmap=True)
            assert cached.shape[1] == args.d_in
            for index, batch in enumerate(batches):
                rows = local_order[index * microbatch:(index + 1) * microbatch]
                batch[hook] = cached.index_select(0, rows).to(f"cuda:{rank}")
            del cached
        torch.cuda.synchronize()
        dist.barrier(group=runtime.control_group)
        steps, initial, storages = [], None, None

        def sample_device():
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
            while not sampler_stop.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                samples.append(dict(time=time.time(), used=info.used))
                sampler_stop.wait(.01)

        for step in range(1, total + 1):
            state["step"] = step
            if step == args.warmup + 1:
                storages = inventory(trainer, batches)
                initial = torch.cuda.memory._snapshot()
                torch.cuda.reset_peak_memory_stats(rank)
                sampler = threading.Thread(target=sample_device, daemon=True)
                sampler.start()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            outputs, _ = train_runtime_window(trainer, batches)
            end.record()
            trainer.lr_scheduler.step(trainer._last_updated_hooks)
            trainer.n_training_steps += 1
            trainer.n_training_samples += microbatch * ga
            torch.cuda.synchronize()
            losses = {h: float(o.loss) for h, o in outputs.items()}
            assert all(math.isfinite(v) for v in losses.values())
            assert all(float(o.losses["auxiliary_reconstruction_loss"]) == 0 for o in outputs.values())
            assert trainer._last_window_microbatches == ga
            assert all(n == args.batch for n in trainer._last_global_tokens_by_hook.values())
            assert all(u.update_count == step for u in trainer.units.values())
            del outputs
            steps.append(dict(step=step, measured=step > args.warmup, ms=start.elapsed_time(end), losses=losses,
                allocated=torch.cuda.memory_allocated(rank), reserved=torch.cuda.memory_reserved(rank),
                peak_allocated=torch.cuda.max_memory_allocated(rank), peak_reserved=torch.cuda.max_memory_reserved(rank)))
        torch.cuda.synchronize()
        sampler_stop.set()
        sampler.join(timeout=5)
        assert samples and not sampler.is_alive()
        snapshot = torch.cuda.memory._snapshot()
        assert len(snapshot["device_traces"][rank]) < 200000, "history truncated"
        with (directory / f"memory_timeline_rank{rank}.pickle").open("wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Compact start state for exact replay of the measured event interval.
        dump(directory / f"start_state_rank{rank}.json", dict(
            segments=initial["segments"], trace_index=len(initial["device_traces"][rank])))
        for hook, unit in trainer.units.items():
            assert calls[f"{hook}:backward"] == total * ga
            assert calls[f"{hook}:adam"] == total
            assert calls.get(f"{hook}:tp_wavefront_encode_launch", 0) == (total * ga if tp > 1 else 0)
            assert {int(s["step"].item()) for s in unit.optimizer.optimizer.state.values()} == {total}
        dump(directory / f"rank{rank}.json", dict(rank=rank, tp=tp, dp=dp, ga=ga,
            local_microbatch=microbatch, global_update_batch=args.batch,
            source=str(sae_lens.__file__), torch_version=torch.__version__, cuda_version=torch.version.cuda,
            flags=dict(wavefront=trainer._runtime_tp_wavefront, optimizer_overlap=trainer._runtime_optimizer_overlap,
                gradient_accumulation_fusion=dp > 1, single_replica_fast_path=dp == 1,
                ga1_mean_loss=trainer._runtime_mean_loss_fast_path, fused_adam=True,
                param_gather_overlap_requested=True, sharded_optimizer=False, autocast=False, tf32=False),
            config=model_cfg.to_dict(), calls=calls, steps=steps, storages=storages,
            memory=dict(peak_allocated=torch.cuda.max_memory_allocated(rank),
                peak_reserved=torch.cuda.max_memory_reserved(rank), end_allocated=torch.cuda.memory_allocated(rank),
                sampled_device_peak=max(x["used"] for x in samples)),
            auxiliary_loss_active=False, vllm_started=False))
        dump(directory / f"host_phase_markers_rank{rank}.json", marks)
        dump(directory / f"device_samples_rank{rank}.json", samples)
        if rank == 0:
            print(args.layout, "GA", ga, "PASS", flush=True)
    except BaseException as exc:
        monitor.fail(exc)
        raise
    finally:
        sampler_stop.set()
        if sampler is not None:
            sampler.join(timeout=5)
        torch.cuda.memory._record_memory_history(enabled=None)
        monitor.close()
        runtime.close()
        dist.destroy_process_group()


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "source_sha256.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert all(sha(args.output / "sources" / p) == digest for p, digest in manifest.items())
        for filename, command in (("gpu_environment.txt", ["nvidia-smi"]), ("gpu_topology.txt", ["nvidia-smi", "topo", "-m"])):
            if not (args.output / filename).exists():
                (args.output / filename).write_text(subprocess.check_output(command, text=True))
        return
    sources = args.output / "sources"
    shutil.copytree(REPO / "sae_lens", sources / "sae_lens", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {str(p.relative_to(sources)): sha(p) for p in sorted((sources / "sae_lens").rglob("*.py"))}
    dump(manifest_path, manifest)
    legacy = [REPO / "scripts/predict_sae_memory.py", REPO / "scripts/profile_sae_phase_v5.py",
              REPO / "sae_lens/autoconfig/phase_memory_model.py"]
    dump(args.output / "legacy_sha256_before.json", {str(p.relative_to(REPO)): sha(p) for p in legacy})
    dump(args.output / "input_sha256.json", {str(p): sha(p) for p in
        [args.cache / "global_row_order.pt", *[args.cache / f"{h}.pt" for h in HOOKS]]})
    (args.output / "runs").mkdir()
    (args.output / "git_revision.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True))
    (args.output / "git_status_before.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True))
    (args.output / "gpu_environment.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    (args.output / "gpu_topology.txt").write_text(subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "results/memory_model_megatron_20260922")
    p.add_argument("--cache", type=Path, default=REPO / "results/nsys_h3_ga_sweep_20260918/inputs")
    p.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    p.add_argument("--gas", nargs="+", type=int, default=[1, 4])
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--d-in", type=int, default=4096)
    p.add_argument("--d-sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--layout", choices=LAYOUTS)
    p.add_argument("--ga", type=int)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    if args.warmup < 1 or args.steps < 1 or any(ga < 1 for ga in args.gas):
        p.error("warmup, steps and GA must be positive")
    if args.worker:
        import torch.multiprocessing as mp
        tp, dp = LAYOUTS[args.layout]
        mp.spawn(worker, args=(args,), nprocs=tp * dp, join=True)
        return
    for layout in args.layouts:
        for ga in args.gas:
            if args.batch % (LAYOUTS[layout][1] * ga):
                p.error("batch must be divisible by DP * GA")
    prepare(args)
    for ga in args.gas:
        for layout in args.layouts:
            directory = args.output / "runs" / f"{layout}_ga{ga}"
            if directory.with_suffix(".result.json").exists():
                if json.loads(directory.with_suffix(".result.json").read_text())["returncode"] == 0:
                    continue
                failed = args.output / "failed_attempts" / f"{directory.name}_{time.time_ns()}"
                failed.mkdir(parents=True)
                for path in (directory, directory.with_suffix(".result.json"), directory.with_suffix(".log")):
                    shutil.move(str(path), str(failed / path.name))
            directory.mkdir(exist_ok=False)
            tp, dp = LAYOUTS[layout]
            overrides = dict(CUDA_VISIBLE_DEVICES=",".join(map(str, range(tp * dp))), OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", SAE_ADAM_IMPL="fused", NCCL_LAUNCH_ORDER_IMPLICIT="1",
                SAE_TP_PHASE_FENCE="auto", WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false",
                PYTORCH_ALLOC_CONF="expandable_segments:True")
            env = {**os.environ, **overrides}
            for key in ("CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
                env.pop(key, None)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(args.output),
                "--cache", str(args.cache), "--layout", layout, "--ga", str(ga)]
            for name in ("batch", "d_in", "d_sae", "k", "warmup", "steps"):
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            dump(directory / "command.json", dict(argv=command, environment=overrides, harness_sha256=sha(__file__)))
            (directory / "command.txt").write_text(shlex.join(["env", *[k + "=" + v for k, v in overrides.items()], *command]) + "\n")
            print("START", layout, "GA", ga, flush=True)
            started = time.time()
            with directory.with_suffix(".log").open("x") as log:
                proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = proc.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    rc = 124
            dump(directory.with_suffix(".result.json"), dict(returncode=rc, elapsed_s=time.time() - started))
            print("END", layout, "GA", ga, "returncode", rc, flush=True)
            if rc:
                raise RuntimeError(f"See {directory.with_suffix('.log')}")


if __name__ == "__main__":
    main()
