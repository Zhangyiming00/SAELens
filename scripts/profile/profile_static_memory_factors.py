"""Controlled static Megatron SAE memory experiments (no elastic or eager wavefront).

Each case gets fresh processes, fixed FP32/k/AuxK/data, unprofiled timing, then
one allocator-history step. Inputs are real cached Llama activations. Global
tokens per optimizer update are invariant under DP/GA changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "results_0/nsys_h3_ga_sweep_20260918/inputs"
HOOKS = [f"blocks.{i}.hook_resid_post" for i in (16, 21, 26)]
BASE = dict(tp=2, dp=1, pp=1, h=3, batch=8192, ga=1, d_in=4096,
            d_sae=16384, backend="sharded_dense", wave="bounded", live=2,
            zero=False, overlap="on", aux="auto", dead=2048,
            main_compute="inherit", aux_compute="inherit")


def cases():
    variants = {
        "base": {}, "wave_off": dict(wave="off"), "wave_lazy": dict(wave="lazy"),
        "wave_bound1": dict(live=1), "full_dense": dict(backend="legacy"),
        "full_dense_off": dict(backend="legacy", wave="off"),
        "full_dense_lazy": dict(backend="legacy", wave="lazy"),
        "aux_local_dense": dict(aux="local_dense"),
        "tp1": dict(tp=1, wave="off"), "tp4": dict(tp=4),
        "dp2": dict(tp=1, dp=2, wave="off"), "dp4": dict(tp=1, dp=4, wave="off"),
        "tp2dp2": dict(dp=2), "dp2_zero": dict(tp=1, dp=2, wave="off", zero=True),
        "dp4_zero": dict(tp=1, dp=4, wave="off", zero=True),
        "tp2dp2_zero": dict(dp=2, zero=True),
        "pp2": dict(pp=2), "pp3_tp1": dict(tp=1, pp=3, wave="off"),
        "h1": dict(h=1, wave="off"), "h2": dict(h=2),
        "width8192": dict(d_sae=8192), "width32768": dict(d_sae=32768),
        "batch4096": dict(batch=4096), "batch16384": dict(batch=16384),
        "ga2": dict(ga=2), "ga4": dict(ga=4), "overlap_off": dict(overlap="off"),
        # Optional: selected-entry storage, never part of the dense-storage suite.
        "ragged_sparse": dict(backend="sharded_ragged", main_compute="sparse", aux_compute="sparse"),
        "ragged_compact": dict(backend="sharded_ragged", main_compute="sparse", aux_compute="compact_dense"),
        "ragged_local": dict(backend="sharded_ragged", main_compute="local_dense", aux_compute="local_dense"),
        "ragged_main_compact": dict(backend="sharded_ragged", main_compute="compact_dense", aux_compute="compact_dense"),
        # Held-out combinations for allocated prediction; excluded from default runs.
        "verify_control": {},
        "verify_dense_off": dict(d_sae=12288, batch=6144, wave="off"),
        "verify_dense_bound": dict(d_sae=12288, batch=6144),
        "verify_dense_lazy": dict(d_sae=12288, batch=6144, wave="lazy"),
        "verify_full_off": dict(d_sae=12288, batch=6144, backend="legacy", wave="off"),
        "verify_full_bound": dict(d_sae=12288, batch=6144, backend="legacy"),
        "verify_full_lazy": dict(d_sae=12288, batch=6144, backend="legacy", wave="lazy"),
        "verify_zero": dict(d_sae=24576, batch=12288, dp=2, zero=True),
        "verify_dp3_zero": dict(d_sae=12288, batch=12288, tp=1, dp=3, zero=True, wave="off"),
        "verify_padding_zero": dict(d_sae=16390, batch=6144, dp=2, zero=True),
        "verify_pp": dict(d_sae=12288, batch=6144, pp=2),
        "verify_ga3": dict(d_sae=24576, batch=12288, ga=3),
        "verify_tp4": dict(d_sae=24576, batch=12288, tp=4),
        "verify_ragged_sparse": dict(d_sae=12288, batch=6144, backend="sharded_ragged", main_compute="sparse", aux_compute="sparse"),
        "verify_ragged_compact": dict(d_sae=12288, batch=6144, backend="sharded_ragged", main_compute="sparse", aux_compute="compact_dense"),
        "verify_ragged_main_compact": dict(d_sae=12288, batch=6144, backend="sharded_ragged", main_compute="compact_dense", aux_compute="compact_dense"),
        "verify_ragged_local": dict(d_sae=12288, batch=6144, backend="sharded_ragged", main_compute="local_dense", aux_compute="local_dense"),
    }
    return {name: dict(BASE, **change) for name, change in variants.items()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def storage_inventory(trainer, batches):
    import torch
    found = {}

    def visit(value, category, name):
        if torch.is_tensor(value) and value.is_cuda:
            storage = value.untyped_storage()
            addr = storage.data_ptr()
            item = found.setdefault(addr, dict(address=addr, bytes=storage.nbytes(), category=category, names=[]))
            item["names"].append(name)
        elif isinstance(value, dict):
            for i, (k, v) in enumerate(value.items()):
                visit(v, category, f"{name}.{k if isinstance(k, (str, int)) else i}")
        elif isinstance(value, (tuple, list)):
            for k, v in enumerate(value):
                visit(v, category, f"{name}.{k}")

    for hook, unit in trainer.units.items():
        for n, p in unit.model.named_parameters():
            visit(p, "parameters", f"{hook}.{n}")
        buffers = unit.ddp.buffers if getattr(unit.ddp, "_sae_megatron_ddp", False) else []
        for i, b in enumerate(buffers):
            visit(b.grad_data, "gradients", f"{hook}.grad_buffer{i}")
            visit(getattr(b, "param_data", None), "parameters", f"{hook}.param_buffer{i}")
        for n, p in unit.model.named_parameters():
            visit(p.grad, "gradients", f"{hook}.{n}.grad")
        opt = unit.optimizer
        visit(opt.optimizer.state, "adam_state", hook)
        for n in ("shard_fp32_from_float32_groups", "shard_fp32_from_float16_groups", "fp32_from_float16_groups"):
            visit(getattr(opt, n, None), "optimizer_parameter_shards", f"{hook}.{n}")
    visit(batches, "inputs", "batches")
    for n in ("act_freq_scores_by_hook", "n_forward_passes_since_fired_by_hook"):
        visit(getattr(trainer, n), "feature_statistics", n)
    return list(found.values())


def worker(rank, args, c):
    import math
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    sys.path.insert(0, str(REPO))
    from sae_lens.config import SAETrainerConfig, LoggingConfig
    from sae_lens.sae_runtime import SAERuntime
    from sae_lens.static_failure import StaticFailureMonitor
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
    from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
    from sae_lens.training.megatron_ddp import wrap_runtime_sae
    from sae_lens.training.multi_sae_trainer import MultiSAETrainer
    from sae_lens.training.gradient_window import train_runtime_window

    directory = args.output / args.case
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=c["tp"] * c["dp"] * c["pp"],
                            init_method=f"file://{directory}/world", timeout=timedelta(seconds=180))
    runtime = SAERuntime.from_layout(tp_size=c["tp"], dp_size=c["dp"], placement_size=c["pp"], hooks=tuple(HOOKS[:c["h"]]))
    monitor = StaticFailureMonitor(runtime, set(runtime._owned_groups))
    runtime.failure_monitor = monitor
    ctx = runtime.require_local()
    hooks = list(ctx.domain.hooks)
    local_total = c["batch"] // c["dp"]
    micro = local_total // c["ga"]
    assert micro * c["ga"] * c["dp"] == c["batch"]
    try:
        cfg = SAETrainerConfig(device=f"cuda:{rank}", n_checkpoints=0,
            total_training_samples=(args.warmup + args.steps + 1) * local_total,
            train_batch_size_samples=micro, output_path=None,
            save_mse_every_n_steps=0, save_timing_every_n_steps=0, save_memory_every_n_steps=0,
            record_memory_empty_cache=False, record_memory_timeline_step=-1, synchronize_timing=False,
            lr=3e-4, lr_end=3e-4, lr_scheduler_name="constant", lr_warm_up_steps=0, lr_decay_steps=0,
            dead_feature_window=0, feature_sampling_window=1000, autocast=False,
            quiesce_checkpoint_path=None, multi_sae_backward_order="forward",
            multi_sae_stats_sync_mode="immediate", multi_sae_stats_sync_interval=1,
            adam_beta1=.9, adam_beta2=.999, n_restart_cycles=1,
            checkpoint_path=str(directory / "unused"), save_final_checkpoint=False,
            logger=LoggingConfig(log_to_wandb=False))
        cfg.gradient_accumulation_steps = c["ga"]
        cfg.ddp_zero_optimizer = c["zero"]
        cfg.multi_sae_distributed_architecture = "legacy_per_hook_wrapper" if c["wave"] == "off" else "unified_multi_hook"
        cfg.multi_sae_tp_wavefront_schedule = "bounded" if c["wave"] == "off" else c["wave"]
        cfg.multi_sae_tp_wavefront_max_live_hooks = c["live"]
        cfg.sae_runtime_output_retention = "summary"
        cfg.multi_sae_optimizer_overlap = c["overlap"]
        cfg.multi_sae_param_gather_overlap = True
        cfg.multi_sae_param_gather_schedule = "one_hook_lag"
        cfg.sae_single_replica_fast_path = True
        cfg.sae_gradient_accumulation_fusion = True
        cfg.sae_ga1_loss_normalization = True
        cfg.multi_sae_tp_phase_fence = "auto"
        model_cfg = TopKTrainingSAEConfig(d_in=c["d_in"], d_sae=c["d_sae"], k=128,
            device=f"cuda:{rank}", dtype="float32", normalize_activations="none",
            use_sparse_activations=False, topk_backend=c["backend"], auxk_decoder_backend=c["aux"],
            ragged_decoder_engine="openai", v5_main_compute=c["main_compute"], v5_aux_compute=c["aux_compute"])
        models, wrapped = {}, {}
        for hook in hooks:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = MegatronTopKSAE(model_cfg, runtime=runtime)
            models[hook] = model
            wrapped[hook] = wrap_runtime_sae(model, runtime, distributed_optimizer=c["zero"],
                single_replica_fast_path=True, gradient_accumulation_fusion=True)
        trainer = MultiSAETrainer(hook_names=hooks, sae_by_hook=wrapped, base_sae_by_hook=models,
            data_provider=iter(()), save_checkpoint_fn=None, cfg=cfg, dp_group=ctx.dp_group,
            token_count_weighted_dp=True, sae_dp_mode="ddp", runtime=runtime)
        assert trainer.global_update_batch_size == c["batch"]
        assert trainer._runtime_tp_wavefront == (c["wave"] != "off" and c["tp"] > 1 and len(hooks) > 1)
        order = torch.load(CACHE / "global_row_order.pt", weights_only=True)[:c["batch"]]
        assert len(order) == c["batch"]
        order = order[ctx.dp_rank * local_total:(ctx.dp_rank + 1) * local_total]
        batches = [{} for _ in range(c["ga"])]
        for hook in hooks:
            cached = torch.load(CACHE / f"{hook}.pt", map_location="cpu", weights_only=True, mmap=True)
            for i, batch in enumerate(batches):
                batch[hook] = cached.index_select(0, order[i * micro:(i + 1) * micro]).float().to(rank)
            del cached
        # Balanced nested dead set identical across TP layouts; no changing AuxK workload.
        mask = torch.zeros(c["d_sae"], dtype=torch.bool, device=rank)
        # Equals the old period mask for divisible widths; also tests non-aligned
        # widths without changing the global dead-feature count.
        dead_ids = torch.arange(c["dead"], device=rank) * c["d_sae"] // c["dead"]
        mask[dead_ids] = True
        del dead_ids
        assert int(mask.sum()) == c["dead"]
        steps = []
        initial = None
        inventory = None
        timing_memory = None
        for step in range(args.warmup + args.steps + 1):
            for ages in trainer.n_forward_passes_since_fired_by_hook.values():
                ages.copy_(mask)
            torch.cuda.synchronize()
            if step == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            if step == args.warmup + args.steps:
                timing_memory = memory()
                inventory = storage_inventory(trainer, batches)
                torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=200000)
                initial = torch.cuda.memory._snapshot()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            outputs, _ = train_runtime_window(trainer, batches)
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - start)
            losses = {h: float(o.loss) for h, o in outputs.items()}
            assert all(math.isfinite(v) for v in losses.values())
            assert all(float(o.losses["auxiliary_reconstruction_loss"]) > 0 for o in outputs.values())
            assert all(u.update_count == step + 1 for u in trainer.units.values())
            assert all(v == c["batch"] for v in trainer._last_global_tokens_by_hook.values())
            del outputs
            if args.warmup <= step < args.warmup + args.steps:
                steps.append(dict(step=step, ms=elapsed, losses=losses, **memory()))
            trainer.lr_scheduler.step(trainer._last_updated_hooks)
            trainer.n_training_steps += 1
            trainer.n_training_samples += local_total
        torch.cuda.synchronize()
        trace_memory = memory()
        snapshot = torch.cuda.memory._snapshot()
        assert len(snapshot["device_traces"][rank]) < 200000
        with (directory / f"rank{rank}.pickle").open("wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        start_state = dict(segments=initial["segments"], trace_index=len(initial["device_traces"][rank]))
        dump(directory / f"start_rank{rank}.json", start_state)
        dump(directory / f"rank{rank}.json", dict(rank=rank, config=c, local_hooks=hooks,
            tp_rank=ctx.tp_rank, dp_rank=ctx.dp_rank,
            native_buffer_layouts=[
                dict(numel=b.numel, buckets=b.bucket_indices)
                for u in trainer.units.values()
                for b in (u.ddp.buffers if getattr(u.ddp, "_sae_megatron_ddp", False) else [])
            ],
            model_config=model_cfg.to_dict(), microbatch=micro,
            effective_wavefront=trainer._runtime_tp_wavefront,
            effective_overlap=trainer._runtime_optimizer_overlap,
            ddp=[type(u.ddp).__name__ for u in trainer.units.values()],
            optimizers=[type(u.optimizer).__name__ for u in trainer.units.values()],
            adam_implementations=[type(u.optimizer.optimizer).__module__ + "." + type(u.optimizer.optimizer).__name__ for u in trainer.units.values()],
            steps=steps, memory=timing_memory, trace_memory=trace_memory, storages=inventory,
            main_execution=[getattr(u.model, "_last_main_execution", None) for u in trainer.units.values()],
            aux_execution=[getattr(u.model, "_last_auxk_execution", None) for u in trainer.units.values()],
            torch=torch.__version__, cuda=torch.version.cuda))
    except BaseException as exc:
        monitor.fail(exc)
        raise
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        monitor.close()
        runtime.close()
        dist.destroy_process_group()


def memory():
    import torch
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        active=stats["active_bytes.all.current"], inactive_split=stats["inactive_split_bytes.all.current"],
        device_used=total-free)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "results/static_memory_factors_20260927")
    p.add_argument("--cases", nargs="+", default=[k for k in cases() if not k.startswith(("ragged", "verify_"))], choices=list(cases()))
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--case", choices=list(cases()))
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        import torch.multiprocessing as mp
        c = cases()[args.case]
        mp.spawn(worker, args=(args, c), nprocs=c["tp"] * c["dp"] * c["pp"], join=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "cases.json", cases())
    sources = {str(f.relative_to(REPO)): sha(f) for f in (REPO / "sae_lens").rglob("*.py")}
    sources[str(Path(__file__).relative_to(REPO))] = sha(__file__)
    source_file = args.output / "source_sha256.json"
    if source_file.exists():
        previous = json.loads(source_file.read_text())
        assert all(sources[k] == v for k, v in previous.items() if k.startswith("sae_lens/")), "Production source changed"
    else:
        dump(source_file, sources)
        dump(args.output / "inputs_sha256.json", {str(f): sha(f) for f in [CACHE / "global_row_order.pt", *[CACHE / f"{h}.pt" for h in HOOKS]]})
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", NCCL_LAUNCH_ORDER_IMPLICIT="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for name in args.cases:
        directory = args.output / name
        if (directory / "result.json").exists():
            if json.loads((directory / "result.json").read_text())["returncode"] == 0:
                continue
            directory.rename(args.output / f"{name}_failed_{time.time_ns()}")
        directory.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", name,
                   "--output", str(args.output), "--warmup", str(args.warmup), "--steps", str(args.steps)]
        dump(directory / "command.json", dict(argv=command, harness_sha256=sha(__file__),
             environment={k: env[k] for k in ("OMP_NUM_THREADS", "NCCL_LAUNCH_ORDER_IMPLICIT", "PYTORCH_CUDA_ALLOC_CONF")}))
        print("START", name, flush=True)
        begin = time.time()
        with (directory / "run.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = 124
        dump(directory / "result.json", dict(returncode=rc, elapsed_s=time.time()-begin))
        print("END", name, rc, flush=True)
        if rc:
            raise RuntimeError(f"Case failed: {directory / 'run.log'}")


if __name__ == "__main__":
    main()
