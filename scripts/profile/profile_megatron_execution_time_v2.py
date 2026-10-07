"""Current native Megatron timing: unprofiled wall time or a separate Nsight run.

Warmup, allocator history and checkpoint/teardown are excluded. Each case gets
fresh ranks. Fixed masks support controlled AuxK audits; natural mode preserves
the trainer's firing ages and applies the requested dead_feature_window.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_static_memory_factors import CACHE, HOOKS, cases as memory_cases, dump, sha


def hook_sources(count, *, repeat=False):
    """Independent SAE owners; repeated cached inputs are explicitly opt-in."""
    if type(count) is not int or count < 1 or (count > len(HOOKS) and not repeat):
        raise ValueError("Invalid hook count; >3 requires --repeat-cached-hooks")
    return {HOOKS[i] if i < len(HOOKS) else f"replica_{i}.{HOOKS[i % len(HOOKS)]}":
            HOOKS[i % len(HOOKS)] for i in range(count)}


def read_local_dead_mask(directory, tp, rank, d_sae):
    """Read the actual owner mask; counts are derived, never used as a proxy."""
    import torch
    from sae_lens.tp_layout import balanced_widths

    path = Path(directory) / f"tp{tp}_rank{rank}.pt"
    mask = torch.load(path, map_location="cpu", weights_only=True)
    width = balanced_widths(d_sae, tp)[rank]
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != (width,)):
        raise ValueError(f"{path}: expected bool [{width}] local dead mask")
    return mask, path


def original_cases():
    result = memory_cases()
    for name in ("base", "full_dense", "tp2dp2", "tp2dp2_zero", "dp2_zero",
                 "tp4", "ragged_compact", "tp1", "dp2", "dp4", "dp4_zero",
                 "ragged_sparse", "ragged_local", "ragged_main_compact", "aux_local_dense",
                 "batch4096", "batch16384", "width8192", "width32768"):
        # Calibration uses window=1 and no optimizer overlap; validation does
        # not contribute timings to the phase profile.
        result["cal_" + name] = dict(result[name], live=1, overlap="off")
    for wave in ("off", "bounded", "lazy"):
        for overlap in ("off", "on"):
            result[f"time_{wave}_{overlap}"] = dict(result["base"], wave=wave, overlap=overlap)
    for gather in ("eager", "one_hook_lag", "after_backward", "deferred"):
        result["time_zero_" + gather] = dict(result["tp2dp2_zero"], gather=gather)
    return result



def cases():
    result = original_cases()
    base = result["base"]
    for suffix, changes in {
        "torch": {}, "triton": {"key_backend": "triton"},
        "zero_torch": {"dp": 2, "zero": True},
        "zero_triton": {"dp": 2, "zero": True, "key_backend": "triton"},
    }.items():
        result["keys_"+suffix] = dict(base, **changes)
    for name, changes in {
        "dense": {"batch": 10240, "d_sae": 20480},
        "small": {"batch": 5120, "d_sae": 10240},
        "tp4": {"batch": 10240, "d_sae": 20480, "tp": 4},
        "ga2": {"batch": 10240, "d_sae": 20480, "ga": 2},
        "pp": {"batch": 10240, "d_sae": 10240, "pp": 2},
        "full": {"batch": 10240, "d_sae": 10240, "backend": "legacy"},
        "ragged": {"batch": 5120, "d_sae": 20480, "backend": "sharded_ragged", "main_compute": "sparse", "aux_compute": "compact_dense"},
        "dp4": {"batch": 10240, "d_sae": 10240, "tp": 1, "dp": 4, "zero": True},
        "triton": {"batch": 10240, "d_sae": 20480, "key_backend": "triton"},
        "zero_triton": {"batch": 10240, "d_sae": 20480, "dp": 2, "zero": True, "key_backend": "triton"},
    }.items():
        result["hold_"+name] = dict(base, **changes)
    for wave in ("off", "bounded", "lazy"):
        result["hold_wave_"+wave] = dict(base, batch=10240, d_sae=20480, wave=wave)
    for gather in ("eager", "one_hook_lag", "after_backward", "deferred"):
        result["hold_zero_"+gather] = dict(base, batch=10240, d_sae=20480, dp=2, zero=True, gather=gather)
    return result


def instrument(trainer, torch, dist):
    import sae_lens.sharded_topk as topk
    original_candidates = topk._local_candidates
    @functools.wraps(original_candidates)
    def candidates(*args, **kwargs):
        with torch.cuda.nvtx.range("time:op:local_candidates"):
            return original_candidates(*args, **kwargs)
    topk._local_candidates = candidates

    # These functions are also called on autograd worker threads. Explicit
    # ranges preserve attribution of mixed sparse/dense decoder stages.
    import sae_lens.sparse_parts as sparse_parts
    for name in ("sparse_forward", "sparse_dvalues", "sparse_dweight"):
        original = getattr(sparse_parts, name)
        def sparse_wrapper(fn, label):
            @functools.wraps(fn)
            def measured(*args, **kwargs):
                with torch.cuda.nvtx.range("time:sparse:" + label):
                    return fn(*args, **kwargs)
            return measured
        setattr(sparse_parts, name, sparse_wrapper(original, name))

    def wrap(obj, name, hook, stage):
        original = getattr(obj, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            with torch.cuda.nvtx.range(f"time:phase:{hook}:{stage}"):
                return original(*args, **kwargs)
        setattr(obj, name, measured)

    for i, unit in enumerate(trainer.units.values()):
        h = str(i)
        for method, stage in (("forward", "forward"), ("backward", "backward"),
                              ("finish_window", "finish_grad"), ("zero_grad", "zero_grad")):
            wrap(unit, method, h, stage)
        for method, stage in (("tp_wavefront_encode_launch", "encode"),
                              ("tp_wavefront_decode_launch", "decode"),
                              ("tp_wavefront_finish", "finish"),
                              ("encode_with_hidden_pre", "encode"),
                              ("decode", "decode"),
                              ("_build_train_step_output", "finish")):
            wrap(unit.model, method, h, stage)
        wrap(unit.optimizer, "step", h, "optimizer")
        if hasattr(unit.ddp, "start_param_sync"):
            wrap(unit.ddp, "start_param_sync", h, "param_gather")
    for name in ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
                 "_all_gather_base", "_reduce_scatter_base", "all_gather"):
        original = getattr(dist, name)
        def make(original, name):
            @functools.wraps(original)
            def measured(*args, **kwargs):
                group = kwargs.get("group")
                group_kind = "other"
                ctx = trainer.runtime.require_local()
                if group is ctx.tp_group:
                    group_kind = "tp"
                elif group is ctx.dp_group:
                    group_kind = "dp"
                with torch.cuda.nvtx.range(f"time:comm:{group_kind}:{name}"):
                    return original(*args, **kwargs)
            return measured
        setattr(dist, name, make(original, name))


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
    from sae_lens.tp_layout import balanced_widths

    directory = args.output / args.case
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=c["tp"] * c["dp"] * c["pp"],
                            init_method=f"file://{directory}/world", timeout=timedelta(seconds=180))
    sources = hook_sources(c["h"], repeat=args.repeat_cached_hooks)
    runtime = SAERuntime.from_layout(tp_size=c["tp"], dp_size=c["dp"], placement_size=c["pp"], hooks=tuple(sources))
    monitor = StaticFailureMonitor(runtime, set(runtime._owned_groups))
    runtime.failure_monitor = monitor
    backward_waits = []
    monitor.backward_wait_observer = backward_waits.append
    ctx = runtime.require_local()
    hooks = list(ctx.domain.hooks)
    local_total = c["batch"] // c["dp"]
    micro = local_total // c["ga"]
    assert micro * c["ga"] * c["dp"] == c["batch"]
    dead_mode = c.get("dead_mode", "fixed")
    if dead_mode not in ("fixed", "natural"):
        raise ValueError("dead_mode must be fixed or natural")
    dead_window = c.get("dead_window", 0)
    if type(dead_window) is not int or dead_window < 0:
        raise ValueError("dead_window must be a nonnegative integer")
    if dead_mode == "natural" and c.get("mask_dir"):
        raise ValueError("Natural firing ages cannot be combined with an injected dead mask")
    try:
        cfg = SAETrainerConfig(device=f"cuda:{rank}", n_checkpoints=0,
            total_training_samples=(args.warmup + args.steps + 1) * local_total,
            train_batch_size_samples=micro, output_path=None,
            save_mse_every_n_steps=0, save_timing_every_n_steps=0, save_memory_every_n_steps=0,
            record_memory_empty_cache=False, record_memory_timeline_step=-1, synchronize_timing=False,
            lr=3e-4, lr_end=3e-4, lr_scheduler_name="constant", lr_warm_up_steps=0, lr_decay_steps=0,
            dead_feature_window=dead_window, feature_sampling_window=1000, autocast=False,
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
        if "execution" in c:
            cfg.sae_tp_overlap = c["wave"]
            cfg.sae_tp_overlap_max_live_hooks = c["live"]
        cfg.sae_runtime_output_retention = "summary"
        cfg.multi_sae_optimizer_overlap = c["overlap"]
        cfg.multi_sae_param_gather_overlap = c.get("gather", "one_hook_lag") != "deferred"
        cfg.multi_sae_param_gather_schedule = c.get("gather", "one_hook_lag") if c.get("gather") != "deferred" else "one_hook_lag"
        cfg.sae_single_replica_fast_path = True
        cfg.sae_gradient_accumulation_fusion = c.get("gradient_fusion", True)
        cfg.sae_ga1_loss_normalization = True
        cfg.multi_sae_tp_phase_fence = "auto"
        model_options = dict(d_in=c["d_in"], d_sae=c["d_sae"], k=c.get("k", 128), auxk=c.get("auxk"),
            device=f"cuda:{rank}", dtype="float32", normalize_activations="none",
            use_sparse_activations=False, topk_backend=c["backend"], topk_key_backend=c.get("key_backend", "torch"), auxk_decoder_backend=c["aux"],
            auxk_async_selection=c.get("aux_selection_overlap", True),
            ragged_decoder_engine="openai", v5_main_compute=c["main_compute"], v5_aux_compute=c["aux_compute"])
        model_options.update(c.get("execution", {}))
        model_cfg = TopKTrainingSAEConfig(**model_options)
        models, wrapped = {}, {}
        observed_dead = {}
        for hook in hooks:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = MegatronTopKSAE(model_cfg, runtime=runtime)
            models[hook] = model
            prepare_metadata = model.prepare_auxk_metadata
            def observe_metadata(mask, count, fn=prepare_metadata, key=hook):
                columns = fn(mask, count)
                # The runtime already read the global count. Tensor shape is
                # host metadata, so this observer adds no GPU synchronization.
                observed_dead[key] = dict(global_dead=count,
                    local_dead=0 if columns is None else columns.numel())
                return columns
            model.prepare_auxk_metadata = observe_metadata
            wrapped[hook] = wrap_runtime_sae(model, runtime, distributed_optimizer=c["zero"],
                single_replica_fast_path=True,
                gradient_accumulation_fusion=c.get("gradient_fusion", True))
        trainer = MultiSAETrainer(hook_names=hooks, sae_by_hook=wrapped, base_sae_by_hook=models,
            data_provider=iter(()), save_checkpoint_fn=None, cfg=cfg, dp_group=ctx.dp_group,
            token_count_weighted_dp=True, sae_dp_mode="ddp", runtime=runtime)
        assert trainer.global_update_batch_size == c["batch"]
        intra_aux = all(getattr(m, "tp_wavefront_aux_selection_supported", lambda: False)()
                        and m.cfg.auxk != 0 for m in models.values())
        assert trainer._runtime_tp_wavefront == (c["wave"] != "off" and (c["tp"] > 1 or intra_aux) and (len(hooks) > 1 or intra_aux))
        order = torch.load(CACHE / "global_row_order.pt", weights_only=True)[:c["batch"]]
        assert len(order) == c["batch"]
        order = order[ctx.dp_rank * local_total:(ctx.dp_rank + 1) * local_total]
        batches = [{} for _ in range(c["ga"])]
        for hook in hooks:
            cached = torch.load(CACHE / f"{sources[hook]}.pt", map_location="cpu", weights_only=True, mmap=True)
            for i, batch in enumerate(batches):
                values = cached.index_select(0, order[i * micro:(i + 1) * micro]).float()
                if values.shape[1] != c["d_in"]:
                    if args.input_width_policy != "slice_repeat":
                        raise ValueError("Cached activation width differs from d_in; explicitly select --input-width-policy slice_repeat for a shape-only experiment")
                    # Input preparation is outside the timed training window.
                    columns = torch.arange(c["d_in"]) % values.shape[1]
                    values = values.index_select(1, columns)
                batch[hook] = values.to(rank)
                del values
            del cached
        mask_path = None
        if dead_mode == "natural":
            mask = torch.zeros(c["d_sae"], dtype=torch.bool, device=rank)
            assert all(not ages.any().item() for ages in trainer.n_forward_passes_since_fired_by_hook.values())
        elif c.get("mask_dir"):
            local_mask, mask_path = read_local_dead_mask(c["mask_dir"], c["tp"], ctx.tp_rank, c["d_sae"])
            parts = [read_local_dead_mask(c["mask_dir"], c["tp"], owner, c["d_sae"])[0]
                     for owner in range(c["tp"])]
            # Trainer ages currently have the replicated global feature axis.
            # Assembly is outside timing; each selector still uses its own slice.
            mask = torch.cat(parts).to(rank)
        else:
            # Compatibility only: one balanced synthetic distribution. It is
            # not a profile of arbitrary masks with the same global dead count.
            mask = torch.zeros(c["d_sae"], dtype=torch.bool, device=rank)
            ids = (torch.arange(c["dead"], device=rank) * c["d_sae"] // c["dead"]
                   if c["dead"] else torch.empty(0, device=rank, dtype=torch.long))
            mask[ids] = True
        widths = balanced_widths(c["d_sae"], c["tp"])
        def count_by_tp(value):
            return [int(part.sum()) for part in value.split(widths)]
        dead_by_tp = count_by_tp(mask)
        c = dict(c, dead_mode=dead_mode, dead_window=dead_window)
        if dead_mode == "fixed":
            c["dead"] = sum(dead_by_tp)
        dump(directory / f"dead_mask_rank{rank}.json", dict(
            source=("natural_cold_start" if dead_mode == "natural" else
                    "explicit_local_bool" if mask_path else "legacy_generated_balanced"),
            dead_window=dead_window,
            dead_by_tp=dead_by_tp, total_dead=sum(dead_by_tp),
            path=str(mask_path) if mask_path else None,
            file_sha256=sha(mask_path) if mask_path else None,
        ))
        if args.trace:
            instrument(trainer, torch, dist)
        # Candidate-stage scalar timers; no CUDA sync or
        # production training changes. Keep the measured CPU issuance visible.
        import sae_lens.sharded_topk as topk
        candidate_cpu = []
        original_candidates = topk._local_candidates
        def timed_candidates(*a, **kw):
            before = time.perf_counter()
            result = original_candidates(*a, **kw)
            candidate_cpu.append((time.perf_counter()-before)*1000)
            return result
        topk._local_candidates = timed_candidates
        steps = []
        dead_history = []
        total = args.warmup + args.steps
        for step in range(total):
            if dead_mode == "fixed":
                for ages in trainer.n_forward_passes_since_fired_by_hook.values():
                    ages.copy_(mask.to(ages.dtype) * (dead_window + 1))
            torch.cuda.synchronize()
            dist.barrier(group=runtime.control_group)
            if step == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            if args.trace and step == args.warmup:
                if rank == 0:
                    torch.cuda.profiler.start()
                dist.barrier(group=runtime.control_group)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            scope = torch.cuda.nvtx.range(f"time:step:{step}") if args.trace else contextlib.nullcontext()
            with scope:
                start = time.perf_counter()
                a.record()
                outputs, _ = train_runtime_window(trainer, batches)
                b.record()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            losses = {h: float(o.loss) for h, o in outputs.items()}
            assert all(math.isfinite(v) for v in losses.values())
            assert all(u.update_count == step + 1 for u in trainer.units.values())
            assert all(v == c["batch"] for v in trainer._last_global_tokens_by_hook.values())
            dead_sample = dict(step=step, timed=step >= args.warmup,
                               dead_by_hook=dict(observed_dead),
                               aux_execution={h: dict(getattr(m, "_last_auxk_execution", None) or {})
                                              for h, m in models.items()},
                               aux_async_launched={h: bool(getattr(m, "_last_auxk_async_selection", False))
                                                   for h, m in models.items()})
            dead_history.append(dead_sample)
            if step >= args.warmup:
                steps.append(dict(step=step, ms=elapsed, cuda_ms=a.elapsed_time(b), losses=losses,
                                  dead_by_hook=dead_sample["dead_by_hook"],
                                  aux_execution=dead_sample["aux_execution"],
                                  aux_async_launched=dead_sample["aux_async_launched"],
                                  backward_waits=list(backward_waits), candidate_cpu_ms=list(candidate_cpu)))
            backward_waits.clear()
            candidate_cpu.clear()
            del outputs
            if "execution" in c and step >= args.warmup:
                steps[-1]["post_output_allocated_bytes"] = torch.cuda.memory_allocated()
            trainer.lr_scheduler.step(trainer._last_updated_hooks)
            trainer.n_training_steps += 1
            trainer.n_training_samples += local_total
        torch.cuda.synchronize()
        dist.barrier(group=runtime.control_group)
        if args.trace and rank == 0:
            torch.cuda.profiler.stop()
        measured_peak_allocated = torch.cuda.max_memory_allocated()
        measured_peak_reserved = torch.cuda.max_memory_reserved()
        measured_main_execution = [getattr(u.model, "_last_main_execution", None) for u in trainer.units.values()]
        measured_aux_execution = [getattr(u.model, "_last_auxk_execution", None) for u in trainer.units.values()]
        # Observe D and actual E/B in an extra UNTIMED update, using the same
        # mask/logits path as training. Never infer E from D or nonzero values.
        from sae_lens.training.sae_trainer import DeadFeatureHistory
        from sae_lens.training.gradient_window import window_metrics
        cfg.output_path, cfg.save_dead_every_n_steps = str(directory), 1
        trainer.dead_feature_history = DeadFeatureHistory(
            cfg, models, writer=ctx.dp_rank == 0 and ctx.tp_rank == 0,
            runtime=runtime, dp_group=ctx.dp_group,
            pp_rank=runtime.domains.index(ctx.domain),
        )
        if dead_mode == "fixed":
            for ages in trainer.n_forward_passes_since_fired_by_hook.values():
                ages.copy_(mask.to(ages.dtype) * (dead_window + 1))
        audit_dead_by_hook = {
            h: count_by_tp(ages > dead_window)
            for h, ages in trainer.n_forward_passes_since_fired_by_hook.items()
        }
        audit, _ = train_runtime_window(trainer, batches)
        trainer.dead_feature_history.write(trainer.n_training_steps + 1,
            trainer.n_training_samples + local_total, **window_metrics(trainer))
        del audit
        dump(directory / f"rank{rank}.json", dict(rank=rank, pid=os.getpid(), config=c,
            actual_dead_by_tp=dead_by_tp if dead_mode == "fixed" else None,
            actual_dead_by_hook=audit_dead_by_hook, dead_history=dead_history,
            dead_observation="update_start_per_step_and_extra_untimed_update",
            local_hooks=hooks, hook_sources=sources, microbatch=micro, trace=args.trace, steps=steps,
            effective_wavefront=trainer._runtime_tp_wavefront,
            effective_overlap=trainer._runtime_optimizer_overlap,
            effective_overlap_reason=trainer._runtime_optimizer_overlap_reason,
            effective_gradient_fusion={h: bool(getattr(m, "gradient_accumulation_fusion", False))
                                       for h, m in models.items()},
            gather=trainer._runtime_param_gather_schedule,
            main_execution=measured_main_execution,
            aux_execution=measured_aux_execution,
            resolved_model_config=model_cfg.to_dict(),
            optimizer_classes=[type(u.optimizer).__module__ + "." + type(u.optimizer).__name__
                               for u in trainer.units.values()],
            input_width_policy=args.input_width_policy,
            peak_allocated_bytes=measured_peak_allocated,
            peak_reserved_bytes=measured_peak_reserved,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda))
    except BaseException as exc:
        monitor.fail(exc)
        raise
    finally:
        monitor.close()
        runtime.close()
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--config-file", type=Path, help="JSON mapping of additional case names to complete configurations")
    p.add_argument("--repeat-cached-hooks", action="store_true",
                   help="Allow >3 independent SAEs by cycling the three cached layer inputs (shape experiment)")
    p.add_argument("--dead-mask-dir", type=Path,
                   help="Explicit bool [d_sae/TP] files tp{TP}_rank{rank}.pt; overrides case mask_dir and dead count. The untimed audit saves D and actual AuxK E/B.")
    p.add_argument("--input-width-policy", choices=("exact", "slice_repeat"), default="exact")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep failed case logs and continue independent cases (e.g. OOM audits)")
    p.add_argument("--case")
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    if args.warmup < 1 or args.steps < 1:
        p.error("warmup and steps must be positive")
    available = cases()
    if args.config_file:
        args.config_file = args.config_file.resolve()
        custom = json.loads(args.config_file.read_text())
        if not isinstance(custom, dict) or not custom or any(not isinstance(c, dict) for c in custom.values()):
            p.error("config-file must be a nonempty mapping of case names to configurations")
        if set(custom) & set(available):
            p.error("Custom case names must not replace built-in cases")
        if any(Path(name).name != name or name in (".", "..") for name in custom):
            p.error("Case names must be single directory names")
        available.update(custom)
    args.cases = args.cases or (list(custom) if args.config_file else ["base"])
    if args.dead_mask_dir:
        args.dead_mask_dir = args.dead_mask_dir.resolve()
        available = {name: dict(c, mask_dir=str(args.dead_mask_dir)) for name,c in available.items()}
    if any(name not in available for name in args.cases) or (args.worker and args.case not in available):
        p.error("Unknown case name")
    args.output = args.output.resolve()
    if args.worker:
        import torch.multiprocessing as mp
        c = available[args.case]
        mp.spawn(worker, args=(args, c), nprocs=c["tp"] * c["dp"] * c["pp"], join=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    source = {str(f.relative_to(REPO)): sha(f) for f in (REPO / "sae_lens").rglob("*.py")
              if "autoconfig" not in f.parts}
    source[str(Path(__file__).resolve().relative_to(REPO))] = sha(__file__)
    source["scripts/profile/profile_static_memory_factors.py"] = sha(Path(__file__).with_name("profile_static_memory_factors.py"))
    for name in args.cases:
        c = available[name]
        if c.get("mask_dir"):
            for rank in range(c["tp"]):
                _, path = read_local_dead_mask(c["mask_dir"], c["tp"], rank, c["d_sae"])
                source[str(path.resolve())] = sha(path)
    if args.config_file:
        source[str(args.config_file)] = sha(args.config_file)
    manifest = args.output / "source_sha256.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == source, "Timing source changed: use a new output directory"
    else:
        dump(manifest, source)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", NCCL_LAUNCH_ORDER_IMPLICIT="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for name in args.cases:
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            if result["returncode"] == 0:
                continue
            raise RuntimeError(f"Failed prior run: {directory}; use a fresh output directory")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--case", name,
                   "--output", str(args.output), "--warmup", str(args.warmup), "--steps", str(args.steps),
                   "--input-width-policy", args.input_width_policy]
        if args.config_file:
            command += ["--config-file", str(args.config_file)]
        if args.repeat_cached_hooks:
            command += ["--repeat-cached-hooks"]
        if args.dead_mask_dir:
            command += ["--dead-mask-dir", str(args.dead_mask_dir)]
        if args.trace:
            command += ["--trace"]
            command = ["nsys", "profile", "--sample=none", "--cpuctxsw=none", "--trace=cuda,nvtx",
                       "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--export=sqlite",
                       f"--output={directory / 'trace'}", *command]
        dump(directory / "command.json", dict(argv=command, config=available[name]))
        print("START", name, flush=True)
        begin = time.monotonic()
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
        dump(directory / "result.json", dict(returncode=rc, elapsed_s=time.monotonic()-begin))
        print("END", name, rc, flush=True)
        if rc and not args.continue_on_error:
            raise RuntimeError(f"Case failed: {directory / 'run.log'}")


if __name__ == "__main__":
    main()
