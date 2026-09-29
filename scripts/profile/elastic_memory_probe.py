"""Repeated native elastic cutovers; invoked by validate_elastic_megatron.py.

Records live allocations, allocator reservations, driver usage and object
liveness separately. No empty_cache is added to steady-state training.
"""

import gc
import json
import os
import time
import weakref
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain
from sae_lens.training.elastic_runtime_state import (
    broadcast_runtime_models,
    capture_runtime_state,
    restore_runtime_state,
    retire_runtime_trainer,
)
from scripts.profile.validate_elastic_megatron import advance, build


def memory_sample():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        active=stats.get("active_bytes.all.current", 0),
        inactive_split=stats.get("inactive_split_bytes.all.current", 0),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        device_used=total - free,
        pg_count=len(dist.distributed_c10d._world.pg_map),
        rss=int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
    )


def training_weakrefs(trainer):
    """Do not accidentally retain the tensors being checked in the probe."""
    refs = {}
    for hook, unit in trainer.units.items():
        for name in ("model", "ddp", "optimizer"):
            refs[f"{hook}.{name}"] = weakref.ref(getattr(unit, name))
        inner = unit.optimizer.optimizer
        refs[f"{hook}.inner_optimizer"] = weakref.ref(inner)
        for name, parameter in unit.model.named_parameters():
            refs[f"{hook}.{name}"] = weakref.ref(parameter)
        for index, state in enumerate(inner.state.values()):
            for name, value in state.items():
                if torch.is_tensor(value):
                    refs[f"{hook}.adam{index}.{name}"] = weakref.ref(value)
        buffers = unit.ddp.buffers if getattr(unit.ddp, "_sae_megatron_ddp", False) else ()
        for index, buffer in enumerate(buffers):
            for name in ("param_data", "grad_data"):
                value = getattr(buffer, name, None)
                if value is not None:
                    refs[f"{hook}.buffer{index}.{name}"] = weakref.ref(value)
    return refs


def worker(rank, rendezvous, output, options):
    import faulthandler

    faulthandler.dump_traceback_later(180, repeat=True)
    with Path(output, f"worker{rank}.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=180))
    control = dist.new_group(backend="gloo")
    reports = []
    specs = [
        ("tp1_dp1_2", [(3,)], [(2, 3)], 1),
        ("tp1_dp2_3", [(2, 3)], [(1, 2, 3)], 1),
        ("tp2_dp1_2", [(2, 3)], [(0, 1, 2, 3)], 2),
        ("placement2_dp1_2", [(2,), (3,)], [(0, 2), (1, 3)], 1),
    ]
    try:
        for name, small, big, tp in specs:
            if options["case"] and name != options["case"]:
                continue
            hooks = [("h0", "h1", "h2")] if len(small) == 1 else [("h0", "h1"), ("h2",)]
            runtimes = [SAERuntime(tuple(
                SAETrainingDomain(f"p{i}", tuple(r), tp, hooks[i])
                for i, r in enumerate(members))) for members in (small, big)]

            def source_for(runtime):
                pp = runtime.domains.index(runtime.require_local().domain)
                return small[pp][runtime.require_local().tp_rank]

            def create(runtime, zero):
                result = build(runtime, zero, output, batch=options["batch"],
                               d_in=options["d_in"], d_sae=options["d_sae"], k=32,
                               topk_backend=options.get("backend", "sharded_sparse"))
                result.cfg.sae_runtime_output_retention = "summary"
                return result

            for zero in (False, True):
                if options["optimizer"] != "both" and zero != (options["optimizer"] == "distributed"):
                    continue
                case = f"{name}_zero{int(zero)}"
                case_start = time.monotonic()
                active = runtimes[0]
                trainer = create(active, zero) if active.local else None
                expected_pg_count = len(dist.distributed_c10d._world.pg_map)
                report = dict(case=case, phases=[], checked_old_objects=0)
                step = 0
                for epoch in range(options["switches"] + 1):
                    torch.cuda.reset_peak_memory_stats()
                    if trainer is not None:
                        for _ in range(options["steps"]):
                            advance(trainer, step, options["batch"])
                            step += 1
                    dist.barrier(group=control)
                    sample = memory_sample()
                    dead_features = {} if trainer is None else {
                        h: int((value > trainer.cfg.dead_feature_window).sum())
                        for h, value in trainer.n_forward_passes_since_fired_by_hook.items()
                    }
                    report["phases"].append(dict(epoch=epoch, phase="trained",
                        topology=0 if options.get("static") else epoch % 2,
                        dead_features=dead_features, consumer=active.local is not None, **sample))
                    if options.get("static") and epoch < options["switches"]:
                        Path(output, f"rank{rank}.json").write_text(json.dumps(reports + [report], indent=2))
                        continue
                    torch.cuda.reset_peak_memory_stats()
                    saved = None
                    if trainer is not None:
                        saved = capture_runtime_state(trainer, source_global_rank=source_for(active))
                        refs = training_weakrefs(trainer)
                        retire_runtime_trainer(trainer)
                        trainer = None
                        gc.collect()
                        alive = [name for name, ref in refs.items() if ref() is not None]
                        if alive:
                            raise AssertionError(f"{case} epoch {epoch}: retained {alive}")
                        report["checked_old_objects"] += len(refs)
                        del refs
                    # Matches the production runner's _elastic_drop_sae.
                    gc.collect()
                    torch.cuda.empty_cache()
                    report["phases"].append(dict(epoch=epoch, phase="retired",
                        topology=0 if options.get("static") else epoch % 2,
                        consumer=active.local is not None, **memory_sample()))
                    dist.barrier(group=control)
                    if epoch == options["switches"]:
                        break
                    active = runtimes[(epoch + 1) % 2]
                    if active.local:
                        trainer = create(active, zero)
                        broadcast_runtime_models(saved, trainer.base_sae_by_hook,
                            group=active.require_local().dp_group, source_global_rank=source_for(active))
                        restore_runtime_state(saved, trainer, source_global_rank=source_for(active))
                        step = trainer.n_training_steps
                    saved = None
                    report["phases"].append(dict(epoch=epoch + 1, phase="restored",
                        topology=(epoch + 1) % 2, consumer=active.local is not None, **memory_sample()))
                    assert len(dist.distributed_c10d._world.pg_map) == expected_pg_count
                    Path(output, f"rank{rank}.json").write_text(json.dumps(reports + [report], indent=2))
                    if rank == 0 and (epoch + 1) % 10 == 0:
                        print(case, epoch + 1, "cutovers", flush=True)
                saved = None
                report.update(status="PASS", switches=0 if options.get("static") else options["switches"],
                              elapsed_s=time.monotonic() - case_start)
                reports.append(report)
                Path(output, f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
                dist.barrier(group=control)
            for runtime in reversed(runtimes):
                runtime.close()
        dist.barrier(group=control)
    except BaseException:
        import traceback

        Path(output, f"error_rank{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()
