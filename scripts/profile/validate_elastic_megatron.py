"""Real NCCL acceptance for DP resizing, including native optimizer resharding.

Four GPUs exercise TP1 DP1/2/3, TP2 DP1/2 and two hook placements, independently
of vLLM. Checks numerical continuation against uninterrupted training, exact
state round trips, native reducer/optimizer ownership and old-model release.
"""
import argparse
import gc
import json
import os
import sys
import time
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.elastic_runtime_state import (
    broadcast_runtime_models,
    capture_runtime_state,
    restore_runtime_state,
    retire_runtime_trainer,
)
from sae_lens.training.exact_dp_batch_provider import ExactDataParallelBatchProvider
from sae_lens.training.megatron_ddp import wrap_runtime_sae
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from tests.saes.test_megatron_sae_trainers import _trainer_config  # noqa: TID251


def build(runtime, zero, output, *, batch=65, d_in=32, d_sae=256, k=32,
          topk_backend="sharded_sparse", accumulation=1, execution=None):
    cfg = _trainer_config(f"cuda:{dist.get_rank()}", output, "unified_multi_hook")
    cfg.ddp_zero_optimizer = zero
    cfg.multi_sae_optimizer_overlap = "on"
    cfg.multi_sae_param_gather_schedule = "one_hook_lag"
    cfg.train_batch_size_samples = batch
    cfg.gradient_accumulation_steps = accumulation
    cfg.total_training_samples = batch * 20 * accumulation
    cfg.lr_scheduler_name = "cosineannealing"
    cfg.dead_feature_window = 2
    if execution:
        cfg.sae_tp_overlap = execution.get('tp_overlap', 'lazy')
    context = runtime.require_local()
    models = {}
    for i, hook in enumerate(context.domain.hooks):
        torch.manual_seed(812 + int(hook[1:]))
        model_cfg = TopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=min(k, d_sae // 8),
            device=cfg.device, dtype="float32", topk_backend=topk_backend,
        )
        if execution:
            for key, value in execution.items():
                if key == 'tp_overlap':continue
                if not hasattr(model_cfg,key):raise ValueError(f'Unknown SAE execution field: {key}')
                setattr(model_cfg,key,value)
            model_cfg.ragged_decoder_engine='openai'
        models[hook] = MegatronTopKSAE(model_cfg, runtime=runtime)
    wrapped = {h: wrap_runtime_sae(m, runtime, distributed_optimizer=zero)
               for h, m in models.items()}
    trainer = MultiSAETrainer(
        hook_names=list(models), sae_by_hook=wrapped, base_sae_by_hook=models,
        data_provider=SimpleNamespace(tracks_global_progress=True),
        cfg=cfg, runtime=runtime, save_checkpoint_fn=lambda *_args, **_kwargs: None,
        dp_group=context.dp_group,
        token_count_weighted_dp=True, sae_dp_mode="ddp", backward_mode="sequential",
        seed_mode="same",
    )
    assert trainer.global_update_batch_size == batch * accumulation
    for unit in trainer.units.values():
        if context.dp_group.size() > 1:
            assert unit.ddp._sae_megatron_ddp
            assert unit.ddp.dp_group is context.dp_group
            assert unit.early_grad_sync
        assert type(unit.optimizer).__name__ == (
            "GPUClipDistributedOptimizer" if zero and context.dp_group.size() > 1
            else "GPUClipFP32Optimizer"
        )
    return trainer


def advance(trainer, step, batch=65):
    ctx = trainer.runtime.require_local()
    d_in = next(iter(trainer.base_sae_by_hook.values())).cfg.d_in
    # Rotate remainder ownership exactly as the streaming exact provider does.
    from sae_lens.training.dp_batch import balanced_token_counts
    accumulation = trainer.cfg.gradient_accumulation_steps
    batches = []
    for micro in range(accumulation):
        index = step * accumulation + micro
        rng = torch.Generator().manual_seed(901 + index)
        full = torch.randn(batch, d_in, generator=rng).to(trainer.cfg.device)
        counts = balanced_token_counts(batch, ctx.dp_group.size(), index)
        start = sum(counts[:ctx.dp_rank])
        local = full[start:start + counts[ctx.dp_rank]]
        batches.append({h: local + int(h[1:]) * .01 for h in trainer.hook_names})
    if accumulation == 1:
        trainer._train_step(batches[0], len(local))
    else:
        from sae_lens.training.gradient_window import train_runtime_window
        train_runtime_window(trainer, batches)
    trainer.lr_scheduler.step(trainer._last_updated_hooks)
    trainer.n_training_steps += 1
    trainer.n_training_samples += batch * accumulation


def compare(before, after, *, exact=False):
    maximum = 0.0
    for category in ("models", "optimizers", "tensors"):
        def walk(a, b):
            nonlocal maximum
            assert a.keys() == b.keys()
            for key in a:
                if isinstance(a[key], dict):
                    walk(a[key], b[key])
                elif torch.is_tensor(a[key]):
                    maximum = max(maximum, float((a[key] - b[key]).abs().max()))
                    torch.testing.assert_close(a[key], b[key], atol=0 if exact else 8e-6,
                                               rtol=0 if exact else 8e-5)
                else:
                    assert a[key] == b[key]
        walk(getattr(before, category), getattr(after, category))
    for key in ("n_training_steps", "n_training_samples", "update_counts", "lr_scheduler",
                "n_frac_active_samples_by_hook", "token_count_remainder", "grad_scaler",
                "activation_scalers"):
        assert before.scalars[key] == after.scalars[key], key
    if exact:
        for key in ("torch_rng_state", "cuda_rng_state"):
            torch.testing.assert_close(before.scalars[key], after.scalars[key], atol=0, rtol=0)
    return maximum


def worker(rank, rendezvous, output, short_batch=False, accumulation=1, execution=None):
    import faulthandler
    faulthandler.dump_traceback_later(90, repeat=True)
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
            hooks = [("h0", "h1", "h2")] if len(small) == 1 else [("h0", "h1"), ("h2",)]
            runtimes = [SAERuntime(tuple(SAETrainingDomain(f"p{i}", tuple(r), tp, hooks[i])
                                        for i, r in enumerate(members))) for members in (small, big)]
            def source_for(runtime):
                pp = runtime.domains.index(runtime.require_local().domain)
                return small[pp][runtime.require_local().tp_rank]
            for zero in (False, True):
                case = f"{name}_zero{int(zero)}"
                reference = None
                if runtimes[0].local is not None:
                    trainer = build(runtimes[0], zero, output, accumulation=accumulation, execution=execution)
                    for step in range(12):
                        advance(trainer, step, 1 if short_batch and step == 4 else 65)
                    reference = capture_runtime_state(trainer, source_global_rank=source_for(runtimes[0]))
                    retire_runtime_trainer(trainer)
                    del trainer
                dist.barrier(group=control)
                trainer = build(runtimes[0], zero, output, accumulation=accumulation, execution=execution) if runtimes[0].local else None
                active = runtimes[0]
                switches = []
                for step in range(12):
                    if step in (3, 5, 7, 9):
                        saved = None
                        refs = []
                        if trainer is not None:
                            context = active.require_local()
                            polls = []
                            def poll():
                                assert rank == small[0][0], "TP/placement peer read control independently"
                                polls.append(step)
                                return step, 2 if step in (3, 7) else 1
                            provider = ExactDataParallelBatchProvider(
                                source=iter([]) if context.dp_rank == 0 else None,
                                dp_group=context.dp_group, dp_idx=context.dp_rank,
                                dp_size=context.dp_group.size(), device=torch.device(trainer.cfg.device),
                                dtype=torch.float32, d_model=32, reconfigure_poll=poll,
                                reconfigure_group=active.training_group,
                                reconfigure_source_global_rank=small[0][0],
                            )
                            assert provider.poll_reconfigure()
                            assert provider.pending_reconfigure == (step, 2 if step in (3, 7) else 1)
                            assert len(polls) == int(rank == small[0][0])
                            assert provider.global_tokens_consumed == 0
                            saved = capture_runtime_state(trainer, source_global_rank=source_for(active))
                            refs = [weakref.ref(u.ddp) for u in trainer.units.values()]
                            retire_runtime_trainer(trainer)
                            del trainer
                            gc.collect()
                            assert all(r() is None for r in refs), "Old native DDP/model leaked"
                        active = runtimes[1] if step in (3, 7) else runtimes[0]
                        trainer = None
                        if active.local:
                            trainer = build(active, zero, output, accumulation=accumulation, execution=execution)
                            # DDP construction has already happened in this standalone test;
                            # weights are broadcast into its native buffer views before stepping.
                            import copy
                            expected = copy.deepcopy(saved)
                            broadcast_runtime_models(saved, trainer.base_sae_by_hook,
                                                     group=active.require_local().dp_group,
                                                     source_global_rank=source_for(active))
                            restore_runtime_state(saved, trainer, source_global_rank=source_for(active))
                            roundtrip = capture_runtime_state(trainer, source_global_rank=source_for(active))
                            if expected:
                                compare(expected, roundtrip, exact=True)
                            switches.append(dict(step=step, dp=active.require_local().dp_group.size()))
                        dist.barrier(group=control)
                    if trainer is not None:
                        advance(trainer, step, 1 if short_batch and step == 4 else 65)
                error = None
                if trainer is not None:
                    final = capture_runtime_state(trainer, source_global_rank=source_for(active))
                    if reference:
                        error = compare(reference, final)
                    retire_runtime_trainer(trainer)
                    del trainer
                reports.append(dict(case=case, switches=switches, max_abs_error=error,
                                    short_batch=short_batch, accumulation=accumulation, status="PASS"))
                Path(output, f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
                dist.barrier(group=control)
                if rank == 0:
                    print("PASS", case, flush=True)
            for runtime in reversed(runtimes):
                runtime.close()
        dist.barrier(group=control)
    except BaseException:
        import traceback
        Path(output, f"error_rank{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--short-batch", action="store_true",
                        help="Include a one-token update with empty DP replicas before resharding")
    parser.add_argument("--accumulation", type=int, default=1,
                        help="Microbatches per update during numerical cutover validation")
    parser.add_argument('--execution-config', help='JSON file with independent representation/compute fields and optional tp_overlap')
    parser.add_argument("--memory-switches", type=int, default=0,
                        help="Run repeated cutovers with memory/object-lifetime observations")
    parser.add_argument("--memory-case", default="", choices=("", "tp1_dp1_2", "tp1_dp2_3", "tp2_dp1_2", "placement2_dp1_2"))
    parser.add_argument("--memory-optimizer", default="both", choices=("both", "adam", "distributed"))
    parser.add_argument("--memory-d-in", type=int, default=256)
    parser.add_argument("--memory-d-sae", type=int, default=2048)
    parser.add_argument("--memory-batch", type=int, default=65)
    parser.add_argument("--memory-steps", type=int, default=3)
    parser.add_argument("--memory-backend", default="sharded_sparse",
                        choices=("sharded_sparse", "sharded_ragged"))
    parser.add_argument("--memory-static", action="store_true",
                        help="Keep the small topology fixed for the same training windows")
    args = parser.parse_args()
    if args.execution_config and args.memory_switches:
        parser.error('--execution-config currently applies to numerical cutover validation, not --memory-switches')
    if args.memory_switches < 0 or min(args.accumulation, args.memory_d_in, args.memory_d_sae, args.memory_batch, args.memory_steps) < 1:
        parser.error("Memory-test dimensions/steps must be positive; switch count must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    started = time.monotonic()
    worker_args = (f"file://{args.output.resolve() / 'rendezvous'}", str(args.output.resolve()))
    if args.memory_switches:
        from scripts.profile.elastic_memory_probe import worker as memory_worker

        options = dict(switches=args.memory_switches, case=args.memory_case,
                       optimizer=args.memory_optimizer, d_in=args.memory_d_in,
                       d_sae=args.memory_d_sae, batch=args.memory_batch, steps=args.memory_steps,
                       backend=args.memory_backend, static=args.memory_static)
        mp.spawn(memory_worker, args=(*worker_args, options), nprocs=4, join=True)
    else:
        execution=json.loads(Path(args.execution_config).read_text()) if args.execution_config else None
        mp.spawn(worker, args=(*worker_args, args.short_batch, args.accumulation, execution), nprocs=4, join=True)
    (args.output / "summary.json").write_text(json.dumps(dict(
        status="PASS", elapsed_s=time.monotonic()-started,
        configuration=vars(args) | {"output": str(args.output)},
        reports=[json.loads((args.output/f"rank{i}.json").read_text()) for i in range(4)],
    ), indent=2))
