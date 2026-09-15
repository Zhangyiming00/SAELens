"""Trace the unchanged static GA=1 path with per-hook NVTX attribution.

Run under nsys --capture-range=cudaProfilerApi --capture-range-end=stop.
The real vLLM producer, routing, shuffle/mixing buffer, failure fences and
optimizer are retained. Only window boundaries synchronize the CUDA device.
"""

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def nvtx_method(obj, method, label):
    original = getattr(obj, method)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with torch.cuda.nvtx.range(label() if callable(label) else label):
            return original(*args, **kwargs)

    setattr(obj, method, wrapped)


def instrument_backward_hooks():
    # Install before DDP construction. The original Megatron callback executes
    # unchanged, including its add_, release of param.grad, and ready register.
    from megatron.core.distributed import DistributedDataParallel

    original = DistributedDataParallel._make_backward_post_hook

    def make_hook(ddp, parameter):
        callback = original(ddp, parameter)
        name = next(n for n, p in ddp.module.named_parameters() if p is parameter)

        def wrapped(*args):
            with torch.cuda.nvtx.range(f"ga1:{ddp._ga1_hook}:grad_add:{name}"):
                return callback(*args)

        return wrapped

    DistributedDataParallel._make_backward_post_hook = make_hook


def _worker(rank, args):
    sys.path.insert(0, str(args.source_root))
    h3 = args.workload == "h3_cached"
    tp, dp = (1, 1) if args.layout == "tp1dp1" else (2, 2)
    world = tp * dp if h3 else 4
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29572",
        SAE_ADAM_IMPL="fused", VLLM_ENABLE_V1_MULTIPROCESSING="0",
        NCCL_LAUNCH_ORDER_IMPLICIT=str(args.implicit_order),
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", rank=rank, world_size=world,
        init_method=f"file://{args.output / 'world'}",
        timeout=timedelta(seconds=180),
    )
    caller_groups = set(dist.distributed_c10d._world.pg_map)
    from datasets import Dataset

    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
    from sae_lens.sae_runtime import SAETrainingDomain
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    instrument_backward_hooks()
    hooks = [f"blocks.{i}.hook_resid_post" for i in ((16, 21, 26) if h3 else (21, 26))]
    global_batch = 2048 if h3 else 4096
    total_steps = args.warmup + args.steps * args.windows
    dataset = None if h3 else Dataset.from_dict({"tokens": [
        [1000] + [1001 + ((i * 37 + j) % 1999) for j in range(31)]
        for i in range(1024)
    ]})
    cfg_kwargs = dict(
        sae=TopKTrainingSAEConfig(
            d_in=4096, d_sae=32768 if h3 else 16384, k=128 if h3 else 4, device=f"cuda:{rank}",
            use_sparse_activations=not h3, normalize_activations="none",
        ),
        model_name=args.model, model_class_name="VLLMModel",
        model_from_pretrained_kwargs=dict(
            tensor_parallel_size=1 if h3 else 4, max_model_len=2049 if h3 else 33,
            gpu_memory_utilization=0.5 if h3 else 0.35, enforce_eager=not h3,
        ),
        hook_name=hooks[0], hook_names=hooks,
        dataset_path=args.dataset if h3 else "local-static-benchmark", is_dataset_tokenized=True,
        use_cached_activations=h3, cached_activations_path=str(args.cache) if h3 else None,
        streaming=False, context_size=2048 if h3 else 32,
        # Keep the aggregate DP mixing capacity at 4096 tokens. At DP1 the
        # entire global batch lives on one rank, so 64*32 would be too small.
        store_batch_size_prompts=2 if h3 else 192, n_batches_in_buffer=2 if h3 else 128 // dp,
        training_tokens=(global_batch if h3 else global_batch // dp) * total_steps,
        train_batch_size_tokens=global_batch if h3 else global_batch // dp,
        routing_dp_batch_mode="exact" if h3 else "equal",
        activations_mixing_fraction=0.5, device=f"cuda:{rank}",
        act_store_device=f"cuda:{rank}", dtype="float32",
        exclude_special_tokens=False if h3 else [1000], prepend_bos=h3, n_eval_batches=0,
        autocast=False, sae_dp_mode="ddp", lr=3e-4, lr_end=3e-5,
        lr_scheduler_name="constant" if h3 else "cosineannealing", n_checkpoints=0,
        dead_feature_window=-1 if h3 else 1000,
        save_final_checkpoint=False, output_path=str(args.output),
        checkpoint_path=str(args.output / "unused_checkpoints"),
        save_mse_every_n_steps=0, save_timing_every_n_steps=0,
        save_memory_every_n_steps=0, logger=LoggingConfig(log_to_wandb=False),
        gradient_accumulation_steps=1, ddp_bucket_cap_mb=None,
        step_window_profile_start_step=args.warmup + 1,
        step_window_profile_window_steps=args.steps,
        step_window_profile_window_count=args.windows,
    )
    # A pre-GA source snapshot is an additional H3 historical control. Missing
    # GA/bucket fields mean its original GA=1/default-bucket behavior.
    for key in ("gradient_accumulation_steps", "ddp_bucket_cap_mb"):
        if key not in LanguageModelSAERunnerConfig.__dataclass_fields__:
            cfg_kwargs.pop(key)
    cfg = LanguageModelSAERunnerConfig(**cfg_kwargs)
    waits, unit_info = [], {}

    class TraceRunner(LanguageModelSAETrainingRunner):
        def run_multi_trainer_with_interruption_handling(self, trainer):
            self.audited_trainer = trainer
            monitor = self.sae_runtime.failure_monitor
            profiler = trainer.step_window_profiler
            step_start, step_end = profiler.on_step_start, profiler.on_step_end
            zero_counts = dict.fromkeys(hooks, 0)

            def on_start(step):
                # Start one warmup step early so rank skew cannot truncate the
                # first measured step on another GPU. No additional barrier.
                if rank == 0 and step == args.warmup:
                    torch.cuda.profiler.start()
                step_start(step)
                zero_counts.update(dict.fromkeys(hooks, 0))
                torch.cuda.nvtx.range_push(f"ga1:step:r{rank}:s{step}")

            def on_end(step, **kwargs):
                step_end(step, **kwargs)
                torch.cuda.nvtx.range_pop()

            profiler.on_step_start, profiler.on_step_end = on_start, on_end

            def observe(event):
                step = trainer.n_training_steps + 1
                if step > args.warmup:
                    waits.append({"step": step, **event})

            monitor.backward_wait_observer = observe
            for hook, unit in trainer.units.items():
                megatron = getattr(unit.ddp, "_sae_megatron_ddp", False)
                if args.pre_ga_reference:
                    assert not megatron, "Historical reference must use the pre-GA source snapshot"
                else:
                    assert megatron
                assert unit.optimizer.defaults["fused"]
                unit.ddp._ga1_hook = hook
                prefix = f"ga1:{hook}:"
                for method, phase in [
                    ("forward", "forward"), ("backward", "backward"),
                    ("finish_window", "grad_normalize"),
                    ("finish_grad_sync", "grad_sync"),
                    ("clip_grad_norm", "clip"),
                ]:
                    if hasattr(unit, method):
                        nvtx_method(unit, method, prefix + phase)
                if megatron:
                    # Native ready hooks dispatch bucket groups directly;
                    # instrument the actual launch point for both schedules.
                    for group in unit.ddp.bucket_groups + unit.ddp.expert_parallel_bucket_groups:
                        nvtx_method(group, "start_grad_sync", prefix + "ddp_launch")
                    nvtx_method(unit.ddp, "finish_grad_sync", prefix + "ddp_wait")
                nvtx_method(unit.optimizer, "step", prefix + "fused_adam")

                def zero_label(h=hook, p=prefix):
                    zero_counts[h] += 1
                    return p + ("grad_zero_start" if zero_counts[h] == 1 else "grad_zero_end")

                if megatron:
                    nvtx_method(unit.ddp, "zero_grad_buffer", zero_label)
                unit_info[hook] = dict(
                    parameters={n: list(p.shape) for n, p in unit.model.named_parameters()},
                    bucket_size=unit.ddp.ddp_config.bucket_size if megatron else None,
                    buckets=[b.grad_data.numel() for buf in unit.ddp.buffers for b in buf.buckets] if megatron else [],
                    grad_buffer_bytes=sum(buf.grad_data.numel() * 4 for buf in unit.ddp.buffers) if megatron else None,
                    fused_adam=unit.optimizer.defaults["fused"],
                    global_update_batch_size=getattr(trainer, "global_update_batch_size", global_batch),
                    early_grad_sync=getattr(unit, "early_grad_sync", False),
                    megatron_ddp=megatron,
                )
            result = super().run_multi_trainer_with_interruption_handling(trainer)
            assert trainer.n_training_steps == total_steps
            assert len(waits) == args.steps * args.windows * len(hooks)
            assert all(w["completed"] for w in waits)
            monitor.backward_wait_observer = None
            return result

    try:
        domains = (SAETrainingDomain("dp1", (0,), 1, tuple(hooks)),) if dp == 1 else None
        runner = TraceRunner(
            cfg, override_dataset=dataset, vllm_tp_size=1 if h3 else 4,
            vllm_dp_size=0 if h3 else 1,
            sae_tp_size=tp, sae_dp_size=dp, sae_training_domains=domains,
        )
        runner.cfg.output_path = str(args.output)
        runner.run()
        assert runner.sae_runtime._closed
        assert set(dist.distributed_c10d._world.pg_map) == caller_groups
        (args.output / f"rank{rank}.json").write_text(json.dumps(dict(
            pid=os.getpid(), rank=rank, backward_waits=waits, units=unit_info,
            runtime_closed=True, owned_process_groups_released=True,
        ), indent=2))
        dist.barrier()  # Only after training/runtime cleanup; outside measurement.
        if rank == 0:
            torch.cuda.profiler.stop()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("tp1dp1", "tp2dp2"), required=True)
    parser.add_argument("--model", default="/root/models/Llama-3.1-8B")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--implicit-order", type=int, choices=(0, 1), default=1)
    parser.add_argument("--workload", choices=("h2_live", "h3_cached"), default="h2_live")
    parser.add_argument("--cache", type=Path, default=Path("results/h3_matrix_20260911/cache"))
    parser.add_argument("--dataset", default="/mnt/L202500425/dzl/datasets/wikitext2_tokenized_llama31_ctx2048")
    parser.add_argument("--pre-ga-reference", action="store_true")
    args = parser.parse_args()
    if args.warmup < 2 or args.steps < 1 or args.windows < 1:
        parser.error("warmup >= 2 and steps >= 1 required")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Requires four visible CUDA GPUs (vLLM TP4)")
    args.output = args.output.resolve()
    args.source_root = args.source_root.resolve()
    args.cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.source_root
    source_paths = list((root / "sae_lens").rglob("*.py"))
    (args.output / "invocation.json").write_text(json.dumps(dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        nccl_version=torch.cuda.nccl.version(),
        nccl_launch_order_implicit=args.implicit_order,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        devices=[torch.cuda.get_device_name(i) for i in range(4)],
        global_update_batch=2048 if args.workload == "h3_cached" else 4096,
        ga=1, hooks=3 if args.workload == "h3_cached" else 2, d_in=4096,
        d_sae=32768 if args.workload == "h3_cached" else 16384,
        k=128 if args.workload == "h3_cached" else 4,
        context=2048 if args.workload == "h3_cached" else 32,
        use_sparse_activations=args.workload != "h3_cached",
        dtype="float32", autocast=False, ddp_bucket_cap_mb=None,
        global_mixing_capacity_tokens=4096 * (2 if args.layout == "tp2dp2" else 1) if args.workload == "h3_cached" else 4096,
        mixing_fraction=0.5,
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    ), indent=2))
    world = 1 if args.workload == "h3_cached" and args.layout == "tp1dp1" else 4
    context = mp.spawn(_worker, args=(args,), nprocs=world, join=False)
    deadline = time.monotonic() + 900
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Trace run exceeded 900s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    active_ranks = [0] if args.layout == "tp1dp1" else list(range(4))
    windows = [[json.loads(line) for line in (args.output / f"step_window_profile_sae_rank{r}.jsonl").read_text().splitlines()] for r in active_ranks]
    assert all(len(ws) == args.windows for ws in windows)
    assert all(w["complete"] and w["steps"] == args.steps for ws in windows for w in ws)
    spans = [max(ws[i]["t_end_unix"] for ws in windows) - min(ws[i]["t_start_unix"] for ws in windows) for i in range(args.windows)]
    span = sum(spans)
    (args.output / "summary.json").write_text(json.dumps(dict(
        result="passed", active_ranks=active_ranks, steps=args.steps * args.windows,
        group_window_per_step_ms=[s / args.steps * 1000 for s in spans],
        wall_span_s=span, per_step_ms=span / (args.steps * args.windows) * 1000,
        global_tokens_per_s=(2048 if args.workload == "h3_cached" else 4096) * args.steps * args.windows / span,
        note=(
            "End-to-end window includes cached activation loading, routing, mixing and SAE; no live vLLM."
            if args.workload == "h3_cached" else
            "End-to-end window includes vLLM, routing, mixing and SAE."
        ) + " Only boundary CUDA syncs are added for measurement.",
    ), indent=2))


if __name__ == "__main__":
    main()
