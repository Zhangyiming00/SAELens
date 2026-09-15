"""Fixed synchronous vLLM -> routing -> SAE baseline, with opt-in wait timing.

No per-step CUDA synchronization, input hashing, or checkpoint I/O in the
measurement windows. All failure guards remain enabled in both timing modes.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, args):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4",
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29572",
        SAE_ADAM_IMPL="forloop", VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", rank=rank, world_size=4,
        init_method=f"file://{args.output / 'world'}",
        timeout=timedelta(seconds=180),
    )
    caller_groups = set(dist.distributed_c10d._world.pg_map)
    from datasets import Dataset

    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    hooks = [f"blocks.{i}.hook_resid_post" for i in (21, 26)[:args.hooks]]
    total_steps = args.warmup + args.window_steps * args.windows
    dataset = Dataset.from_dict({"tokens": [
        [1000] + [1001 + ((i * 37 + j) % 1999) for j in range(31)]
        for i in range(1024)
    ]})
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=4096, d_sae=args.d_sae, k=4, device=f"cuda:{rank}",
            use_sparse_activations=True, normalize_activations="none",
        ),
        model_name=args.model, model_class_name="VLLMModel",
        model_from_pretrained_kwargs=dict(
            tensor_parallel_size=4, max_model_len=33,
            gpu_memory_utilization=0.35, enforce_eager=True,
        ),
        hook_name=hooks[0], hook_names=hooks if args.hooks > 1 else None,
        dataset_path="local-static-benchmark", is_dataset_tokenized=True,
        streaming=False, context_size=32,
        store_batch_size_prompts=192, n_batches_in_buffer=64,
        # Like the CLI, translate equal-mode global counts to local DP counts
        # before constructing the runner (exact mode has a different contract).
        training_tokens=(args.global_batch // 2) * total_steps,
        train_batch_size_tokens=args.global_batch // 2, routing_dp_batch_mode="equal",
        activations_mixing_fraction=0.5, device=f"cuda:{rank}",
        act_store_device=f"cuda:{rank}", dtype="float32",
        exclude_special_tokens=[1000], prepend_bos=False, n_eval_batches=0,
        autocast=False, sae_dp_mode="ddp", lr=3e-4, lr_end=3e-5,
        lr_scheduler_name="cosineannealing", n_checkpoints=0,
        save_final_checkpoint=False, output_path=str(args.output),
        checkpoint_path=str(args.output / "unused_checkpoints"),
        save_mse_every_n_steps=0, save_timing_every_n_steps=0,
        save_memory_every_n_steps=0, logger=LoggingConfig(log_to_wandb=False),
        step_window_profile_start_step=args.warmup + 1,
        step_window_profile_window_steps=args.window_steps,
        step_window_profile_window_count=args.windows,
    )
    waits, refills, generations = [], [], []

    class BenchmarkRunner(LanguageModelSAETrainingRunner):
        def benchmark(self, trainer, run):
            self.audited_trainer = trainer
            monitor = self.sae_runtime.failure_monitor

            def observe(event):
                step = trainer.n_training_steps + 1
                if step > args.warmup:
                    waits.append({"step": step, **event})

            if args.observe_backward:
                monitor.backward_wait_observer = observe
            result = run(trainer)
            assert trainer.n_training_steps == total_steps
            monitor.backward_wait_observer = None
            return result

        def run_trainer_with_interruption_handling(self, trainer):
            return self.benchmark(trainer, super().run_trainer_with_interruption_handling)

        def run_multi_trainer_with_interruption_handling(self, trainer):
            return self.benchmark(trainer, super().run_multi_trainer_with_interruption_handling)

    try:
        runner = BenchmarkRunner(
            cfg, override_dataset=dataset, vllm_tp_size=4, sae_tp_size=2, sae_dp_size=2,
        )
        runner.cfg.output_path = str(args.output)
        refill = runner.activations_store._synchronized_serving_batches
        generate = runner.model.run_with_cache

        def step_number():
            return runner.audited_trainer.n_training_steps + 1

        def measured_refill(count):
            started = time.perf_counter()
            agreed = refill(count)
            elapsed = time.perf_counter() - started
            if step_number() > args.warmup:
                refills.append(dict(step=step_number(), elapsed_s=elapsed,
                                    proposed=count, agreed=agreed))
            return agreed

        def counted_generate(*a, **kw):
            if step_number() > args.warmup:
                generations.append(step_number())
            return generate(*a, **kw)

        runner.activations_store._synchronized_serving_batches = measured_refill
        runner.model.run_with_cache = counted_generate
        runner.run()
        assert runner.sae_runtime._closed
        assert set(dist.distributed_c10d._world.pg_map) == caller_groups
        if args.observe_backward:
            assert len(waits) == args.window_steps * args.windows * args.hooks
            assert all(w["completed"] for w in waits)
        (args.output / f"rank{rank}.json").write_text(json.dumps(dict(
            backward_waits=waits, buffer_consensus=refills,
            generation_steps=generations, runtime_closed=True,
            owned_process_groups_released=True,
        ), indent=2))
    finally:
        dist.destroy_process_group()


def describe(values):
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values),
                p50=statistics.median(values), p95=ordered[int((len(ordered)-1)*0.95)],
                minimum=ordered[0], maximum=ordered[-1]) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/root/models/Llama-3.1-8B")
    parser.add_argument("--hooks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--global-batch", type=int, default=4096)
    parser.add_argument("--d-sae", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--window-steps", type=int, default=16)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--observe-backward", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.global_batch % 2 or not 2 <= args.global_batch <= 4096:
        parser.error("global batch must be even and between 2 and 4096")
    if args.warmup < 1 or args.window_steps < 1 or args.windows < 1:
        parser.error("warmup, window steps and windows must be positive")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Requires four visible CUDA GPUs")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "scripts/benchmark_sae_runner_static.py", "sae_lens/config.py",
        "sae_lens/llm_sae_training_runner.py", "sae_lens/sae_runtime.py",
        "sae_lens/distributed_v2.py", "sae_lens/static_failure.py",
        "sae_lens/training/sae_train_unit.py", "sae_lens/training/sae_trainer.py",
        "sae_lens/training/multi_sae_trainer.py", "sae_lens/training/mixing_buffer.py",
        "sae_lens/training/activations_store.py", "sae_lens/training/step_window_profiler.py",
        "sae_lens/saes/megatron_topk_sae.py", "sae_lens/vllm_model.py",
    ]
    (args.output / "invocation.json").write_text(json.dumps(dict(
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        devices=[torch.cuda.get_device_name(i) for i in range(4)],
        sae_tp=2, sae_dp=2, vllm_tp=4, dtype="float32", autocast=False,
        source_sha256={p:hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths},
    ), indent=2))
    context = mp.spawn(_worker, args=(args,), nprocs=4, join=False)
    deadline = time.monotonic() + 900
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark exceeded 900s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    ranks = [json.loads((args.output / f"rank{r}.json").read_text()) for r in range(4)]
    windows = [[json.loads(line) for line in (
        args.output / f"step_window_profile_sae_rank{rank}.jsonl"
    ).read_text().splitlines()] for rank in range(4)]
    assert all(len(w) == args.windows for w in windows)
    assert all(w["complete"] and w["steps"] == args.window_steps for ws in windows for w in ws)
    group_windows = []
    for index in range(args.windows):
        span = max(w[index]["t_end_unix"] for w in windows) - min(w[index]["t_start_unix"] for w in windows)
        group_windows.append(dict(window=index+1, wall_span_s=span,
                                  per_step_s=span/args.window_steps,
                                  global_tokens_per_s=args.global_batch*args.window_steps/span))
    by_hook = {}
    for hook in sorted({w["hook"] for r in ranks for w in r["backward_waits"]}):
        entries = [w for r in ranks for w in r["backward_waits"] if w["hook"] == hook]
        by_hook[hook] = dict(
            elapsed_s=describe([w["elapsed_s"] for w in entries]),
            sleep_s=describe([w["sleep_s"] for w in entries]),
            poll_sleeps=describe([w["poll_sleeps"] for w in entries]),
            no_sleep_calls=sum(w["poll_sleeps"] == 0 for w in entries),
        )
    summary = dict(
        result="passed", group_windows=group_windows,
        global_per_step_s=describe([w["per_step_s"] for w in group_windows]),
        backward_wait_by_hook=by_hook,
        per_rank=[dict(rank=i,
                       window_wall_s=sum(w["window_time_s"] for w in windows[i]),
                       backward_wait_s=sum(w["elapsed_s"] for w in r["backward_waits"]),
                       buffer_consensus_s=sum(w["elapsed_s"] for w in r["buffer_consensus"]),
                       generations=len(r["generation_steps"])) for i,r in enumerate(ranks)],
        notes=["CUDA synchronization occurs only at window boundaries; window wall time includes generation, routing, buffer, training and loop overhead.",
               "Host backward readiness wait is included in the backward/SAE time, not an additive GPU phase or a measured counterfactual slowdown.",
               "Sleep counts and measured sleep durations describe polling; 5ms is not charged to calls that do not sleep.",
               "Both observer modes keep the failure handshake and its 5ms interval enabled; the observer adds optional CPU measurements.",
               "Uniform filtering is intentional for a repeatable performance baseline; uneven filtering is covered by separate correctness acceptance."],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
