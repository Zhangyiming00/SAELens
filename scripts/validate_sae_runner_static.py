"""Four-GPU acceptance through the real vLLM runner, mixer, and checkpoint APIs.

No activations or model outputs are substituted. Each worker starts a fresh
runner; baseline and resume must see identical mixed training batches.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def assert_state_equal(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def run_workers(args, timeout):
    context = mp.spawn(_worker, args=args, nprocs=4, join=False)
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Runner workers did not exit within {timeout}s")
    finally:
        # A watchdog intervention is a failed test, never a successful exit.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def _worker(
    rank,
    rendezvous,
    directory,
    model_path,
    hook_count,
    resume,
    empty_steps=False,
    layout="full",
    batch_mode="equal",
    global_batch=32,
    placement_amp=False,
    failure=None,
    uneven_filter=False,
):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="4",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29571",
        SAE_ADAM_IMPL="forloop",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=180),
    )
    caller_groups = set(dist.distributed_c10d._world.pg_map)
    from datasets import Dataset
    from torch.utils._pytree import tree_map

    from sae_lens import distributed_v2
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
    from sae_lens.sae_runtime import SAETrainingDomain
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

    directory = Path(directory)
    phase = "resume" if resume else "continuous"
    # The equal runner expects local batch/token counts, unlike exact mode.
    # For the uneven-filter regression reproduce precisely local batch 16.
    cfg_batch = global_batch // 2 if uneven_filter and batch_mode == "equal" else global_batch
    checkpoint_tokens = cfg_batch * 3
    hooks = [f"blocks.{i}.hook_resid_post" for i in (21, 26)[:hook_count]]
    # Pretokenized, deterministic local rows exercise vLLM prefill and real
    # activation extraction without network or dataset download variability.
    dataset = Dataset.from_dict(
        {
            "tokens": [
                ([1000] if not uneven_filter or i % 4 == 2 else [1001])
                + [1001 + ((i * 37 + j) % 1999) for j in range(31)]
                for i in range(256)
            ]
        }
    )
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=4096,
            d_sae=64,
            k=4,
            device=f"cuda:{rank}",
            use_sparse_activations=True,
            normalize_activations="none",
        ),
        model_name=model_path,
        model_class_name="VLLMModel",
        model_from_pretrained_kwargs=dict(
            tensor_parallel_size=4,
            max_model_len=33,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
        ),
        hook_name=hooks[0],
        hook_names=hooks if hook_count > 1 else None,
        dataset_path="local-static-acceptance",
        is_dataset_tokenized=True,
        streaming=False,
        context_size=32,
        store_batch_size_prompts=192 if global_batch == 4096 else 4,
        n_batches_in_buffer=64
        if global_batch == 4096
        else (2 if batch_mode == "exact" or uneven_filter else 4),
        training_tokens=cfg_batch * 6,
        train_batch_size_tokens=cfg_batch,
        routing_dp_batch_mode=batch_mode,
        activations_mixing_fraction=0.5,
        device=f"cuda:{rank}",
        act_store_device=f"cuda:{rank}",
        dtype="float32",
        exclude_special_tokens=[1000],
        prepend_bos=False,
        n_eval_batches=0,
        autocast=empty_steps or placement_amp,
        sae_dp_mode="ddp",
        lr=3e-4,
        lr_end=3e-5,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_checkpoints=2,
        checkpoint_path=str(directory / "checkpoints"),
        save_final_checkpoint=False,
        output_path=None,
        save_mse_every_n_steps=0,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=0,
        logger=LoggingConfig(log_to_wandb=False),
        resume_from_checkpoint=str(directory / "checkpoints" / str(checkpoint_tokens))
        if resume
        else None,
    )

    class AuditRunner(LanguageModelSAETrainingRunner):
        def audit_train(self, trainer, run):
            self.audited_trainer = trainer
            self.steps = []
            if failure == "consumer_wait" and rank == 0:
                raise RuntimeError("injected consumer waiting boundary")
            if failure == "consumer_backward" and rank == 0:

                def fail_backward(_grad):
                    raise RuntimeError("injected consumer backward")

                next(iter(trainer.optimizer.param_groups[0]["params"])).register_hook(
                    fail_backward
                )
            if placement_amp and not resume:
                trainer.grad_scaler = torch.amp.GradScaler(
                    "cuda", init_scale=64, growth_interval=2
                )
            if placement_amp and runtime.local.domain.name == "second":

                def overflow(grad):
                    return (
                        torch.full_like(grad, float("inf"))
                        if trainer.n_training_steps == 1
                        else grad
                    )

                trainer.optimizer.param_groups[0]["params"][0].register_hook(overflow)
            train_step = trainer._train_step
            if empty_steps:
                provider = trainer.data_provider

                class EmptyBatchProbe:
                    def __getattr__(self, name):
                        return getattr(provider, name)

                    def __next__(self):
                        batch = next(provider)
                        index = trainer.n_training_steps
                        empty = index in (2, 3) or (
                            index == 1 and runner.sae_runtime.local.dp_rank == 0
                        )
                        if empty:
                            return (
                                {h: t[:0] for h, t in batch.items()}
                                if isinstance(batch, dict)
                                else batch[:0]
                            )
                        return batch

                trainer.data_provider = EmptyBatchProbe()

            def audited_step(*args, **kwargs):
                if trainer.n_training_steps == 3:
                    models = getattr(
                        trainer,
                        "base_sae_by_hook",
                        {hooks[0]: getattr(trainer, "_base_sae", None)},
                    )
                    torch.save(
                        tree_map(
                            lambda x: x.detach().cpu() if torch.is_tensor(x) else x,
                            dict(
                                models={h: m.state_dict() for h, m in models.items()},
                                optimizer=trainer.optimizer.state_dict(),
                                scheduler=trainer.lr_scheduler.state_dict(),
                                scaler=trainer.grad_scaler.state_dict(),
                                n_training_samples=trainer.n_training_samples,
                                n_training_steps=trainer.n_training_steps,
                                token_count_remainder=getattr(
                                    trainer, "_token_count_remainder", 0
                                ),
                            ),
                        ),
                        directory / f"{phase}_checkpoint_entry_rank{rank}.pt",
                    )
                batch = kwargs["sae_in"] if "sae_in" in kwargs else args[0]
                if not isinstance(batch, dict):
                    batch = {hooks[0]: batch}
                hashes = {
                    h: hashlib.sha256(
                        t.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    for h, t in batch.items()
                }
                rates = [g["lr"] for g in trainer.optimizer.param_groups]
                scaler_at_start = trainer.grad_scaler.state_dict()
                if empty_steps and trainer.n_training_steps in (2, 3):
                    import copy

                    before = copy.deepcopy(trainer.optimizer.state_dict())
                    scaler_before = copy.deepcopy(trainer.grad_scaler.state_dict())
                result = train_step(*args, **kwargs)
                if empty_steps and trainer.n_training_steps in (2, 3):
                    assert_state_equal(before, trainer.optimizer.state_dict())
                    assert_state_equal(scaler_before, trainer.grad_scaler.state_dict())
                self.steps.append(
                    dict(
                        batch=hashes,
                        learning_rates=rates,
                        local_rows=len(next(iter(batch.values()))),
                        scaler=scaler_at_start,
                    )
                )
                return result

            trainer._train_step = audited_step
            result = run(trainer)
            models = getattr(
                trainer,
                "base_sae_by_hook",
                {hooks[0]: getattr(trainer, "_base_sae", None)},
            )
            state = dict(
                models={h: m.state_dict() for h, m in models.items()},
                optimizer=trainer.optimizer.state_dict(),
                scheduler=trainer.lr_scheduler.state_dict(),
                n_training_steps=trainer.n_training_steps,
                n_training_samples=trainer.n_training_samples,
                token_count_remainder=getattr(trainer, "_token_count_remainder", 0),
                scaler=trainer.grad_scaler.state_dict(),
            )
            torch.save(
                tree_map(
                    lambda x: x.detach().cpu() if torch.is_tensor(x) else x, state
                ),
                directory / f"{phase}_rank{rank}.pt",
            )
            return result

        def run_trainer_with_interruption_handling(self, trainer):
            return self.audit_train(
                trainer, super().run_trainer_with_interruption_handling
            )

        def run_multi_trainer_with_interruption_handling(self, trainer):
            return self.audit_train(
                trainer, super().run_multi_trainer_with_interruption_handling
            )

    try:
        domains = None
        if layout == "prefix":
            domains = (SAETrainingDomain("prefix", (0, 1), 2, tuple(hooks)),)
        elif layout == "placement":
            domains = (
                SAETrainingDomain("first", (0, 1), 2, (hooks[0],)),
                SAETrainingDomain("second", (2, 3), 2, (hooks[1],)),
            )
        elif layout == "dp3":
            domains = (SAETrainingDomain("dp3", (0, 1, 2), 1, tuple(hooks)),)
        runner = AuditRunner(
            cfg,
            override_dataset=dataset,
            vllm_tp_size=4,
            sae_tp_size=1 if layout == "dp3" else 2,
            sae_dp_size=3 if layout == "dp3" else (2 if layout == "full" else 1),
            sae_pp_size=2 if layout == "placement" else 1,
            sae_training_domains=domains,
        )
        runner.steps = []
        # The runner appends its run ID to checkpoint_path. Use a fixed shared
        # directory after construction for cross-process resume verification.
        runner.cfg.checkpoint_path = str(directory / "checkpoints")
        generated = 0
        filtered_rows = []
        generate = runner.model.run_with_cache

        def counted_generate(*args, **kwargs):
            nonlocal generated
            generated += 1
            if failure == "producer_generate" and rank == 3:
                raise RuntimeError("injected producer generation")
            return generate(*args, **kwargs)

        runner.model.run_with_cache = counted_generate
        runtime = runner.sae_runtime
        produce = runner.activations_store._produce_one_v2_assembled_batch

        def checked_produce():
            batch = produce()
            if batch is not None:
                sample = (
                    next(iter(batch.values())) if isinstance(batch, dict) else batch
                )
                from sae_lens.shard_routing import routes_for_consumer

                routes = routes_for_consumer(
                    distributed_v2.get_routing_table(), runtime.local.dp_rank
                )
                expected = sum(
                    sum(
                        not (i % 32 == 0 and (not uneven_filter or i // 32 % 4 == 2))
                        for i in range(r.row_start, r.row_end)
                    )
                    for r in routes
                )
                assert sample.shape == (
                    expected,
                    4096,
                ), "token exclusion or TP input shape differs"
                filtered_rows.append(len(sample))
            return batch

        runner.activations_store._produce_one_v2_assembled_batch = checked_produce
        if failure:
            import faulthandler

            trace_file = (directory / f"failure_stack_rank{rank}.log").open("w")
            faulthandler.dump_traceback_later(20, file=trace_file)
            started = time.monotonic()
            try:
                runner.run()
            except BaseException as exc:
                assert runtime.failure_monitor.error is not None
                (directory / f"failure_rank{rank}.json").write_text(
                    json.dumps(
                        dict(
                            error=str(exc),
                            first_failure=json.loads(runtime.failure_monitor.error),
                            elapsed_s=time.monotonic() - started,
                            runtime_closed=runtime._closed,
                            producer_only=runtime.local is None,
                            owned_process_groups_released=set(
                                dist.distributed_c10d._world.pg_map
                            )
                            == caller_groups,
                        ),
                        indent=2,
                    )
                )
            else:
                raise AssertionError("Injected failure returned success")
            finally:
                faulthandler.cancel_dump_traceback_later()
                trace_file.close()
            return
        runner.run()
        assert runtime._closed and not distributed_v2._initialized
        assert dist.is_initialized(), "runner destroyed caller-owned world"
        from vllm.distributed import parallel_state as vllm_state

        assert vllm_state._WORLD is None and vllm_state._TP is None
        assert set(dist.distributed_c10d._world.pg_map) == caller_groups
        runner.close()
        (directory / f"{phase}_rank{rank}.json").write_text(
            json.dumps(
                dict(
                    steps=runner.steps,
                    vllm_generations=generated,
                    runtime_closed=True,
                    producer_only=runtime.local is None,
                    filtered_batch_rows=filtered_rows,
                    owned_process_groups_released=True,
                ),
                indent=2,
            )
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/root/models/Llama-3.1-8B")
    parser.add_argument("--hooks", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument(
        "--layout", choices=("full", "prefix", "placement", "dp3"), default="full"
    )
    parser.add_argument("--batch-mode", choices=("equal", "exact"), default="equal")
    parser.add_argument("--global-batch", type=int, choices=(32, 4096), default=32)
    parser.add_argument("--placement-amp", action="store_true")
    parser.add_argument("--uneven-filter", action="store_true")
    parser.add_argument(
        "--failure", choices=("consumer_wait", "producer_generate", "consumer_backward")
    )
    parser.add_argument(
        "--empty-steps",
        action="store_true",
        help="After the real mixer, inject one empty replica and two global empty batches into the runner loop",
    )
    args = parser.parse_args()
    if args.layout == "placement" and 1 in args.hooks:
        parser.error("placement requires --hooks 2")
    if args.empty_steps and args.layout != "full":
        parser.error("empty-step acceptance uses the full TP2 x DP2 layout")
    if args.layout == "dp3" and args.batch_mode != "exact":
        parser.error("DP3 acceptance requires --batch-mode exact")
    if args.uneven_filter and (args.layout != "full" or args.global_batch != 32):
        parser.error("uneven filtering requires the full layout and global batch 32")
    if args.placement_amp and args.layout != "placement":
        parser.error("placement AMP requires --layout placement --hooks 2")
    if args.failure and (
        args.layout != "prefix" or args.empty_steps or args.placement_amp
    ):
        parser.error(
            "failure injection requires the prefix layout without other injections"
        )
    if torch.cuda.device_count() < 4:
        raise RuntimeError("Requires four CUDA GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        "run_sae_runner_gpu.py",
        "sae_lens/llm_sae_training_runner.py",
        "sae_lens/vllm_model.py",
        "sae_lens/distributed.py",
        "sae_lens/distributed_v2.py",
        "sae_lens/megatron_tp.py",
        "sae_lens/sae_runtime.py",
        "sae_lens/static_failure.py",
        "sae_lens/training/sae_trainer.py",
        "sae_lens/training/multi_sae_trainer.py",
        "sae_lens/training/runtime_checkpoint.py",
        "sae_lens/training/activations_store.py",
        "sae_lens/training/mixing_buffer.py",
        "sae_lens/training/sae_train_unit.py",
        "sae_lens/training/optimizer_checkpoint.py",
        "sae_lens/saes/megatron_topk_sae.py",
        "sae_lens/config.py",
        "scripts/validate_sae_runner_static.py",
    ]
    (args.output / "invocation.json").write_text(
        json.dumps(
            dict(
                arguments={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                devices=[torch.cuda.get_device_name(i) for i in range(4)],
                source_sha256={
                    p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                    for p in sources
                },
            ),
            indent=2,
        )
    )

    summary = []
    for hooks in args.hooks:
        directory = args.output / f"h{hooks}"
        directory.mkdir()
        for resume in (False,) if args.empty_steps or args.failure else (False, True):
            run_workers(
                args=(
                    f"file://{directory.resolve() / ('resume_rdzv' if resume else 'continuous_rdzv')}",
                    str(directory.resolve()),
                    args.model,
                    hooks,
                    resume,
                    args.empty_steps,
                    args.layout,
                    args.batch_mode,
                    args.global_batch,
                    args.placement_amp,
                    args.failure,
                    args.uneven_filter,
                ),
                timeout=120 if args.failure else 600,
            )
        if args.failure:
            reports = [
                json.loads((directory / f"failure_rank{r}.json").read_text())
                for r in range(4)
            ]
            assert all(
                r["runtime_closed"] and r["owned_process_groups_released"]
                for r in reports
            )
            assert max(r["elapsed_s"] for r in reports) < 60
            assert (
                len({json.dumps(r["first_failure"], sort_keys=True) for r in reports})
                == 1
            )
            summary.append(
                dict(
                    hooks=hooks,
                    layout=args.layout,
                    failure=args.failure,
                    result="passed",
                    all_ranks_exited=True,
                    reports=reports,
                )
            )
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            continue
        if args.empty_steps:
            for rank in range(2):
                a = torch.load(
                    directory / f"continuous_rank{rank}.pt", weights_only=True
                )
                b = torch.load(
                    directory / f"continuous_rank{rank + 2}.pt", weights_only=True
                )
                assert_state_equal(a, b)
                assert a["n_training_steps"] == 9
                assert all(
                    int(s["step"].item()) == 7 for s in a["optimizer"]["state"].values()
                )
            summary.append(
                dict(
                    hooks=hooks,
                    tp=2,
                    dp=2,
                    result="passed",
                    empty_replica=True,
                    global_empty_steps=2,
                    enabled_scaler=True,
                )
            )
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            continue
        resume_bitwise_equal = True
        maximum_errors = {"models": 0.0, "optimizer": 0.0}
        for rank in range(4):
            before = json.loads((directory / f"continuous_rank{rank}.json").read_text())
            after = json.loads((directory / f"resume_rank{rank}.json").read_text())
            if before["producer_only"]:
                assert after["producer_only"] and before["vllm_generations"] > 0
                continue
            baseline = torch.load(
                directory / f"continuous_rank{rank}.pt", weights_only=True
            )
            restored = torch.load(
                directory / f"resume_rank{rank}.pt", weights_only=True
            )
            assert_state_equal(
                torch.load(
                    directory / f"continuous_checkpoint_entry_rank{rank}.pt",
                    weights_only=True,
                ),
                torch.load(
                    directory / f"resume_checkpoint_entry_rank{rank}.pt",
                    weights_only=True,
                ),
            )
            if args.layout == "dp3":
                # DDP reconstructs its reducer buckets on a fresh process. With
                # DP3, the summation order can differ after resume; checkpoint
                # state and input/LR are still required to be bitwise identical.
                def compare_tensors(a, b, section):
                    nonlocal resume_bitwise_equal
                    if isinstance(a, torch.Tensor):
                        error = float((a - b).abs().max()) if a.numel() else 0.0
                        maximum_errors[section] = max(maximum_errors[section], error)
                        resume_bitwise_equal &= torch.equal(a, b)
                    elif isinstance(a, dict):
                        for key in a:
                            compare_tensors(a[key], b[key], section)
                    elif isinstance(a, (tuple, list)):
                        for x, y in zip(a, b, strict=True):
                            compare_tensors(x, y, section)

                for section in maximum_errors:
                    compare_tensors(baseline[section], restored[section], section)
                    torch.testing.assert_close(
                        baseline[section], restored[section], rtol=1e-6, atol=1e-7
                    )
                assert_state_equal(
                    {k: v for k, v in baseline.items() if k not in maximum_errors},
                    {k: v for k, v in restored.items() if k not in maximum_errors},
                )
            else:
                assert_state_equal(baseline, restored)
            assert len(before["steps"]) == 6 and len(after["steps"]) == 3
            assert before["steps"][3:] == after["steps"]
            assert before["vllm_generations"] > 0 and after["vllm_generations"] > 0
            if args.batch_mode == "exact":
                from sae_lens.training.dp_batch import balanced_token_counts

                dp = 3 if args.layout == "dp3" else (2 if args.layout == "full" else 1)
                dp_index = (
                    rank
                    if args.layout == "dp3"
                    else (rank // 2 if args.layout == "full" else 0)
                )
                assert all(
                    s["local_rows"]
                    == balanced_token_counts(args.global_batch, dp)[dp_index]
                    for s in before["steps"]
                )
        if args.placement_amp:
            a = torch.load(
                directory
                / "checkpoints"
                / str(args.global_batch * 3)
                / "placement_state_0.pt",
                weights_only=True,
            )
            b = torch.load(
                directory
                / "checkpoints"
                / str(args.global_batch * 3)
                / "placement_state_1.pt",
                weights_only=True,
            )
            assert a["grad_scaler"]["scale"] != b["grad_scaler"]["scale"]
        if args.uneven_filter:
            for phase in ("continuous", "resume"):
                replicas = [
                    json.loads((directory / f"{phase}_rank{rank}.json").read_text())
                    for rank in (0, 2)
                ]
                assert all(x == 64 for x in replicas[0]["filtered_batch_rows"])
                assert all(x == 63 for x in replicas[1]["filtered_batch_rows"])
                assert all(
                    step["local_rows"] == 16
                    for replica in replicas for step in replica["steps"]
                )
                assert len(replicas[0]["filtered_batch_rows"]) == len(
                    replicas[1]["filtered_batch_rows"]
                )
        summary.append(
            dict(
                hooks=hooks,
                tp=1 if args.layout == "dp3" else 2,
                dp=3 if args.layout == "dp3" else (2 if args.layout == "full" else 1),
                layout=args.layout,
                result="passed",
                resume_bitwise_equal=resume_bitwise_equal,
                resume_max_abs_errors=maximum_errors,
                checkpoint_state_bitwise_equal=True,
                resumed_inputs_lr_scaler_equal=True,
                routing_dp_batch_mode=args.batch_mode,
                configured_global_batch=args.global_batch
                if args.batch_mode == "exact" or args.uneven_filter
                else None,
                placement_amp=args.placement_amp,
                uneven_filter=args.uneven_filter,
            )
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
