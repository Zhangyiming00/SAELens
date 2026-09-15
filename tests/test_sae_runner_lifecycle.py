"""Real Gloo group ownership at runner initialization and exit boundaries."""

import json
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens import distributed_v2 as routing
from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore


def config():
    return LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(d_in=16, d_sae=32, k=4, device="cpu"),
        model_name="unused",
        model_class_name="HookedTransformer",
        device="cpu",
        act_store_device="cpu",
        hook_name="blocks.0.hook_resid_post",
        context_size=8,
        store_batch_size_prompts=1,
        train_batch_size_tokens=8,
        training_tokens=16,
        n_batches_in_buffer=2,
        n_eval_batches=0,
        logger=LoggingConfig(log_to_wandb=False),
        output_path=None,
        save_final_checkpoint=False,
        n_checkpoints=0,
    )


@pytest.mark.parametrize("failure", ["initialize", "run", "none"])
def test_runner_cleans_groups_and_preserves_callers_world(tmp_path, failure):
    dist.init_process_group(
        "gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'world'}"
    )
    before = set(dist.distributed_c10d._world.pg_map)
    try:
        if failure == "initialize":
            with (
                patch.object(
                    ActivationsStore,
                    "from_config",
                    side_effect=RuntimeError("store failed"),
                ),
                pytest.raises(RuntimeError, match="store failed"),
            ):
                LanguageModelSAETrainingRunner(config(), override_model=MagicMock())
        else:
            with patch.object(
                ActivationsStore, "from_config", return_value=MagicMock()
            ):
                runner = LanguageModelSAETrainingRunner(
                    config(), override_model=MagicMock()
                )
            assert len(dist.distributed_c10d._world.pg_map) > len(before)
            if failure == "run":
                with (
                    patch.object(
                        runner, "_run", side_effect=RuntimeError("trainer failed")
                    ),
                    pytest.raises(RuntimeError, match="trainer failed"),
                ):
                    runner.run()
            else:
                with patch.object(runner, "_run", return_value=runner.sae):
                    assert runner.run() is runner.sae
            runner.close()
            assert runner.sae_runtime._closed
        assert not routing._initialized and routing.get_sae_runtime() is None
        assert dist.is_initialized()
        assert set(dist.distributed_c10d._world.pg_map) == before
    finally:
        dist.destroy_process_group()


def test_single_process_runner_initializes_and_releases_its_world():
    assert not dist.is_initialized()
    with patch.object(ActivationsStore, "from_config", return_value=MagicMock()):
        runner = LanguageModelSAETrainingRunner(config(), override_model=MagicMock())
    assert dist.is_initialized() and runner.sae_runtime is not None
    runner.close()
    assert not dist.is_initialized()


def test_removed_static_routing_flag_fails_before_creating_groups():
    with pytest.raises(ValueError, match="has been removed"):
        LanguageModelSAETrainingRunner(config(), use_shard_routing=False)
    assert not dist.is_initialized()


def _mixed_failure_worker(rank, rendezvous, output, failed_rank, phase):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=15),
    )
    before = set(dist.distributed_c10d._world.pg_map)
    try:
        with patch.object(ActivationsStore, "from_config", return_value=MagicMock()):
            runner = LanguageModelSAETrainingRunner(
                config(),
                override_model=MagicMock(),
                vllm_tp_size=4,
                sae_tp_size=2,
                sae_dp_size=1,
            )

        def fail_or_wait():
            monitor = runner.sae_runtime.failure_monitor
            if phase == "backward" and rank < 2:
                domain = runner.sae_runtime.require_local().domain
                for _ in range(3):
                    monitor.complete_backward(domain, "test_hook")
            if rank == failed_rank:
                # Peers have time to enter their actual helper command wait.
                time.sleep(0.1)
                raise RuntimeError(f"injected rank {rank}")
            if rank >= 2:
                return runner._run_producer_helper_loop_v2()
            if phase == "backward":
                monitor.complete_backward(domain, "test_hook")
                raise AssertionError("A failed peer must prevent the next update")
            while True:
                monitor.check()
                time.sleep(0.01)

        started = time.monotonic()
        with (
            patch.object(runner, "_run", side_effect=fail_or_wait),
            pytest.raises(RuntimeError, match="injected rank"),
        ):
            runner.run()
        assert set(dist.distributed_c10d._world.pg_map) == before
        assert not routing._initialized
        Path(output, f"rank{rank}.json").write_text(
            json.dumps(
                dict(
                    elapsed=time.monotonic() - started,
                    error=runner.sae_runtime.failure_monitor.error,
                )
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "failed_rank,phase", [(0, "command"), (2, "command"), (0, "backward")]
)
def test_mixed_roles_receive_failure_and_exit(tmp_path, failed_rank, phase):
    mp.spawn(
        _mixed_failure_worker,
        args=(f"file://{tmp_path / 'world'}", str(tmp_path), failed_rank, phase),
        nprocs=4,
    )
    reports = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(4)]
    assert max(r["elapsed"] for r in reports) < 10
    assert len({r["error"] for r in reports}) == 1
