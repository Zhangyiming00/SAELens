from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sae_lens.autoconfig import explore_vllm_activation_profile_v3 as explore
from sae_lens.autoconfig import profile_vllm_saturation_frontier_v2 as profile


def _args(tmp_path: Path, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_name="/tmp/model",
        stop_at_layer=1,
        hook_name="blocks.0.hook_resid_post",
        dtype="bfloat16",
        warmup=0,
        repeats=1,
        effective_max_model_len=17,
        gpu_memory_utilization=0.5,
        case_order="ascending",
        order_seed=123,
        no_resume=True,
        cuda_devices="0,1",
        vllm_executor_backend=backend,
        output_dir=str(tmp_path),
    )


def _case() -> explore.Case:
    return explore.Case(
        case_name="case_tp2",
        family="test",
        experiment_id="test",
        B=2,
        context_size=16,
        mbt=32,
        kv_pool_tokens=32,
        target_total_tokens=32,
    )


def _fake_base() -> SimpleNamespace:
    return SimpleNamespace(
        SCRIPT_VERSION="test_base",
        _task_hash=explore._task_hash,
        _safe_int=explore._safe_int,
    )


def test_profile_group_task_uses_torchrun_for_external_launcher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str], dict[str, Any]]] = []

    def fake_run(command, **kwargs):
        task_path = Path(command[command.index("--task-file") + 1])
        result_path = Path(command[command.index("--result-file") + 1])
        task = json.loads(task_path.read_text())
        result_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "task_hash": task["task_hash"],
                    "rows": [{"status": "ok", "case_signature": "case"}],
                }
            )
        )
        calls.append((list(command), dict(kwargs["env"]), task))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(profile.subprocess, "run", fake_run)

    rows, record = profile._run_group_task(
        base=_fake_base(),
        base_script=tmp_path / "worker.py",
        output_dir=tmp_path,
        stage="frontier",
        task_name="external_task",
        tp=2,
        mbt=32,
        pool=32,
        capture_batch_size=2,
        capture_context_size=16,
        cases=[_case()],
        args=_args(tmp_path, "external_launcher"),
    )

    command, env, task = calls[0]
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in command
    assert task["vllm_executor_backend"] == "external_launcher"
    assert task["worker_launch_mode"] == "torchrun_external_launcher"
    assert record["worker_launch_mode"] == "torchrun_external_launcher"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert rows[0]["stage"] == "frontier"


def test_profile_group_task_uses_single_worker_for_vllm_mp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str], dict[str, Any]]] = []
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")

    def fake_run(command, **kwargs):
        task_path = Path(command[command.index("--task-file") + 1])
        result_path = Path(command[command.index("--result-file") + 1])
        task = json.loads(task_path.read_text())
        result_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "task_hash": task["task_hash"],
                    "rows": [{"status": "ok", "case_signature": "case"}],
                }
            )
        )
        calls.append((list(command), dict(kwargs["env"]), task))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(profile.subprocess, "run", fake_run)

    _rows, record = profile._run_group_task(
        base=_fake_base(),
        base_script=tmp_path / "worker.py",
        output_dir=tmp_path,
        stage="frontier",
        task_name="mp_task",
        tp=2,
        mbt=32,
        pool=32,
        capture_batch_size=2,
        capture_context_size=16,
        cases=[_case()],
        args=_args(tmp_path, "mp"),
    )

    command, env, task = calls[0]
    assert command[:2] == [sys.executable, str(tmp_path / "worker.py")]
    assert "torch.distributed.run" not in command
    assert task["vllm_executor_backend"] == "mp"
    assert task["worker_launch_mode"] == "single_process_vllm_multiproc_executor"
    assert record["worker_launch_mode"] == "single_process_vllm_multiproc_executor"
    assert "RANK" not in env
    assert "WORLD_SIZE" not in env
    assert "LOCAL_RANK" not in env


def test_explore_worker_command_supports_both_tp2_launchers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_file = tmp_path / "task.json"
    result_file = tmp_path / "result.json"
    script_path = tmp_path / "explore.py"

    external_command = explore._worker_command(
        script_path=script_path,
        task_file=task_file,
        result_file=result_file,
        tp=2,
        vllm_executor_backend="external_launcher",
        no_resume=False,
    )
    mp_command = explore._worker_command(
        script_path=script_path,
        task_file=task_file,
        result_file=result_file,
        tp=2,
        vllm_executor_backend="mp",
        no_resume=False,
    )

    monkeypatch.setenv("RANK", "1")
    mp_env = explore._worker_subprocess_env(tp=2, vllm_executor_backend="mp")

    assert external_command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert mp_command[:2] == [sys.executable, str(script_path)]
    assert "torch.distributed.run" not in mp_command
    assert mp_env["SAELENS_VLLM_EXECUTOR_BACKEND"] == "mp"
    assert "RANK" not in mp_env
