"""
Topology supervisor for streaming vLLM + SAE training.

Keeps a lightweight control plane alive across topology switches. GPU workers
(torchrun process groups) are disposable — torn down and relaunched on each
topology change. The /dev/shm buffer survives across restarts.

Usage:
    python3 scripts/topology_supervisor.py \\
        --run-dir /checkpoints/my_run \\
        --worker-script scripts/run_sae_runner_gpu.py \\
        --worker-args "--model-name /data/Llama-3.1-8B --streaming-mode ..." \\
        --vllm-tp 2 --vllm-dp 2 --sae-tp 1 \\
        --num-gpus 8

To request a topology switch while the run is live, write:
    echo '{"vllm_tp": 2, "vllm_dp": 4, "sae_tp": 2}' > /checkpoints/my_run/topology_request.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Supervisor polls these files at this interval (seconds).
_POLL_INTERVAL_S = 5.0
# After quiesce acks received, wait this long for workers to exit before SIGKILL.
_WORKER_EXIT_TIMEOUT_S = 60.0


def _run_dir_markers(run_dir: Path) -> tuple[str, ...]:
    markers = {str(run_dir)}
    try:
        markers.add(str(run_dir.resolve()))
    except Exception:
        pass
    return tuple(marker for marker in markers if marker)


def _active_topology_process_pids(
    ps_lines: list[str],
    run_dir: Path,
    *,
    current_pid: int | None = None,
) -> list[int]:
    markers = _run_dir_markers(run_dir)
    process_markers = (
        "scripts/topology_supervisor.py",
        "scripts/run_sae_runner_gpu.py",
        "torch.distributed.run",
    )
    pids: list[int] = []
    for line in ps_lines:
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[2]
        if pid == current_pid:
            continue
        if (
            any(marker in command for marker in markers)
            and any(marker in command for marker in process_markers)
        ):
            pids.append(pid)
    return sorted(set(pids))


def _pid_is_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    except FileNotFoundError:
        return False
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_existing_run_processes(run_dir: Path, timeout_s: float = 15.0) -> None:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[supervisor] could not inspect process table: {result.stderr.strip()}")
        return
    pids = _active_topology_process_pids(
        result.stdout.splitlines(),
        run_dir,
        current_pid=os.getpid(),
    )
    if not pids:
        return
    print(f"[supervisor] stopping existing topology processes for {run_dir}: {pids}")
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_is_alive(pid)]
        if not alive:
            return
        time.sleep(0.2)
    alive = [pid for pid in pids if _pid_is_alive(pid)]
    if alive:
        print(f"[supervisor] force-killing existing topology processes: {alive}")
        for pid in sorted(alive, reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@dataclass(frozen=True)
class StartupMemoryPlan:
    shm_path: Path
    shm_total_bytes: int
    shm_free_bytes: int
    buffer_bytes: int
    checkpoint_bytes: int
    required_bytes: int
    d_in: int
    d_sae: int
    num_hooks: int
    dtype: str
    checkpoint_storage: str


# ---------------------------------------------------------------------------
# Switch event logger
# ---------------------------------------------------------------------------

class SwitchLogger:
    """Append-only JSONL log of topology switch events with wall-time and elapsed time."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._t0 = time.time()
        self._switch_count = 0

    def _write(self, event: str, **fields: Any) -> None:
        now = time.time()
        record = {
            "event": event,
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "elapsed_s": round(now - self._t0, 3),
            **fields,
        }
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[topo-log] {record}")

    def supervisor_start(self, topology: dict, run_dir: str) -> None:
        self._write("supervisor_start", topology=topology, run_dir=run_dir)

    def switch_triggered(
        self,
        trigger: str,
        from_topology: dict,
        to_topology: dict,
        buffer_state: dict | None,
    ) -> None:
        self._switch_count += 1
        self._write(
            "switch_triggered",
            switch_index=self._switch_count,
            trigger=trigger,
            from_topology=from_topology,
            to_topology=to_topology,
            buffer_state=buffer_state,
        )

    def quiesce_signaled(self, from_topology: dict) -> None:
        self._write("quiesce_signaled", from_topology=from_topology)

    def all_acks_received(self, from_topology: dict) -> None:
        self._write("all_acks_received", from_topology=from_topology)

    def workers_exited(self, exit_code: int | None, from_topology: dict) -> None:
        self._write("workers_exited", exit_code=exit_code, from_topology=from_topology)

    def checkpoint_found(self, checkpoint_path: str | None) -> None:
        self._write("checkpoint_found", checkpoint_path=checkpoint_path)

    def new_workers_launched(self, topology: dict, pid: int) -> None:
        self._write("new_workers_launched", topology=topology, pid=pid)

    def workers_crashed(self, exit_code: int, topology: dict) -> None:
        self._write("workers_crashed", exit_code=exit_code, topology=topology)

    def run_complete(self, topology: dict) -> None:
        self._write("run_complete", topology=topology)


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _build_torchrun_cmd(
    *,
    run_dir: Path,
    worker_script: str,
    worker_args: str,
    vllm_tp: int,
    vllm_dp: int,
    sae_tp: int,
    sae_dp: int,
    sae_pp_size: int,
    master_port: int,
    control_state_path: Path,
    checkpoint_storage: str = "disk",
) -> list[str]:
    nproc = vllm_tp * vllm_dp + sae_tp * sae_dp * sae_pp_size
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_addr=localhost",
        f"--master_port={master_port}",
        worker_script,
        "--streaming-mode",
        f"--vllm-tp-size={vllm_tp}",
        f"--vllm-dp-size={vllm_dp}",
        f"--sae-tp-size={sae_tp}",
        f"--sae-dp-size={sae_dp}",
        f"--sae-pp-size={sae_pp_size}",
        f"--control-state-path={control_state_path}",
        f"--checkpoint-storage={checkpoint_storage}",
        "--append-history-logs",
    ]
    if checkpoint_storage == "memory":
        cmd.append(
            f"--quiesce-checkpoint-path={_memory_quiesce_checkpoint_base(run_dir)}"
        )
    if worker_args:
        cmd.extend(worker_args.split())
    return cmd


def _quiesce_ack_paths(run_dir: Path, vllm_dp: int, sae_dp: int, sae_pp_size: int) -> list[Path]:
    paths = [run_dir / f"quiesce_ack_producer_{i}" for i in range(vllm_dp)]
    # One ack per PP-stage TP-root in each DP replica.
    for d in range(sae_dp):
        for p in range(max(sae_pp_size, 1)):
            paths.append(run_dir / f"quiesce_ack_consumer_d{d}_pp{p}")
    return paths


def _all_acks_present(run_dir: Path, vllm_dp: int, sae_dp: int, sae_pp_size: int) -> bool:
    return all(p.exists() for p in _quiesce_ack_paths(run_dir, vllm_dp, sae_dp, sae_pp_size))


def _all_acks_present_since(
    run_dir: Path,
    vllm_dp: int,
    sae_dp: int,
    sae_pp_size: int,
    since: float,
) -> bool:
    for path in _quiesce_ack_paths(run_dir, vllm_dp, sae_dp, sae_pp_size):
        if not path.exists() or path.stat().st_mtime < since:
            return False
    return True


def _vllm_stopped_ack_paths(run_dir: Path, vllm_dp: int) -> list[Path]:
    return [
        run_dir / f"vllm_stopped_produce_ack_producer_{i}"
        for i in range(vllm_dp)
    ]


def _vllm_finished_ack_paths(run_dir: Path, vllm_dp: int) -> list[Path]:
    return [run_dir / f"vllm_finished_ack_producer_{i}" for i in range(vllm_dp)]


def _sae_drain_ack_paths(run_dir: Path, sae_dp: int, sae_pp_size: int) -> list[Path]:
    return [
        run_dir / f"sae_drain_ack_consumer_d{d}_pp{p}"
        for d in range(sae_dp)
        for p in range(max(sae_pp_size, 1))
    ]


def _sae_finished_ack_paths(run_dir: Path, sae_dp: int, sae_pp_size: int) -> list[Path]:
    return [
        run_dir / f"sae_finished_ack_consumer_d{d}_pp{p}"
        for d in range(sae_dp)
        for p in range(max(sae_pp_size, 1))
    ]


def _all_paths_present_since(paths: list[Path], since: float) -> bool:
    for path in paths:
        if not path.exists() or path.stat().st_mtime < since:
            return False
    return True


def _missing_paths_since(paths: list[Path], since: float) -> list[Path]:
    return [
        path
        for path in paths
        if not path.exists() or path.stat().st_mtime < since
    ]


def _final_acks_complete_for_switch(
    *,
    run_dir: Path,
    sae_dp: int,
    sae_pp_size: int,
    vllm_dp: int,
    since: float,
    vllm_already_done: bool,
) -> bool:
    sae_paths = _sae_finished_ack_paths(run_dir, sae_dp, sae_pp_size)
    if not _all_paths_present_since(sae_paths, since):
        return False
    vllm_paths = _vllm_finished_ack_paths(run_dir, vllm_dp)
    if vllm_already_done:
        return all(path.exists() for path in vllm_paths)
    return _all_paths_present_since(vllm_paths, since)


def _missing_final_ack_paths_for_switch(
    *,
    run_dir: Path,
    sae_dp: int,
    sae_pp_size: int,
    vllm_dp: int,
    since: float,
    vllm_already_done: bool,
) -> list[Path]:
    missing = _missing_paths_since(
        _sae_finished_ack_paths(run_dir, sae_dp, sae_pp_size),
        since,
    )
    vllm_paths = _vllm_finished_ack_paths(run_dir, vllm_dp)
    if vllm_already_done:
        missing.extend(path for path in vllm_paths if not path.exists())
    else:
        missing.extend(_missing_paths_since(vllm_paths, since))
    return missing


def _cleanup_quiesce_files(run_dir: Path, vllm_dp: int, sae_dp: int, sae_pp_size: int) -> None:
    (run_dir / "quiesce_request").unlink(missing_ok=True)
    (run_dir / "sae_stop_acquire_request").unlink(missing_ok=True)
    (run_dir / "vllm_stop_produce_request").unlink(missing_ok=True)
    for p in _quiesce_ack_paths(run_dir, vllm_dp, sae_dp, sae_pp_size):
        p.unlink(missing_ok=True)
    for p in (
        _vllm_stopped_ack_paths(run_dir, vllm_dp)
        + _vllm_finished_ack_paths(run_dir, vllm_dp)
        + _sae_drain_ack_paths(run_dir, sae_dp, sae_pp_size)
        + _sae_finished_ack_paths(run_dir, sae_dp, sae_pp_size)
    ):
        p.unlink(missing_ok=True)


def _reset_buffer(
    run_dir: Path,
    control_state_path: Path,
    new_vllm_dp: int,
) -> None:
    """Reset buffer header for the new producer group and update control state."""
    from sae_lens.topology_control import read_control_state, write_control_state
    from sae_lens.training.shared_activation_buffer import SharedActivationBuffer

    state = read_control_state(control_state_path)
    buf = SharedActivationBuffer(
        name=state.buffer_name,
        num_chunks=state.buffer_params.num_chunks,
        chunk_size_tokens=state.buffer_params.chunk_size_tokens,
        d_model=state.buffer_params.d_model,
        num_producers=new_vllm_dp,
        create=False,
    )
    # Read next_claim_seq before reset (it is preserved across restarts).
    next_claim_seq = int(buf._header[3])
    buf.reset_for_restart(new_vllm_dp)
    buf.close()

    state.next_claim_seq_at_quiesce = next_claim_seq
    write_control_state(control_state_path, state)
    print(f"[supervisor] buffer reset: new_num_producers={new_vllm_dp} next_claim_seq={next_claim_seq}")


def _apply_new_topology(
    control_state_path: Path,
    new_vllm_tp: int,
    new_vllm_dp: int,
    new_sae_tp: int,
    new_sae_dp: int,
    new_sae_pp_size: int,
    new_checkpoint_path: str | None,
) -> None:
    from sae_lens.topology_control import TopologySpec, read_control_state, write_control_state

    state = read_control_state(control_state_path)
    state.topology = TopologySpec(
        vllm_tp=new_vllm_tp,
        vllm_dp=new_vllm_dp,
        sae_tp=new_sae_tp,
        sae_dp=new_sae_dp,
        sae_pp_size=new_sae_pp_size,
    )
    state.phase = "RUNNING"
    if new_checkpoint_path is not None:
        state.checkpoint_path = new_checkpoint_path
    write_control_state(control_state_path, state)


def _memory_quiesce_checkpoint_base(run_dir: Path) -> Path:
    safe = str(run_dir.resolve()).strip("/").replace("/", "_")
    return Path("/dev/shm") / "saelens_topology_checkpoints" / safe


def _cleanup_runtime_shm(
    *,
    run_dir: Path,
    control_state_path: Path,
    shm_base_dir: Path = Path("/dev/shm"),
    cleanup_memory_checkpoints: bool = True,
) -> None:
    """Best-effort cleanup for runtime-only shared memory artifacts."""
    try:
        from sae_lens.topology_control import read_control_state
        from sae_lens.training.shared_activation_buffer import SharedActivationBuffer

        state = read_control_state(control_state_path)
        if state.buffer_name:
            SharedActivationBuffer.cleanup_files(
                state.buffer_name,
                base_dir=str(shm_base_dir),
            )
            print(f"[supervisor] cleaned shared buffer from {shm_base_dir}: {state.buffer_name}")
    except Exception as e:
        print(f"[supervisor] shared-buffer cleanup failed (ignoring): {e}")

    if cleanup_memory_checkpoints:
        ckpt_base = _memory_quiesce_checkpoint_base(run_dir)
        try:
            if ckpt_base.exists():
                shutil.rmtree(ckpt_base)
                print(f"[supervisor] cleaned memory checkpoints: {ckpt_base}")
        except Exception as e:
            print(f"[supervisor] memory-checkpoint cleanup failed (ignoring): {e}")

    # Clean up any leftover SAE buffer files matching the run_dir pattern
    try:
        run_dir_markers = _run_dir_markers(run_dir)
        for marker in run_dir_markers:
            # Find all buffer files related to this run
            for pattern in ("sae_buf_*", "*_state.bin", "*_header.bin"):
                for file_path in shm_base_dir.glob(pattern):
                    try:
                        # Check if this file is related to our run by checking its name
                        if any(marker.replace("/", "_") in str(file_path) for marker in run_dir_markers):
                            file_path.unlink(missing_ok=True)
                            print(f"[supervisor] cleaned buffer file: {file_path}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"[supervisor] buffer file cleanup failed (ignoring): {e}")



def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def _worker_arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for idx, arg in enumerate(args):
        if arg == name and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _worker_arg_int(args: list[str], name: str, default: int) -> int:
    value = _worker_arg_value(args, name)
    return int(value) if value is not None else default


def _resolve_hidden_size_for_estimate(model_name: str | None) -> int:
    if model_name:
        cfg_path = Path(model_name) / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                hidden_size = cfg.get("hidden_size") or cfg.get("n_embd")
                if hidden_size is not None:
                    return int(hidden_size)
            except Exception:
                pass
    # Llama-3.1-8B default used by the GPU runner demo.
    return 4096


def _dtype_nbytes(dtype: str) -> int:
    normalized = dtype.replace("torch.", "").lower()
    if normalized in {"float32", "fp32"}:
        return 4
    if normalized in {"bfloat16", "bf16", "float16", "fp16"}:
        return 2
    return 4


def _estimate_startup_memory_plan(
    *,
    run_dir: Path,
    worker_args: str,
    checkpoint_storage: str,
    include_buffer: bool = True,
) -> StartupMemoryPlan:
    args = shlex.split(worker_args)
    model_name = _worker_arg_value(args, "--model-name")
    d_in = _worker_arg_int(args, "--d-in", _resolve_hidden_size_for_estimate(model_name))
    d_sae = _worker_arg_int(args, "--d-sae", 32768)
    dtype = _worker_arg_value(args, "--dtype", "float32") or "float32"
    dtype_bytes = _dtype_nbytes(dtype)
    hook_names = _worker_arg_value(args, "--hook-names")
    num_hooks = len([h for h in hook_names.split(",") if h]) if hook_names else 1
    chunk_tokens_per_hook = _worker_arg_int(args, "--streaming-chunk-size-tokens", 4096)
    num_chunks = _worker_arg_int(args, "--streaming-num-chunks", 32)
    buffer_chunk_rows = chunk_tokens_per_hook * num_hooks
    buffer_bytes = (
        num_chunks * buffer_chunk_rows * d_in * dtype_bytes
        if include_buffer
        else 0
    )

    param_elems_per_hook = 2 * d_in * d_sae + d_sae + d_in
    weight_bytes_per_hook = param_elems_per_hook * dtype_bytes
    # Adam stores exp_avg and exp_avg_sq for every parameter. Use fp32 for a
    # conservative estimate even if the SAE parameter dtype is lower precision.
    adam_bytes_per_hook = param_elems_per_hook * 2 * 4
    stats_and_pt_overhead = num_hooks * (3 * d_sae * 4 + 4 * 1024 * 1024)
    checkpoint_bytes = (
        num_hooks * (weight_bytes_per_hook + adam_bytes_per_hook)
        + stats_and_pt_overhead
    )
    usage = shutil.disk_usage("/dev/shm")
    required_bytes = buffer_bytes + (
        checkpoint_bytes if checkpoint_storage == "memory" else 0
    )
    return StartupMemoryPlan(
        shm_path=Path("/dev/shm"),
        shm_total_bytes=usage.total,
        shm_free_bytes=usage.free,
        buffer_bytes=int(buffer_bytes),
        checkpoint_bytes=int(checkpoint_bytes),
        required_bytes=int(required_bytes),
        d_in=d_in,
        d_sae=d_sae,
        num_hooks=num_hooks,
        dtype=dtype,
        checkpoint_storage=checkpoint_storage,
    )


def _assert_memory_plan_has_space(
    plan: StartupMemoryPlan,
    *,
    available_bytes: int | None = None,
) -> None:
    if available_bytes is None:
        usage = shutil.disk_usage("/dev/shm")
        available = usage.free
    else:
        available = available_bytes
    if available < plan.required_bytes:
        checkpoint_detail = (
            _format_bytes(plan.checkpoint_bytes)
            if plan.checkpoint_storage == "memory"
            else f"{_format_bytes(plan.checkpoint_bytes)} estimate, stored on disk"
        )
        raise RuntimeError(
            "Insufficient /dev/shm space during preflight; no shared activation "
            "buffer was allocated by this supervisor run: "
            f"need {_format_bytes(plan.required_bytes)} "
            f"(buffer={_format_bytes(plan.buffer_bytes)}, "
            f"checkpoint={checkpoint_detail}), "
            f"available={_format_bytes(available)}"
        )


def _print_startup_memory_plan(plan: StartupMemoryPlan) -> None:
    print(
        "[supervisor] /dev/shm space: "
        f"free={_format_bytes(plan.shm_free_bytes)} "
        f"total={_format_bytes(plan.shm_total_bytes)}"
    )
    print(
        "[supervisor] preflight allocation estimate: "
        f"buffer_required={_format_bytes(plan.buffer_bytes)} "
        f"checkpoint_estimate={_format_bytes(plan.checkpoint_bytes)} "
        f"shm_required={_format_bytes(plan.required_bytes)} "
        f"d_in={plan.d_in} d_sae={plan.d_sae} hooks={plan.num_hooks} "
        f"dtype={plan.dtype} checkpoint_storage={plan.checkpoint_storage}"
    )


def _is_complete_quiesce_checkpoint(checkpoint_dir: Path) -> bool:
    if not checkpoint_dir.is_dir():
        return False
    if not (checkpoint_dir / "COMPLETED").exists():
        return False
    manifest_path = checkpoint_dir / "multi_sae_manifest.json"
    if not manifest_path.exists():
        return (
            (checkpoint_dir / "trainer_state.pt").exists()
            and (checkpoint_dir / "sae_weights.safetensors").exists()
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    hook_names = manifest.get("hook_names")
    hook_to_dir = manifest.get("hook_to_dir", {})
    if not isinstance(hook_names, list) or not isinstance(hook_to_dir, dict):
        return False
    if not (checkpoint_dir / "trainer_state.pt").exists():
        return False
    for hook_name in hook_names:
        hook_dir = checkpoint_dir / str(
            hook_to_dir.get(
                hook_name,
                hook_name.replace(".", "_").replace("/", "_"),
            )
        )
        if not (hook_dir / "sae_weights.safetensors").exists():
            return False
        if not (hook_dir / "hook_state.pt").exists():
            return False
    return True


def _quiesce_switch_succeeded(acks_complete: bool, exit_code: int | None) -> bool:
    return acks_complete and exit_code == 0


def _worker_crash_is_retryable(state: Any) -> bool:
    return bool(getattr(state, "buffer_name", ""))


_TOPOLOGY_TARGET_PRESETS: dict[str, dict[str, int]] = {
    "1vllm_1sae": {
        "vllm_tp": 1,
        "vllm_dp": 1,
        "sae_tp": 1,
        "sae_dp": 1,
        "sae_pp_size": 1,
    },
    "sae_pp2": {
        "vllm_tp": 1,
        "vllm_dp": 0,
        "sae_tp": 1,
        "sae_dp": 1,
        "sae_pp_size": 2,
    },
}


def _parse_topology_target(value: str) -> dict[str, int]:
    normalized = value.strip().lower()
    if normalized in _TOPOLOGY_TARGET_PRESETS:
        return dict(_TOPOLOGY_TARGET_PRESETS[normalized])
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("topology target must be a JSON object or known preset")
    required = ("vllm_tp", "vllm_dp", "sae_tp")
    for key in required:
        if key not in raw:
            raise ValueError(f"topology target missing required key: {key}")
    return {
        "vllm_tp": int(raw["vllm_tp"]),
        "vllm_dp": int(raw["vllm_dp"]),
        "sae_tp": int(raw["sae_tp"]),
        "sae_dp": int(raw.get("sae_dp", 1)),
        "sae_pp_size": int(raw.get("sae_pp_size", 1)),
    }


def _topology_matches(current: dict[str, Any], target: dict[str, int]) -> bool:
    return all(int(current.get(k, -1)) == int(v) for k, v in target.items())


def _buffer_producers_done(buffer_state: dict[str, Any] | None) -> bool:
    if buffer_state is None:
        return False
    num_producers = int(buffer_state.get("num_producers", 0))
    if num_producers <= 0:
        return True
    return int(buffer_state.get("done_count", 0)) >= num_producers


def _done_buffer_switch_request(
    *,
    enabled: bool,
    already_checked: bool,
    current_topology: dict[str, Any],
    buffer_state: dict[str, Any] | None,
    threshold: float,
    target_topology: dict[str, int],
) -> dict[str, int | str] | None:
    if not enabled or already_checked or not _buffer_producers_done(buffer_state):
        return None
    if _topology_matches(current_topology, target_topology):
        return None
    assert buffer_state is not None
    total = int(buffer_state.get("total", 0))
    if total <= 0:
        return None
    ready_pct = int(buffer_state.get("ready", 0)) / total
    if ready_pct <= threshold:
        return None
    return {**target_topology, "_trigger": "vllm_done_buffer_ready_pct"}


def _read_shared_buffer_state(
    control_state_path: Path,
    shm_base_dir: Path = Path("/dev/shm"),
) -> dict[str, int | float] | None:
    try:
        from sae_lens.topology_control import read_control_state
        import numpy as np

        state = read_control_state(control_state_path)
        if not state.buffer_name:
            return None
        state_path = shm_base_dir / f"{state.buffer_name}_state.bin"
        header_path = shm_base_dir / f"{state.buffer_name}_header.bin"
        if not state_path.exists() or not header_path.exists():
            return None
        total = int(state.buffer_params.num_chunks)
        arr = np.memmap(str(state_path), dtype=np.int8, mode="r", shape=(total,))
        header = np.memmap(str(header_path), dtype=np.int32, mode="r", shape=(8,))
        counts = {int(v): 0 for v in range(4)}
        for value in arr:
            counts[int(value)] += 1
        return {
            "free": counts[0],
            "writing": counts[1],
            "ready": counts[2],
            "consuming": counts[3],
            "total": total,
            "ready_pct": round(counts[2] / total, 4) if total else 0.0,
            "num_producers": int(header[0]),
            "done_count": int(header[1]),
            "target_chunks": int(header[2]),
            "next_claim_seq": int(header[3]),
        }
    except Exception:
        return None


def _find_latest_quiesce_checkpoint(run_dir: Path, since: float | None = None) -> str | None:
    """Find the most recent quiesce_* checkpoint directory under checkpoints/."""
    # cfg.checkpoint_path is set to run_dir/checkpoints, so quiesce checkpoints
    # are saved directly as run_dir/checkpoints/quiesce_N.
    # Also search one level deeper (run_dir/checkpoints/*/quiesce_N) for runs
    # that use a unique run-id subdirectory.
    if not run_dir.exists():
        return None
    candidates = []
    for ckpt_base in [run_dir / "checkpoints", _memory_quiesce_checkpoint_base(run_dir)]:
        if not ckpt_base.is_dir():
            continue
        # Direct: run_dir/checkpoints/quiesce_*
        for ckpt in ckpt_base.glob("quiesce_*"):
            if ckpt.is_dir():
                candidates.append(ckpt)
        # One level deep: run_dir/checkpoints/*/quiesce_*
        for subdir in ckpt_base.iterdir():
            if not subdir.is_dir():
                continue
            for ckpt in subdir.glob("quiesce_*"):
                if ckpt.is_dir():
                    candidates.append(ckpt)
    candidates = [
        ckpt
        for ckpt in candidates
        if (since is None or ckpt.stat().st_mtime >= since)
        and _is_complete_quiesce_checkpoint(ckpt)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return str(candidates[-1])


def _wait_for_workers(proc: subprocess.Popen, timeout_s: float) -> None:
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[supervisor] workers did not exit within {timeout_s}s — sending SIGKILL")
        proc.kill()
        proc.wait()


def run_supervisor(
    *,
    run_dir: Path,
    worker_script: str,
    worker_args: str,
    initial_vllm_tp: int,
    initial_vllm_dp: int,
    initial_sae_tp: int,
    initial_sae_dp: int = 1,
    initial_sae_pp_size: int = 1,
    num_gpus: int,
    resume_from_checkpoint: str | None,
    log_path: Path | None = None,
    checkpoint_storage: str = "disk",
    cleanup_shm: bool = True,
    done_buffer_ready_pct_threshold: float | None = None,
    done_buffer_target_topology: dict[str, int] | None = None,
) -> None:
    from sae_lens.topology_control import (
        BufferParams,
        ControlState,
        TopologySpec,
        read_control_state,
        write_control_state,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    control_state_path = run_dir / "control_state.json"
    topology_request_path = run_dir / "topology_request.json"
    quiesce_request_path = run_dir / "quiesce_request"
    sae_stop_acquire_request_path = run_dir / "sae_stop_acquire_request"
    vllm_stop_produce_request_path = run_dir / "vllm_stop_produce_request"

    _terminate_existing_run_processes(run_dir)

    # --- Clean up /dev/shm before starting ---
    if cleanup_shm:
        _cleanup_runtime_shm(
            run_dir=run_dir,
            control_state_path=control_state_path,
            cleanup_memory_checkpoints=True,
        )

    # --- Set up switch event logger ---
    if log_path is None:
        ts = time.strftime("%y_%m_%d_%H_%M_%S")
        log_path = Path("results/topology") / f"topology_run_{ts}.jsonl"
    logger = SwitchLogger(log_path)
    print(f"[supervisor] topology log: {log_path}")

    # --- Initialise or recover control state ---
    if control_state_path.exists():
        state = read_control_state(control_state_path)
        print(f"[supervisor] recovered control state: phase={state.phase} topology={state.topology}")
        vllm_tp = state.topology.vllm_tp
        vllm_dp = state.topology.vllm_dp
        sae_tp = state.topology.sae_tp
        sae_dp = state.topology.sae_dp
        sae_pp_size = state.topology.sae_pp_size
    else:
        vllm_tp = initial_vllm_tp
        vllm_dp = initial_vllm_dp
        sae_tp = initial_sae_tp
        sae_dp = initial_sae_dp
        sae_pp_size = initial_sae_pp_size
        state = ControlState(
            phase="RUNNING",
            topology=TopologySpec(
                vllm_tp=vllm_tp,
                vllm_dp=vllm_dp,
                sae_tp=sae_tp,
                sae_dp=sae_dp,
                sae_pp_size=sae_pp_size,
            ),
            buffer_name="",  # workers will generate and we'll read it back
            buffer_params=BufferParams(num_chunks=0, chunk_size_tokens=0, d_model=0, dtype="bfloat16"),
            checkpoint_path=resume_from_checkpoint,
            next_claim_seq_at_quiesce=0,
            target_chunks=0,
        )
        write_control_state(control_state_path, state)

    include_buffer_in_space_check = not bool(state.buffer_name)
    memory_plan = _estimate_startup_memory_plan(
        run_dir=run_dir,
        worker_args=worker_args,
        checkpoint_storage=checkpoint_storage,
        include_buffer=include_buffer_in_space_check,
    )
    _print_startup_memory_plan(memory_plan)
    _assert_memory_plan_has_space(memory_plan)

    logger.supervisor_start(
        topology={
            "vllm_tp": vllm_tp, "vllm_dp": vllm_dp,
            "sae_tp": sae_tp, "sae_dp": sae_dp, "sae_pp_size": sae_pp_size,
        },
        run_dir=str(run_dir),
    )

    # --- Main supervisor loop ---
    proc: subprocess.Popen | None = None
    master_port = _find_free_port()
    done_buffer_switch_checked = False

    def _launch() -> subprocess.Popen:
        nonlocal master_port
        master_port = _find_free_port()
        cmd = _build_torchrun_cmd(
            run_dir=run_dir,
            worker_script=worker_script,
            worker_args=worker_args,
            vllm_tp=vllm_tp,
            vllm_dp=vllm_dp,
            sae_tp=sae_tp,
            sae_dp=sae_dp,
            sae_pp_size=sae_pp_size,
            master_port=master_port,
            control_state_path=control_state_path,
            checkpoint_storage=checkpoint_storage,
        )
        print(f"[supervisor] launching: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        logger.new_workers_launched(
            topology={
                "vllm_tp": vllm_tp, "vllm_dp": vllm_dp,
                "sae_tp": sae_tp, "sae_dp": sae_dp, "sae_pp_size": sae_pp_size,
            },
            pid=p.pid,
        )
        return p

    # Handle recovery from a previous quiesce that was interrupted.
    if state.phase == "QUIESCING":
        print("[supervisor] recovering from interrupted quiesce")
        if _all_acks_present(run_dir, vllm_dp, sae_dp, sae_pp_size):
            # Acks are present but supervisor crashed before resetting buffer.
            # Read the pending topology request if it still exists.
            new_topo = None
            if topology_request_path.exists():
                new_topo = json.loads(topology_request_path.read_text())
            _reset_buffer(run_dir, control_state_path, new_topo["vllm_dp"] if new_topo else vllm_dp)
            if new_topo:
                ckpt = _find_latest_quiesce_checkpoint(run_dir, since=quiesce_signal_time)
                _apply_new_topology(
                    control_state_path,
                    new_topo["vllm_tp"], new_topo["vllm_dp"], new_topo["sae_tp"],
                    int(new_topo.get("sae_dp", 1)),
                    int(new_topo.get("sae_pp_size", 1)),
                    ckpt,
                )
                vllm_tp = new_topo["vllm_tp"]
                vllm_dp = new_topo["vllm_dp"]
                sae_tp = new_topo["sae_tp"]
                sae_dp = int(new_topo.get("sae_dp", 1))
                sae_pp_size = int(new_topo.get("sae_pp_size", 1))
            _cleanup_quiesce_files(run_dir, vllm_dp, sae_dp, sae_pp_size)
            topology_request_path.unlink(missing_ok=True)
        else:
            # Acks not all present — workers may have crashed during quiesce.
            # Restart with same topology from last checkpoint.
            state = read_control_state(control_state_path)
            state.phase = "RUNNING"
            write_control_state(control_state_path, state)
            _cleanup_quiesce_files(run_dir, vllm_dp, sae_dp, sae_pp_size)

    # Belt-and-braces: even when phase != QUIESCING, scrub leftover quiesce
    # signal files from a previous run before launching workers. Otherwise the
    # next worker generation reads `quiesce_request` on its first step and
    # quits after a single batch.
    _cleanup_quiesce_files(run_dir, vllm_dp, sae_dp, sae_pp_size)
    # Also remove any stale producer/consumer ack files from older topologies
    # that no longer match the current shape.
    for stale in run_dir.glob("quiesce_ack_consumer*"):
        stale.unlink(missing_ok=True)
    for stale in run_dir.glob("quiesce_ack_producer_*"):
        stale.unlink(missing_ok=True)
    (run_dir / "quiesce_request").unlink(missing_ok=True)
    for pattern in (
        "sae_drain_ack_consumer*",
        "sae_finished_ack_consumer*",
        "vllm_stopped_produce_ack_producer_*",
        "vllm_finished_ack_producer_*",
    ):
        for stale in run_dir.glob(pattern):
            stale.unlink(missing_ok=True)
    sae_stop_acquire_request_path.unlink(missing_ok=True)
    vllm_stop_produce_request_path.unlink(missing_ok=True)

    proc = _launch()

    try:
        while True:
            time.sleep(_POLL_INTERVAL_S)

            # Check if workers crashed.
            if proc.poll() is not None:
                exit_code = proc.returncode
                if exit_code == 0:
                    # Check if a topology switch is pending — if so, the consumer
                    # drained the buffer (vllm_dp=0) and we should apply the switch
                    # instead of treating this as run complete.
                    if topology_request_path.exists():
                        print("[supervisor] workers exited cleanly but switch pending — applying switch")
                    else:
                        print("[supervisor] workers exited cleanly — run complete")
                        logger.run_complete(
                            topology={
                                "vllm_tp": vllm_tp, "vllm_dp": vllm_dp,
                                "sae_tp": sae_tp, "sae_dp": sae_dp,
                                "sae_pp_size": sae_pp_size,
                            }
                        )
                        break
                print(f"[supervisor] workers crashed (exit_code={exit_code}) — restarting")
                logger.workers_crashed(
                    exit_code=exit_code,
                    topology={
                        "vllm_tp": vllm_tp, "vllm_dp": vllm_dp,
                        "sae_tp": sae_tp, "sae_dp": sae_dp,
                        "sae_pp_size": sae_pp_size,
                    },
                )
                state = read_control_state(control_state_path)
                if not _worker_crash_is_retryable(state):
                    print(
                        "[supervisor] worker crashed before shared buffer initialization; "
                        "not retrying automatically"
                    )
                    break
                # Reset any WRITING chunks left by the crash.
                if state.buffer_name:
                    try:
                        _reset_buffer(run_dir, control_state_path, vllm_dp)
                    except Exception as e:
                        print(f"[supervisor] buffer reset failed (ignoring): {e}")
                _cleanup_quiesce_files(run_dir, vllm_dp, sae_dp, sae_pp_size)
                proc = _launch()
                continue

            # Check for topology switch request.
            if (
                not topology_request_path.exists()
                and done_buffer_ready_pct_threshold is not None
                and done_buffer_target_topology is not None
            ):
                buf_state = _read_shared_buffer_state(control_state_path)
                if _buffer_producers_done(buf_state):
                    auto_request = _done_buffer_switch_request(
                        enabled=True,
                        already_checked=done_buffer_switch_checked,
                        current_topology={
                            "vllm_tp": vllm_tp,
                            "vllm_dp": vllm_dp,
                            "sae_tp": sae_tp,
                            "sae_dp": sae_dp,
                            "sae_pp_size": sae_pp_size,
                        },
                        buffer_state=buf_state,
                        threshold=done_buffer_ready_pct_threshold,
                        target_topology=done_buffer_target_topology,
                    )
                    done_buffer_switch_checked = True
                    if auto_request is not None:
                        topology_request_path.write_text(json.dumps(auto_request))

            if topology_request_path.exists():
                try:
                    new_topo = json.loads(topology_request_path.read_text())
                    new_vllm_tp = int(new_topo["vllm_tp"])
                    new_vllm_dp = int(new_topo["vllm_dp"])
                    new_sae_tp = int(new_topo["sae_tp"])
                    new_sae_dp = int(new_topo.get("sae_dp", 1))
                    new_sae_pp_size = int(new_topo.get("sae_pp_size", 1))
                except Exception as e:
                    print(f"[supervisor] invalid topology_request.json: {e} — ignoring")
                    topology_request_path.unlink(missing_ok=True)
                    continue

                required_gpus = (
                    new_vllm_tp * new_vllm_dp
                    + new_sae_tp * new_sae_dp * new_sae_pp_size
                )
                if required_gpus > num_gpus:
                    print(
                        f"[supervisor] topology request requires {required_gpus} GPUs "
                        f"but only {num_gpus} available — ignoring"
                    )
                    topology_request_path.unlink(missing_ok=True)
                    continue

                from_topo = {
                    "vllm_tp": vllm_tp, "vllm_dp": vllm_dp,
                    "sae_tp": sae_tp, "sae_dp": sae_dp,
                    "sae_pp_size": sae_pp_size,
                }
                to_topo = {
                    "vllm_tp": new_vllm_tp, "vllm_dp": new_vllm_dp,
                    "sae_tp": new_sae_tp, "sae_dp": new_sae_dp,
                    "sae_pp_size": new_sae_pp_size,
                }

                # Read buffer state for the log record (best-effort).
                buf_state: dict | None = None
                try:
                    state_snap = read_control_state(control_state_path)
                    if state_snap.buffer_name:
                        import numpy as np
                        sp = Path("/dev/shm") / f"{state_snap.buffer_name}_state.bin"
                        if sp.exists():
                            arr = np.memmap(str(sp), dtype=np.int8, mode="r",
                                            shape=(state_snap.buffer_params.num_chunks,))
                            counts = {int(v): 0 for v in range(4)}
                            for v in arr:
                                counts[int(v)] += 1
                            total = state_snap.buffer_params.num_chunks
                            buf_state = {
                                "free": counts[0], "writing": counts[1],
                                "ready": counts[2], "consuming": counts[3],
                                "total": total,
                                "ready_pct": round(counts[2] / total, 4) if total else 0,
                            }
                except Exception:
                    pass

                print(
                    f"[supervisor] topology switch requested: "
                    f"vllm_tp={new_vllm_tp} vllm_dp={new_vllm_dp} "
                    f"sae_tp={new_sae_tp} sae_dp={new_sae_dp} sae_pp_size={new_sae_pp_size}"
                )
                # 1. Log trigger with buffer state at the moment of detection
                trigger = new_topo.get("_trigger", "manual")
                logger.switch_triggered(
                    trigger=trigger,
                    from_topology=from_topo,
                    to_topology=to_topo,
                    buffer_state=buf_state,
                )

                # Mark quiescing in control state.
                state = read_control_state(control_state_path)
                state.phase = "QUIESCING"
                write_control_state(control_state_path, state)

                # 2. First stop SAE acquisition. SAE drains only its local pool
                # while vLLM is still allowed to produce into shm.
                quiesce_signal_time = time.time()
                sae_stop_acquire_request_path.touch()
                logger.quiesce_signaled(from_topology=from_topo)

                print("[supervisor] waiting for SAE drain acks...")
                sae_drain_paths = _sae_drain_ack_paths(run_dir, sae_dp, sae_pp_size)
                sae_drain_complete = _all_paths_present_since(
                    sae_drain_paths, quiesce_signal_time
                )
                while not sae_drain_complete:
                    if proc.poll() is not None:
                        print("[supervisor] workers exited during SAE drain wait")
                        break
                    time.sleep(1.0)
                    sae_drain_complete = _all_paths_present_since(
                        sae_drain_paths, quiesce_signal_time
                    )

                if sae_drain_complete:
                    print("[supervisor] SAE drained local pools; stopping vLLM producers")

                # 3. Stop vLLM after SAE no longer acquires. Producers finish
                # their current chunk, save dataset state, signal buffer done,
                # then continue exiting while SAE may still be checkpointing.
                vllm_stop_signal_time = time.time()
                vllm_stop_produce_request_path.touch()
                vllm_already_done = _buffer_producers_done(
                    _read_shared_buffer_state(control_state_path)
                )
                vllm_stopped_paths = (
                    []
                    if vllm_already_done
                    else _vllm_stopped_ack_paths(run_dir, vllm_dp)
                )
                vllm_stopped_complete = _all_paths_present_since(
                    vllm_stopped_paths, vllm_stop_signal_time
                )
                if vllm_already_done:
                    print("[supervisor] vLLM producers already finished; skipping vLLM stop acks")
                elif vllm_stopped_paths:
                    print("[supervisor] waiting for vLLM stop acks...")
                else:
                    print("[supervisor] no vLLM producers in current topology; skipping vLLM stop acks")
                while not vllm_stopped_complete:
                    if proc.poll() is not None:
                        print("[supervisor] workers exited during vLLM stop wait")
                        break
                    time.sleep(1.0)
                    vllm_stopped_complete = _all_paths_present_since(
                        vllm_stopped_paths, vllm_stop_signal_time
                    )

                print("[supervisor] waiting for final worker acks...")
                final_acks_complete = _final_acks_complete_for_switch(
                    run_dir=run_dir,
                    sae_dp=sae_dp,
                    sae_pp_size=sae_pp_size,
                    vllm_dp=vllm_dp,
                    since=quiesce_signal_time,
                    vllm_already_done=vllm_already_done,
                )
                while not final_acks_complete:
                    if proc.poll() is not None:
                        final_acks_complete = _final_acks_complete_for_switch(
                            run_dir=run_dir,
                            sae_dp=sae_dp,
                            sae_pp_size=sae_pp_size,
                            vllm_dp=vllm_dp,
                            since=quiesce_signal_time,
                            vllm_already_done=vllm_already_done,
                        )
                        break
                    time.sleep(1.0)
                    final_acks_complete = _final_acks_complete_for_switch(
                        run_dir=run_dir,
                        sae_dp=sae_dp,
                        sae_pp_size=sae_pp_size,
                        vllm_dp=vllm_dp,
                        since=quiesce_signal_time,
                        vllm_already_done=vllm_already_done,
                    )
                if not final_acks_complete:
                    missing = _missing_final_ack_paths_for_switch(
                        run_dir=run_dir,
                        sae_dp=sae_dp,
                        sae_pp_size=sae_pp_size,
                        vllm_dp=vllm_dp,
                        since=quiesce_signal_time,
                        vllm_already_done=vllm_already_done,
                    )
                    print(
                        "[supervisor] final worker acks missing: "
                        + ", ".join(str(path.name) for path in missing)
                    )

                acks_complete = (
                    sae_drain_complete
                    and vllm_stopped_complete
                    and final_acks_complete
                )
                if acks_complete:
                    logger.all_acks_received(from_topology=from_topo)

                # Wait for workers to exit.
                _wait_for_workers(proc, _WORKER_EXIT_TIMEOUT_S)
                logger.workers_exited(exit_code=proc.returncode, from_topology=from_topo)

                # Reset buffer for new producer group (only if workers created it).
                state = read_control_state(control_state_path)
                if state.buffer_name:
                    _reset_buffer(run_dir, control_state_path, new_vllm_dp)

                # 4. Find the quiesce checkpoint written by the consumer.
                ckpt = _find_latest_quiesce_checkpoint(run_dir)
                print(f"[supervisor] quiesce checkpoint: {ckpt}")
                logger.checkpoint_found(checkpoint_path=ckpt)

                # Apply new topology to control state.
                _apply_new_topology(
                    control_state_path,
                    new_vllm_tp, new_vllm_dp, new_sae_tp, new_sae_dp, new_sae_pp_size,
                    ckpt,
                )
                old_vllm_dp = vllm_dp
                old_sae_dp = sae_dp
                old_sae_pp_size = sae_pp_size
                vllm_tp = new_vllm_tp
                vllm_dp = new_vllm_dp
                sae_tp = new_sae_tp
                sae_dp = new_sae_dp
                sae_pp_size = new_sae_pp_size

                # Clean up signal files using the OLD topology shape (so we
                # remove every file that was created during quiesce).
                _cleanup_quiesce_files(run_dir, old_vllm_dp, old_sae_dp, old_sae_pp_size)
                topology_request_path.unlink(missing_ok=True)

                print(
                    f"[supervisor] launching new topology: "
                    f"vllm_tp={vllm_tp} vllm_dp={vllm_dp} sae_tp={sae_tp} "
                    f"sae_dp={sae_dp} sae_pp_size={sae_pp_size}"
                )
                # 5. Launch new workers — logged inside _launch()
                proc = _launch()
                done_buffer_switch_checked = False

    except KeyboardInterrupt:
        print("[supervisor] interrupted — sending quiesce to workers")
        quiesce_request_path.touch()
        sae_stop_acquire_request_path.touch()
        vllm_stop_produce_request_path.touch()
        _wait_for_workers(proc, _WORKER_EXIT_TIMEOUT_S)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()
        if cleanup_shm:
            _cleanup_runtime_shm(
                run_dir=run_dir,
                control_state_path=control_state_path,
                cleanup_memory_checkpoints=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Topology supervisor for streaming SAE training")
    parser.add_argument("--run-dir", required=True, help="Directory for control state and signal files")
    parser.add_argument("--worker-script", default="scripts/run_sae_runner_gpu.py")
    parser.add_argument(
        "--worker-args",
        default="",
        help="Extra args forwarded to the worker script (space-separated string)",
    )
    parser.add_argument("--vllm-tp", type=int, default=1)
    parser.add_argument("--vllm-dp", type=int, default=1)
    parser.add_argument("--sae-tp", type=int, default=1)
    parser.add_argument("--sae-dp", type=int, default=1, help="0=no SAE (vLLM-only), 1=has SAE")
    parser.add_argument(
        "--sae-pp-size", type=int, default=1,
        help="Number of SAE pipeline stages (each trains a hook subset).",
    )
    parser.add_argument("--num-gpus", type=int, required=True, help="Total available GPUs")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-storage",
        choices=["memory", "disk"],
        default="disk",
        help="Storage backend for quiesce checkpoints.",
    )
    parser.add_argument(
        "--no-cleanup-shm",
        action="store_true",
        help="Keep runtime /dev/shm buffer and memory quiesce checkpoints after supervisor exits.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Path for JSONL event log. Default: results/topology/topology_run_YY_MM_DD_HH_MM_SS.jsonl",
    )
    parser.add_argument(
        "--done-buffer-ready-pct-threshold",
        type=float,
        default=None,
        help=(
            "Optional one-shot trigger after all vLLM producers finish: if "
            "READY chunks / total chunks is greater than this threshold, request "
            "the topology from --done-buffer-target-topology."
        ),
    )
    parser.add_argument(
        "--done-buffer-target-topology",
        default=None,
        help=(
            "Target for --done-buffer-ready-pct-threshold. Accepts JSON topology "
            "or preset names: 1vllm_1sae, sae_pp2."
        ),
    )
    args = parser.parse_args()

    done_buffer_target = None
    if args.done_buffer_ready_pct_threshold is not None:
        if not 0 <= args.done_buffer_ready_pct_threshold <= 1:
            parser.error("--done-buffer-ready-pct-threshold must be in [0, 1]")
        if args.done_buffer_target_topology is None:
            parser.error(
                "--done-buffer-target-topology is required when "
                "--done-buffer-ready-pct-threshold is set"
            )
        try:
            done_buffer_target = _parse_topology_target(args.done_buffer_target_topology)
        except Exception as exc:
            parser.error(f"invalid --done-buffer-target-topology: {exc}")

    run_supervisor(
        run_dir=Path(args.run_dir),
        worker_script=args.worker_script,
        worker_args=args.worker_args,
        initial_vllm_tp=args.vllm_tp,
        initial_vllm_dp=args.vllm_dp,
        initial_sae_tp=args.sae_tp,
        initial_sae_dp=args.sae_dp,
        initial_sae_pp_size=args.sae_pp_size,
        num_gpus=args.num_gpus,
        resume_from_checkpoint=args.resume_from_checkpoint,
        log_path=Path(args.log_path) if args.log_path else None,
        checkpoint_storage=args.checkpoint_storage,
        cleanup_shm=not args.no_cleanup_shm,
        done_buffer_ready_pct_threshold=args.done_buffer_ready_pct_threshold,
        done_buffer_target_topology=done_buffer_target,
    )


if __name__ == "__main__":
    main()
