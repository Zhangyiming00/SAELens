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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Supervisor polls these files at this interval (seconds).
_POLL_INTERVAL_S = 5.0
# After quiesce acks received, wait this long for workers to exit before SIGKILL.
_WORKER_EXIT_TIMEOUT_S = 60.0


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
    master_port: int,
    control_state_path: Path,
) -> list[str]:
    nproc = vllm_tp * vllm_dp + sae_tp * sae_dp
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
        f"--control-state-path={control_state_path}",
    ]
    if worker_args:
        cmd.extend(worker_args.split())
    return cmd


def _quiesce_ack_paths(run_dir: Path, vllm_dp: int) -> list[Path]:
    paths = [run_dir / f"quiesce_ack_producer_{i}" for i in range(vllm_dp)]
    paths.append(run_dir / "quiesce_ack_consumer")
    return paths


def _all_acks_present(run_dir: Path, vllm_dp: int) -> bool:
    return all(p.exists() for p in _quiesce_ack_paths(run_dir, vllm_dp))


def _cleanup_quiesce_files(run_dir: Path, vllm_dp: int) -> None:
    (run_dir / "quiesce_request").unlink(missing_ok=True)
    for p in _quiesce_ack_paths(run_dir, vllm_dp):
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
    new_checkpoint_path: str | None,
) -> None:
    from sae_lens.topology_control import TopologySpec, read_control_state, write_control_state

    state = read_control_state(control_state_path)
    state.topology = TopologySpec(
        vllm_tp=new_vllm_tp, vllm_dp=new_vllm_dp, sae_tp=new_sae_tp, sae_dp=new_sae_dp
    )
    state.phase = "RUNNING"
    if new_checkpoint_path is not None:
        state.checkpoint_path = new_checkpoint_path
    write_control_state(control_state_path, state)


def _find_latest_quiesce_checkpoint(run_dir: Path) -> str | None:
    """Find the most recent quiesce_* checkpoint directory under checkpoints/."""
    # cfg.checkpoint_path is set to run_dir/checkpoints, so quiesce checkpoints
    # are saved directly as run_dir/checkpoints/quiesce_N.
    # Also search one level deeper (run_dir/checkpoints/*/quiesce_N) for runs
    # that use a unique run-id subdirectory.
    if not run_dir.exists():
        return None
    candidates = []
    ckpt_base = run_dir / "checkpoints"
    if ckpt_base.is_dir():
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
    num_gpus: int,
    resume_from_checkpoint: str | None,
    log_path: Path | None = None,
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
    else:
        vllm_tp = initial_vllm_tp
        vllm_dp = initial_vllm_dp
        sae_tp = initial_sae_tp
        sae_dp = initial_sae_dp
        state = ControlState(
            phase="RUNNING",
            topology=TopologySpec(vllm_tp=vllm_tp, vllm_dp=vllm_dp, sae_tp=sae_tp, sae_dp=sae_dp),
            buffer_name="",  # workers will generate and we'll read it back
            buffer_params=BufferParams(num_chunks=0, chunk_size_tokens=0, d_model=0, dtype="bfloat16"),
            checkpoint_path=resume_from_checkpoint,
            next_claim_seq_at_quiesce=0,
            target_chunks=0,
        )
        write_control_state(control_state_path, state)

    logger.supervisor_start(
        topology={"vllm_tp": vllm_tp, "vllm_dp": vllm_dp, "sae_tp": sae_tp, "sae_dp": sae_dp},
        run_dir=str(run_dir),
    )

    # --- Main supervisor loop ---
    proc: subprocess.Popen | None = None
    master_port = _find_free_port()

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
            master_port=master_port,
            control_state_path=control_state_path,
        )
        print(f"[supervisor] launching: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        logger.new_workers_launched(
            topology={"vllm_tp": vllm_tp, "vllm_dp": vllm_dp, "sae_tp": sae_tp, "sae_dp": sae_dp},
            pid=p.pid,
        )
        return p

    # Handle recovery from a previous quiesce that was interrupted.
    if state.phase == "QUIESCING":
        print("[supervisor] recovering from interrupted quiesce")
        if _all_acks_present(run_dir, vllm_dp):
            # Acks are present but supervisor crashed before resetting buffer.
            # Read the pending topology request if it still exists.
            new_topo = None
            if topology_request_path.exists():
                new_topo = json.loads(topology_request_path.read_text())
            _reset_buffer(run_dir, control_state_path, new_topo["vllm_dp"] if new_topo else vllm_dp)
            if new_topo:
                ckpt = _find_latest_quiesce_checkpoint(run_dir)
                _apply_new_topology(
                    control_state_path,
                    new_topo["vllm_tp"], new_topo["vllm_dp"], new_topo["sae_tp"],
                    int(new_topo.get("sae_dp", 1)),
                    ckpt,
                )
                vllm_tp = new_topo["vllm_tp"]
                vllm_dp = new_topo["vllm_dp"]
                sae_tp = new_topo["sae_tp"]
                sae_dp = int(new_topo.get("sae_dp", 1))
            _cleanup_quiesce_files(run_dir, vllm_dp)
            topology_request_path.unlink(missing_ok=True)
        else:
            # Acks not all present — workers may have crashed during quiesce.
            # Restart with same topology from last checkpoint.
            state = read_control_state(control_state_path)
            state.phase = "RUNNING"
            write_control_state(control_state_path, state)
            _cleanup_quiesce_files(run_dir, vllm_dp)

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
                            topology={"vllm_tp": vllm_tp, "vllm_dp": vllm_dp, "sae_tp": sae_tp, "sae_dp": sae_dp}
                        )
                        break
                print(f"[supervisor] workers crashed (exit_code={exit_code}) — restarting")
                logger.workers_crashed(
                    exit_code=exit_code,
                    topology={"vllm_tp": vllm_tp, "vllm_dp": vllm_dp, "sae_tp": sae_tp, "sae_dp": sae_dp},
                )
                # Reset any WRITING chunks left by the crash.
                state = read_control_state(control_state_path)
                if state.buffer_name:
                    try:
                        _reset_buffer(run_dir, control_state_path, vllm_dp)
                    except Exception as e:
                        print(f"[supervisor] buffer reset failed (ignoring): {e}")
                _cleanup_quiesce_files(run_dir, vllm_dp)
                proc = _launch()
                continue

            # Check for topology switch request.
            if topology_request_path.exists():
                try:
                    new_topo = json.loads(topology_request_path.read_text())
                    new_vllm_tp = int(new_topo["vllm_tp"])
                    new_vllm_dp = int(new_topo["vllm_dp"])
                    new_sae_tp = int(new_topo["sae_tp"])
                    new_sae_dp = int(new_topo.get("sae_dp", 1))
                except Exception as e:
                    print(f"[supervisor] invalid topology_request.json: {e} — ignoring")
                    topology_request_path.unlink(missing_ok=True)
                    continue

                required_gpus = new_vllm_tp * new_vllm_dp + new_sae_tp * new_sae_dp
                if required_gpus > num_gpus:
                    print(
                        f"[supervisor] topology request requires {required_gpus} GPUs "
                        f"but only {num_gpus} available — ignoring"
                    )
                    topology_request_path.unlink(missing_ok=True)
                    continue

                from_topo = {"vllm_tp": vllm_tp, "vllm_dp": vllm_dp, "sae_tp": sae_tp, "sae_dp": sae_dp}
                to_topo = {"vllm_tp": new_vllm_tp, "vllm_dp": new_vllm_dp, "sae_tp": new_sae_tp, "sae_dp": new_sae_dp}

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
                    f"sae_tp={new_sae_tp} sae_dp={new_sae_dp}"
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

                # 2. Signal workers to quiesce — log the moment quiesce is sent
                quiesce_request_path.touch()
                logger.quiesce_signaled(from_topology=from_topo)

                # Wait for all acks.
                print("[supervisor] waiting for quiesce acks...")
                while not _all_acks_present(run_dir, vllm_dp):
                    if proc.poll() is not None:
                        print("[supervisor] workers exited during quiesce wait")
                        break
                    time.sleep(1.0)

                # 3. All acks received = consumer saved checkpoint and stopped consuming
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
                    new_vllm_tp, new_vllm_dp, new_sae_tp, new_sae_dp,
                    ckpt,
                )
                old_vllm_dp = vllm_dp
                vllm_tp = new_vllm_tp
                vllm_dp = new_vllm_dp
                sae_tp = new_sae_tp
                sae_dp = new_sae_dp

                # Clean up signal files and request (use old vllm_dp for ack paths).
                _cleanup_quiesce_files(run_dir, old_vllm_dp)
                topology_request_path.unlink(missing_ok=True)

                print(
                    f"[supervisor] launching new topology: "
                    f"vllm_tp={vllm_tp} vllm_dp={vllm_dp} sae_tp={sae_tp} sae_dp={sae_dp}"
                )
                # 5. Launch new workers — logged inside _launch()
                proc = _launch()

    except KeyboardInterrupt:
        print("[supervisor] interrupted — sending quiesce to workers")
        quiesce_request_path.touch()
        _wait_for_workers(proc, _WORKER_EXIT_TIMEOUT_S)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()


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
    parser.add_argument("--num-gpus", type=int, required=True, help="Total available GPUs")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--log-path",
        default=None,
        help="Path for JSONL event log. Default: results/topology/topology_run_YY_MM_DD_HH_MM_SS.jsonl",
    )
    args = parser.parse_args()

    run_supervisor(
        run_dir=Path(args.run_dir),
        worker_script=args.worker_script,
        worker_args=args.worker_args,
        initial_vllm_tp=args.vllm_tp,
        initial_vllm_dp=args.vllm_dp,
        initial_sae_tp=args.sae_tp,
        initial_sae_dp=args.sae_dp,
        num_gpus=args.num_gpus,
        resume_from_checkpoint=args.resume_from_checkpoint,
        log_path=Path(args.log_path) if args.log_path else None,
    )


if __name__ == "__main__":
    main()
