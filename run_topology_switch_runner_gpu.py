#!/usr/bin/env python3
"""
Demo: topology switch between (vllm_tp=1 sae_tp=1) and (vllm_dp=0 sae_tp=2)

Phase 1 — initial topology (2 GPUs):
  GPU 0: vLLM producer (vllm_tp=1, vllm_dp=1)
  GPU 1: SAE consumer  (sae_tp=1)

Phase 2 — after switch (2 GPUs):
  GPU 0+1: SAE consumer (vllm_dp=0, sae_tp=2) — no vLLM, SAE spans both GPUs

Usage:
  python3 scripts/run_topology_switch_runner_gpu.py run                # start supervisor
  python3 scripts/run_topology_switch_runner_gpu.py run --no_cleanup   # keep runtime shared memory
  python3 scripts/demo_topology_switch.py watch                       # watch buffer state
  python3 scripts/demo_topology_switch.py monitor [--verbose]         # auto-switch monitor
  python3 scripts/demo_topology_switch.py switch --topo TOPO_0VLLM_SAE2
  python3 scripts/demo_topology_switch.py switch --topo '{"vllm_tp":1,"vllm_dp":0,"sae_tp":2}'
  python3 scripts/demo_topology_switch.py clean                       # wipe run dir
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_DIR = Path("results/topology_runs/demo_topo_switch1")
LATEST_RUN_FILE = "latest_run.txt"
MODEL = "/data/models/Llama-3.1-8B"
DATASET = "../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048"
HOOK = "blocks.21.hook_resid_post"
HOOKS_2 = "blocks.21.hook_resid_post,blocks.31.hook_resid_post"
HOOKS_4 = "blocks.16.hook_resid_post,blocks.21.hook_resid_post,blocks.26.hook_resid_post,blocks.31.hook_resid_post"
NUM_CHUNKS = 96  # must match --streaming-num-chunks
STREAMING_PREFETCH_CHUNKS = 2
STREAMING_MIX_CHUNKS = 2
DONE_BUFFER_READY_PCT_THRESHOLD: float | None = 0.25
DONE_BUFFER_TARGET_TOPOLOGY = "TOPO_0VLLM_SAE_PP2"

BASE_WORKER_ARGS = [
    "--model-name", MODEL,
    "--dataset-path", DATASET,
    "--hook-name", HOOK,
    "--hook-names", HOOKS_2,
    "--k", "128",
    "--training-tokens", "8388608",
    "--train-batch-size-tokens", "2048",
    "--context-size", "2048",
    "--max-model-len", "2049",
    "--gpu-memory-utilization", "0.45",
    "--streaming-mode",
    "--streaming-chunk-size-tokens", "32768",
    "--streaming-num-chunks", str(NUM_CHUNKS),
    "--streaming-prefetch-chunks", str(STREAMING_PREFETCH_CHUNKS),
    "--streaming-mix-chunks", str(STREAMING_MIX_CHUNKS),
    "--streaming-mix-fraction", "0.5",
    "--save-timing-every-n-steps", "1",
    "--save-mse-every-n-steps", "1",
    "--save-memory-every-n-steps", "0",
    "--no-is-dataset-tokenized",
]


def _run_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _make_timestamped_run_dir(root: Path, *, timestamp: str | None = None) -> Path:
    timestamp = timestamp or _run_timestamp()
    base = root / f"run_{timestamp}"
    if not base.exists():
        return base
    for i in range(1, 1000):
        candidate = root / f"run_{timestamp}_{i:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique run directory under {root}")


def _write_latest_run_dir(run_dir: Path, *, root: Path | None = None) -> None:
    root = RUN_DIR if root is None else Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / LATEST_RUN_FILE).write_text(f"{run_dir.name}\n")


def _resolve_run_dir(root: Path | None = None) -> Path:
    root = RUN_DIR if root is None else Path(root)
    latest = root / LATEST_RUN_FILE
    try:
        text = latest.read_text().strip()
    except FileNotFoundError:
        return root
    if not text:
        return root
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if len(candidate.parts) == 1:
        return root / candidate
    return candidate


def _worker_args_for_run_dir(run_dir: Path) -> list[str]:
    return [
        *BASE_WORKER_ARGS,
        "--output-path", str(run_dir / "output"),
        "--checkpoint-path", str(run_dir / "checkpoints"),
    ]


WORKER_ARGS = _worker_args_for_run_dir(RUN_DIR)

# ---------------------------------------------------------------------------
# Topology configs
# ---------------------------------------------------------------------------

TOPO_1VLLM_1SAE = {"vllm_tp": 1, "vllm_dp": 1, "sae_tp": 1, "sae_dp": 1, "sae_pp_size": 1}
TOPO_0VLLM_SAE2 = {"vllm_tp": 1, "vllm_dp": 0, "sae_tp": 2, "sae_dp": 1, "sae_pp_size": 1}
TOPO_2VLLM_0SAE = {"vllm_tp": 1, "vllm_dp": 2, "sae_tp": 1, "sae_dp": 0, "sae_pp_size": 1}
TOPO_VLLMTP2_0SAE = {"vllm_tp": 2, "vllm_dp": 1, "sae_tp": 1, "sae_dp": 0, "sae_pp_size": 1}
# 4-hook PP demo: 1 vLLM + 1 SAE rank trains all 4 hooks ↔ 0 vLLM + 2 SAE PP ranks
# splitting 4 hooks 2+2.  Both topologies fit in 2 GPUs.
TOPO_1VLLM_1SAE_PP1 = {"vllm_tp": 1, "vllm_dp": 1, "sae_tp": 1, "sae_dp": 1, "sae_pp_size": 1}
TOPO_0VLLM_SAE_PP2 = {"vllm_tp": 1, "vllm_dp": 0, "sae_tp": 1, "sae_dp": 1, "sae_pp_size": 2}

# Named presets for --topo argument
PRESETS: dict[str, dict[str, int]] = {
    "TOPO_1VLLM_1SAE": TOPO_1VLLM_1SAE,
    "TOPO_0VLLM_SAE2": TOPO_0VLLM_SAE2,
    "TOPO_2VLLM_0SAE": TOPO_2VLLM_0SAE,
    "TOPO_1VLLM_1SAE_PP1": TOPO_1VLLM_1SAE_PP1,
    "TOPO_0VLLM_SAE_PP2": TOPO_0VLLM_SAE_PP2,
}

# ---------------------------------------------------------------------------
# Auto-switch condition types
# ---------------------------------------------------------------------------

@dataclass
class BufferReadyPct:
    """Trigger when buffer crosses a threshold and stays there.

    When the buffer first crosses `threshold`, records time0 and pct0.
    Fires if, at any of the `check_after_s` checkpoints, the buffer is still
    at or beyond pct0 (i.e. hasn't recovered). Satisfying any one checkpoint
    is enough. Timer resets if the buffer recovers past `reset_band` from the
    threshold (e.g. drops from 87% back below 75% for an "above 0.85" rule).
    """
    threshold: float
    direction: str            # "above" or "below"
    target_topology: dict[str, int]
    label: str = ""
    check_after_s: list[int] = field(default_factory=lambda: [10, 20])
    _first_crossed_at: float | None = field(default=None, repr=False, compare=False)
    _first_crossed_pct: float | None = field(default=None, repr=False, compare=False)

    def update(self, buf: dict | None) -> None:
        """Update timer state. Call every poll cycle regardless of cooldown."""
        if buf is None or buf.get("total", 0) == 0:
            return
        pct = buf["ready"] / buf["total"]
        crossed = (self.direction == "above" and pct >= self.threshold) or \
                  (self.direction == "below" and pct <= self.threshold)
        if crossed and self._first_crossed_at is None:
            self._first_crossed_at = time.time()
            self._first_crossed_pct = pct
            print(f"  [timer-start] '{self.label}': pct={pct:.0%} {self.direction} {self.threshold:.0%} → timer started")

    def check(self, state: dict[str, Any]) -> bool:
        if self._first_crossed_at is None or self._first_crossed_pct is None:
            return False
        buf = state.get("buffer")
        if buf is None or buf.get("total", 0) == 0:
            return False
        pct = buf["ready"] / buf["total"]
        elapsed = time.time() - self._first_crossed_at
        for t in self.check_after_s:
            if elapsed < t:
                continue
            # At this checkpoint: is buffer still above/below the threshold?
            still_holds = (self.direction == "above" and pct >= self.threshold) or \
                          (self.direction == "below" and pct <= self.threshold)
            if still_holds:
                return True
        return False

    def description(self) -> str:
        lbl = self.label or f"buffer_ready_{self.direction}_{self.threshold:.0%}"
        checks = "/".join(f"{t}s" for t in self.check_after_s)
        return f"{lbl}: ready {self.direction} {self.threshold:.0%} then [{checks} any] → {self.target_topology}"


@dataclass
class TokensConsumed:
    """Trigger when total consumed training tokens reaches a threshold."""
    threshold: int            # absolute token count
    target_topology: dict[str, int]
    label: str = ""

    def check(self, state: dict[str, Any]) -> bool:
        return state.get("tokens_consumed", 0) >= self.threshold

    def description(self) -> str:
        lbl = self.label or f"tokens_consumed_ge_{self.threshold}"
        return f"{lbl}: tokens_consumed >= {self.threshold} → {self.target_topology}"


@dataclass
class MonitorRuntime:
    cooldown_s: float = 30.0
    last_fired: dict[int, float] = field(default_factory=dict)
    prev_topology: dict[str, int] | None = None


def _format_topology(topo: dict[str, Any] | None) -> str:
    topo = topo or {}
    return (
        f"[vtp={int(topo.get('vllm_tp', 1))} "
        f"vdp={int(topo.get('vllm_dp', 0))} "
        f"stp={int(topo.get('sae_tp', 1))} "
        f"sdp={int(topo.get('sae_dp', 1))} "
        f"spp={int(topo.get('sae_pp_size', 1))}]"
    )


def _topology_matches(current: dict[str, Any], target: dict[str, int]) -> bool:
    return all(int(current.get(k, 1 if k in {"sae_dp", "sae_pp_size"} else 0)) == v for k, v in target.items())


def _monitor_state_line(
    *,
    phase: str,
    tokens_consumed: int,
    buf: dict[str, int] | None,
) -> str:
    line = f"monitor state phase={phase} tokens={tokens_consumed}"
    if buf is not None:
        total = int(buf.get("total", 0))
        ready = int(buf.get("ready", 0))
        pct = round(100 * ready / total) if total else 0
        line += (
            f" buffer ready={ready}/{total}({pct}%)"
            f" free={int(buf.get('free', 0))}"
            f" writ={int(buf.get('writing', 0))}"
            f" cons={int(buf.get('consuming', 0))}"
        )
    return line


def _monitor_signal_steps_line() -> str:
    return (
        "monitor signal steps topology_request -> phase=QUIESCING -> "
        "sae_stop_acquire_request -> sae_drain_ack -> "
        "vllm_stop_produce_request -> vllm_stopped_ack -> "
        "sae_finished/vllm_finished -> checkpoint -> relaunch"
    )


def _ack_paths_for_topology(
    run_dir: Path,
    topo: dict[str, Any],
) -> tuple[list[Path], list[Path], list[Path]]:
    vllm_dp = int(topo.get("vllm_dp", 0))
    sae_dp = int(topo.get("sae_dp", 1))
    sae_pp_size = int(topo.get("sae_pp_size", 1))
    sae_drain = [
        run_dir / f"sae_drain_ack_consumer_d{d}_pp{p}"
        for d in range(sae_dp)
        for p in range(max(sae_pp_size, 1))
    ]
    vllm_stopped = [
        run_dir / f"vllm_stopped_produce_ack_producer_{i}"
        for i in range(vllm_dp)
    ]
    final = [
        run_dir / f"sae_finished_ack_consumer_d{d}_pp{p}"
        for d in range(sae_dp)
        for p in range(max(sae_pp_size, 1))
    ] + [
        run_dir / f"vllm_finished_ack_producer_{i}"
        for i in range(vllm_dp)
    ]
    return sae_drain, vllm_stopped, final


def _monitor_ack_status_lines(
    *,
    run_dir: Path,
    ctrl: dict[str, Any],
) -> list[str]:
    topo = ctrl.get("topology", {})
    phase = str(ctrl.get("phase", "?"))
    sae_drain, vllm_stopped, final = _ack_paths_for_topology(run_dir, topo)
    request_path = run_dir / "topology_request.json"
    stop_acquire_path = run_dir / "sae_stop_acquire_request"
    stop_produce_path = run_dir / "vllm_stop_produce_request"

    steps: list[tuple[str, bool]] = [
        ("topology_request", request_path.exists()),
        ("phase=QUIESCING", phase == "QUIESCING"),
        ("sae_stop_acquire_request", stop_acquire_path.exists()),
    ]
    steps.extend((path.name, path.exists()) for path in sae_drain)
    steps.append(("vllm_stop_produce_request", stop_produce_path.exists()))
    if vllm_stopped:
        steps.extend((path.name, path.exists()) for path in vllm_stopped)
    else:
        steps.append(("vllm_stopped=skip", True))
    steps.extend((path.name, path.exists()) for path in final)

    done = [name for name, ok in steps if ok]
    missing = [name for name, ok in steps if not ok]
    non_skip_done = [name for name in done if not name.endswith("=skip")]
    current = non_skip_done[-1] if non_skip_done else (done[-1] if done else "none")

    def _group(paths: list[Path], skip_label: str | None = None) -> str:
        if not paths and skip_label is not None:
            return skip_label
        return ",".join(path.name for path in paths) if paths else "none"

    required = (
        f"sae_drain:{_group(sae_drain)}; "
        f"vllm_stopped:{_group(vllm_stopped, 'skip(no-vllm-producers)')}; "
        f"final:{_group(final)}"
    )
    return [
        f"monitor ack status current={current} required={required}",
        "monitor ack done " + (", ".join(done) if done else "none"),
        "monitor ack missing " + (", ".join(missing) if missing else "none"),
    ]


def _monitor_lines_and_maybe_fire(
    *,
    rules: list[BufferReadyPct | TokensConsumed],
    runtime: MonitorRuntime,
    ctrl: dict[str, Any],
    buf: dict[str, int] | None,
    tokens_consumed: int,
    now: float,
) -> list[str]:
    """Evaluate monitor rules once and return console lines for watch/monitor."""
    lines: list[str] = []
    run_dir = _resolve_run_dir()
    current_topo = ctrl.get("topology", {})
    phase = str(ctrl.get("phase", "?"))
    request_path = run_dir / "topology_request.json"

    if runtime.prev_topology is not None and current_topo != runtime.prev_topology:
        for rule in rules:
            if isinstance(rule, BufferReadyPct):
                rule._first_crossed_at = None
                rule._first_crossed_pct = None
    runtime.prev_topology = dict(current_topo)

    if request_path.exists():
        try:
            requested = json.loads(request_path.read_text())
        except json.JSONDecodeError:
            requested = {}
        lines.append(
            "monitor switch pending "
            f"current={_format_topology(current_topo)} "
            f"requested={_format_topology(requested)}"
        )
        lines.append(
            _monitor_state_line(
                phase=phase, tokens_consumed=tokens_consumed, buf=buf
            )
        )
        lines.extend(_monitor_ack_status_lines(run_dir=run_dir, ctrl=ctrl))
        return lines

    if phase == "QUIESCING":
        lines.append(
            _monitor_state_line(
                phase=phase, tokens_consumed=tokens_consumed, buf=buf
            )
        )
        lines.extend(_monitor_ack_status_lines(run_dir=run_dir, ctrl=ctrl))
        return lines

    for rule in rules:
        if isinstance(rule, BufferReadyPct):
            rule.update(buf)

    fired = False
    for i, rule in enumerate(rules):
        target = rule.target_topology
        target_s = _format_topology(target)
        if _topology_matches(current_topo, target):
            lines.append(f"monitor rule[{i}] {rule.label} target={target_s} status=already-target")
            continue
        remaining = runtime.cooldown_s - (now - runtime.last_fired.get(i, 0.0))
        if remaining > 0:
            lines.append(
                f"monitor rule[{i}] {rule.label} target={target_s} "
                f"status=cooldown remaining={remaining:.0f}s"
            )
            continue

        state = {
            "buffer": buf,
            "tokens_consumed": tokens_consumed,
            "topology": current_topo,
        }
        check = rule.check(state)
        if isinstance(rule, TokensConsumed):
            lines.append(
                f"monitor rule[{i}] {rule.label} target={target_s} "
                f"status=tokens consumed={tokens_consumed}/{rule.threshold} "
                f"check={check}"
            )
        else:
            pct = None
            if buf is not None and int(buf.get("total", 0)) > 0:
                pct = int(buf.get("ready", 0)) / int(buf["total"])
            pct_s = "n/a" if pct is None else f"{pct:.0%}"
            lines.append(
                f"monitor rule[{i}] {rule.label} target={target_s} "
                f"status=buffer ready={pct_s} threshold={rule.threshold:.0%} "
                f"check={check}"
            )
        if check:
            _request_switch(target, trigger="auto")
            runtime.last_fired[i] = now
            lines.insert(
                0,
                f"monitor FIRE {_format_topology(current_topo)} -> {target_s}",
            )
            lines.insert(1, _monitor_signal_steps_line())
            fired = True
            break

    if fired:
        lines.append(
            _monitor_state_line(
                phase=phase, tokens_consumed=tokens_consumed, buf=buf
            )
        )
    return lines


# Default rules:
#   1. After 2M tokens consumed → switch to no-vLLM SAE-TP2 (reliable demo trigger)
#   2. buffer ≥85% for 10s or 20s → no-vLLM SAE-TP2 (vLLM is faster than SAE)
#   3. buffer ≤15% for 10s or 20s → 1vLLM 1SAE (SAE is faster than vLLM)
DEFAULT_RULES: list[BufferReadyPct | TokensConsumed] = [
    BufferReadyPct(
        threshold=0.85,
        direction="above",
        target_topology=TOPO_0VLLM_SAE_PP2,#TOPO_0VLLM_SAE_PP2,
        label="buffer_full",
        check_after_s=[3, 6, 12, 18, 24, 36, 48, 60, 75, 90, 120, 150, 180, 240],
    ),
    BufferReadyPct(
        threshold=0.15,
        direction="below",
        target_topology=TOPO_1VLLM_1SAE,
        label="buffer_low",
        check_after_s=[3, 6, 12, 18, 24, 36, 48, 60, 75, 90, 120, 150, 180, 240], 
    ),
]

# ---------------------------------------------------------------------------
# Shared state readers
# ---------------------------------------------------------------------------

def _read_control_state() -> dict | None:
    p = _resolve_run_dir() / "control_state.json"
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_buffer_state() -> dict | None:
    """Read buffer slot counts directly from /dev/shm memmaps."""
    ctrl = _read_control_state()
    if ctrl is None or not ctrl.get("buffer_name"):
        return None
    try:
        import numpy as np
        name = ctrl["buffer_name"]
        bp = ctrl.get("buffer_params", {})
        num_chunks = bp.get("num_chunks", NUM_CHUNKS)
        state_path = Path("/dev/shm") / f"{name}_state.bin"
        if not state_path.exists():
            return None
        arr = np.memmap(str(state_path), dtype=np.int8, mode="r", shape=(num_chunks,))
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for v in arr:
            counts[int(v)] += 1
        return {
            "free": counts[0],
            "writing": counts[1],
            "ready": counts[2],
            "consuming": counts[3],
            "total": num_chunks,
        }
    except Exception:
        return None


def _read_tokens_consumed() -> int:
    """Read latest n_training_samples from timing_history.jsonl."""
    p = _resolve_run_dir() / "output" / "timing_history.jsonl"
    try:
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        if not lines:
            return 0
        r = json.loads(lines[-1])
        return r.get("n_training_samples", 0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


def _read_tokens_produced() -> int:
    """Cumulative tokens produced by vLLM (latest chunk_written.total_tokens)."""
    p = _resolve_run_dir() / "output" / "shm_log_vllm.jsonl"
    try:
        lines = p.read_text().splitlines()
        for line in reversed(lines):
            if not line.strip() or '"event": "chunk_written"' not in line:
                continue
            r = json.loads(line)
            return int(r.get("total_tokens", 0))
        return 0
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


def _current_topology() -> dict | None:
    ctrl = _read_control_state()
    if ctrl is None:
        return None
    return ctrl.get("topology")


def _request_switch(topo: dict[str, int], trigger: str = "manual") -> None:
    req = _resolve_run_dir() / "topology_request.json"
    req.write_text(json.dumps({**topo, "_trigger": trigger}))


def _topology_arg_for_supervisor(topo: str | dict[str, int]) -> str:
    if isinstance(topo, dict):
        return json.dumps(topo)
    if topo in PRESETS:
        return json.dumps(PRESETS[topo])
    return topo


def _active_topology_process_pids(
    ps_lines: list[str],
    run_dir: Path,
    *,
    current_pid: int | None = None,
) -> list[int]:
    marker = str(run_dir)
    process_markers = (
        "scripts/topology_supervisor.py",
        "run_sae_runner_gpu.py",
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
        if marker in command and any(token in command for token in process_markers):
            pids.append(pid)
    return sorted(set(pids))


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_existing_run_processes(
    run_dir: Path | None = None,
    timeout_s: float = 15.0,
) -> None:
    run_dir = _resolve_run_dir() if run_dir is None else Path(run_dir)
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: could not inspect process table: {result.stderr.strip()}")
        return
    pids = _active_topology_process_pids(
        result.stdout.splitlines(),
        run_dir,
        current_pid=os.getpid(),
    )
    if not pids:
        return
    print(f"Stopping existing topology processes for {run_dir}: {pids}")
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
        print(f"Force-killing topology processes that did not exit: {alive}")
        for pid in sorted(alive, reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _cleanup_shm_buffers() -> None:
    for f in Path("/dev/shm").glob("sae_buf_*"):
        f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(cleanup: bool = True) -> None:
    previous_run_dir = _resolve_run_dir()
    _terminate_existing_run_processes(run_dir=previous_run_dir)
    if cleanup:
        _cleanup_shm_buffers()
    else:
        print("Skipping topology runner shared-memory cleanup (--no_cleanup).")

    run_dir = _make_timestamped_run_dir(RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_latest_run_dir(run_dir)
    worker_args = _worker_args_for_run_dir(run_dir)

    print("=== Starting topology supervisor (initial: vllm_tp=1 vllm_dp=1 sae_tp=1) ===")
    print(f"Run dir: {run_dir}")
    print(f"Logs:    {run_dir}/output/shm_log_vllm.jsonl  (producer)")
    print(f"         {run_dir}/output/shm_log_sae.jsonl   (consumer)")
    if DONE_BUFFER_READY_PCT_THRESHOLD is not None:
        print(
            "Done-buffer trigger: "
            f"ready>{DONE_BUFFER_READY_PCT_THRESHOLD:.0%} -> "
            f"{DONE_BUFFER_TARGET_TOPOLOGY}"
        )
    print()
    print("In another terminal:")
    print(f"  python3 {__file__} watch                              # live buffer state")
    print(f"  python3 {__file__} monitor [--verbose]                # auto-switch monitor")
    print(f"  python3 {__file__} switch --topo TOPO_0VLLM_SAE2      # switch to no-vLLM SAE-TP2")
    print(f"  python3 {__file__} switch --topo TOPO_1VLLM_1SAE      # switch back to 1vLLM 1SAE")
    print(f"  python3 {__file__} switch --topo '{{\"vllm_tp\":1,\"vllm_dp\":0,\"sae_tp\":2}}'  # JSON")
    print()

    cmd = [
        sys.executable, "scripts/topology_supervisor.py",
        "--run-dir", str(run_dir),
        "--worker-script", "scripts/run_sae_runner_gpu.py",
        "--worker-args", " ".join(worker_args),
        "--vllm-tp", "1",
        "--vllm-dp", "1",
        "--sae-tp", "1",
        "--sae-pp-size", "1",
        "--num-gpus", "2",
    ]
    if DONE_BUFFER_READY_PCT_THRESHOLD is not None:
        target_topology_arg = _topology_arg_for_supervisor(DONE_BUFFER_TARGET_TOPOLOGY)
        cmd.extend(
            [
                "--done-buffer-ready-pct-threshold",
                str(DONE_BUFFER_READY_PCT_THRESHOLD),
                "--done-buffer-target-topology",
                target_topology_arg,
            ]
        )
    if not cleanup:
        # The supervisor has its own startup/finally cleanup path. Keep the
        # launcher flag effective for the full lifetime of this run.
        cmd.append("--no-cleanup-shm")
    subprocess.run(cmd, check=False)


def cmd_watch() -> None:
    start_time = time.time()
    monitor_runtime = MonitorRuntime(cooldown_s=30)
    print("=== Buffer state (Ctrl-C to stop) ===")

    # Read total training tokens from worker args
    try:
        idx = WORKER_ARGS.index("--training-tokens")
        total_training_tokens = int(WORKER_ARGS[idx + 1])
    except (ValueError, IndexError):
        total_training_tokens = 0

    while True:
        print("\033[2J\033[H", end="")  # clear screen
        # Two blank lines so the live header isn't shadowed when the shell
        # echoes a command above the watch output.
        print()
        print()
        elapsed = time.time() - start_time

        # --- Header: topology + phase (single line) ---
        ctrl_data = _read_control_state()
        if ctrl_data:
            topo = ctrl_data.get("topology", {})
            phase = ctrl_data.get("phase", "?")
            tp = (
                f"vtp={topo.get('vllm_tp',0)} vdp={topo.get('vllm_dp',0)} "
                f"stp={topo.get('sae_tp',0)} sdp={topo.get('sae_dp',0)} "
                f"spp={topo.get('sae_pp_size',1)}"
            )
            print(f"elapsed={elapsed:.0f}s  phase={phase}  topo=[{tp}]")
        else:
            print(f"elapsed={elapsed:.0f}s  (no control state yet)")

        # --- Token progress bars: vLLM produced + SAE consumed ---
        produced = _read_tokens_produced()
        consumed = _read_tokens_consumed()
        bar_len = 40

        def _bar(n: int, total: int) -> str:
            if total <= 0:
                return "░" * bar_len
            pct = min(n / total, 1.0)
            filled = int(bar_len * pct)
            return "█" * filled + "░" * (bar_len - filled)

        if total_training_tokens > 0:
            p_pct = min(produced / total_training_tokens, 1.0)
            c_pct = min(consumed / total_training_tokens, 1.0)
            print(
                f"vllm  [{_bar(produced, total_training_tokens)}] "
                f"{produced:,}/{total_training_tokens:,} ({p_pct:.1%})"
            )
            print(
                f"sae   [{_bar(consumed, total_training_tokens)}] "
                f"{consumed:,}/{total_training_tokens:,} ({c_pct:.1%})"
            )
        else:
            print(f"vllm  produced={produced:,}")
            print(f"sae   consumed={consumed:,}")

        # --- Buffer state ---
        print()
        buf = _read_buffer_state()
        if buf:
            total = buf["total"]
            ready = buf["ready"]
            pct = ready / total if total else 0
            bar_len = 40
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"buffer [{bar}] {pct:.0%}  free={buf['free']} writ={buf['writing']} ready={buf['ready']} cons={buf['consuming']}/{total}")
        else:
            print("buffer (not available yet)")

        # --- vLLM producer (last 3 events) ---
        print()
        run_dir = _resolve_run_dir()
        vllm_log = run_dir / "output" / "shm_log_vllm.jsonl"
        try:
            lines = [l for l in vllm_log.read_text().splitlines() if '"event": "chunk_written"' in l][-3:]
            if lines:
                print("vLLM  step   seq  infer")
                for line in lines:
                    r = json.loads(line)
                    print(f"      {r['step']:4d}  {r['seq_no']:4d}  {r['inference_time_s']:.2f}s")
            else:
                print("vLLM  (no data yet)")
        except (FileNotFoundError, KeyError):
            print("vLLM  (no data yet)")

        # --- SAE consumer (last 3 steps) ---
        print()
        timing_log = run_dir / "output" / "timing_history.jsonl"
        try:
            lines = [l for l in timing_log.read_text().splitlines() if '"vllm_step_time_s"' in l][-3:]
            if lines:
                print("SAE   step  wait    sae")
                for line in lines:
                    r = json.loads(line)
                    print(f"      {r['step']:4d}  {r['vllm_step_time_s']:.2f}s  {r['sae_time_s']:.2f}s")
            else:
                print("SAE   (no data yet)")
        except (FileNotFoundError, KeyError):
            print("SAE   (no data yet)")

        # --- Auto-switch monitor ---
        print()
        if ctrl_data:
            monitor_lines = _monitor_lines_and_maybe_fire(
                rules=DEFAULT_RULES,
                runtime=monitor_runtime,
                ctrl=ctrl_data,
                buf=buf,
                tokens_consumed=consumed,
                now=time.time(),
            )
            if monitor_lines:
                print("monitor")
                for line in monitor_lines:
                    print(f"  {line}")
            else:
                print("monitor (no trigger checks available)")
        else:
            print("monitor (waiting for control state)")

        # --- Quiesce files ---
        qfiles = list(run_dir.glob("quiesce_*"))
        if qfiles:
            print()
            print("quiesce: " + "  ".join(f.name for f in qfiles))

        time.sleep(1)


def cmd_monitor(rules: list | None = None, verbose: bool = False) -> None:
    """Auto-switch monitor. Evaluates rules every second and fires when conditions are met.

    Rules are evaluated independently. A rule is skipped if the current topology
    already matches its target (prevents re-triggering). A pending topology_request.json
    suppresses all triggers until the switch completes.
    """
    if rules is None:
        rules = DEFAULT_RULES

    print("=== Auto-switch monitor (Ctrl-C to stop) ===")
    for r in rules:
        print(f"  rule: {r.description()}")
    print()

    runtime = MonitorRuntime(cooldown_s=30)

    while True:
        ctrl = _read_control_state()
        if ctrl is None:
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for control_state.json...")
            time.sleep(1)
            continue

        buf = _read_buffer_state()
        tokens = _read_tokens_consumed()
        lines = _monitor_lines_and_maybe_fire(
            rules=rules,
            runtime=runtime,
            ctrl=ctrl,
            buf=buf,
            tokens_consumed=tokens,
            now=time.time(),
        )
        if verbose or any("FIRE" in line or "pending" in line for line in lines):
            for line in lines:
                print(f"[{time.strftime('%H:%M:%S')}] {line}")

        time.sleep(1)


def cmd_switch(topo_str: str) -> None:
    """Request a topology switch. topo_str is a preset name or JSON dict."""
    if topo_str in PRESETS:
        topo = PRESETS[topo_str]
    else:
        try:
            topo = json.loads(topo_str)
        except json.JSONDecodeError as e:
            print(f"Error: --topo must be a preset name or valid JSON: {e}")
            print(f"Available presets: {', '.join(PRESETS)}")
            sys.exit(1)

    required_keys = {"vllm_tp", "vllm_dp", "sae_tp"}
    missing = required_keys - topo.keys()
    if missing:
        print(f"Error: topology JSON missing keys: {missing}")
        sys.exit(1)
    topo.setdefault("sae_dp", 1)
    topo.setdefault("sae_pp_size", 1)

    print(f"=== Requesting topology switch → {topo} ===")
    req = _resolve_run_dir() / "topology_request.json"
    req.write_text(json.dumps(topo))
    print(f"Written to {req}")
    print("Watch the supervisor terminal for quiesce progress.")


def cmd_clean() -> None:
    print("=== Cleaning up ===")
    run_dir = _resolve_run_dir()
    _terminate_existing_run_processes(run_dir=run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    latest = RUN_DIR / LATEST_RUN_FILE
    if latest.exists() and _resolve_run_dir(RUN_DIR) == run_dir:
        latest.unlink()
    _cleanup_shm_buffers()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo topology switch controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    parser.add_argument(
        "--no_cleanup",
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=True,
        help=(
            "Keep topology-runner shared-memory artifacts when starting streaming "
            "mode (cleanup is enabled by default)."
        ),
    )

    run = sub.add_parser("run", help="Start the topology supervisor")
    # Also accept the option after the subcommand. SUPPRESS avoids replacing a
    # value supplied before the subcommand with the subparser default.
    run.add_argument(
        "--no_cleanup",
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep topology-runner shared-memory artifacts.",
    )
    sub.add_parser("watch", help="Live buffer state (1s refresh)")

    mon = sub.add_parser("monitor", help="Auto-switch monitor")
    mon.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-second state and rule skip reasons",
    )

    sw = sub.add_parser("switch", help="Request a topology switch")
    sw.add_argument(
        "--topo", required=True,
        help=(
            "Preset name or JSON topology. "
            f"Presets: {', '.join(PRESETS)}. "
            "JSON example: '{\"vllm_tp\":1,\"vllm_dp\":0,\"sae_tp\":2,\"sae_dp\":1}'. "
            "sae_dp=1 → has SAE, sae_dp=0 → no SAE (vLLM-only)."
        ),
    )

    sub.add_parser("clean", help="Wipe run dir and /dev/shm buffers")

    # Support legacy positional usage: demo_topology_switch.py switch (no --topo)
    args = parser.parse_args()
    if args.cmd is None:
        args.cmd = "run"

    if args.cmd == "run":
        cmd_run(cleanup=args.cleanup)
    elif args.cmd == "watch":
        cmd_watch()
    elif args.cmd == "monitor":
        cmd_monitor(verbose=args.verbose)
    elif args.cmd == "switch":
        cmd_switch(args.topo)
    elif args.cmd == "clean":
        cmd_clean()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
