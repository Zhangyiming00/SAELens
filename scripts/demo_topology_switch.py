#!/usr/bin/env python3
"""
Demo: topology switch between (vllm_tp=1 sae_tp=1) and (vllm_dp=0 sae_tp=2)

Phase 1 — initial topology (2 GPUs):
  GPU 0: vLLM producer (vllm_tp=1, vllm_dp=1)
  GPU 1: SAE consumer  (sae_tp=1)

Phase 2 — after switch (2 GPUs):
  GPU 0+1: SAE consumer (vllm_dp=0, sae_tp=2) — no vLLM, SAE spans both GPUs

Usage:
  python3 scripts/demo_topology_switch.py run                         # start supervisor
  python3 scripts/demo_topology_switch.py watch                       # watch buffer state
  python3 scripts/demo_topology_switch.py monitor [--verbose]         # auto-switch monitor
  python3 scripts/demo_topology_switch.py switch --topo TOPO_0VLLM_SAE2
  python3 scripts/demo_topology_switch.py switch --topo '{"vllm_tp":1,"vllm_dp":0,"sae_tp":2}'
  python3 scripts/demo_topology_switch.py clean                       # wipe run dir
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_DIR = Path("/tmp/demo_topo_switch")
MODEL = "/data/models/Llama-3.1-8B"
DATASET = "/data/fineweb-edu/sample/10BT/000_00000.parquet"
HOOK = "blocks.21.hook_resid_post"
NUM_CHUNKS = 128  # must match --streaming-num-chunks

WORKER_ARGS = [
    "--model-name", MODEL,
    "--dataset-path", DATASET,
    "--hook-name", HOOK,
    "--d-sae", "65536",
    "--k", "128",
    "--training-tokens", "2048000",
    "--train-batch-size-tokens", "2048",
    "--context-size", "2048",
    "--max-model-len", "2049",
    "--gpu-memory-utilization", "0.45",
    "--streaming-chunk-size-tokens", "131072",
    "--streaming-num-chunks", str(NUM_CHUNKS),
    "--output-path", str(RUN_DIR / "output"),
    "--checkpoint-path", str(RUN_DIR / "checkpoints"),
    "--save-timing-every-n-steps", "1",
    "--save-mse-every-n-steps", "1",
    "--save-memory-every-n-steps", "0",
    "--no-is-dataset-tokenized",
]

# ---------------------------------------------------------------------------
# Topology configs
# ---------------------------------------------------------------------------

TOPO_1VLLM_1SAE = {"vllm_tp": 1, "vllm_dp": 1, "sae_tp": 1, "sae_dp": 1}
TOPO_0VLLM_SAE2 = {"vllm_tp": 1, "vllm_dp": 0, "sae_tp": 2, "sae_dp": 1}
TOPO_2VLLM_0SAE = {"vllm_tp": 1, "vllm_dp": 2, "sae_tp": 1, "sae_dp": 0}
TOPO_VLLMTP2_0SAE = {"vllm_tp": 2, "vllm_dp": 1, "sae_tp": 1, "sae_dp": 0}

# Named presets for --topo argument
PRESETS: dict[str, dict[str, int]] = {
    "TOPO_1VLLM_1SAE": TOPO_1VLLM_1SAE,
    "TOPO_0VLLM_SAE2": TOPO_0VLLM_SAE2,
    "TOPO_2VLLM_0SAE": TOPO_2VLLM_0SAE,
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


# Default rules:
#   1. After 2M tokens consumed → switch to no-vLLM SAE-TP2 (reliable demo trigger)
#   2. buffer ≥85% for 10s or 20s → no-vLLM SAE-TP2 (vLLM is faster than SAE)
#   3. buffer ≤15% for 10s or 20s → 1vLLM 1SAE (SAE is faster than vLLM)
DEFAULT_RULES: list[BufferReadyPct | TokensConsumed] = [
    BufferReadyPct(
        threshold=0.82,
        direction="above",
        target_topology=TOPO_0VLLM_SAE2,
        label="buffer_full",
        check_after_s=[6, 12,18],
    ),
    BufferReadyPct(
        threshold=0.18,
        direction="below",
        target_topology=TOPO_1VLLM_1SAE,
        label="buffer_low",
        check_after_s=[6, 12,18],
    ),
]

# ---------------------------------------------------------------------------
# Shared state readers
# ---------------------------------------------------------------------------

def _read_control_state() -> dict | None:
    p = RUN_DIR / "control_state.json"
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
    p = RUN_DIR / "output" / "timing_history.jsonl"
    try:
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        if not lines:
            return 0
        r = json.loads(lines[-1])
        return r.get("n_training_samples", 0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


def _current_topology() -> dict | None:
    ctrl = _read_control_state()
    if ctrl is None:
        return None
    return ctrl.get("topology")


def _request_switch(topo: dict[str, int], trigger: str = "manual") -> None:
    req = RUN_DIR / "topology_request.json"
    req.write_text(json.dumps({**topo, "_trigger": trigger}))

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run() -> None:
    import subprocess
    print("=== Starting topology supervisor (initial: vllm_tp=1 vllm_dp=1 sae_tp=1) ===")
    print(f"Run dir: {RUN_DIR}")
    print(f"Logs:    {RUN_DIR}/output/shm_log_vllm.jsonl  (producer)")
    print(f"         {RUN_DIR}/output/shm_log_sae.jsonl   (consumer)")
    print()
    print("In another terminal:")
    print(f"  python3 {__file__} watch                              # live buffer state")
    print(f"  python3 {__file__} monitor [--verbose]                # auto-switch monitor")
    print(f"  python3 {__file__} switch --topo TOPO_0VLLM_SAE2      # switch to no-vLLM SAE-TP2")
    print(f"  python3 {__file__} switch --topo TOPO_1VLLM_1SAE      # switch back to 1vLLM 1SAE")
    print(f"  python3 {__file__} switch --topo '{{\"vllm_tp\":1,\"vllm_dp\":0,\"sae_tp\":2}}'  # JSON")
    print()

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/topology_supervisor.py",
        "--run-dir", str(RUN_DIR),
        "--worker-script", "scripts/run_sae_runner_gpu.py",
        "--worker-args", " ".join(WORKER_ARGS),
        "--vllm-tp", "1",
        "--vllm-dp", "1",
        "--sae-tp", "1",
        "--num-gpus", "2",
    ]
    subprocess.run(cmd, check=False)


def cmd_watch() -> None:
    start_time = time.time()
    print("=== Buffer state (Ctrl-C to stop) ===")

    # Read total training tokens from worker args
    try:
        idx = WORKER_ARGS.index("--training-tokens")
        total_training_tokens = int(WORKER_ARGS[idx + 1])
    except (ValueError, IndexError):
        total_training_tokens = 0

    while True:
        print("\033[2J\033[H", end="")  # clear screen
        elapsed = time.time() - start_time

        # --- Header: topology + phase (single line) ---
        ctrl_data = _read_control_state()
        if ctrl_data:
            topo = ctrl_data.get("topology", {})
            phase = ctrl_data.get("phase", "?")
            tp = f"vtp={topo.get('vllm_tp',0)} vdp={topo.get('vllm_dp',0)} stp={topo.get('sae_tp',0)} sdp={topo.get('sae_dp',0)}"
            print(f"elapsed={elapsed:.0f}s  phase={phase}  topo=[{tp}]")
        else:
            print(f"elapsed={elapsed:.0f}s  (no control state yet)")

        # --- Token progress bar ---
        tokens = _read_tokens_consumed()
        if total_training_tokens > 0:
            tok_pct = min(tokens / total_training_tokens, 1.0)
            bar_len = 40
            filled = int(bar_len * tok_pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"tokens [{bar}] {tokens:,}/{total_training_tokens:,} ({tok_pct:.1%})")
        else:
            print(f"tokens {tokens:,}")

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
        vllm_log = RUN_DIR / "output" / "shm_log_vllm.jsonl"
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
        timing_log = RUN_DIR / "output" / "timing_history.jsonl"
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

        # --- Quiesce files ---
        qfiles = list(RUN_DIR.glob("quiesce_*"))
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

    # Cooldown: after firing a rule, wait this many seconds before re-evaluating
    COOLDOWN_S = 30
    last_fired: dict[int, float] = {}  # rule index → timestamp
    prev_topo: dict | None = None

    while True:
        ctrl = _read_control_state()
        if ctrl is None:
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for control_state.json...")
            time.sleep(1)
            continue

        current_topo = ctrl.get("topology", {})

        # Reset all timers when topology changes — new topology gets a fresh window
        if prev_topo is not None and current_topo != prev_topo:
            print(f"  [topo-changed] {prev_topo} → {current_topo}: resetting all rule timers")
            for rule in rules:
                if isinstance(rule, BufferReadyPct):
                    rule._first_crossed_at = None
                    rule._first_crossed_pct = None
        prev_topo = current_topo

        # Don't fire while a switch is already pending
        if (RUN_DIR / "topology_request.json").exists():
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] switch pending, skipping")
            time.sleep(1)
            continue

        # Don't fire while quiescing
        if ctrl.get("phase") == "QUIESCING":
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] QUIESCING, skipping")
            time.sleep(1)
            continue

        buf = _read_buffer_state()
        tokens = _read_tokens_consumed()

        # Always update timers so they stay current even during cooldown/skip
        for rule in rules:
            if isinstance(rule, BufferReadyPct):
                rule.update(buf)

        # Always print a compact status line every second
        buf_str = "no-buf"
        if buf:
            pct = buf["ready"] / buf["total"] if buf["total"] else 0
            buf_str = f"ready={buf['ready']}/{buf['total']}({pct:.0%})"
        rule_states = []
        for i, rule in enumerate(rules):
            if isinstance(rule, BufferReadyPct):
                if rule._first_crossed_at is None:
                    rule_states.append(f"r{i}[{rule.label}:waiting]")
                else:
                    elapsed = time.time() - rule._first_crossed_at
                    rule_states.append(f"r{i}[{rule.label}:pct0={rule._first_crossed_pct:.0%} t={elapsed:.0f}s]")
            else:
                rule_states.append(f"r{i}[{rule.label}:tokens={tokens}]")
        print(f"[{time.strftime('%H:%M:%S')}] topo={current_topo} {buf_str} tokens={tokens} | {' '.join(rule_states)}")

        state = {
            "buffer": buf,
            "tokens_consumed": tokens,
            "topology": current_topo,
        }

        now = time.time()
        for i, rule in enumerate(rules):
            target = rule.target_topology
            # Skip if already on target topology
            if all(current_topo.get(k) == v for k, v in target.items()):
                print(f"  rule[{i}] SKIP: already on target {target}")
                continue

            # Skip if in cooldown
            remaining = COOLDOWN_S - (now - last_fired.get(i, 0))
            if remaining > 0:
                print(f"  rule[{i}] SKIP: cooldown {remaining:.0f}s remaining")
                continue

            result = rule.check(state)
            if isinstance(rule, BufferReadyPct):
                if rule._first_crossed_at is None:
                    reason = "no threshold cross yet"
                else:
                    elapsed = time.time() - rule._first_crossed_at
                    buf_pct = (buf["ready"] / buf["total"]) if buf and buf["total"] else None
                    passed_checks = [t for t in rule.check_after_s if elapsed >= t]
                    pending_checks = [t for t in rule.check_after_s if elapsed < t]
                    if buf_pct is not None:
                        still_holds = (rule.direction == "above" and buf_pct >= rule.threshold) or \
                                      (rule.direction == "below" and buf_pct <= rule.threshold)
                        reason = (
                            f"elapsed={elapsed:.0f}s pct0={rule._first_crossed_pct:.0%} "
                            f"cur={buf_pct:.0%} still_holds={still_holds} "
                            f"passed_checks={passed_checks} pending={pending_checks}"
                        )
                    else:
                        reason = f"elapsed={elapsed:.0f}s buf=None"
                print(f"  rule[{i}] check={result} ({reason})")
            else:
                print(f"  rule[{i}] check={result} (tokens={tokens} need={rule.threshold})")

            if result:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] >>> rule '{rule.description()}' FIRED → requesting {target}")
                _request_switch(target, trigger="auto")
                last_fired[i] = now
                break

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

    print(f"=== Requesting topology switch → {topo} ===")
    req = RUN_DIR / "topology_request.json"
    req.write_text(json.dumps(topo))
    print(f"Written to {req}")
    print("Watch the supervisor terminal for quiesce progress.")


def cmd_clean() -> None:
    print("=== Cleaning up ===")
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    for f in Path("/dev/shm").glob("sae_buf_*"):
        f.unlink(missing_ok=True)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo topology switch controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    sub.add_parser("run", help="Start the topology supervisor")
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
        cmd_run()
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
