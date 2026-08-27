#!/usr/bin/env python3
"""Derive the per-step vLLM cost and combine it with the measured total time.

Two inputs, both raw measurements:

1. The step-window total, `mean_ms` per step, from the sweep's window profiles.
   This is wall-clock, synced only at window boundaries.
2. The standalone vLLM profile, which reports the median wall time of one prefill
   call at a given (num_hooks, tp, B, context).

The vLLM cost per step is *derived here* rather than measured inside the training
step, because the in-step CPU timer is a launch span and cannot be subtracted
from a wall-clock window.

Deriving it needs the number of prefill calls a step pays for. That count is
COUNTED from the run's own timing_history.jsonl, not computed from a formula:

    calls_per_step = (total prefill calls in the window) / (steps in the window)

The mixing buffer decouples fetches from steps, so a step either pays for a whole
fetch or nothing, and a fetch may itself span several prefill calls. The number
of calls inside one fetch is recovered by dividing the fetch's measured duration
by the profiled per-call median. An earlier version of this script used
`batch_tokens / 2048`, which overestimated by up to 2x at batch 4096 and ignored
`sae_dp_size` entirely; the docstring of scripts/plan_step_window_sweep.py had
already recorded that this cadence must be measured rather than derived.

`residual_ms = total - vllm` is what remains after the vLLM prefill: SAE compute,
NCCL collectives, and per-step CPU work. It is reported as one quantity because
the run carries no sync-free way to split it; use nsys
(scripts/analyze_nsys_gap.py) when the split matters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_windows(run_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(run_dir.glob("step_window_profile_sae_rank*.jsonl")):
        rank = int(path.stem.rsplit("rank", 1)[1])
        for line in path.read_text().splitlines():
            if line:
                record = json.loads(line)
                record["_rank"] = rank
                records.append(record)
    return records


def collapse(records: list[dict]) -> list[dict]:
    """One entry per window, taking the slowest rank: a step ends when the last
    rank finishes it."""
    out = []
    for window in sorted({r["window"] for r in records}):
        group = [r for r in records if r["window"] == window]
        out.append(dict(max(group, key=lambda r: r["window_time_s"])))
    return out


def load_timing(run_dir: Path) -> list[dict]:
    path = run_dir / "timing_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def count_prefill_calls(
    timing: list[dict], lo: int, hi: int, per_call_ms: float
) -> tuple[float, int, list[float]]:
    """Prefill calls paid for by steps in [lo, hi].

    Each fetch that invokes vLLM contributes its measured duration; dividing by
    the profiled per-call median recovers how many prefill calls that fetch ran,
    since a fetch large enough to need several calls costs a multiple of one.
    Returns (calls, number of vLLM-invoking steps, the fetch durations in ms).
    """
    hits = [
        r
        for r in timing
        if lo <= r["step"] <= hi and (r.get("vllm_step_time_s") or 0.0) > 0.0
    ]
    durations = [r["vllm_step_time_s"] * 1000.0 for r in hits]
    calls = sum(d / per_call_ms for d in durations)
    return calls, len(hits), durations


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root", type=Path, default=ROOT / "results" / "step_window_sweep_20"
    )
    p.add_argument(
        "--vllm-json",
        type=Path,
        default=ROOT
        / "sae_lens"
        / "autoconfig"
        / "profile_results"
        / "vllm_multihook_v1"
        / "vllm_multihook_profile.json",
    )
    args = p.parse_args()

    # vLLM runs at tp=1 in every sweep config, so key the raw profile by hook
    # count. B and context are fixed at 1 x 2048 across these profile rows.
    vllm_rows = json.loads(args.vllm_json.read_text())["rows"]
    per_call_ms = {
        r["num_hooks"]: r["wall_ms_median"] for r in vllm_rows if r["tp"] == 1
    }

    manifest = [
        json.loads(line)
        for line in (args.sweep_root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    rows: list[dict] = []
    for spec in manifest:
        run_dir = Path(spec["output_path"])
        windows = collapse(load_windows(run_dir))
        if not windows:
            continue
        timing = load_timing(run_dir)
        call_ms = per_call_ms[spec["hook_count"]]

        steps = sum(w["steps"] for w in windows)
        total_ms = sum(w["window_time_s"] for w in windows) / steps * 1000.0

        calls = 0.0
        vllm_steps = 0
        durations: list[float] = []
        for w in windows:
            c, n, d = count_prefill_calls(
                timing, w["start_step"], w["end_step"], call_ms
            )
            calls += c
            vllm_steps += n
            durations.extend(d)

        calls_per_step = calls / steps if steps else 0.0
        vllm_ms = calls_per_step * call_ms
        residual_ms = total_ms - vllm_ms
        rows.append(
            {
                "run_id": spec["run_id"],
                "mode": spec["mode"],
                "H": spec["hook_count"],
                "d_sae": spec["d_sae"],
                "batch": spec["batch_tokens"],
                "tp": spec["sae_tp_size"],
                "dp": spec["sae_dp_size"],
                "steps": steps,
                "vllm_per_call_ms": call_ms,
                # Counted from the run, not derived from batch size.
                "vllm_invoking_steps": vllm_steps,
                "calls_per_step": calls_per_step,
                "vllm_ms": vllm_ms,
                "residual_ms": residual_ms,
                "total_ms": total_ms,
                "vllm_pct": vllm_ms / total_ms * 100.0,
                "residual_pct": residual_ms / total_ms * 100.0,
                "fetch_ms_observed": ";".join(f"{d:.0f}" for d in durations),
            }
        )

    header = (
        f"{'run_id':26s} {'H':>2s} {'d_sae':>7s} {'batch':>6s} {'tp':>3s} {'dp':>3s} "
        f"{'calls/step':>10s} {'vllm_ms':>9s} {'residual':>9s} {'total_ms':>9s}  "
        f"{'vllm%':>6s} {'resid%':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['run_id']:26s} {r['H']:2d} {r['d_sae']:7d} {r['batch']:6d} "
            f"{r['tp']:3d} {r['dp']:3d} {r['calls_per_step']:10.4f} "
            f"{r['vllm_ms']:9.1f} {r['residual_ms']:9.1f} {r['total_ms']:9.1f}  "
            f"{r['vllm_pct']:6.1f} {r['residual_pct']:7.1f}"
        )
    table = "\n".join(lines)

    out_dir = args.sweep_root
    (out_dir / "vllm_sae_breakdown.txt").write_text(table + "\n")
    with open(out_dir / "vllm_sae_breakdown.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(table)


if __name__ == "__main__":
    main()
