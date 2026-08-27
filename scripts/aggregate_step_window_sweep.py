#!/usr/bin/env python3
"""Aggregate step-window profiling results into the total-time table.

This reports wall-clock time only: per-window per-step time and its mean. It
deliberately does NOT split the step into vLLM and SAE parts. Those splits were
previously taken from CPU timers measured without syncs inside the window, so
they were launch spans rather than costs and did not sum to the window. The
vLLM attribution now happens in scripts/build_vllm_sae_breakdown.py, from the
measured vLLM cadence plus the standalone per-call profile.

With TP/DDP/FSDP every rank writes its own profile file. The reported window
time is the max across ranks: a step is not finished until the slowest rank
finishes it, so the max is the step's true wall cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def per_window_max(records: list[dict]) -> dict[int, dict]:
    """Collapse ranks to one entry per window, keeping the slowest rank's time."""
    windows = sorted({record["window"] for record in records})
    by_window: dict[int, dict] = {}
    for window in windows:
        group = [r for r in records if r["window"] == window]
        slowest = max(group, key=lambda r: r["window_time_s"])
        entry = dict(slowest)
        entry["_slowest_rank"] = slowest["_rank"]
        by_window[window] = entry
    return by_window


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "step_window_sweep_20",
    )
    args = p.parse_args()

    manifest = [
        json.loads(line)
        for line in (args.out_root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    rows: list[dict] = []
    for spec in manifest:
        run_dir = Path(spec["output_path"])
        records = load_windows(run_dir)
        windows = per_window_max(records)
        row = {
            "run_id": spec["run_id"],
            "hooks": spec["hook_count"],
            "d_in": 4096,
            "d_sae": spec["d_sae"],
            "batch": spec["batch_tokens"],
            "k": 256,
            "tp": spec["sae_tp_size"],
            "dp": spec["sae_dp_size"],
            "mode": spec["mode"],
            "adam": "fused",
            "start_step": spec["start_step"],
            "window_steps": spec["window_steps"],
            "n_ranks": len({r["_rank"] for r in records}),
            "windows": [],
        }
        for idx in sorted(windows):
            record = windows[idx]
            row["windows"].append(
                {
                    "window": idx,
                    "steps": f"{record['start_step']}-{record['end_step']}",
                    "n_steps": record["steps"],
                    "complete": record["complete"],
                    "window_time_s": record["window_time_s"],
                    "per_step_ms": record["per_step_s"] * 1000.0
                    if record["per_step_s"]
                    else None,
                }
            )
        per_steps = [w["per_step_ms"] for w in row["windows"] if w["per_step_ms"]]
        row["mean_per_step_ms"] = sum(per_steps) / len(per_steps) if per_steps else None
        row["spread_pct"] = (
            (max(per_steps) - min(per_steps)) / min(per_steps) * 100.0
            if len(per_steps) > 1
            else None
        )
        rows.append(row)

    header = (
        f"{'run_id':26s} {'H':>2s} {'d_in':>5s} {'d_sae':>7s} {'batch':>6s} "
        f"{'k':>4s} {'tp':>3s} {'dp':>3s} {'mode':>7s} {'adam':>6s} "
        f"{'w1_steps':>9s} {'w1_ms':>9s} {'w2_steps':>9s} {'w2_ms':>9s} "
        f"{'mean_ms':>9s} {'spread%':>8s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        w = row["windows"]
        w1 = w[0] if len(w) > 0 else None
        w2 = w[1] if len(w) > 1 else None

        def fmt_steps(entry: dict | None) -> str:
            if entry is None:
                return "n/a"
            return entry["steps"] + ("" if entry["complete"] else "*")

        def fmt_ms(entry: dict | None) -> str:
            if entry is None or entry["per_step_ms"] is None:
                return "n/a"
            return f"{entry['per_step_ms']:.2f}"

        mean = (
            f"{row['mean_per_step_ms']:.2f}"
            if row["mean_per_step_ms"] is not None
            else "n/a"
        )
        spread = f"{row['spread_pct']:.2f}" if row["spread_pct"] is not None else "n/a"
        lines.append(
            f"{row['run_id']:26s} {row['hooks']:2d} {row['d_in']:5d} "
            f"{row['d_sae']:7d} {row['batch']:6d} {row['k']:4d} {row['tp']:3d} "
            f"{row['dp']:3d} {row['mode']:>7s} {row['adam']:>6s} "
            f"{fmt_steps(w1):>9s} {fmt_ms(w1):>9s} "
            f"{fmt_steps(w2):>9s} {fmt_ms(w2):>9s} "
            f"{mean:>9s} {spread:>8s}"
        )
    table = "\n".join(lines)
    (args.out_root / "step_window_table.txt").write_text(table + "\n")
    (args.out_root / "step_window_summary.json").write_text(json.dumps(rows, indent=2))
    print(table)


if __name__ == "__main__":
    main()
