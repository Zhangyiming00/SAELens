#!/usr/bin/env python3
"""Summarize streaming-mode producer/consumer window timings.

Streaming runs are split across two independent loops:
- vLLM producer timing lives in `timing_history_vllm.jsonl`
- SAE consumer timing lives in `timing_history.jsonl`

This script reads both, collapses each side to per-step wall time, and reports
the slow side as the bottleneck for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _candidate_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "output").is_dir():
        return [run_dir / "output", run_dir]
    return [run_dir]


def collapse_windows(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"total_ms": 0.0, "per_step_ms": 0.0, "steps": 0}
    steps = sum(int(r.get("steps", 0)) for r in records)
    total_s = sum(float(r.get("window_time_s", 0.0)) for r in records)
    return {
        "total_ms": total_s * 1000.0,
        "per_step_ms": (total_s / steps * 1000.0) if steps else 0.0,
        "steps": steps,
    }


def load_streaming_side(run_dir: Path, name: str) -> dict[str, float]:
    if name == "vllm":
        paths = sorted(run_dir.glob("timing_history_vllm*.jsonl"))
    else:
        paths = sorted(run_dir.glob("timing_history*.jsonl"))
        paths = [p for p in paths if p.name != "timing_history_vllm.jsonl"]
    if not paths:
        for candidate in _candidate_dirs(run_dir):
            if name == "vllm":
                paths = sorted(candidate.glob("timing_history_vllm*.jsonl"))
            else:
                paths = sorted(candidate.glob("timing_history*.jsonl"))
                paths = [p for p in paths if p.name != "timing_history_vllm.jsonl"]
            if paths:
                break
    total: list[dict] = []
    for path in paths:
        total.extend(load_jsonl(path))
    return collapse_windows(total)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results" / "streaming_single",
        help="Root containing run subdirs with streaming timing logs.",
    )
    args = p.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        vllm = load_streaming_side(run_dir, "vllm")
        sae = load_streaming_side(run_dir, "sae")
        if vllm["steps"] == 0 and sae["steps"] == 0:
            continue
        bottleneck = max(vllm["per_step_ms"], sae["per_step_ms"])
        rows.append(
            {
                "run": run_dir.name,
                "vllm_ms": round(vllm["per_step_ms"], 2),
                "sae_ms": round(sae["per_step_ms"], 2),
                "total_ms": round(bottleneck, 2),
                "vllm_steps": vllm["steps"],
                "sae_steps": sae["steps"],
            }
        )

    if not rows:
        print("no streaming runs found")
        return

    rows.sort(key=lambda r: r["run"])
    header = f"{'run':30s} {'vllm_ms':>10s} {'sae_ms':>10s} {'total_ms':>10s}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['run']:30s} {r['vllm_ms']:10.2f} {r['sae_ms']:10.2f} {r['total_ms']:10.2f}"
        )
    table = "\n".join(lines)

    out_dir = args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "streaming_summary.txt").write_text(table + "\n")
    with open(out_dir / "streaming_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(table)


if __name__ == "__main__":
    main()
