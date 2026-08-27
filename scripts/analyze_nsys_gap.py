#!/usr/bin/env python3
"""Attribute the step-window `gap` column against nsys hardware timelines.

The wall-clock table reports `gap = total - vllm - sae`, where `vllm` and `sae`
come from CPU timers taken without syncs inside the window. Those timers measure
*launch* spans, so anything whose GPU tail outlives its launch span, and every
CPU stretch not inside either timer, lands in `gap`. This script replaces the
subtraction with a measured timeline.

For each profiled step it builds the union of GPU activity intervals (kernel,
memcpy, memset) and reports, over the step's wall interval:

- `gpu_busy_ms`: wall time in which at least one GPU activity was in flight
- `gpu_idle_ms`: the complement, i.e. wall time with no GPU work anywhere
- per-NVTX-phase GPU busy time, attributing each activity to the innermost
  enclosing NVTX range on the launching thread via `correlationId`

With more than one rank, per-rank timelines are computed separately and the step
is summarized by its slowest rank, matching how the wall-clock table collapses
ranks.

Requires the sqlite export produced by scripts/run_nsys_gap_diagnosis.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# NVTX ranges to attribute GPU work to, innermost first: an activity is charged
# to the first range in this list that encloses its launch.
PHASE_TEXTS = [
    "multi_sae:unified_forward",
    "multi_sae:combined_backward",
    "nccl:multi_sae_ddp_combined_backward",
    "multi_sae:optimizer_step",
    "multi_sae:optimizer_unscale",
    "multi_sae:scaler_update",
    "multi_sae:stats_sync_tail",
    "multi_sae:data_fetch",
    "multi_sae:train_step",
]


@dataclass
class Interval:
    start: int
    end: int


def union_length(intervals: list[Interval], lo: int, hi: int) -> int:
    """Total length of the union of `intervals`, clipped to [lo, hi)."""
    clipped = [
        (max(i.start, lo), min(i.end, hi))
        for i in intervals
        if i.end > lo and i.start < hi
    ]
    if not clipped:
        return 0
    clipped.sort()
    total = 0
    cur_start, cur_end = clipped[0]
    for start, end in clipped[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def table_names(con: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }


def load_activities(
    con: sqlite3.Connection,
) -> dict[int, list[tuple[int, int, int, str]]]:
    """GPU activities per pid: (start, end, correlationId, name)."""
    names = table_names(con)
    out: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    if "CUPTI_ACTIVITY_KIND_KERNEL" in names:
        for start, end, corr, pid, name in con.execute(
            "select k.start, k.end, k.correlationId, k.globalPid >> 24, "
            "coalesce(s.value, 'kernel') "
            "from CUPTI_ACTIVITY_KIND_KERNEL k "
            "left join StringIds s on s.id = k.shortName "
            "where k.start is not null and k.end is not null"
        ):
            out[int(pid)].append((int(start), int(end), int(corr), str(name)))
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]"),
    ):
        if table not in names:
            continue
        for start, end, corr, pid in con.execute(
            f"select start, end, correlationId, globalPid >> 24 from {table} "
            "where start is not null and end is not null"
        ):
            out[int(pid)].append((int(start), int(end), int(corr), label))
    return out


def load_runtime_by_corr(con: sqlite3.Connection) -> dict[int, tuple[int, int, int]]:
    """correlationId -> (launch start, launch end, launching globalTid)."""
    out: dict[int, tuple[int, int, int]] = {}
    for start, end, corr, tid in con.execute(
        "select start, end, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where correlationId is not null"
    ):
        out[int(corr)] = (int(start), int(end), int(tid))
    return out


def load_nvtx(con: sqlite3.Connection) -> dict[str, list[tuple[int, int, int]]]:
    """NVTX text -> list of (start, end, globalTid)."""
    out: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for text, start, end, tid in con.execute(
        "select text, start, end, globalTid from NVTX_EVENTS "
        "where text is not null and start is not null and end is not null"
    ):
        out[str(text)].append((int(start), int(end), int(tid)))
    for ranges in out.values():
        ranges.sort()
    return out


def phase_of_launch(
    nvtx: dict[str, list[tuple[int, int, int]]], t: int, tid: int
) -> str:
    for text in PHASE_TEXTS:
        for start, end, range_tid in nvtx.get(text, []):
            if range_tid == tid and start <= t < end:
                return text
            if start > t:
                break
    return "outside_nvtx"


def analyze_run(sqlite_path: Path, spec: dict) -> dict:
    con = sqlite3.connect(sqlite_path)
    activities = load_activities(con)
    runtime = load_runtime_by_corr(con)
    nvtx = load_nvtx(con)
    con.close()

    steps = sorted(nvtx.get("multi_sae:train_step", []))
    fetches = sorted(nvtx.get("multi_sae:data_fetch", []))
    if not steps:
        return {"run_id": spec["run_id"], "error": "no multi_sae:train_step NVTX"}

    pids = sorted(activities)
    per_pid_steps: dict[int, list[dict]] = {}
    for pid in pids:
        acts = activities[pid]
        pid_steps = [s for s in steps if _tid_pid(s[2]) == pid]
        pid_fetches = [f for f in fetches if _tid_pid(f[2]) == pid]
        if not pid_steps:
            continue
        # A step's wall interval runs from the start of the data fetch that feeds
        # it to the end of its train_step range, which is what the window timer
        # covers per step.
        rows = []
        for idx, (s_start, s_end, _tid) in enumerate(pid_steps):
            prior = [f for f in pid_fetches if f[0] <= s_start]
            wall_start = prior[-1][0] if prior else s_start
            wall_end = s_end
            busy = union_length(
                [Interval(a, b) for a, b, _, _ in acts], wall_start, wall_end
            )
            wall = wall_end - wall_start
            by_phase: dict[str, list[Interval]] = defaultdict(list)
            by_thread: dict[str, list[Interval]] = defaultdict(list)
            kernel_sum: dict[str, float] = defaultdict(float)
            # Launch-to-completion lag: how far a phase's GPU work runs past the
            # end of the CPU range that launched it. This is what the CPU timers
            # miss and the `gap` column absorbs.
            phase_last_end: dict[str, int] = {}
            for a_start, a_end, corr, name in acts:
                if a_end <= wall_start or a_start >= wall_end:
                    continue
                launch = runtime.get(corr)
                phase = (
                    phase_of_launch(nvtx, launch[0], launch[2])
                    if launch
                    else "no_runtime_record"
                )
                by_phase[phase].append(Interval(a_start, a_end))
                kernel_sum[name] += (min(a_end, wall_end) - max(a_start, wall_start)) / 1e6
                phase_last_end[phase] = max(phase_last_end.get(phase, 0), a_end)
                if phase in ("outside_nvtx", "no_runtime_record"):
                    tid = launch[2] if launch else 0
                    by_thread[f"{phase}_tid{tid & 0xFFFFFF}"].append(
                        Interval(a_start, a_end)
                    )
            tails: dict[str, float] = {}
            for text in PHASE_TEXTS:
                enclosing = [
                    (start, end)
                    for start, end, tid in nvtx.get(text, [])
                    if _tid_pid(tid) == pid and start >= wall_start and end <= wall_end
                ]
                if enclosing and text in phase_last_end:
                    cpu_end = max(end for _, end in enclosing)
                    tails[text] = max(0, phase_last_end[text] - cpu_end) / 1e6
            rows.append(
                {
                    "idx": idx,
                    "pid": pid,
                    "wall_ms": wall / 1e6,
                    "gpu_busy_ms": busy / 1e6,
                    "gpu_idle_ms": (wall - busy) / 1e6,
                    "phase_busy_ms": {
                        phase: union_length(ivs, wall_start, wall_end) / 1e6
                        for phase, ivs in sorted(by_phase.items())
                    },
                    "phase_gpu_tail_ms": tails,
                    "unattributed_by_thread_ms": {
                        key: union_length(ivs, wall_start, wall_end) / 1e6
                        for key, ivs in sorted(by_thread.items())
                    },
                    "top_kernels_ms": dict(
                        sorted(kernel_sum.items(), key=lambda kv: -kv[1])[:12]
                    ),
                }
            )
        per_pid_steps[pid] = rows

    # Steady-state steps only: drop the first two, which carry warmup.
    summary_rows = []
    n_steps = min(len(rows) for rows in per_pid_steps.values())
    for idx in range(2, n_steps):
        group = [per_pid_steps[pid][idx] for pid in per_pid_steps]
        slowest = max(group, key=lambda r: r["wall_ms"])
        summary_rows.append(slowest)

    if not summary_rows:
        return {"run_id": spec["run_id"], "error": "no steady-state steps"}

    def avg(key: str) -> float:
        return sum(r[key] for r in summary_rows) / len(summary_rows)

    phase_keys = sorted({k for r in summary_rows for k in r["phase_busy_ms"]})
    phase_avg = {
        k: sum(r["phase_busy_ms"].get(k, 0.0) for r in summary_rows)
        / len(summary_rows)
        for k in phase_keys
    }
    tail_keys = sorted({k for r in summary_rows for k in r["phase_gpu_tail_ms"]})
    tail_avg = {
        k: sum(r["phase_gpu_tail_ms"].get(k, 0.0) for r in summary_rows)
        / len(summary_rows)
        for k in tail_keys
    }
    thread_keys = sorted(
        {k for r in summary_rows for k in r["unattributed_by_thread_ms"]}
    )
    thread_avg = {
        k: sum(r["unattributed_by_thread_ms"].get(k, 0.0) for r in summary_rows)
        / len(summary_rows)
        for k in thread_keys
    }
    kernel_keys = {k for r in summary_rows for k in r["top_kernels_ms"]}
    kernel_avg = {
        k: sum(r["top_kernels_ms"].get(k, 0.0) for r in summary_rows)
        / len(summary_rows)
        for k in kernel_keys
    }
    return {
        "phase_gpu_tail_ms": tail_avg,
        "unattributed_by_thread_ms": thread_avg,
        "top_kernels_ms": dict(
            sorted(kernel_avg.items(), key=lambda kv: -kv[1])[:15]
        ),
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "H": spec["hook_count"],
        "d_sae": spec["d_sae"],
        "batch": spec["batch_tokens"],
        "tp": spec["sae_tp_size"],
        "dp": spec["sae_dp_size"],
        "n_ranks": len(per_pid_steps),
        "n_steps_analyzed": len(summary_rows),
        "wall_ms": avg("wall_ms"),
        "gpu_busy_ms": avg("gpu_busy_ms"),
        "gpu_idle_ms": avg("gpu_idle_ms"),
        "gpu_idle_pct": avg("gpu_idle_ms") / avg("wall_ms") * 100.0,
        "phase_busy_ms": phase_avg,
        "per_step": summary_rows,
    }


def _tid_pid(global_tid: int) -> int:
    return global_tid >> 24


def load_window_profile(run_dir: Path) -> dict:
    """Wall-clock ground truth written by the step-window profiler."""
    out: dict[str, float] = {}
    records = []
    for path in sorted(run_dir.glob("step_window_profile_sae_rank*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                records.append(json.loads(line))
    if not records:
        return out
    by_window: dict[int, dict] = {}
    for r in records:
        w = r["window"]
        if w not in by_window or r["window_time_s"] > by_window[w]["window_time_s"]:
            by_window[w] = r
    steps = sum(r["steps"] for r in by_window.values())
    out["window_total_ms"] = (
        sum(r["window_time_s"] for r in by_window.values()) / steps * 1000.0
    )
    rank0 = [r for r in records if r.get("rank", 0) == 0]
    if rank0:
        n = sum(r["steps"] for r in rank0)
        out["window_vllm_ms"] = (
            sum(r["components"].get("data_time_s", 0.0) for r in rank0) / n * 1000.0
        )
        out["window_sae_ms"] = (
            sum(r["components"].get("sae_time_s", 0.0) for r in rank0) / n * 1000.0
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", type=Path, default=ROOT / "results" / "nsys_gap_diagnosis"
    )
    args = p.parse_args()

    manifest = [
        json.loads(line)
        for line in (args.root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    results = []
    for spec in manifest:
        sqlite_path = Path(spec["nsys_base"] + ".sqlite")
        if not sqlite_path.exists():
            print(f"{spec['run_id']}: missing {sqlite_path}")
            continue
        record = analyze_run(sqlite_path, spec)
        record.update(load_window_profile(Path(spec["output_path"])))
        results.append(record)
        print(f"analyzed {spec['run_id']}")

    out_dir = args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gap_analysis.json").write_text(json.dumps(results, indent=2))

    ok = [r for r in results if "error" not in r]
    if ok:
        header = (
            f"{'run_id':26s} {'ranks':>5s} {'steps':>5s} {'wall_ms':>9s} "
            f"{'gpu_busy':>9s} {'gpu_idle':>9s} {'idle%':>6s}  "
            f"{'win_total':>9s} {'win_vllm':>9s} {'win_sae':>9s}"
        )
        lines = [header, "-" * len(header)]
        for r in ok:
            lines.append(
                f"{r['run_id']:26s} {r['n_ranks']:5d} {r['n_steps_analyzed']:5d} "
                f"{r['wall_ms']:9.1f} {r['gpu_busy_ms']:9.1f} "
                f"{r['gpu_idle_ms']:9.1f} {r['gpu_idle_pct']:6.1f}  "
                f"{r.get('window_total_ms', float('nan')):9.1f} "
                f"{r.get('window_vllm_ms', float('nan')):9.1f} "
                f"{r.get('window_sae_ms', float('nan')):9.1f}"
            )
        lines.append("")
        lines.append("GPU busy time per NVTX phase (ms/step, union within step wall):")
        phases = sorted({k for r in ok for k in r["phase_busy_ms"]})
        lines.append(f"{'run_id':26s} " + " ".join(f"{p[-22:]:>24s}" for p in phases))
        for r in ok:
            cells = " ".join(
                f"{r['phase_busy_ms'].get(p, 0.0):24.1f}" for p in phases
            )
            lines.append(f"{r['run_id']:26s} {cells}")

        lines.append("")
        lines.append(
            "GPU tail past the launching CPU range (ms/step) — what CPU timers miss:"
        )
        tails = sorted({k for r in ok for k in r["phase_gpu_tail_ms"]})
        lines.append(f"{'run_id':26s} " + " ".join(f"{t[-22:]:>24s}" for t in tails))
        for r in ok:
            cells = " ".join(
                f"{r['phase_gpu_tail_ms'].get(t, 0.0):24.1f}" for t in tails
            )
            lines.append(f"{r['run_id']:26s} {cells}")

        lines.append("")
        lines.append(
            "GPU work with no enclosing NVTX range, by launching thread (ms/step):"
        )
        for r in ok:
            cells = ", ".join(
                f"{k}={v:.1f}"
                for k, v in sorted(
                    r["unattributed_by_thread_ms"].items(), key=lambda kv: -kv[1]
                )
            )
            lines.append(f"  {r['run_id']:26s} {cells or '(none)'}")

        lines.append("")
        lines.append("Top kernels by GPU time inside the step (ms/step):")
        for r in ok:
            lines.append(f"  {r['run_id']}:")
            for name, ms in r["top_kernels_ms"].items():
                lines.append(f"    {name[:60]:62s} {ms:8.2f}")
        table = "\n".join(lines)
        (out_dir / "gap_analysis.txt").write_text(table + "\n")
        print()
        print(table)

        with open(out_dir / "gap_analysis.csv", "w", newline="") as f:
            fields = [
                "run_id", "mode", "H", "d_sae", "batch", "tp", "dp", "n_ranks",
                "n_steps_analyzed", "wall_ms", "gpu_busy_ms", "gpu_idle_ms",
                "gpu_idle_pct", "window_total_ms", "window_vllm_ms", "window_sae_ms",
            ] + [f"phase_{p}" for p in phases]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in ok:
                row = {k: r.get(k) for k in fields}
                for p in phases:
                    row[f"phase_{p}"] = r["phase_busy_ms"].get(p, 0.0)
                writer.writerow(row)


if __name__ == "__main__":
    main()
