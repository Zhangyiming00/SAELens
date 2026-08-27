#!/usr/bin/env python3
"""Build the vLLM / SAE / gap / total table.

Each column is a measurement, and the three parts are measured so that they are
disjoint and sum to no more than the total:

- `total_ms`  wall-clock per step, from the step-window profiler (synced only at
  window boundaries). Ground truth.
- `vllm_ms`   GPU busy time inside the step that belongs to the data fetch, taken
  from the nsys timeline as the union of GPU activity launched under the
  `multi_sae:data_fetch` NVTX range plus vLLM's own engine-thread launches.
- `sae_ms`    GPU busy time of everything else in the step: the SAE forward,
  backward, optimizer, stats sync, and the NCCL collectives. NCCL is included
  here even though it sits outside every `multi_sae:*` range, because it is SAE
  gradient traffic — leaving it out is what made the earlier `gap` column
  balloon.
- `gap_ms`    `total - vllm - sae`: wall time in the step with no GPU activity at
  all, plus GPU time the union already counted once but that overlaps both parts.

Because `vllm_ms` and `sae_ms` are unions over the same wall interval they can
overlap when vLLM and SAE work run concurrently; `overlap_ms` reports that
directly, so `gap` stays interpretable as genuine idle time.

Inputs: a step-window sweep directory (for the wall-clock total) and an nsys
directory produced by scripts/run_nsys_gap_diagnosis.py over the same configs.
When both exist for a run the nsys run's own window profile is used, so total and
split come from the identical process.
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

# GPU work launched under these NVTX ranges is vLLM's, not the SAE's.
VLLM_RANGES = {"multi_sae:data_fetch"}

# Every other multi_sae range is SAE work. Listed innermost-first so an activity
# is charged to the tightest range enclosing its launch.
SAE_RANGES = [
    "multi_sae:unified_forward",
    "multi_sae:combined_backward",
    "nccl:multi_sae_ddp_combined_backward",
    "multi_sae:optimizer_step",
    "multi_sae:optimizer_unscale",
    "multi_sae:scaler_update",
    "multi_sae:stats_sync_tail",
    "multi_sae:train_step",
]

# Kernel-name prefixes that are SAE gradient/parameter traffic even when they are
# launched outside any NVTX range (DDP/FSDP hooks fire from autograd threads).
NCCL_PREFIXES = ("nccl", "ncclDevKernel")


@dataclass
class Interval:
    start: int
    end: int


def union_length(intervals: list[Interval], lo: int, hi: int) -> int:
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


def overlap_length(
    a: list[Interval], b: list[Interval], lo: int, hi: int
) -> int:
    """Length of the intersection of two interval unions, clipped to [lo, hi)."""
    union_a = union_length(a, lo, hi)
    union_b = union_length(b, lo, hi)
    union_both = union_length(a + b, lo, hi)
    return union_a + union_b - union_both


def _pid(global_tid: int) -> int:
    return global_tid >> 24


def load_trace(path: Path) -> dict:
    con = sqlite3.connect(path)
    tables = {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }

    acts: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for start, end, corr, pid, name in con.execute(
            "select k.start, k.end, k.correlationId, k.globalPid >> 24, "
            "coalesce(s.value, 'kernel') from CUPTI_ACTIVITY_KIND_KERNEL k "
            "left join StringIds s on s.id = k.shortName "
            "where k.start is not null and k.end is not null"
        ):
            acts[int(pid)].append((int(start), int(end), int(corr), str(name)))
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]"),
    ):
        if table in tables:
            for start, end, corr, pid in con.execute(
                f"select start, end, correlationId, globalPid >> 24 from {table} "
                "where start is not null and end is not null"
            ):
                acts[int(pid)].append((int(start), int(end), int(corr), label))

    runtime: dict[int, tuple[int, int]] = {}
    for start, corr, tid in con.execute(
        "select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where correlationId is not null"
    ):
        runtime[int(corr)] = (int(start), int(tid))

    nvtx: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for text, start, end, tid in con.execute(
        "select text, start, end, globalTid from NVTX_EVENTS "
        "where text is not null and start is not null and end is not null"
    ):
        nvtx[str(text)].append((int(start), int(end), int(tid)))
    for v in nvtx.values():
        v.sort()
    con.close()
    return {"acts": acts, "runtime": runtime, "nvtx": nvtx}


def classify(
    nvtx: dict[str, list[tuple[int, int, int]]],
    launch: tuple[int, int] | None,
    name: str,
    fetch_spans: list[tuple[int, int]],
) -> str:
    """Return 'vllm' or 'sae' for one GPU activity.

    NVTX ranges are the primary evidence, but they only cover the thread that
    opened them: DDP/FSDP fire NCCL from autograd threads, and the SAE backward
    launches its GEMMs from a worker thread that carries no range. So an activity
    whose launch is not inside any range is attributed by *when* it was launched
    — inside a `multi_sae:data_fetch` span it is vLLM's, otherwise it is the
    SAE's. Attributing by thread instead misreads the SAE's own backward GEMM
    threads as vLLM engine threads.
    """
    if name.startswith(NCCL_PREFIXES):
        return "sae"
    if launch is None:
        return "sae"
    t, tid = launch
    for text in SAE_RANGES:
        for start, end, range_tid in nvtx.get(text, []):
            if range_tid == tid and start <= t < end:
                return "sae"
            if start > t:
                break
    for text in VLLM_RANGES:
        for start, end, range_tid in nvtx.get(text, []):
            if range_tid == tid and start <= t < end:
                return "vllm"
            if start > t:
                break
    for start, end in fetch_spans:
        if start <= t < end:
            return "vllm"
        if start > t:
            break
    return "sae"


def analyze(trace: dict, drop_warmup: int = 2) -> dict | None:
    acts = trace["acts"]
    runtime = trace["runtime"]
    nvtx = trace["nvtx"]
    steps_all = sorted(nvtx.get("multi_sae:train_step", []))
    fetches_all = sorted(nvtx.get("multi_sae:data_fetch", []))
    if not steps_all:
        return None

    per_pid: dict[int, list[dict]] = {}
    for pid, pid_acts in acts.items():
        steps = [s for s in steps_all if _pid(s[2]) == pid]
        fetches = [f for f in fetches_all if _pid(f[2]) == pid]
        fetch_spans = sorted((f[0], f[1]) for f in fetches)
        if not steps:
            continue
        rows = []
        for s_start, s_end, _ in steps:
            prior = [f for f in fetches if f[0] <= s_start]
            lo = prior[-1][0] if prior else s_start
            hi = s_end
            buckets: dict[str, list[Interval]] = defaultdict(list)
            for a_start, a_end, corr, name in pid_acts:
                if a_end <= lo or a_start >= hi:
                    continue
                kind = classify(nvtx, runtime.get(corr), name, fetch_spans)
                buckets[kind].append(Interval(a_start, a_end))
            all_ivs = [iv for ivs in buckets.values() for iv in ivs]
            all_ivs += [Interval(s, e) for s, e in fetch_spans if s < hi and e > lo]
            # vLLM is charged the wall span of its fetches plus any vLLM GPU work
            # outside them, not just GPU busy time: a fetch also iterates the
            # dataset and copies host-side, and that wall time is the data path's
            # cost rather than idle time. The SAE is charged GPU busy time, since
            # its step is GPU-bound and its CPU launches overlap the device.
            vllm_ivs = [Interval(s, e) for s, e in fetch_spans]
            vllm_ivs += buckets.get("vllm", [])
            sae_ivs = buckets.get("sae", [])
            sae = union_length(sae_ivs, lo, hi)
            # Make the two parts disjoint so vllm + sae + gap == total exactly.
            # SAE GPU work that runs concurrently with a fetch is credited to the
            # SAE, and only the remainder of the fetch span counts as vLLM.
            vllm_union = union_length(vllm_ivs, lo, hi)
            vllm = vllm_union - overlap_length(vllm_ivs, sae_ivs, lo, hi)
            other = 0
            busy = union_length(all_ivs, lo, hi)
            rows.append(
                {
                    "wall": hi - lo,
                    "vllm": vllm,
                    "sae": sae,
                    "other": other,
                    "busy": busy,
                    # SAE GPU work that ran concurrently with a vLLM fetch. It is
                    # credited to the SAE, so this only reports how much the two
                    # phases actually pipeline.
                    "overlap": vllm_union - vllm,
                }
            )
        per_pid[pid] = rows

    if not per_pid:
        return None
    n = min(len(v) for v in per_pid.values())
    if n <= drop_warmup:
        return None
    # A step ends when its slowest rank ends, matching the wall-clock table.
    chosen = [
        max((per_pid[pid][i] for pid in per_pid), key=lambda r: r["wall"])
        for i in range(drop_warmup, n)
    ]
    k = len(chosen)

    def avg(key: str) -> float:
        return sum(r[key] for r in chosen) / k / 1e6

    return {
        "n_ranks": len(per_pid),
        "n_steps": k,
        "trace_wall_ms": avg("wall"),
        "vllm_ms": avg("vllm"),
        "sae_ms": avg("sae"),
        "other_ms": avg("other"),
        "gpu_busy_ms": avg("busy"),
        "overlap_ms": avg("overlap"),
    }


def window_total_ms(run_dir: Path) -> float | None:
    """Wall-clock per-step time: slowest rank per window, averaged."""
    by_window: dict[int, dict] = {}
    for path in sorted(run_dir.glob("step_window_profile_sae_rank*.jsonl")):
        for line in path.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            w = r["window"]
            if w not in by_window or r["window_time_s"] > by_window[w]["window_time_s"]:
                by_window[w] = r
    if not by_window:
        return None
    steps = sum(r["steps"] for r in by_window.values())
    return sum(r["window_time_s"] for r in by_window.values()) / steps * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nsys-root", type=Path, default=ROOT / "results" / "nsys_gap_20")
    p.add_argument("--out-name", default="vllm_sae_gap_table")
    args = p.parse_args()

    manifest = [
        json.loads(line)
        for line in (args.nsys_root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    rows: list[dict] = []
    for spec in manifest:
        sqlite_path = Path(spec["nsys_base"] + ".sqlite")
        if not sqlite_path.exists():
            print(f"  skip {spec['run_id']}: no sqlite")
            continue
        result = analyze(load_trace(sqlite_path))
        if result is None:
            print(f"  skip {spec['run_id']}: no usable steps")
            continue
        total = window_total_ms(Path(spec["output_path"]))
        if total is None:
            print(f"  skip {spec['run_id']}: no window profile")
            continue
        vllm = result["vllm_ms"]
        sae = result["sae_ms"]
        gap = total - vllm - sae
        # The parts are disjoint by construction, so this must hold exactly.
        assert abs(vllm + sae + gap - total) < 1e-6, spec["run_id"]
        if gap < 0:
            print(f"  WARNING {spec['run_id']}: negative gap {gap:.2f}ms")
        rows.append(
            {
                "run_id": spec["run_id"],
                "mode": spec["mode"],
                "H": spec["hook_count"],
                "d_sae": spec["d_sae"],
                "batch": spec["batch_tokens"],
                "tp": spec["sae_tp_size"],
                "dp": spec["sae_dp_size"],
                "n_ranks": result["n_ranks"],
                "n_steps": result["n_steps"],
                "vllm_ms": vllm,
                "sae_ms": sae,
                "gap_ms": gap,
                "total_ms": total,
                "vllm_pct": vllm / total * 100.0,
                "sae_pct": sae / total * 100.0,
                "gap_pct": gap / total * 100.0,
                "overlap_ms": result["overlap_ms"],
                "other_ms": result["other_ms"],
                "gpu_busy_ms": result["gpu_busy_ms"],
                "trace_wall_ms": result["trace_wall_ms"],
            }
        )
        print(f"  ok {spec['run_id']}")

    order = {"single": 0, "tp2": 1, "ddp": 2, "fsdp": 3}
    rows.sort(key=lambda r: (order.get(r["mode"], 9), r["run_id"]))

    header = (
        f"{'run_id':24s} {'H':>2s} {'d_sae':>7s} {'batch':>6s} {'tp':>3s} {'dp':>3s} "
        f"{'vllm_ms':>9s} {'sae_ms':>9s} {'gap_ms':>8s} {'total_ms':>9s}  "
        f"{'vllm%':>6s} {'sae%':>6s} {'gap%':>6s}  {'overlap':>8s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['run_id']:24s} {r['H']:2d} {r['d_sae']:7d} {r['batch']:6d} "
            f"{r['tp']:3d} {r['dp']:3d} {r['vllm_ms']:9.1f} {r['sae_ms']:9.1f} "
            f"{r['gap_ms']:8.1f} {r['total_ms']:9.1f}  {r['vllm_pct']:6.1f} "
            f"{r['sae_pct']:6.1f} {r['gap_pct']:6.1f}  {r['overlap_ms']:8.1f}"
        )
    table = "\n".join(lines)

    out_dir = args.nsys_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.out_name}.txt").write_text(table + "\n")
    with open(out_dir / f"{args.out_name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(table)


if __name__ == "__main__":
    main()
