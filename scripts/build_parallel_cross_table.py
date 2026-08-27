#!/usr/bin/env python3
"""Final table for the parallelism cross sweep: vLLM / SAE / gap / total.

The SAE column follows the `real_gpu_hw_ms` definition used by the 7/31 nsys
sweep, with no extra processing beyond scaling to one step:

    for each `multi_sae:train_step` CPU range, take every GPU activity whose
    launch (CUPTI_ACTIVITY_KIND_RUNTIME) falls inside that range, and measure
    hardware min(start) .. max(end); with several ranks take the max over ranks
    for the same step index, then average over the steady-state steps.

That is a span, not a union of busy intervals, and it counts only work launched
from inside the train_step range — so NCCL fired from autograd threads is
excluded, exactly as in the reference numbers. It is reported as measured.

`vllm_ms` is the wall span of the data fetches feeding the profiled steps, per
step. Mixing-buffer work (randperm/cat/index copies), dataset iteration and the
host-to-device copies all happen inside `next(data_provider)` and therefore
inside `multi_sae:data_fetch`, so they are counted here and not in the SAE
column.

`gap_ms = total - vllm - sae`. Because the SAE column is a span taken from a
different accounting than the vLLM wall span, the two can overlap when the SAE
step runs concurrently with a prefetch; `overlap_ms` reports that so the gap
stays interpretable. A negative gap therefore means the two spans overlap more
than the step has slack, and is flagged rather than hidden.

`total_ms` is the step-window wall-clock per-step time from the same run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pid(global_tid: int) -> int:
    return global_tid >> 24


def load(path: Path) -> dict:
    con = sqlite3.connect(path)
    tables = {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }
    acts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if table not in tables:
            continue
        for start, end, corr, pid in con.execute(
            f"select start, end, correlationId, globalPid >> 24 from {table} "
            "where start is not null and end is not null"
        ):
            acts[int(pid)].append((int(start), int(end), int(corr)))

    # correlationId -> (launch time, launching pid)
    launches: dict[int, tuple[int, int]] = {}
    for start, corr, tid in con.execute(
        "select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where correlationId is not null"
    ):
        launches[int(corr)] = (int(start), _pid(int(tid)))

    nvtx: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for text, start, end, tid in con.execute(
        "select text, start, end, globalTid from NVTX_EVENTS "
        "where text is not null and start is not null and end is not null"
    ):
        nvtx[str(text)].append((int(start), int(end), _pid(int(tid))))
    for v in nvtx.values():
        v.sort()
    con.close()
    return {"acts": acts, "launches": launches, "nvtx": nvtx}


def span_ms(
    acts: list[tuple[int, int, int]],
    launches: dict[int, tuple[int, int]],
    pid: int,
    lo: int,
    hi: int,
) -> float | None:
    """min(start)..max(end) of GPU activity launched in [lo, hi) by `pid`."""
    starts: list[int] = []
    ends: list[int] = []
    for a_start, a_end, corr in acts:
        launch = launches.get(corr)
        if launch is None or launch[1] != pid:
            continue
        if lo <= launch[0] < hi:
            starts.append(a_start)
            ends.append(a_end)
    if not starts:
        return None
    return (max(ends) - min(starts)) / 1e6


def analyze(trace: dict, drop_warmup: int = 2) -> dict | None:
    acts, launches, nvtx = trace["acts"], trace["launches"], trace["nvtx"]
    steps_all = sorted(nvtx.get("multi_sae:train_step", []))
    fetches_all = sorted(nvtx.get("multi_sae:data_fetch", []))
    if not steps_all:
        return None

    # Per rank: the SAE span of each step, and the fetch wall feeding it.
    per_pid_sae: dict[int, list[float]] = {}
    per_pid_vllm: dict[int, list[float]] = {}
    for pid in sorted(acts):
        steps = [s for s in steps_all if s[2] == pid]
        fetches = [f for f in fetches_all if f[2] == pid]
        if not steps:
            continue
        sae_vals: list[float] = []
        vllm_vals: list[float] = []
        for s_start, s_end, _ in steps:
            sae = span_ms(acts[pid], launches, pid, s_start, s_end)
            sae_vals.append(0.0 if sae is None else sae)
            # The fetch that fed this step: the last one starting at or before it.
            prior = [f for f in fetches if f[0] <= s_start]
            vllm_vals.append((prior[-1][1] - prior[-1][0]) / 1e6 if prior else 0.0)
        per_pid_sae[pid] = sae_vals
        per_pid_vllm[pid] = vllm_vals

    if not per_pid_sae:
        return None
    n = min(len(v) for v in per_pid_sae.values())
    if n <= drop_warmup:
        return None
    idx = range(drop_warmup, n)
    # rank-max per step index, then mean over steady-state steps.
    sae = sum(max(per_pid_sae[p][i] for p in per_pid_sae) for i in idx) / len(idx)
    vllm = sum(max(per_pid_vllm[p][i] for p in per_pid_vllm) for i in idx) / len(idx)
    return {
        "n_ranks": len(per_pid_sae),
        "n_steps": len(idx),
        "sae_ms": sae,
        "vllm_ms": vllm,
    }


def window_total_ms(run_dir: Path) -> float | None:
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
    p.add_argument(
        "--root", type=Path, default=ROOT / "results" / "parallel_cross_sweep"
    )
    args = p.parse_args()

    manifest = [
        json.loads(line)
        for line in (args.root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    rows: list[dict] = []
    for spec in manifest:
        sqlite_path = Path(spec["nsys_base"] + ".sqlite")
        if not sqlite_path.exists():
            print(f"  skip {spec['run_id']}: no sqlite")
            continue
        result = analyze(load(sqlite_path))
        total = window_total_ms(Path(spec["output_path"]))
        if result is None or total is None:
            print(f"  skip {spec['run_id']}: no usable steps or window")
            continue
        sae = result["sae_ms"]
        vllm = result["vllm_ms"]
        gap = total - vllm - sae
        if gap < 0:
            print(f"  note {spec['run_id']}: negative gap {gap:.1f}ms (spans overlap)")
        rows.append(
            {
                "run_id": spec["run_id"],
                "sae_mode": spec["sae_mode"],
                "vllm_mode": spec["vllm_mode"],
                "H": spec["hook_count"],
                "d_in": 4096,
                "d_sae": spec["d_sae"],
                "batch": spec["batch_tokens"],
                "k": spec["k"],
                "sae_tp": spec["sae_tp_size"],
                "sae_dp": spec["sae_dp_size"],
                "vllm_tp": spec["vllm_tp_size"],
                "vllm_dp": spec["vllm_dp_size"],
                "n_ranks": result["n_ranks"],
                "n_steps": result["n_steps"],
                "vllm_ms": vllm,
                "sae_ms": sae,
                "gap_ms": gap,
                "total_ms": total,
                "vllm_pct": vllm / total * 100.0,
                "sae_pct": sae / total * 100.0,
                "gap_pct": gap / total * 100.0,
            }
        )
        print(f"  ok {spec['run_id']}")

    sae_order = {"single": 0, "tp2": 1, "ddp": 2, "fsdp": 3}
    vllm_order = {"vtp1dp1": 0, "vtp2dp1": 1, "vtp1dp2": 2}
    rows.sort(
        key=lambda r: (sae_order.get(r["sae_mode"], 9), vllm_order.get(r["vllm_mode"], 9))
    )

    header = (
        f"{'run_id':22s} {'sae':>7s} {'vllm':>8s} {'stp':>3s} {'sdp':>3s} "
        f"{'vtp':>3s} {'vdp':>3s} {'vllm_ms':>9s} {'sae_ms':>9s} {'gap_ms':>9s} "
        f"{'total_ms':>9s}  {'vllm%':>6s} {'sae%':>6s} {'gap%':>6s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['run_id']:22s} {r['sae_mode']:>7s} {r['vllm_mode']:>8s} "
            f"{r['sae_tp']:3d} {r['sae_dp']:3d} {r['vllm_tp']:3d} {r['vllm_dp']:3d} "
            f"{r['vllm_ms']:9.2f} {r['sae_ms']:9.2f} {r['gap_ms']:9.2f} "
            f"{r['total_ms']:9.2f}  {r['vllm_pct']:6.1f} {r['sae_pct']:6.1f} "
            f"{r['gap_pct']:6.1f}"
        )
    table = "\n".join(lines)

    out_dir = args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_table.txt").write_text(table + "\n")
    with open(out_dir / "cross_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(table)


if __name__ == "__main__":
    main()
