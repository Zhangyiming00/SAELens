#!/usr/bin/env python3
"""Locate why `total - vllm - sae` goes negative, using an nsys timeline.

The wall-clock table subtracts a profiled vLLM prefill cost and a reference SAE
GPU-hardware time from the measured per-step wall time, and the result comes out
slightly below zero. On one GPU the two phases cannot genuinely run at the same
time, so a negative residual means at least one of the two subtracted terms is
larger than the wall time it actually occupies in the run.

This script measures, per step, from the trace:

- `wall_ms`        the step's wall interval (fetch start .. train_step end)
- `gpu_busy_ms`    union of all GPU activity in that interval
- `gpu_idle_ms`    the complement: wall time with no GPU work at all
- `fetch_gpu_ms`   GPU busy inside the `multi_sae:data_fetch` range
- `sae_gpu_ms`     GPU busy attributable to the SAE (everything else, plus NCCL,
                   which DDP/FSDP launch from autograd threads outside any range)
- `overlap_ms`     GPU time counted by both, i.e. genuine concurrency
- `sae_span_ms`    the `real_gpu_hw_ms` definition: min(start)..max(end) of the
                   activity launched inside `multi_sae:train_step`. This is a SPAN,
                   so it includes any idle gaps between those kernels, which is
                   the suspected source of the over-subtraction.

Comparing `sae_span_ms` against `sae_gpu_ms` shows how much of the reference SAE
number is device-idle time rather than SAE work.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NCCL_PREFIXES = ("nccl", "ncclDevKernel")


@dataclass
class Iv:
    start: int
    end: int


def union_len(ivs: list[Iv], lo: int, hi: int) -> int:
    clipped = sorted(
        (max(i.start, lo), min(i.end, hi))
        for i in ivs
        if i.end > lo and i.start < hi
    )
    if not clipped:
        return 0
    total = 0
    cur_s, cur_e = clipped[0]
    for s, e in clipped[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + cur_e - cur_s


def load(path: Path) -> dict:
    con = sqlite3.connect(path)
    tables = {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }
    acts: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for s, e, c, pid, name in con.execute(
            "select k.start, k.end, k.correlationId, k.globalPid >> 24, "
            "coalesce(sid.value, 'kernel') from CUPTI_ACTIVITY_KIND_KERNEL k "
            "left join StringIds sid on sid.id = k.shortName "
            "where k.start is not null and k.end is not null"
        ):
            acts[int(pid)].append((int(s), int(e), int(c), str(name)))
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]"),
    ):
        if table in tables:
            for s, e, c, pid in con.execute(
                f"select start, end, correlationId, globalPid >> 24 from {table} "
                "where start is not null and end is not null"
            ):
                acts[int(pid)].append((int(s), int(e), int(c), label))

    launches: dict[int, tuple[int, int]] = {}
    for s, c, tid in con.execute(
        "select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where correlationId is not null"
    ):
        launches[int(c)] = (int(s), int(tid) >> 24)

    nvtx: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for text, s, e, tid in con.execute(
        "select text, start, end, globalTid from NVTX_EVENTS "
        "where text is not null and start is not null and end is not null"
    ):
        nvtx[str(text)].append((int(s), int(e), int(tid) >> 24))
    for v in nvtx.values():
        v.sort()
    con.close()
    return {"acts": acts, "launches": launches, "nvtx": nvtx}


def analyze(trace: dict, drop_warmup: int = 2) -> list[dict]:
    acts, launches, nvtx = trace["acts"], trace["launches"], trace["nvtx"]
    steps_all = sorted(nvtx.get("multi_sae:train_step", []))
    fetches_all = sorted(nvtx.get("multi_sae:data_fetch", []))
    out: list[dict] = []
    for pid in sorted(acts):
        steps = [s for s in steps_all if s[2] == pid]
        fetches = [f for f in fetches_all if f[2] == pid]
        pid_acts = acts[pid]
        for idx, (s_start, s_end, _) in enumerate(steps):
            if idx < drop_warmup:
                continue
            prior = [f for f in fetches if f[0] <= s_start]
            lo = prior[-1][0] if prior else s_start
            hi = s_end
            fetch_span = (prior[-1][0], prior[-1][1]) if prior else (lo, lo)

            all_ivs: list[Iv] = []
            fetch_ivs: list[Iv] = []
            sae_ivs: list[Iv] = []
            train_launched: list[Iv] = []
            for a_s, a_e, corr, name in pid_acts:
                if a_e <= lo or a_s >= hi:
                    continue
                iv = Iv(a_s, a_e)
                all_ivs.append(iv)
                launch = launches.get(corr)
                lt = launch[0] if launch else None
                in_fetch = lt is not None and fetch_span[0] <= lt < fetch_span[1]
                in_train = lt is not None and s_start <= lt < s_end
                if name.startswith(NCCL_PREFIXES):
                    sae_ivs.append(iv)
                elif in_fetch:
                    fetch_ivs.append(iv)
                else:
                    sae_ivs.append(iv)
                if in_train:
                    train_launched.append(iv)

            wall = hi - lo
            busy = union_len(all_ivs, lo, hi)
            f_gpu = union_len(fetch_ivs, lo, hi)
            s_gpu = union_len(sae_ivs, lo, hi)
            # Overlap between the two buckets = sum of parts minus their union.
            both = union_len(fetch_ivs + sae_ivs, lo, hi)
            overlap = f_gpu + s_gpu - both
            # real_gpu_hw_ms definition: span of train_step-launched activity.
            if train_launched:
                span = max(i.end for i in train_launched) - min(
                    i.start for i in train_launched
                )
                span_busy = union_len(train_launched, 0, 1 << 62)
            else:
                span = 0
                span_busy = 0
            out.append(
                {
                    "pid": pid,
                    "step_idx": idx,
                    "wall_ms": wall / 1e6,
                    "gpu_busy_ms": busy / 1e6,
                    "gpu_idle_ms": (wall - busy) / 1e6,
                    "fetch_gpu_ms": f_gpu / 1e6,
                    "sae_gpu_ms": s_gpu / 1e6,
                    "overlap_ms": overlap / 1e6,
                    "fetch_wall_ms": (fetch_span[1] - fetch_span[0]) / 1e6,
                    "sae_span_ms": span / 1e6,
                    "sae_span_busy_ms": span_busy / 1e6,
                    "sae_span_idle_ms": (span - span_busy) / 1e6,
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sqlite",
        type=Path,
        default=ROOT / "results" / "gap_probe" / "ddp_vtp1dp1.sqlite",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gap_probe")
    args = p.parse_args()

    rows = analyze(load(args.sqlite))
    if not rows:
        print("no steady-state steps found")
        return

    keys = [
        "wall_ms",
        "gpu_busy_ms",
        "gpu_idle_ms",
        "fetch_wall_ms",
        "fetch_gpu_ms",
        "sae_gpu_ms",
        "overlap_ms",
        "sae_span_ms",
        "sae_span_busy_ms",
        "sae_span_idle_ms",
    ]
    pids = sorted({r["pid"] for r in rows})
    print(f"steps analyzed: {len(rows)} across {len(pids)} rank(s)\n")
    print(f"{'metric':20s}" + "".join(f"{f'rank{i}':>14s}" for i in range(len(pids))))
    print("-" * (20 + 14 * len(pids)))
    summary: dict[str, list[float]] = {}
    for key in keys:
        cells = []
        for pid in pids:
            vals = [r[key] for r in rows if r["pid"] == pid]
            cells.append(sum(vals) / len(vals) if vals else 0.0)
        summary[key] = cells
        print(f"{key:20s}" + "".join(f"{c:14.2f}" for c in cells))

    (args.out_dir / "gap_probe.json").write_text(
        json.dumps({"per_step": rows, "mean_by_rank": summary}, indent=2)
    )
    print(f"\nwrote {args.out_dir / 'gap_probe.json'}")


if __name__ == "__main__":
    main()
