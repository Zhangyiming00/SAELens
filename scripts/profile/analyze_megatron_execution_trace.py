"""Extract current-runtime phase work and actual overlap from Nsight SQLite.

CUDA API correlation assigns activities to nested NVTX phases. GPU activity
durations are distinct from CPU range durations and stream span (which includes
waits). No GPU duration is inferred by adding CPU forward/backward timers.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b-a for a, b in merged)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def extract(directory: Path):
    ranks = [json.loads(p.read_text()) for p in sorted(directory.glob("rank*.json"))
             if p.stem.removeprefix("rank").isdigit()]
    bypid = {r["pid"]: r for r in ranks}
    if not ranks or not all(r["trace"] for r in ranks):
        raise ValueError(f"Not a completed timing trace: {directory}")
    db = sqlite3.connect(f"file:{directory/'trace.sqlite'}?mode=ro", uri=True)
    strings = dict(db.execute("select id,value from StringIds"))
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    spans, steps = defaultdict(list), defaultdict(list)
    for a, b, tid, text, text_id in db.execute("select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null"):
        pid = (tid >> 24) & 0xFFFFFF
        if pid not in bypid:
            continue
        tag = text or strings.get(text_id, "")
        if tag.startswith("time:step:"):
            steps[pid].append((a, b, int(tag.rsplit(":", 1)[1]), tid))
        elif tag.startswith(("time:phase:", "time:comm:")):
            spans[tid].append((a, b, tag))
    starts = {}
    for pid in bypid:
        steps[pid].sort()
        starts[pid] = [s[0] for s in steps[pid]]
        if len(steps[pid]) != len(bypid[pid]["steps"]):
            raise ValueError("Incomplete trace capture")
    phase_spans = defaultdict(list)
    for tid, records in spans.items():
        pid = (tid >> 24) & 0xFFFFFF
        for a, b, tag in records:
            i = bisect.bisect_right(starts[pid], a)-1
            if i >= 0 and a < steps[pid][i][1]:
                phase_spans[(pid, steps[pid][i][2], tid)].append((a, b, tag))
    launches = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for a, b, tid, corr in db.execute(f"select start,end,globalTid,correlationId from {table}"):
            pid = (tid >> 24) & 0xFFFFFF
            if pid not in starts:
                continue
            i = bisect.bisect_right(starts[pid], a)-1
            if i < 0 or a >= steps[pid][i][1]:
                continue
            start, _, step, main_tid = steps[pid][i]
            tags = [r for r in phase_spans[(pid, step, tid)] if r[0] <= a and b <= r[1]]
            if not tags and tid != main_tid:
                tags = [r for r in phase_spans[(pid, step, main_tid)] if r[0] <= a and b <= r[1]]
            tags.sort(key=lambda r: r[1]-r[0])
            phase_tag = next((r[2] for r in tags if r[2].startswith("time:phase:")), "time:phase:all:other")
            _, _, hook, phase = phase_tag.split(":", 3)
            comm = next((r[2].split(":")[2] for r in tags if r[2].startswith("time:comm:")), "unknown")
            launches.setdefault((pid, corr), dict(rank=bypid[pid]["rank"], step=step, hook=hook,
                                                 phase=phase, comm=comm, origin=start, api_start=a))
    activities = []
    for a, b, gpid, corr, stream, name_id in db.execute(
            "select start,end,globalPid,correlationId,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL"):
        pid = (gpid >> 24) & 0xFFFFFF
        loc = launches.get((pid, corr))
        if loc is None:
            continue
        name = strings[name_id]
        kind = "comm" if "nccl" in name.lower() else "gemm" if re.search("gemm|xmma|cutlass|sparse", name, re.I) else "other"
        activities.append(dict(**loc, stream=stream, name=name, kind=kind,
                               start_ms=(a-loc["origin"])/1e6, end_ms=(b-loc["origin"])/1e6,
                               ms=(b-a)/1e6))
    for table in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if table not in tables:
            continue
        for a, b, gpid, corr, stream in db.execute(f"select start,end,globalPid,correlationId,streamId from {table}"):
            pid = (gpid >> 24) & 0xFFFFFF
            loc = launches.get((pid, corr))
            if loc:
                activities.append(dict(**loc, stream=stream, name=table, kind="other",
                                       start_ms=(a-loc["origin"])/1e6, end_ms=(b-loc["origin"])/1e6, ms=(b-a)/1e6))
    db.close()
    primary = {}
    for r in ranks:
        counter = Counter()
        for a in activities:
            if a["rank"] == r["rank"] and a["phase"] == "backward" and a["kind"] != "comm":
                counter[a["stream"]] += a["ms"]
        primary[r["rank"]] = counter.most_common(1)[0][0]
    totals = defaultdict(lambda: defaultdict(float))
    grad = defaultdict(float)
    gather = defaultdict(float)
    other = defaultdict(float)
    for a in activities:
        rank, step, hook, phase, kind = (a[k] for k in ("rank", "step", "hook", "phase", "kind"))
        key = (rank, step, hook)
        if phase == "param_gather" and kind == "comm":
            gather[key] += a["ms"]
        elif phase in ("backward", "finish_grad") and kind == "comm" and a["comm"] != "tp":
            grad[key] += a["ms"]
        elif phase in ("encode", "decode", "finish", "backward", "optimizer", "zero_grad"):
            field = "tp_ms" if kind == "comm" else "async_compute_ms" if phase == "encode" and a["stream"] != primary[rank] else kind+"_ms"
            totals[(*key, phase)][field] += a["ms"]
        else:
            other[(rank, step)] += a["ms"]
    phases = {}
    keys = [(r["rank"], s["step"], str(h)) for r in ranks for s in r["steps"] for h in range(len(r["local_hooks"]))]
    for phase in ("encode", "decode", "finish", "backward", "optimizer", "zero_grad"):
        phases[phase] = {field: statistics.mean(totals[(*key, phase)][field] for key in keys)
                         for field in ("gemm_ms", "other_ms", "tp_ms", "async_compute_ms")}
    fractions, clip_fractions = [], []
    for key in keys:
        acts = [a for a in activities if (a["rank"], a["step"], a["hook"]) == key]
        bw = [a for a in acts if a["phase"] == "backward" and a["kind"] != "comm"]
        comm = [a for a in acts if a["phase"] in ("backward", "finish_grad") and a["kind"] == "comm" and a["comm"] != "tp"]
        if comm and bw:
            first = min(a["start_ms"] for a in comm)
            fractions.append(sum(a["ms"] for a in bw if a["end_ms"] <= first)/sum(a["ms"] for a in bw))
        opt = [a for a in acts if a["phase"] == "optimizer" and a["kind"] != "comm"]
        adam = [a for a in opt if "adam" in a["name"].lower()]
        if opt and adam:
            first = min(a["start_ms"] for a in adam)
            clip_fractions.append(sum(a["ms"] for a in opt if a["end_ms"] <= first)/sum(a["ms"] for a in opt))
    metrics = []
    for r in ranks:
        for s in r["steps"]:
            acts = [a for a in activities if a["rank"] == r["rank"] and a["step"] == s["step"]]
            comp = [(a["start_ms"], a["end_ms"]) for a in acts if a["kind"] != "comm"]
            comm = [(a["start_ms"], a["end_ms"]) for a in acts if a["kind"] == "comm"]
            busy = union(comp+comm)
            metrics.append(dict(rank=r["rank"], step=s["step"], wall_ms=s["ms"], cuda_ms=s["cuda_ms"],
                                compute_busy_ms=union(comp), comm_busy_ms=union(comm),
                                compute_comm_overlap_ms=union(comp)+union(comm)-busy,
                                gpu_idle_ms=max(0, s["cuda_ms"]-busy),
                                kernel_work_ms=sum(a["ms"] for a in acts)))
    row = dict(name=directory.name, config=ranks[0]["config"], phases=phases,
               grad_reduce_ms=statistics.mean(grad[k] for k in keys),
               param_gather_ms=statistics.mean(gather[k] for k in keys),
               grad_release_fraction=statistics.median(fractions) if fractions else 1,
               clip_fraction=statistics.median(clip_fractions) if clip_fractions else .35,
               overhead_ms=statistics.mean(other.values())/len(ranks[0]["local_hooks"]) if other else 0,
               warnings=["Phase work comes from traced kernels; rank skew, queue waits and contention may affect NCCL durations"],
               measured_gpu=ranks[0]["gpu"], torch=ranks[0]["torch"], cuda=ranks[0]["cuda"])
    with (directory/"activities.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(activities[0]))
        w.writeheader()
        w.writerows(activities)
    dump(directory/"phase_profile.json", row)
    dump(directory/"trace_metrics.json", metrics)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", type=Path, nargs="+")
    args = p.parse_args()
    for d in args.directories:
        row = extract(d)
        print(d.name, json.dumps(row["phases"]), flush=True)
