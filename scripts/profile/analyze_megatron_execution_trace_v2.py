"""CPU submission, synchronization and CUDA stream audit of native Nsight runs.

Traced CPU durations include profiler overhead. They are diagnostic and are
never substituted for unprofiled end-to-end timing. Event generations, not just
reused CUDA event IDs, are used when auditing stream-wait dependencies.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics

from analyze_megatron_execution_trace import extract, union


def audit(directory):
    directory = Path(directory)
    if not (directory / "activities.csv").exists():
        extract(directory)
    ranks = [json.loads(p.read_text()) for p in directory.glob("rank[0-9]*.json")]
    rank_pid = {r["pid"]: r for r in ranks}
    db = sqlite3.connect(f"file:{directory/'trace.sqlite'}?mode=ro", uri=True)
    strings = dict(db.execute("select id,value from StringIds"))
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    steps, phases, guards = defaultdict(list), defaultdict(list), defaultdict(list)
    for a, b, tid, txt, textid in db.execute("select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null"):
        pid = (tid >> 24) & 0xFFFFFF
        if pid not in rank_pid:
            continue
        tag = txt or strings.get(textid, "")
        if tag.startswith("time:step:"):
            steps[pid].append((a, b, int(tag.split(":")[-1]), tid))
        elif tag.startswith("time:phase:"):
            _, _, hook, phase = tag.split(":", 3)
            phases[pid].append((a, b, hook, phase, tid))
        elif tag.endswith(":backward_ready_wait"):
            guards[pid].append((a, b))
    for spans in steps.values():
        spans.sort()
    starts = {pid: [x[0] for x in spans] for pid, spans in steps.items()}
    spans = defaultdict(list)
    for pid, records in phases.items():
        for a, b, hook, phase, tid in records:
            i = bisect.bisect_right(starts[pid], a)-1
            if i >= 0 and b <= steps[pid][i][1]:
                spans[pid, steps[pid][i][2], hook, phase].append((a, b))
    api_by_pid = defaultdict(list)
    api_counts = Counter()
    for a, b, tid, corr, nameid in db.execute("select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        pid = (tid >> 24) & 0xFFFFFF
        if pid in steps:
            name = strings[nameid]
            api_by_pid[pid].append((a, b, name, corr))
            api_counts[name] += 1
    rows = []
    for (pid, step, hook, phase), ranges in sorted(spans.items()):
        lo, hi = min(a for a,b in ranges), max(b for a,b in ranges)
        apis = [(a,b,n) for a,b,n,_ in api_by_pid[pid]
                if lo <= a and b <= hi and any(x <= a and b <= y for x,y in ranges)]
        blocked = [(a,b) for a,b,n in apis if "Synchronize" in n or n in ("cudaMemcpy_v3020", "cudaMemcpy")]
        wait = [(a,b) for a,b in guards[pid] if any(x <= a and b <= y for x,y in ranges)]
        full = union(ranges)/1e6
        blocking = union(blocked)/1e6
        guard = union(wait)/1e6
        rows.append(dict(rank=rank_pid[pid]["rank"], step=step, hook=hook, phase=phase,
                         span_ms=full, blocking_api_ms=blocking, guard_ms=guard,
                         non_sync_span_ms=max(0., full-union(blocked+wait)/1e6),
                         long_launch_api_ms=union([(a,b) for a,b,n in apis if "LaunchKernel" in n and b-a>200000])/1e6,
                         launches=sum("LaunchKernel" in n for a,b,n in apis)))
    phase_summary = {}
    for phase in sorted({r["phase"] for r in rows}):
        subset = [r for r in rows if r["phase"] == phase]
        phase_summary[phase] = {k:statistics.median(r[k] for r in subset)
                               for k in ("span_ms", "blocking_api_ms", "guard_ms", "non_sync_span_ms", "long_launch_api_ms", "launches")}
    streams, kernels = defaultdict(lambda: defaultdict(float)), Counter()
    with (directory/"activities.csv").open() as f:
        for a in csv.DictReader(f):
            streams[f"rank{a['rank']}/stream{a['stream']}"][f"{a['phase']}:{a['kind']}"] += float(a["ms"])
            kernels[a["name"]] += 1
    # An empirical effective launch queue: issued kernel APIs minus completed
    # GPU kernels at the start of a long launch call. This is an observation,
    # not an assertion about a documented CUDA queue capacity.
    gpu_by_pid = defaultdict(dict)
    for a,b,gpid,corr in db.execute("select start,end,globalPid,correlationId from CUPTI_ACTIVITY_KIND_KERNEL"):
        pid=(gpid >> 24) & 0xFFFFFF
        if pid in rank_pid:
            gpu_by_pid[pid][corr]=(a,b)
    launch_stalls=[]
    for pid,apis in api_by_pid.items():
        launches=[(a,b,c) for a,b,n,c in apis if "LaunchKernel" in n and c in gpu_by_pid[pid]]
        begin=sorted(a for a,b,c in launches)
        done=sorted(b for a,b in gpu_by_pid[pid].values())
        for a,b,c in launches:
            if b-a > 200000:
                launch_stalls.append(dict(rank=rank_pid[pid]["rank"], duration_ms=(b-a)/1e6,
                                          outstanding_kernels=bisect.bisect_right(begin,a)-bisect.bisect_right(done,a)))
    event_audit = dict(records=0, waits=0, resolved_generations=0, unresolved_generations=0)
    if {"CUPTI_ACTIVITY_KIND_CUDA_EVENT", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"} <= tables:
        records = set(db.execute("select globalPid,contextId,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT"))
        event_audit["records"] = len(records)
        for pid, ctx, event, generation in db.execute("select globalPid,contextId,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2"):
            event_audit["waits"] += 1
            event_audit["resolved_generations" if (pid,ctx,event,generation) in records else "unresolved_generations"] += 1
    db.close()
    result = dict(case=directory.name, config=ranks[0]["config"], cpu_phases=phase_summary,
                  cpu_rows=rows, cuda_api_counts=dict(api_counts), streams=dict(streams),
                  top_kernel_counts=kernels.most_common(20), event_audit=event_audit,
                  launch_stalls=launch_stalls,
                  median_outstanding_at_stall=statistics.median(r["outstanding_kernels"] for r in launch_stalls) if launch_stalls else None,
                  notes=["CPU spans are from Nsight and include instrumentation overhead",
                         "Non-sync span still includes implicit driver backpressure inside kernel launch APIs",
                         "Blocking API time and failure-monitor CPU polling are distinct",
                         "Unresolved event generations can precede the capture; no dependency is invented"])
    (directory/"mechanism_v2.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", type=Path, nargs="+")
    for directory in p.parse_args().directories:
        result = audit(directory)
        print(directory.name, json.dumps(result["cpu_phases"]), flush=True)
