"""Attribute captured CUDA kernels to the actual sparse-parts launch ranges.

Kernel-sum time is diagnostic, not wall time; kernel execution can overlap.
Correlate launch PID+ID and host thread, including autograd worker threads.
Exclude launches outside native measured updates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def analyze(directory):
    directory = Path(directory)
    db = sqlite3.connect(f"file:{directory/'trace.sqlite'}?mode=ro", uri=True)
    strings = dict(db.execute("select id,value from StringIds"))
    ranks = [json.loads(p.read_text()) for p in directory.glob("rank[0-9]*.json")]
    pid_rank = {r["pid"]: r["rank"] for r in ranks}
    ranges, steps = defaultdict(list), defaultdict(list)
    for a, b, tid, text, textid in db.execute("select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null"):
        pid = (tid >> 24) & 0xFFFFFF
        tag = text or strings.get(textid, "")
        if pid not in pid_rank:
            continue
        if tag.startswith("time:sparse:"):
            ranges[tid].append((a, b, tag.removeprefix("time:sparse:")))
        elif tag.startswith("time:step:"):
            steps[pid].append((a, b, int(tag.split(":")[-1])))
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    launches = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for a, b, tid, corr in db.execute(f"select start,end,globalTid,correlationId from {table}"):
            pid = (tid >> 24) & 0xFFFFFF
            if pid in pid_rank:
                launches[pid, corr].append((a, b, tid))
    totals = defaultdict(lambda: dict(kernel_ms=0., kernels=0))
    missing = defaultdict(int)
    for a, b, gpid, corr in db.execute("select start,end,globalPid,correlationId from CUPTI_ACTIVITY_KIND_KERNEL"):
        pid = (gpid >> 24) & 0xFFFFFF
        if pid not in pid_rank:
            continue
        candidates = [(x, y, tid) for x, y, tid in launches[pid, corr] if x <= a]
        if not candidates:
            if any(x <= a <= y for x, y, _ in steps[pid]):
                missing[pid] += 1
            continue
        x, y, tid = max(candidates)
        step = next((n for lo, hi, n in steps[pid] if lo <= x and y <= hi), None)
        if step is None:
            continue
        matched = [(hi-lo, tag) for lo, hi, tag in ranges[tid] if lo <= x and y <= hi]
        stage = min(matched)[1] if matched else "other"
        row = totals[pid_rank[pid], step, stage]
        row["kernel_ms"] += (b-a)/1e6
        row["kernels"] += 1
    counts = {pid_rank[pid]: len(spans) for pid, spans in steps.items()}
    summary = {}
    for rank, n in counts.items():
        summary[str(rank)] = {stage: dict(kernel_ms_per_update=sum(v["kernel_ms"] for (r, _, s), v in totals.items() if r == rank and s == stage)/n,
                                          kernels_per_update=sum(v["kernels"] for (r, _, s), v in totals.items() if r == rank and s == stage)/n)
                              for stage in ("sparse_forward", "sparse_dvalues", "sparse_dweight", "other")}
    result = dict(case=directory.name, steps_per_rank=counts, stages_by_rank=summary,
                  unmatched_kernel_launches=dict(missing),
                  note="Mean per-update kernel duration sum; overlaps and host gaps prevent interpreting it as wall time")
    (directory/"sparse_stages.json").write_text(json.dumps(result, indent=2)+"\n")
    db.close()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(analyze(p.parse_args().directory), indent=2))
