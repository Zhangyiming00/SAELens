"""Attribute GA=1 Nsight CUDA activities through launch-thread NVTX ranges.

Correlation IDs are process-local. GPU durations include actual kernel/memcpy/
memset execution, without charging asynchronous host launch time as GPU time.
"""

import argparse
import bisect
import csv
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def union_ns(intervals, lo=None, hi=None):
    merged = []
    for a, b in sorted(intervals):
        a = max(a, lo) if lo is not None else a
        b = min(b, hi) if hi is not None else b
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def describe(values):
    values = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values),
                p50=statistics.median(values), p95=values[int((len(values)-1)*0.95)],
                min=values[0], max=values[-1]) if values else None


def phase_of(text):
    if text.startswith("ga1:blocks."):
        _, hook, phase, *_ = text.split(":")
        return hook, phase
    if text.startswith("sae:") and text.endswith(":backward_ready_wait"):
        return text.split(":")[1], "hook_wait"
    if text == "multi_sae:train_step":
        return "all", "train_other"
    if text == "multi_sae:data_fetch":
        return "all", "data_fetch"
    return "all", "loop_other"


def analyze(sqlite_path, run_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    inv = json.loads((run_dir / "invocation.json").read_text())
    warmup, count = inv["arguments"]["warmup"], inv["arguments"]["steps"]
    count *= inv["arguments"].get("windows", 1)
    ranks = [json.loads(p.read_text()) for p in sorted(run_dir.glob("rank*.json"))]
    pid_rank = {r["pid"]: r["rank"] for r in ranks if r["units"]}
    con = sqlite3.connect(sqlite_path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    strings = dict(con.execute("select id, value from StringIds"))
    nvtx = []
    for text, text_id, a, b, tid in con.execute(
        "select text, textId, start, end, globalTid from NVTX_EVENTS where end is not null"
    ):
        text = text or strings.get(text_id, "")
        pid = (tid >> 24) & 0xFFFFFF
        if pid in pid_rank and (text.startswith("ga1:") or text.startswith("sae:") or text.startswith("multi_sae:")):
            nvtx.append(dict(text=text, a=a, b=b, tid=tid, pid=pid))
    steps = defaultdict(list)
    for n in nvtx:
        match = re.fullmatch(r"ga1:step:r(\d+):s(\d+)", n["text"])
        if match and warmup < int(match[2]) <= warmup + count:
            steps[n["pid"]].append(dict(**n, step=int(match[2]), rank=int(match[1])))
    assert set(steps) == set(pid_rank), (set(steps), set(pid_rank))
    for pid, values in steps.items():
        values.sort(key=lambda s: s["a"])
        assert [s["step"] for s in values] == list(range(warmup + 1, warmup + count + 1))
    step_starts = {pid: [s["a"] for s in ss] for pid, ss in steps.items()}

    def step_at(pid, t):
        i = bisect.bisect_right(step_starts[pid], t) - 1
        if i >= 0 and t < steps[pid][i]["b"]:
            return steps[pid][i]
        return None

    ranges = defaultdict(list)
    for n in nvtx:
        step = step_at(n["pid"], n["a"])
        if step and not n["text"].startswith("ga1:step:"):
            n["hook"], n["phase"] = phase_of(n["text"])
            ranges[(n["pid"], step["step"])].append(n)
    runtime = {}
    runtime_events = defaultdict(list)
    for a, b, corr, tid, name in con.execute(
        "select start, end, correlationId, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        pid = (tid >> 24) & 0xFFFFFF
        if pid in pid_rank:
            runtime[(pid, corr)] = (a, b, tid)
            runtime_events[pid].append((a, b, strings.get(name, "")))
    activities = defaultdict(list)
    for a, b, corr, pid, name in con.execute(
        "select start, end, correlationId, (globalPid >> 24) & 16777215, shortName from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        if pid in pid_rank:
            activities[pid].append(dict(a=a, b=b, corr=corr, name=strings[name], kind="kernel"))
    for table, kind in [("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")]:
        if table in tables:
            for a, b, corr, pid in con.execute(f"select start, end, correlationId, (globalPid >> 24) & 16777215 from {table}"):
                if pid in pid_rank:
                    activities[pid].append(dict(a=a, b=b, corr=corr, name=kind, kind=kind))
    con.close()
    attributed = defaultdict(list)
    kernel_totals = defaultdict(lambda: [0, 0.0])
    for pid, acts in activities.items():
        for act in acts:
            launch = runtime.get((pid, act["corr"]))
            if not launch:
                continue
            step = step_at(pid, launch[0])
            if not step:
                continue
            enclosing = [n for n in ranges[(pid, step["step"])]
                         if n["tid"] == launch[2] and n["a"] <= launch[0] < n["b"]]
            if not enclosing:
                # Autograd executes GPU launches on its worker thread while
                # backward() is open on the main thread. That parent range is
                # valid here because hooks/backwards are strictly sequential.
                enclosing = [n for n in ranges[(pid, step["step"])]
                             if n["tid"] == step["tid"] and n["a"] <= launch[0] < n["b"]]
            n = min(enclosing, key=lambda n: n["b"] - n["a"]) if enclosing else None
            hook, phase = (n["hook"], n["phase"]) if n else ("all", "loop_other")
            act.update(hook=hook, phase=phase, launch=launch[0], step=step["step"])
            attributed[(pid, step["step"])].append(act)
            k = (pid_rank[pid], hook, phase, act["name"])
            kernel_totals[k][0] += 1
            kernel_totals[k][1] += (act["b"] - act["a"]) / 1e6
    phase_rows, step_rows, wait_rows = [], [], []
    for pid, ss in steps.items():
        all_gpu = [(a["a"], a["b"]) for a in activities[pid]]
        for step in ss:
            rank, sn = step["rank"], step["step"]
            ns, acts = ranges[(pid, sn)], attributed[(pid, sn)]
            phases = sorted({(n["hook"], n["phase"]) for n in ns} | {(a["hook"], a["phase"]) for a in acts})
            for hook, phase in phases:
                pr = [n for n in ns if (n["hook"], n["phase"]) == (hook, phase)]
                pa = [a for a in acts if (a["hook"], a["phase"]) == (hook, phase)]
                inclusive = sum(n["b"] - n["a"] for n in pr)
                # Exclusive CPU time removes every nested NVTX phase, so e.g.
                # finish_window is normalization alone after grad_sync returns.
                exclusive = sum(n["b"] - n["a"] - union_ns([
                    (c["a"], c["b"]) for c in ns if c is not n and c["tid"] == n["tid"]
                    and n["a"] <= c["a"] and c["b"] <= n["b"]
                ]) for n in pr)
                phase_rows.append(dict(rank=rank, step=sn, hook=hook, phase=phase,
                    calls=len(pr), cpu_inclusive_ms=inclusive / 1e6, cpu_exclusive_ms=exclusive / 1e6,
                    gpu_events=len(pa), gpu_sum_ms=sum(a["b"]-a["a"] for a in pa) / 1e6,
                    gpu_union_ms=union_ns([(a["a"], a["b"]) for a in pa]) / 1e6,
                    nccl_gpu_sum_ms=sum(a["b"]-a["a"] for a in pa if "nccl" in a["name"].lower()) / 1e6,
                ))
            for n in ns:
                if n["phase"] == "hook_wait":
                    busy = union_ns(all_gpu, n["a"], n["b"])
                    wait_rows.append(dict(rank=rank, step=sn, hook=n["hook"],
                        host_ms=(n["b"]-n["a"])/1e6, gpu_busy_during_ms=busy/1e6,
                        gpu_idle_during_ms=(n["b"]-n["a"]-busy)/1e6))
            train = next(n for n in ns if n["text"] == "multi_sae:train_step")
            train_acts = [a for a in acts if train["a"] <= a["launch"] < train["b"]]
            gpu_intervals = [(a["a"], a["b"]) for a in train_acts]
            ddp = [(a["a"], a["b"]) for a in train_acts if a["phase"] == "ddp_launch"]
            other = [(a["a"], a["b"]) for a in train_acts if a["phase"] != "ddp_launch"]
            backward_done = max(a["b"] for a in train_acts if a["phase"] in ("backward", "grad_add"))
            syncs = [(a,b) for a,b,name in runtime_events[pid]
                     if train["a"] <= a < train["b"] and "Synchronize" in name]
            step_rows.append(dict(rank=rank, step=sn,
                host_step_ms=(step["b"]-step["a"])/1e6,
                host_train_ms=(train["b"]-train["a"])/1e6,
                train_to_gpu_done_ms=(max([train["b"]]+[a["b"] for a in train_acts])-train["a"])/1e6,
                train_gpu_union_ms=union_ns(gpu_intervals)/1e6,
                train_gpu_sum_ms=sum(b-a for a,b in gpu_intervals)/1e6,
                ddp_gpu_union_ms=union_ns(ddp)/1e6,
                ddp_overlap_other_gpu_ms=(union_ns(ddp)+union_ns(other)-union_ns(gpu_intervals))/1e6,
                ddp_without_other_gpu_ms=(union_ns(gpu_intervals)-union_ns(other))/1e6,
                ddp_before_backward_done_ms=union_ns(ddp,hi=backward_done)/1e6,
                ddp_after_backward_done_ms=union_ns(ddp,lo=backward_done)/1e6,
                train_host_cuda_sync_ms=union_ns(syncs)/1e6,
                host_train_gpu_idle_ms=((train["b"]-train["a"])-union_ns(all_gpu,train["a"],train["b"]))/1e6,
            ))
    for name, rows in [("phases",phase_rows),("steps",step_rows),("waits",wait_rows)]:
        with (output / f"{name}.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with (output / "timeline.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "step", "lane", "hook", "phase", "start_ns", "end_ns", "name", "launch_ns"])
        for (pid, sn), acts in attributed.items():
            ns = ranges[(pid, sn)]
            train = next(n for n in ns if n["text"] == "multi_sae:train_step")
            for a in acts:
                if train["a"] <= a["launch"] < train["b"]:
                    w.writerow([pid_rank[pid], sn, "GPU", a["hook"], a["phase"], a["a"], a["b"], a["name"], a["launch"]])
            for n in ns:
                if n["phase"] == "hook_wait":
                    w.writerow([pid_rank[pid], sn, "CPU wait", n["hook"], n["phase"], n["a"], n["b"], n["text"], ""])
    for r in phase_rows:
        expected = {"grad_add": 4, "grad_normalize": 4,
                    "grad_zero_start": 1, "grad_zero_end": 1}
        if r["phase"] in expected:
            assert r["gpu_events"] == expected[r["phase"]], r
        if r["phase"] == "ddp_launch":
            unit = next(rank["units"][r["hook"]] for rank in ranks if rank["rank"] == r["rank"])
            assert r["gpu_events"] == (0 if len(pid_rank) == 1 else len(unit["buckets"])), r
    with (output / "kernels.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["rank","hook","phase","kernel","count","total_ms","per_step_ms"])
        for key,(num,ms) in sorted(kernel_totals.items(), key=lambda kv: -kv[1][1]):
            w.writerow([*key,num,ms,ms/count])
    summary = dict(
        trace=str(sqlite_path), run=str(run_dir), measured_steps=count,
        end_to_end=json.loads((run_dir / "summary.json").read_text()),
        ranks={}, hooks={}, phases={},
        method=[
            "GPU activities attributed by (PID, correlationId) to innermost NVTX on the CUDA launch thread.",
            "Autograd worker launches without their own annotation inherit the enclosing main-thread phase (sequential hook execution).",
            "GPU sum includes kernel/memcpy/memset durations; concurrent GPU work can overlap, so it is not wall time.",
            "Host wait idle is the complement of all GPU activity on that rank during its readiness fence; not a counterfactual speedup.",
            "NCCL kernel time includes device-side peer waiting, and cannot be interpreted as pure wire time.",
            "Per-rank and per-hook phase means are per measured step; no summing across DP/TP ranks.",
        ],
    )
    for rank in pid_rank.values():
        rows = [r for r in step_rows if r["rank"] == rank]
        summary["ranks"][rank] = {k: describe([r[k] for r in rows]) for k in rows[0] if k.endswith("_ms")}
    for hook in sorted({r["hook"] for r in wait_rows}):
        wr = [r for r in wait_rows if r["hook"] == hook]
        summary["hooks"][hook] = {k: describe([r[k] for r in wr]) for k in ("host_ms","gpu_busy_during_ms","gpu_idle_during_ms")}
    for hook,phase in sorted({(r["hook"],r["phase"]) for r in phase_rows}):
        pr = [r for r in phase_rows if (r["hook"],r["phase"]) == (hook,phase)]
        summary["phases"][f"{hook}:{phase}"] = {k: describe([r[k] for r in pr]) for k in ("calls","cpu_inclusive_ms","cpu_exclusive_ms","gpu_events","gpu_sum_ms","gpu_union_ms","nccl_gpu_sum_ms")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(output=str(output), steps=count, ranks=list(pid_rank.values()), end_to_end_ms=summary["end_to_end"]["per_step_ms"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.sqlite, args.run, args.output)
