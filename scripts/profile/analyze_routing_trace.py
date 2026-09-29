"""Extract routing copy sections and NCCL waits from paired nsys SQLite files."""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean


def analyze(path):
    connection = sqlite3.connect(path)
    by_thread = defaultdict(list)
    phases = defaultdict(list)
    for start, end, tid, label in connection.execute(
        "select n.start,n.end,n.globalTid,coalesce(n.text,s.value) from NVTX_EVENTS n "
        "left join StringIds s on n.textId=s.id where n.end is not null"
    ):
        if label and (label.startswith("routing:") or label.startswith("nccl:") or label.startswith("multi_sae:")):
            by_thread[tid].append((start, end, label))
            phases[(tid >> 24, label)].append((end - start) / 1e6)
    host_sections = defaultdict(list)
    for tid, ranges in by_thread.items():
        waits = [(s, e) for s, e, label in ranges if label == "routing:AsyncRoutingTransport.wait_until"]
        for start, end, label in ranges:
            if label not in ("routing:_Ring.publish", "routing:_Ring.read"):
                continue
            # Payload memcpy plus small header/ACK/locking work. Waits on an
            # empty/full ring are excluded; this is NOT a standalone memcpy timer.
            excluded = sum(e - s for s, e in waits if start <= s and e <= end)
            host_sections[(tid >> 24, label)].append((end - start - excluded) / 1e6)
    copies = [dict(pid=pid, gpu=gpu, kind=kind, bytes=size, count=count, mean_ms=duration)
              for pid, gpu, kind, size, count, duration in connection.execute(
                  "select globalPid >> 24,deviceId,copyKind,bytes,count(*),avg(end-start)/1e6 "
                  "from CUPTI_ACTIVITY_KIND_MEMCPY where bytes >= 1048576 and copyKind in (1,2) group by 1,2,3,4")]
    collectives = [dict(gpu=gpu, name=name, count=count, total_ms=total, mean_ms=duration)
                   for gpu, name, count, total, duration in connection.execute(
                       "select k.deviceId,s.value,count(*),sum(k.end-k.start)/1e6,avg(k.end-k.start)/1e6 "
                       "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id "
                       "where s.value like 'nccl%' group by 1,2")]
    launches = {(tid >> 24, corr): (start, tid) for start, tid, corr in connection.execute(
        "select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    attributed = defaultdict(list)
    for start, end, pid, corr in connection.execute(
        "select k.start,k.end,k.globalPid >> 24,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on k.shortName=s.id where k.deviceId=0 and s.value like '%u64%'"
    ):
        launch = launches.get((pid, corr))
        if launch is None:
            continue
        issued, tid = launch
        enclosing = [(e - s, label) for s, e, label in by_thread[tid] if s <= issued <= e]
        label = min(enclosing)[1] if enclosing else "unattributed"
        attributed[label].append((end - start) / 1e6)
    connection.close()
    return dict(sqlite=str(path), phases=[dict(pid=p, label=l, count=len(v), mean_ms=mean(v))
                                       for (p, l), v in phases.items()],
                host_copy_and_header_sections=[dict(pid=p, label=l, count=len(v), mean_ms=mean(v))
                                               for (p, l), v in host_sections.items()],
                copies=copies, collectives=collectives,
                gpu0_u64_launch_scope=[dict(label=label, count=len(v), mean_ms=mean(v))
                                       for label, v in attributed.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = {mode: analyze(args.directory / f"trace_tp4_{mode}.sqlite")
              for mode in ("nccl", "shm", "shm_precise")
              if (args.directory / f"trace_tp4_{mode}.sqlite").exists()}
    (args.directory / "trace_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    for mode, report in result.items():
        print(mode, "host sections", report["host_copy_and_header_sections"])
        print("GPU0 u64 launch scope", report["gpu0_u64_launch_scope"])
        for row in report["collectives"]:
            if row["gpu"] == 0 and ("u64" in row["name"] or "Broadcast" in row["name"]):
                print(row)


if __name__ == "__main__":
    main()
