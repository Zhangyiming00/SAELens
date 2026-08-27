#!/usr/bin/env python3
"""Two tables for the parallelism cross sweep.

Table 1 -- everything from one nsys timeline, so the parts share units:

    nsys_vllm    GPU-busy time inside the `multi_sae:data_fetch` range
    nsys_sae     GPU-busy time of the rest of the step, NCCL included (DDP/FSDP
                 launch it from autograd threads outside every NVTX range)
    nsys_gap     nsys_total - nsys_vllm - nsys_sae
    nsys_total   the step's wall interval on the timeline: from the start of the
                 fetch that feeds it to the end of its `multi_sae:train_step`
    non_nsys_total  the same quantity measured by the step-window profiler in a
                 separate un-traced run, for reference

Because vLLM and SAE do not run concurrently on one device (verified:
GPU-busy buckets overlap by 0.00ms), nsys_gap is genuine device idle plus CPU
stretches with nothing in flight, and stays positive.

Table 2 -- the predictive model, with no gap term:

    profiled_vllm   calls_per_step / vllm_dp * per_call_median(vllm_tp), from
                    explore_vllm_activation_profile_v3_2.py --suite runmatch
    profiled_sae    the simulator's per-step SAE estimate (sim_ms)
    profiled_total  profiled_vllm + profiled_sae, added directly
    non_nsys_total  measured wall per step (step-window profiler)
    error           profiled_total - non_nsys_total, and the same as a percentage

With multiple ranks a step is summarized by its slowest rank, since a step is not
finished until its last rank finishes.
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

NCCL_PREFIXES = ("nccl", "ncclDevKernel")

# The activation delivery work between producer and SAE ranks. Routing is its
# own phase -- neither the model's prefill nor SAE compute -- so it gets its own
# column rather than being folded into either.
#
# `nccl:shard_routing_p2p_exchange` covers remote producer->consumer transfer.
# It is measured with the usual cross-rank max.
ROUTING_MAX_RANGES = ("nccl:shard_routing_p2p_exchange",)

# `nccl:shard_routing_sae_tp_broadcast` covers the consumer-root fanout to other
# SAE TP ranks. It is a routing cost, but one side can enter the collective
# before the root has finished vLLM generation. The wait-inclusive NCCL kernel
# span can therefore be as long as vLLM itself. For this range, use the nonzero
# cross-rank min as the transfer-side cost and avoid double-counting producer
# vLLM wait.
ROUTING_MIN_RANGES = ("nccl:shard_routing_sae_tp_broadcast",)

ROUTING_RANGES = ROUTING_MAX_RANGES + ROUTING_MIN_RANGES

# Both range groups are activation delivery, and excluding the broadcast makes
# SAE-TP routing look artificially free.
#
# `nccl:shard_routing_p2p_barrier` is NOT routing work: it is a consumer waiting
# for a producer to finish generating, and it lasts as long as that wait. Treating
# it as routing would double-count the producer's vLLM work across ranks.

# The mixing buffer runs inside the caller's data-fetch range but is SAE-side
# work (concat, shuffle, slice of the activation pool), so it is attributed to
# the SAE, not to vLLM.
MIXING_RANGES = (
    "mixing_buffer:append",
    "mixing_buffer:shuffle",
)

# Simulator per-step SAE estimate (sim_ms) at H=1, d_in=4096, d_sae=65536,
# batch=4096, k=256, fused Adam, keyed by SAE topology.
PROFILED_SAE_MS = {
    "single": 723.75,
    "tp2": 478.40,
    "ddp": 585.23,
    "fsdp": 729.97,
}

# Measured GPU-hardware SAE step time (real_gpu_hw_ms) for the same configs, used
# only to cross-check the timeline measurement.
REF_SAE_GPU_HW_MS = {
    "single": 757.19,
    "tp2": 489.62,
    "ddp": 621.69,
    "fsdp": 755.74,
}


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


def load_trace(path: Path) -> dict:
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

    launches: dict[int, int] = {}
    for s, c in con.execute(
        "select start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where correlationId is not null"
    ):
        launches[int(c)] = int(s)

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


def analyze(trace: dict, drop_warmup: int = 2) -> dict | None:
    acts, launches, nvtx = trace["acts"], trace["launches"], trace["nvtx"]
    steps_all = sorted(nvtx.get("multi_sae:train_step", []))
    fetches_all = sorted(nvtx.get("multi_sae:data_fetch", []))
    if not steps_all:
        return None

    def ranges_for(names: tuple[str, ...], pid: int) -> list[tuple[int, int]]:
        out = [
            (s, e) for name in names for s, e, p in nvtx.get(name, []) if p == pid
        ]
        out.sort()
        return out

    per_pid: dict[int, list[dict]] = {}
    for pid in sorted(acts):
        steps = [s for s in steps_all if s[2] == pid]
        fetches = [f for f in fetches_all if f[2] == pid]
        if not steps:
            continue
        routing_max = ranges_for(ROUTING_MAX_RANGES, pid)
        routing_min = ranges_for(ROUTING_MIN_RANGES, pid)
        mixing = ranges_for(MIXING_RANGES, pid)
        rows = []
        for s_start, s_end, _ in steps:
            prior = [f for f in fetches if f[0] <= s_start]
            fetch = (prior[-1][0], prior[-1][1]) if prior else (s_start, s_start)
            # Correlation ids by launching phase. Routing and the mixing buffer
            # both run inside the fetch's CPU range, so they are matched first and
            # what remains of the fetch is the model's prefill.
            def corrs_in(windows: list[tuple[int, int]]) -> set[int]:
                return {
                    c
                    for c, lt in launches.items()
                    if any(w_s <= lt < w_e for w_s, w_e in windows)
                }

            routing_max_corrs = corrs_in(routing_max)
            routing_min_corrs = corrs_in(routing_min)
            routing_corrs = routing_max_corrs | routing_min_corrs
            mixing_corrs = corrs_in(mixing)
            fetch_corrs = {
                c
                for c, lt in launches.items()
                if fetch[0] <= lt < fetch[1]
            } - routing_corrs - mixing_corrs
            step_corrs = corrs_in([(s_start, s_end)])
            # The interval is one step PERIOD: from this step's fetch start to the
            # next step's fetch start (the last step runs to where its own GPU work
            # ends). A step's own GPU work outlives the CPU range that launched it --
            # `multi_sae:train_step` closes after ~218ms of launches while its
            # kernels keep running for ~750ms -- so ending the interval at `s_end`
            # drops ~530ms of SAE work per step. Extending it to where that work
            # finishes instead double-counts, because consecutive steps pipeline and
            # the tail overlaps the next step's own work. A period boundary keeps
            # each step's slice disjoint while still charging the step for whatever
            # GPU work occupies its slot.
            later = [f[0] for f in fetches if f[0] > fetch[0]]
            own_corrs = step_corrs | fetch_corrs | routing_corrs | mixing_corrs
            gpu_end = max(
                (a_e for a_s, a_e, corr, _ in acts[pid] if corr in own_corrs),
                default=s_end,
            )
            lo = fetch[0]
            hi = later[0] if later else max(s_end, gpu_end)
            buckets: dict[str, list[Iv]] = {
                "vllm": [],
                "sae": [],
                "routing_max": [],
                "routing_min": [],
            }
            for a_s, a_e, corr, name in acts[pid]:
                if a_e <= lo or a_s >= hi:
                    continue
                iv = Iv(a_s, a_e)
                # Which phase an activity belongs to is decided by WHO LAUNCHED it,
                # then by where it ran. Hardware position alone is not enough in
                # either direction: a prefill kernel whose tail lands inside the
                # next `multi_sae:train_step` window would be credited to the SAE,
                # and the previous step's SAE GEMMs draining through a fetch window
                # would be credited to vLLM.
                if corr in routing_max_corrs:
                    buckets["routing_max"].append(iv)
                elif corr in routing_min_corrs:
                    buckets["routing_min"].append(iv)
                elif corr in mixing_corrs:
                    buckets["sae"].append(iv)
                elif corr in fetch_corrs:
                    buckets["vllm"].append(iv)
                elif name.startswith(NCCL_PREFIXES):
                    # DDP/FSDP gradient traffic: launched from autograd threads,
                    # so no range encloses it, but it is SAE work.
                    buckets["sae"].append(iv)
                else:
                    buckets["sae"].append(iv)
            v = union_len(buckets["vllm"], lo, hi)
            s = union_len(buckets["sae"], lo, hi)
            r_max = union_len(buckets["routing_max"], lo, hi)
            r_min = union_len(buckets["routing_min"], lo, hi)
            both = union_len(
                buckets["vllm"]
                + buckets["sae"]
                + buckets["routing_max"]
                + buckets["routing_min"],
                lo,
                hi,
            )
            rows.append(
                {
                    "wall": hi - lo,
                    "vllm": v,
                    "sae": s,
                    "routing_max": r_max,
                    "routing_min": r_min,
                    "routing": r_max + r_min,
                    "overlap": v + s + r_max + r_min - both,
                }
            )
        per_pid[pid] = rows

    if not per_pid:
        return None
    n = min(len(v) for v in per_pid.values())
    if n <= drop_warmup:
        return None
    idx = list(range(drop_warmup, n))
    # `nsys_total` is the slowest rank's wall: the step is not done until its last
    # rank is done.
    #
    # The phase columns are each taken from the rank that PERFORMED that phase, by
    # taking the max across ranks. With vllm_dp=1 only rank 0 prefills, so reading
    # vLLM off the slowest rank -- often a consumer whose own device never
    # prefills -- reports ~0 and dumps the wait into whatever bucket absorbed it.
    # Since vLLM and SAE do not run concurrently on one device (`overlap` is 0),
    # taking each phase from its own performer does not double-count.
    def mean_phase(key: str) -> float:
        return sum(max(per_pid[p][i][key] for p in per_pid) for i in idx) / len(
            idx
        ) / 1e6

    def mean_nonzero_min_phase(key: str) -> float:
        total = 0
        for i in idx:
            values = [per_pid[p][i][key] for p in per_pid if per_pid[p][i][key] > 0]
            total += min(values) if values else 0
        return total / len(idx) / 1e6

    def mean_slowest(key: str) -> float:
        return (
            sum(
                max((per_pid[p][i] for p in per_pid), key=lambda r: r["wall"])[key]
                for i in idx
            )
            / len(idx)
            / 1e6
        )

    nsys_routing_ms = mean_phase("routing_max") + mean_nonzero_min_phase(
        "routing_min"
    )

    return {
        "n_ranks": len(per_pid),
        "n_steps": len(idx),
        "nsys_total_ms": mean_slowest("wall"),
        "nsys_vllm_ms": mean_phase("vllm"),
        "nsys_sae_ms": mean_phase("sae"),
        "nsys_routing_ms": nsys_routing_ms,
        "overlap_ms": mean_slowest("overlap"),
        "vllm_producer_ms": mean_phase("vllm"),
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
    p.add_argument("--nsys-root", type=Path, default=ROOT / "results" / "cross_nsys")
    p.add_argument(
        "--window-root",
        type=Path,
        default=ROOT / "results" / "cross_all",
        help="un-traced sweep providing non_nsys_total",
    )
    p.add_argument(
        "--vllm-json",
        type=Path,
        default=ROOT / "results" / "vllm_runmatch" / "profile.json",
    )
    args = p.parse_args()

    profile = json.loads(args.vllm_json.read_text())
    per_call = {r["tp"]: r["wall_ms_median"] for r in profile["rows"]}
    tokens_per_call = {r["tp"]: r["total_tokens"] for r in profile["rows"]}

    manifest = [
        json.loads(line)
        for line in (args.nsys_root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    rows: list[dict] = []
    for spec in manifest:
        sqlite_path = args.nsys_root / "nsys" / f"{spec['run_id']}.sqlite"
        if not sqlite_path.exists():
            print(f"  skip {spec['run_id']}: no sqlite")
            continue
        nsys = analyze(load_trace(sqlite_path))
        if nsys is None:
            print(f"  skip {spec['run_id']}: no usable steps")
            continue
        non_nsys = window_total_ms(
            args.window_root / "runner_outputs" / spec["run_id"]
        )
        vtp, vdp = spec["vllm_tp_size"], spec["vllm_dp_size"]
        calls = spec["batch_tokens"] / tokens_per_call[vtp]
        prof_vllm = calls / vdp * per_call[vtp]
        prof_sae = PROFILED_SAE_MS[spec["sae_mode"]]
        prof_total = prof_vllm + prof_sae
        rows.append(
            {
                "run_id": spec["run_id"],
                "sae_mode": spec["sae_mode"],
                "vllm_mode": spec["vllm_mode"],
                "sae_tp": spec["sae_tp_size"],
                "sae_dp": spec["sae_dp_size"],
                "vllm_tp": vtp,
                "vllm_dp": vdp,
                "n_ranks": nsys["n_ranks"],
                "n_steps": nsys["n_steps"],
                "nsys_vllm_ms": nsys["nsys_vllm_ms"],
                "nsys_sae_ms": nsys["nsys_sae_ms"],
                "nsys_routing_ms": nsys["nsys_routing_ms"],
                "vllm_producer_ms": nsys["vllm_producer_ms"],
                "nsys_gap_ms": nsys["nsys_total_ms"]
                - nsys["nsys_vllm_ms"]
                - nsys["nsys_sae_ms"]
                - nsys["nsys_routing_ms"],
                "nsys_total_ms": nsys["nsys_total_ms"],
                "overlap_ms": nsys["overlap_ms"],
                "non_nsys_total_ms": non_nsys,
                "profiled_vllm_ms": prof_vllm,
                "profiled_sae_ms": prof_sae,
                "profiled_total_ms": prof_total,
                "error_ms": (prof_total - non_nsys) if non_nsys else None,
                "error_pct": ((prof_total - non_nsys) / non_nsys * 100.0)
                if non_nsys
                else None,
                "ref_sae_gpu_hw_ms": REF_SAE_GPU_HW_MS[spec["sae_mode"]],
                "nsys_sae_vs_ref_pct": (
                    nsys["nsys_sae_ms"] - REF_SAE_GPU_HW_MS[spec["sae_mode"]]
                )
                / REF_SAE_GPU_HW_MS[spec["sae_mode"]]
                * 100.0,
                "nsys_total_vs_non_nsys_pct": (
                    (nsys["nsys_total_ms"] - non_nsys) / non_nsys * 100.0
                )
                if non_nsys
                else None,
                # Per-component error of the profiled model against the timeline.
                "err_vllm_pct": (prof_vllm - nsys["nsys_vllm_ms"])
                / nsys["nsys_vllm_ms"]
                * 100.0
                if nsys["nsys_vllm_ms"]
                else None,
                "err_sae_pct": (prof_sae - nsys["nsys_sae_ms"])
                / nsys["nsys_sae_ms"]
                * 100.0
                if nsys["nsys_sae_ms"]
                else None,
            }
        )
        print(f"  ok {spec['run_id']}")

    if not rows:
        print("no runs with a usable trace yet")
        return

    sae_order = {"single": 0, "tp2": 1, "ddp": 2, "fsdp": 3}
    vllm_order = {"vtp1dp1": 0, "vtp2dp1": 1, "vtp1dp2": 2}
    # Keep the table order stable even when some runs are missing from a manifest.
    # This makes missing-config gaps obvious in the wide outputs.
    rows.sort(
        key=lambda r: (
            sae_order.get(r["sae_mode"], 9),
            vllm_order.get(r["vllm_mode"], 9),
        )
    )

    def fmt(value: float | None, width: int = 9, nd: int = 2) -> str:
        return "n/a".rjust(width) if value is None else f"{value:{width}.{nd}f}"

    def config_name(r: dict) -> str:
        """vLLM topology first, then SAE: e.g. vtp1dp1_stp1sdp1, vtp2dp1_stp1fsdp2."""
        vllm = f"vtp{r['vllm_tp']}dp{r['vllm_dp']}"
        # FSDP is a sharded form of data parallel, so it is named in the dp slot.
        dp_label = "fsdp" if r["sae_mode"] == "fsdp" else "sdp"
        return f"{vllm}_stp{r['sae_tp']}{dp_label}{r['sae_dp']}"

    for r in rows:
        r["config"] = config_name(r)

    cfg_w = max(len("config"), max(len(r["config"]) for r in rows))

    lines: list[str] = []
    lines.append("Table 1 -- measured on one nsys timeline (ms)")
    h1 = (
        f"{'config':{cfg_w}s} {'nsys_vllm':>10s} {'nsys_sae':>10s} "
        f"{'nsys_routing':>13s} {'nsys_gap':>9s} {'nsys_total':>11s} "
        f"{'non_nsys_total':>15s}"
    )
    lines += [h1, "-" * len(h1)]
    for r in rows:
        lines.append(
            f"{r['config']:{cfg_w}s} {r['nsys_vllm_ms']:10.2f} "
            f"{r['nsys_sae_ms']:10.2f} {r['nsys_routing_ms']:13.2f} "
            f"{r['nsys_gap_ms']:9.2f} {r['nsys_total_ms']:11.2f} "
            f"{fmt(r['non_nsys_total_ms'], 15)}"
        )

    lines.append("")
    lines.append("Table 2 -- per-component error of the profiled model")
    h2 = (
        f"{'config':{cfg_w}s} {'nsys_vllm':>10s} {'prof_vllm':>10s} "
        f"{'err_vllm%':>10s} {'nsys_sae':>10s} {'prof_sae':>10s} {'err_sae%':>9s} "
        f"{'nsys_total':>11s} {'prof_total':>11s} {'non_nsys_total':>15s} "
        f"{'err%':>7s}"
    )
    lines += [h2, "-" * len(h2)]
    for r in rows:
        lines.append(
            f"{r['config']:{cfg_w}s} {r['nsys_vllm_ms']:10.2f} "
            f"{r['profiled_vllm_ms']:10.2f} {fmt(r['err_vllm_pct'], 10)} "
            f"{r['nsys_sae_ms']:10.2f} {r['profiled_sae_ms']:10.2f} "
            f"{fmt(r['err_sae_pct'], 9)} {r['nsys_total_ms']:11.2f} "
            f"{r['profiled_total_ms']:11.2f} "
            f"{fmt(r['non_nsys_total_ms'], 15)} {fmt(r['error_pct'], 7)}"
        )
    table = "\n".join(lines)

    out_dir = args.nsys_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nsys_vs_profiled.txt").write_text(table + "\n")

    # One wide CSV: config, then every table-1 column, then every table-2 column,
    # plus the reference SAE number and the per-column errors against it.
    columns = [
        "config",
        "run_id",
        "sae_mode",
        "vllm_mode",
        "H",
        "d_in",
        "d_sae",
        "batch",
        "k",
        "sae_tp",
        "sae_dp",
        "vllm_tp",
        "vllm_dp",
        "n_ranks",
        "n_steps",
        # Table 1: measured on the nsys timeline.
        "nsys_vllm_ms",
        "nsys_sae_ms",
        "nsys_routing_ms",
        "nsys_gap_ms",
        "nsys_total_ms",
        "overlap_ms",
        "vllm_producer_ms",
        # Table 2: profiled model.
        "profiled_vllm_ms",
        "profiled_sae_ms",
        "profiled_total_ms",
        "non_nsys_total_ms",
        "error_ms",
        "error_pct",
        # Per-component error of the model against the timeline.
        "err_vllm_pct",
        "err_sae_pct",
        # Cross-checks.
        "ref_sae_gpu_hw_ms",
        "nsys_sae_vs_ref_pct",
        "nsys_total_vs_non_nsys_pct",
    ]
    def write_csv(path: Path, fields: list[str], nd: int = 2) -> None:
        """Write `fields` for every row, rounding floats to `nd` decimals."""
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                out: dict[str, object] = {}
                for key in fields:
                    value = r.get(key)
                    out[key] = round(value, nd) if isinstance(value, float) else value
                writer.writerow(out)

    write_csv(out_dir / "nsys_vs_profiled.csv", columns)

    # Table 1 on its own: the nsys timeline measurement.
    write_csv(
        out_dir / "table1_nsys.csv",
        [
            "config",
            "nsys_vllm_ms",
            "nsys_sae_ms",
            "nsys_routing_ms",
            "nsys_gap_ms",
            "nsys_total_ms",
            "non_nsys_total_ms",
        ],
    )

    # Table 2 on its own: the profiled model's error, per component and in total.
    write_csv(
        out_dir / "table2_profiled_error.csv",
        [
            "config",
            "nsys_vllm_ms",
            "profiled_vllm_ms",
            "err_vllm_pct",
            "nsys_sae_ms",
            "profiled_sae_ms",
            "err_sae_pct",
            "nsys_total_ms",
            "profiled_total_ms",
            "non_nsys_total_ms",
            "error_pct",
        ],
    )
    print()
    print(table)
    print()
    print(f"wrote {out_dir / 'table1_nsys.csv'}")
    print(f"wrote {out_dir / 'table2_profiled_error.csv'}")
    print(f"wrote {out_dir / 'nsys_vs_profiled.csv'} (wide)")


if __name__ == "__main__":
    main()
