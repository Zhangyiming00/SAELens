#!/usr/bin/env python3
"""vLLM / SAE / gap / total table for the parallelism cross sweep.

`total_ms` -- measured. Per-step wall clock from the step-window profiler: the
window interval, synced only at the two window boundaries, divided by the
window's step count. With several ranks the slowest rank's window is used, since
a step ends when its last rank ends.

`vllm_ms` -- the vLLM prefill cost per SAE step, from the standalone profile:

    calls_per_step = train_batch_size_tokens / tokens_per_call
    vllm_ms        = calls_per_step / vllm_dp * per_call_median(vllm_tp)

`tokens_per_call` is what one activation-generation call covers,
`store_batch_size_prompts * context_size` = 1 * 2048, so a 4096-token step needs
2 calls. TP shards a single call across its ranks and so does not change the
count, only the per-call cost, which the profile measures separately per tp. DP
replicas split the calls and run concurrently, hence the division by `vllm_dp`.

The profile comes from `explore_vllm_activation_profile_v3_2.py --suite runmatch`,
whose case reproduces the runs exactly: hook `blocks.21.hook_resid_post`,
stop_at_layer 22 (the runs derive the same as `max(hook_layers) + 1`), B=1,
context 2048, mbt at its 4096 default, max_model_len 2049, gpu_memory_utilization
0.5, bfloat16 (the runs' `--dtype float32` sets the SAE dtype; vLLM itself runs
bf16, confirmed in the run logs). mbt is deliberately left at the default: it only
bounds how many chunks one prefill is split into, so raising it would measure a
different call than the runs make.

Do NOT use the older `vllm_multihook_profile.json` here. Its per-call figure for
this case is 415.76ms against 203.83ms measured by the runmatch case, and its
`ms_per_1k_tokens` (203.01) equals the runmatch per-call total for 2048 tokens --
i.e. it double-counts each call. Using it drove `vllm + sae` above `total`.

`measured_fetch_ms` is the run's own fetch timer for comparison, max over DP
replicas and, within a replica, max over its TP ranks. It runs above `vllm_ms`
because a fetch also iterates the dataset, runs the mixing buffer's
randperm/cat/index copies, and does the host-to-device copy.

`sae_ms` -- measured GPU-hardware time per SAE step, supplied per topology from
the nsys reference (`real_gpu_hw_ms`): for each `multi_sae:train_step` range, the
hardware span of the GPU activity it launched, rank-max, averaged over
steady-state steps. The in-step CPU timer (`sae_time_s`) is NOT used: it is a
launch span and underestimates this by 2.2x to 5.4x, worst for ddp where the
gradient allreduce is launched from autograd threads that no timer covers.

`gap_ms = total - vllm - sae`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Measured GPU-hardware SAE step time (real_gpu_hw_ms) at
# H=1, d_in=4096, d_sae=65536, batch=4096, k=256, fused Adam, keyed by SAE mode.
SAE_GPU_HW_MS = {
    "single": 757.19,
    "tp2": 489.62,
    "ddp": 621.69,
    "fsdp": 755.74,
}

def load_timing(run_dir: Path) -> list[dict]:
    path = run_dir / "timing_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fetch_ms_per_rank(records: list[dict]) -> dict[int, float]:
    """Per-step fetch time each rank reports, averaged over its windows."""
    out: dict[int, float] = {}
    for rank in sorted({r["_rank"] for r in records}):
        mine = [r for r in records if r["_rank"] == rank]
        steps = sum(r["steps"] for r in mine)
        if steps == 0:
            continue
        total = sum(r["components"].get("vllm_step_time_s", 0.0) for r in mine)
        out[rank] = total / steps * 1000.0
    return out


def load_windows(run_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(run_dir.glob("step_window_profile_sae_rank*.jsonl")):
        rank = int(path.stem.rsplit("rank", 1)[1])
        for line in path.read_text().splitlines():
            if line:
                record = json.loads(line)
                record["_rank"] = rank
                records.append(record)
    return records


def window_totals(records: list[dict]) -> dict | None:
    """Per-step wall clock per window, taking the slowest rank in each window."""
    if not records:
        return None
    per_window: list[float] = []
    ranges: list[tuple[int, int]] = []
    total = 0.0
    steps = 0
    for window in sorted({r["window"] for r in records}):
        group = [r for r in records if r["window"] == window]
        slowest = max(group, key=lambda r: r["window_time_s"])
        per_window.append(slowest["window_time_s"] / slowest["steps"] * 1000.0)
        ranges.append((slowest["start_step"], slowest["end_step"]))
        total += slowest["window_time_s"]
        steps += slowest["steps"]
    if steps == 0:
        return None
    return {
        "n_ranks": len({r["_rank"] for r in records}),
        "steps": steps,
        "ranges": ranges,
        "w1_ms": per_window[0],
        "w2_ms": per_window[1] if len(per_window) > 1 else None,
        "spread_pct": (max(per_window) - min(per_window)) / min(per_window) * 100.0
        if len(per_window) > 1
        else None,
        "total_ms": total / steps * 1000.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "results" / "cross_all")
    p.add_argument(
        "--vllm-json",
        type=Path,
        default=ROOT / "results" / "vllm_runmatch" / "profile.json",
        help=(
            "Output of explore_vllm_activation_profile_v3_2.py --suite runmatch, "
            "which profiles the exact call the training loop makes."
        ),
    )
    args = p.parse_args()

    profile = json.loads(args.vllm_json.read_text())
    # tp -> median wall ms for one prefill call, plus the tokens that call covers.
    per_call: dict[int, float] = {}
    tokens_per_call: dict[int, int] = {}
    for r in profile["rows"]:
        per_call[r["tp"]] = r["wall_ms_median"]
        tokens_per_call[r["tp"]] = r["total_tokens"]

    manifest = [
        json.loads(line)
        for line in (args.root / "run_manifest.jsonl").read_text().splitlines()
        if line
    ]

    rows: list[dict] = []
    missing_profile: list[str] = []
    for spec in manifest:
        totals = window_totals(load_windows(Path(spec["output_path"])))
        if totals is None:
            print(f"  skip {spec['run_id']}: no window records")
            continue
        vtp = spec["vllm_tp_size"]
        vdp = spec["vllm_dp_size"]
        if vtp not in per_call:
            missing_profile.append(f"{spec['run_id']} (vllm_tp={vtp})")
            continue
        # Global prefill calls a step needs: its tokens over the tokens one call
        # covers (store_batch_size_prompts * context). TP shards one call across
        # its ranks, so it does not change the count -- only the per-call cost,
        # which the profile already measures per tp. DP replicas split those calls
        # and run concurrently, so the wall cost per step divides by vllm_dp.
        calls_per_step = spec["batch_tokens"] / tokens_per_call[vtp]
        vllm_ms = calls_per_step / vdp * per_call[vtp]
        # Measured fetch wall for comparison: max over DP replicas, and within a
        # replica the max over its TP ranks (they run one call together).
        by_rank = fetch_ms_per_rank(load_windows(Path(spec["output_path"])))
        replica_ms = [
            max(
                [by_rank.get(replica * vtp + i, 0.0) for i in range(vtp)] or [0.0]
            )
            for replica in range(vdp)
        ]
        measured_fetch_ms = max(replica_ms) if replica_ms else 0.0
        sae_ms = SAE_GPU_HW_MS[spec["sae_mode"]]
        total_ms = totals["total_ms"]
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
                "n_ranks": totals["n_ranks"],
                "steps": totals["steps"],
                "w1_ms": totals["w1_ms"],
                "w2_ms": totals["w2_ms"],
                "spread_pct": totals["spread_pct"],
                "calls_per_step": calls_per_step,
                "vllm_per_call_ms": per_call[vtp],
                "tokens_per_call": tokens_per_call[vtp],
                "vllm_ms": vllm_ms,
                "measured_fetch_ms": measured_fetch_ms,
                "sae_ms": sae_ms,
                "gap_ms": total_ms - vllm_ms - sae_ms,
                "total_ms": total_ms,
                "vllm_pct": vllm_ms / total_ms * 100.0,
                "sae_pct": sae_ms / total_ms * 100.0,
                "gap_pct": (total_ms - vllm_ms - sae_ms) / total_ms * 100.0,
            }
        )
        print(f"  ok {spec['run_id']}")

    for entry in missing_profile:
        print(f"  skip {entry}: no vLLM per-call profile for that (H, tp)")

    sae_order = {"single": 0, "tp2": 1, "ddp": 2, "fsdp": 3}
    vllm_order = {"vtp1dp1": 0, "vtp2dp1": 1, "vtp1dp2": 2}
    rows.sort(
        key=lambda r: (
            sae_order.get(r["sae_mode"], 9),
            vllm_order.get(r["vllm_mode"], 9),
        )
    )

    header = (
        f"{'run_id':20s} {'sae':>7s} {'vllm':>8s} {'stp':>3s} {'sdp':>3s} "
        f"{'vtp':>3s} {'vdp':>3s} {'w1_ms':>8s} {'w2_ms':>8s} {'spr%':>5s} "
        f"{'c/step':>6s} {'vllm_ms':>9s} {'sae_ms':>9s} {'gap_ms':>9s} "
        f"{'total_ms':>9s}  {'vllm%':>6s} {'sae%':>6s} {'gap%':>6s}  {'fetch':>8s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        w2 = "n/a" if r["w2_ms"] is None else f"{r['w2_ms']:.2f}"
        spr = "n/a" if r["spread_pct"] is None else f"{r['spread_pct']:.2f}"
        lines.append(
            f"{r['run_id']:20s} {r['sae_mode']:>7s} {r['vllm_mode']:>8s} "
            f"{r['sae_tp']:3d} {r['sae_dp']:3d} {r['vllm_tp']:3d} {r['vllm_dp']:3d} "
            f"{r['w1_ms']:8.2f} {w2:>8s} {spr:>5s} {r['calls_per_step']:6.2f} "
            f"{r['vllm_ms']:9.2f} {r['sae_ms']:9.2f} {r['gap_ms']:9.2f} "
            f"{r['total_ms']:9.2f}  {r['vllm_pct']:6.1f} {r['sae_pct']:6.1f} "
            f"{r['gap_pct']:6.1f}  {r['measured_fetch_ms']:8.1f}"
        )
    table = "\n".join(lines)

    out_dir = args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_window_table.txt").write_text(table + "\n")
    with open(out_dir / "cross_window_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(table)


if __name__ == "__main__":
    main()
