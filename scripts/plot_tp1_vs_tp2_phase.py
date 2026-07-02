"""Plot TP1 vs TP2 SAE training-phase memory profiles.

Reads memory_phase_history_rank*.jsonl from a TP1 run and a TP2 run, then
produces two figures:

  * tp1_vs_tp2_phase_peak.png            — per-phase peak_allocated_mb
                                           (median over the stable-step window),
                                           grouped bar chart.
  * tp1_vs_tp2_phase_component_stacked.png — per-phase stacked components
                                             (params/grads/adam/raw+scaled batch
                                             /retained outputs/unattributed),
                                             stable-step median, side-by-side.

For TP2 we report rank-0 numbers (rank-1 is identical in our 2-hook run; both
are dumped to a CSV alongside the figures so anyone can verify symmetry).

Usage::

    python3 scripts/plot_tp1_vs_tp2_phase.py \\
        --tp1-dir results/memory_model/sae_phase_v4/tp1 \\
        --tp2-dir results/memory_model/sae_phase_v4/tp2 \\
        --output-dir results/memory_model/sae_phase_v4/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

PHASE_ORDER = [
    "after_data_fetch",
    "after_scale_to_device",
    "after_zero_grad_start",
    "after_forward_all",
    "after_combined_backward",
    "after_post_backward",
    "after_optimizer_step",
    "after_stats_tail",
]

COMPONENTS = [
    ("params_mb", "params"),
    ("grads_mb", "grads"),
    ("optimizer_state_mb", "adam"),
    ("trainer_buffers_mb", "trainer_buf"),
    ("raw_batch_mb", "raw_batch"),
    ("scaled_batch_mb", "scaled_batch"),
    ("retained_outputs_mb", "retained_outputs"),
    ("current_outputs_mb", "current_outputs"),
    ("data_provider_buffers_mb", "data_provider_buffers"),
    ("unattributed_allocated_mb", "unattributed (autograd saved)"),
]

# Transient (peak_allocated - allocated): memory that lived only *inside* the
# phase — autograd workspace during backward, Adam temporaries during the
# optimizer step — and was freed before the phase-end snapshot. The COMPONENTS
# above sum to allocated_mb (the resting snapshot), so without this band the
# backward/optimizer phases look far cheaper than the watermark that actually
# governs whether the run fits in VRAM.
TRANSIENT_LABEL = "transient (peak-alloc)"
TRANSIENT_COLOR = "0.7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tp1-dir", type=Path, required=True)
    p.add_argument("--tp2-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--stable-from-step",
        type=int,
        default=10,
        help="Aggregate from this step onwards (skips init / warmup transients).",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_phase_summary(
    rows: list[dict], stable_from_step: int
) -> dict[str, dict[str, float]]:
    by_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["step"] < stable_from_step:
            continue
        phase = row["phase"]
        by_phase[phase]["peak_allocated_mb"].append(row["peak_allocated_mb"])
        by_phase[phase]["allocated_mb"].append(row["allocated_mb"])
        # Track reserved + driver too — peak_allocated does NOT determine
        # whether the run actually fits in GPU VRAM. driver_used (what
        # nvidia-smi sees) is the binding constraint.
        by_phase[phase]["peak_reserved_mb"].append(row["peak_reserved_mb"])
        by_phase[phase]["driver_used_mb"].append(row["driver_used_mb"])
        for key, _label in COMPONENTS:
            by_phase[phase][key].append(row.get(key, 0.0))
    out: dict[str, dict[str, float]] = {}
    for phase, vals in by_phase.items():
        out[phase] = {k: median(v) for k, v in vals.items()}
    return out


def run_max(rows: list[dict], stable_from_step: int, key: str) -> float:
    return max(r[key] for r in rows if r["step"] >= stable_from_step)


def plot_phase_peak(
    tp1: dict[str, dict[str, float]],
    tp2: dict[str, dict[str, float]],
    out_path: Path,
    tp1_run_max: dict[str, float],
    tp2_run_max: dict[str, float],
) -> None:
    phases = [p for p in PHASE_ORDER if p in tp1 and p in tp2]
    tp1_vals = [tp1[p]["peak_allocated_mb"] for p in phases]
    tp2_vals = [tp2[p]["peak_allocated_mb"] for p in phases]

    x = np.arange(len(phases))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, tp1_vals, width, label="TP=1 peak_allocated", color="#3b82f6")
    bars2 = ax.bar(x + width / 2, tp2_vals, width, label="TP=2 (rank0) peak_allocated", color="#ef4444")

    # Reference lines: peak_reserved (allocator pool with fragmentation) and
    # driver_used (what nvidia-smi sees, the actual VRAM the OS treats as
    # used). peak_allocated alone does NOT determine fit — these lines show
    # how much extra real VRAM is needed beyond the bar heights.
    ax.axhline(
        tp1_run_max["peak_reserved_mb"],
        color="#3b82f6",
        linestyle="--",
        linewidth=1,
        alpha=0.55,
        label=f"TP=1 peak_reserved ({tp1_run_max['peak_reserved_mb']:.0f})",
    )
    ax.axhline(
        tp1_run_max["driver_used_mb"],
        color="#1e3a8a",
        linestyle="-",
        linewidth=1.4,
        alpha=0.85,
        label=f"TP=1 driver_used ({tp1_run_max['driver_used_mb']:.0f}) ← real VRAM",
    )
    ax.axhline(
        tp2_run_max["peak_reserved_mb"],
        color="#ef4444",
        linestyle="--",
        linewidth=1,
        alpha=0.55,
        label=f"TP=2 peak_reserved ({tp2_run_max['peak_reserved_mb']:.0f})",
    )
    ax.axhline(
        tp2_run_max["driver_used_mb"],
        color="#7f1d1d",
        linestyle="-",
        linewidth=1.4,
        alpha=0.85,
        label=f"TP=2 driver_used ({tp2_run_max['driver_used_mb']:.0f}) ← real VRAM",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("after_", "") for p in phases], rotation=20, ha="right")
    ax.set_ylabel("MB (median over stable steps for bars; run-max for ref lines)")
    ax.set_title(
        "TP=1 vs TP=2 — per-phase peak_allocated, with run-wide "
        "peak_reserved + driver_used reference lines\n"
        "(driver_used is what determines whether the run fits on the GPU; "
        "bars do NOT)"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bars, vals in ((bars1, tp1_vals), (bars2, tp2_vals)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_component_stacked(
    tp1: dict[str, dict[str, float]],
    tp2: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    phases = [p for p in PHASE_ORDER if p in tp1 and p in tp2]
    n = len(phases)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, (_, label) in enumerate(COMPONENTS)}

    def stacked(side: str, x_off: float, summary: dict[str, dict[str, float]]):
        bottom = np.zeros(n)
        for key, label in COMPONENTS:
            vals = np.array([summary[p].get(key, 0.0) for p in phases])
            ax.bar(
                x + x_off,
                vals,
                width,
                bottom=bottom,
                color=colors[label],
                label=f"{label}" if side == "TP=1" else None,
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += vals
        # Transient band on top: peak_allocated - allocated. The components
        # above sum to allocated (resting snapshot); this band restores the
        # in-phase watermark so backward/optimizer peaks are visible and the
        # stack total matches plot_phase_peak's peak_allocated bars.
        transient = np.array(
            [
                max(
                    0.0,
                    summary[p].get("peak_allocated_mb", 0.0)
                    - summary[p].get("allocated_mb", 0.0),
                )
                for p in phases
            ]
        )
        ax.bar(
            x + x_off,
            transient,
            width,
            bottom=bottom,
            color=TRANSIENT_COLOR,
            hatch="//",
            label=TRANSIENT_LABEL if side == "TP=1" else None,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += transient
        # Add total label on top of stack (now == peak_allocated_mb)
        for i, total in enumerate(bottom):
            ax.text(
                x[i] + x_off,
                total,
                f"{total:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        # group label
        ax.text(
            (x[0] + x_off + x[-1] + x_off) / 2,
            -max(bottom) * 0.05,
            side,
            ha="center",
            va="top",
            fontsize=10,
            transform=ax.transData,
        )

    stacked("TP=1", -width / 2, tp1)
    stacked("TP=2 (rank0)", width / 2, tp2)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("after_", "") for p in phases], rotation=20, ha="right")
    ax.set_ylabel("peak_allocated_mb, decomposed (median over stable steps)")
    ax.set_title(
        "TP=1 vs TP=2 — per-phase memory components + transient watermark "
        "(multi-hook training)\n"
        "(stack total = peak_allocated_mb; grey hatched band = in-phase "
        "transient freed before the snapshot)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(
    tp1: dict[str, dict[str, float]],
    tp2_rank0: dict[str, dict[str, float]],
    tp2_rank1: dict[str, dict[str, float]] | None,
    out_path: Path,
) -> None:
    fields = [
        "phase",
        "tp1_peak_mb",
        "tp1_allocated_mb",
        "tp1_peak_reserved_mb",
        "tp1_driver_used_mb",
        "tp2_r0_peak_mb",
        "tp2_r0_allocated_mb",
        "tp2_r0_peak_reserved_mb",
        "tp2_r0_driver_used_mb",
        "tp2_r1_peak_mb",
        "tp2_r1_allocated_mb",
        "tp1_unattributed_mb",
        "tp2_r0_unattributed_mb",
    ]
    phases = [p for p in PHASE_ORDER if p in tp1]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in phases:
            row = {
                "phase": p,
                "tp1_peak_mb": f"{tp1[p]['peak_allocated_mb']:.2f}",
                "tp1_allocated_mb": f"{tp1[p]['allocated_mb']:.2f}",
                "tp1_peak_reserved_mb": f"{tp1[p].get('peak_reserved_mb', 0.0):.2f}",
                "tp1_driver_used_mb": f"{tp1[p].get('driver_used_mb', 0.0):.2f}",
                "tp2_r0_peak_mb": f"{tp2_rank0[p]['peak_allocated_mb']:.2f}"
                if p in tp2_rank0
                else "",
                "tp2_r0_allocated_mb": f"{tp2_rank0[p]['allocated_mb']:.2f}"
                if p in tp2_rank0
                else "",
                "tp2_r0_peak_reserved_mb": f"{tp2_rank0[p].get('peak_reserved_mb', 0.0):.2f}"
                if p in tp2_rank0
                else "",
                "tp2_r0_driver_used_mb": f"{tp2_rank0[p].get('driver_used_mb', 0.0):.2f}"
                if p in tp2_rank0
                else "",
                "tp2_r1_peak_mb": f"{tp2_rank1[p]['peak_allocated_mb']:.2f}"
                if tp2_rank1 and p in tp2_rank1
                else "",
                "tp2_r1_allocated_mb": f"{tp2_rank1[p]['allocated_mb']:.2f}"
                if tp2_rank1 and p in tp2_rank1
                else "",
                "tp1_unattributed_mb": f"{tp1[p].get('unattributed_allocated_mb', 0.0):.2f}",
                "tp2_r0_unattributed_mb": f"{tp2_rank0[p].get('unattributed_allocated_mb', 0.0):.2f}"
                if p in tp2_rank0
                else "",
            }
            w.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tp1_rows = read_jsonl(args.tp1_dir / "memory_phase_history_rank0.jsonl")
    tp2_r0_rows = read_jsonl(args.tp2_dir / "memory_phase_history_rank0.jsonl")
    tp2_r1_path = args.tp2_dir / "memory_phase_history_rank1.jsonl"
    tp2_r1_rows = read_jsonl(tp2_r1_path) if tp2_r1_path.exists() else []

    tp1 = stable_phase_summary(tp1_rows, args.stable_from_step)
    tp2_r0 = stable_phase_summary(tp2_r0_rows, args.stable_from_step)
    tp2_r1 = stable_phase_summary(tp2_r1_rows, args.stable_from_step) if tp2_r1_rows else None

    tp1_run_max = {
        "peak_reserved_mb": run_max(tp1_rows, args.stable_from_step, "peak_reserved_mb"),
        "driver_used_mb": run_max(tp1_rows, args.stable_from_step, "driver_used_mb"),
    }
    tp2_run_max = {
        "peak_reserved_mb": run_max(tp2_r0_rows, args.stable_from_step, "peak_reserved_mb"),
        "driver_used_mb": run_max(tp2_r0_rows, args.stable_from_step, "driver_used_mb"),
    }

    plot_phase_peak(
        tp1,
        tp2_r0,
        args.output_dir / "tp1_vs_tp2_phase_peak.png",
        tp1_run_max,
        tp2_run_max,
    )
    plot_component_stacked(
        tp1, tp2_r0, args.output_dir / "tp1_vs_tp2_phase_component_stacked.png"
    )
    write_csv(tp1, tp2_r0, tp2_r1, args.output_dir / "tp1_vs_tp2_phase_summary.csv")
    print(f"wrote figures + CSV to {args.output_dir}")
    print(
        f"  TP=1 run-max: peak_reserved={tp1_run_max['peak_reserved_mb']:.0f} MB, "
        f"driver_used={tp1_run_max['driver_used_mb']:.0f} MB"
    )
    print(
        f"  TP=2 run-max: peak_reserved={tp2_run_max['peak_reserved_mb']:.0f} MB, "
        f"driver_used={tp2_run_max['driver_used_mb']:.0f} MB"
    )


if __name__ == "__main__":
    main()
