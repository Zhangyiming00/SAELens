"""Compare ``driver_used_mb`` per phase under two recording strategies:

  watermark   — empty_cache OFF: driver_used reflects the cumulative
                allocator pool watermark; once a step has needed N MB the
                driver still sees N MB even after the tensors are freed.
                This is what real training looks like — the OS-level
                "in use" never decreases until the process exits.

  freed       — empty_cache ON: every phase recording is preceded by
                ``torch.cuda.empty_cache()``, which forces the allocator
                to return idle blocks to the driver. driver_used now
                tracks "live tensors right now". This is what an
                instantaneous ``cat /proc/X/status`` would show if you
                also free()'d caches; it is NOT what nvidia-smi reports
                during real training.

The freed series is interesting for understanding *which phase is
expensive*; the watermark series is what determines whether the run
fits on the GPU.

Inputs: two output dirs each containing ``memory_phase_history_rank0.jsonl``
for TP=1 and TP=2, one with empty_cache OFF and one with empty_cache ON.

Outputs (--output-dir):
  - tp1_vs_tp2_driver_watermark_vs_freed.png : 2x1 grouped bar chart
  - tp1_vs_tp2_driver_watermark_vs_freed.csv : per-phase numbers
  - timing_overhead.txt                      : empty_cache wall-clock cost
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

PHASE_ORDER = [
    "after_data_fetch",
    "after_scale_to_device",
    "after_forward_all",
    "after_combined_backward",
    "after_post_backward",
    "after_optimizer_step",
    "after_zero_grad_start",
    "after_stats_tail",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--watermark-tp1-dir", type=Path, required=True)
    p.add_argument("--watermark-tp2-dir", type=Path, required=True)
    p.add_argument("--freed-tp1-dir", type=Path, required=True)
    p.add_argument("--freed-tp2-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stable-from-step", type=int, default=10)
    return p.parse_args()


def per_phase_driver_median(jsonl: Path, stable_from: int) -> dict[str, float]:
    rows = [json.loads(line) for line in jsonl.open() if line.strip()]
    rows = [r for r in rows if r["step"] >= stable_from]
    by_phase: dict[str, list[float]] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r["driver_used_mb"])
    return {ph: median(vs) for ph, vs in by_phase.items()}


def total_runtime(dir_path: Path) -> float | None:
    """Read ``total_time_history.jsonl`` from the parent results dir; the
    runner appends one record per run, keyed by the leaf output dir name
    (``run_id``). We pick the most recent matching record."""
    parent = dir_path.parent / "total_time_history.jsonl"
    if not parent.exists():
        return None
    run_id = dir_path.name
    last_run_secs: float | None = None
    for line in parent.open():
        rec = json.loads(line)
        if rec.get("run_id") == run_id:
            last_run_secs = rec.get("total_time_s") or rec.get("total_runtime_seconds")
    return last_run_secs


def plot_pair(
    tp1_watermark: dict[str, float],
    tp1_freed: dict[str, float],
    tp2_watermark: dict[str, float],
    tp2_freed: dict[str, float],
    out_path: Path,
) -> None:
    phases = [p for p in PHASE_ORDER if p in tp1_watermark and p in tp1_freed]
    n = len(phases)
    x = np.arange(n)
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    def panel(ax, watermark, freed, title, color_water, color_freed):
        wm = [watermark[p] for p in phases]
        fr = [freed[p] for p in phases]
        b1 = ax.bar(x - width / 2, wm, width, color=color_water,
                    label="empty_cache OFF (watermark, what nvidia-smi sees)")
        b2 = ax.bar(x + width / 2, fr, width, color=color_freed,
                    label="empty_cache ON (instantaneous live tensors)")
        for bars, vals in ((b1, wm), (b2, fr)):
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7,
                )
        # Annotate the gap (watermark − freed) for each phase
        for xi, w, f in zip(x, wm, fr):
            gap = w - f
            ax.annotate(
                f"+{gap:.0f}",
                xy=(xi, max(w, f) + 200),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444",
            )
        ax.set_title(title)
        ax.set_ylabel("driver_used_mb (median over stable steps)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    panel(axes[0], tp1_watermark, tp1_freed,
          "TP=1 — driver_used per phase (gap = allocator pool fragment)",
          "#3b82f6", "#93c5fd")
    panel(axes[1], tp2_watermark, tp2_freed,
          "TP=2 (rank0) — driver_used per phase",
          "#ef4444", "#fca5a5")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [p.replace("after_", "") for p in phases], rotation=20, ha="right"
    )
    fig.suptitle(
        "Caching-allocator watermark vs. freed driver_used — same training, "
        "different memory accounting",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    tp1_watermark: dict[str, float],
    tp1_freed: dict[str, float],
    tp2_watermark: dict[str, float],
    tp2_freed: dict[str, float],
    out_path: Path,
) -> None:
    phases = [p for p in PHASE_ORDER if p in tp1_watermark]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "phase",
            "tp1_watermark_mb", "tp1_freed_mb", "tp1_gap_mb",
            "tp2_watermark_mb", "tp2_freed_mb", "tp2_gap_mb",
        ])
        for p in phases:
            tp1w = tp1_watermark[p]
            tp1f = tp1_freed[p]
            tp2w = tp2_watermark[p]
            tp2f = tp2_freed[p]
            w.writerow([
                p,
                f"{tp1w:.1f}", f"{tp1f:.1f}", f"{tp1w - tp1f:.1f}",
                f"{tp2w:.1f}", f"{tp2f:.1f}", f"{tp2w - tp2f:.1f}",
            ])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tp1w = per_phase_driver_median(
        args.watermark_tp1_dir / "memory_phase_history_rank0.jsonl",
        args.stable_from_step,
    )
    tp1f = per_phase_driver_median(
        args.freed_tp1_dir / "memory_phase_history_rank0.jsonl",
        args.stable_from_step,
    )
    tp2w = per_phase_driver_median(
        args.watermark_tp2_dir / "memory_phase_history_rank0.jsonl",
        args.stable_from_step,
    )
    tp2f = per_phase_driver_median(
        args.freed_tp2_dir / "memory_phase_history_rank0.jsonl",
        args.stable_from_step,
    )

    plot_pair(
        tp1w, tp1f, tp2w, tp2f,
        args.output_dir / "tp1_vs_tp2_driver_watermark_vs_freed.png",
    )
    write_csv(
        tp1w, tp1f, tp2w, tp2f,
        args.output_dir / "tp1_vs_tp2_driver_watermark_vs_freed.csv",
    )

    # Timing overhead: read total_runtime from each parent's
    # total_time_history.jsonl (the runner appends a record per run).
    timing_lines = []
    for label, dir_path in [
        ("watermark TP1", args.watermark_tp1_dir),
        ("watermark TP2", args.watermark_tp2_dir),
        ("freed     TP1", args.freed_tp1_dir),
        ("freed     TP2", args.freed_tp2_dir),
    ]:
        runtime = total_runtime(dir_path)
        timing_lines.append(
            f"{label}: {runtime:.2f} s" if runtime else f"{label}: (no runtime record)"
        )
    (args.output_dir / "timing_overhead.txt").write_text("\n".join(timing_lines) + "\n")

    print(f"wrote outputs to {args.output_dir}")
    for line in timing_lines:
        print("  " + line)


if __name__ == "__main__":
    main()
