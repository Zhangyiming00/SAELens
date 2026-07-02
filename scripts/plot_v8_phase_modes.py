"""Plot per-phase SAE training memory components for N parallelism modes.

Generalizes ``plot_tp1_vs_tp2_phase.py`` from a fixed TP1-vs-TP2 pair to an
arbitrary set of modes (e.g. single / tp2 / ddp2 / fsdp2). Each mode is a run
directory containing ``memory_phase_history_rank0.jsonl``; for DP modes rank0
is reported (replicas are symmetric, and rank1 is dumped to the CSV).

Produces ``tp1_vs_tp2_phase_component_stacked.png``: one stacked bar per mode
per phase, where the stack decomposes peak_allocated_mb into params/grads/adam/
batch/retained-outputs/unattributed + the in-phase transient watermark.

Usage::

    python3 scripts/plot_v8_phase_modes.py \\
        --mode single=results/memory_model/sae_phase_v8/single \\
        --mode tp2=results/memory_model/sae_phase_v8/tp2 \\
        --mode ddp2=results/memory_model/sae_phase_v8/ddp2 \\
        --mode fsdp2=results/memory_model/sae_phase_v8/fsdp2 \\
        --output-dir results/memory_model/sae_phase_v8/figures
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

TRANSIENT_LABEL = "transient (peak-alloc)"
TRANSIENT_COLOR = "0.7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="A mode to plot, e.g. tp2=results/.../tp2. Repeat for each mode; "
        "order is preserved left-to-right.",
    )
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
        by_phase[phase]["peak_reserved_mb"].append(row["peak_reserved_mb"])
        by_phase[phase]["driver_used_mb"].append(row["driver_used_mb"])
        for key, _label in COMPONENTS:
            by_phase[phase][key].append(row.get(key, 0.0))
    return {phase: {k: median(v) for k, v in vals.items()} for phase, vals in by_phase.items()}


def plot_component_stacked(
    summaries: list[tuple[str, dict[str, dict[str, float]]]],
    out_path: Path,
) -> None:
    # Phases present in every mode, in canonical order.
    phases = [p for p in PHASE_ORDER if all(p in s for _, s in summaries)]
    n = len(phases)
    n_modes = len(summaries)
    x = np.arange(n)
    # Cluster the modes within each phase slot, leaving a gap between phases.
    group_width = 0.82
    width = group_width / n_modes
    offsets = [(-group_width / 2) + width * (i + 0.5) for i in range(n_modes)]

    fig, ax = plt.subplots(figsize=(2.0 + 3.4 * n_modes, 6.5))

    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, (_, label) in enumerate(COMPONENTS)}

    def stacked(label: str, x_off: float, summary: dict[str, dict[str, float]], first: bool):
        bottom = np.zeros(n)
        for key, comp_label in COMPONENTS:
            vals = np.array([summary[p].get(key, 0.0) for p in phases])
            ax.bar(
                x + x_off,
                vals,
                width,
                bottom=bottom,
                color=colors[comp_label],
                label=comp_label if first else None,
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += vals
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
            label=TRANSIENT_LABEL if first else None,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += transient
        for i, total in enumerate(bottom):
            ax.text(
                x[i] + x_off,
                total,
                f"{total:.0f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
        ymax = max(bottom) if len(bottom) else 1.0
        for i in range(n):
            ax.text(
                x[i] + x_off,
                -ymax * 0.04,
                label,
                ha="center",
                va="top",
                fontsize=7,
                rotation=90,
            )

    for i, (label, summary) in enumerate(summaries):
        stacked(label, offsets[i], summary, first=(i == 0))

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("after_", "") for p in phases], rotation=20, ha="right")
    ax.set_ylabel("peak_allocated_mb, decomposed (median over stable steps)")
    mode_names = " vs ".join(label for label, _ in summaries)
    ax.set_title(
        f"{mode_names} — per-phase memory components + transient watermark "
        "(multi-hook cached training)\n"
        "(stack total = peak_allocated_mb; grey hatched band = in-phase "
        "transient freed before the snapshot; DP modes show rank0)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(
    summaries: list[tuple[str, dict[str, dict[str, float]]]],
    out_path: Path,
) -> None:
    phases = [p for p in PHASE_ORDER if any(p in s for _, s in summaries)]
    fields = ["phase"]
    for label, _ in summaries:
        fields += [
            f"{label}_peak_mb",
            f"{label}_allocated_mb",
            f"{label}_peak_reserved_mb",
            f"{label}_driver_used_mb",
        ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in phases:
            row: dict[str, object] = {"phase": p}
            for label, summary in summaries:
                if p in summary:
                    row[f"{label}_peak_mb"] = f"{summary[p]['peak_allocated_mb']:.2f}"
                    row[f"{label}_allocated_mb"] = f"{summary[p]['allocated_mb']:.2f}"
                    row[f"{label}_peak_reserved_mb"] = f"{summary[p].get('peak_reserved_mb', 0.0):.2f}"
                    row[f"{label}_driver_used_mb"] = f"{summary[p].get('driver_used_mb', 0.0):.2f}"
                else:
                    for suffix in ("peak_mb", "allocated_mb", "peak_reserved_mb", "driver_used_mb"):
                        row[f"{label}_{suffix}"] = ""
            w.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[str, dict[str, dict[str, float]]]] = []
    for spec in args.mode:
        if "=" not in spec:
            raise ValueError(f"--mode expects LABEL=DIR, got: {spec}")
        label, dir_str = spec.split("=", 1)
        run_dir = Path(dir_str)
        rows = read_jsonl(run_dir / "memory_phase_history_rank0.jsonl")
        summaries.append((label, stable_phase_summary(rows, args.stable_from_step)))

    out_png = args.output_dir / "tp1_vs_tp2_phase_component_stacked.png"
    plot_component_stacked(summaries, out_png)
    write_csv(summaries, args.output_dir / "phase_modes_summary.csv")
    print(f"wrote {out_png}")
    print(f"wrote {args.output_dir / 'phase_modes_summary.csv'}")


if __name__ == "__main__":
    main()
