"""Combine PyTorch allocator stats with driver watermark/freed series in one
figure, so the gap between every layer of the memory hierarchy is visible
at a glance.

Layers (low → high) for each phase:

  allocated_mb              — live tensors right now (PyTorch)
  peak_allocated_mb         — transient peak within phase (PyTorch)
  peak_reserved_mb          — allocator pool watermark (PyTorch)
  driver_used_mb (freed)    — driver bytes after empty_cache (instantaneous)
  driver_used_mb (watermark)— driver bytes without empty_cache (nvidia-smi)

The watermark experiment provides allocated / peak_allocated / peak_reserved /
driver_used(watermark). The freed experiment provides driver_used(freed).
allocated / peak_allocated should match between experiments — we use the
watermark run as the source of truth for those.

Usage::

    python3 scripts/plot_memory_layers_combined.py \
        --watermark-tp1-dir results/memory_model/sae_phase_v4/tp1 \
        --watermark-tp2-dir results/memory_model/sae_phase_v4/tp2 \
        --freed-tp1-dir     results/memory_model/sae_phase_v4_freed/tp1 \
        --freed-tp2-dir     results/memory_model/sae_phase_v4_freed/tp2 \
        --output-dir        results/memory_model/sae_phase_v4_freed/figures
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

LAYERS = [
    ("allocated_mb",        "allocated (live)",          "#1d4ed8"),
    ("peak_allocated_mb",   "peak_allocated (transient)","#3b82f6"),
    ("peak_reserved_mb",    "peak_reserved (pool)",      "#93c5fd"),
    ("driver_freed_mb",     "driver freed (empty_cache)","#fb923c"),
    ("driver_watermark_mb", "driver watermark (nvidia-smi)", "#dc2626"),
]
PROXY_KEY = "phase_driver_proxy_mb"
PROXY_LABEL = "phase driver proxy = peak_alloc + (freed - alloc)"
PROXY_COLOR = "#16a34a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--watermark-tp1-dir", type=Path, required=True)
    p.add_argument("--watermark-tp2-dir", type=Path, required=True)
    p.add_argument("--freed-tp1-dir", type=Path, required=True)
    p.add_argument("--freed-tp2-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stable-from-step", type=int, default=10)
    return p.parse_args()


def per_phase_median(jsonl: Path, stable_from: int, key: str) -> dict[str, float]:
    rows = [json.loads(line) for line in jsonl.open() if line.strip()]
    rows = [r for r in rows if r["step"] >= stable_from]
    by_phase: dict[str, list[float]] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r.get(key, 0.0))
    return {ph: median(vs) for ph, vs in by_phase.items()}


def collect(watermark_dir: Path, freed_dir: Path, stable_from: int) -> dict[str, dict[str, float]]:
    wm_jsonl = watermark_dir / "memory_phase_history_rank0.jsonl"
    fr_jsonl = freed_dir / "memory_phase_history_rank0.jsonl"
    allocated = per_phase_median(wm_jsonl, stable_from, "allocated_mb")
    peak_alloc = per_phase_median(wm_jsonl, stable_from, "peak_allocated_mb")
    peak_resv = per_phase_median(wm_jsonl, stable_from, "peak_reserved_mb")
    driver_wm = per_phase_median(wm_jsonl, stable_from, "driver_used_mb")
    driver_fr = per_phase_median(fr_jsonl, stable_from, "driver_used_mb")
    out: dict[str, dict[str, float]] = {}
    for ph in PHASE_ORDER:
        if ph not in allocated:
            continue
        out[ph] = {
            "allocated_mb": allocated[ph],
            "peak_allocated_mb": peak_alloc[ph],
            "peak_reserved_mb": peak_resv[ph],
            "driver_freed_mb": driver_fr.get(ph, float("nan")),
            "driver_watermark_mb": driver_wm[ph],
            "phase_driver_proxy_mb": (
                peak_alloc[ph] + (driver_fr.get(ph, float("nan")) - allocated[ph])
            ),
        }
    return out


def plot_panel(ax, summary: dict[str, dict[str, float]], title: str) -> None:
    phases = [p for p in PHASE_ORDER if p in summary]
    n = len(phases)
    n_layers = len(LAYERS)
    x = np.arange(n)
    width = 0.85 / n_layers

    for i, (key, label, color) in enumerate(LAYERS):
        vals = [summary[p][key] for p in phases]
        offset = (i - (n_layers - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=color, label=label,
                      edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.0f}",
                ha="center", va="bottom", fontsize=6, rotation=90,
            )

    # Per-phase driver proxy = peak_alloc + (freed - alloc). Draw as a
    # horizontal segment spanning the bar group so it reads as a "this is
    # what the driver would peak at if only this phase happened" line.
    proxy_vals = [summary[p][PROXY_KEY] for p in phases]
    half = (n_layers / 2) * width
    proxy_handle = None
    for xi, v in zip(x, proxy_vals):
        line, = ax.plot(
            [xi - half, xi + half],
            [v, v],
            color=PROXY_COLOR, linewidth=2.2, solid_capstyle="butt",
        )
        if proxy_handle is None:
            proxy_handle = line
            line.set_label(PROXY_LABEL)
        ax.text(
            xi, v, f"{v:.0f}",
            ha="center", va="bottom", fontsize=7,
            color=PROXY_COLOR, fontweight="bold",
        )

    # Phase whose proxy is the max — that's the binding phase for the
    # driver-peak model (D_total = max_p proxy_p + pool fragmentation).
    max_phase_idx = int(np.argmax(proxy_vals))
    max_proxy = proxy_vals[max_phase_idx]
    driver_wm = summary[phases[max_phase_idx]]["driver_watermark_mb"]
    pool_frag_tax = driver_wm - max_proxy
    ax.axhline(
        max_proxy,
        color=PROXY_COLOR, linestyle="--", linewidth=1.0, alpha=0.55,
        label=f"max_p proxy = {max_proxy:.0f}",
    )
    ax.axhline(
        driver_wm,
        color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.45,
        label=f"driver watermark = {driver_wm:.0f}  (pool tax +{pool_frag_tax:.0f})",
    )

    # Annotate the cumulative gap from allocated → driver_watermark per phase,
    # broken down by hop, just above each bar group.
    y_top = max(summary[p]["driver_watermark_mb"] for p in phases)
    for xi, p in enumerate(phases):
        s = summary[p]
        hops = [
            ("Δpeak",   s["peak_allocated_mb"] - s["allocated_mb"]),
            ("Δpool",   s["peak_reserved_mb"]  - s["peak_allocated_mb"]),
            ("Δfreed",  s["driver_freed_mb"]   - s["peak_reserved_mb"]),
            ("Δwm",     s["driver_watermark_mb"] - s["driver_freed_mb"]),
        ]
        text = "\n".join(f"{lbl}+{val:.0f}" for lbl, val in hops)
        ax.text(
            xi,
            y_top * 1.05,
            text,
            ha="center", va="bottom", fontsize=6, color="#444",
        )

    ax.set_title(title)
    ax.set_ylabel("MB (median over stable steps)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("after_", "") for p in phases],
                       rotation=20, ha="right")
    ax.set_ylim(0, y_top * 1.32)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.95)


def write_csv(tp1: dict[str, dict[str, float]],
              tp2: dict[str, dict[str, float]],
              out_path: Path) -> None:
    csv_keys = [k for k, _, _ in LAYERS] + [PROXY_KEY]
    fields = ["phase"]
    for tp in ("tp1", "tp2"):
        for key in csv_keys:
            fields.append(f"{tp}_{key}")
    phases = [p for p in PHASE_ORDER if p in tp1]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for p in phases:
            row = [p]
            for src in (tp1, tp2):
                for key in csv_keys:
                    row.append(f"{src[p][key]:.1f}")
            w.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tp1 = collect(args.watermark_tp1_dir, args.freed_tp1_dir, args.stable_from_step)
    tp2 = collect(args.watermark_tp2_dir, args.freed_tp2_dir, args.stable_from_step)

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    plot_panel(axes[0], tp1, "TP=1 — memory hierarchy per phase (allocated → driver)")
    plot_panel(axes[1], tp2, "TP=2 (rank0) — memory hierarchy per phase (allocated → driver)")
    fig.suptitle(
        "Per-phase PyTorch + driver memory layers — gap between layers shows "
        "where MB are 'lost' to allocator + driver",
        y=1.01,
    )
    fig.tight_layout()
    out_png = args.output_dir / "tp1_vs_tp2_memory_layers_combined.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    write_csv(tp1, tp2, args.output_dir / "tp1_vs_tp2_memory_layers_combined.csv")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
