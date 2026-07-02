"""Plot the three Adam optimizer modes' memory behavior from the phases grid.

Reads ``results/memory_model/optim_phases/results.json`` and renders:

  1. Isolated optimizer transient vs n_hooks (one line per mode) at a fixed
     (d_in, d_sae, dtype) — shows foreach scaling with hooks while for_loop is
     flat at 3x biggest and fused at 0.
  2. Step peak by mode across a config sweep (grouped bars) — shows which mode
     is the lowest ceiling and the gap.
  3. Phase-peak stack for one config per mode — forward/backward/optimizer.

Run as::

    python3 -m scripts.plot_sae_optim_phases \\
        --input results/memory_model/optim_phases/results.json \\
        --out-dir results/memory_model/optim_phases/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODE_COLOR = {"fused": "#4e79a7", "foreach": "#e15759", "for_loop": "#59a14f"}
MODE_ORDER = ["fused", "foreach", "for_loop"]


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())["results"]


def plot_transient_vs_hooks(rows, out_dir, d_in, d_sae, dtype):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in MODE_ORDER:
        pts = sorted(
            (r["n_hooks"], r["iso_transient_mb"])
            for r in rows
            if r["optim_mode"] == mode
            and r["d_in"] == d_in
            and r["d_sae"] == d_sae
            and r["dtype"] == dtype
            and r["ok"]
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=MODE_COLOR[mode], label=mode, lw=2, ms=7)
    ax.set_xlabel("n_hooks (independent SAEs sharing one Adam)")
    ax.set_ylabel("isolated optimizer.step transient (MiB)")
    ax.set_title(f"Adam step transient vs hooks  (d_in={d_in}, d_sae={d_sae}, {dtype})")
    ax.legend(title="optim_mode")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"transient_vs_hooks_din{d_in}_d{d_sae}_{dtype}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def plot_step_peak_bars(rows, out_dir, dtype):
    """Grouped bars: step peak by mode across (d_sae, n_hooks) at fixed d_in."""
    d_in = 4096
    keys = sorted(
        {
            (r["d_sae"], r["n_hooks"])
            for r in rows
            if r["d_in"] == d_in and r["dtype"] == dtype and r["ok"]
        }
    )
    by = defaultdict(dict)
    for r in rows:
        if r["d_in"] == d_in and r["dtype"] == dtype and r["ok"]:
            by[(r["d_sae"], r["n_hooks"])][r["optim_mode"]] = r["step_peak_mb"]
    x = np.arange(len(keys), dtype=float)
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, mode in enumerate(MODE_ORDER):
        ys = [by[k].get(mode, np.nan) for k in keys]
        ax.bar(x + (i - 1) * w, ys, w, color=MODE_COLOR[mode], label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d_sae={ds}\nH={h}" for ds, h in keys], fontsize=8)
    ax.set_ylabel("full step peak allocated (MiB)")
    ax.set_title(f"Step memory ceiling by optim_mode  (d_in={d_in}, {dtype})")
    ax.legend(title="optim_mode")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"step_peak_bars_din{d_in}_{dtype}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def plot_phase_peaks(rows, out_dir, d_in, d_sae, n_hooks, dtype):
    """For one config: each mode's forward/backward/optimizer phase peak."""
    sel = {
        r["optim_mode"]: r
        for r in rows
        if r["d_in"] == d_in
        and r["d_sae"] == d_sae
        and r["n_hooks"] == n_hooks
        and r["dtype"] == dtype
        and r["ok"]
    }
    if not sel:
        return None
    phases = [
        "forward_phase_peak_mb",
        "backward_phase_peak_mb",
        "optimizer_phase_peak_mb",
    ]
    labels = ["forward", "backward", "optimizer"]
    x = np.arange(len(MODE_ORDER), dtype=float)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for j, (ph, lab) in enumerate(zip(phases, labels)):
        ys = [sel[m][ph] if m in sel else np.nan for m in MODE_ORDER]
        ax.bar(x + (j - 1) * w, ys, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(MODE_ORDER)
    ax.set_ylabel("phase peak allocated (MiB)")
    ax.set_title(f"Per-phase peak  (d_in={d_in}, d_sae={d_sae}, H={n_hooks}, {dtype})")
    ax.legend(title="phase")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"phase_peaks_din{d_in}_d{d_sae}_h{n_hooks}_{dtype}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/memory_model/optim_phases/results.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/memory_model/optim_phases/figures"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(args.input)
    written = [
        plot_transient_vs_hooks(rows, args.out_dir, 4096, 65536, "fp32"),
        plot_transient_vs_hooks(rows, args.out_dir, 4096, 65536, "bf16"),
        plot_step_peak_bars(rows, args.out_dir, "fp32"),
        plot_step_peak_bars(rows, args.out_dir, "bf16"),
        plot_phase_peaks(rows, args.out_dir, 4096, 65536, 4, "fp32"),
        plot_phase_peaks(rows, args.out_dir, 4096, 65536, 1, "fp32"),
    ]
    for p in written:
        if p:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
