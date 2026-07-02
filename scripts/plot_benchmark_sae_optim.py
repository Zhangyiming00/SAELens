"""Plot ``benchmark_sae_optim`` results.

Reads the JSON written by ``scripts.benchmark_sae_optim`` and produces two
figures:

- ``benchmark_sae_optim_step_time.png`` — median step time (ms) vs n_hooks,
  with one subplot per (d_sae, dtype) and one bar group per optim_mode.
- ``benchmark_sae_optim_peak_mem.png`` — peak GPU memory (MB) with the same
  layout.

Run as:

    python3 -m scripts.plot_benchmark_sae_optim --input bench.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OPTIM_ORDER = ("for_loop", "foreach", "fused")
OPTIM_COLORS = {
    "for_loop": "#888888",
    "foreach": "#1f77b4",
    "fused": "#d62728",
}


def _select(rows: list[dict], **filters) -> list[dict]:
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def _plot_metric(
    rows: list[dict],
    *,
    d_saes: list[int],
    dtypes: list[str],
    n_hooks_list: list[int],
    optim_modes: list[str],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    device_name: str,
) -> None:
    n_rows = len(d_saes)
    n_cols = len(dtypes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.4 * n_rows),
        squeeze=False,
        sharex=True,
    )

    bar_width = 0.8 / max(len(optim_modes), 1)
    x = np.arange(len(n_hooks_list), dtype=float)

    for i, d_sae in enumerate(d_saes):
        for j, dtype in enumerate(dtypes):
            ax = axes[i][j]
            for m_idx, mode in enumerate(optim_modes):
                values: list[float] = []
                for n_hooks in n_hooks_list:
                    sel = _select(
                        rows,
                        d_sae=d_sae,
                        dtype=dtype,
                        optim_mode=mode,
                        n_hooks=n_hooks,
                    )
                    if sel and sel[0].get("ok", False):
                        v = sel[0].get(metric, math.nan)
                    else:
                        v = math.nan
                    values.append(v)
                offsets = x + (m_idx - (len(optim_modes) - 1) / 2.0) * bar_width
                bars = ax.bar(
                    offsets,
                    values,
                    width=bar_width,
                    label=mode,
                    color=OPTIM_COLORS.get(mode, None),
                )
                for rect, v in zip(bars, values):
                    if not math.isnan(v):
                        ax.text(
                            rect.get_x() + rect.get_width() / 2.0,
                            v,
                            f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )
            ax.set_title(f"d_sae={d_sae}, dtype={dtype}")
            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in n_hooks_list])
            if i == n_rows - 1:
                ax.set_xlabel("n_hooks")
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(optim_modes),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(f"{title}  ({device_name})", y=1.04)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_sae_optim_results.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    rows: list[dict] = payload["results"]
    device_name = payload.get("device_name", "")

    d_saes = sorted({r["d_sae"] for r in rows})
    dtypes = sorted({r["dtype"] for r in rows}, reverse=True)  # bf16 first
    n_hooks_list = sorted({r["n_hooks"] for r in rows})
    optim_modes = [m for m in OPTIM_ORDER if any(r["optim_mode"] == m for r in rows)]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        metric="median_step_ms",
        ylabel="median step time (ms)",
        title="SAE training step time",
        output=args.out_dir / "benchmark_sae_optim_step_time.png",
        device_name=device_name,
    )

    _plot_metric(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        metric="peak_mem_mb",
        ylabel="peak GPU memory (MB)",
        title="SAE peak memory",
        output=args.out_dir / "benchmark_sae_optim_peak_mem.png",
        device_name=device_name,
    )


if __name__ == "__main__":
    main()
