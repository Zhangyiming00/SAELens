"""Plot v2 benchmark results — two separate figures.

Each figure is a grid of panels:

                  n_hooks=1     n_hooks=2     n_hooks=4
    d_sae=32768   panel          panel          panel
    d_sae=65536   panel          panel          panel

  * <prefix>_time.png — median step time (ms) only.
  * <prefix>_mem.png  — transient peak memory (MB) only.

Each panel: x-axis = optim_mode (for_loop / foreach / fused), and bf16/fp32
are paired bars, so all three modes for both dtypes can be compared at a
glance. Memory panels plot transient = peak − resident, where resident is
the empirical ``torch.cuda.memory_allocated()`` reading after warmup
(params + grads + Adam m/v + input batch). Forward autograd-saved tensors
and the per-hook reconstruction outputs only exist inside a step (freed
by backward), so they end up in the transient bucket — which is why the
transient grows roughly linearly in n_hooks for every optim mode.

Run as:

    python3 -m scripts.plot_benchmark_sae_optim_v2 \\
        --input results/memory_model_v2/optim_v1/results.json \\
        --out-dir results/memory_model_v2/optim_v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OPTIM_ORDER = ("for_loop", "foreach", "fused")
DTYPE_ORDER = ("bf16", "fp32")
DTYPE_COLORS = {"bf16": "#2ca02c", "fp32": "#9467bd"}


def _select(rows: list[dict], **filters) -> dict | None:
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            return r
    return None


def _val(row: dict | None, metric: str) -> float:
    if row is None or not row.get("ok", False):
        return math.nan
    v = row.get(metric, math.nan)
    return float(v) if v is not None else math.nan


def _draw_paired_bars(
    ax: plt.Axes,
    rows: list[dict],
    *,
    d_sae: int,
    n_hooks: int,
    optim_modes: list[str],
    dtypes: list[str],
    metric: str,
    label_fmt: str,
) -> None:
    bar_width = 0.8 / max(len(dtypes), 1)
    x = np.arange(len(optim_modes), dtype=float)
    for d_idx, dtype in enumerate(dtypes):
        values = [
            _val(
                _select(
                    rows,
                    d_sae=d_sae,
                    n_hooks=n_hooks,
                    dtype=dtype,
                    optim_mode=mode,
                ),
                metric,
            )
            for mode in optim_modes
        ]
        offsets = x + (d_idx - (len(dtypes) - 1) / 2.0) * bar_width
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=dtype,
            color=DTYPE_COLORS.get(dtype),
        )
        for rect, v in zip(bars, values):
            if not math.isnan(v):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    v,
                    label_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(optim_modes)
    ax.grid(True, axis="y", alpha=0.3)


def _draw_memory_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    d_sae: int,
    n_hooks: int,
    optim_modes: list[str],
    dtypes: list[str],
) -> None:
    bar_width = 0.8 / max(len(dtypes), 1)
    x = np.arange(len(optim_modes), dtype=float)
    bars_max = 0.0
    for d_idx, dtype in enumerate(dtypes):
        transients = [
            _val(
                _select(
                    rows,
                    d_sae=d_sae,
                    n_hooks=n_hooks,
                    dtype=dtype,
                    optim_mode=mode,
                ),
                "transient_mem_mb",
            )
            for mode in optim_modes
        ]
        offsets = x + (d_idx - (len(dtypes) - 1) / 2.0) * bar_width
        bars = ax.bar(
            offsets,
            transients,
            width=bar_width,
            label=dtype,
            color=DTYPE_COLORS.get(dtype),
        )
        for rect, v in zip(bars, transients):
            if not math.isnan(v):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    v,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
                bars_max = max(bars_max, v)

    ax.set_xticks(x)
    ax.set_xticklabels(optim_modes)
    ax.grid(True, axis="y", alpha=0.3)
    if bars_max > 0:
        ax.set_ylim(0, bars_max * 1.18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/memory_model_v2/optim_v1/results.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/memory_model_v2/optim_v1"),
    )
    parser.add_argument(
        "--prefix",
        default="benchmark_sae_optim_v2",
    )
    return parser.parse_args()


def _plot_grid(
    rows: list[dict],
    *,
    d_saes: list[int],
    n_hooks_list: list[int],
    optim_modes: list[str],
    dtypes: list[str],
    metric: str,
    label_fmt: str,
    ylabel: str,
    title: str,
    out_path: Path,
    is_memory: bool,
) -> None:
    """One figure: rows = d_sae, cols = n_hooks, one metric only."""
    n_rows = len(d_saes)
    n_cols = len(n_hooks_list)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    # Share y across columns within each row so panels in the same row are
    # directly comparable.
    for r in range(n_rows):
        for c in range(1, n_cols):
            axes[r][c].sharey(axes[r][0])

    for d_idx, d_sae in enumerate(d_saes):
        for c, n_hooks in enumerate(n_hooks_list):
            ax = axes[d_idx][c]
            if is_memory:
                _draw_memory_panel(
                    ax,
                    rows,
                    d_sae=d_sae,
                    n_hooks=n_hooks,
                    optim_modes=optim_modes,
                    dtypes=dtypes,
                )
            else:
                _draw_paired_bars(
                    ax,
                    rows,
                    d_sae=d_sae,
                    n_hooks=n_hooks,
                    optim_modes=optim_modes,
                    dtypes=dtypes,
                    metric=metric,
                    label_fmt=label_fmt,
                )
            ax.set_title(f"d_sae={d_sae}, n_hooks={n_hooks}")
            if c == 0:
                ax.set_ylabel(f"d_sae={d_sae}\n\n{ylabel}")
            ax.set_xlabel("optim mode")
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    rows: list[dict] = payload["results"]
    device_name = payload.get("device_name", "")

    d_saes = sorted({r["d_sae"] for r in rows})
    n_hooks_list = sorted({r["n_hooks"] for r in rows})
    optim_modes = [m for m in OPTIM_ORDER if any(r["optim_mode"] == m for r in rows)]
    dtypes = [d for d in DTYPE_ORDER if any(r["dtype"] == d for r in rows)]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot_grid(
        rows,
        d_saes=d_saes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        dtypes=dtypes,
        metric="median_step_ms",
        label_fmt="{:.0f}",
        ylabel="median step time (ms)",
        title=(
            f"SAE training step — median step time  —  {device_name}\n"
            "bars: bf16 vs fp32 paired   |   lower is faster"
        ),
        out_path=args.out_dir / f"{args.prefix}_time.png",
        is_memory=False,
    )

    _plot_grid(
        rows,
        d_saes=d_saes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        dtypes=dtypes,
        metric="transient_mem_mb",
        label_fmt="{:.0f}",
        ylabel="transient peak mem (MB)\n= peak − resident",
        title=(
            f"SAE training step — transient peak memory  —  {device_name}\n"
            "bars: bf16 vs fp32 paired   |   "
            "transient = peak − resident,  resident = params + grads + m + v + input"
        ),
        out_path=args.out_dir / f"{args.prefix}_mem.png",
        is_memory=True,
    )


if __name__ == "__main__":
    main()
