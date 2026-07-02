"""Detailed plots for ``benchmark_sae_optim`` results.

In addition to the basic facets in ``plot_benchmark_sae_optim``, this script
produces:

- ``..._scaling_lines.png``    line plots of step time / peak mem vs n_hooks
                               on linear and log axes for each (d_sae, dtype, mode).
- ``..._per_hook.png``         per-hook (amortized) step time and per-hook peak mem,
                               useful for spotting nonlinearity wrt n_hooks.
- ``..._relative.png``         speedup and memory ratio of foreach/fused vs for_loop.
- ``..._dtype_compare.png``    bf16 vs fp32 paired bars (for each n_hooks/d_sae/mode).
- ``..._heatmap.png``          time/memory heatmap over (n_hooks × mode) per (d_sae, dtype).

Run as:

    python3 -m scripts.plot_benchmark_sae_optim_detailed \\
        --input benchmark_sae_optim_results.json --out-dir .
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
OPTIM_MARKERS = {"for_loop": "o", "foreach": "s", "fused": "D"}

DTYPE_COLORS = {"bf16": "#2ca02c", "fp32": "#9467bd"}


def _select(rows: list[dict], **filters) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]


def _value(rows: list[dict], metric: str, **filters) -> float:
    sel = _select(rows, **filters)
    if sel and sel[0].get("ok", False):
        v = sel[0].get(metric, math.nan)
        return float(v) if v is not None else math.nan
    return math.nan


# ---------------------------------------------------------------------------
# Plot 1: scaling lines (step time / peak mem vs n_hooks)
# ---------------------------------------------------------------------------


def plot_scaling_lines(
    rows: list[dict],
    *,
    d_saes: list[int],
    dtypes: list[str],
    n_hooks_list: list[int],
    optim_modes: list[str],
    output: Path,
    device_name: str,
) -> None:
    n_rows = len(d_saes)
    n_cols = len(dtypes)
    fig, axes = plt.subplots(
        2 * n_rows, 2 * n_cols, figsize=(5.0 * n_cols * 2, 3.4 * n_rows * 2)
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(2 * n_rows, 2 * n_cols)

    for i, d_sae in enumerate(d_saes):
        for j, dtype in enumerate(dtypes):
            for col_off, ylabel, metric in (
                (0, "median step time (ms)", "median_step_ms"),
                (1, "peak GPU memory (MB)", "peak_mem_mb"),
            ):
                ax_lin = axes[i * 2 + 0][j * 2 + col_off]
                ax_log = axes[i * 2 + 1][j * 2 + col_off]
                for mode in optim_modes:
                    ys = [
                        _value(
                            rows,
                            metric,
                            d_sae=d_sae,
                            dtype=dtype,
                            optim_mode=mode,
                            n_hooks=h,
                        )
                        for h in n_hooks_list
                    ]
                    color = OPTIM_COLORS.get(mode)
                    marker = OPTIM_MARKERS.get(mode, "o")
                    ax_lin.plot(
                        n_hooks_list, ys, marker=marker, color=color, label=mode
                    )
                    ax_log.plot(
                        n_hooks_list, ys, marker=marker, color=color, label=mode
                    )
                title = f"d_sae={d_sae}, dtype={dtype}"
                ax_lin.set_title(f"{title}  (linear)")
                ax_log.set_title(f"{title}  (log-log)")
                for ax in (ax_lin, ax_log):
                    ax.set_xlabel("n_hooks")
                    ax.set_ylabel(ylabel)
                    ax.set_xticks(n_hooks_list)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
                ax_log.set_xscale("log")
                ax_log.set_yscale("log")
                ax_log.set_xticks(n_hooks_list)
                ax_log.set_xticklabels([str(h) for h in n_hooks_list])

    fig.suptitle(
        f"SAE training step: scaling vs n_hooks  ({device_name})", y=1.0
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Plot 2: per-hook amortized step time / peak mem
# ---------------------------------------------------------------------------


def plot_per_hook(
    rows: list[dict],
    *,
    d_saes: list[int],
    dtypes: list[str],
    n_hooks_list: list[int],
    optim_modes: list[str],
    output: Path,
    device_name: str,
) -> None:
    fig, axes = plt.subplots(
        len(d_saes),
        2 * len(dtypes),
        figsize=(5.0 * len(dtypes) * 2, 3.4 * len(d_saes)),
        squeeze=False,
    )
    bar_width = 0.8 / max(len(optim_modes), 1)
    x = np.arange(len(n_hooks_list), dtype=float)

    for i, d_sae in enumerate(d_saes):
        for j, dtype in enumerate(dtypes):
            for col_off, ylabel, metric in (
                (0, "step time per hook (ms)", "median_step_ms"),
                (1, "peak mem per hook (MB)", "peak_mem_mb"),
            ):
                ax = axes[i][j * 2 + col_off]
                for m_idx, mode in enumerate(optim_modes):
                    values = []
                    for h in n_hooks_list:
                        v = _value(
                            rows,
                            metric,
                            d_sae=d_sae,
                            dtype=dtype,
                            optim_mode=mode,
                            n_hooks=h,
                        )
                        values.append(v / h if not math.isnan(v) and h > 0 else math.nan)
                    offsets = x + (m_idx - (len(optim_modes) - 1) / 2.0) * bar_width
                    bars = ax.bar(
                        offsets,
                        values,
                        width=bar_width,
                        label=mode,
                        color=OPTIM_COLORS.get(mode),
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
                ax.set_xticks(x)
                ax.set_xticklabels([str(n) for n in n_hooks_list])
                ax.set_xlabel("n_hooks")
                ax.set_ylabel(ylabel)
                ax.set_title(f"d_sae={d_sae}, dtype={dtype}")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(fontsize=8)
    fig.suptitle(
        f"SAE training: per-hook (amortized) cost  ({device_name})", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Plot 3: relative speedup / mem-ratio vs for_loop baseline
# ---------------------------------------------------------------------------


def plot_relative(
    rows: list[dict],
    *,
    d_saes: list[int],
    dtypes: list[str],
    n_hooks_list: list[int],
    output: Path,
    device_name: str,
) -> None:
    other_modes = [m for m in ("foreach", "fused") if m in OPTIM_ORDER]
    fig, axes = plt.subplots(
        len(d_saes),
        2 * len(dtypes),
        figsize=(5.0 * len(dtypes) * 2, 3.2 * len(d_saes)),
        squeeze=False,
    )
    bar_width = 0.4
    x = np.arange(len(n_hooks_list), dtype=float)

    for i, d_sae in enumerate(d_saes):
        for j, dtype in enumerate(dtypes):
            for col_off, ylabel, metric, baseline_better in (
                (0, "speedup vs for_loop", "median_step_ms", True),
                (1, "mem ratio vs for_loop", "peak_mem_mb", False),
            ):
                ax = axes[i][j * 2 + col_off]
                for m_idx, mode in enumerate(other_modes):
                    ratios = []
                    for h in n_hooks_list:
                        base = _value(
                            rows,
                            metric,
                            d_sae=d_sae,
                            dtype=dtype,
                            optim_mode="for_loop",
                            n_hooks=h,
                        )
                        cur = _value(
                            rows,
                            metric,
                            d_sae=d_sae,
                            dtype=dtype,
                            optim_mode=mode,
                            n_hooks=h,
                        )
                        if math.isnan(base) or math.isnan(cur) or cur == 0:
                            ratios.append(math.nan)
                        elif baseline_better:
                            ratios.append(base / cur)  # >1 == faster
                        else:
                            ratios.append(cur / base)  # >1 == more mem
                    offsets = x + (m_idx - (len(other_modes) - 1) / 2.0) * bar_width
                    bars = ax.bar(
                        offsets,
                        ratios,
                        width=bar_width,
                        label=mode,
                        color=OPTIM_COLORS.get(mode),
                    )
                    for rect, v in zip(bars, ratios):
                        if not math.isnan(v):
                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                v,
                                f"{v:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                            )
                ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
                ax.set_xticks(x)
                ax.set_xticklabels([str(n) for n in n_hooks_list])
                ax.set_xlabel("n_hooks")
                ax.set_ylabel(ylabel)
                ax.set_title(f"d_sae={d_sae}, dtype={dtype}")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(fontsize=8)
    fig.suptitle(
        f"foreach / fused vs for_loop  ({device_name})  "
        "— left: speedup (>1 better),  right: peak-mem ratio (>1 worse)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Plot 4: bf16 vs fp32 paired bars
# ---------------------------------------------------------------------------


def plot_dtype_compare(
    rows: list[dict],
    *,
    d_saes: list[int],
    dtypes: list[str],
    n_hooks_list: list[int],
    optim_modes: list[str],
    output: Path,
    device_name: str,
) -> None:
    fig, axes = plt.subplots(
        len(d_saes),
        2 * len(optim_modes),
        figsize=(4.0 * len(optim_modes) * 2, 3.2 * len(d_saes)),
        squeeze=False,
    )
    bar_width = 0.4
    x = np.arange(len(n_hooks_list), dtype=float)

    for i, d_sae in enumerate(d_saes):
        for m_idx, mode in enumerate(optim_modes):
            for col_off, ylabel, metric in (
                (0, "median step time (ms)", "median_step_ms"),
                (1, "peak GPU memory (MB)", "peak_mem_mb"),
            ):
                ax = axes[i][m_idx * 2 + col_off]
                for d_idx, dtype in enumerate(dtypes):
                    values = [
                        _value(
                            rows,
                            metric,
                            d_sae=d_sae,
                            dtype=dtype,
                            optim_mode=mode,
                            n_hooks=h,
                        )
                        for h in n_hooks_list
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
                                f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                            )
                ax.set_xticks(x)
                ax.set_xticklabels([str(n) for n in n_hooks_list])
                ax.set_xlabel("n_hooks")
                ax.set_ylabel(ylabel)
                ax.set_title(f"d_sae={d_sae}, mode={mode}")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend(fontsize=8)
    fig.suptitle(f"bf16 vs fp32  ({device_name})", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Plot 5: heatmaps (n_hooks × mode) per (d_sae, dtype)
# ---------------------------------------------------------------------------


def plot_heatmap(
    rows: list[dict],
    *,
    d_saes: list[int],
    dtypes: list[str],
    n_hooks_list: list[int],
    optim_modes: list[str],
    output: Path,
    device_name: str,
) -> None:
    n_panels = len(d_saes) * len(dtypes)
    fig, axes = plt.subplots(
        n_panels,
        2,
        figsize=(7.5, 2.6 * n_panels),
        squeeze=False,
    )
    panel = 0
    for d_sae in d_saes:
        for dtype in dtypes:
            for col, metric, label, cmap in (
                (0, "median_step_ms", "step time (ms)", "viridis"),
                (1, "peak_mem_mb", "peak mem (MB)", "magma"),
            ):
                ax = axes[panel][col]
                grid = np.full((len(optim_modes), len(n_hooks_list)), math.nan)
                for r, mode in enumerate(optim_modes):
                    for c, h in enumerate(n_hooks_list):
                        grid[r, c] = _value(
                            rows,
                            metric,
                            d_sae=d_sae,
                            dtype=dtype,
                            optim_mode=mode,
                            n_hooks=h,
                        )
                im = ax.imshow(grid, aspect="auto", cmap=cmap)
                ax.set_xticks(range(len(n_hooks_list)))
                ax.set_xticklabels([str(n) for n in n_hooks_list])
                ax.set_yticks(range(len(optim_modes)))
                ax.set_yticklabels(optim_modes)
                ax.set_xlabel("n_hooks")
                ax.set_title(f"d_sae={d_sae}, dtype={dtype} — {label}")
                for r in range(grid.shape[0]):
                    for c in range(grid.shape[1]):
                        v = grid[r, c]
                        if not math.isnan(v):
                            ax.text(
                                c,
                                r,
                                f"{v:.0f}",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                            )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            panel += 1
    fig.suptitle(f"Step time / peak memory heatmaps  ({device_name})", y=1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--prefix",
        default="benchmark_sae_optim_detailed",
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
    p = args.prefix

    plot_scaling_lines(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        output=args.out_dir / f"{p}_scaling_lines.png",
        device_name=device_name,
    )
    plot_per_hook(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        output=args.out_dir / f"{p}_per_hook.png",
        device_name=device_name,
    )
    plot_relative(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        output=args.out_dir / f"{p}_relative.png",
        device_name=device_name,
    )
    plot_dtype_compare(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        output=args.out_dir / f"{p}_dtype_compare.png",
        device_name=device_name,
    )
    plot_heatmap(
        rows,
        d_saes=d_saes,
        dtypes=dtypes,
        n_hooks_list=n_hooks_list,
        optim_modes=optim_modes,
        output=args.out_dir / f"{p}_heatmap.png",
        device_name=device_name,
    )


if __name__ == "__main__":
    main()
