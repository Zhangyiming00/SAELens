"""Stacked bar chart of per-rank vLLM memory for bf16 Llama-3.1-8B.

Config: mbt=8192, B=16, S=2048 (T=32768, chunked into 4 prefill steps).
Formulas come from results/memory_model/vllm/vllm_tp1_tp2_report.md
(validated to <1 MiB error in fp32; rescaled to bf16 here by bpe ratio).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GiB = 2**30

H = 4096
I_MLP = 14336  # noqa: N816 - intermediate_size
L = 32
N_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
PARAMS_TOTAL_BYTES = 8_030_261_248 * 2  # 8.03B params, bf16

BPE = 2  # bf16
B = 16
S = 2048
MBT = 8192
T = B * S
T_EFF = min(T, MBT)


def weight_gib(tp: int) -> float:
    return PARAMS_TOTAL_BYTES / tp / GiB


def kv_min_gib(tp: int) -> float:
    num_blocks = math.ceil(MBT / BLOCK_SIZE) + 1
    return L * num_blocks * 2 * BLOCK_SIZE * (KV_HEADS / tp) * HEAD_DIM * BPE / GiB


def runtime_residual_gib(tp: int) -> float:
    # Empirical one-shot CUDA / NCCL / cuBLAS workspace residual measured in
    # the validated fp32 report; dtype-independent (driver/handle state).
    return 0.239805 if tp == 1 else 0.247617


def tmp_gib(tp: int) -> float:
    coeff = 25 / 2 if tp == 1 else 29 / 4
    return coeff * T_EFF * H * BPE / GiB


def persistent_gib() -> float:
    return T * H * BPE / GiB


COMPONENTS = [
    ("Weights", weight_gib, "#4C72B0"),
    ("KV cache (min)", kv_min_gib, "#55A868"),
    ("Runtime residual\n(CUDA/NCCL/cuBLAS)", runtime_residual_gib, "#8172B2"),
    ("Forward workspace (tmp)", lambda tp: tmp_gib(tp), "#C44E52"),
    ("Hook capture (persistent)", lambda _tp: persistent_gib(), "#CCB974"),
]


def main() -> None:
    tps = [1, 2]
    bottoms = np.zeros(len(tps), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    for name, fn, color in COMPONENTS:
        values = np.array([fn(tp) for tp in tps], dtype=float)
        bars = ax.bar(
            [f"TP={tp}" for tp in tps],
            values,
            bottom=bottoms,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if value < 0.05:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + value / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 1.0 else "black",
            )
        bottoms += values

    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.25, f"total {total:.2f} GiB",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Per-rank memory (GiB)")
    ax.set_title(
        "vLLM per-rank memory — Llama-3.1-8B bf16\n"
        f"mbt={MBT}, B={B}, S={S} (T={T}, chunks={math.ceil(T/MBT)})",
        fontsize=11,
    )
    ax.set_ylim(0, max(bottoms) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    out = Path("results/memory_model/vllm/vllm_tp1_tp2_bf16_b16_s2048.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print("Per-rank components (GiB):")
    for tp in tps:
        print(f"  TP={tp}")
        for name, fn, _ in COMPONENTS:
            print(f"    {name.replace(chr(10), ' '):<40} {fn(tp):.3f}")
        total = sum(fn(tp) for _, fn, _ in COMPONENTS)
        print(f"    {'TOTAL':<40} {total:.3f}")


if __name__ == "__main__":
    main()
