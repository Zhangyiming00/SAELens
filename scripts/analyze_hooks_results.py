#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sae_lens.hooks_results_analysis import build_report


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze hooks benchmark results while preserving failed slots and "
            "normalizing throughput by a common token budget."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results/results_1.44_hooks",
        help="Results root containing saelens_runner_gpu_* folders.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints/checkpoint2",
        help="Checkpoint root used to backfill missing runner_cfg.json.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=32,
        help="Minimum number of timing steps required to consider a run complete.",
    )
    parser.add_argument(
        "--min-completion-ratio",
        type=float,
        default=0.98,
        help="Minimum final_samples/training_tokens required to consider a run complete.",
    )
    args = parser.parse_args()

    report = build_report(
        Path(args.results_dir),
        checkpoints_dir=Path(args.checkpoints_dir),
        min_steps=args.min_steps,
        min_completion_ratio=args.min_completion_ratio,
    )

    for hook_count in sorted(report.groups):
        group = report.groups[hook_count]
        print(f"\n=== {hook_count} hooks ===")
        print(
            "slot scenario              status steps final_samples elapsed_s "
            "norm_tok/s completion notes run_id"
        )
        for row in group.rows:
            status = "OK" if row.complete else "FAIL"
            notes = ",".join(row.notes) if row.notes else "-"
            print(
                f"{row.slot:>2}   "
                f"{row.scenario:<20} "
                f"{status:<5} "
                f"{row.steps:>5} "
                f"{row.final_samples:>13} "
                f"{row.elapsed_s:>8.3f} "
                f"{row.normalized_tokens_per_s:>9.1f} "
                f"{_format_ratio(row.local_completion_ratio):>10} "
                f"{notes:<20} "
                f"{row.run_id}"
            )

        best = group.best_complete
        if best is None:
            print("best_complete: none")
        else:
            print(
                "best_complete: "
                f"slot={best.slot} scenario={best.scenario} "
                f"norm_tok/s={best.normalized_tokens_per_s:.1f} run_id={best.run_id}"
            )


if __name__ == "__main__":
    main()
