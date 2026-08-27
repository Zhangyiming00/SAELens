#!/usr/bin/env python3
"""Map vLLM producer cycles to SAE step numbers for step-window profiling.

In co-located mode vLLM runs inside ``next(data_provider)``, so a producer cycle
is one data fetch that invokes vLLM: the mixing buffer pulls from the source
until it can serve, then serves some SAE steps from that pull without touching
vLLM again. A cycle is identified by the SAE step whose fetch paid for it.

The cadence per (batch_tokens, sae_dp_size) is MEASURED from short calibration
runs (steps with vllm_step_time_s > 0 in timing_history.jsonl) rather than
derived from the mixing-buffer arithmetic. A pure replay of ``mixing_buffer``
matched 5 of the 6 configs but predicted 1,2,3,5,6,7,9 for
(3072, dp=2) where vLLM in fact runs on every step, so the measured table is
authoritative. Re-measure with --calibrate when the data path changes.

Given `--skip-cycles 2 --cycles-per-window 2 --window-count 2`, profiling starts
at the 3rd cycle's step and each window encloses two cycles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured vLLM-invoking SAE steps, keyed by (batch_tokens, sae_dp_size).
# Sources: results/step_window_sweep_20 timing_history.jsonl for every config,
# plus a 30-step run for (3072, 2). All other sweep knobs at their defaults
# (context 2048, store_batch_size_prompts 1, n_batches_in_buffer 2,
# activations_mixing_fraction 0.5). Re-verified unchanged on idle GPUs against
# results/nsys_gap_diagnosis for (768, 1) and (768, 2).
#
# This table only places the windows, i.e. picks which steps to profile. It is
# not used to attribute time: scripts/build_vllm_sae_breakdown.py counts each
# run's own vLLM-invoking steps from its timing_history.jsonl.
MEASURED_CYCLES: dict[tuple[int, int], list[int]] = {
    (768, 1): [1, 3, 6, 9, 11, 14, 17, 19, 22, 25, 27, 30],
    (768, 2): [1, 6, 11, 17, 22, 27, 33, 38, 43, 49, 54, 59],
    (3072, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (3072, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (4096, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (4096, 2): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}


def measured_cycles(batch_tokens: int, sae_dp_size: int) -> list[int]:
    key = (batch_tokens, sae_dp_size)
    if key not in MEASURED_CYCLES:
        raise KeyError(
            f"no measured vLLM cadence for batch_tokens={batch_tokens} "
            f"sae_dp_size={sae_dp_size}; run with --calibrate on a real run first"
        )
    return MEASURED_CYCLES[key]


def cycles_from_timing(timing_path: Path) -> list[int]:
    """Extract the vLLM-invoking steps from a run's timing_history.jsonl."""
    rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line]
    return [r["step"] for r in rows if r.get("vllm_step_time_s", 0.0) > 0.0]


def plan(
    *,
    batch_tokens: int,
    sae_dp_size: int,
    skip_cycles: int = 2,
    cycles_per_window: int = 2,
    window_count: int = 2,
) -> dict:
    cycles = measured_cycles(batch_tokens, sae_dp_size)
    needed = skip_cycles + cycles_per_window * window_count
    if len(cycles) <= needed:
        raise ValueError(f"only {len(cycles)} cycles known, need > {needed}")

    start_step = cycles[skip_cycles]
    # Window width = distance to the cycle `cycles_per_window` later, so each
    # window opens on a vLLM step and closes just before the cycle two later.
    window_steps = cycles[skip_cycles + cycles_per_window] - start_step
    enclosed = [
        [
            c
            for c in cycles
            if start_step + i * window_steps
            <= c
            <= start_step + (i + 1) * window_steps - 1
        ]
        for i in range(window_count)
    ]
    last_step = start_step + window_steps * window_count - 1
    # One spare step so training does not end exactly as the last window closes.
    total_steps = last_step + 1
    return {
        "batch_tokens": batch_tokens,
        "sae_dp_size": sae_dp_size,
        "local_batch_tokens": batch_tokens // sae_dp_size,
        "vllm_cycle_steps": cycles,
        "start_step": start_step,
        "window_steps": window_steps,
        "window_count": window_count,
        "cycles_enclosed_per_window": enclosed,
        "last_profiled_step": last_step,
        "total_steps": total_steps,
        "training_tokens": total_steps * batch_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-tokens", type=int)
    p.add_argument("--sae-dp-size", type=int, default=1)
    p.add_argument("--skip-cycles", type=int, default=2)
    p.add_argument("--cycles-per-window", type=int, default=2)
    p.add_argument("--window-count", type=int, default=2)
    p.add_argument(
        "--calibrate",
        type=Path,
        default=None,
        help="Print the vLLM-invoking steps of a run's timing_history.jsonl",
    )
    args = p.parse_args()

    if args.calibrate is not None:
        print(json.dumps(cycles_from_timing(args.calibrate)))
        return

    if args.batch_tokens is None:
        p.error("--batch-tokens is required unless --calibrate is given")
    print(
        json.dumps(
            plan(
                batch_tokens=args.batch_tokens,
                sae_dp_size=args.sae_dp_size,
                skip_cycles=args.skip_cycles,
                cycles_per_window=args.cycles_per_window,
                window_count=args.window_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
