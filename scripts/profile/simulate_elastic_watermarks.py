"""CPU-only fluid model of a calibrated DP2/DP3 SHM watermark cycle.

Input rates are measured online (including routing/compute overlap), not the
native SAE kernel-only capacity. The model accounts for the producer restart,
expansion preparation and the two optimizer-state migration pauses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate(profile, cycles=4):
    capacity = profile["capacity_tokens"]
    low = profile["low"] * capacity
    high = profile["high"] * capacity
    p2, p3 = profile["production_dp2"], profile["production_dp3"]
    c2, c3 = profile["consumption_dp2"], profile["consumption_dp3"]
    if not (p2 > c2 and c3 > p3):
        raise ValueError("Natural cycling requires P2 > C2 and C3 > P3")
    if not 0 < low < high < capacity:
        raise ValueError("Require 0 < low < high < 1")
    hysteresis = profile["stable_delay_s"]
    t = 0.0
    # Begin immediately after a low-water request's shrink migration.
    q = max(0, low-(c3-p3)*hysteresis) + p3*profile["shrink_cutover_s"]
    q = min(capacity, q)
    events = []

    def stage(name, seconds, production, consumption, dp):
        nonlocal q, t
        unbounded = q + (production-consumption)*seconds
        events.append(dict(stage=name, dp=dp, start_s=t, end_s=t+seconds,
                           start_fill=q/capacity, end_fill=min(capacity,max(0,unbounded))/capacity,
                           producer_tps=production, consumer_tps=consumption,
                           hits_empty=unbounded < 0, hits_full=unbounded > capacity))
        q = min(capacity, max(0, unbounded))
        t += seconds

    decisions = []
    for cycle in range(cycles):
        stage("producer_reload", profile["producer_reload_s"], p3, c2, 2)
        fill_time = max(0, (high-q)/(p2-c2)) + hysteresis
        fill_time = max(fill_time, profile["cooldown_s"]-profile["producer_reload_s"])
        stage("dp2_fill", fill_time, p2, c2, 2)
        decisions.append(dict(cycle=cycle, target_dp=3, timestamp_s=t, fill=q/capacity))
        stage("expand_prepare", profile["expand_prepare_s"], p3, c2, 2)
        stage("expand_cutover", profile["expand_cutover_s"], p3, 0, 2)
        drain_time = max(0, (q-low)/(c3-p3)) + hysteresis
        stage("dp3_drain", max(drain_time,profile["cooldown_s"]), p3, c3, 3)
        decisions.append(dict(cycle=cycle, target_dp=2, timestamp_s=t, fill=q/capacity))
        stage("shrink_cutover", profile["shrink_cutover_s"], p3, 0, 3)
    intervals = [dict(source_target=a["target_dp"], next_target=b["target_dp"],
                      seconds=b["timestamp_s"]-a["timestamp_s"])
                 for a,b in zip(decisions,decisions[1:])]
    return dict(profile=profile, events=events, decisions=decisions, intervals=intervals,
                duration_s=t, hardware_scope="Only the measured online configuration",
                caveat="Constant-rate approximation; startup/EOF, chunk quantization and changing AuxK work are excluded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    result = simulate(json.loads(args.profile.read_text()), args.cycles)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
