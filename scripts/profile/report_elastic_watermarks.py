"""Summarize real auto-controller runs; rates count logical tokens per hook.

The native time model excludes SHM routing. This report keeps that prediction
separate from the measured online rate and its fluid-buffer projection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as stats


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def median(values):
    values = list(values)
    return stats.median(values) if values else None


def summarize(root):
    options = json.loads((root / "command.json").read_text())["options"]
    low, high = options.get("low_watermark", .25), options.get("high_watermark", .70)
    predictions = json.loads((root / "predictions.json").read_text())
    result = json.loads((root / "result.json").read_text())
    control = read_rows(root / "controller.jsonl")
    observations = {int(p.stem.removeprefix("observations_rank")): read_rows(p)
                    for p in root.glob("observations_rank*.jsonl")}
    samples = [r for r in control if r["event"] == "sample"]
    decisions = [r for r in control if r["event"] == "decision"]
    source = observations[2]
    steps = [r for r in source if r["event"] == "step"]
    update_tokens = options["microbatch"] * options["ga"]
    epochs = []
    for epoch in sorted({r["epoch"] for r in samples}):
        active = [r for r in samples if r["epoch"] == epoch and r["phase"] == "active"]
        if not active:
            continue
        # Discard startup/cutover and partial rate windows, plus buffer backpressure.
        stable = [r for r in active if r["timestamp"] >= active[0]["timestamp"] + 10
                  and r.get("window_seconds", 0) >= 3.9 and .05 < r["fill_ratio"] < .95]
        p = median(r["vllm_tokens_per_s"] for r in stable)
        c = median(r["sae_tokens_per_s"] for r in stable)
        decision = next((r for r in decisions if r["epoch"] == epoch), None)
        epoch_steps = [r for r in steps if r["epoch"] == epoch
                       and r["timestamp"] >= active[0]["timestamp"] + 10]
        epochs.append(dict(epoch=epoch, dp=active[0]["active_sae_dp"], samples=len(stable),
                           producer_tps=p, consumer_tps=c,
                           net_tps=p-c if p is not None else None,
                           online_cuda_window_ms=median(r["cuda_ms"] for r in epoch_steps),
                           start_unix=active[0]["timestamp"],
                           residence_until_request_s=(decision["timestamp"]-active[0]["timestamp"]) if decision else None,
                           start_fill=active[0]["fill_ratio"], end_fill=active[-1]["fill_ratio"]))
    switches = []
    for decision in decisions:
        epoch = decision["epoch"] + 1
        begin = next((r for r in source if r["event"] == "switch_begin" and r["epoch"] == epoch), None)
        end = next((r for r in source if r["event"] == "switch_end" and r["epoch"] == epoch), None)
        switches.append(dict(epoch=epoch, target_dp=decision["decision_target_sae_dp"],
                             fill=decision["fill_ratio"], reason=decision["reason"],
                             producer_tps=decision.get("vllm_tokens_per_s"),
                             consumer_tps=decision.get("sae_tokens_per_s"),
                             decision_unix=decision["timestamp"],
                             cutover_s=end["duration_s"] if end else None,
                             request_to_restored_s=end["timestamp"]-decision["timestamp"] if end else None,
                             progress=end["progress"] if end else None,
                             dead_ages_preserved=end.get("dead_ages_preserved") if end else None,
                             progress_preserved=bool(begin and end and begin["progress"] == end["progress"])))
    memory = []
    for dp in [2, 3]:
        for rank, rows in observations.items():
            subset = [r for r in rows if r["event"] == "step" and r["dp"] == dp]
            if subset:
                memory.append(dict(dp=dp, rank=rank, updates=len(subset),
                                   peak_allocated_gib=max(r["peak_allocated"] for r in subset)/2**30,
                                   peak_reserved_gib=max(r["reserved"] for r in subset)/2**30))
    losses = [r for r in source if r["event"] == "loss"]
    health = dict(updates=len(steps), final_tokens=steps[-1]["tokens"] if steps else 0,
                  steps_contiguous=[r["step"] for r in steps] == list(range(1, options["updates"]+1)),
                  exact_global_tokens=all(r["tokens"] == r["step"]*update_tokens for r in steps),
                  exact_micro_progress=all(r["micro_index"] == r["step"]*options["ga"] for r in steps),
                  full_gradient_windows=all(r["microbatches"] == options["ga"] for r in steps),
                  loss_observations=len(losses),
                  first_loss_mean=stats.mean(losses[0]["losses"].values()) if losses else None,
                  last_loss_mean=stats.mean(losses[-1]["losses"].values()) if losses else None)
    # Calibrate each flow direction on the first residence interval and compare
    # its sign/rate against the next interval of the same topology.
    flows = []
    capacity = options["chunks"] * options["chunk_tokens"]
    for dp in [2, 3]:
        matching = [e for e in epochs if e["dp"] == dp and e["samples"] >= 4]
        if not matching:
            continue
        first = matching[0]
        flows.append(dict(dp=dp, calibration_epoch=first["epoch"],
                          producer_tps=first["producer_tps"], consumer_tps=first["consumer_tps"],
                          net_tps=first["net_tps"],
                          threshold_span_s=(high-low)*capacity/abs(first["net_tps"]) if first["net_tps"] else None,
                          later_epochs=[dict(epoch=e["epoch"], producer_tps=e["producer_tps"],
                                             consumer_tps=e["consumer_tps"], net_tps=e["net_tps"],
                                             consumer_error_pct=100*(first["consumer_tps"]/e["consumer_tps"]-1),
                                             # Stable-sample hysteresis adds about 1 second.
                                             predicted_residence_s=max(12, abs((high if dp==2 else low)-e["start_fill"])*capacity/abs(first["net_tps"]))+1 if first["net_tps"] else None,
                                             observed_residence_s=e["residence_until_request_s"])
                                        for e in matching[1:]]))
    summary = dict(result=result, options=options, native_predictions=predictions["selected"],
                   epochs=epochs, switches=switches, memory=memory, health=health,
                   measured_fluid_model=flows,
                   max_observed_allocated_gib=max(r.get("peak_allocated", 0) for rows in observations.values() for r in rows)/2**30,
                   success=(result["returncode"] == 0 and len(switches) >= options["switches"]
                            and all(s["progress_preserved"] for s in switches)
                            and all(health[k] for k in ["steps_contiguous", "exact_global_tokens", "exact_micro_progress", "full_gradient_windows"])))
    summary["retired_objects_checked"] = sum(r.get("retired_objects_checked", 0)
                                            for rows in observations.values() for r in rows)
    summary["elastic_rank_after_shrink"] = [dict(epoch=r["epoch"], allocated_mib=r["allocated"]/2**20)
                                            for r in observations[1]
                                            if r["event"] == "switch_end" and r["target"] == 2]
    dead_steps = [r for r in steps if any(a.get("num_dead",0)>0 for a in r.get("auxk",{}).values())]
    if any("auxk" in r for r in steps):
        first_dead = dead_steps[0] if dead_steps else None
        first_dead_step = first_dead["step"] if first_dead else None
        hooks = sorted({h for r in steps for h in r.get("auxk",{})})
        active_switches = [s for s in switches if s["progress"] and first_dead_step is not None
                           and s["progress"][0]>=first_dead_step]
        # Use completed optimizer steps to locate rolling-rate windows in each
        # phase. Exclude the first four seconds after AuxK first appears.
        phase_statistics = []
        for label, active in [("before_dead",False),("after_dead_onset",True)]:
            for dp in [2,3]:
                subset=[r for r in steps if r["dp"]==dp and r["step"]>16
                        and ((first_dead_step is not None and r["step"]>=first_dead_step)==active)]
                if not subset:
                    continue
                rate_subset=[]
                for epoch in epochs:
                    if epoch["dp"]!=dp:continue
                    for r in samples:
                        if r["epoch"]!=epoch["epoch"] or r["phase"]!="active":continue
                        if r["timestamp"]<epoch["start_unix"]+10 or r.get("window_seconds",0)<3.9:continue
                        if not .02<r["fill_ratio"]<.98:continue
                        is_after=first_dead is not None and r["timestamp"]>=first_dead["timestamp"]
                        if is_after!=active:continue
                        if active and r["timestamp"]<first_dead["timestamp"]+4:continue
                        rate_subset.append(r)
                phase_statistics.append(dict(phase=label,dp=dp,updates=len(subset),
                    updates_with_dead=sum(any(a.get("num_dead",0)>0 for a in r["auxk"].values()) for r in subset),
                    median_cuda_ms=median(r["cuda_ms"] for r in subset),
                    peak_source_allocated_gib=max(r["peak_allocated"] for r in subset)/2**30,
                    producer_tps=median(r["vllm_tokens_per_s"] for r in rate_subset),
                    consumer_tps=median(r["sae_tokens_per_s"] for r in rate_subset),rate_samples=len(rate_subset)))
        aux_positive = {h:[r for r in losses if r.get("components",{}).get(h,{}).get("auxiliary_reconstruction_loss",0)>0]
                        for h in hooks}
        summary["dead_coverage"] = dict(dead_window=options.get("dead_window",1000),
            first_dead_step=first_dead_step,first_dead_unix=first_dead["timestamp"] if first_dead else None,
            active_updates=len(dead_steps),
            dead_max_by_hook={h:max(r.get("auxk",{}).get(h,{}).get("num_dead",0) for r in steps) for h in hooks},
            dead_final_by_hook={h:steps[-1].get("auxk",{}).get(h,{}).get("num_dead",0) for h in hooks},
            aux_execution_variants={h:sorted({(r["auxk"][h].get("selection"),r["auxk"][h].get("decoder"))
                                             for r in dead_steps if r["auxk"][h].get("num_dead",0)>0}) for h in hooks},
            positive_aux_loss_samples={h:len(rows) for h,rows in aux_positive.items()},
            first_positive_aux_loss_step={h:rows[0]["step"] if rows else None for h,rows in aux_positive.items()},
            switches_after_dead_onset=len(active_switches),
            target_dp_after_dead_onset=[s["target_dp"] for s in active_switches],
            age_migration_checks=sum(s["dead_ages_preserved"] is True for s in switches),
            phase_statistics=phase_statistics,
            both_topologies_have_active_aux=all(any(r["dp"]==dp for r in dead_steps) for dp in [2,3]),
            repeated_round_trips_after_dead=all(sum(s["target_dp"]==dp for s in active_switches)>=2 for dp in [2,3]))
        coverage=summary["dead_coverage"]
        coverage["validation_passed"] = bool(
            summary["success"] and first_dead_step is not None
            and first_dead_step>=coverage["dead_window"]+2
            and all(coverage["positive_aux_loss_samples"].values())
            and coverage["both_topologies_have_active_aux"]
            and coverage["repeated_round_trips_after_dead"]
            and all(s["dead_ages_preserved"] is True for s in active_switches))
        for endpoint, record in ([("first", losses[0]), ("last", losses[-1])] if losses else []):
            summary["health"][f"{endpoint}_sampled_mse_by_hook"] = {
                h:v.get("mse_loss") for h,v in record.get("components",{}).items()}
            summary["health"][f"{endpoint}_sampled_aux_by_hook"] = {
                h:v.get("auxiliary_reconstruction_loss") for h,v in record.get("components",{}).items()}
    flow_path = root / "flow_prediction.json"
    if flow_path.exists():
        frozen = json.loads(flow_path.read_text())
        created = frozen["profile"]["created_unix"]
        by_direction = {(r["source_target"],r["next_target"]):r["seconds"] for r in frozen["intervals"]}
        comparison = []
        for a,b in zip(decisions,decisions[1:]):
            if a["timestamp"] <= created:
                continue  # Require the whole interval to be in the future.
            predicted = by_direction[(a["decision_target_sae_dp"],b["decision_target_sae_dp"])]
            actual = b["timestamp"]-a["timestamp"]
            comparison.append(dict(from_epoch=a["epoch"]+1, to_epoch=b["epoch"]+1,
                                   source_target=a["decision_target_sae_dp"], next_target=b["decision_target_sae_dp"],
                                   predicted_s=predicted, measured_s=actual,
                                   error_pct=100*(predicted/actual-1)))
        validation = dict(calibration_run=frozen["profile"]["calibration_run"], created_unix=created,
                          intervals=comparison, mape_pct=stats.mean(abs(r["error_pct"]) for r in comparison) if comparison else None)
        summary["frozen_flow_validation"] = validation
        (root / "flow_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    start = samples[0]["timestamp"]
    times = [r["timestamp"]-start for r in samples]
    axes[0].plot(times, [r["fill_ratio"] for r in samples], color="#277591")
    for level in [low, high]:
        axes[0].axhline(level, color="gray", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="SHM occupancy", ylim=(0, 1.04), title="Real automatic DP2 / DP3 switching; no inserted delays")
    for key, label, color in [("vllm_tokens_per_s", "Production", "#d28946"), ("sae_tokens_per_s", "Consumption", "#277591")]:
        # A reset rate window has no estimate; it is not zero production.
        axes[1].plot(times, [r.get(key, float("nan"))/1000 for r in samples], label=label, color=color, alpha=.8)
    axes[1].set(ylabel="Logical ktokens/s"); axes[1].legend()
    axes[2].step(times, [r["active_sae_dp"] for r in samples], where="post", color="#64764e")
    axes[2].set(ylabel="SAE DP", yticks=[2, 3], xlabel="Time since first controller sample (s)")
    for ax in axes:
        ax.grid(alpha=.2)
        for row in decisions:
            ax.axvline(row["timestamp"]-start, color="#a54a42", linewidth=.8, alpha=.6)
        if dead_steps:
            ax.axvline(dead_steps[0]["timestamp"]-start, color="#7b3fa1",linestyle="--",label="First active AuxK")
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(root / f"watermarks.{extension}", dpi=160)
    plt.close(fig)
    if dead_steps:
        fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
        for h in hooks:
            axes[0].plot([r["step"] for r in steps],[r["auxk"][h]["num_dead"] for r in steps],label=h)
            axes[1].plot([r["step"] for r in losses],
                         [r["components"][h]["auxiliary_reconstruction_loss"] for r in losses],label=h)
        for dp in [2,3]:
            rows=[r for r in steps if r["dp"]==dp]
            axes[2].scatter([r["step"] for r in rows],[r["peak_allocated"]/2**30 for r in rows],s=5,label=f"SAE DP{dp}")
        axes[0].set(ylabel="Dead features / hook",title=f"Fresh training; dead window {options.get('dead_window',1000)} optimizer updates")
        axes[1].set(ylabel="AuxK loss (local GA window)")
        axes[2].set(ylabel="Source peak allocated (GiB)",xlabel="Optimizer update")
        for ax in axes:
            ax.legend();ax.grid(alpha=.2);ax.axvline(first_dead_step,color="#7b3fa1",linestyle="--")
        fig.tight_layout()
        for extension in ["png","svg"]:fig.savefig(root/f"dead_auxk.{extension}",dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
