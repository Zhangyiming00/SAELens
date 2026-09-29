"""Summarize completed routing transport runs; failed pilots are excluded."""
import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = []
    for summary in sorted(args.directory.glob("*/summary.json")):
        directory = summary.parent
        spec = json.loads((directory / "spec.json").read_text())
        if spec.get("trace"):
            continue  # Profiling overhead is excluded from performance comparisons.
        records = [json.loads(line) for p in directory.glob("rank*.jsonl") for line in p.read_text().splitlines()]
        measurements = [r for r in records if r["event"] == "measurement"]
        for result in json.loads(summary.read_text()):
            ranks = [r for r in measurements if r["name"] == result["name"]]
            root = next(r for r in ranks if r["rank"] == 0)
            updates = root["update_cuda_ms"]
            half = len(updates) // 2
            blocks = [mean(updates[i:i + 4]) for i in range(0, len(updates) - 3, 4)]
            runs.append(dict(directory=directory.name, layout=spec["layout"], transport=spec["transport"],
                             **result, warmup=spec["warmup"], measured=spec["measured"],
                             cuda_mean_ms=mean(updates), cuda_cv_pct=100 * stdev(updates) / mean(updates),
                             four_step_cv_pct=100 * stdev(blocks) / mean(blocks) if len(blocks) > 1 else None,
                             half_drift_pct=100 * (mean(updates[half:]) / mean(updates[:half]) - 1),
                             data_cuda_ms=mean(root["data_cuda_ms"]), sae_cuda_ms=mean(root["sae_cuda_ms"]),
                             capture_ms_per_update=1000 * sum(root["capture_seconds"]) / root["updates"],
                             capture_count=len(root["capture_seconds"]),
                             losses={str(r["rank"]): r["losses"] for r in ranks},
                             final_input_sha256={str(r["rank"]): r.get("final_input_sha256") for r in ranks},
                             auxiliary={str(r["rank"]): r["auxiliary"] for r in ranks}))
    comparisons = []
    input_checks = []
    within_transport_loss = []
    for layout in ("tp4", "dp4", "pp3"):
        for name in ("noaux", "dead_s1600", "dead_s1200", "dead_s832"):
            modes = {mode: [r for r in runs if r["layout"] == layout and r["name"] == name and r["transport"] == mode]
                     for mode in ("nccl", "shm_async")}
            if not all(modes.values()):
                continue
            nccl, shm = (median(r["ms_per_update"] for r in modes[m]) for m in modes)
            losses = []
            # Same initialization, dataset order, phase schedule, and fixed masks.
            # Compare all terminal per-hook losses, without disturbing timed runs.
            for left in modes["nccl"]:
                for right in modes["shm_async"]:
                    if all(left["final_input_sha256"].values()) and all(right["final_input_sha256"].values()):
                        equal = left["final_input_sha256"] == right["final_input_sha256"]
                        input_checks.append(dict(layout=layout, name=name, left=left["directory"],
                                                 right=right["directory"], sampled_inputs_equal=equal))
                        assert equal, input_checks[-1]
                    for rank, hooks in left["losses"].items():
                        for h, components in hooks.items():
                            for k, a in components.items():
                                b = right["losses"][rank][h][k]
                                losses.append(dict(abs=abs(a - b), rel=abs(a - b) / max(abs(a), abs(b), 1e-12)))
            comparisons.append(dict(layout=layout, name=name, nccl_ms=nccl, shm_ms=shm,
                                    throughput_gain_pct=100 * (nccl / shm - 1),
                                    latency_reduction_pct=100 * (1 - shm / nccl),
                                    repeat_count={m: len(v) for m, v in modes.items()},
                                    repeat_range_pct={m: 100 * (max(r["ms_per_update"] for r in v) - min(r["ms_per_update"] for r in v))
                                                      / median(r["ms_per_update"] for r in v) for m, v in modes.items()},
                                    loss_max_abs=max(v["abs"] for v in losses), loss_max_rel=max(v["rel"] for v in losses),
                                    nccl_tokens_s=4096000 / nccl, shm_tokens_s=4096000 / shm))
            for mode, values in modes.items():
                if len(values) < 2:
                    continue
                errors = []
                for left, right in zip(values, values[1:], strict=False):
                    for rank, hooks in left["losses"].items():
                        for h, components in hooks.items():
                            for k, a in components.items():
                                b = right["losses"][rank][h][k]
                                errors.append((abs(a - b), abs(a - b) / max(abs(a), abs(b), 1e-12)))
                within_transport_loss.append(dict(layout=layout, name=name, transport=mode,
                                                  max_abs=max(x[0] for x in errors), max_rel=max(x[1] for x in errors)))
    result = dict(runs=runs, comparisons=comparisons, sampled_input_checks=input_checks,
                  within_transport_loss_variation=within_transport_loss)
    (args.directory / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("layout state NCCL_ms SHM_ms throughput_gain% repeats max_loss_abs")
    for row in comparisons:
        print(row["layout"], row["name"], f'{row["nccl_ms"]:.2f}', f'{row["shm_ms"]:.2f}',
              f'{row["throughput_gain_pct"]:+.2f}', row["repeat_count"], row["loss_max_abs"])


if __name__ == "__main__":
    main()
