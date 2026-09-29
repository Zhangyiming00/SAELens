"""Replay saved allocator pickles and summarize controlled static memory cases."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import pickle
import statistics
import sys
import importlib.util

REPO = Path(__file__).resolve().parents[2]
# This analyzer needs only pure Python. Avoid importing the package __init__,
# which probes CUDA even though allocator replay can run without GPU access.
spec = importlib.util.spec_from_file_location("memory_model", REPO / "sae_lens/autoconfig/megatron_memory_model.py")
memory_model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memory_model
spec.loader.exec_module(memory_model)
replay_allocator_window = memory_model.replay_allocator_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/static_memory_factors_20260927")
    args = parser.parse_args()
    rows = []
    for directory in sorted((args.root / "native").glob("*")):
        if not (directory / "result.json").exists() or "_failed_" in directory.name:
            continue
        status = json.loads((directory / "result.json").read_text())
        if status["returncode"] != 0:
            continue
        ranks = []
        for file in sorted(directory.glob("rank*.json")):
            if ".replay." in file.name:
                continue
            data = json.loads(file.read_text())
            rank = data["rank"]
            with (directory / f"rank{rank}.pickle").open("rb") as f:
                snapshot = pickle.load(f)  # Only snapshots made by this experiment.
            start = json.loads((directory / f"start_rank{rank}.json").read_text())
            replay = replay_allocator_window(snapshot, start, rank, data["storages"])
            assert replay["final_accounting_error"] == 0, (directory.name, rank, "final accounting")
            error = replay["peak_allocated"] - data["trace_memory"]["peak_allocated"]
            assert error == 0, (directory.name, rank, error)
            replay["peak_api_error"] = error
            # Keep the trace series on disk for plots without inflating the summary.
            (directory / f"rank{rank}.replay.json").write_text(json.dumps(replay, indent=2))
            data["replay_summary"] = {k: v for k, v in replay.items() if k != "series"}
            ranks.append(data)
        if not ranks:
            continue
        cfg = ranks[0]["config"]
        assert len(ranks) == cfg["tp"]*cfg["dp"]*cfg["pp"]
        step_ms = [max(r["steps"][i]["ms"] for r in ranks) for i in range(len(ranks[0]["steps"]))]
        worst = max(ranks, key=lambda r: r["memory"]["peak_allocated"])
        gi = 1024**3
        categories = {}
        for s in worst["storages"]:
            categories[s["category"]] = categories.get(s["category"], 0) + s["bytes"]
        row = dict(case=directory.name, **cfg, gpus=len(ranks), max_local_hooks=max(len(r["local_hooks"]) for r in ranks),
            ms=statistics.median(step_ms), ms_min=min(step_ms), ms_max=max(step_ms),
            peak_gib=max(r["memory"]["peak_allocated"] for r in ranks)/gi,
            trace_peak_gib=max(r["trace_memory"]["peak_allocated"] for r in ranks)/gi,
            reserved_gib=max(r["memory"]["peak_reserved"] for r in ranks)/gi,
            resident_gib=max(r["memory"]["allocated"] for r in ranks)/gi,
            device_end_gib=max(r["memory"]["device_used"] for r in ranks)/gi,
            params_gib=categories.get("parameters", 0)/gi,
            adam_gib=categories.get("adam_state", 0)/gi,
            grad_gib=categories.get("gradients", 0)/gi,
            inputs_gib=categories.get("inputs", 0)/gi,
            replay_exact=True)
        rows.append(row)
    (args.root / "native_summary.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (args.root / "native_summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(f'{row["case"]:22s} {row["peak_gib"]:6.3f} GiB {row["ms"]:7.2f} ms resident {row["resident_gib"]:5.3f} GiB')
    online = []
    for directory in sorted((args.root / "online").glob("*")):
        if not (directory / "result.json").exists() or "_failed_" in directory.name:
            continue
        status = json.loads((directory / "result.json").read_text())
        if status["returncode"] != 0:
            continue
        ranks = [json.loads(f.read_text()) for f in sorted(directory.glob("rank[0-9].json"))]
        cfg = ranks[0]["config"]
        assert len(ranks) == max(cfg["vtp"]*cfg["vdp"], cfg["stp"]*cfg["sdp"]*cfg["spp"])
        for rank, data in enumerate(ranks):
            assert len(data["steps"]) == 10
            assert data["global_update_batch"] == 4096
            path, = list(directory.rglob(f"memory_timeline_rank{rank}.pickle"))
            with path.open("rb") as f:
                snapshot = pickle.load(f)
            start = json.loads((directory / f"start_rank{rank}.json").read_text())
            replay = replay_allocator_window(snapshot, start, rank, [])
            traced, = [s for s in data["steps"] if s["traced"]]
            assert replay["peak_allocated"] == traced["peak_allocated"], (directory, rank, replay["peak_allocated"], traced["peak_allocated"])
            assert replay["final_accounting_error"] == 0
            (directory / f"rank{rank}.replay.json").write_text(json.dumps(replay, indent=2))
        # Larger/no-mix buffers alternate refill and serving-only steps.
        # Four steps cover two complete 2-step cycles; five would bias timing
        # toward cheap serving steps. Step 8 is traced and excluded.
        ms = [max(r["steps"][i]["ms"] for r in ranks) for i in range(4, 8)]
        weights = [json.loads(f.read_text()) for f in directory.glob("model_init_rank*.json")]
        online.append(dict(case=directory.name, **cfg, gpus=len(ranks), ms=statistics.mean(ms),
            peak_gib=max(s["peak_allocated"] for r in ranks for s in r["steps"][4:8])/2**30,
            init_peak_gib=max(w["memory"]["peak_allocated"] for w in weights)/2**30,
            reserved_gib=max(s["peak_reserved"] for r in ranks for s in r["steps"][4:8])/2**30,
            sampled_device_lifecycle_peak_gib=max(status["sampled_device_peak"])/2**30,
            weight_per_producer_rank_gib=max(sum(w["weight_bytes_param_scan"] for w in d["workers"]) for d in weights)/2**30,
            kv_per_producer_rank_gib=max(sum(w["kv_cache_bytes"] for w in d["workers"]) for d in weights)/2**30,
            replay_exact=True))
    (args.root / "online_summary.json").write_text(json.dumps(online, indent=2))
    if online:
        with (args.root / "online_summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(online[0]))
            writer.writeheader()
            writer.writerows(online)
    for row in online:
        print(f'online {row["case"]:16s} {row["peak_gib"]:6.3f} GiB {row["ms"]:7.2f} ms')
    plot(args.root, rows, online)


def plot(root, rows, online):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {r["case"]: r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    groups = [
        ("Wavefront / representation (TP2, H3)", ["wave_off", "wave_bound1", "base", "wave_lazy", "full_dense", "ragged_compact"]),
        ("Parallelism (fixed 8192 tokens/update)", ["tp1", "base", "tp4", "dp2", "dp2_zero", "dp4", "dp4_zero", "tp2dp2", "tp2dp2_zero", "pp2", "pp3_tp1"]),
        ("Dimensions and accumulation", ["batch4096", "base", "batch16384", "ga2", "ga4", "width8192", "width32768", "h1", "h2"]),
    ]
    for ax, (title, names) in zip(axes.flat, groups):
        values = [data[n] for n in names if n in data]
        ax.bar(range(len(values)), [r["peak_gib"] for r in values], color="#287d8e")
        ax.set_xticks(range(len(values)), [r["case"] for r in values], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Max rank allocated peak (GiB)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        for i, r in enumerate(values):
            ax.text(i, r["peak_gib"]+.06, f'{r["peak_gib"]:.2f}', ha="center", fontsize=8)
    ax = axes[1, 1]
    if online:
        ax.bar(range(len(online)), [r["peak_gib"] for r in online], color="#bb6f2a")
        ax.set_xticks(range(len(online)), [r["case"] for r in online], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Max rank allocated peak (GiB)")
        ax.set_title("Real Llama-3.1-8B online, SAE TP2 DP2")
        ax.grid(axis="y", alpha=.2)
    else:
        ax.set_visible(False)
    fig.suptitle("Static memory factors: current Megatron runtime, RTX 5090, FP32 SAE\nNative SAE has fixed active AuxK; online short runs have inactive AuxK", fontsize=12)
    fig.savefig(root / "memory_factors.png", dpi=160)
    fig.savefig(root / "memory_factors.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
