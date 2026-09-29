"""Replay trusted local native/online snapshots; report allocated bytes only.

This validates allocation lifetimes, not an unseen configuration's peak. No CUDA
or model weights are needed. Use --root for the parent of native/ and online/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "_allocated_replay", REPO / "sae_lens/autoconfig/megatron_memory_model.py"
)
assert spec is not None and spec.loader is not None
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def replay_case(directory: Path, kind: str) -> list[dict]:
    status = json.loads((directory / "result.json").read_text())
    if status["returncode"] != 0:
        return []
    records = []
    for start_file in sorted(directory.glob("start_rank*.json")):
        rank = int(start_file.stem.removeprefix("start_rank"))
        data = json.loads((directory / f"rank{rank}.json").read_text())
        if kind == "native":
            snapshot_file = directory / f"rank{rank}.pickle"
            expected = data["trace_memory"]["peak_allocated"]
            storages = data["storages"]
        else:
            snapshot_file, = directory.rglob(f"memory_timeline_rank{rank}.pickle")
            traced, = (s for s in data["steps"] if s["traced"])
            expected = traced["peak_allocated"]
            storages = []
        with snapshot_file.open("rb") as handle:
            snapshot = pickle.load(handle)  # Only load locally generated, trusted snapshots.
        result = model.replay_allocator_window(
            snapshot, json.loads(start_file.read_text()), rank, storages
        )
        error = result["peak_allocated"] - expected
        if error or result["final_accounting_error"]:
            raise ValueError(f"{directory}, rank {rank}: inconsistent allocated accounting")
        records.append(dict(
            case=str(directory), kind=kind, rank=rank,
            snapshot=str(snapshot_file),
            baseline_allocated=result["baseline_allocated"],
            peak_allocated=result["peak_allocated"],
            end_allocated=result["end_allocated"],
            api_peak_allocated=expected, peak_error_bytes=error,
            final_error_bytes=result["final_accounting_error"],
            categories_at_peak=result["categories_at_allocated_peak"],
            largest_sites_at_peak=result["largest_sites_at_allocated_peak"],
        ))
    if not records:
        raise ValueError(f"Successful case has no training snapshots: {directory}")
    cfg = data["config"]
    expected_ranks = (
        cfg["tp"] * cfg["dp"] * cfg["pp"] if kind == "native"
        else max(cfg["vtp"] * cfg["vdp"], cfg["stp"] * cfg["sdp"] * cfg["spp"])
    )
    if len(records) != expected_ranks:
        raise ValueError(f"{directory}: incomplete rank coverage")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for root in args.root:
        for kind in ("native", "online"):
            for status_file in sorted((root / kind).glob("*/result.json")):
                if "_failed_" not in status_file.parent.name:
                    records.extend(replay_case(status_file.parent, kind))
    if not records:
        raise ValueError("No successful training snapshots found")
    report = dict(
        metric="torch.cuda.memory_allocated (bytes)",
        method="measured allocator event replay; not configuration extrapolation",
        cases=len({r["case"] for r in records}), ranks=len(records),
        max_peak_error_bytes=max(abs(r["peak_error_bytes"]) for r in records),
        max_final_error_bytes=max(abs(r["final_error_bytes"]) for r in records),
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
