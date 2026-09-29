"""Add row-tile knots to a coarse sparse table; validate on NEW shapes.

The original holdouts become diagnostic development data. None of the new
holdout shapes are imported into calibration. Interpolation is unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

from profile_sparse_parts import REPO, dump, measure
from execution_time_model import predict_sparse
from interpolation_model import canonical, family


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=9)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a fresh output directory")
    base = json.loads((args.base/"profile.json").read_text())
    for path, digest in base["source_sha256"].items():
        if sha(REPO/path) != digest:
            raise ValueError(f"Base measurement source changed: {path}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from sae_lens.openai_sae_adapter import _row_tile
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    families = {family(r["config"], base["axes"]): {k: v for k, v in r["config"].items() if k not in base["axes"]}
                for r in base["rows"]}
    cal, hold = {}, {}
    for c in families.values():
        key = f"{c['engine']}_k{c['k']}_{c['pattern']}_w{c['workspace_mib']}"
        knots = {1024, 2048, 4096, 8192}
        if c["engine"] == "openai":
            lengths = [c["k"]] if c["pattern"] == "uniform" else [c["k"]-c["k"]//4, c["k"]+c["k"]//4]
            multiplier = len(lengths)
            for length in lengths:
                padded = 1 << (length-1).bit_length()
                tile = _row_tile(c["d_in"], padded, c["workspace_mib"])
                # First and second full row tiles: saturation and coexistence
                # of the previous page's temporaries. Keep both sides of steps.
                for count in (1, 2):
                    boundary = multiplier*tile*count
                    knots.update(n for n in (boundary, boundary+multiplier) if 1024 <= n <= 8192)
        for b in sorted(knots):
            for f in (16384, 32768, 65536):
                cal[f"{key}_b{b}_f{f}"] = dict(c, rows=b, width=f)
        for b, f in ((3072, 24576), (6144, 49152)):
            hold[f"{key}_b{b}_f{f}"] = dict(c, rows=b, width=f)
    if {canonical(c) for c in cal.values()} & {canonical(c) for c in hold.values()}:
        raise ValueError("Holdout leaked into calibration")
    args.output.mkdir(parents=True)
    dump(args.output/"calibration_configs.json", cal)
    dump(args.output/"holdout_configs.json", hold)
    dump(args.output/"correctness.json", json.loads((args.base/"correctness.json").read_text()))
    previous = {canonical(r["config"]): r for r in base["rows"]}
    rows, added = [], 0
    with (args.output/"calibration.jsonl").open("x") as stream:
        for name, cfg in cal.items():
            row = previous.get(canonical(cfg))
            if row is None:
                row = measure(name, cfg, args, torch)
                added += 1
                print("CAL", name, flush=True)
            rows.append(row)
            stream.write(json.dumps(row)+"\n")
            stream.flush()
    profile = dict(base, rows=rows, created_utc=datetime.now(timezone.utc).isoformat(),
                   refinement=dict(base_profile_sha256=sha(args.base/"profile.json"),
                                   diagnostic_validation_sha256=sha(args.base/"validation.json"),
                                   reused_calibration_rows=len(rows)-added, new_calibration_rows=added,
                                   new_holdout_shapes=len(hold),
                                   reason="Add row-tile saturation boundaries and intermediate width/rows"))
    profile["source_sha256"][str(Path(__file__).resolve().relative_to(REPO))] = sha(__file__)
    dump(args.output/"profile.json", profile)
    predictions = {name: predict_sparse(cfg, profile) for name, cfg in hold.items()}
    dump(args.output/"predictions.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                             profile_sha256=sha(args.output/"profile.json"), predictions=predictions))
    validation = []
    with (args.output/"holdout.jsonl").open("x") as stream:
        for name, cfg in hold.items():
            obs = measure(name, cfg, args, torch)
            stream.write(json.dumps(obs)+"\n")
            stream.flush()
            pred = predictions[name]
            if canonical(obs["regime"]) != canonical(pred["regime"]):
                raise ValueError("Holdout changed sparse algorithm regime")
            for stage in obs["metrics"]:
                for metric, actual in obs["metrics"][stage].items():
                    predicted = pred["metrics"][stage][metric]
                    validation.append(dict(name=name, stage=stage, metric=metric, actual=actual,
                                           predicted=predicted, error_pct=100*(predicted/actual-1)))
            print("HOLD", name, flush=True)
    summary = {metric: dict(mape_pct=statistics.mean(abs(r["error_pct"]) for r in validation if r["metric"] == metric),
                            max_error_pct=max(abs(r["error_pct"]) for r in validation if r["metric"] == metric))
               for metric in ("wall_ms", "cuda_ms", "peak_extra_bytes")}
    dump(args.output/"validation.json", dict(summary=summary, rows=validation))
    print(json.dumps(dict(calibration_rows=len(rows), added=added, holdouts=len(hold), summary=summary), indent=2))


if __name__ == "__main__":
    main()
