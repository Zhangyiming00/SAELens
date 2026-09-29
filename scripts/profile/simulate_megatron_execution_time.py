"""Freeze strict interpolation tables, predict before holdouts, then validate."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sae_lens/autoconfig"))
from allocated_peak_model import state_payload  # noqa: E402
from execution_time_model import (  # noqa: E402
    SCHEMA,
    predict_native,
    predict_sparse,
    validate_config,
)
from interpolation_model import canonical  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def measurement(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result["returncode"]:
        raise ValueError(f"Failed measurement: {directory}")
    paths = sorted(directory.glob("rank[0-9]*.json"), key=lambda p: int(p.stem[4:]))
    ranks = [json.loads(p.read_text()) for p in paths]
    if not ranks or any(r["trace"] for r in ranks):
        raise ValueError("Need unprofiled native measurements")
    cfg = ranks[0]["config"]
    validate_config(cfg)
    for r in ranks:
        model = r["resolved_model_config"]
        if model["dtype"] != "float32" or any(model[k] != cfg[k] for k in ("d_in", "d_sae", "k", "auxk")):
            raise ValueError("Resolved runtime model differs from workload/FP32 memory assumptions")
    if [r["rank"] for r in ranks] != list(range(cfg["tp"]*cfg["dp"]*cfg["pp"])):
        raise ValueError("Incomplete rank set")
    ids = [s["step"] for s in ranks[0]["steps"]]
    if len(ids) < 3 or any(r["config"] != cfg or [s["step"] for s in r["steps"]] != ids for r in ranks):
        raise ValueError("Mismatched or insufficient timing samples")
    samples = [max(r["steps"][i]["ms"] for r in ranks) for i in range(len(ids))]
    def branch(info):
        # Counts and nnz are observations, not algorithm names. Keep every
        # categorical diagnostic, including direct-Aux versus packed-Aux.
        return {k: v for k, v in (info or {}).items() if isinstance(v, str)}
    regime = [dict(main=[branch(x) for x in r["main_execution"]],
                   aux=[branch(x) for x in r["aux_execution"]],
                   optimizer=r["optimizer_classes"], wave=r["effective_wavefront"],
                   overlap=r["effective_overlap"], gather=r["gather"]) for r in ranks]
    memory = []
    for r in ranks:
        structural = sum(state_payload(cfg, r["rank"]).values())
        resident = statistics.median(s["post_output_allocated_bytes"] for s in r["steps"])
        memory.append(dict(peak_residual_bytes=r["peak_allocated_bytes"]-structural,
                           resident_residual_bytes=resident-structural))
    env = {key: ranks[0][key] for key in ("gpu", "torch", "cuda")}
    if any(any(r[k] != env[k] for k in env) for r in ranks):
        raise ValueError("Heterogeneous runtime environment")
    return dict(name=directory.name, config=cfg, regime=regime,
                metrics=dict(wall_ms=statistics.median(samples), sample_min_ms=min(samples),
                             sample_max_ms=max(samples), rank_memory=memory),
                samples=samples, environment=env,
                input_sha256={str(p): sha(p) for p in paths})


def calibrate(args):
    configs = json.loads(args.configs.read_text())
    rows = [measurement(args.root/name) for name in configs]
    if any(r["config"] != configs[r["name"]] for r in rows):
        raise ValueError("Measurement does not match requested calibration")
    if len({canonical(r["environment"]) for r in rows}) != 1:
        raise ValueError("Do not mix hardware or software stacks")
    runtime_sources = json.loads((args.root / "source_sha256.json").read_text())
    # Absolute keys also pin explicit dead masks and input configuration files.
    sources = dict(runtime_sources)
    from profile_static_memory_factors import CACHE, HOOKS
    for path in [CACHE/"global_row_order.pt", *(CACHE/f"{h}.pt" for h in HOOKS[:max(c["h"] for c in configs.values())])]:
        sources[str(path)] = sha(path)
    for name in ("interpolation_model.py", "execution_time_model.py", "allocated_peak_model.py"):
        path = REPO / "sae_lens/autoconfig" / name
        sources[str(path.relative_to(REPO))] = sha(path)
    profile = dict(schema=SCHEMA, axes=["batch", "d_sae"], rows=rows,
                   environment=rows[0]["environment"], source_sha256=sources,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   scope="Native FP32 cached SAE updates; exact discrete execution/scheduler families")
    for row in rows:
        predict_native(row["config"], profile)
    dump(args.output, profile)
    print(json.dumps(dict(profile=str(args.output), calibration_rows=len(rows))))


def check_sources(profile):
    for rel, digest in profile["source_sha256"].items():
        if not (REPO/rel).is_file():
            raise ValueError(f"Profile source no longer exists: {rel}; re-freeze the calibration with the current model")
        if sha(REPO/rel) != digest:
            raise ValueError(f"Source changed since profiling: {rel}")


def predict(args):
    profile = json.loads(args.profile.read_text())
    check_sources(profile)
    configs = json.loads(args.configs.read_text())
    predictor = predict_sparse if profile["schema"] == "sparse_parts_interpolation_v1" else predict_native
    predictions = {name: predictor(cfg, profile) for name, cfg in configs.items()}
    dump(args.output, dict(created_utc=datetime.now(timezone.utc).isoformat(),
                           profile_sha256=sha(args.profile), predictions=predictions))
    print(json.dumps({n: (dict(ms=r["total_ms"], gib=r["peak_allocated_bytes"]/2**30)
                         if "total_ms" in r else r["metrics"])
                      for n, r in predictions.items()}, indent=2))


def validate(args):
    frozen = json.loads(args.predictions.read_text())
    profile = json.loads(args.profile.read_text())
    if sha(args.profile) != frozen["profile_sha256"]:
        raise ValueError("Validation profile differs from the frozen prediction profile")
    check_sources(profile)
    hold_sources = json.loads((args.root/"source_sha256.json").read_text())
    for path, digest in hold_sources.items():
        if path in profile["source_sha256"] and digest != profile["source_sha256"][path]:
            raise ValueError(f"Calibration/holdout runtime source mismatch: {path}")
    rows = []
    for name, pred in frozen["predictions"].items():
        obs = measurement(args.root/name)
        if obs["environment"] != profile["environment"]:
            raise ValueError(f"Calibration/holdout environment mismatch: {name}")
        if obs["config"] != pred["config"] or obs["regime"] != pred["regime"]:
            raise ValueError(f"Holdout configuration/observed path mismatch: {name}")
        peak = max(sum(state_payload(obs["config"], i).values()) + r["peak_residual_bytes"]
                   for i, r in enumerate(obs["metrics"]["rank_memory"]))
        actual = obs["metrics"]["wall_ms"]
        rows.append(dict(name=name, predicted_ms=pred["total_ms"], actual_ms=actual,
                         time_error_pct=100*(pred["total_ms"]/actual-1),
                         predicted_peak_bytes=pred["peak_allocated_bytes"], actual_peak_bytes=peak,
                         memory_error_pct=100*(pred["peak_allocated_bytes"]/peak-1),
                         actual_samples_ms=obs["samples"]))
    report = dict(predictions_sha256=sha(args.predictions), rows=rows,
                  time_mape_pct=statistics.mean(abs(r["time_error_pct"]) for r in rows),
                  time_max_error_pct=max(abs(r["time_error_pct"]) for r in rows),
                  memory_mape_pct=statistics.mean(abs(r["memory_error_pct"]) for r in rows),
                  memory_max_error_pct=max(abs(r["memory_error_pct"]) for r in rows))
    dump(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--configs", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--configs", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    for p in sub.choices.values():
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
