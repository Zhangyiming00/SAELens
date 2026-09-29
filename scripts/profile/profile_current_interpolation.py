"""Reproducible H1/H3 native calibration and holdout grid for interpolation v3."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def configurations():
    base = dict(tp=1, dp=1, pp=1, h=1, batch=4096, ga=1, d_in=4096,
                d_sae=65536, k=128, auxk=0, backend="sharded_dense", key_backend="torch",
                wave="off", live=1, zero=False, overlap="off", aux="auto", dead=0,
                main_compute="inherit", aux_compute="inherit",
                execution=dict(main_representation="none", aux_representation="none",
                               main_compute="none", aux_compute="none",
                               ragged_decoder_engine="openai"))
    families = {}
    for name, changes in {
        "tp1": {}, "tp2": dict(tp=2), "tp4": dict(tp=4),
        "dp2": dict(dp=2), "dp4": dict(dp=4), "tp2dp2": dict(tp=2, dp=2),
        "dp4zero": dict(dp=4, zero=True), "tp2dp2zero": dict(tp=2, dp=2, zero=True),
        "aux_tp2": dict(tp=2, auxk=2048, dead=2048),
        "h3_tp4_lazy": dict(tp=4, h=3, wave="lazy"),
        "triton_tp1": {}, "mixed_tp1": {},
    }.items():
        c = copy.deepcopy(base)
        c.update(changes)
        if name == "triton_tp1":
            c["execution"]["ragged_decoder_engine"] = "triton"
        if name == "mixed_tp1":
            c["execution"]["main_dweight"] = "dense"
        families[name] = c
    cal, hold = {}, {}
    for name, cfg in families.items():
        for batch in (4096, 8192):
            for width in (32768, 65536):
                cal[f"{name}_b{batch}_f{width}"] = dict(cfg, batch=batch, d_sae=width)
        hold[f"{name}_b6144_f49152"] = dict(cfg, batch=6144, d_sae=49152)
    cal["tp1_b18432_f65536"] = dict(families["tp1"], batch=18432)
    cal["tp1_b4096_f163840"] = dict(families["tp1"], d_sae=163840)
    hold["tp1_b12288_f65536"] = dict(families["tp1"], batch=12288)
    hold["tp1_b4096_f131072"] = dict(families["tp1"], d_sae=131072)
    return cal, hold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("prepare", "calibration", "freeze", "holdout", "validate"), required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cal, hold = configurations()
    for name, data in (("calibration", cal), ("holdout", hold)):
        path = out/f"{name}_configs.json"
        if path.exists():
            if json.loads(path.read_text()) != data:
                raise ValueError("Experiment configuration changed; choose a new directory")
        else:
            path.write_text(json.dumps(data, indent=2)+"\n")
    def run(script, *argv):
        subprocess.run([sys.executable, str(HERE/script), *map(str, argv)], cwd=REPO, check=True)
    if args.phase in ("calibration", "holdout"):
        run("profile_megatron_execution_time_v2.py", "--output", out/args.phase,
            "--config-file", out/f"{args.phase}_configs.json", "--warmup", 4, "--steps", 12)
    elif args.phase == "freeze":
        run("simulate_megatron_execution_time.py", "calibrate", "--root", out/"calibration",
            "--configs", out/"calibration_configs.json", "--output", out/"profile.json")
        run("simulate_megatron_execution_time.py", "predict", "--profile", out/"profile.json",
            "--configs", out/"holdout_configs.json", "--output", out/"predictions.json")
    elif args.phase == "validate":
        run("simulate_megatron_execution_time.py", "validate", "--root", out/"holdout",
            "--profile", out/"profile.json", "--predictions", out/"predictions.json", "--output", out/"validation.json")


if __name__ == "__main__":
    main()
