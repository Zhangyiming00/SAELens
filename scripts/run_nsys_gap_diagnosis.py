#!/usr/bin/env python3
"""nsys traces for three step-window sweep configs, to diagnose the gap column.

Same runner configuration and nsys flags as
results/nsys_sweep_dsae_batch_20_260731_fused_75s, plus the step-window profiler
arguments from results/step_window_sweep_20 so the trace timeline carries the
same window boundaries the wall-clock table was measured over. Window edges are
recoverable from the NVTX ranges and from
`step_window_profile_sae_rank*.jsonl` written next to each run's output.

The three configs share (H=4, d_sae=32768, batch=768) and differ only in SAE
topology (single / tp2 / ddp), which is where the gap column jumps from 522ms to
919ms in the wall-clock table.

Unlike the 7/31 sweep this does not cap the run at 75s: the run must reach the
last profiled step for the windows to close, so the timeout is generous and the
process is left to exit on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_step_window_sweep import plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "/data/models/Llama-3.1-8B"
DATASET_PATH = "../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048"
RUN_TIMEOUT_S = 1800

HOOK_POOL = [
    "blocks.21.hook_resid_post",
    "blocks.31.hook_resid_post",
    "blocks.11.hook_resid_post",
    "blocks.26.hook_resid_post",
]

# Same five (batch, d_sae, hook_count) groups as the step-window sweep.
RUN_GROUPS = [
    {"batch_tokens": 768, "d_sae": 32_768, "hook_count": 4},
    {"batch_tokens": 768, "d_sae": 131_072, "hook_count": 1},
    {"batch_tokens": 3_072, "d_sae": 49_152, "hook_count": 2},
    {"batch_tokens": 4_096, "d_sae": 16_384, "hook_count": 3},
    {"batch_tokens": 4_096, "d_sae": 65_536, "hook_count": 1},
]

# mode: (sae_tp_size, sae_dp_size, sae_dp_mode, nproc, visible_devices)
MODE_TOPOLOGY = {
    "single": (1, 1, "ddp", 1, "0"),
    "tp2": (2, 1, "ddp", 2, "0,1"),
    "ddp": (1, 2, "ddp", 2, "0,1"),
    "fsdp": (1, 2, "fsdp", 2, "0,1"),
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    mode: str
    d_sae: int
    hook_count: int
    batch_tokens: int
    local_batch_tokens: int
    sae_tp_size: int
    sae_dp_size: int
    sae_dp_mode: str
    nproc: int
    start_step: int
    window_steps: int
    window_count: int
    total_steps: int
    training_tokens: int
    vllm_cycle_steps: list[int]
    cycles_enclosed_per_window: list[list[int]]
    nsys_base: str
    output_path: str
    log_path: str
    command: list[str]
    env: dict[str, str]


def build_specs(out_root: Path, modes: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for group in RUN_GROUPS:
        hook_count = int(group["hook_count"])
        hooks = HOOK_POOL[:hook_count]
        batch_tokens = int(group["batch_tokens"])
        d_sae = int(group["d_sae"])
        for mode in modes:
          tp, dp, dp_mode, nproc, devices = MODE_TOPOLOGY[mode]
          window = plan(batch_tokens=batch_tokens, sae_dp_size=dp)
          architecture = (
              "legacy_per_hook_wrapper" if hook_count == 1 else "unified_multi_hook"
          )
          run_id = f"{mode}_H{hook_count}_d{d_sae}_b{batch_tokens}"
          output_path = out_root / "runner_outputs" / run_id
          checkpoint_path = out_root / "runner_outputs" / f"{run_id}_checkpoints"
          nsys_base = out_root / "nsys" / run_id

          runner_args = [
              "scripts/run_sae_runner_gpu.py",
              "--model-name",
              MODEL_NAME,
              "--dataset-path",
              DATASET_PATH,
              "--hook-name",
              hooks[0],
              "--hook-names",
              ",".join(hooks),
              "--d-sae",
              str(d_sae),
              "--k",
              "256",
              "--vllm-tp-size",
              "1",
              "--sae-tp-size",
              str(tp),
              "--vllm-dp-size",
              "1",
              "--sae-dp-size",
              str(dp),
              "--sae-pp-size",
              "1",
              "--sae-dp-mode",
              dp_mode,
              "--multi-sae-distributed-architecture",
              architecture,
              "--multi-sae-backward-mode",
              "combined",
              "--multi-sae-backward-order",
              "forward",
              "--multi-sae-stats-sync-mode",
              "immediate",
              "--multi-sae-stats-sync-interval",
              "1",
              "--multi-sae-seed-mode",
              "same",
              "--training-tokens",
              str(window["training_tokens"]),
              "--train-batch-size-tokens",
              str(batch_tokens),
              "--context-size",
              "2048",
              "--store-batch-size-prompts",
              "1",
              "--n-batches-in-buffer",
              "2",
              "--max-model-len",
              "2049",
              "--max-num-batched-tokens",
              "4096",
              "--gpu-memory-utilization",
              "0.5",
              "--dtype",
              "float32",
              "--act-store-device",
              "cuda",
              # Keep the step free of extra syncs, exactly as in the wall-clock
              # sweep, so the trace measures the same step the table measured.
              "--save-memory-every-n-steps",
              "0",
              "--save-mse-every-n-steps",
              "0",
              "--save-timing-every-n-steps",
              "1",
              "--step-window-profile-start-step",
              str(window["start_step"]),
              "--step-window-profile-window-steps",
              str(window["window_steps"]),
              "--step-window-profile-window-count",
              str(window["window_count"]),
              "--no-save-final-checkpoint",
              "--checkpoint-path",
              str(checkpoint_path),
              "--output-path",
              str(output_path),
          ]

          if mode == "fsdp":
              runner_args.extend(
                  [
                      "--fsdp-sharding-strategy",
                      "shard_grad_op",
                      "--fsdp-forward-prefetch",
                      "--fsdp-backward-prefetch",
                      "backward_post",
                  ]
              )

          launcher = (
              ["python3"]
              if nproc == 1
              else ["torchrun", "--standalone", f"--nproc_per_node={nproc}"]
          )
          command = [
              "nsys",
              "profile",
              "--trace=cuda,nvtx,osrt",
              "--sample=none",
              "--cpuctxsw=none",
              "--stats=false",
              "--force-overwrite=true",
              "--export=sqlite",
              f"--output={nsys_base}",
              *launcher,
              *runner_args,
          ]

          specs.append(
              RunSpec(
                  run_id=run_id,
                  mode=mode,
                  d_sae=d_sae,
                  hook_count=hook_count,
                  batch_tokens=batch_tokens,
                  local_batch_tokens=int(window["local_batch_tokens"]),
                  sae_tp_size=tp,
                  sae_dp_size=dp,
                  sae_dp_mode=dp_mode,
                  nproc=nproc,
                  start_step=int(window["start_step"]),
                  window_steps=int(window["window_steps"]),
                  window_count=int(window["window_count"]),
                  total_steps=int(window["total_steps"]),
                  training_tokens=int(window["training_tokens"]),
                  vllm_cycle_steps=list(window["vllm_cycle_steps"]),
                  cycles_enclosed_per_window=window["cycles_enclosed_per_window"],
                  nsys_base=str(nsys_base),
                  output_path=str(output_path),
                  log_path=str(out_root / "logs" / f"{run_id}.log"),
                  command=command,
                  env={
                      "CUDA_VISIBLE_DEVICES": devices,
                      "SAE_ADAM_IMPL": "fused",
                      "SAE_FUSED_ADAM": "1",
                  },
              )
          )
    return specs


def run_one(spec: RunSpec, env_base: dict[str, str]) -> dict:
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(spec.output_path).mkdir(parents=True, exist_ok=True)
    Path(spec.nsys_base).parent.mkdir(parents=True, exist_ok=True)
    env = {**env_base, **spec.env}
    t0 = time.perf_counter()
    with open(spec.log_path, "w") as log:
        proc = subprocess.run(
            spec.command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=RUN_TIMEOUT_S,
        )
    elapsed = time.perf_counter() - t0
    return {
        "run_id": spec.run_id,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "status": "ok" if proc.returncode == 0 else "error",
        "sqlite_exists": Path(spec.nsys_base + ".sqlite").exists(),
        "rep_exists": Path(spec.nsys_base + ".nsys-rep").exists(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root", type=Path, default=ROOT / "results" / "nsys_gap_diagnosis"
    )
    p.add_argument("--modes", default="single,tp2,ddp,fsdp")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_root: Path = args.out_root
    specs = build_specs(out_root, args.modes.split(","))

    if args.dry_run:
        for spec in specs:
            print(f"{spec.run_id}: start={spec.start_step} "
                  f"win={spec.window_steps}x{spec.window_count} "
                  f"total_steps={spec.total_steps}")
            print("  " + " ".join(spec.command))
        return

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "run_manifest.jsonl", "w") as f:
        for spec in specs:
            json.dump(asdict(spec), f)
            f.write("\n")

    env_base = dict(os.environ)
    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx}/{len(specs)}] {spec.run_id} ...", flush=True)
        try:
            record = run_one(spec, env_base)
        except subprocess.TimeoutExpired:
            record = {
                "run_id": spec.run_id,
                "returncode": None,
                "elapsed_s": RUN_TIMEOUT_S,
                "status": "timeout",
            }
        print(f"    {record['status']} in {record['elapsed_s']:.1f}s", flush=True)
        with open(out_root / "run_status.jsonl", "a") as f:
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    main()
