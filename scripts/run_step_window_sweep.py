#!/usr/bin/env python3
"""Step-window profiling sweep over the 20 (mode, hooks, d_sae, batch) configs.

Reuses the configuration of results/nsys_sweep_dsae_batch_20_260731_fused_75s
(Llama-3.1-8B, vLLM tp=dp=1, k=256, ctx 2048, float32, fused Adam) and replaces
nsys tracing with the built-in step-window profiler: no per-step syncs, only one
sync at each window boundary.

Window placement comes from scripts/plan_step_window_sweep.py: profiling starts
at the 3rd vLLM producer cycle and each of the 2 windows encloses 2 cycles.

Everything else is left at the sweep's defaults. Run with --dry-run to print the
planned commands without launching anything.
"""

from __future__ import annotations

import argparse
import json
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
RUN_TIMEOUT_S = 1200

HOOK_POOL = [
    "blocks.21.hook_resid_post",
    "blocks.31.hook_resid_post",
    "blocks.11.hook_resid_post",
    "blocks.26.hook_resid_post",
]

RUN_GROUPS = [
    {"batch_tokens": 768, "d_sae": 32_768, "requested_hook_count": 4},
    {"batch_tokens": 768, "d_sae": 131_072, "requested_hook_count": 1},
    {"batch_tokens": 3_072, "d_sae": 49_152, "requested_hook_count": 2},
    {"batch_tokens": 4_096, "d_sae": 16_384, "requested_hook_count": 3},
    {"batch_tokens": 4_096, "d_sae": 65_536, "requested_hook_count": 1},
]
MODES = ["single", "tp2", "ddp", "fsdp"]

MODE_TOPOLOGY = {
    # mode: (sae_tp_size, sae_dp_size, sae_dp_mode, nproc, visible_devices)
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
    output_path: str
    log_path: str
    command: list[str]
    env: dict[str, str]


def build_specs(out_root: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for group in RUN_GROUPS:
        hook_count = min(int(group["requested_hook_count"]), len(HOOK_POOL))
        hooks = HOOK_POOL[:hook_count]
        batch = int(group["batch_tokens"])
        d_sae = int(group["d_sae"])
        for mode in MODES:
            tp, dp, dp_mode, nproc, devices = MODE_TOPOLOGY[mode]
            window = plan(batch_tokens=batch, sae_dp_size=dp)
            architecture = (
                "legacy_per_hook_wrapper" if hook_count == 1 else "unified_multi_hook"
            )
            run_id = f"{mode}_H{hook_count}_d{d_sae}_b{batch}"
            output_path = out_root / "runner_outputs" / run_id
            checkpoint_path = out_root / "runner_outputs" / f"{run_id}_checkpoints"
            log_path = out_root / "logs" / f"{run_id}.log"

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
                str(batch),
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
                # No per-step syncs: memory profiling syncs at every phase and
                # --synchronize-timing syncs inside the step, both of which would
                # inflate the window. timing_history stays on (pure perf_counter)
                # so the per-step breakdown is available for cross-checking.
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

            if nproc == 1:
                command = ["python3", *runner_args]
            else:
                command = [
                    "torchrun",
                    "--standalone",
                    f"--nproc_per_node={nproc}",
                    *runner_args,
                ]

            specs.append(
                RunSpec(
                    run_id=run_id,
                    mode=mode,
                    d_sae=d_sae,
                    hook_count=hook_count,
                    batch_tokens=batch,
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
                    output_path=str(output_path),
                    log_path=str(log_path),
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
    }


def main() -> None:
    import os

    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results" / "step_window_sweep_20",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated run_id substrings")
    args = p.parse_args()

    out_root: Path = args.out_root
    specs = build_specs(out_root)
    if args.only:
        needles = args.only.split(",")
        specs = [s for s in specs if any(n in s.run_id for n in needles)]

    if args.dry_run:
        for spec in specs:
            print(
                f"{spec.run_id}: start={spec.start_step} "
                f"win={spec.window_steps}x{spec.window_count} "
                f"cycles={spec.cycles_enclosed_per_window} "
                f"total_steps={spec.total_steps} tokens={spec.training_tokens}"
            )
        return

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "run_manifest.jsonl"
    status_path = out_root / "run_status.jsonl"
    with open(manifest_path, "w") as f:
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
        print(
            f"    {record['status']} in {record['elapsed_s']:.1f}s",
            flush=True,
        )
        with open(status_path, "a") as f:
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    main()
