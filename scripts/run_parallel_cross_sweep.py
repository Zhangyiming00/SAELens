#!/usr/bin/env python3
"""Step-window sweep over the SAE-parallelism x vLLM-parallelism cross product.

Fixed model config: H=1, d_in=4096, d_sae=65536, batch=4096, k=256, fused Adam,
float32, context 2048. Only the topology varies:

- SAE: single (tp1 dp1), tp2, ddp (dp2), fsdp (dp2, shard_grad_op)
- vLLM: tp1/dp1, tp2/dp1, and tp1/dp2

Timing comes from the built-in step-window profiler: two windows, each enclosing
two vLLM producer cycles, with syncs only at the window boundaries. No nsys.

batch=4096 invokes vLLM on every step, so the vLLM cadence period is one step
and the window width divides it exactly. That removes the phase-alignment
problem the b768 configs had, where a 5-step window straddled an 8-step cadence
and the two windows enclosed fetches of different cost.

Colocated routing places producers on ranks [0, vllm_tp*vllm_dp) and consumers
on ranks [0, sae_tp*sae_dp). Producer-only ranks run the same producer protocol
as consumer ranks but do not create an independent step-window profiler: the
consumer window already includes their vLLM work, and an extra CUDA sync would
split the vLLM collective protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

HOOK = "blocks.21.hook_resid_post"
D_SAE = 65_536
BATCH_TOKENS = 4_096
K = 256

# name: (sae_tp_size, sae_dp_size, sae_dp_mode)
SAE_TOPOLOGY = {
    "single": (1, 1, "ddp"),
    "tp2": (2, 1, "ddp"),
    "ddp": (1, 2, "ddp"),
    "fsdp": (1, 2, "fsdp"),
}
# name: (vllm_tp_size, vllm_dp_size)
VLLM_TOPOLOGY = {
    "vtp1dp1": (1, 1),
    "vtp2dp1": (2, 1),
    "vtp1dp2": (1, 2),
}

# Trace each run with nsys, so vLLM, SAE and the step wall can all be measured on
# one timeline in the same units. Set by --nsys.
USE_NSYS = False


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    sae_mode: str
    vllm_mode: str
    d_sae: int
    hook_count: int
    batch_tokens: int
    k: int
    sae_tp_size: int
    sae_dp_size: int
    sae_dp_mode: str
    vllm_tp_size: int
    vllm_dp_size: int
    nproc: int
    start_step: int
    window_steps: int
    window_count: int
    total_steps: int
    training_tokens: int
    output_path: str
    log_path: str
    command: list[str]
    env: dict[str, str]


def build_specs(out_root: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for sae_mode, (stp, sdp, dp_mode) in SAE_TOPOLOGY.items():
        for vllm_mode, (vtp, vdp) in VLLM_TOPOLOGY.items():
            # Colocated routing: producers and consumers share the same devices.
            nproc = max(vdp * vtp, sdp * stp)
            window = plan(batch_tokens=BATCH_TOKENS, sae_dp_size=sdp)
            run_id = f"sae{sae_mode}_{vllm_mode}"
            output_path = out_root / "runner_outputs" / run_id
            checkpoint_path = out_root / "runner_outputs" / f"{run_id}_checkpoints"

            runner_args = [
                "scripts/run_sae_runner_gpu.py",
                "--model-name",
                MODEL_NAME,
                "--dataset-path",
                DATASET_PATH,
                "--hook-name",
                HOOK,
                "--hook-names",
                HOOK,
                "--d-sae",
                str(D_SAE),
                "--k",
                str(K),
                "--vllm-tp-size",
                str(vtp),
                "--vllm-dp-size",
                str(vdp),
                "--sae-tp-size",
                str(stp),
                "--sae-dp-size",
                str(sdp),
                "--sae-pp-size",
                "1",
                "--sae-dp-mode",
                dp_mode,
                "--multi-sae-distributed-architecture",
                "legacy_per_hook_wrapper",
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
                str(BATCH_TOKENS),
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
            if sae_mode == "fsdp":
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
            if USE_NSYS:
                command = [
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx,osrt",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--stats=false",
                    "--force-overwrite=true",
                    "--export=sqlite",
                    f"--output={out_root / 'nsys' / run_id}",
                    *launcher,
                    *runner_args,
                ]
            else:
                command = [*launcher, *runner_args]

            specs.append(
                RunSpec(
                    run_id=run_id,
                    sae_mode=sae_mode,
                    vllm_mode=vllm_mode,
                    d_sae=D_SAE,
                    hook_count=1,
                    batch_tokens=BATCH_TOKENS,
                    k=K,
                    sae_tp_size=stp,
                    sae_dp_size=sdp,
                    sae_dp_mode=dp_mode,
                    vllm_tp_size=vtp,
                    vllm_dp_size=vdp,
                    nproc=nproc,
                    start_step=int(window["start_step"]),
                    window_steps=int(window["window_steps"]),
                    window_count=int(window["window_count"]),
                    total_steps=int(window["total_steps"]),
                    training_tokens=int(window["training_tokens"]),
                    output_path=str(output_path),
                    log_path=str(out_root / "logs" / f"{run_id}.log"),
                    command=command,
                    env={
                        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(nproc)),
                        "SAE_ADAM_IMPL": "fused",
                        "SAE_FUSED_ADAM": "1",
                    },
                )
            )
    return specs


MIN_FREE_GB = 20


def free_gb(path: Path) -> float:
    stat = shutil.disk_usage(path)
    return stat.free / 1024**3


def run_one(spec: RunSpec, env_base: dict[str, str]) -> dict:
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(spec.output_path).mkdir(parents=True, exist_ok=True)
    if USE_NSYS:
        # nsys will not create the report's directory itself; without this it
        # fails with "No such file or directory" and drops the report in /tmp
        # while the run otherwise succeeds.
        for part in spec.command:
            if part.startswith("--output="):
                Path(part.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    # A full disk has previously killed runs mid-flight in ways that look like a
    # topology bug rather than an out-of-space failure. Fail loudly instead.
    available = free_gb(Path(spec.output_path))
    if available < MIN_FREE_GB:
        return {
            "run_id": spec.run_id,
            "returncode": None,
            "elapsed_s": 0.0,
            "status": "skipped_low_disk",
            "free_gb": available,
        }
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
        "free_gb_after": free_gb(Path(spec.output_path)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root", type=Path, default=ROOT / "results" / "parallel_cross_sweep"
    )
    p.add_argument("--only", default=None, help="comma-separated run_id substrings")
    p.add_argument(
        "--nsys",
        action="store_true",
        help="trace each run with nsys so vLLM/SAE/wall share one timeline",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    global USE_NSYS
    USE_NSYS = args.nsys

    specs = build_specs(args.out_root)
    if args.only:
        needles = args.only.split(",")
        specs = [s for s in specs if any(n in s.run_id for n in needles)]

    if args.dry_run:
        for s in specs:
            print(
                f"{s.run_id:22s} nproc={s.nproc} sae_tp={s.sae_tp_size} "
                f"sae_dp={s.sae_dp_size} vllm_tp={s.vllm_tp_size} "
                f"vllm_dp={s.vllm_dp_size} start={s.start_step} "
                f"win={s.window_steps}x{s.window_count} steps={s.total_steps}"
            )
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    with open(args.out_root / "run_manifest.jsonl", "w") as f:
        for s in specs:
            json.dump(asdict(s), f)
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
        with open(args.out_root / "run_status.jsonl", "a") as f:
            json.dump(record, f)
            f.write("\n")


if __name__ == "__main__":
    main()
