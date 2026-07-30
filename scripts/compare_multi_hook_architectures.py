#!/usr/bin/env python
"""Compare legacy and unified multi-hook SAE training on fixed activations.

This is a synthetic SAE-side check for the two multi-hook distributed
architectures:

* legacy_per_hook_wrapper
* unified_multi_hook

It reports host completion time for a manual training loop without calling
``torch.cuda.synchronize()``. It also runs an accuracy pass that writes normal
MSE histories and final checkpoints, then compares every step's MSE and every
final checkpoint tensor.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/compare_multi_hook_architectures.py
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
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae_lens.config import LoggingConfig, SAETrainerConfig  # noqa: E402
from sae_lens.constants import MSE_HISTORY_FILENAME, SAE_WEIGHTS_FILENAME  # noqa: E402
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig  # noqa: E402
from sae_lens.training.multi_hook_sae import MultiHookSAE  # noqa: E402
from sae_lens.training.multi_sae_trainer import (  # noqa: E402
    MultiSAETrainer,
    sanitize_hook_name_for_path,
)


ARCHITECTURES = ("legacy_per_hook_wrapper", "unified_multi_hook")


@dataclass(frozen=True)
class BenchConfig:
    d_in: int = 128
    d_sae: int = 512
    k: int = 16
    batch_size: int = 64
    n_steps: int = 6
    n_hooks: int = 2
    lr: float = 1e-3
    seed: int = 12345
    dtype: str = "float32"

    @property
    def total_samples(self) -> int:
        return self.batch_size * self.n_steps


class FixedBatchProvider:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self._batches = batches
        self._idx = 0

    def __iter__(self) -> "FixedBatchProvider":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._idx >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._idx]
        self._idx += 1
        return batch


def _hook_names(n_hooks: int) -> list[str]:
    return [f"blocks.{idx}.hook_resid_post" for idx in range(n_hooks)]


def _make_sae_cfg(cfg: BenchConfig, device: str) -> TopKTrainingSAEConfig:
    return TopKTrainingSAEConfig(
        d_in=cfg.d_in,
        d_sae=cfg.d_sae,
        k=cfg.k,
        dtype=cfg.dtype,
        device=device,
        normalize_activations="none",
        decoder_init_norm=0.1,
        apply_b_dec_to_input=False,
        rescale_acts_by_decoder_norm=True,
    )


def _make_sae(cfg: BenchConfig, device: str) -> TopKTrainingSAE:
    return TopKTrainingSAE(_make_sae_cfg(cfg, device))


def _make_initial_state_by_hook(
    cfg: BenchConfig,
    hook_names: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    state_by_hook: dict[str, dict[str, torch.Tensor]] = {}
    for idx, hook_name in enumerate(hook_names):
        torch.manual_seed(cfg.seed + idx)
        sae = _make_sae(cfg, "cpu")
        state_by_hook[hook_name] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in sae.state_dict().items()
        }
    return state_by_hook


def _make_batches(
    cfg: BenchConfig,
    hook_names: list[str],
) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 10_000)
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(cfg.n_steps):
        batches.append(
            {
                hook_name: torch.randn(
                    cfg.batch_size,
                    cfg.d_in,
                    generator=gen,
                    dtype=getattr(torch, cfg.dtype),
                )
                for hook_name in hook_names
            }
        )
    return batches


def _write_payload(path: Path, cfg: BenchConfig) -> None:
    hook_names = _hook_names(cfg.n_hooks)
    payload = {
        "config": asdict(cfg),
        "hook_names": hook_names,
        "state_by_hook": _make_initial_state_by_hook(cfg, hook_names),
        "batches": _make_batches(cfg, hook_names),
    }
    torch.save(payload, path)


def _trainer_cfg(
    *,
    cfg: BenchConfig,
    device: str,
    architecture: str,
    output_path: Path | None,
    checkpoint_path: Path | None,
    record_mse: bool,
    save_final_checkpoint: bool,
    timing_pass: bool,
) -> SAETrainerConfig:
    return SAETrainerConfig(
        device=device,
        n_checkpoints=0,
        total_training_samples=cfg.total_samples,
        train_batch_size_samples=cfg.batch_size,
        output_path=str(output_path) if output_path is not None else None,
        save_mse_every_n_steps=1 if record_mse else 0,
        save_timing_every_n_steps=0,
        save_memory_every_n_steps=0,
        record_memory_empty_cache=False,
        record_memory_timeline_step=-1,
        synchronize_timing=False,
        multi_sae_backward_order="forward",
        multi_sae_stats_sync_mode="periodic" if timing_pass else "immediate",
        multi_sae_stats_sync_interval=cfg.n_steps + 100,
        multi_sae_distributed_architecture=architecture,  # type: ignore[arg-type]
        lr=cfg.lr,
        lr_end=None,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        lr_decay_steps=0,
        n_restart_cycles=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        dead_feature_window=1000,
        feature_sampling_window=1000,
        autocast=False,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        quiesce_checkpoint_path=None,
        save_final_checkpoint=save_final_checkpoint,
        logger=LoggingConfig(log_to_wandb=False),
    )


def _build_trainer(
    *,
    payload: dict[str, Any],
    architecture: str,
    device: str,
    output_path: Path | None,
    checkpoint_path: Path | None,
    record_mse: bool,
    save_final_checkpoint: bool,
    timing_pass: bool,
) -> MultiSAETrainer:
    cfg = BenchConfig(**payload["config"])
    hook_names: list[str] = payload["hook_names"]
    base_sae_by_hook: dict[str, TopKTrainingSAE] = {}
    for hook_name in hook_names:
        sae = _make_sae(cfg, device)
        sae.load_state_dict(
            {
                name: tensor.to(device)
                for name, tensor in payload["state_by_hook"][hook_name].items()
            }
        )
        base_sae_by_hook[hook_name] = sae

    multi_hook_sae = None
    if architecture == "unified_multi_hook":
        multi_hook_sae = MultiHookSAE(hook_names, base_sae_by_hook)

    trainer_cfg = _trainer_cfg(
        cfg=cfg,
        device=device,
        architecture=architecture,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        record_mse=record_mse,
        save_final_checkpoint=save_final_checkpoint,
        timing_pass=timing_pass,
    )
    return MultiSAETrainer(
        hook_names=hook_names,
        sae_by_hook=base_sae_by_hook,
        base_sae_by_hook=base_sae_by_hook,
        multi_hook_sae=multi_hook_sae,
        data_provider=FixedBatchProvider(payload["batches"]),
        save_checkpoint_fn=None,
        cfg=trainer_cfg,
        dp_group=None,
        token_count_weighted_dp=False,
        sae_dp_mode="ddp",
        backward_mode="combined",
    )


def _run_timing_worker(
    *,
    payload_path: Path,
    architecture: str,
    device: str,
    result_path: Path,
) -> None:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    cfg = BenchConfig(**payload["config"])
    trainer = _build_trainer(
        payload=payload,
        architecture=architecture,
        device=device,
        output_path=None,
        checkpoint_path=None,
        record_mse=False,
        save_final_checkpoint=False,
        timing_pass=True,
    )

    t0 = time.perf_counter()
    for batch in payload["batches"]:
        local_n = next(iter(batch.values())).shape[0]
        scaled_batch = {
            hook_name: trainer.activation_scaler_by_hook[hook_name](
                batch[hook_name].to(device)
            )
            for hook_name in payload["hook_names"]
        }
        trainer.n_training_samples += local_n
        trainer._train_step(scaled_batch, local_n)
        trainer.n_training_steps += 1
        trainer.lr_scheduler.step()
    completion_time_s = time.perf_counter() - t0

    result = {
        "architecture": architecture,
        "kind": "timing_no_cuda_sync",
        "completion_time_s": completion_time_s,
        "samples": cfg.total_samples,
        "steps": cfg.n_steps,
        "tokens_per_s": cfg.total_samples / completion_time_s,
        "note": "No explicit torch.cuda.synchronize() is called in the timed region.",
    }
    result_path.write_text(json.dumps(result, indent=2))


def _run_accuracy_worker(
    *,
    payload_path: Path,
    architecture: str,
    device: str,
    run_dir: Path,
    result_path: Path,
) -> None:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    cfg = BenchConfig(**payload["config"])
    output_path = run_dir / "output"
    checkpoint_path = run_dir / "checkpoints"
    trainer = _build_trainer(
        payload=payload,
        architecture=architecture,
        device=device,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        record_mse=True,
        save_final_checkpoint=True,
        timing_pass=False,
    )

    t0 = time.perf_counter()
    trainer.fit()
    completion_time_s = time.perf_counter() - t0

    final_checkpoint = checkpoint_path / f"final_{cfg.total_samples}"
    result = {
        "architecture": architecture,
        "kind": "accuracy",
        "completion_time_s": completion_time_s,
        "mse_history_path": str(output_path / MSE_HISTORY_FILENAME),
        "final_checkpoint_path": str(final_checkpoint),
    }
    result_path.write_text(json.dumps(result, indent=2))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _compare_mse_histories(
    legacy_path: Path,
    unified_path: Path,
    *,
    atol: float,
) -> dict[str, Any]:
    legacy_rows = _read_jsonl(legacy_path)
    unified_rows = _read_jsonl(unified_path)
    if len(legacy_rows) != len(unified_rows):
        raise AssertionError(
            f"MSE history length mismatch: {len(legacy_rows)} vs {len(unified_rows)}"
        )

    max_abs_diff = 0.0
    mismatches: list[dict[str, Any]] = []
    for legacy_row, unified_row in zip(legacy_rows, unified_rows, strict=True):
        if legacy_row["step"] != unified_row["step"]:
            raise AssertionError(
                f"MSE step mismatch: {legacy_row['step']} vs {unified_row['step']}"
            )
        for hook_name, legacy_hook in legacy_row["hooks"].items():
            unified_hook = unified_row["hooks"][hook_name]
            diff = abs(float(legacy_hook["mse_loss"]) - float(unified_hook["mse_loss"]))
            max_abs_diff = max(max_abs_diff, diff)
            if diff > atol:
                mismatches.append(
                    {
                        "step": legacy_row["step"],
                        "hook_name": hook_name,
                        "legacy_mse": float(legacy_hook["mse_loss"]),
                        "unified_mse": float(unified_hook["mse_loss"]),
                        "abs_diff": diff,
                    }
                )
    if mismatches:
        raise AssertionError(f"MSE mismatches above atol={atol}: {mismatches[:5]}")
    return {
        "steps": len(legacy_rows),
        "max_abs_mse_diff": max_abs_diff,
        "atol": atol,
    }


def _checkpoint_weight_files(
    checkpoint_path: Path,
    hook_names: list[str],
) -> dict[str, Path]:
    return {
        hook_name: checkpoint_path
        / sanitize_hook_name_for_path(hook_name)
        / SAE_WEIGHTS_FILENAME
        for hook_name in hook_names
    }


def _compare_final_checkpoints(
    legacy_checkpoint: Path,
    unified_checkpoint: Path,
    hook_names: list[str],
    *,
    atol: float,
) -> dict[str, Any]:
    max_abs_diff = 0.0
    mismatches: list[dict[str, Any]] = []
    legacy_files = _checkpoint_weight_files(legacy_checkpoint, hook_names)
    unified_files = _checkpoint_weight_files(unified_checkpoint, hook_names)
    for hook_name in hook_names:
        legacy_state = load_file(legacy_files[hook_name])
        unified_state = load_file(unified_files[hook_name])
        if sorted(legacy_state) != sorted(unified_state):
            raise AssertionError(f"Checkpoint key mismatch for {hook_name}")
        for name, legacy_tensor in legacy_state.items():
            unified_tensor = unified_state[name]
            diff = (legacy_tensor - unified_tensor).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            if diff > atol:
                mismatches.append(
                    {
                        "hook_name": hook_name,
                        "tensor": name,
                        "max_abs_diff": diff,
                    }
                )
    if mismatches:
        raise AssertionError(
            f"Final checkpoint mismatches above atol={atol}: {mismatches[:5]}"
        )
    return {
        "max_abs_checkpoint_diff": max_abs_diff,
        "atol": atol,
    }


def _run_child(
    *,
    script_path: Path,
    out_dir: Path,
    payload_path: Path,
    architecture: str,
    kind: str,
    gpu: str,
    device: str,
) -> Path:
    result_path = out_dir / f"{architecture}_{kind}.json"
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-kind",
        kind,
        "--architecture",
        architecture,
        "--payload-path",
        str(payload_path),
        "--out-dir",
        str(out_dir),
        "--device",
        device,
        "--result-path",
        str(result_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return result_path


def _run_main(args: argparse.Namespace) -> None:
    cfg = BenchConfig(
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_hooks=args.n_hooks,
        lr=args.lr,
        seed=args.seed,
        dtype=args.dtype,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "payload.pt"
    _write_payload(payload_path, cfg)

    script_path = Path(__file__).resolve()
    timing_results: dict[str, Any] = {}
    accuracy_results: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        timing_path = _run_child(
            script_path=script_path,
            out_dir=out_dir,
            payload_path=payload_path,
            architecture=architecture,
            kind="timing",
            gpu=args.gpu,
            device=args.device,
        )
        timing_results[architecture] = json.loads(timing_path.read_text())

    for architecture in ARCHITECTURES:
        accuracy_path = _run_child(
            script_path=script_path,
            out_dir=out_dir,
            payload_path=payload_path,
            architecture=architecture,
            kind="accuracy",
            gpu=args.gpu,
            device=args.device,
        )
        accuracy_results[architecture] = json.loads(accuracy_path.read_text())

    hook_names = _hook_names(cfg.n_hooks)
    mse_comparison = _compare_mse_histories(
        Path(accuracy_results["legacy_per_hook_wrapper"]["mse_history_path"]),
        Path(accuracy_results["unified_multi_hook"]["mse_history_path"]),
        atol=args.mse_atol,
    )
    checkpoint_comparison = _compare_final_checkpoints(
        Path(accuracy_results["legacy_per_hook_wrapper"]["final_checkpoint_path"]),
        Path(accuracy_results["unified_multi_hook"]["final_checkpoint_path"]),
        hook_names,
        atol=args.checkpoint_atol,
    )

    summary = {
        "config": asdict(cfg),
        "gpu": args.gpu,
        "device_inside_worker": args.device,
        "timing": timing_results,
        "accuracy_runs": accuracy_results,
        "mse_comparison": mse_comparison,
        "checkpoint_comparison": checkpoint_comparison,
    }
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/multi_hook_arch_compare_gpu0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-in", type=int, default=128)
    parser.add_argument("--d-sae", type=int, default=512)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--n-hooks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    parser.add_argument("--mse-atol", type=float, default=1e-6)
    parser.add_argument("--checkpoint-atol", type=float, default=1e-6)
    parser.add_argument("--worker-kind", choices=["timing", "accuracy"], default=None)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=None)
    parser.add_argument("--payload-path", default=None)
    parser.add_argument("--result-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_kind is None:
        _run_main(args)
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this comparison.")
    if args.architecture is None or args.payload_path is None or args.result_path is None:
        raise SystemExit("Worker mode requires architecture, payload-path, and result-path.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    payload_path = Path(args.payload_path)
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_kind == "timing":
        _run_timing_worker(
            payload_path=payload_path,
            architecture=args.architecture,
            device=args.device,
            result_path=result_path,
        )
    else:
        run_dir = Path(args.out_dir) / args.architecture
        run_dir.mkdir(parents=True, exist_ok=True)
        _run_accuracy_worker(
            payload_path=payload_path,
            architecture=args.architecture,
            device=args.device,
            run_dir=run_dir,
            result_path=result_path,
        )


if __name__ == "__main__":
    main()
