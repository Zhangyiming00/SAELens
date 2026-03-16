#!/usr/bin/env python
"""
Benchmark the *complete* production pipeline:
  ActivationsStore (LLM inference) → SAETrainer (TopK SAE forward + backward + step)

Three backends compared:
  - HookedVLLMModel TP=1
  - HookedVLLMModel TP=2
  - HookedTransformer (single GPU)

Each backend runs in its own subprocess so GPU memory is fully released between runs.
The subprocess prints a JSON result line to stdout; the main process collects and
prints a comparison table.

Usage:
    python scripts/benchmark_full_pipeline.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048

Key measured metrics:
    - activation_tok_per_s   : tokens through the LLM per second (inference bottleneck)
    - training_tok_per_s     : tokens through the full pipeline per second
    - sae_step_per_s         : SAE optimizer steps per second
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYER = 21
DEFAULT_HOOK = "hook_resid_post"
D_IN = 4096        # Llama-3.1-8B hidden dim
D_SAE = 16384      # 4x expansion
K = 100

# Pipeline config
CONTEXT_SIZE = 128
STORE_BATCH_PROMPTS = 4     # sequences per LLM call
N_BATCHES_IN_BUFFER = 32     # LLM calls buffered before training
TRAIN_BATCH_TOKENS = 2048   # tokens per SAE training step

# Timing
N_WARMUP_STEPS = 3          # SAE training steps to discard
N_MEASURE_STEPS = 10        # SAE training steps to time


# ---- subprocess worker ----

def _run_pipeline(
    backend: str,          # "vllm_tp1" | "vllm_tp2" | "hooked"
    model_path: str,
    hooked_model_name: str,
    dataset_path: str,
    layer: int,
    hook: str,
    device: str,
) -> None:
    """Full pipeline run in subprocess. Prints one JSON line to stdout."""
    import gc

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.saes.sae import SAEMetadata
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
    from sae_lens.training.activations_store import ActivationsStore
    from sae_lens.training.sae_trainer import SAETrainer, SAETrainerConfig

    hook_name = f"blocks.{layer}.{hook}"
    total_steps = N_WARMUP_STEPS + N_MEASURE_STEPS

    # --- Build model ---
    if backend.startswith("vllm"):
        from sae_lens.vllm_model import HookedVLLMModel
        tp = 2 if backend == "vllm_tp2" else 1
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = HookedVLLMModel(
            model_path, tokenizer,
            tensor_parallel_size=tp,
            max_model_len=CONTEXT_SIZE + 1,
            enable_prefix_caching=False,
        )
    else:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
            device=device, dtype=torch.bfloat16, local_files_only=True,
        )
        model.eval()

    # --- Build ActivationsStore directly (bypass from_config to avoid loading model twice) ---
    dataset = load_from_disk(dataset_path)
    store = ActivationsStore(
        model=model,  # type: ignore[arg-type]
        dataset=dataset,
        streaming=False,
        hook_name=hook_name,
        hook_head_index=None,
        context_size=CONTEXT_SIZE,
        d_in=D_IN,
        n_batches_in_buffer=N_BATCHES_IN_BUFFER,
        total_training_tokens=total_steps * TRAIN_BATCH_TOKENS * 10,  # large enough
        store_batch_size_prompts=STORE_BATCH_PROMPTS,
        train_batch_size_tokens=TRAIN_BATCH_TOKENS,
        prepend_bos=False,
        normalize_activations="none",
        device=torch.device(device),
        dtype="bfloat16",
        disable_concat_sequences=False,
        sequence_separator_token=None,
        activations_mixing_fraction=0.0,  # no mixing → pure throughput
    )

    # --- Build small TopK SAE ---
    sae_cfg = TopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        k=K,
        dtype="float32",
        device=device,
        metadata=SAEMetadata(hook_name=hook_name),
    )
    sae = TopKTrainingSAE(sae_cfg)

    trainer_cfg = SAETrainerConfig(
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        total_training_samples=total_steps * TRAIN_BATCH_TOKENS,
        device=device,
        autocast=False,
        lr=1e-4,
        lr_end=1e-5,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_decay_steps=0,
        n_restart_cycles=1,
        train_batch_size_samples=TRAIN_BATCH_TOKENS,
        dead_feature_window=1000,
        feature_sampling_window=1000,
        logger=LoggingConfig(log_to_wandb=False, log_weights_to_wandb=False),
    )

    # --- Manual training loop with timing ---
    sae.to(device)

    # Track LLM inference time separately
    lm_time = 0.0
    lm_tokens = 0
    step_times: list[float] = []

    # Monkey-patch get_activations to track LLM time
    _orig_get_acts = store.get_activations
    def _timed_get_acts(batch_tokens: torch.Tensor):
        nonlocal lm_time, lm_tokens
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        result = _orig_get_acts(batch_tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        lm_time += time.perf_counter() - t0
        lm_tokens += batch_tokens.numel()
        return result
    store.get_activations = _timed_get_acts  # type: ignore[method-assign]

    from torch.optim import Adam
    optimizer = Adam(sae.parameters(), lr=trainer_cfg.lr)

    from tqdm import tqdm
    from sae_lens.saes.sae import TrainStepInput
    pbar = tqdm(total=total_steps, desc=f"{backend}", file=sys.stderr)

    for step in range(total_steps):
        t0 = time.perf_counter()
        batch = next(store).to(device)
        out = sae.training_forward_pass(
            TrainStepInput(
                sae_in=batch,
                coefficients=sae.get_coefficients(),
                dead_neuron_mask=torch.zeros(D_SAE, dtype=torch.bool, device=device),
                n_training_steps=step,
                is_logging_step=False,
            )
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if step >= N_WARMUP_STEPS:
            step_times.append(elapsed)
        pbar.update(1)
    pbar.close()

    measure_tokens = N_MEASURE_STEPS * TRAIN_BATCH_TOKENS
    total_wall = sum(step_times)
    training_tps = measure_tokens / total_wall
    # LLM tps: accumulate only for measure steps (warmup has different mix)
    # Approximate: total lm_tokens / lm_time
    lm_tps = lm_tokens / lm_time if lm_time > 0 else 0.0
    sae_steps_per_s = N_MEASURE_STEPS / total_wall

    result = {
        "backend": backend,
        "training_tok_per_s": training_tps,
        "activation_tok_per_s": lm_tps,
        "sae_step_per_s": sae_steps_per_s,
        "wall_s_per_step": total_wall / N_MEASURE_STEPS,
    }
    print(f"RESULT:{json.dumps(result)}", flush=True)

    # cleanup
    del model, store, sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- subprocess launcher ----

def launch_subprocess(backend: str, args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable, __file__,
        "--_run-backend", backend,
        "--model-path", args.model_path,
        "--hooked-model-name", args.hooked_model_name,
        "--dataset-path", args.dataset_path,
        "--layer", str(args.layer),
        "--device", args.device,
    ]
    print(f"\n[{backend}] Launching subprocess ...", flush=True)
    proc = subprocess.run(cmd, check=False, capture_output=False,
                          stdout=subprocess.PIPE, stderr=sys.stderr,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} subprocess failed (exit {proc.returncode})")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise RuntimeError(f"No RESULT line from {backend} subprocess")


# ---- arg parsing ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    # Internal
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---- report ----

def print_report(results: list[dict]) -> None:
    print()
    print("=" * 85)
    print("  Full Pipeline Benchmark  (ActivationsStore + TopK SAE training)")
    print(f"  Layer {DEFAULT_LAYER} | d_in={D_IN} | d_sae={D_SAE} | k={K}")
    print(f"  context={CONTEXT_SIZE} | store_batch={STORE_BATCH_PROMPTS}seq | train_batch={TRAIN_BATCH_TOKENS}tok")
    print("=" * 85)
    header = f"{'backend':<18} {'train tok/s':>12} {'LLM tok/s':>12} {'SAE step/s':>11} {'s/step':>8}"
    print(header)
    print("-" * 85)
    for r in results:
        print(
            f"{r['backend']:<18} "
            f"{r['training_tok_per_s']:>12.0f} "
            f"{r['activation_tok_per_s']:>12.0f} "
            f"{r['sae_step_per_s']:>11.2f} "
            f"{r['wall_s_per_step']:>8.3f}"
        )
    print("-" * 85)
    print()
    print("Columns:")
    print("  train tok/s   : tokens/s through the full pipeline (LLM + SAE + optimizer)")
    print("  LLM tok/s     : tokens/s through the LLM only (inference bottleneck)")
    print("  SAE step/s    : SAE optimizer steps per second")
    print("  s/step        : wall seconds per training step")


# ---- main ----

def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        _run_pipeline(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            dataset_path=args.dataset_path,
            layer=args.layer,
            hook=DEFAULT_HOOK,
            device=args.device,
        )
        return

    print(f"Model:         {args.model_path}")
    print(f"Dataset:       {args.dataset_path}")
    print(f"Layer:         {args.layer} | hook: {DEFAULT_HOOK}")
    print(f"SAE:           d_sae={D_SAE}, k={K}")
    print(f"Warmup steps:  {N_WARMUP_STEPS}  |  Measure steps: {N_MEASURE_STEPS}")

    results = []
    results.append(launch_subprocess("vllm_tp1", args))
    if not args.skip_tp2:
        results.append(launch_subprocess("vllm_tp2", args))
    if not args.skip_hooked:
        results.append(launch_subprocess("hooked", args))

    print_report(results)


if __name__ == "__main__":
    main()
