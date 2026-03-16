#!/usr/bin/env python
"""
Compare SAE training accuracy across three backends:
  - HookedVLLMModel TP=1
  - HookedVLLMModel TP=2
  - HookedTransformer

All backends use the *same* initial SAE weights (saved to a temp file by
the main process and loaded by each subprocess), so differences in the
training curve reflect only activation quality, not random initialisation.

Metrics reported per step:
  - mse_loss
  - aux_loss (TopK dead-neuron pressure)
  - explained_variance  (1 - mse / var(input))

At the end a summary table and pairwise final-loss comparison is printed.

Usage:
    python scripts/benchmark_training_accuracy.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \
        --n-steps 50
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYER = 21
DEFAULT_HOOK = "hook_resid_post"
D_IN = 4096
D_SAE = 16384
K = 100
CONTEXT_SIZE = 128
STORE_BATCH_PROMPTS = 4
N_BATCHES_IN_BUFFER = 32
TRAIN_BATCH_TOKENS = 2048


# ---- subprocess worker ----

def _train(
    backend: str,
    model_path: str,
    hooked_model_name: str,
    dataset_path: str,
    layer: int,
    hook: str,
    device: str,
    n_steps: int,
    init_weights_path: str,
    out_path: str,
) -> None:
    """Train SAE for n_steps, save per-step metrics to out_path as JSON."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from datasets import load_from_disk
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from sae_lens.config import LoggingConfig
    from sae_lens.saes.sae import SAEMetadata, TrainStepInput
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
    from sae_lens.training.activations_store import ActivationsStore

    hook_name = f"blocks.{layer}.{hook}"

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
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
        model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
            device=device, dtype=torch.bfloat16, local_files_only=True,
        )
        model.eval()

    # --- ActivationsStore ---
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
        total_training_tokens=n_steps * TRAIN_BATCH_TOKENS * 10,
        store_batch_size_prompts=STORE_BATCH_PROMPTS,
        train_batch_size_tokens=TRAIN_BATCH_TOKENS,
        prepend_bos=False,
        normalize_activations="none",
        device=torch.device(device),
        dtype="bfloat16",
        disable_concat_sequences=False,
        sequence_separator_token=None,
        activations_mixing_fraction=0.0,
    )

    # --- SAE: load shared initial weights ---
    sae_cfg = TopKTrainingSAEConfig(
        d_in=D_IN, d_sae=D_SAE, k=K,
        dtype="float32", device=device,
        metadata=SAEMetadata(hook_name=hook_name),
    )
    sae = TopKTrainingSAE(sae_cfg)
    sae.load_state_dict(torch.load(init_weights_path, map_location=device))
    sae.to(device)

    from torch.optim import Adam
    optimizer = Adam(sae.parameters(), lr=1e-4)

    records: list[dict] = []
    pbar = tqdm(total=n_steps, desc=backend, file=sys.stderr)

    for step in range(n_steps):
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

        with torch.no_grad():
            var = batch.var(dim=0).sum().item()
            mse = out.losses["mse_loss"].item()
            ev = 1.0 - mse / max(var / D_IN, 1e-8)
            aux = out.losses.get("auxiliary_reconstruction_loss", torch.tensor(0.0)).item()

        records.append({
            "step": step,
            "mse_loss": mse,
            "aux_loss": aux,
            "explained_variance": ev,
        })
        pbar.update(1)
    pbar.close()

    with open(out_path, "w") as f:
        json.dump({"backend": backend, "records": records}, f)
    print(f"DONE:{backend}", flush=True)

    del model, store, sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- subprocess launcher ----

def launch(backend: str, args: argparse.Namespace, init_weights_path: str, out_path: str) -> dict:
    cmd = [
        sys.executable, __file__,
        "--_run-backend", backend,
        "--model-path", args.model_path,
        "--hooked-model-name", args.hooked_model_name,
        "--dataset-path", args.dataset_path,
        "--layer", str(args.layer),
        "--device", args.device,
        "--n-steps", str(args.n_steps),
        "--init-weights-path", init_weights_path,
        "--out-path", out_path,
    ]
    print(f"\n[{backend}] Training {args.n_steps} steps ...", flush=True)
    proc = subprocess.run(cmd, check=False, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed (exit {proc.returncode})")
    with open(out_path) as f:
        return json.load(f)


# ---- reporting ----

def print_report(results: list[dict], n_steps: int) -> None:
    log_steps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    log_steps = sorted(set(max(0, min(n_steps - 1, s)) for s in log_steps))

    print()
    print("=" * 80)
    print("  SAE Training Accuracy  —  per-step metrics")
    print(f"  d_in={D_IN} | d_sae={D_SAE} | k={K} | train_batch={TRAIN_BATCH_TOKENS} tok")
    print("=" * 80)

    # Loss curve table
    backends = [r["backend"] for r in results]
    header = f"{'step':>6}"
    for b in backends:
        header += f"  {b+' mse':>14}  {b+' ev%':>9}"
    print(header)
    print("-" * 80)

    for s in log_steps:
        row = f"{s:>6}"
        for r in results:
            rec = r["records"][s]
            row += f"  {rec['mse_loss']:>14.4f}  {rec['explained_variance']*100:>9.2f}"
        print(row)

    print("-" * 80)

    # Summary: final values
    print("\nFinal step summary:")
    ref = results[0]["records"][-1]
    for r in results:
        final = r["records"][-1]
        mse_delta = final["mse_loss"] - ref["mse_loss"]
        print(
            f"  {r['backend']:<18}  mse={final['mse_loss']:.4f} "
            f"({'ref' if r is results[0] else f'{mse_delta:+.4f} vs ref'})  "
            f"ev={final['explained_variance']*100:.2f}%  "
            f"aux={final['aux_loss']:.4f}"
        )

    # Pairwise final mse agreement
    print("\nPairwise final mse difference (should be ~0 if activations match):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a_mse = results[i]["records"][-1]["mse_loss"]
            b_mse = results[j]["records"][-1]["mse_loss"]
            print(
                f"  {results[i]['backend']} vs {results[j]['backend']}: "
                f"Δmse = {abs(a_mse - b_mse):.4f}  "
                f"({abs(a_mse - b_mse) / max(a_mse, 1e-8) * 100:.2f}%)"
            )


# ---- arg parsing ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    # Internal
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--init-weights-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---- main ----

def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        _train(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            dataset_path=args.dataset_path,
            layer=args.layer,
            hook=DEFAULT_HOOK,
            device=args.device,
            n_steps=args.n_steps,
            init_weights_path=args.init_weights_path,
            out_path=args.out_path,
        )
        return

    print(f"Model:  {args.model_path}")
    print(f"Layer:  {args.layer}.{DEFAULT_HOOK}  |  SAE: d_sae={D_SAE}, k={K}")
    print(f"Steps:  {args.n_steps}  |  train_batch={TRAIN_BATCH_TOKENS} tok")

    # Create shared initial SAE weights
    from sae_lens.saes.sae import SAEMetadata
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

    hook_name = f"blocks.{args.layer}.{DEFAULT_HOOK}"
    sae_cfg = TopKTrainingSAEConfig(
        d_in=D_IN, d_sae=D_SAE, k=K,
        dtype="float32", device="cpu",
        metadata=SAEMetadata(hook_name=hook_name),
    )
    sae_init = TopKTrainingSAE(sae_cfg)
    print("Saving shared initial SAE weights ...")

    with tempfile.TemporaryDirectory(prefix="sae_acc_bench_") as tmp:
        init_path = f"{tmp}/init.pt"
        torch.save(sae_init.state_dict(), init_path)

        backends = ["vllm_tp1"]
        if not args.skip_tp2:
            backends.append("vllm_tp2")
        if not args.skip_hooked:
            backends.append("hooked")

        results = []
        for backend in backends:
            out_path = f"{tmp}/{backend}.json"
            results.append(launch(backend, args, init_path, out_path))

    print_report(results, args.n_steps)


if __name__ == "__main__":
    main()
