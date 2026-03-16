#!/usr/bin/env python
"""
Train SAE on each backend for N steps, then compare which features activate
on the same held-out token sequences.

Pipeline:
  1. Main process saves shared initial SAE weights + fixed test token batches.
  2. Each backend subprocess: trains N steps, then encodes the fixed test
     tokens and saves feature_acts to disk.
  3. Main process loads feature_acts from all backends and reports:
       - mean Jaccard similarity of top-k feature sets per token position
       - mean cosine similarity of feature activation vectors
       - mean overlap count (out of k)

This tests whether the trained SAEs, despite being trained on slightly
different (bfloat16-rounded) activations, discover the same features.

Usage:
    python scripts/compare_feature_activation.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \
        --n-train-steps 200 \
        --n-test-batches 5
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

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

def _worker(
    backend: str,
    model_path: str,
    hooked_model_name: str,
    dataset_path: str,
    layer: int,
    hook: str,
    device: str,
    n_train_steps: int,
    init_weights_path: str,
    test_tokens_path: str,   # fixed test batches saved by main process
    out_path: str,
) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    import gc
    from datasets import load_from_disk
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from sae_lens.saes.sae import SAEMetadata, TrainStepInput
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
    from sae_lens.training.activations_store import ActivationsStore
    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

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
        total_training_tokens=n_train_steps * TRAIN_BATCH_TOKENS * 10,
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
    pbar = tqdm(total=n_train_steps, desc=f"[{backend}] train", file=sys.stderr)

    # --- Training loop ---
    for step in range(n_train_steps):
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
        pbar.update(1)
    pbar.close()

    final_loss = out.losses["mse_loss"].item()
    print(f"[{backend}] final mse_loss={final_loss:.2f}", file=sys.stderr)

    # --- Eval: encode fixed test tokens with the trained SAE ---
    test_batches: list[torch.Tensor] = torch.load(test_tokens_path, map_location="cpu")
    stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)

    all_feature_acts: list[torch.Tensor] = []
    sae.eval()
    with torch.no_grad():
        for test_batch in test_batches:
            test_batch = test_batch.to(device if backend == "hooked" else "cpu")
            # Get LLM activations for test tokens
            _, cache = model.run_with_cache(
                test_batch if backend == "hooked" else test_batch.cpu(),
                names_filter=[hook_name],
                stop_at_layer=stop,
                prepend_bos=False,
            )
            acts = cache[hook_name]  # (B, S, D_IN)
            B, S, _ = acts.shape
            acts_flat = acts.reshape(-1, D_IN).to(device).float()

            # SAE encode
            feature_acts, _ = sae.encode_with_hidden_pre(acts_flat)  # (B*S, D_SAE)
            all_feature_acts.append(feature_acts.cpu())

    all_feature_acts_tensor = torch.cat(all_feature_acts, dim=0)  # (N_tokens, D_SAE)

    torch.save({
        "backend": backend,
        "feature_acts": all_feature_acts_tensor,
        "final_mse": final_loss,
    }, out_path)
    print(f"DONE:{backend} saved {all_feature_acts_tensor.shape}", flush=True)

    del model, store, sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- subprocess launcher ----

def launch(backend: str, args: argparse.Namespace,
           init_weights_path: str, test_tokens_path: str, out_path: str) -> dict:
    cmd = [
        sys.executable, __file__,
        "--_run-backend", backend,
        "--model-path", args.model_path,
        "--hooked-model-name", args.hooked_model_name,
        "--dataset-path", args.dataset_path,
        "--layer", str(args.layer),
        "--device", args.device,
        "--n-train-steps", str(args.n_train_steps),
        "--init-weights-path", init_weights_path,
        "--test-tokens-path", test_tokens_path,
        "--out-path", out_path,
    ]
    print(f"\n[{backend}] Training {args.n_train_steps} steps + feature eval ...", flush=True)
    proc = subprocess.run(cmd, check=False, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed (exit {proc.returncode})")
    return torch.load(out_path, map_location="cpu")


# ---- metrics ----

def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token Jaccard similarity of top-k feature sets."""
    # a, b: (N, D_SAE) sparse feature_acts; nonzero = fired
    assert a.shape == b.shape
    a_fired = a > 0
    b_fired = b > 0
    inter = (a_fired & b_fired).float().sum(dim=-1)  # (N,)
    union = (a_fired | b_fired).float().sum(dim=-1)   # (N,)
    jaccard_per_tok = inter / union.clamp(min=1)
    return jaccard_per_tok.mean().item()


def overlap_count(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token count of shared fired features."""
    inter = ((a > 0) & (b > 0)).float().sum(dim=-1)
    return inter.mean().item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def top_k_rank_corr(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    """
    For each token, take the union of top-k features from both a and b,
    and compute Spearman rank correlation of their values within that union.
    """
    n = a.shape[0]
    corrs = []
    for i in range(n):
        ai, bi = a[i], b[i]
        # union of fired features
        union_idx = (ai > 0) | (bi > 0)
        if union_idx.sum() < 2:
            continue
        idxs = union_idx.nonzero(as_tuple=True)[0]
        av = ai[idxs].numpy()
        bv = bi[idxs].numpy()
        from scipy.stats import spearmanr
        r, _ = spearmanr(av, bv)
        if not (r != r):  # NaN check
            corrs.append(r)
    return float(sum(corrs) / len(corrs)) if corrs else 0.0


# ---- report ----

def print_report(results: list[dict], k: int) -> None:
    print()
    print("=" * 70)
    print("  Feature Activation Comparison  (same tokens, different backends)")
    print(f"  d_sae={D_SAE} | k={K} | test tokens per backend: {results[0]['feature_acts'].shape[0]}")
    print("=" * 70)
    print(f"{'pair':<30} {'jaccard':>9} {'overlap/k':>10} {'cosine':>9} {'rank_r':>8}")
    print("-" * 70)

    names = [r["backend"] for r in results]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]["feature_acts"]
            b = results[j]["feature_acts"]
            jac = jaccard(a, b)
            ov = overlap_count(a, b)
            cos = cosine_sim(a, b)
            rr = top_k_rank_corr(a, b, k)
            pair = f"{names[i]} vs {names[j]}"
            print(f"{pair:<30} {jac:>9.4f} {ov/k:>10.4f} {cos:>9.4f} {rr:>8.4f}")

    print("-" * 70)
    print()
    print("Columns:")
    print("  jaccard   : |A∩B| / |A∪B| of fired feature sets per token (1.0 = identical)")
    print("  overlap/k : mean shared features / k (fraction of top-k in common)")
    print("  cosine    : cosine similarity of full D_SAE feature vectors per token")
    print("  rank_r    : Spearman rank corr of activation magnitudes in union set")
    print()
    print("Final training MSE per backend:")
    for r in results:
        print(f"  {r['backend']:<18} mse = {r['final_mse']:.2f}")


# ---- arg parsing ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-train-steps", type=int, default=200)
    p.add_argument("--n-test-batches", type=int, default=5)
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    # Internal
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--init-weights-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-tokens-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---- main ----

def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        _worker(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            dataset_path=args.dataset_path,
            layer=args.layer,
            hook=DEFAULT_HOOK,
            device=args.device,
            n_train_steps=args.n_train_steps,
            init_weights_path=args.init_weights_path,
            test_tokens_path=args.test_tokens_path,
            out_path=args.out_path,
        )
        return

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    print(f"Model:        {args.model_path}")
    print(f"Layer:        {args.layer}.{DEFAULT_HOOK}  |  SAE: d_sae={D_SAE}, k={K}")
    print(f"Train steps:  {args.n_train_steps}  |  Test batches: {args.n_test_batches}")

    # Build fixed test token batches from end of dataset (after training data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dataset = load_from_disk(args.dataset_path)
    for key in ("input_ids", "tokens", "token_ids"):
        if key in dataset.column_names:
            break
    # Use last N samples as test set to avoid overlap with training
    n_test = args.n_test_batches * STORE_BATCH_PROMPTS
    test_rows = list(dataset)[-n_test:]
    test_batches: list[torch.Tensor] = []
    for start in range(0, len(test_rows), STORE_BATCH_PROMPTS):
        chunk = test_rows[start : start + STORE_BATCH_PROMPTS]
        seqs = [torch.tensor(list(r[key]), dtype=torch.long)[:CONTEXT_SIZE] for r in chunk]
        ml = max(s.numel() for s in seqs)
        b = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            b[i, : s.numel()] = s
        test_batches.append(b)

    # Create shared SAE init weights
    from sae_lens.saes.sae import SAEMetadata
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

    hook_name = f"blocks.{args.layer}.{DEFAULT_HOOK}"
    sae_cfg = TopKTrainingSAEConfig(
        d_in=D_IN, d_sae=D_SAE, k=K, dtype="float32", device="cpu",
        metadata=SAEMetadata(hook_name=hook_name),
    )
    sae_init = TopKTrainingSAE(sae_cfg)
    print("Saving shared initial SAE weights and test tokens ...")

    with tempfile.TemporaryDirectory(prefix="feat_cmp_") as tmp:
        init_path = f"{tmp}/init.pt"
        test_tokens_path = f"{tmp}/test_tokens.pt"
        torch.save(sae_init.state_dict(), init_path)
        torch.save(test_batches, test_tokens_path)

        backends = ["vllm_tp1"]
        if not args.skip_tp2:
            backends.append("vllm_tp2")
        if not args.skip_hooked:
            backends.append("hooked")

        results = []
        for backend in backends:
            out_path = f"{tmp}/{backend}.pt"
            results.append(launch(backend, args, init_path, test_tokens_path, out_path))

    print_report(results, K)


if __name__ == "__main__":
    main()
