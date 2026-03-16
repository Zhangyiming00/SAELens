#!/usr/bin/env python
"""
Train SAE on each backend for all supported hook points, then compare which
features activate on the same held-out token sequences.

Each backend runs in its own subprocess (model loaded once, all hooks tested
sequentially).  For each hook the worker:
  1. Auto-detects d_in via a tiny forward pass.
  2. Creates an ActivationsStore and TopK SAE sized to that d_in.
  3. Trains for --n-train-steps steps from the same shared initial weights.
  4. Encodes fixed test tokens and saves feature_acts.

The main process compares backends pairwise per hook type and reports:
  - mean Jaccard similarity of top-k feature sets per token position
  - mean overlap/k (fraction of top-k features in common)
  - mean cosine similarity of feature activation vectors
  - Spearman rank correlation of activation magnitudes (union of fired features)

Usage:
    python scripts/compare_feature_activation.py \\
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \\
        --n-train-steps 200 --n-test-batches 5

    # Test only specific hooks:
    python scripts/compare_feature_activation.py \\
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \\
        --hooks hook_resid_post attn.hook_q mlp.hook_post
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYER = 21

ALL_HOOK_TYPES = [
    "hook_resid_pre",
    "hook_resid_mid",
    "hook_resid_post",
    "hook_attn_out",
    "hook_mlp_out",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_z",
    "mlp.hook_pre",
    "mlp.hook_post",
]

CONTEXT_SIZE = 128
STORE_BATCH_PROMPTS = 4
N_BATCHES_IN_BUFFER = 32
TRAIN_BATCH_TOKENS = 2048

D_SAE_MULT = 4
D_SAE_MAX = 32768
D_K_DIV = 64
D_K_MIN = 10


# ---------------------------------------------------------------------------
# subprocess worker
# ---------------------------------------------------------------------------


def _run_all_hooks(
    backend: str,
    model_path: str,
    hooked_model_name: str,
    dataset_path: str,
    layer: int,
    device: str,
    n_train_steps: int,
    init_weights_dir: str,
    test_tokens_path: str,
    hook_types: list[str],
    out_path: str,
) -> None:
    """Train SAE + eval feature activations for each hook; save results as .pt."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from datasets import load_from_disk
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from sae_lens.saes.sae import SAEMetadata, TrainStepInput
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
    from sae_lens.training.activations_store import ActivationsStore
    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

    effective_backend = "hooked" if backend in ("hooked_1", "hooked_2") else backend

    if effective_backend.startswith("vllm"):
        from sae_lens.vllm_model import HookedVLLMModel
        tp = 2 if effective_backend == "vllm_tp2" else 1
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        model = HookedVLLMModel(
            model_path, tokenizer,
            tensor_parallel_size=tp,
            max_model_len=CONTEXT_SIZE + 1,
            enable_prefix_caching=False,
        )
    else:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
        model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
            device=device, dtype=torch.bfloat16, local_files_only=True,
        )
        model.eval()

    dataset = load_from_disk(dataset_path)
    test_batches: list[torch.Tensor] = torch.load(test_tokens_path, map_location="cpu")

    results: dict[str, dict] = {}

    for hook_type in hook_types:
        hook_name = f"blocks.{layer}.{hook_type}"
        print(f"\n[{backend}] Testing {hook_name} ...", file=sys.stderr)

        # --- Auto-detect d_in ---
        try:
            stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)
            test_tok = torch.zeros(1, 2, dtype=torch.long)
            if effective_backend == "hooked":
                test_tok = test_tok.to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    test_tok, names_filter=[hook_name],
                    stop_at_layer=stop, prepend_bos=False,
                )
            act = cache[hook_name]
            d_in = int(torch.prod(torch.tensor(act.shape[2:])).item())
        except Exception as e:
            print(f"[{backend}] {hook_name}: d_in detection failed: {e}", file=sys.stderr)
            results[hook_type] = {"error": str(e), "d_in": -1}
            continue

        d_sae = min(D_SAE_MULT * d_in, D_SAE_MAX)
        k = max(D_K_MIN, d_sae // D_K_DIV)
        print(f"[{backend}] {hook_type}: d_in={d_in}, d_sae={d_sae}, k={k}", file=sys.stderr)

        # --- ActivationsStore ---
        try:
            store = ActivationsStore(
                model=model,  # type: ignore[arg-type]
                dataset=dataset,
                streaming=False,
                hook_name=hook_name,
                hook_head_index=None,
                context_size=CONTEXT_SIZE,
                d_in=d_in,
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
        except Exception as e:
            print(f"[{backend}] {hook_name}: ActivationsStore failed: {e}", file=sys.stderr)
            results[hook_type] = {"error": str(e), "d_in": d_in}
            continue

        # --- SAE: load or create shared initial weights ---
        init_filename = f"{hook_type.replace('.', '_')}_d{d_in}_s{d_sae}.pt"
        init_path = os.path.join(init_weights_dir, init_filename)
        sae_cfg = TopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            dtype="float32", device=device,
            metadata=SAEMetadata(hook_name=hook_name),
        )
        sae = TopKTrainingSAE(sae_cfg)
        if os.path.exists(init_path):
            sae.load_state_dict(torch.load(init_path, map_location=device))
        else:
            torch.save(sae.state_dict(), init_path)
        sae.to(device)

        from torch.optim import Adam
        optimizer = Adam(sae.parameters(), lr=1e-4)

        # --- Training ---
        pbar = tqdm(total=n_train_steps, desc=f"{backend}/{hook_type}", file=sys.stderr)
        final_mse = 0.0
        try:
            for step in range(n_train_steps):
                batch = next(store).to(device)
                out = sae.training_forward_pass(
                    TrainStepInput(
                        sae_in=batch,
                        coefficients=sae.get_coefficients(),
                        dead_neuron_mask=torch.zeros(d_sae, dtype=torch.bool, device=device),
                        n_training_steps=step,
                        is_logging_step=False,
                    )
                )
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
            final_mse = out.losses["mse_loss"].item()
        except Exception as e:
            pbar.close()
            print(f"[{backend}] {hook_name}: training failed: {e}", file=sys.stderr)
            results[hook_type] = {"error": str(e), "d_in": d_in, "d_sae": d_sae, "k": k}
            del store, sae, optimizer
            gc.collect()
            continue
        pbar.close()
        print(f"[{backend}] {hook_type}: final mse={final_mse:.4f}", file=sys.stderr)

        # --- Eval: encode fixed test tokens ---
        stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)
        all_feature_acts: list[torch.Tensor] = []
        sae.eval()
        try:
            with torch.no_grad():
                for test_batch in test_batches:
                    if effective_backend == "hooked":
                        test_batch = test_batch.to(device)
                    _, cache = model.run_with_cache(
                        test_batch,
                        names_filter=[hook_name],
                        stop_at_layer=stop,
                        prepend_bos=False,
                    )
                    acts = cache[hook_name]  # (B, S, d_in)
                    acts_flat = acts.reshape(-1, d_in).to(device).float()
                    feature_acts, _ = sae.encode_with_hidden_pre(acts_flat)
                    all_feature_acts.append(feature_acts.cpu())
        except Exception as e:
            print(f"[{backend}] {hook_name}: eval failed: {e}", file=sys.stderr)
            results[hook_type] = {"error": str(e), "d_in": d_in, "d_sae": d_sae, "k": k,
                                  "final_mse": final_mse}
            del store, sae, optimizer
            gc.collect()
            continue

        feature_acts_tensor = torch.cat(all_feature_acts, dim=0)  # (N_tok, d_sae)
        results[hook_type] = {
            "d_in": d_in, "d_sae": d_sae, "k": k,
            "final_mse": final_mse,
            "feature_acts": feature_acts_tensor,
        }
        print(f"[{backend}] {hook_type}: encoded {feature_acts_tensor.shape[0]} tokens",
              file=sys.stderr)

        del store, sae, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save({"backend": backend, "hooks": results}, out_path)
    print(f"DONE:{backend}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# subprocess launcher
# ---------------------------------------------------------------------------


def launch(
    backend: str,
    args: argparse.Namespace,
    init_weights_dir: str,
    test_tokens_path: str,
    hook_types: list[str],
    out_path: str,
) -> dict:
    cmd = [
        sys.executable, __file__,
        "--_run-backend", backend,
        "--model-path", args.model_path,
        "--hooked-model-name", args.hooked_model_name,
        "--dataset-path", args.dataset_path,
        "--layer", str(args.layer),
        "--device", args.device,
        "--n-train-steps", str(args.n_train_steps),
        "--init-weights-dir", init_weights_dir,
        "--test-tokens-path", test_tokens_path,
        "--hook-types", *hook_types,
        "--out-path", out_path,
    ]
    print(f"\n[{backend}] Training {args.n_train_steps} steps on {len(hook_types)} hooks ...",
          flush=True)
    proc = subprocess.run(cmd, check=False, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed (exit {proc.returncode})")
    return torch.load(out_path, map_location="cpu")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token Jaccard similarity of top-k feature sets."""
    a_fired = a > 0
    b_fired = b > 0
    inter = (a_fired & b_fired).float().sum(dim=-1)
    union = (a_fired | b_fired).float().sum(dim=-1)
    return (inter / union.clamp(min=1)).mean().item()


def overlap_count(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token count of shared fired features."""
    return ((a > 0) & (b > 0)).float().sum(dim=-1).mean().item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def top_k_rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of activation magnitudes in union of fired features."""
    from scipy.stats import spearmanr
    corrs = []
    for ai, bi in zip(a, b):
        union_idx = ((ai > 0) | (bi > 0)).nonzero(as_tuple=True)[0]
        if union_idx.numel() < 2:
            continue
        r, _ = spearmanr(ai[union_idx].numpy(), bi[union_idx].numpy())
        if r == r:  # NaN check
            corrs.append(r)
    return float(sum(corrs) / len(corrs)) if corrs else 0.0


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_report(results: list[dict], n_train_steps: int) -> None:
    backends = [r["backend"] for r in results]

    print()
    print("=" * 110)
    print("  Feature Activation Comparison  —  all hook types")
    print(f"  n_train_steps={n_train_steps} | context={CONTEXT_SIZE} | train_batch={TRAIN_BATCH_TOKENS}")
    print("=" * 110)

    all_hooks: list[str] = []
    for r in results:
        for h in r["hooks"]:
            if h not in all_hooks:
                all_hooks.append(h)

    for hook_type in all_hooks:
        hook_results = [r["hooks"].get(hook_type, {}) for r in results]

        # Skip if any backend has an error or is missing
        if any("error" in h or not h for h in hook_results):
            errors = [(backends[i], hook_results[i].get("error", "SKIP"))
                      for i in range(len(backends))
                      if "error" in hook_results[i] or not hook_results[i]]
            print(f"\n  {hook_type}: SKIPPED — {errors}")
            continue

        # Check d_in consistency
        d_ins = [h["d_in"] for h in hook_results]
        d_sae = hook_results[0]["d_sae"]
        k = hook_results[0]["k"]
        mismatch = len(set(d_ins)) > 1

        mse_str = "  ".join(
            f"{backends[i]}={hook_results[i]['final_mse']:.4f}"
            for i in range(len(backends))
        )
        print(f"\n  {hook_type}  [d_in={d_ins[0]}{' ← MISMATCH' if mismatch else ''}  "
              f"d_sae={d_sae}  k={k}]")
        print(f"    MSE: {mse_str}")

        if mismatch:
            print(f"    (d_in differs across backends — skipping pairwise comparison)")
            continue

        print(f"    {'pair':<26} {'jaccard':>9} {'overlap/k':>10} {'cosine':>9} {'rank_r':>8}")
        print(f"    {'-'*66}")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a = hook_results[i]["feature_acts"]
                b = hook_results[j]["feature_acts"]
                jac = jaccard(a, b)
                ov = overlap_count(a, b)
                cos = cosine_sim(a, b)
                rr = top_k_rank_corr(a, b)
                pair = f"{backends[i]} vs {backends[j]}"
                print(f"    {pair:<26} {jac:>9.4f} {ov/k:>10.4f} {cos:>9.4f} {rr:>8.4f}")

    print()
    print("=" * 110)
    print()
    print("Columns:")
    print("  jaccard   : |A∩B| / |A∪B| of fired feature sets per token (1.0 = identical)")
    print("  overlap/k : mean shared features / k")
    print("  cosine    : cosine similarity of full d_sae feature vectors per token")
    print("  rank_r    : Spearman rank corr of activation magnitudes in union set")
    print()
    print("Note: mlp.hook_pre d_in differs between vLLM (gate+up merged) "
          "and HookedTransformer (gate only) — pairwise comparison skipped for that hook.")


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-train-steps", type=int, default=200)
    p.add_argument("--n-test-batches", type=int, default=5)
    p.add_argument("--hooks", nargs="+", default=["all"],
                   help="Hook types to test. Use 'all' for all hooks, "
                        "or list them e.g.: hook_resid_post attn.hook_q")
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    p.add_argument("--hooked-twice", action="store_true",
                   help="Run HookedTransformer twice (hooked_1 vs hooked_2) as a same-backend baseline.")
    # Internal
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--init-weights-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-tokens-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--hook-types", nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        _run_all_hooks(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            dataset_path=args.dataset_path,
            layer=args.layer,
            device=args.device,
            n_train_steps=args.n_train_steps,
            init_weights_dir=args.init_weights_dir,
            test_tokens_path=args.test_tokens_path,
            hook_types=args.hook_types,
            out_path=args.out_path,
        )
        return

    hook_types = ALL_HOOK_TYPES if args.hooks == ["all"] else args.hooks

    print(f"Model:       {args.model_path}")
    print(f"Layer:       {args.layer}")
    print(f"Train steps: {args.n_train_steps}  |  Test batches: {args.n_test_batches}")
    print(f"Hooks:       {hook_types}")

    # Build fixed test token batches (last N samples — no overlap with training)
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dataset = load_from_disk(args.dataset_path)
    for key in ("input_ids", "tokens", "token_ids"):
        if key in dataset.column_names:
            break

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

    backends = ["vllm_tp1"]
    if not args.skip_tp2:
        backends.append("vllm_tp2")
    if not args.skip_hooked:
        backends.append("hooked")

    if args.hooked_twice:
        backends = ["hooked_1", "hooked_2"]

    with tempfile.TemporaryDirectory(prefix="feat_cmp_") as tmp:
        test_tokens_path = f"{tmp}/test_tokens.pt"
        torch.save(test_batches, test_tokens_path)
        print(f"Saved {len(test_batches)} test batches.")

        results = []
        for backend in backends:
            out_path = f"{tmp}/{backend}.pt"
            results.append(launch(backend, args, tmp, test_tokens_path, hook_types, out_path))

    print_report(results, args.n_train_steps)


if __name__ == "__main__":
    main()
