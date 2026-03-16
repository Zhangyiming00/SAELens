#!/usr/bin/env python
"""
Compare SAE training accuracy across three backends for ALL supported hook points:
  - HookedVLLMModel TP=1
  - HookedVLLMModel TP=2
  - HookedTransformer

Each backend runs in its own subprocess (one subprocess per backend, testing all
hooks sequentially so the model is loaded only once per backend).

For each hook the worker:
  1. Auto-detects d_in by running a tiny forward pass.
  2. Creates an ActivationsStore and TopK SAE sized to that d_in.
  3. Trains for --n-steps steps starting from the same shared initial weights.
  4. Reports final mse_loss and explained_variance.

The main process compares backends pairwise (Δmse, Δev) and flags hooks where
backends produce activations with incompatible d_in (e.g., mlp.hook_pre differs
between vLLM and HookedTransformer due to merged vs split gate/up projections).

Usage:
    python scripts/benchmark_training_accuracy.py \\
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \\
        --n-steps 50

    # Test only specific hooks:
    python scripts/benchmark_training_accuracy.py \\
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \\
        --hooks hook_resid_post attn.hook_q mlp.hook_post
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

# All hook types we test by default (hook_embed excluded: it is global, not per-layer)
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

# Pipeline config (kept modest so the benchmark finishes quickly)
CONTEXT_SIZE = 128
STORE_BATCH_PROMPTS = 4
N_BATCHES_IN_BUFFER = 32
TRAIN_BATCH_TOKENS = 2048

# SAE sizing: d_sae = D_SAE_MULT * d_in, capped at D_SAE_MAX.
# K = max(D_K_MIN, d_sae // D_K_DIV).
D_SAE_MULT = 4
D_SAE_MAX = 32768
D_K_DIV = 64
D_K_MIN = 10


# ---------------------------------------------------------------------------
# subprocess worker
# ---------------------------------------------------------------------------


def _train_all_hooks(
    backend: str,
    model_path: str,
    hooked_model_name: str,
    dataset_path: str,
    layer: int,
    device: str,
    n_steps: int,
    init_weights_dir: str,   # directory containing per-hook init weights
    hook_types: list[str],   # hook types to test
    out_path: str,
) -> None:
    """Train SAE for each hook in one subprocess; save per-hook metrics as JSON."""
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

    # Build model once.
    if backend.startswith("vllm"):
        from sae_lens.vllm_model import HookedVLLMModel
        tp = 2 if backend == "vllm_tp2" else 1
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

    results: dict[str, dict] = {}

    for hook_type in hook_types:
        hook_name = f"blocks.{layer}.{hook_type}"
        print(f"\n[{backend}] Testing {hook_name} ...", file=sys.stderr)

        # --- Auto-detect d_in ---
        try:
            stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)
            test_tok = torch.zeros(1, 2, dtype=torch.long)
            if backend == "hooked":
                test_tok = test_tok.to(device)
            with torch.no_grad() if backend == "hooked" else torch.no_grad():
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
        except Exception as e:
            print(f"[{backend}] {hook_name}: ActivationsStore failed: {e}", file=sys.stderr)
            results[hook_type] = {"error": str(e), "d_in": d_in}
            continue

        # --- SAE: load or create shared initial weights ---
        # Filename encodes d_in and d_sae so backends with different d_in
        # (e.g. mlp.hook_pre: vLLM=28672 vs TL=14336) use separate inits.
        init_filename = f"{hook_type.replace('.', '_')}_d{d_in}_s{d_sae}.pt"
        init_path = os.path.join(init_weights_dir, init_filename)
        sae_cfg = TopKTrainingSAEConfig(
            d_in=d_in, d_sae=d_sae, k=k,
            dtype="float32", device=device,
            metadata=SAEMetadata(hook_name=hook_name),
        )
        sae = TopKTrainingSAE(sae_cfg)
        if os.path.exists(init_path):
            # Another backend already created this init file → load for fair comparison
            sae.load_state_dict(torch.load(init_path, map_location=device))
        else:
            # First backend to reach this hook → save init so others can load it
            torch.save(sae.state_dict(), init_path)
        sae.to(device)

        from torch.optim import Adam
        optimizer = Adam(sae.parameters(), lr=1e-4)

        # --- Training ---
        records: list[dict] = []
        pbar = tqdm(total=n_steps, desc=f"{backend}/{hook_type}", file=sys.stderr)
        try:
            for step in range(n_steps):
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

                with torch.no_grad():
                    var = batch.var(dim=0).sum().item()
                    mse = out.losses["mse_loss"].item()
                    ev = 1.0 - mse / max(var / d_in, 1e-8)
                    aux = out.losses.get(
                        "auxiliary_reconstruction_loss", torch.tensor(0.0)
                    ).item()

                records.append({
                    "step": step, "mse_loss": mse,
                    "explained_variance": ev, "aux_loss": aux,
                })
                pbar.update(1)
        except Exception as e:
            pbar.close()
            print(f"[{backend}] {hook_name}: training failed at step {len(records)}: {e}",
                  file=sys.stderr)
            results[hook_type] = {"error": str(e), "d_in": d_in, "d_sae": d_sae, "k": k}
            del store, sae, optimizer
            gc.collect()
            continue
        pbar.close()

        results[hook_type] = {
            "d_in": d_in, "d_sae": d_sae, "k": k,
            "final_mse": records[-1]["mse_loss"],
            "final_ev": records[-1]["explained_variance"],
            "final_aux": records[-1]["aux_loss"],
            "records": records,
        }
        print(
            f"[{backend}] {hook_type}: final mse={records[-1]['mse_loss']:.4f} "
            f"ev={records[-1]['explained_variance']*100:.2f}%",
            file=sys.stderr,
        )

        del store, sae, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    with open(out_path, "w") as f:
        json.dump({"backend": backend, "hooks": results}, f)
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
        "--n-steps", str(args.n_steps),
        "--init-weights-dir", init_weights_dir,
        "--hook-types", *hook_types,
        "--out-path", out_path,
    ]
    print(f"\n[{backend}] Training {args.n_steps} steps on {len(hook_types)} hooks ...",
          flush=True)
    proc = subprocess.run(cmd, check=False, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed (exit {proc.returncode})")
    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_report(results: list[dict], n_steps: int) -> None:
    backends = [r["backend"] for r in results]

    print()
    print("=" * 100)
    print("  SAE Training Accuracy  —  all hook types")
    print(f"  context={CONTEXT_SIZE} | train_batch={TRAIN_BATCH_TOKENS} tok "
          f"| n_steps={n_steps}")
    print("=" * 100)

    # Collect all hook types that at least one backend ran successfully
    all_hooks: list[str] = []
    for r in results:
        for h in r["hooks"]:
            if h not in all_hooks:
                all_hooks.append(h)

    # Header
    bw = 10
    print(f"\n{'hook':<22} {'d_in':>6}", end="")
    for b in backends:
        print(f"  {b+' mse':>{bw+4}}  {b+' ev%':>{bw-1}}", end="")
    print()
    print("-" * 100)

    for hook_type in all_hooks:
        # Collect per-backend info
        infos: list[dict] = []
        for r in results:
            infos.append(r["hooks"].get(hook_type, {}))

        # Check d_in consistency
        d_ins = [i.get("d_in", -1) for i in infos if "error" not in i]
        d_in_str = str(d_ins[0]) if d_ins else "?"
        mismatch = len(set(d_ins)) > 1

        print(f"  {hook_type:<20} {d_in_str:>6}", end="")
        for info in infos:
            if "error" in info:
                print(f"  {'ERROR':>{bw+4}}  {'---':>{bw-1}}", end="")
            elif not info:
                print(f"  {'SKIP':>{bw+4}}  {'---':>{bw-1}}", end="")
            else:
                mse = info["final_mse"]
                ev = info["final_ev"] * 100
                print(f"  {mse:>{bw+4}.4f}  {ev:>{bw-1}.2f}", end="")
        if mismatch:
            print("  ← d_in mismatch!", end="")
        print()

    print("-" * 100)

    # Pairwise Δmse summary
    print("\nPairwise final Δmse (should be ~0 if activations match):")
    for hook_type in all_hooks:
        infos = [r["hooks"].get(hook_type, {}) for r in results]
        valid = [(backends[i], infos[i]) for i in range(len(backends))
                 if "error" not in infos[i] and infos[i]]
        if len(valid) < 2:
            continue
        d_ins = [v[1].get("d_in", -1) for v in valid]
        if len(set(d_ins)) > 1:
            # Different d_in: skip pairwise comparison
            print(f"  {hook_type:<22}  (d_in mismatch across backends, skipped)")
            continue
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                bname_i, info_i = valid[i]
                bname_j, info_j = valid[j]
                a = info_i["final_mse"]
                b = info_j["final_mse"]
                delta = abs(a - b)
                pct = delta / max(a, 1e-8) * 100
                print(f"  {hook_type:<22}  {bname_i} vs {bname_j}: "
                      f"Δmse={delta:.4f} ({pct:.2f}%)")

    print()
    print("Columns: mse = mean-squared reconstruction error | ev% = explained variance %")
    print("Note: mlp.hook_pre d_in differs between vLLM (gate+up merged) "
          "and HookedTransformer (gate only)")


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
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--hooks", nargs="+", default=["all"],
                   help="Hook types to test. Use 'all' for all hooks, "
                        "or list them e.g.: hook_resid_post attn.hook_q")
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    # Internal
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--init-weights-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--hook-types", nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        _train_all_hooks(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            dataset_path=args.dataset_path,
            layer=args.layer,
            device=args.device,
            n_steps=args.n_steps,
            init_weights_dir=args.init_weights_dir,
            hook_types=args.hook_types,
            out_path=args.out_path,
        )
        return

    # Resolve hook types
    if args.hooks == ["all"]:
        hook_types = ALL_HOOK_TYPES
    else:
        hook_types = args.hooks

    print(f"Model:  {args.model_path}")
    print(f"Layer:  {args.layer}")
    print(f"Steps:  {args.n_steps}  |  Hooks: {hook_types}")
    print()
    print("Init weights: first backend to run each hook creates them; "
          "later backends load them for a fair comparison.")
    print("Hooks where d_in differs across backends use separate inits "
          "(mlp.hook_pre: vLLM=gate+up merged, TL=gate only).")

    backends = ["vllm_tp1"]
    if not args.skip_tp2:
        backends.append("vllm_tp2")
    if not args.skip_hooked:
        backends.append("hooked")

    with tempfile.TemporaryDirectory(prefix="sae_acc_bench_") as tmp:
        # init_weights_dir: workers will create/read per-hook .pt files here.
        init_weights_dir = tmp

        results = []
        for backend in backends:
            out_path = f"{tmp}/{backend}.json"
            results.append(launch(backend, args, init_weights_dir, hook_types, out_path))

    print_report(results, args.n_steps)


if __name__ == "__main__":
    main()
