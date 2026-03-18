#!/usr/bin/env python
"""
SAE convergence validation: train on a single hook with one backend.

Single backend (typical usage):
    python scripts/validate_convergence.py \\
        --dataset-path ~/datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048 \\
        --hook hook_resid_post --layer 21 --backend vllm_tp1 \\
        --n-steps 5000 --save-every 100 --output-dir ./convergence_results

Run all backends sequentially (orchestrator):
    python scripts/validate_convergence.py \\
        --dataset-path ~/datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048 \\
        --hook hook_resid_post --layer 21 \\
        --n-steps 5000 --output-dir ./convergence_results

Plot from existing results (no training):
    python scripts/validate_convergence.py \\
        --dataset-path ~/datasets/... --hook hook_resid_post --layer 21 \\
        --plot-only --output-dir ./convergence_results

Output layout:
    {output_dir}/blocks.{layer}.{hook}/
        init_weights.pt       shared SAE init weights across backends
        vllm_tp1.json         written every --save-every steps
        vllm_tp2.json
        hooked.json
        mse_loss.csv
        explained_variance.csv
        l0.csv
        convergence_curves.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYER = 21
DEFAULT_HOOK = "hook_resid_post"
DEFAULT_N_STEPS = 5000
DEFAULT_TRAIN_BATCH_TOKENS = 8192
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_SAVE_EVERY = 100

STORE_BATCH_PROMPTS = 4
N_BATCHES_IN_BUFFER = 16
# HookedTransformer materializes the full (B, H, S, S) attention pattern.
# With B=8, H=32, S=2048 that's 4 GiB → OOM on a 44 GiB GPU.
# Reduce batch size and buffer for hooked to keep peak memory under budget.
HOOKED_STORE_BATCH_PROMPTS = 4   # attention pattern: 4×32×2048×2048×4B = 2 GiB
HOOKED_N_BATCHES_IN_BUFFER = 16  # activation buffer: 16×8192×4096×4B ≈ 2 GiB
D_SAE_MULT = 16
D_SAE_MAX = 65536 * 2
D_K_DIV = 256
D_K_MIN = 16

ALL_BACKENDS = ["vllm_tp1", "vllm_tp2", "hooked"]


# ---------------------------------------------------------------------------
# training worker (runs directly when --backend is given)
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    """Train SAE for one backend, saving every --save-every steps."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from datasets import load_from_disk
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from sae_lens.saes.sae import SAEMetadata, TrainStepInput
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
    from sae_lens.training.activations_store import ActivationsStore
    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

    backend = args.backend
    hook_name = f"blocks.{args.layer}.{args.hook}"
    run_dir = Path(args.output_dir) / hook_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{backend}.json"
    init_weights_path = run_dir / "init_weights.pt"
    log_path = run_dir / f"{backend}.log"

    # Tee stderr to log file (captures Python-level output; vLLM C logs go via
    # the subprocess pipe when launched by the orchestrator).
    class _Tee:
        def __init__(self, a: object, b: object) -> None:
            self._a, self._b = a, b
        def write(self, s: str) -> None:
            self._a.write(s)  # type: ignore[union-attr]
            self._b.write(s)  # type: ignore[union-attr]
        def flush(self) -> None:
            self._a.flush()  # type: ignore[union-attr]
            self._b.flush()  # type: ignore[union-attr]
        def fileno(self) -> int:
            return self._a.fileno()  # type: ignore[union-attr]

    log_file = open(log_path, "w")
    sys.stderr = _Tee(sys.__stderr__, log_file)  # type: ignore[assignment]

    # Build model
    if backend.startswith("vllm"):
        from sae_lens.vllm_model import HookedVLLMModel
        tp = 2 if backend == "vllm_tp2" else 1
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, use_fast=True
        )
        model = HookedVLLMModel(
            args.model_path, tokenizer,
            tensor_parallel_size=tp,
            max_model_len=args.context_size + 1,
            enable_prefix_caching=False,
        )
    else:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, use_fast=True
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        model = HookedTransformer.from_pretrained_no_processing(
            args.hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
            device=args.device, dtype=torch.bfloat16, local_files_only=True,
        )
        model.eval()

    dataset = load_from_disk(args.dataset_path)

    # Detect d_in
    stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)
    test_tok = torch.zeros(1, 2, dtype=torch.long)
    if backend == "hooked":
        test_tok = test_tok.to(args.device)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            test_tok, names_filter=[hook_name],
            stop_at_layer=stop, prepend_bos=False,
        )
    d_in = int(torch.prod(torch.tensor(cache[hook_name].shape[2:])).item())
    d_sae = min(D_SAE_MULT * d_in, D_SAE_MAX)
    k = max(D_K_MIN, d_sae // D_K_DIV)
    print(f"[{backend}] d_in={d_in}, d_sae={d_sae}, k={k}", file=sys.stderr)

    store_batch_prompts = HOOKED_STORE_BATCH_PROMPTS if backend == "hooked" else STORE_BATCH_PROMPTS
    n_batches_in_buffer = HOOKED_N_BATCHES_IN_BUFFER if backend == "hooked" else N_BATCHES_IN_BUFFER

    store = ActivationsStore(
        model=model,  # type: ignore[arg-type]
        dataset=dataset,
        streaming=False,
        hook_name=hook_name,
        hook_head_index=None,
        context_size=args.context_size,
        d_in=d_in,
        n_batches_in_buffer=n_batches_in_buffer,
        total_training_tokens=args.n_steps * args.train_batch_tokens * 10,
        store_batch_size_prompts=store_batch_prompts,
        train_batch_size_tokens=args.train_batch_tokens,
        prepend_bos=False,
        normalize_activations="none",
        device=torch.device(args.device),
        dtype="bfloat16",
        disable_concat_sequences=False,
        sequence_separator_token=None,
        activations_mixing_fraction=0.0,
    )

    sae = TopKTrainingSAE(TopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        dtype="float32", device=args.device,
        metadata=SAEMetadata(hook_name=hook_name),
    ))

    # Shared init weights: first backend saves, rest load.
    if init_weights_path.exists():
        sae.load_state_dict(torch.load(str(init_weights_path), map_location=args.device))
        print(f"[{backend}] Loaded init weights from {init_weights_path}", file=sys.stderr)
    else:
        torch.save(sae.state_dict(), str(init_weights_path))
        print(f"[{backend}] Saved init weights to {init_weights_path}", file=sys.stderr)
    sae.to(args.device)

    from torch.optim import Adam
    optimizer = Adam(sae.parameters(), lr=2e-4)

    meta = {"backend": backend, "d_in": d_in, "d_sae": d_sae, "k": k,
            "n_steps": args.n_steps, "train_batch_tokens": args.train_batch_tokens}

    records: list[dict] = []
    pbar = tqdm(total=args.n_steps, desc=backend, file=sys.stderr)

    for step in range(args.n_steps):
        batch = next(store).to(args.device)
        out = sae.training_forward_pass(
            TrainStepInput(
                sae_in=batch,
                coefficients=sae.get_coefficients(),
                dead_neuron_mask=torch.zeros(d_sae, dtype=torch.bool, device=args.device),
                n_training_steps=step,
                is_logging_step=False,
            )
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            var = batch.var(dim=0).mean().item()
            mse = out.losses["mse_loss"].item()
            ev = 1.0 - mse / max(var, 1e-8)
            l0 = (out.feature_acts > 0).float().sum(dim=-1).mean().item()

        records.append({"step": step, "mse_loss": mse, "explained_variance": ev, "l0": l0})
        pbar.set_postfix(mse=f"{mse:.4f}", ev=f"{ev*100:.1f}%", l0=f"{l0:.1f}")
        pbar.update(1)

        if (step + 1) % args.save_every == 0 or step == args.n_steps - 1:
            with open(out_path, "w") as f:
                json.dump({**meta, "records": records}, f)

    pbar.close()

    sae_path = run_dir / f"{backend}_sae.pt"
    torch.save(sae.state_dict(), str(sae_path))
    print(f"[{backend}] Done. Results at {out_path}, weights at {sae_path}", flush=True)
    log_file.close()
    sys.stderr = sys.__stderr__  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# orchestrator: launch each backend as a subprocess
# ---------------------------------------------------------------------------


def launch_all(args: argparse.Namespace) -> None:
    hook_name = f"blocks.{args.layer}.{args.hook}"
    run_dir = Path(args.output_dir) / hook_name

    print(f"Model:    {args.model_path}")
    print(f"Hook:     {hook_name}")
    print(f"Steps:    {args.n_steps}  |  Batch: {args.train_batch_tokens} tok"
          f"  |  Save every: {args.save_every}")
    print(f"Output:   {run_dir.resolve()}")
    print(f"Backends: {ALL_BACKENDS}\n")

    for backend in ALL_BACKENDS:
        cmd = [
            sys.executable, __file__,
            "--backend", backend,
            "--dataset-path", args.dataset_path,
            "--hook", args.hook,
            "--layer", str(args.layer),
            "--model-path", args.model_path,
            "--hooked-model-name", args.hooked_model_name,
            "--device", args.device,
            "--n-steps", str(args.n_steps),
            "--train-batch-tokens", str(args.train_batch_tokens),
            "--context-size", str(args.context_size),
            "--save-every", str(args.save_every),
            "--output-dir", args.output_dir,
        ]
        run_dir = Path(args.output_dir) / f"blocks.{args.layer}.{args.hook}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"{backend}.log"
        print(f"\n[{backend}] Starting {args.n_steps} steps ... (log: {log_path})", flush=True)
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            def _relay(src: object, log: object) -> None:
                for line in src:  # type: ignore[union-attr]
                    decoded = line.decode(errors="replace")
                    sys.stderr.write(decoded)
                    sys.stderr.flush()
                    log.write(decoded)  # type: ignore[union-attr]
                    log.flush()  # type: ignore[union-attr]
            t = threading.Thread(target=_relay, args=(proc.stderr, log_f), daemon=True)
            t.start()
            proc.wait()
            t.join()
        if proc.returncode != 0:
            print(f"[{backend}] FAILED (exit {proc.returncode}), continuing ...", flush=True)

    # Generate plots + CSVs from whatever results exist
    run_dir = Path(args.output_dir) / hook_name
    results = []
    for backend in ALL_BACKENDS:
        p = run_dir / f"{backend}.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
    if len(results) >= 2:
        print("\nGenerating summary ...")
        print_report(results)
        save_csv(results, run_dir)
        plot_curves(results, run_dir, hook_name)
    else:
        print("\nNot enough results to compare.")


# ---------------------------------------------------------------------------
# plot-only mode: regenerate plots from existing JSON files
# ---------------------------------------------------------------------------


def plot_only(args: argparse.Namespace) -> None:
    hook_name = f"blocks.{args.layer}.{args.hook}"
    run_dir = Path(args.output_dir) / hook_name
    results = []
    for backend in ALL_BACKENDS:
        p = run_dir / f"{backend}.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
            print(f"  Loaded {p}")
    if not results:
        print(f"No result files found in {run_dir}")
        return
    print_report(results)
    save_csv(results, run_dir)
    plot_curves(results, run_dir, hook_name)


# ---------------------------------------------------------------------------
# reporting & plotting
# ---------------------------------------------------------------------------


def print_report(results: list[dict]) -> None:
    print()
    print("=" * 70)
    print("  Convergence Validation — Final Metrics")
    print("=" * 70)
    print(f"  {'backend':<20} {'d_in':>6} {'d_sae':>7} {'k':>4} "
          f"{'final_mse':>10} {'final_ev%':>10} {'final_l0':>9}")
    print("-" * 70)
    for r in results:
        last = r["records"][-1]
        print(f"  {r['backend']:<20} {r['d_in']:>6} {r['d_sae']:>7} {r['k']:>4} "
              f"{last['mse_loss']:>10.4f} {last['explained_variance']*100:>9.2f}% "
              f"{last['l0']:>9.1f}")
    print("-" * 70)
    print("\nPairwise final Δmse:")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            delta = abs(a["records"][-1]["mse_loss"] - b["records"][-1]["mse_loss"])
            pct = delta / max(a["records"][-1]["mse_loss"], 1e-8) * 100
            print(f"  {a['backend']} vs {b['backend']}: Δmse={delta:.4f} ({pct:.2f}%)")
    print()


def save_csv(results: list[dict], out_dir: Path) -> None:
    for metric in ["mse_loss", "explained_variance", "l0"]:
        csv_path = out_dir / f"{metric}.csv"
        by_backend = {r["backend"]: {rec["step"]: rec[metric] for rec in r["records"]}
                      for r in results}
        backends = [r["backend"] for r in results]
        all_steps = sorted({s for b in by_backend.values() for s in b})
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + backends)
            for step in all_steps:
                writer.writerow([step] + [by_backend[b].get(step, "") for b in backends])
        print(f"  Saved {csv_path}")


def plot_curves(results: list[dict], out_dir: Path, hook_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    metrics = [
        ("mse_loss", "MSE Loss", "lower is better"),
        ("explained_variance", "Explained Variance", "higher is better"),
        ("l0", "L0 (avg active features)", ""),
    ]
    colors = {"vllm_tp1": "#1f77b4", "vllm_tp2": "#ff7f0e", "hooked": "#2ca02c"}
    labels = {"vllm_tp1": "vLLM TP=1", "vllm_tp2": "vLLM TP=2", "hooked": "HookedTransformer"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        for r in results:
            b = r["backend"]
            ax.plot([rec["step"] for rec in r["records"]],
                    [rec[metric] for rec in r["records"]],
                    label=labels.get(b, b), color=colors.get(b), linewidth=1.5, alpha=0.85)
        ax.set_title(f"{title}\n({subtitle})" if subtitle else title)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    r0 = results[0]
    fig.suptitle(
        f"SAE Convergence — {hook_name}\n"
        f"d_in={r0['d_in']}  d_sae={r0['d_sae']}  k={r0['k']}  "
        f"batch={r0['train_batch_tokens']} tok",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = out_dir / "convergence_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# arg parsing & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--hook", default=DEFAULT_HOOK)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--train-batch-tokens", type=int, default=DEFAULT_TRAIN_BATCH_TOKENS)
    p.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--output-dir", default="./convergence_results")
    p.add_argument("--backend", choices=ALL_BACKENDS, default=None,
                   help="Single backend to run. Omit to run all sequentially.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip training; regenerate plots from existing JSON files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        plot_only(args)
    elif args.backend is not None:
        run_training(args)
    else:
        launch_all(args)


if __name__ == "__main__":
    main()
