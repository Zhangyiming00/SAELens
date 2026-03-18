#!/usr/bin/env python
"""
Parametric benchmark for activation generation (prefill + hook capture only).

Sweeps batch_size × context_size for each backend:
  - vLLM TP=1
  - vLLM TP=2
  - HookedTransformer

Metrics per (backend, batch_size, context_size, layer):
  - tokens/s  (throughput)
  - s/batch   (latency)
  - peak_gpu_mb (GPU memory)

Results saved to CSV and printed as tables.

Usage:
    python scripts/benchmark_parametric.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \
        [--model-path /data/models/Llama-3.1-8B] \
        [--skip-tp2] [--skip-hooked]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Sweep dimensions
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
CONTEXT_SIZES = [128, 512, 1024, 2048]
LAYERS = [0, 15, 31]
HOOK = "hook_resid_post"

NUM_WARMUP = 2
NUM_MEASURE = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batches(
    dataset_path: str,
    num_batches: int,
    batch_size: int,
    context_size: int,
    pad_id: int,
) -> list[torch.Tensor]:
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    for key in ("input_ids", "tokens", "token_ids"):
        if key in dataset.column_names:
            break
    else:
        raise ValueError(f"No token column found in {dataset_path}")

    batches: list[torch.Tensor] = []
    cur: list[torch.Tensor] = []
    for row in dataset:
        ids = torch.tensor(list(row[key]), dtype=torch.long)[:context_size]
        # Pad to context_size so every sequence is the same length
        if ids.numel() < context_size:
            pad = torch.full((context_size - ids.numel(),), pad_id, dtype=torch.long)
            ids = torch.cat([ids, pad])
        cur.append(ids)
        if len(cur) == batch_size:
            batches.append(torch.stack(cur))  # (B, S)
            cur = []
            if len(batches) >= num_batches:
                break

    # Repeat if dataset too small
    while len(batches) < num_batches:
        batches = (batches * ((num_batches // len(batches)) + 2))[:num_batches]

    return batches


# ---------------------------------------------------------------------------
# Subprocess worker: one backend, all (batch_size, context_size, layer) combos
# ---------------------------------------------------------------------------

def _run_backend(
    backend: str,
    model_path: str,
    hooked_model_name: str,
    dataset_path: str,
    batch_sizes: list[int],
    context_sizes: list[int],
    layers: list[int],
    hook: str,
    num_warmup: int,
    num_measure: int,
    out_path: str,
) -> None:
    """Run inside subprocess. Writes JSON results to out_path."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from transformers import AutoTokenizer

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Build model once
    if backend.startswith("vllm"):
        from sae_lens.vllm_model import HookedVLLMModel
        tp = 2 if backend == "vllm_tp2" else 1
        # Use the largest context_size so we only need one model instance
        max_ctx = max(context_sizes)
        model = HookedVLLMModel(
            model_path, tokenizer,
            tensor_parallel_size=tp,
            max_model_len=max_ctx + 1,
            enable_prefix_caching=False,
        )
    else:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
            device="cuda", dtype=torch.bfloat16, local_files_only=True,
        )
        model.eval()
        hf_model = None  # let GC collect it

    results: list[dict] = []
    total_configs = len(batch_sizes) * len(context_sizes) * len(layers)
    done = 0

    for layer in layers:
        hook_name = f"blocks.{layer}.{hook}"
        stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)

        for context_size in context_sizes:
            for batch_size in batch_sizes:
                total_batches = num_warmup + num_measure
                batches = load_batches(dataset_path, total_batches, batch_size, context_size, pad_id)

                # Reset peak memory counter
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                try:
                    if backend.startswith("vllm"):
                        # Warmup
                        for b in batches[:num_warmup]:
                            model.run_with_cache(b, names_filter=[hook_name],
                                                 stop_at_layer=stop, prepend_bos=False)
                        # Measure
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for b in batches[num_warmup:]:
                            model.run_with_cache(b, names_filter=[hook_name],
                                                 stop_at_layer=stop, prepend_bos=False)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t0
                    else:
                        # Warmup
                        for b in batches[:num_warmup]:
                            with torch.inference_mode():
                                model.run_with_cache(
                                    b.cuda(), names_filter=[hook_name],
                                    stop_at_layer=stop, prepend_bos=False,
                                    return_cache_object=False,
                                )
                        # Measure
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for b in batches[num_warmup:]:
                            with torch.inference_mode():
                                model.run_with_cache(
                                    b.cuda(), names_filter=[hook_name],
                                    stop_at_layer=stop, prepend_bos=False,
                                    return_cache_object=False,
                                )
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t0

                    total_tokens = num_measure * batch_size * context_size
                    tps = total_tokens / elapsed
                    lat = elapsed / num_measure
                    peak_mb = (
                        torch.cuda.max_memory_allocated() / (1024 ** 2)
                        if torch.cuda.is_available() else 0.0
                    )

                    results.append({
                        "backend": backend,
                        "layer": layer,
                        "batch_size": batch_size,
                        "context_size": context_size,
                        "tokens_per_s": tps,
                        "s_per_batch": lat,
                        "peak_gpu_mb": peak_mb,
                    })

                    done += 1
                    print(
                        f"[{backend}] layer={layer:2d} bs={batch_size:3d} ctx={context_size:5d} "
                        f"→ {tps:>10.0f} tok/s  {lat:.3f} s/batch  {peak_mb:.0f} MB "
                        f"({done}/{total_configs})",
                        flush=True,
                    )

                except Exception as e:
                    print(f"[{backend}] ERROR layer={layer} bs={batch_size} ctx={context_size}: {e}", flush=True)
                    results.append({
                        "backend": backend,
                        "layer": layer,
                        "batch_size": batch_size,
                        "context_size": context_size,
                        "tokens_per_s": float("nan"),
                        "s_per_batch": float("nan"),
                        "peak_gpu_mb": float("nan"),
                        "error": str(e),
                    })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(results, f)


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def launch_subprocess(backend: str, args: argparse.Namespace, out_path: str) -> list[dict]:
    cmd = [
        sys.executable, __file__,
        "--_run-backend", backend,
        "--model-path", args.model_path,
        "--hooked-model-name", args.hooked_model_name,
        "--dataset-path", args.dataset_path,
        "--batch-sizes", ",".join(str(x) for x in args.batch_sizes),
        "--context-sizes", ",".join(str(x) for x in args.context_sizes),
        "--layers", ",".join(str(x) for x in args.layers),
        "--hook", args.hook,
        "--num-warmup", str(args.num_warmup),
        "--num-measure", str(args.num_measure),
        "--out-path", out_path,
    ]
    print(f"\n[{backend}] Launching subprocess ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{backend} subprocess failed (exit {result.returncode})")
    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(
    results: list[dict],
    backend: str,
    batch_sizes: list[int],
    context_sizes: list[int],
    layer: int,
    metric: str,
    label: str,
) -> None:
    rows = [r for r in results if r["backend"] == backend and r["layer"] == layer]
    lookup: dict[tuple[int, int], float] = {
        (r["batch_size"], r["context_size"]): r[metric] for r in rows
    }

    header = f"  ctx \\ bs" + "".join(f"  bs={b:>4}" for b in batch_sizes)
    print(f"\n=== {backend}  layer={layer}  {label} ===")
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"  {ctx:>5} "
        for bs in batch_sizes:
            val = lookup.get((bs, ctx), float("nan"))
            if metric == "tokens_per_s":
                row += f"  {val:>8.0f}"
            elif metric == "peak_gpu_mb":
                row += f"  {val:>8.0f}"
            else:
                row += f"  {val:>8.3f}"
        print(row)


def print_speedup_table(
    results: list[dict],
    backend: str,
    ref_backend: str,
    batch_sizes: list[int],
    context_sizes: list[int],
    layer: int,
) -> None:
    def lookup(bk: str, bs: int, ctx: int) -> float:
        for r in results:
            if r["backend"] == bk and r["layer"] == layer and r["batch_size"] == bs and r["context_size"] == ctx:
                return r["tokens_per_s"]
        return float("nan")

    header = f"  ctx \\ bs" + "".join(f"  bs={b:>4}" for b in batch_sizes)
    print(f"\n=== speedup: {backend} / {ref_backend}  layer={layer} ===")
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"  {ctx:>5} "
        for bs in batch_sizes:
            v = lookup(backend, bs, ctx)
            ref = lookup(ref_backend, bs, ctx)
            speedup = v / ref if ref > 0 else float("nan")
            row += f"  {speedup:>8.2f}x"
        print(row)


def save_csv(results: list[dict], path: str) -> None:
    if not results:
        return
    fields = ["backend", "layer", "batch_size", "context_size",
              "tokens_per_s", "s_per_batch", "peak_gpu_mb"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--batch-sizes", type=_parse_int_list, default=BATCH_SIZES,
                   metavar="1,2,4,8")
    p.add_argument("--context-sizes", type=_parse_int_list, default=CONTEXT_SIZES,
                   metavar="128,512,1024,2048")
    p.add_argument("--layers", type=_parse_int_list, default=LAYERS,
                   metavar="0,15,31")
    p.add_argument("--hook", default=HOOK)
    p.add_argument("--num-warmup", type=int, default=NUM_WARMUP)
    p.add_argument("--num-measure", type=int, default=NUM_MEASURE)
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    p.add_argument("--output-dir", default="results")
    # Internal
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        _run_backend(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            dataset_path=args.dataset_path,
            batch_sizes=args.batch_sizes,
            context_sizes=args.context_sizes,
            layers=args.layers,
            hook=args.hook,
            num_warmup=args.num_warmup,
            num_measure=args.num_measure,
            out_path=args.out_path,
        )
        return

    print(f"Model:          {args.model_path}")
    print(f"Dataset:        {args.dataset_path}")
    print(f"Hook:           {args.hook}")
    print(f"Layers:         {args.layers}")
    print(f"Batch sizes:    {args.batch_sizes}")
    print(f"Context sizes:  {args.context_sizes}")
    print(f"Warmup/measure: {args.num_warmup} / {args.num_measure} batches")

    backends = ["vllm_tp1"]
    if not args.skip_tp2:
        backends.append("vllm_tp2")
    if not args.skip_hooked:
        backends.append("hooked")

    all_results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="bench_param_") as tmp:
        for backend in backends:
            out_path = f"{tmp}/{backend}.json"
            rows = launch_subprocess(backend, args, out_path)
            all_results.extend(rows)

    # Print tables
    for layer in args.layers:
        for backend in backends:
            print_table(all_results, backend, args.batch_sizes, args.context_sizes,
                        layer, "tokens_per_s", "tokens/s")
        for backend in backends:
            print_table(all_results, backend, args.batch_sizes, args.context_sizes,
                        layer, "s_per_batch", "s/batch")
        if "hooked" in backends:
            for backend in [b for b in backends if b != "hooked"]:
                print_speedup_table(all_results, backend, "hooked",
                                    args.batch_sizes, args.context_sizes, layer)

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_parametric_{ts}.csv"
    save_csv(all_results, str(csv_path))


if __name__ == "__main__":
    main()
