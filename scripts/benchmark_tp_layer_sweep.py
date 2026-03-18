#!/usr/bin/env python
"""
Sweep stop_at_layer for vLLM TP=1 vs TP=2 to isolate per-layer communication overhead.

Each TP size runs in its own subprocess. For each layer depth, we measure
tokens/s and s/batch. The throughput difference between TP=1 and TP=2 grows
with depth: the slope reveals per-layer AllReduce cost.

Usage:
    python scripts/benchmark_tp_layer_sweep.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048

    python scripts/benchmark_tp_layer_sweep.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \
        --layers 1,4,8,16,24,32 --batch-size 16 --context-size 512
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_LAYERS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
DEFAULT_BATCH_SIZE = 16
DEFAULT_CONTEXT_SIZE = 512
NUM_WARMUP = 3
NUM_MEASURE = 10


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_worker(
    tp: int,
    model_path: str,
    dataset_path: str,
    layers: list[int],
    batch_size: int,
    context_size: int,
    num_warmup: int,
    num_measure: int,
    out_path: str,
) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

    backend = f"vllm_native_tp{tp}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Build once with max context
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        max_model_len=context_size + 1,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    # Load data once
    dataset = load_from_disk(dataset_path)
    for key in ("input_ids", "tokens", "token_ids"):
        if key in dataset.column_names:
            break

    total_batches = num_warmup + num_measure
    raw: list[list[int]] = []
    for row in dataset:
        ids = list(row[key])[:context_size]
        if len(ids) < context_size:
            ids += [pad_id] * (context_size - len(ids))
        raw.append(ids)
        if len(raw) >= total_batches * batch_size:
            break
    while len(raw) < total_batches * batch_size:
        raw = (raw * 2)[:total_batches * batch_size]

    prompt_batches = [
        [{"prompt_token_ids": raw[i * batch_size + j]} for j in range(batch_size)]
        for i in range(total_batches)
    ]

    results = []
    for layer in layers:
        # vLLM stop_at_layer: stop after processing layer `layer`
        # We use hook_resid_post at this layer as the reference hook
        hook_name = f"blocks.{layer - 1}.hook_resid_post"
        stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            # Warmup
            for b in prompt_batches[:num_warmup]:
                llm.generate(b, sampling_params=sampling)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for b in prompt_batches[num_warmup:]:
                llm.generate(b, sampling_params=sampling)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - t0

            total_tokens = num_measure * batch_size * context_size
            tps = total_tokens / elapsed
            lat = elapsed / num_measure
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available() else 0.0
            )

            results.append({
                "backend": backend,
                "layer": layer,
                "tokens_per_s": tps,
                "s_per_batch": lat,
                "peak_gpu_mb": peak_mb,
            })
            print(
                f"[{backend}] layer={layer:3d} → {tps:>9.0f} tok/s  {lat:.3f} s/batch",
                flush=True,
            )

        except Exception as e:
            print(f"[{backend}] ERROR layer={layer}: {e}", flush=True)
            results.append({
                "backend": backend,
                "layer": layer,
                "tokens_per_s": float("nan"),
                "s_per_batch": float("nan"),
                "peak_gpu_mb": float("nan"),
            })

    del llm
    with open(out_path, "w") as f:
        json.dump(results, f)


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def launch(tp: int, args: argparse.Namespace, out_path: str) -> list[dict]:
    cmd = [
        sys.executable, __file__,
        "--_run-tp", str(tp),
        "--model-path", args.model_path,
        "--dataset-path", args.dataset_path,
        "--layers", ",".join(str(x) for x in args.layers),
        "--batch-size", str(args.batch_size),
        "--context-size", str(args.context_size),
        "--num-warmup", str(args.num_warmup),
        "--num-measure", str(args.num_measure),
        "--out-path", out_path,
    ]
    print(f"\n[vllm_native_tp{tp}] Sweeping {len(args.layers)} layers ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"TP={tp} subprocess failed (exit {result.returncode})")
    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_results: list[dict], layers: list[int], tp_sizes: list[int]) -> None:
    by_backend: dict[str, dict[int, dict]] = {}
    for r in all_results:
        by_backend.setdefault(r["backend"], {})[r["layer"]] = r

    print("\n=== tokens/s by layer ===")
    header = f"  {'layer':>6}" + "".join(f"  {'tp'+str(tp):>12}" for tp in tp_sizes)
    if len(tp_sizes) >= 2:
        header += f"  {'overhead_ms':>12}"
    print(header)
    print("-" * len(header))

    for layer in layers:
        row = f"  {layer:>6}"
        tps_values = []
        lat_values = []
        for tp in tp_sizes:
            b = f"vllm_native_tp{tp}"
            info = by_backend.get(b, {}).get(layer, {})
            tps = info.get("tokens_per_s", float("nan"))
            lat = info.get("s_per_batch", float("nan"))
            row += f"  {tps:>12.0f}"
            tps_values.append(tps)
            lat_values.append(lat)

        if len(tp_sizes) >= 2 and 1 in tp_sizes and 2 in tp_sizes:
            idx1 = tp_sizes.index(1)
            idx2 = tp_sizes.index(2)
            lat1 = lat_values[idx1]
            lat2 = lat_values[idx2]
            # overhead = extra time TP=2 spends vs TP=1 per batch, in ms
            if lat1 == lat1 and lat2 == lat2:  # not nan
                overhead_ms = (lat2 - lat1) * 1000
                row += f"  {overhead_ms:>+12.2f}ms"

        print(row)

    # Linear fit: overhead_ms vs num_layers → slope = per-layer AllReduce cost
    if len(tp_sizes) >= 2 and 1 in tp_sizes and 2 in tp_sizes:
        idx1 = tp_sizes.index(1)
        idx2 = tp_sizes.index(2)
        xs, ys = [], []
        for layer in layers:
            b1 = f"vllm_native_tp{tp_sizes[idx1]}"
            b2 = f"vllm_native_tp{tp_sizes[idx2]}"
            lat1 = by_backend.get(b1, {}).get(layer, {}).get("s_per_batch", float("nan"))
            lat2 = by_backend.get(b2, {}).get(layer, {}).get("s_per_batch", float("nan"))
            if lat1 == lat1 and lat2 == lat2:
                xs.append(layer)
                ys.append((lat2 - lat1) * 1000)

        if len(xs) >= 2:
            import numpy as np
            slope, intercept = np.polyfit(xs, ys, 1)
            print(f"\nLinear fit: overhead_ms = {slope:.3f} × num_layers + {intercept:.3f}")
            print(f"  → per-layer AllReduce cost ≈ {slope:.3f} ms")
            print(f"  → fixed overhead (intercept) ≈ {intercept:.3f} ms")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    def int_list(s: str) -> list[int]:
        return [int(x) for x in s.split(",")]

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    p.add_argument("--layers", type=int_list, default=DEFAULT_LAYERS,
                   metavar="1,4,8,16,32")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    p.add_argument("--num-warmup", type=int, default=NUM_WARMUP)
    p.add_argument("--num-measure", type=int, default=NUM_MEASURE)
    p.add_argument("--output-dir", default="results")
    # Internal
    p.add_argument("--_run-tp", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args._run_tp is not None:
        _run_worker(
            tp=args._run_tp,
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            layers=args.layers,
            batch_size=args.batch_size,
            context_size=args.context_size,
            num_warmup=args.num_warmup,
            num_measure=args.num_measure,
            out_path=args.out_path,
        )
        return

    print(f"Model:          {args.model_path}")
    print(f"TP sizes:       {args.tp}")
    print(f"Layers:         {args.layers}")
    print(f"Batch size:     {args.batch_size}  |  Context: {args.context_size}")
    print(f"Warmup/measure: {args.num_warmup} / {args.num_measure} batches")

    import tempfile
    all_results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="bench_layer_sweep_") as tmp:
        for tp in args.tp:
            out_path = f"{tmp}/tp{tp}.json"
            rows = launch(tp, args, out_path)
            all_results.extend(rows)

    print_report(all_results, args.layers, args.tp)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_tp_layer_sweep_{ts}.csv"
    fields = ["backend", "layer", "tokens_per_s", "s_per_batch", "peak_gpu_mb"]
    import csv
    with open(csv_path, "w", newline="") as f:
        import csv as csv_mod
        writer = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
