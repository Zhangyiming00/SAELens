#!/usr/bin/env python
"""
Benchmark activation capture throughput for:
  - HookedVLLMModel TP=1
  - HookedVLLMModel TP=2
  - HookedTransformer

Reports tokens/second for each (layer, backend) combination.
Each vLLM backend runs in its own subprocess so GPU memory is fully released.

Usage:
    python scripts/benchmark_activation_capture.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYERS = [0, 7, 15, 23, 31]
DEFAULT_HOOK_TYPES = ("hook_resid_pre", "hook_attn_out", "hook_mlp_out", "hook_resid_post")

# ---- data loading ----

def load_batches(
    dataset_path: str,
    num_batches: int,
    batch_size: int,
    max_length: int,
    pad_id: int,
) -> list[torch.Tensor]:
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    for key in ("input_ids", "tokens", "token_ids"):
        if key in dataset.column_names:
            break
    else:
        raise ValueError(f"No token column in {dataset_path}")

    batches: list[torch.Tensor] = []
    cur: list[torch.Tensor] = []
    for row in dataset:
        ids = torch.tensor(list(row[key]), dtype=torch.long)[:max_length]
        cur.append(ids)
        if len(cur) == batch_size:
            ml = max(t.numel() for t in cur)
            b = torch.full((len(cur), ml), pad_id, dtype=torch.long)
            for j, t in enumerate(cur):
                b[j, : t.numel()] = t
            batches.append(b)
            cur = []
            if len(batches) >= num_batches:
                break
    return batches


# ---- subprocess worker: benchmark one vLLM backend ----

def _benchmark_vllm(
    model_path: str,
    dataset_path: str,
    tp: int,
    layers: list[int],
    hook_types: list[str],
    num_warmup: int,
    num_measure: int,
    batch_size: int,
    max_length: int,
    out_path: str,
) -> None:
    """Run inside subprocess: benchmark capture, write JSON results."""
    import json

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from transformers import AutoTokenizer

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name
    from sae_lens.vllm_model import HookedVLLMModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    total_batches = num_warmup + num_measure
    batches = load_batches(dataset_path, total_batches, batch_size, max_length, pad_id)
    # Repeat if dataset is small
    while len(batches) < total_batches:
        batches = (batches * ((total_batches // len(batches)) + 2))[:total_batches]

    effective_max = max_length + 1
    model = HookedVLLMModel(
        model_path, tokenizer,
        tensor_parallel_size=tp,
        max_model_len=effective_max,
        enable_prefix_caching=False,
    )

    results: dict[str, float] = {}
    try:
        for layer in layers:
            hook_names = [f"blocks.{layer}.{h}" for h in hook_types]
            stop = extract_stop_at_layer_from_tlens_hook_name(hook_names[0])

            # warmup
            for batch in batches[:num_warmup]:
                model.run_with_cache(batch, names_filter=hook_names,
                                     stop_at_layer=stop, prepend_bos=False)

            # measure
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_tokens = 0
            for batch in batches[num_warmup:]:
                model.run_with_cache(batch, names_filter=hook_names,
                                     stop_at_layer=stop, prepend_bos=False)
                total_tokens += batch.numel()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            tps = total_tokens / elapsed
            results[str(layer)] = tps
            print(f"  layer {layer:2d}: {tps:>10.0f} tok/s  ({elapsed:.2f}s for {total_tokens} tokens)")
            sys.stdout.flush()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    import json
    with open(out_path, "w") as f:
        json.dump(results, f)


# ---- subprocess launcher ----

def run_vllm_subprocess(
    tp: int,
    out_path: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    cmd = [
        sys.executable, __file__,
        "--_benchmark-vllm",
        "--model-path", args.model_path,
        "--dataset-path", args.dataset_path,
        "--tp", str(tp),
        "--num-warmup", str(args.num_warmup),
        "--num-measure", str(args.num_measure),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--out-path", out_path,
    ]
    print(f"\n[vLLM TP={tp}] Starting benchmark ...")
    sys.stdout.flush()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"vLLM TP={tp} subprocess failed (exit {result.returncode})")
    import json
    with open(out_path) as f:
        return json.load(f)


# ---- HookedTransformer benchmark ----

def benchmark_hooked_transformer(
    dataset_path: str,
    layers: list[int],
    hook_types: tuple[str, ...],
    *,
    model_path: str,
    hooked_model_name: str,
    num_warmup: int,
    num_measure: int,
    batch_size: int,
    max_length: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, float]:
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    total_batches = num_warmup + num_measure
    batches = load_batches(dataset_path, total_batches, batch_size, max_length, pad_id)
    while len(batches) < total_batches:
        batches = (batches * ((total_batches // len(batches)) + 2))[:total_batches]

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, local_files_only=True)
    model = HookedTransformer.from_pretrained_no_processing(
        hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=dtype, local_files_only=True,
    )
    model.eval()

    results: dict[str, float] = {}
    try:
        for layer in layers:
            hook_names = [f"blocks.{layer}.{h}" for h in hook_types]
            stop = extract_stop_at_layer_from_tlens_hook_name(hook_names[0])

            # warmup
            for batch in batches[:num_warmup]:
                with torch.inference_mode():
                    model.run_with_cache(
                        batch.to(device), names_filter=hook_names,
                        stop_at_layer=stop, prepend_bos=False, return_cache_object=False,
                    )

            # measure
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_tokens = 0
            for batch in batches[num_warmup:]:
                with torch.inference_mode():
                    model.run_with_cache(
                        batch.to(device), names_filter=hook_names,
                        stop_at_layer=stop, prepend_bos=False, return_cache_object=False,
                    )
                total_tokens += batch.numel()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            tps = total_tokens / elapsed
            results[str(layer)] = tps
            print(f"  layer {layer:2d}: {tps:>10.0f} tok/s  ({elapsed:.2f}s for {total_tokens} tokens)")
            sys.stdout.flush()
    finally:
        del model, hf_model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ---- arg parsing ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--num-warmup", type=int, default=2)
    p.add_argument("--num-measure", type=int, default=10)
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    # Internal
    p.add_argument("--_benchmark-vllm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--tp", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---- report ----

def print_report(
    layers: list[int],
    tp1: dict[str, float],
    tp2: dict[str, float] | None,
    hooked: dict[str, float],
) -> None:
    backends = ["vLLM TP=1", "HookedTransformer"]
    if tp2 is not None:
        backends = ["vLLM TP=1", "vLLM TP=2", "HookedTransformer"]

    header = f"{'layer':>6}"
    for b in backends:
        header += f"  {b:>16}"
    if tp2 is not None:
        header += f"  {'TP1/Hooked':>10}  {'TP2/Hooked':>10}"
    else:
        header += f"  {'TP1/Hooked':>10}"
    print()
    print("=" * len(header))
    print("  Throughput (tokens/second)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for layer in layers:
        k = str(layer)
        t1 = tp1.get(k, float("nan"))
        th = hooked.get(k, float("nan"))
        row = f"{layer:>6}  {t1:>16.0f}"
        if tp2 is not None:
            t2 = tp2.get(k, float("nan"))
            row += f"  {t2:>16.0f}"
        row += f"  {th:>16.0f}"
        ratio1 = t1 / th if th > 0 else float("nan")
        row += f"  {ratio1:>10.2f}x"
        if tp2 is not None:
            t2 = tp2.get(k, float("nan"))
            ratio2 = t2 / th if th > 0 else float("nan")
            row += f"  {ratio2:>10.2f}x"
        print(row)

    print("-" * len(header))


# ---- main ----

def main() -> None:
    args = parse_args()

    if args._benchmark_vllm:
        _benchmark_vllm(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            tp=args.tp,
            layers=DEFAULT_LAYERS,
            hook_types=list(DEFAULT_HOOK_TYPES),
            num_warmup=args.num_warmup,
            num_measure=args.num_measure,
            batch_size=args.batch_size,
            max_length=args.max_length,
            out_path=args.out_path,
        )
        return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Model:      {args.model_path}")
    print(f"Dataset:    {args.dataset_path}")
    print(f"Layers:     {DEFAULT_LAYERS}")
    print(f"Hooks/layer: {list(DEFAULT_HOOK_TYPES)}")
    print(f"Batch size: {args.batch_size}  max_length: {args.max_length}")
    print(f"Warmup: {args.num_warmup}  Measure: {args.num_measure} batches")

    with tempfile.TemporaryDirectory(prefix="bench_act_") as tmp:
        tp1_path = f"{tmp}/tp1.json"
        tp2_path = f"{tmp}/tp2.json"

        tp1 = run_vllm_subprocess(1, tp1_path, args)

        tp2 = None
        if not args.skip_tp2:
            tp2 = run_vllm_subprocess(2, tp2_path, args)

    print("\n[HookedTransformer] Starting benchmark ...")
    hooked = benchmark_hooked_transformer(
        dataset_path=args.dataset_path,
        layers=DEFAULT_LAYERS,
        hook_types=DEFAULT_HOOK_TYPES,
        model_path=args.model_path,
        hooked_model_name=args.hooked_model_name,
        num_warmup=args.num_warmup,
        num_measure=args.num_measure,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=dtype,
        device=args.device,
    )

    print_report(DEFAULT_LAYERS, tp1, tp2, hooked)


if __name__ == "__main__":
    main()
