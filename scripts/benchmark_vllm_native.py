#!/usr/bin/env python
"""
Benchmark native vLLM prefill throughput (no hook capture).

Measures raw vLLM performance as a baseline to compare against
benchmark_parametric.py which uses HookedVLLMModel with activation capture.

Each TP size runs in its own subprocess to avoid CUDA re-initialization errors.

Usage:
    python scripts/benchmark_vllm_native.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048

    python scripts/benchmark_vllm_native.py \
        --dataset-path ../datasets/TinyTok_tokenized_llama31_ctx2048 \
        --tp 1 2 \
        --batch-sizes 8,16,32 \
        --context-sizes 128,256,512,1024
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
BATCH_SIZES = [16, 32]
CONTEXT_SIZES = [128, 512]
NUM_WARMUP = 2
NUM_MEASURE = 2


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
        if ids.numel() < context_size:
            pad = torch.full((context_size - ids.numel(),), pad_id, dtype=torch.long)
            ids = torch.cat([ids, pad])
        cur.append(ids)
        if len(cur) == batch_size:
            batches.append(torch.stack(cur))
            cur = []
            if len(batches) >= num_batches:
                break

    while len(batches) < num_batches:
        batches = (batches * ((num_batches // len(batches)) + 2))[:num_batches]
    return batches


# ---------------------------------------------------------------------------
# Worker: runs inside subprocess for one TP size
# ---------------------------------------------------------------------------

def _run_worker(
    tp: int,
    model_path: str,
    dataset_path: str,
    batch_sizes: list[int],
    context_sizes: list[int],
    num_warmup: int,
    num_measure: int,
    out_path: str,
) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    backend = f"vllm_native_tp{tp}"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    max_ctx = max(context_sizes)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        max_model_len=max_ctx + 1,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    # prefill-only: max_tokens=1 to minimize decode overhead
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    results: list[dict] = []
    total = len(batch_sizes) * len(context_sizes)
    done = 0

    for context_size in context_sizes:
        for batch_size in batch_sizes:
            total_batches = num_warmup + num_measure
            batches = load_batches(dataset_path, total_batches, batch_size, context_size, pad_id)
            prompt_batches = [b.tolist() for b in batches]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            try:
                for b in prompt_batches[:num_warmup]:
                    llm.generate(
                        [{"prompt_token_ids": seq} for seq in b],
                        sampling_params=sampling,
                    )

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.perf_counter()
                for b in prompt_batches[num_warmup:]:
                    llm.generate(
                        [{"prompt_token_ids": seq} for seq in b],
                        sampling_params=sampling,
                    )
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
                    "batch_size": batch_size,
                    "context_size": context_size,
                    "tokens_per_s": tps,
                    "s_per_batch": lat,
                    "peak_gpu_mb": peak_mb,
                })
                done += 1
                print(
                    f"[{backend}] bs={batch_size:3d} ctx={context_size:5d}"
                    f" → {tps:>10.0f} tok/s  {lat:.3f} s/batch  {peak_mb:.0f} MB"
                    f"  ({done}/{total})",
                    flush=True,
                )

            except Exception as e:
                print(f"[{backend}] ERROR bs={batch_size} ctx={context_size}: {e}", flush=True)
                results.append({
                    "backend": backend,
                    "batch_size": batch_size,
                    "context_size": context_size,
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
        "--batch-sizes", ",".join(str(x) for x in args.batch_sizes),
        "--context-sizes", ",".join(str(x) for x in args.context_sizes),
        "--num-warmup", str(args.num_warmup),
        "--num-measure", str(args.num_measure),
        "--out-path", out_path,
    ]
    print(f"\n[vllm_native_tp{tp}] Launching subprocess ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"TP={tp} subprocess failed (exit {result.returncode})")
    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict], backend: str, batch_sizes: list[int], context_sizes: list[int]) -> None:
    rows = [r for r in results if r["backend"] == backend]
    lookup = {(r["batch_size"], r["context_size"]): r["tokens_per_s"] for r in rows}
    header = "  ctx \\ bs" + "".join(f"  bs={b:>4}" for b in batch_sizes)
    print(f"\n=== {backend}  tokens/s ===")
    print(header)
    print("-" * len(header))
    for ctx in context_sizes:
        row = f"  {ctx:>5} "
        for bs in batch_sizes:
            val = lookup.get((bs, ctx), float("nan"))
            row += f"  {val:>8.0f}"
        print(row)


def save_csv(results: list[dict], path: str) -> None:
    fields = ["backend", "batch_size", "context_size", "tokens_per_s", "s_per_batch", "peak_gpu_mb"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    def int_list(s: str) -> list[int]:
        return [int(x) for x in s.split(",")]

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2],
                   help="Tensor parallel sizes to test (e.g. --tp 1 2)")
    p.add_argument("--batch-sizes", type=int_list, default=BATCH_SIZES, metavar="8,16,32")
    p.add_argument("--context-sizes", type=int_list, default=CONTEXT_SIZES, metavar="128,512,...")
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
            batch_sizes=args.batch_sizes,
            context_sizes=args.context_sizes,
            num_warmup=args.num_warmup,
            num_measure=args.num_measure,
            out_path=args.out_path,
        )
        return

    print(f"Model:          {args.model_path}")
    print(f"TP sizes:       {args.tp}")
    print(f"Batch sizes:    {args.batch_sizes}")
    print(f"Context sizes:  {args.context_sizes}")
    print(f"Warmup/measure: {args.num_warmup} / {args.num_measure} batches")

    import tempfile
    all_results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="bench_native_") as tmp:
        for tp in args.tp:
            out_path = f"{tmp}/tp{tp}.json"
            rows = launch(tp, args, out_path)
            all_results.extend(rows)

    backends = [f"vllm_native_tp{tp}" for tp in args.tp]
    for backend in backends:
        print_table(all_results, backend, args.batch_sizes, args.context_sizes)

    if 1 in args.tp and 2 in args.tp:
        print("\n=== speedup: vllm_native_tp2 / vllm_native_tp1 ===")
        tp1_lu = {(r["batch_size"], r["context_size"]): r["tokens_per_s"]
                  for r in all_results if r["backend"] == "vllm_native_tp1"}
        tp2_lu = {(r["batch_size"], r["context_size"]): r["tokens_per_s"]
                  for r in all_results if r["backend"] == "vllm_native_tp2"}
        header = "  ctx \\ bs" + "".join(f"  bs={b:>4}" for b in args.batch_sizes)
        print(header)
        print("-" * len(header))
        for ctx in args.context_sizes:
            row = f"  {ctx:>5} "
            for bs in args.batch_sizes:
                v1 = tp1_lu.get((bs, ctx), float("nan"))
                v2 = tp2_lu.get((bs, ctx), float("nan"))
                sp = v2 / v1 if v1 > 0 else float("nan")
                row += f"  {sp:>8.2f}x"
            print(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"benchmark_vllm_native_{ts}.csv"
    save_csv(all_results, str(csv_path))


if __name__ == "__main__":
    main()
