#!/usr/bin/env python
"""
Measure NCCL AllReduce latency and bandwidth for tensor sizes
relevant to Llama-3.1-8B TP=2 prefill.

In Llama TP=2, each layer performs 2 AllReduces:
  - after attention output projection: (B*S, d_model)
  - after MLP down projection:        (B*S, d_model)

Run with:
    torchrun --nproc_per_node=2 scripts/benchmark_nccl_allreduce.py
    torchrun --nproc_per_node=2 scripts/benchmark_nccl_allreduce.py \
        --d-model 4096 --num-layers 32 --num-iters 100
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

D_MODEL = 4096       # Llama-3.1-8B hidden size
NUM_LAYERS = 32      # Llama-3.1-8B layers
NUM_WARMUP = 10
NUM_ITERS = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=D_MODEL)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--num-warmup", type=int, default=NUM_WARMUP)
    p.add_argument("--num-iters", type=int, default=NUM_ITERS)
    return p.parse_args()


def bench_allreduce(
    tensor_size: int,
    dtype: torch.dtype,
    num_warmup: int,
    num_iters: int,
    rank: int,
) -> tuple[float, float]:
    """Returns (mean_latency_ms, bandwidth_GBs)."""
    x = torch.randn(tensor_size, dtype=dtype, device=f"cuda:{rank}")

    # Warmup
    for _ in range(num_warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / num_iters * 1000
    # AllReduce on 2 GPUs transfers 2*(N-1)/N * data = 1x data total per GPU
    bytes_transferred = x.numel() * x.element_size()
    bandwidth_GBs = bytes_transferred / (elapsed / num_iters) / 1e9
    return latency_ms, bandwidth_GBs


def main() -> None:
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"World size: {world_size}  |  d_model={args.d_model}  |  layers={args.num_layers}")
        print(f"Warmup: {args.num_warmup}  |  Iters: {args.num_iters}")
        print()
        print(f"{'config':<30} {'tensor_bytes':>12} {'latency_ms':>12} {'BW (GB/s)':>12}")
        print("-" * 70)

    dtype = torch.bfloat16

    # Sweep over (batch_size, seq_len) combinations typical in SAE training
    configs = [
        (1, 128), (4, 128), (8, 128), (16, 128), (32, 128),
        (1, 512), (4, 512), (8, 512), (16, 512), (32, 512),
        (1, 1024), (4, 1024), (8, 1024),
        (1, 2048), (4, 2048),
    ]

    results = []
    for batch_size, seq_len in configs:
        # One AllReduce: (B*S, d_model) flattened
        tensor_size = batch_size * seq_len * args.d_model
        lat_ms, bw = bench_allreduce(tensor_size, dtype, args.num_warmup, args.num_iters, rank)

        # Total TP=2 AllReduce cost per full forward pass (2 per layer)
        total_lat_ms = lat_ms * 2 * args.num_layers

        results.append((batch_size, seq_len, tensor_size, lat_ms, bw, total_lat_ms))

        if rank == 0:
            nbytes = tensor_size * 2  # bfloat16
            label = f"bs={batch_size:3d} ctx={seq_len:5d}"
            print(f"  {label:<28} {nbytes:>12,}  {lat_ms:>11.3f}ms  {bw:>11.2f}")

    if rank == 0:
        print()
        print("Estimated TP=2 AllReduce overhead per full forward pass (2 AllReduces/layer × 32 layers):")
        print(f"{'config':<30} {'total_allreduce_ms':>20} {'tokens':>10}")
        print("-" * 65)
        for batch_size, seq_len, _, lat_ms, _, total_lat_ms in results:
            tokens = batch_size * seq_len
            label = f"bs={batch_size:3d} ctx={seq_len:5d}"
            print(f"  {label:<28} {total_lat_ms:>19.2f}ms  {tokens:>10,} tok")

        print()
        print("Note: actual inference time = compute time + allreduce overhead")
        print("      TP=2 is beneficial only if compute time >> allreduce overhead")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
