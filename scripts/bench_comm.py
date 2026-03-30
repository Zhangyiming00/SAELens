"""Benchmark communication bandwidth for each collective used in SAELens.

Covers:
  - Gloo CPU broadcast  (worker_cpu_group path: activations + tokens)
  - Gloo CPU send/recv  (vllm_dp P2P / shard-routing P2P)
  - NCCL GPU all_gather (vllm_tp_group, split-activation gather)
  - NCCL GPU broadcast  (sae_tp_group, fan-in / v2 Phase-6)
  - NCCL GPU all_reduce (sae_dp_group, gradient sync)

Usage (2 GPUs):
    torchrun --nproc_per_node=2 scripts/bench_comm.py
    torchrun --nproc_per_node=2 scripts/bench_comm.py --sizes-mb 32 128 512
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensor(n_bytes: int, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n_elem = n_bytes // dtype.itemsize  # type: ignore[attr-defined]
    return torch.zeros(n_elem, dtype=dtype, device=device)


def _bandwidth_gb_s(n_bytes: int, elapsed_s: float) -> float:
    return n_bytes / elapsed_s / 1e9


def _run(label: str, fn, n_bytes: int, warmup: int, iters: int) -> None:
    rank = dist.get_rank()
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dist.barrier()
    elapsed = time.perf_counter() - t0
    avg_s = elapsed / iters
    bw = _bandwidth_gb_s(n_bytes, avg_s)
    if rank == 0:
        print(f"  {label:<45s}  size={n_bytes/1e6:7.1f} MB  avg={avg_s*1e3:7.2f} ms  bw={bw:6.2f} GB/s")


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_gloo_broadcast(group: dist.ProcessGroup, n_bytes: int, src: int,
                          warmup: int, iters: int) -> None:
    rank = dist.get_rank()
    t = _make_tensor(n_bytes, device="cpu")
    label = f"Gloo broadcast (src={src}, world={dist.get_world_size(group)})"
    _run(label, lambda: dist.broadcast(t, src=src, group=group), n_bytes, warmup, iters)


def bench_gloo_sendrecv(group: dist.ProcessGroup, n_bytes: int,
                         warmup: int, iters: int) -> None:
    rank = dist.get_rank()
    world = dist.get_world_size(group)
    # rank 0 sends to rank 1 (inside the group); group members are [0, 1]
    members = [dist.get_global_rank(group, i) for i in range(world)]
    sender = members[0]
    recver = members[1]
    t = _make_tensor(n_bytes, device="cpu")

    def fn():
        if rank == sender:
            dist.send(t, dst=recver, group=group)
        elif rank == recver:
            dist.recv(t, src=sender, group=group)

    label = f"Gloo send/recv (rank {sender}→{recver})"
    _run(label, fn, n_bytes, warmup, iters)


def bench_nccl_all_gather(group: dist.ProcessGroup, n_bytes: int,
                            warmup: int, iters: int) -> None:
    world = dist.get_world_size(group)
    # Each rank holds a shard of size n_bytes/world; all_gather reassembles
    shard_bytes = n_bytes // world
    local = _make_tensor(shard_bytes, device="cuda")
    shards = [torch.empty_like(local) for _ in range(world)]
    label = f"NCCL all_gather (world={world}, shard={shard_bytes/1e6:.1f} MB)"
    _run(label, lambda: dist.all_gather(shards, local.contiguous(), group=group),
         n_bytes, warmup, iters)


def bench_nccl_broadcast(group: dist.ProcessGroup, n_bytes: int, src: int,
                           warmup: int, iters: int) -> None:
    world = dist.get_world_size(group)
    t = _make_tensor(n_bytes, device="cuda")
    label = f"NCCL broadcast   (src={src}, world={world})"
    _run(label, lambda: dist.broadcast(t, src=src, group=group), n_bytes, warmup, iters)


def bench_nccl_all_reduce(group: dist.ProcessGroup, n_bytes: int,
                            warmup: int, iters: int) -> None:
    world = dist.get_world_size(group)
    t = _make_tensor(n_bytes, device="cuda")
    label = f"NCCL all_reduce  (world={world})"
    _run(label, lambda: dist.all_reduce(t, group=group), n_bytes, warmup, iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sizes-mb", nargs="+", type=float,
        default=[1, 8, 32, 128, 512],
        help="Payload sizes in MB to benchmark (default: 1 8 32 128 512)",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")  # NCCL as default; Gloo groups created explicitly

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    if rank == 0:
        print(f"world_size={world_size}  GPUs available={torch.cuda.device_count()}")

    # Build groups (must be created on ALL ranks simultaneously)
    all_ranks = list(range(world_size))

    # Gloo groups (CPU comms)
    gloo_group_all  = dist.new_group(all_ranks, backend="gloo")

    # NCCL groups (GPU comms)
    nccl_group_all  = dist.new_group(all_ranks, backend="nccl")

    # For send/recv we need at least 2 ranks; skip if world_size==1
    can_p2p = world_size >= 2

    for size_mb in args.sizes_mb:
        n_bytes = int(size_mb * 1e6)
        # round to float32 element boundary
        n_bytes = (n_bytes // 4) * 4

        if rank == 0:
            print(f"\n=== {size_mb:.0f} MB ===")

        # --- Gloo CPU broadcast (all ranks) ---
        bench_gloo_broadcast(gloo_group_all, n_bytes, src=0, warmup=args.warmup, iters=args.iters)

        # --- Gloo CPU send/recv (rank 0 → rank 1) ---
        if can_p2p:
            bench_gloo_sendrecv(gloo_group_all, n_bytes, warmup=args.warmup, iters=args.iters)

        # --- NCCL GPU broadcast (all ranks) ---
        bench_nccl_broadcast(nccl_group_all, n_bytes, src=0, warmup=args.warmup, iters=args.iters)

        # --- NCCL GPU all_gather (all ranks) ---
        if world_size >= 2:
            bench_nccl_all_gather(nccl_group_all, n_bytes, warmup=args.warmup, iters=args.iters)

        # --- NCCL GPU all_reduce (all ranks) ---
        bench_nccl_all_reduce(nccl_group_all, n_bytes, warmup=args.warmup, iters=args.iters)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
