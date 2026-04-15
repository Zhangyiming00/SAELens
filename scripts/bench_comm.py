"""Benchmark communication bandwidth: Gloo CPU vs NCCL GPU collectives + P2P.

Sizes: 256 KB → 8 GB (powers of 4).  GPU ops timed with CUDA events.

Collectives tested:
  Gloo   : broadcast, send/recv
  NCCL   : all_reduce, all_gather, reduce_scatter, broadcast, send/recv

Bandwidth reported:
  algbw  = payload / time          (algorithmic BW — what the app sees)
  busbw  = algbw * bus_factor      (bus BW — what the NIC/NVLink actually carries)
    AllReduce      bus_factor = 2(n-1)/n
    AllGather      bus_factor = (n-1)/n
    ReduceScatter  bus_factor = (n-1)/n
    Broadcast      bus_factor = (n-1)/n
    P2P            bus_factor = 1

Usage:
    torchrun --nproc_per_node=2 scripts/bench_comm.py
    torchrun --nproc_per_node=2 scripts/bench_comm.py --warmup 10 --iters 50
    torchrun --nproc_per_node=2 scripts/bench_comm.py --min-kb 256 --max-gb 8
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    label: str
    n_bytes: int
    avg_ms: float
    algbw: float   # GB/s
    busbw: float   # GB/s


def _cuda_timed(fn, warmup: int, iters: int, group: dist.ProcessGroup) -> float:
    """Return average elapsed GPU milliseconds over `iters` calls, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms


def _cpu_timed(fn, warmup: int, iters: int, group: dist.ProcessGroup) -> float:
    """Return average elapsed CPU milliseconds over `iters` calls (Gloo/CPU path)."""
    for _ in range(warmup):
        fn()
    dist.barrier(group=group)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dist.barrier(group=group)
    return (time.perf_counter() - t0) / iters * 1e3   # ms


def _result(label: str, n_bytes: int, avg_ms: float, bus_factor: float) -> TimingResult:
    algbw = n_bytes / (avg_ms * 1e-3) / 1e9
    return TimingResult(label, n_bytes, avg_ms, algbw, algbw * bus_factor)


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_gloo_broadcast(group: dist.ProcessGroup, n_bytes: int,
                          warmup: int, iters: int) -> TimingResult:
    n = dist.get_world_size(group)
    t = torch.zeros(n_bytes // 4, dtype=torch.float32)
    avg_ms = _cpu_timed(lambda: dist.broadcast(t, src=0, group=group), warmup, iters, group)
    return _result(f"Gloo  broadcast   (n={n})", n_bytes, avg_ms, (n - 1) / n)


def bench_gloo_sendrecv(group: dist.ProcessGroup, n_bytes: int,
                         warmup: int, iters: int) -> TimingResult | None:
    n = dist.get_world_size(group)
    if n < 2:
        return None
    rank = dist.get_rank()
    members = [dist.get_global_rank(group, i) for i in range(n)]
    sender, recver = members[0], members[1]
    t = torch.zeros(n_bytes // 4, dtype=torch.float32)

    def fn():
        if rank == sender:
            dist.send(t, dst=recver, group=group)
        elif rank == recver:
            dist.recv(t, src=sender, group=group)
        else:
            pass  # idle ranks just wait at next barrier

    avg_ms = _cpu_timed(fn, warmup, iters, group)
    return _result(f"Gloo  send/recv   ({sender}→{recver})", n_bytes, avg_ms, 1.0)


def bench_nccl_all_reduce(group: dist.ProcessGroup, n_bytes: int,
                            warmup: int, iters: int) -> TimingResult:
    n = dist.get_world_size(group)
    t = torch.zeros(n_bytes // 4, dtype=torch.float32, device="cuda")
    avg_ms = _cuda_timed(lambda: dist.all_reduce(t, group=group), warmup, iters, group)
    return _result(f"NCCL  all_reduce  (n={n})", n_bytes, avg_ms, 2 * (n - 1) / n)


def bench_nccl_all_gather(group: dist.ProcessGroup, n_bytes: int,
                            warmup: int, iters: int) -> TimingResult:
    n = dist.get_world_size(group)
    shard = torch.zeros(n_bytes // 4 // n, dtype=torch.float32, device="cuda")
    out = [torch.empty_like(shard) for _ in range(n)]
    avg_ms = _cuda_timed(
        lambda: dist.all_gather(out, shard.contiguous(), group=group),
        warmup, iters, group,
    )
    return _result(f"NCCL  all_gather  (n={n})", n_bytes, avg_ms, (n - 1) / n)


def bench_nccl_reduce_scatter(group: dist.ProcessGroup, n_bytes: int,
                                warmup: int, iters: int) -> TimingResult:
    n = dist.get_world_size(group)
    inp = torch.zeros(n_bytes // 4, dtype=torch.float32, device="cuda")
    out = torch.zeros(n_bytes // 4 // n, dtype=torch.float32, device="cuda")
    avg_ms = _cuda_timed(
        lambda: dist.reduce_scatter_tensor(out, inp, group=group),
        warmup, iters, group,
    )
    return _result(f"NCCL  reduce_scat (n={n})", n_bytes, avg_ms, (n - 1) / n)


def bench_nccl_broadcast(group: dist.ProcessGroup, n_bytes: int,
                           warmup: int, iters: int) -> TimingResult:
    n = dist.get_world_size(group)
    t = torch.zeros(n_bytes // 4, dtype=torch.float32, device="cuda")
    avg_ms = _cuda_timed(lambda: dist.broadcast(t, src=0, group=group), warmup, iters, group)
    return _result(f"NCCL  broadcast   (n={n})", n_bytes, avg_ms, (n - 1) / n)


def bench_nccl_sendrecv(group: dist.ProcessGroup, n_bytes: int,
                          warmup: int, iters: int) -> TimingResult | None:
    n = dist.get_world_size(group)
    if n < 2:
        return None
    rank = dist.get_rank()
    members = [dist.get_global_rank(group, i) for i in range(n)]
    sender, recver = members[0], members[1]
    send_t = torch.zeros(n_bytes // 4, dtype=torch.float32, device="cuda")
    recv_t = torch.zeros(n_bytes // 4, dtype=torch.float32, device="cuda")

    def fn():
        ops = []
        if rank == sender:
            ops.append(dist.P2POp(dist.isend, send_t, recver, group=group))
        if rank == recver:
            ops.append(dist.P2POp(dist.irecv, recv_t, sender, group=group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

    avg_ms = _cuda_timed(fn, warmup, iters, group)
    return _result(f"NCCL  send/recv   ({sender}→{recver})", n_bytes, avg_ms, 1.0)


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print(f"\n{'Operation':<38}  {'Size':>9}  {'Time':>9}  {'algbw':>9}  {'busbw':>9}")
    print("-" * 82)


def _print_result(r: TimingResult) -> None:
    size_mb = r.n_bytes / 1e6
    if size_mb < 1:
        size_str = f"{r.n_bytes / 1024:.0f} KB"
    elif size_mb < 1024:
        size_str = f"{size_mb:.1f} MB"
    else:
        size_str = f"{size_mb / 1024:.2f} GB"
    print(
        f"  {r.label:<36}  {size_str:>9}  {r.avg_ms:>7.2f} ms"
        f"  {r.algbw:>7.2f} GB/s  {r.busbw:>7.2f} GB/s"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-kb", type=float, default=256,
                   help="Minimum payload size in KB (default: 256)")
    p.add_argument("--max-gb", type=float, default=8.0,
                   help="Maximum payload size in GB (default: 8)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--skip-gloo", action="store_true",
                   help="Skip Gloo benchmarks (much slower, skip for large sizes)")
    return p.parse_args()


def _size_range(min_kb: float, max_gb: float) -> list[int]:
    """Powers-of-4 sizes from min_kb to max_gb, aligned to float32."""
    sizes: list[int] = []
    n_bytes = int(min_kb * 1024)
    max_bytes = int(max_gb * 1024 ** 3)
    while n_bytes <= max_bytes:
        aligned = (n_bytes // 4) * 4
        if aligned > 0:
            sizes.append(aligned)
        n_bytes *= 4
    return sizes


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    all_ranks = list(range(world))
    gloo_group = dist.new_group(all_ranks, backend="gloo")
    nccl_group = dist.new_group(all_ranks, backend="nccl")

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device)
        print(f"world={world}  GPU={gpu_name}  device=cuda:{local_rank}")
        print(f"sizes: {args.min_kb:.0f} KB → {args.max_gb:.0f} GB  "
              f"warmup={args.warmup}  iters={args.iters}")
        _print_header()

    sizes = _size_range(args.min_kb, args.max_gb)

    for n_bytes in sizes:
        results: list[TimingResult] = []

        if not args.skip_gloo:
            results.append(bench_gloo_broadcast(gloo_group, n_bytes, args.warmup, args.iters))
            r = bench_gloo_sendrecv(gloo_group, n_bytes, args.warmup, args.iters)
            if r is not None:
                results.append(r)

        results.append(bench_nccl_all_reduce(nccl_group, n_bytes, args.warmup, args.iters))
        results.append(bench_nccl_all_gather(nccl_group, n_bytes, args.warmup, args.iters))
        results.append(bench_nccl_reduce_scatter(nccl_group, n_bytes, args.warmup, args.iters))
        results.append(bench_nccl_broadcast(nccl_group, n_bytes, args.warmup, args.iters))
        r = bench_nccl_sendrecv(nccl_group, n_bytes, args.warmup, args.iters)
        if r is not None:
            results.append(r)

        if rank == 0:
            for r in results:
                _print_result(r)
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
