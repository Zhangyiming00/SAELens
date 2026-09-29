"""Unprofiled local candidate latency and CPU issue time, with optional Nsight.

This exercises the production implementation without changing training defaults.
CPU issue latency is measured before synchronization; wall latency is after it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--widths", type=int, nargs="+", default=[2048,4096,5120,6144,8192,10240,12288,16384])
    p.add_argument("--rows", type=int, nargs="+", default=[2048,4096,8192,12288])
    p.add_argument("--backends", nargs="+", default=["torch", "triton"], choices=["torch", "triton"])
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    if args.k < 1 or any(width < args.k for width in args.widths):
        p.error("k must be positive and no larger than any local width")
    if args.output.exists():
        p.error("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    sys.path.insert(0,str(REPO))
    from sae_lens.sharded_topk import _local_candidates
    result = []
    for backend in args.backends:
        for width in args.widths:
            for rows in args.rows:
                torch.manual_seed(73)
                scores = torch.randn(rows, width, device="cuda")
                for _ in range(args.warmup):
                    out = _local_candidates(scores, args.k, 0, None, backend)
                    del out
                torch.cuda.synchronize()
                samples=[]
                if args.trace:
                    torch.cuda.profiler.start()
                for i in range(args.steps):
                    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                    start=time.perf_counter()
                    if args.trace:
                        torch.cuda.nvtx.range_push(f"candidates:{backend}:{rows}:{width}:{i}")
                    out = _local_candidates(scores,args.k,0,None,backend)
                    issue_ms=1000*(time.perf_counter()-start)
                    b.record()
                    if args.trace:
                        torch.cuda.nvtx.range_pop()
                    b.synchronize()
                    wall_ms=1000*(time.perf_counter()-start)
                    samples.append(dict(issue_ms=issue_ms,wall_ms=wall_ms,cuda_ms=a.elapsed_time(b)))
                    del out
                if args.trace:
                    torch.cuda.profiler.stop()
                tile=max(1, (32*1024**2)//(width*(8 if backend=="triton" else 64)))
                row=dict(backend=backend,rows=rows,width=width,k=args.k,tile_rows=tile,tiles=math.ceil(rows/tile),
                         issue_ms=statistics.median(s["issue_ms"] for s in samples),
                         wall_ms=statistics.median(s["wall_ms"] for s in samples),
                         samples=samples)
                result.append(row)
                print(backend,rows,width,f"issue={row['issue_ms']:.3f} wall={row['wall_ms']:.3f} ms",flush=True)
                del scores
    args.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                                         cuda=torch.version.cuda,trace=args.trace,rows=result),indent=2)+"\n")


if __name__ == "__main__":
    main()
