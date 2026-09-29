"""Reprofile production sparse decoder stages, including adapter workspace.

No dense surrogate or vLLM. Forward, dvalues and dweight call sparse_parts
directly. CUDA-event elapsed, synchronized host elapsed and incremental Torch
allocated peaks are separate metrics. dvalues reuses forward's page groups,
as actual autograd does. Synthetic row ownership is an explicit family key.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"sae_lens/autoconfig"))
from execution_time_model import predict_sparse


def dump(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def configurations():
    families = []
    for engine in ("openai", "triton"):
        for k in (32, 64, 128, 256):
            for pattern in ("uniform", "alternating"):
                families.append(dict(engine=engine, k=k, pattern=pattern, d_in=4096,
                                     workspace_mib=256, page_k=512, split=1,
                                     index_backend="sort", forward_mode="bucketed", dtype="float32"))
    families.append(dict(families[4], workspace_mib=64))
    cal, hold = {}, {}
    for c in families:
        family = f"{c['engine']}_k{c['k']}_{c['pattern']}_w{c['workspace_mib']}"
        for rows in (1024, 8192):
            for width in (16384, 65536):
                cal[f"{family}_b{rows}_f{width}"] = dict(c, rows=rows, width=width)
        hold[family+"_b4096_f32768"] = dict(c, rows=4096, width=32768)
    return cal, hold


def inputs(c, torch):
    gen = torch.Generator().manual_seed(42)
    b, w, d, k = (c[n] for n in ("rows", "width", "d_in", "k"))
    lengths = torch.full((b,), k, dtype=torch.long)
    if c["pattern"] == "alternating":
        lengths[::2] -= k//4
        lengths[1::2] += k//4
    if int(lengths.max()) > w:
        raise ValueError("Row entries exceed width")
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    row_ids = torch.arange(b).repeat_interleave(lengths)
    within = torch.arange(row_ids.numel()) - offsets[row_ids]
    # Odd stride is coprime to our power-of-two widths: no duplicate ids/row.
    ids = (row_ids*1543 + within*7919) % w
    values = torch.randn(row_ids.numel(), generator=gen)
    values[::17] = 0  # Selected zero is still a differentiable entry.
    vectors = torch.randn(w, d, generator=gen)/d**.5
    grad = torch.randn(b, d, generator=gen)/d**.5
    return [x.cuda() for x in (vectors, values, ids, row_ids, offsets, grad)]


def operations(data, cfg):
    from sae_lens.sparse_parts import sparse_forward, sparse_dvalues, sparse_dweight
    vectors, values, ids, rows, offsets, grad = data
    opts = dict(engine=cfg["engine"], split=cfg["split"], index_backend=cfg["index_backend"],
                page_k=cfg["page_k"], openai_workspace_mib=cfg["workspace_mib"],
                forward_mode=cfg["forward_mode"])
    output, groups = sparse_forward(vectors, values, ids, rows, offsets, opts)
    del output
    return dict(forward=lambda: sparse_forward(vectors, values, ids, rows, offsets, opts),
                dvalues=lambda: sparse_dvalues(vectors, values, ids, rows, offsets, grad, opts, groups),
                dweight=lambda: sparse_dweight(vectors, values, ids, rows, grad, opts)), groups


def verify(torch):
    reports = []
    for engine in ("openai", "triton"):
        c = dict(engine=engine, rows=9, width=256, d_in=128, k=16,
                 pattern="alternating", split=1, index_backend="sort", page_k=512,
                 workspace_mib=256, forward_mode="bucketed")
        data = inputs(c, torch)
        w, v, ids, rows, offsets, grad = data
        ops, _ = operations(data, c)
        dense = torch.zeros((c["rows"], c["width"]), device="cuda")
        dense.index_put_((rows, ids), v, accumulate=True)
        expected = dict(forward=dense@w, dvalues=(grad@w.T)[rows, ids], dweight=dense.T@grad)
        errors = {}
        for stage, fn in ops.items():
            actual = fn()
            actual = actual[0] if stage == "forward" else actual
            torch.testing.assert_close(actual, expected[stage], rtol=2e-4, atol=2e-5)
            errors[stage] = (actual-expected[stage]).abs().max().item()
        reports.append(dict(engine=engine, max_absolute_errors=errors, selected_zeros=True))
    return reports


def measure(name, cfg, args, torch):
    data = inputs(cfg, torch)
    ops, groups = operations(data, cfg)
    metrics = {}
    for stage, fn in ops.items():
        for _ in range(args.warmup):
            result = fn()
            del result
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.steps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            a.record()
            result = fn()
            b.record()
            b.synchronize()
            samples.append(dict(wall_ms=(time.perf_counter()-start)*1000, cuda_ms=a.elapsed_time(b)))
            del result
        metrics[stage] = dict(wall_ms=statistics.median(s["wall_ms"] for s in samples),
                              cuda_ms=statistics.median(s["cuda_ms"] for s in samples),
                              peak_extra_bytes=torch.cuda.max_memory_allocated()-baseline)
    regime = dict(page_groups=sorted({(g.base, g.k) for g in groups}) if groups is not None else [],
                  engine=cfg["engine"])
    del ops, data, groups
    gc.collect()
    torch.cuda.empty_cache()
    return dict(name=name, config=cfg, metrics=metrics, regime=regime)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=9)
    args = p.parse_args()
    if args.warmup < 1 or args.steps < 3:
        p.error("Need warmup>=1 and steps>=3")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"profile.json").exists():
        p.error("Choose a fresh output directory")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dump(args.output/"correctness.json", verify(torch))
    cal, hold = configurations()
    dump(args.output/"calibration_configs.json", cal)
    dump(args.output/"holdout_configs.json", hold)
    rows = []
    with (args.output/"calibration.jsonl").open("x") as stream:
        for name, cfg in cal.items():
            row = measure(name, cfg, args, torch)
            rows.append(row)
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            print("CAL", name, flush=True)
    files = [p for p in (REPO/"sae_lens").rglob("*.py") if "autoconfig" not in p.parts]
    files += [Path(__file__), REPO/"sae_lens/autoconfig/interpolation_model.py",
              REPO/"sae_lens/autoconfig/execution_time_model.py"]
    profile = dict(schema="sparse_parts_interpolation_v1", axes=["rows", "width"], rows=rows,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                    gpu=torch.cuda.get_device_name()),
                   source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                   warmup=args.warmup, steps=args.steps)
    dump(args.output/"profile.json", profile)
    predictions = {name: predict_sparse(cfg, profile) for name, cfg in hold.items()}
    dump(args.output/"predictions.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), predictions=predictions))
    validation = []
    with (args.output/"holdout.jsonl").open("x") as stream:
        for name, cfg in hold.items():
            obs = measure(name, cfg, args, torch)
            stream.write(json.dumps(obs)+"\n")
            stream.flush()
            pred = predictions[name]
            # JSON serialization normalizes tuple page keys to lists.
            if json.dumps(obs["regime"], sort_keys=True) != json.dumps(pred["regime"], sort_keys=True):
                raise ValueError("Holdout changed sparse algorithm regime")
            for stage in obs["metrics"]:
                for metric, actual in obs["metrics"][stage].items():
                    predicted = pred["metrics"][stage][metric]
                    validation.append(dict(name=name, stage=stage, metric=metric, actual=actual,
                                           predicted=predicted, error_pct=100*(predicted/actual-1)))
            print("HOLD", name, flush=True)
    summary = {metric: dict(mape_pct=statistics.mean(abs(r["error_pct"]) for r in validation if r["metric"] == metric),
                            max_error_pct=max(abs(r["error_pct"]) for r in validation if r["metric"] == metric))
               for metric in ("wall_ms", "cuda_ms", "peak_extra_bytes")}
    dump(args.output/"validation.json", dict(summary=summary, rows=validation))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
