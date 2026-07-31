#!/usr/bin/env python3
"""Phase-aware, low-dimensional SAE compute profiler.

Design contract
---------------
This profiler deliberately does *not* profile whole SAE phases as high-dimensional
black boxes. Every timed surface is keyed only by the natural dimensions of the
operation family that owns the kernels:

* GEMM: (B, D, F_local)
* BD local work: (B, D)
* BF local work: (B, F_local)
* DF local work: (D, F_local)
* global statistics: (B, F_global)
* D-only local work: (D)
* TP wrapper-local BF/BD/D work: corresponding natural shape + tp
* optimizer: (D, F_local) for one SAE

Rows additionally carry ``phase`` and ``schedule_stage``. Those are scheduling
metadata for the simulator; they are NOT extra interpolation dimensions.

The stage split follows the current TopK SAE autograd graph used by SAELens and
the unified DDP/FSDP Nsight traces:

forward_core
  preprocess/encode/norm/activation/decode/loss for one hook.
backward_pre_wdec_ready
  loss -> decoder backward -> activation backward -> decoder-norm branches;
  after this stage the first large matrix gradient (W_dec) is modeled ready.
backward_wdec_to_wenc_ready
  encoder matmul backward; after this stage W_enc is modeled ready.
backward_post_wenc
  input-preprocess backward / b_dec branch accumulation.
post_backward
  local gradient clipping before optimizer.
stats
  per-hook feature statistics (unified trainer runs these after root forward).

The exact DDP bucket packing is intentionally *not* profiled here. The simulator
constructs buckets from parameter bytes and these release events.
"""
from __future__ import annotations

# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS -- preserve the current profile grid
# =============================================================================
DEFAULT_D_IN_VALUES: list[int] = [1024, 4096]
DEFAULT_LOCAL_D_SAE_VALUES: list[int] = [8192, 16384, 32768, 65536, 131072]
DEFAULT_BATCH_SIZES: list[int] = [256, 512, 1024, 2048, 4096]
DEFAULT_TP_VALUES: list[int] = [1, 2]
DEFAULT_OPTIMIZER_IMPLS: list[str] = ["fused"]
DEFAULT_STATS_SYNC_MODE: str = "immediate"
DEFAULT_STATS_SYNC_INTERVAL: int = 1
DEFAULT_NORMALIZE_ACTIVATIONS: str = "none"
DEFAULT_DEVICES: list[str] = ["cuda:0"]
DEFAULT_DTYPES: list[str] = ["float32"]
DEFAULT_WARMUP: int = 5
DEFAULT_REPEATS: int = 8
DEFAULT_OUTPUT_DIR: str = "sae_lens/autoconfig/profile_results"
DEFAULT_OUTPUT_NAME: str = "sae_compute_profile"
DEFAULT_SUBPROCESS_ISOLATION: bool = True
DEFAULT_TASK_TIMEOUT_SECONDS: float = 1800.0
# =============================================================================

import argparse
import csv
import gc
import itertools
import json
import math
import multiprocessing as mp
import os
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype {name!r}")
    return DTYPE_ALIASES[key]


def canonical_dtype(name: str) -> str:
    dt = parse_dtype(name)
    return {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}[dt]


def uniq(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def pct(xs: Sequence[float], q: float) -> float:
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1 - w) + xs[hi] * w)


@dataclass(frozen=True)
class TimingStats:
    samples: int
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    p10_ms: float
    p90_ms: float
    max_ms: float


def summarize(samples: Sequence[float]) -> TimingStats:
    xs = sorted(float(x) for x in samples)
    return TimingStats(
        samples=len(xs),
        median_ms=float(statistics.median(xs)),
        mean_ms=float(statistics.fmean(xs)),
        std_ms=float(statistics.pstdev(xs) if len(xs) > 1 else 0.0),
        min_ms=xs[0], p10_ms=pct(xs, 0.10), p90_ms=pct(xs, 0.90), max_ms=xs[-1],
    )


def add_stats(parts: Sequence[TimingStats]) -> TimingStats:
    """Add independently measured GPU-op groups without creating a larger live tensor set."""
    if not parts:
        return TimingStats(0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    return TimingStats(
        samples=min(p.samples for p in parts),
        median_ms=sum(p.median_ms for p in parts),
        mean_ms=sum(p.mean_ms for p in parts),
        std_ms=math.sqrt(sum(p.std_ms*p.std_ms for p in parts)),
        min_ms=sum(p.min_ms for p in parts),
        p10_ms=sum(p.p10_ms for p in parts),
        p90_ms=sum(p.p90_ms for p in parts),
        max_ms=sum(p.max_ms for p in parts),
    )


class Timer:
    def __init__(self, device: torch.device, warmup: int, repeats: int) -> None:
        self.device, self.warmup, self.repeats = device, warmup, repeats

    def measure(self, op: Callable[[], Any], *, inference: bool = True) -> TimingStats:
        if self.device.type == "cuda":
            ctx = torch.inference_mode() if inference else _Null()
            with torch.cuda.device(self.device), ctx:
                for _ in range(self.warmup):
                    y = op(); torch.cuda.synchronize(self.device); del y
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                samples: list[float] = []
                for _ in range(self.repeats):
                    start.record(); y = op(); end.record(); end.synchronize()
                    samples.append(float(start.elapsed_time(end))); del y
            return summarize(samples)
        ctx = torch.inference_mode() if inference else _Null()
        with ctx:
            for _ in range(self.warmup):
                y = op(); del y
            samples = []
            for _ in range(self.repeats):
                t0 = time.perf_counter_ns(); y = op(); t1 = time.perf_counter_ns(); del y
                samples.append((t1 - t0) / 1e6)
        return summarize(samples)

    def measure_staged(
        self,
        forward: Callable[[], Any],
        prepare_backward: Callable[[Any], Any],
        backward: Callable[[Any, Any], Any],
    ) -> tuple[TimingStats, TimingStats]:
        """Measure forward and backward separately without timing setup/upstream allocation."""
        fwd_samples: list[float] = []
        bwd_samples: list[float] = []
        if self.device.type != "cuda":
            for i in range(self.warmup + self.repeats):
                t0 = time.perf_counter_ns(); out = forward(); t1 = time.perf_counter_ns()
                upstream = prepare_backward(out)
                t2 = time.perf_counter_ns(); bout = backward(out, upstream); t3 = time.perf_counter_ns()
                if i >= self.warmup:
                    fwd_samples.append((t1-t0)/1e6); bwd_samples.append((t3-t2)/1e6)
                del bout, upstream, out
            return summarize(fwd_samples), summarize(bwd_samples)

        with torch.cuda.device(self.device):
            for _ in range(self.warmup):
                out = forward(); torch.cuda.synchronize(self.device)
                upstream = prepare_backward(out); bout = backward(out, upstream)
                torch.cuda.synchronize(self.device); del bout, upstream, out
            fs, fe = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            bs, be = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(self.repeats):
                fs.record(); out = forward(); fe.record(); fe.synchronize()
                fwd_samples.append(float(fs.elapsed_time(fe)))
                upstream = prepare_backward(out)
                bs.record(); bout = backward(out, upstream); be.record(); be.synchronize()
                bwd_samples.append(float(bs.elapsed_time(be)))
                del bout, upstream, out
        return summarize(fwd_samples), summarize(bwd_samples)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *args): return False


@dataclass(frozen=True)
class Case:
    B: int
    D: int
    F: int
    tp: int
    optimizer_impl: str
    device: str
    dtype_name: str

    @property
    def F_global(self) -> int: return self.F * self.tp
    @property
    def dtype(self) -> torch.dtype: return parse_dtype(self.dtype_name)
    @property
    def torch_device(self) -> torch.device: return torch.device(self.device)


CSV_FIELDS = [
    "task_index","task_wall_seconds","profiler","status","error_type","error_message","error_traceback",
    "semantic_op","phase","schedule_stage","release_event","shape_family","driving_vars","equation","op_list",
    "B","D","F_local_d_sae","F_global_d_sae","tp","parameter_numel","stats_sync_mode","stats_sync_interval","normalize_activations","device","dtype","optimizer_impl",
    "warmup","repeats","samples","median_ms","mean_ms","std_ms","min_ms","p10_ms","p90_ms","max_ms",
]


def blank() -> dict[str, Any]: return {k: "" for k in CSV_FIELDS}


def row_base(case: Case, semantic_op: str, *, profiler: str, phase: str, stage: str,
             release: str, family: str, driving: str, equation: str, op_list: Sequence[str]) -> dict[str, Any]:
    r = blank()
    r.update({
        "profiler": profiler, "status": "ok", "semantic_op": semantic_op,
        "phase": phase, "schedule_stage": stage, "release_event": release,
        "shape_family": family, "driving_vars": driving, "equation": equation,
        "op_list": json.dumps(list(op_list), separators=(",", ":")),
        "B": case.B if "B" in driving.split(",") else "",
        "D": case.D if "D" in driving.split(",") else "",
        "F_local_d_sae": case.F if "F_local" in driving.split(",") else "",
        "F_global_d_sae": case.F_global if "F_global" in driving.split(",") else "",
        "tp": case.tp if "tp" in driving.split(",") else "",
        "device": str(case.torch_device), "dtype": case.dtype_name,
        "optimizer_impl": case.optimizer_impl if profiler == "optimizer" else "",
    })
    return r


def finish_row(r: dict[str, Any], stats: TimingStats, warmup: int, repeats: int) -> dict[str, Any]:
    r.update({"warmup": warmup, "repeats": repeats, **asdict(stats)})
    return r


def rand(shape: Sequence[int], c: Case, *, grad: bool=False, positive: bool=False) -> torch.Tensor:
    x = torch.randn(tuple(shape), device=c.torch_device, dtype=c.dtype)
    if positive: x = x.abs().add_(0.5)
    x.requires_grad_(grad)
    return x


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()


# ---------- task implementations ----------

def profile_gemm(c: Case, warmup: int, repeats: int) -> list[dict[str, Any]]:
    timer = Timer(c.torch_device, warmup, repeats)
    specs = [
        ("encoder_forward","forward","forward_core","", "[B,D]@[D,F]", (c.B,c.D),(c.D,c.F)),
        ("decoder_forward","forward","forward_core","", "[B,F]@[F,D]", (c.B,c.F),(c.F,c.D)),
        ("decoder_dgrad","backward","backward_pre_wdec_ready","", "[B,D]@[D,F]", (c.B,c.D),(c.D,c.F)),
        ("decoder_wgrad","backward","backward_pre_wdec_ready","", "[F,B]@[B,D]", (c.F,c.B),(c.B,c.D)),
        ("encoder_dgrad","backward","backward_wdec_to_wenc_ready","", "[B,F]@[F,D]", (c.B,c.F),(c.F,c.D)),
        ("encoder_wgrad","backward","backward_wdec_to_wenc_ready","ddp_wenc_ready", "[D,B]@[B,F]", (c.D,c.B),(c.B,c.F)),
    ]
    out=[]
    for name, phase, stage, release, eq, s1, s2 in specs:
        a=rand(s1,c); b=rand(s2,c)
        stats=timer.measure(lambda a=a,b=b: torch.mm(a,b))
        r=row_base(c,name,profiler="gemm",phase=phase,stage=stage,release=release,
                   family="GEMM",driving="B,D,F_local",equation=eq,op_list=["torch.mm"])
        out.append(finish_row(r,stats,warmup,repeats)); del a,b
    return out


def profile_bd(c: Case, warmup: int, repeats: int, normalize: str) -> list[dict[str, Any]]:
    timer=Timer(c.torch_device,warmup,repeats); rows=[]
    x=rand((c.B,c.D),c,grad=True); b=rand((c.D,),c,grad=True)
    expected_average_scale=1.125
    def pf():
        if normalize=="constant_norm_rescale":
            coeff=(c.D**0.5)/x.norm(dim=-1,keepdim=True); sae_in=x*coeff
        elif normalize=="layer_norm":
            mu=x.mean(dim=-1,keepdim=True); centered=x-mu; std=centered.std(dim=-1,keepdim=True); sae_in=centered/(std+1e-5)
        elif normalize=="expected_average_only_in":
            sae_in=x*expected_average_scale
        else:
            sae_in=x
        return sae_in-b
    def pbprep(o): return torch.empty_like(o)
    def pb(o,u): return torch.autograd.grad(o,(x,b),grad_outputs=u)
    sf,sb=timer.measure_staged(pf,pbprep,pb)
    r=row_base(c,"preprocess_bd_forward",profiler="preprocess",phase="forward",stage="forward_core",release="",family="BD",driving="B,D",equation="process_sae_in forward",op_list=["normalization if configured","subtract b_dec"]); r["normalize_activations"]=normalize
    rows.append(finish_row(r,sf,warmup,repeats))
    r=row_base(c,"preprocess_bd_backward",profiler="preprocess",phase="backward",stage="backward_post_wenc",release="ddp_backward_compute_done",family="BD",driving="B,D",equation="process_sae_in backward",op_list=["normalization backward","b_dec encode-branch reduction"]); r["normalize_activations"]=normalize
    rows.append(finish_row(r,sb,warmup,repeats)); del x,b

    sae_in=rand((c.B,c.D),c,grad=True); dec=rand((c.B,c.D),c,grad=True); b=rand((c.D,),c,grad=True)
    norm_aux: tuple[torch.Tensor,...]
    if normalize=="constant_norm_rescale": norm_aux=(rand((c.B,1),c,grad=True,positive=True),)
    elif normalize=="layer_norm": norm_aux=(rand((c.B,1),c,grad=True),rand((c.B,1),c,grad=True,positive=True))
    else: norm_aux=()
    def lf():
        out=dec+b
        if normalize=="constant_norm_rescale": out=out/norm_aux[0]
        elif normalize=="layer_norm": out=out*norm_aux[1]+norm_aux[0]
        return (out-sae_in).pow(2).sum(dim=-1).mean()
    def lbprep(o): return torch.ones_like(o)
    def lb(o,u): return torch.autograd.grad(o,(dec,sae_in,b,*norm_aux),grad_outputs=u)
    sf,sb=timer.measure_staged(lf,lbprep,lb)
    r=row_base(c,"local_bd_loss_forward",profiler="local_compute",phase="forward",stage="forward_core",release="",family="BD",driving="B,D",equation="decoder output restore + reconstruction loss",op_list=["decoder+b_dec","output normalization restore if configured","MSE"]); r["normalize_activations"]=normalize
    rows.append(finish_row(r,sf,warmup,repeats))
    r=row_base(c,"local_bd_loss_backward",profiler="local_compute",phase="backward",stage="backward_pre_wdec_ready",release="",family="BD",driving="B,D",equation="reconstruction-loss backward",op_list=["loss backward to decoder output/input/b_dec"]); r["normalize_activations"]=normalize
    rows.append(finish_row(r,sb,warmup,repeats))
    return rows


def profile_d(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    timer=Timer(c.torch_device,warmup,repeats)
    a=rand((c.D,),c); b=rand((c.D,),c)
    stats=timer.measure(lambda: a+b)
    return [finish_row(row_base(c,"local_d_bdec_grad_accum",profiler="local_compute",phase="backward",stage="backward_post_wenc",release="ddp_small_grads_ready",family="D",driving="D",equation="b_dec grad branch accumulation",op_list=["vector add"]),stats,warmup,repeats)]


def profile_bf(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    """BF-local phase rows with bounded live memory.

    The old staged-autograd aggregate can hold input, forward output, upstream
    and multiple backward outputs simultaneously for the largest B*F point.
    Here we measure the same natural BF kernels as separate sub-ops in ONE task
    and add their isolated GPU times. This keeps the model two-dimensional and
    avoids profiler-only OOM at otherwise meaningful points.
    """
    timer=Timer(c.torch_device,warmup,repeats); rows=[]
    mm=rand((c.B,c.F),c); up=rand((c.B,c.F),c); b=rand((c.F,),c); norm=rand((c.F,),c,positive=True)
    ef=add_stats([
        timer.measure(lambda: mm+b),
        timer.measure(lambda: mm*norm),
    ])
    eb=add_stats([
        timer.measure(lambda: up*norm),
        timer.measure(lambda: up.sum(dim=0)),
        timer.measure(lambda: (up*mm).sum(dim=0)),
    ])
    rows.append(finish_row(row_base(c,"local_bf_encoder_forward",profiler="local_compute",phase="forward",stage="forward_core",release="",family="BF_LOCAL",driving="B,F_local",equation="encoder bias add + decoder-norm scale",op_list=["[B,F]+[F]","[B,F]*[F]"]),ef,warmup,repeats))
    rows.append(finish_row(row_base(c,"local_bf_encoder_backward",profiler="local_compute",phase="backward",stage="backward_pre_wdec_ready",release="",family="BF_LOCAL",driving="B,F_local",equation="encoder BF backward",op_list=["grad hidden","reduce b_enc grad","reduce decoder-norm upstream"]),eb,warmup,repeats))
    del mm,up,b,norm; cleanup(c.torch_device)

    acts=rand((c.B,c.F),c); up=rand((c.B,c.F),c); inv=rand((c.F,),c,positive=True)
    df=timer.measure(lambda: acts*inv)
    db=add_stats([
        timer.measure(lambda: up*inv),
        timer.measure(lambda: (up*acts).sum(dim=0)),
    ])
    rows.append(finish_row(row_base(c,"local_bf_decoder_forward",profiler="local_compute",phase="forward",stage="forward_core",release="",family="BF_LOCAL",driving="B,F_local",equation="decoder inverse-norm scale",op_list=["[B,F]*[F]"]),df,warmup,repeats))
    rows.append(finish_row(row_base(c,"local_bf_decoder_backward",profiler="local_compute",phase="backward",stage="backward_pre_wdec_ready",release="",family="BF_LOCAL",driving="B,F_local",equation="decoder BF backward",op_list=["grad feature acts","reduce inverse-norm upstream"]),db,warmup,repeats))
    return rows


def profile_df(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    timer=Timer(c.torch_device,warmup,repeats); rows=[]
    w=rand((c.F,c.D),c,grad=True)
    def nf(): return w.norm(dim=-1), 1.0/w.norm(dim=-1)
    def prep(o): return torch.empty_like(o[0]), torch.empty_like(o[1])
    def nb(o,u): return torch.autograd.grad(o,w,grad_outputs=u)[0]
    sf,sb=timer.measure_staged(nf,prep,nb)
    rows.append(finish_row(row_base(c,"local_fd_norm_forward",profiler="local_compute",phase="forward",stage="forward_core",release="",family="FD_LOCAL",driving="D,F_local",equation="two W_dec row norms + reciprocal",op_list=["W_dec.norm x2","reciprocal"]),sf,warmup,repeats))
    rows.append(finish_row(row_base(c,"local_fd_norm_backward",profiler="local_compute",phase="backward",stage="backward_pre_wdec_ready",release="ddp_wdec_ready",family="FD_LOCAL",driving="D,F_local",equation="combined W_dec norm backward",op_list=["two norm backward paths accumulate to W_dec"]),sb,warmup,repeats))
    return rows


def profile_clip(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    """Profile clip without adding a new dimension.

    total matches single/DDP torch.nn.utils.clip_grad_norm_.  norm_scan and
    grad_scale expose the FSDP dependency boundary around the DP scalar AR.
    All three rows remain on the same natural (D,F_local) surface and are
    measured in one task/process to limit profiling overhead.
    """
    timer=Timer(c.torch_device,warmup,repeats)
    ps=(torch.nn.Parameter(torch.empty((c.D,c.F),device=c.torch_device,dtype=c.dtype)),
        torch.nn.Parameter(torch.empty((c.F,c.D),device=c.torch_device,dtype=c.dtype)),
        torch.nn.Parameter(torch.empty((c.F,),device=c.torch_device,dtype=c.dtype)),
        torch.nn.Parameter(torch.empty((c.D,),device=c.torch_device,dtype=c.dtype)))
    for p in ps: p.grad=torch.full_like(p,1e-8)
    total=timer.measure(lambda: torch.nn.utils.clip_grad_norm_(ps,1.0), inference=True)

    grads=[p.grad for p in ps if p.grad is not None]
    def norm_scan():
        acc=torch.zeros((),device=c.torch_device,dtype=torch.float32)
        for g in grads: acc.add_(g.float().pow(2).sum())
        return acc
    norm=timer.measure(norm_scan,inference=True)
    coef=torch.tensor(0.999,device=c.torch_device,dtype=torch.float32)
    def scale():
        for g in grads: g.mul_(coef.to(device=g.device,dtype=g.dtype))
        return grads[0]
    scale_stats=timer.measure(scale,inference=True)
    n=sum(p.numel() for p in ps)
    rows=[]
    for op,st,eq,ops in [
        ("local_grad_clip_total",total,"torch.nn.utils.clip_grad_norm_ over one full SAE",["norm scan","clip coefficient","gradient scale"]),
        ("local_grad_norm_scan",norm,"FSDP-compatible local squared-gradient norm scan",["grad.float().pow(2).sum per parameter"]),
        ("local_grad_scale",scale_stats,"FSDP-compatible local gradient scale",["multiply each gradient by clip coefficient"]),
    ]:
        r=row_base(c,op,profiler="local_compute",phase="post_backward",stage="post_backward",release="",family="PARAM_DF",driving="D,F_local",equation=eq,op_list=ops); r["parameter_numel"]=n
        rows.append(finish_row(r,st,warmup,repeats))
    return rows


def profile_stats(c: Case,warmup:int,repeats:int,mode:str) -> list[dict[str,Any]]:
    timer=Timer(c.torch_device,warmup,repeats)
    feats=rand((c.B,c.F_global),c)
    scores=torch.zeros((c.F_global,),device=c.torch_device,dtype=torch.float32)
    nf=torch.zeros_like(scores); pending=torch.zeros((c.F_global,),device=c.torch_device,dtype=torch.int32)
    if mode=="immediate":
        def op():
            firing=feats.bool().float(); did=firing.sum(dim=-2).bool().to(torch.int32).contiguous(); scores.add_(firing.sum(dim=0)); nf.add_(1); nf[did.bool()]=0; return did
    else:
        def op():
            firing=feats.bool().float(); did=firing.sum(dim=-2).bool().to(torch.int32).contiguous(); scores.add_(firing.sum(dim=0)); return torch.maximum(pending,did)
    stats=timer.measure(op)
    r=row_base(c,"local_bf_global_stats",profiler="local_compute",phase="stats",stage="stats",release="",family="BF_GLOBAL",driving="B,F_global",equation="dense feature firing statistics; NCCL excluded",op_list=["bool/float","reductions over B","feature-vector update"])
    r["F_global_d_sae"]=c.F_global; r["stats_sync_mode"]=mode
    return [finish_row(r,stats,warmup,repeats)]


def profile_tp_bf(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    if c.tp<=1: return []
    timer=Timer(c.torch_device,warmup,repeats); rows=[]
    local=rand((c.B,c.F),c)
    stats=timer.measure(lambda: torch.cat([torch.zeros_like(local) for _ in range(c.tp)],dim=-1))
    rows.append(finish_row(row_base(c,"local_bf_tp_forward",profiler="local_compute",phase="forward",stage="tp_layout_forward",release="",family="BF_LOCAL_TP",driving="B,F_local,tp",equation="TP allgather wrapper output allocation/cat; NCCL excluded",op_list=["zeros_like x tp","cat"]),stats,warmup,repeats))
    full=rand((c.B,c.F_global),c)
    stats=timer.measure(lambda: full[...,:c.F].contiguous())
    rows.append(finish_row(row_base(c,"local_bf_tp_backward",profiler="local_compute",phase="backward",stage="tp_layout_backward",release="",family="BF_LOCAL_TP",driving="B,F_local,tp",equation="TP allgather backward shard slice",op_list=["slice","contiguous"]),stats,warmup,repeats))
    return rows


def profile_tp_bd(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    if c.tp<=1: return []
    timer=Timer(c.torch_device,warmup,repeats); rows=[]
    x=rand((c.B,c.D),c); stats=timer.measure(lambda: x.clone())
    rows.append(finish_row(row_base(c,"local_bd_tp_forward",profiler="local_compute",phase="forward",stage="tp_layout_forward",release="",family="BD_TP",driving="B,D,tp",equation="TP decoder allreduce input clone",op_list=["clone"]),stats,warmup,repeats))
    b=rand((c.D,),c,grad=True)
    def f(): return _ScaleGrad.apply(b,1.0/c.tp)
    def prep(o): return torch.empty_like(o)
    def bw(o,u): return torch.autograd.grad(o,b,grad_outputs=u)[0]
    _,sb=timer.measure_staged(f,prep,bw)
    rows.append(finish_row(row_base(c,"local_d_tp_backward",profiler="local_compute",phase="backward",stage="tp_layout_backward",release="",family="D_TP",driving="D,tp",equation="decode-bias 1/tp gradient scaling",op_list=["identity autograd","gradient multiply"]),sb,warmup,repeats))
    return rows


class _ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale): ctx.scale=scale; return x
    @staticmethod
    def backward(ctx,g): return g*ctx.scale,None


def profile_optimizer(c: Case,warmup:int,repeats:int) -> list[dict[str,Any]]:
    params=[torch.nn.Parameter(rand((c.D,c.F),c)),torch.nn.Parameter(rand((c.F,c.D),c)),torch.nn.Parameter(rand((c.F,),c)),torch.nn.Parameter(rand((c.D,),c))]
    impl=c.optimizer_impl
    kwargs={"lr":1e-3,"foreach": True if impl=="foreach" else False if impl=="forloop" else None,
            "fused": True if impl=="fused" else None}
    kwargs={k:v for k,v in kwargs.items() if v is not None}
    opt=torch.optim.Adam(params,**kwargs)
    for p in params: p.grad=torch.randn_like(p)
    opt.step()  # initialize state
    for p in params: p.grad=torch.randn_like(p)
    timer=Timer(c.torch_device,warmup,repeats)
    stats=timer.measure(lambda: opt.step(),inference=False)
    n=sum(p.numel() for p in params)
    r=row_base(c,f"{impl}_adam_steady_step",profiler="optimizer",phase="optimizer",stage="optimizer",release="",family="OPT_DF",driving="D,F_local",equation="Adam steady step for one SAE",op_list=["torch.optim.Adam.step"])
    r["parameter_numel"]=n
    return [finish_row(r,stats,warmup,repeats)]


TASK_FAMILIES=("gemm","bd","d","bf","df","clip","stats","tp_bf","tp_bd","optimizer")


def task_key(fam:str,c:Case,stats_mode:str,normalize:str)->tuple[Any,...]:
    dev=(c.device,c.dtype_name)
    if fam=="gemm": return fam,c.B,c.D,c.F,*dev
    if fam=="bd": return fam,c.B,c.D,normalize,*dev
    if fam=="d": return fam,c.D,*dev
    if fam=="bf": return fam,c.B,c.F,*dev
    if fam in {"df","clip"}: return fam,c.D,c.F,*dev
    if fam=="stats": return fam,c.B,c.F_global,stats_mode,*dev
    if fam=="tp_bf": return fam,c.B,c.F,c.tp,*dev
    if fam=="tp_bd": return fam,c.B,c.D,c.tp,*dev
    if fam=="optimizer": return fam,c.D,c.F,c.optimizer_impl,*dev
    raise KeyError(fam)


def run_family(fam:str,c:Case,warmup:int,repeats:int,stats_mode:str,normalize:str)->list[dict[str,Any]]:
    if fam=="gemm": return profile_gemm(c,warmup,repeats)
    if fam=="bd": return profile_bd(c,warmup,repeats,normalize)
    if fam=="d": return profile_d(c,warmup,repeats)
    if fam=="bf": return profile_bf(c,warmup,repeats)
    if fam=="df": return profile_df(c,warmup,repeats)
    if fam=="clip": return profile_clip(c,warmup,repeats)
    if fam=="stats": return profile_stats(c,warmup,repeats,stats_mode)
    if fam=="tp_bf": return profile_tp_bf(c,warmup,repeats)
    if fam=="tp_bd": return profile_tp_bd(c,warmup,repeats)
    if fam=="optimizer": return profile_optimizer(c,warmup,repeats)
    raise KeyError(fam)


def worker(q:mp.Queue,fam:str,c:Case,warmup:int,repeats:int,stats_mode:str,normalize:str)->None:
    try:
        torch.manual_seed(0)
        if c.torch_device.type=="cuda": torch.cuda.set_device(c.torch_device)
        q.put((True,run_family(fam,c,warmup,repeats,stats_mode,normalize)))
    except BaseException as exc:
        q.put((False,(type(exc).__name__,str(exc),traceback.format_exc(limit=12))))


def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--d-in",nargs="+",type=int,default=DEFAULT_D_IN_VALUES)
    p.add_argument("--local-d-sae",nargs="+",type=int,default=DEFAULT_LOCAL_D_SAE_VALUES)
    p.add_argument("--batch-size",nargs="+",type=int,default=DEFAULT_BATCH_SIZES)
    p.add_argument("--tp",nargs="+",type=int,default=DEFAULT_TP_VALUES)
    p.add_argument("--optimizer-impl",nargs="+",default=DEFAULT_OPTIMIZER_IMPLS,choices=["foreach","forloop","fused","default"])
    p.add_argument("--stats-sync-mode",default=DEFAULT_STATS_SYNC_MODE,choices=["immediate","periodic","deferred"])
    p.add_argument("--stats-sync-interval",type=int,default=DEFAULT_STATS_SYNC_INTERVAL)
    p.add_argument("--normalize-activations",default=DEFAULT_NORMALIZE_ACTIVATIONS,choices=["none","expected_average_only_in","constant_norm_rescale","layer_norm"])
    p.add_argument("--device",nargs="+",default=DEFAULT_DEVICES)
    p.add_argument("--dtype",nargs="+",default=DEFAULT_DTYPES)
    p.add_argument("--warmup",type=int,default=DEFAULT_WARMUP)
    p.add_argument("--repeats",type=int,default=DEFAULT_REPEATS)
    p.add_argument("--output-dir",default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-name",default=DEFAULT_OUTPUT_NAME)
    p.add_argument("--no-subprocess-isolation",action="store_true")
    p.add_argument("--task-timeout-seconds",type=float,default=DEFAULT_TASK_TIMEOUT_SECONDS)
    return p.parse_args(argv)


def main(argv:Sequence[str]|None=None)->int:
    args=parse_args(argv)
    cases=[Case(B,D,F,tp,opt,dev,canonical_dtype(dt)) for D,F,B,tp,opt,dev,dt in itertools.product(args.d_in,args.local_d_sae,args.batch_size,args.tp,args.optimizer_impl,args.device,args.dtype)]
    tasks=[]; seen=set()
    for c in cases:
        for fam in TASK_FAMILIES:
            if fam.startswith("tp_") and c.tp<=1: continue
            k=task_key(fam,c,args.stats_sync_mode,args.normalize_activations)
            if k in seen: continue
            seen.add(k); tasks.append((fam,c))
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    stem=Path(args.output_name).stem; csvp=outdir/f"{stem}.csv"; jp=outdir/f"{stem}.json"; i=1
    while csvp.exists() or jp.exists(): csvp=outdir/f"{stem}_{i}.csv"; jp=outdir/f"{stem}_{i}.json"; i+=1
    print(f"[phase-profiler] tasks={len(tasks)} CSV={csvp}")
    all_rows=[]; t_all=time.perf_counter()
    ctx=mp.get_context("spawn")
    for idx,(fam,c) in enumerate(tasks,1):
        t0=time.perf_counter(); print(f"[{idx}/{len(tasks)}] {fam} B={c.B} D={c.D} F={c.F} tp={c.tp} {c.device} {c.dtype_name}",flush=True)
        if args.no_subprocess_isolation:
            try: rows=run_family(fam,c,args.warmup,args.repeats,args.stats_sync_mode,args.normalize_activations)
            except Exception as exc:
                rows=[]; err=(type(exc).__name__,str(exc),traceback.format_exc(limit=12))
        else:
            q=ctx.Queue(); p=ctx.Process(target=worker,args=(q,fam,c,args.warmup,args.repeats,args.stats_sync_mode,args.normalize_activations)); p.start(); p.join(None if args.task_timeout_seconds<=0 else args.task_timeout_seconds)
            if p.is_alive():
                p.terminate(); p.join(5); rows=[]; err=("TimeoutError",f"task exceeded {args.task_timeout_seconds}s","")
            elif not q.empty():
                ok,payload=q.get(); rows=payload if ok else []; err=None if ok else payload
            else:
                rows=[]; err=("WorkerError",f"worker exitcode={p.exitcode}","")
        if not rows:
            r=blank(); r.update({"profiler":fam,"status":"error","error_type":err[0],"error_message":err[1],"error_traceback":err[2],"B":c.B,"D":c.D,"F_local_d_sae":c.F,"F_global_d_sae":c.F_global,"tp":c.tp,"device":c.device,"dtype":c.dtype_name})
            rows=[r]
        elapsed=time.perf_counter()-t0
        for r in rows: r["task_index"]=idx; r["task_wall_seconds"]=elapsed
        all_rows.extend(rows)
        if args.no_subprocess_isolation:
            cleanup(c.torch_device)
        with csvp.open("w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=CSV_FIELDS,extrasaction="ignore"); w.writeheader(); w.writerows(all_rows)
    summary={"schema_version":"phase_lowdim_v1","wall_seconds":time.perf_counter()-t_all,"rows":all_rows,
             "contract":{"phase_is_not_interpolation_dimension":True,"overlap_contention_profiled":False,"default_points_preserved":True}}
    jp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"CSV : {csvp}\nJSON: {jp}")
    return 0

if __name__=="__main__": raise SystemExit(main())
