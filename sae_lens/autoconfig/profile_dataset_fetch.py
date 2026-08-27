#!/usr/bin/env python3
"""Profile the dataset -> vLLM-step-input segment of SAE training.

Scope
=====
This profiler measures exactly the interval that ends where the vLLM activation
profilers begin:

    HF dataset row read
      -> per-sequence slice/clone/dtype cast (+ per-sequence .to(device))
      -> torch.stack into (B, ctx)
      -> [optional] token broadcast to non-root TP ranks
      -> batch_tokens ready as the input of model.run_with_cache

It deliberately excludes vLLM prefill, hook capture, activation routing,
mixing_buffer, and all SAE compute.  Those are covered by
``profile_vllm_two_stage_v4.py``, ``profile_activation_routing.py`` and
``simulate_sae_step_time_step_v6.py``.

It drives the real ``ActivationsStore`` (not a reimplementation), so the
measured path is the one training uses: ``_get_batch_tokens_local`` for the
local read and ``get_batch_tokens`` for the broadcast variant.

Two device conventions are measured separately because they differ by ~100x on
a busy GPU:

  * ``per_seq_h2d``  -- current code: every sequence is moved to ``device``
    individually inside ``_iterate_tokenized_sequences_unsharded``, so one batch
    issues B separate 16 KB H2D copies.
  * ``cpu_stack``    -- ``act_store_device=cpu``: sequences stay on CPU and the
    single (B, ctx) tensor is moved once.

DP sharding is measured with the real strided iterator, so the reported cost
includes the rows this rank skips (``i % shard_count == shard_idx``): with
map-style datasets a skipped row is still fully decoded.

Stability
=========
``--stability-mode`` runs a long single-config trace and reports the latency
distribution plus a cold/warm split, which is what determines whether disk
reads are a jitter source or a flat additive cost for the step-time model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

DEFAULT_OUTPUT_DIR = Path("sae_lens/autoconfig/profile_results/dataset_fetch")


@dataclass
class Case:
    """One measured configuration."""

    case_id: str
    batch_size: int
    context_size: int
    shard_count: int
    shard_index: int
    device_mode: str  # per_seq_h2d | cpu_stack
    streaming: bool
    tp_size: int  # 1 = local read only; >1 = read + CPU broadcast
    warmup: int
    repeats: int


@dataclass
class Result:
    case: dict[str, Any]
    status: str
    samples: list[float] = field(default_factory=list)
    error: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def _summarize(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)
    n = len(ordered)

    def q(p: float) -> float:
        return ordered[min(n - 1, int(p * n))]

    mean = statistics.mean(ordered)
    return {
        "samples": float(n),
        "mean_ms": mean,
        "median_ms": statistics.median(ordered),
        "std_ms": statistics.stdev(ordered) if n > 1 else 0.0,
        "cov": (statistics.stdev(ordered) / mean) if (n > 1 and mean > 0) else 0.0,
        "min_ms": ordered[0],
        "p50_ms": q(0.50),
        "p90_ms": q(0.90),
        "p99_ms": q(0.99),
        "max_ms": ordered[-1],
    }


def build_store(
    *,
    dataset_path: str,
    model_name: str,
    batch_size: int,
    context_size: int,
    shard_index: int,
    shard_count: int,
    device: str,
    streaming: bool,
    train_batch_size_tokens: int,
):
    """Construct the real ActivationsStore against a tokenizer-only model.

    A tokenizer-only model is enough: this profiler stops at ``batch_tokens``
    and never calls ``run_with_cache``, so no weights are loaded.
    """
    from sae_lens.load_model import load_tokenizer_only_model
    from sae_lens.training.activations_store import ActivationsStore

    model = load_tokenizer_only_model(model_name, device)
    return ActivationsStore(
        model=model,  # type: ignore[arg-type]
        dataset=dataset_path,
        streaming=streaming,
        hook_name="blocks.0.hook_resid_post",
        hook_head_index=None,
        context_size=context_size,
        d_in=1,  # unused: no activation is ever produced here
        n_batches_in_buffer=1,
        total_training_tokens=batch_size * context_size,
        store_batch_size_prompts=batch_size,
        train_batch_size_tokens=train_batch_size_tokens,
        prepend_bos=False,
        normalize_activations="none",
        device=torch.device(device),
        dtype="float32",
        dataset_shard_index=shard_index,
        dataset_shard_count=shard_count,
    )


def measure_local_fetch(
    store: Any,
    *,
    batch_size: int,
    device_mode: str,
    device: str,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Time ``_get_batch_tokens_local`` + the device move for one batch.

    ``per_seq_h2d`` reproduces the current code path, where the store's own
    iterator already moved each sequence to ``device`` one at a time.
    ``cpu_stack`` keeps sequences on CPU and issues a single H2D for the batch.
    """
    dev = torch.device(device)
    use_cuda = dev.type == "cuda"

    def one() -> None:
        batch = store._get_batch_tokens_local(batch_size=batch_size)
        if device_mode == "cpu_stack" and use_cuda:
            batch.to(dev)
        if use_cuda:
            torch.cuda.synchronize(dev)

    for _ in range(warmup):
        one()

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        one()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def measure_broadcast_fetch(
    store: Any,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Time ``get_batch_tokens`` including the root-read + CPU broadcast.

    Requires an initialized worker group (``init_distributed``); every TP rank
    must call this together.  Only the root rank actually reads the dataset.
    """
    for _ in range(warmup):
        store.get_batch_tokens(batch_size=batch_size)

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        store.get_batch_tokens(batch_size=batch_size)
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def run_case(
    case: Case,
    *,
    dataset_path: str,
    model_name: str,
    device: str,
    train_batch_size_tokens: int,
) -> Result:
    store_device = device if case.device_mode == "per_seq_h2d" else "cpu"
    try:
        store = build_store(
            dataset_path=dataset_path,
            model_name=model_name,
            batch_size=case.batch_size,
            context_size=case.context_size,
            shard_index=case.shard_index,
            shard_count=case.shard_count,
            device=store_device,
            streaming=case.streaming,
            train_batch_size_tokens=train_batch_size_tokens,
        )
        if case.tp_size > 1:
            samples = measure_broadcast_fetch(
                store,
                batch_size=case.batch_size,
                warmup=case.warmup,
                repeats=case.repeats,
            )
        else:
            samples = measure_local_fetch(
                store,
                batch_size=case.batch_size,
                device_mode=case.device_mode,
                device=device,
                warmup=case.warmup,
                repeats=case.repeats,
            )
    except Exception as exc:  # noqa: BLE001 - recorded per case, scan continues
        return Result(
            case=asdict(case), status="error", error=f"{type(exc).__name__}: {exc}"
        )

    metrics = _summarize(samples)
    tokens = case.batch_size * case.context_size
    if metrics.get("mean_ms", 0.0) > 0:
        metrics["tokens_per_s"] = tokens / (metrics["mean_ms"] / 1e3)
        metrics["rows_read_per_step"] = float(tokens * case.shard_count)
    metrics["useful_tokens_per_step"] = float(tokens)
    return Result(case=asdict(case), status="ok", samples=samples, metrics=metrics)


def build_cases(args: argparse.Namespace) -> list[Case]:
    cases: list[Case] = []
    for ctx in args.context_sizes:
        for b in args.batch_sizes:
            for sc in args.shard_counts:
                for mode in args.device_modes:
                    cases.append(
                        Case(
                            case_id=f"ctx{ctx}_B{b}_shard{sc}_{mode}_tp{args.tp_size}",
                            batch_size=b,
                            context_size=ctx,
                            shard_count=sc,
                            shard_index=0,
                            device_mode=mode,
                            streaming=args.streaming,
                            tp_size=args.tp_size,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                    )
    return cases


def run_stability(args: argparse.Namespace) -> dict[str, Any]:
    """Long single-config trace: is the disk read a jitter source or flat cost?

    Reports the full latency distribution, plus a first-N vs last-N split so a
    cold page-cache region is distinguishable from steady state.
    """
    b = args.batch_sizes[0]
    ctx = args.context_sizes[0]
    mode = args.device_modes[0]
    store_device = args.device if mode == "per_seq_h2d" else "cpu"
    store = build_store(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        batch_size=b,
        context_size=ctx,
        shard_index=0,
        shard_count=args.shard_counts[0],
        device=store_device,
        streaming=args.streaming,
        train_batch_size_tokens=args.train_batch_size_tokens,
    )
    samples = measure_local_fetch(
        store,
        batch_size=b,
        device_mode=mode,
        device=args.device,
        warmup=0,
        repeats=args.stability_steps,
    )
    head = max(1, len(samples) // 10)
    return {
        "batch_size": b,
        "context_size": ctx,
        "device_mode": mode,
        "streaming": args.streaming,
        "steps": len(samples),
        "overall": _summarize(samples),
        "first_decile": _summarize(samples[:head]),
        "last_decile": _summarize(samples[-head:]),
        "trace_ms": samples,
    }


def write_outputs(
    results: list[Result],
    stability: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for res in results:
        row: dict[str, Any] = {"status": res.status, "error": res.error}
        row.update(res.case)
        row.update(res.metrics)
        rows.append(row)

    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        csv_path = output_dir / "dataset_fetch_profile.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"wrote {csv_path}")

    payload: dict[str, Any] = {"cases": [asdict(r) for r in results]}
    if stability is not None:
        payload["stability"] = stability
    json_path = output_dir / "dataset_fetch_profile.json"
    json_path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {json_path}")


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="../datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048",
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=_int_list, default=[8, 16, 32, 64])
    parser.add_argument("--context-sizes", type=_int_list, default=[2048])
    parser.add_argument(
        "--shard-counts",
        type=_int_list,
        default=[1, 2, 4],
        help="vLLM DP degrees to emulate via the store's strided sharding.",
    )
    parser.add_argument(
        "--device-modes",
        type=_str_list,
        default=["per_seq_h2d", "cpu_stack"],
        help="per_seq_h2d = current act_store_device=cuda; cpu_stack = act_store_device=cpu.",
    )
    parser.add_argument("--train-batch-size-tokens", type=int, default=4096)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="1 profiles the local read; >1 profiles get_batch_tokens incl. broadcast "
        "and requires a torchrun launch with an initialized worker group.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--stability-mode", action="store_true")
    parser.add_argument("--stability-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.tp_size > 1:
        from sae_lens.distributed import init_distributed

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        init_distributed(sae_dp_size=1, vllm_tp_size=args.tp_size)

    stability = run_stability(args) if args.stability_mode else None
    if stability is not None:
        overall = stability["overall"]
        print(
            f"stability: mean {overall['mean_ms']:.2f} ms  p99 {overall['p99_ms']:.2f}  "
            f"max {overall['max_ms']:.2f}  CoV {overall['cov']:.3f}  "
            f"first-decile mean {stability['first_decile']['mean_ms']:.2f} -> "
            f"last-decile mean {stability['last_decile']['mean_ms']:.2f}"
        )

    results: list[Result] = []
    for case in build_cases(args):
        res = run_case(
            case,
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            device=args.device,
            train_batch_size_tokens=args.train_batch_size_tokens,
        )
        results.append(res)
        if res.status == "ok":
            m = res.metrics
            print(
                f"{case.case_id}: mean {m['mean_ms']:.2f} ms  p90 {m['p90_ms']:.2f}  "
                f"max {m['max_ms']:.2f}  CoV {m['cov']:.3f}  "
                f"{m.get('tokens_per_s', 0.0) / 1e6:.2f} M useful tok/s"
            )
        else:
            print(f"{case.case_id}: {res.error}")

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        write_outputs(results, stability, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
