#!/usr/bin/env python3
"""External profiler for streaming activation transfer.

This script profiles only raw transfer cost:

* host -> device copy for a byte buffer of a given size
* device -> host copy for a byte buffer of a given size

The only sweep dimension is transfer size in bytes.  Warmup is the only other
user-facing knob.  The output is intentionally unprocessed: no chunks, no
window simulation, no buffer state machine, and no downstream compute.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import socket
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================
_KIB = 1024
_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024

DEFAULT_BUFFER_BYTES: list[int] = [
    512,
    1 * _KIB,
    2 * _KIB,
    4 * _KIB,
    8 * _KIB,
    16 * _KIB,
    32 * _KIB,
    64 * _KIB,
    128 * _KIB,
    256 * _KIB,
    512 * _KIB,
    1 * _MIB,
    2 * _MIB,
    4 * _MIB,
    8 * _MIB,
    16 * _MIB,
    32 * _MIB,
    64 * _MIB,
    128 * _MIB,
    256 * _MIB,
    512 * _MIB,
    1 * _GIB,
]
DEFAULT_WARMUP: int = 8
DEFAULT_REPEATS: int = 16
DEFAULT_OUTPUT_DIR: str = "sae_lens/autoconfig/profile_results/streaming_transfer"
DEFAULT_OUTPUT_NAME: str = "streaming_transfer_profile"
# =============================================================================


DTYPE_NAME = "uint8"
DEVICE_NAME = "cuda:0"


def canonical_bytes_label(num_bytes: int) -> str:
    if num_bytes >= _GIB and num_bytes % _GIB == 0:
        return f"{num_bytes // _GIB}GiB"
    if num_bytes >= _MIB and num_bytes % _MIB == 0:
        return f"{num_bytes // _MIB}MiB"
    if num_bytes >= _KIB and num_bytes % _KIB == 0:
        return f"{num_bytes // _KIB}KiB"
    return f"{num_bytes}B"


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def pct(xs: Sequence[float], q: float) -> float:
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1 - w) + xs[hi] * w)


def summarize(samples: Sequence[float]) -> "TimingStats":
    xs = sorted(float(x) for x in samples)
    return TimingStats(
        samples=len(xs),
        median_ms=float(statistics.median(xs)),
        mean_ms=float(statistics.fmean(xs)),
        std_ms=float(statistics.pstdev(xs) if len(xs) > 1 else 0.0),
        min_ms=float(xs[0]),
        p10_ms=pct(xs, 0.10),
        p90_ms=pct(xs, 0.90),
        max_ms=float(xs[-1]),
    )


@dataclass(frozen=True)
class TransferCase:
    buffer_bytes: int
    direction: str
    device: str = DEVICE_NAME
    dtype_name: str = DTYPE_NAME

    @property
    def case_id(self) -> str:
        return f"{self.direction}_buf{self.buffer_bytes}_{self.dtype_name}_{self.device.replace(':', '_')}"


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


@dataclass
class TransferSweepConfig:
    buffer_bytes_values: list[int]
    warmup: int = DEFAULT_WARMUP
    repeats: int = DEFAULT_REPEATS
    output_dir: str = DEFAULT_OUTPUT_DIR
    output_name: str = DEFAULT_OUTPUT_NAME

    def normalize(self) -> "TransferSweepConfig":
        self.buffer_bytes_values = unique_preserve_order(
            int(x) for x in self.buffer_bytes_values
        )
        return self

    def validate_inputs(self) -> None:
        if not self.buffer_bytes_values:
            raise ValueError("buffer_bytes_values must not be empty")
        invalid = [x for x in self.buffer_bytes_values if x <= 0]
        if invalid:
            raise ValueError(f"buffer_bytes_values must be positive integers; got {invalid}")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _measure_transfer(case: TransferCase, warmup: int, repeats: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for streaming transfer profiling")

    device = torch.device(case.device)
    if device.type != "cuda":
        raise ValueError(f"Unsupported device for transfer profiling: {case.device}")

    torch.cuda.set_device(device)
    gc.collect()
    torch.cuda.empty_cache()

    src_cpu = torch.empty(case.buffer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    dst_cpu = torch.empty(case.buffer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    src_gpu = torch.empty(case.buffer_bytes, dtype=torch.uint8, device=device)
    dst_gpu = torch.empty(case.buffer_bytes, dtype=torch.uint8, device=device)

    # Touch once so we measure transfer bandwidth, not first-touch allocation work.
    src_cpu.fill_(1)
    dst_cpu.zero_()
    src_gpu.zero_()
    dst_gpu.zero_()

    def op() -> None:
        if case.direction == "h2d":
            dst_gpu.copy_(src_cpu, non_blocking=True)
        elif case.direction == "d2h":
            dst_cpu.copy_(src_gpu, non_blocking=True)
        else:
            raise ValueError(f"Unknown direction: {case.direction}")

    for _ in range(warmup):
        op()
        torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        op()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)

    stats = summarize(samples)
    return {
        "case_id": case.case_id,
        "buffer_bytes": case.buffer_bytes,
        "buffer_label": canonical_bytes_label(case.buffer_bytes),
        "direction": case.direction,
        "device": case.device,
        "dtype_name": case.dtype_name,
        "warmup": warmup,
        "repeats": repeats,
        "samples_ms": samples,
        "stats": asdict(stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-bytes-values", type=str, default=",".join(str(x) for x in DEFAULT_BUFFER_BYTES))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()

    buffer_bytes_values = unique_preserve_order(
        int(part) for part in args.buffer_bytes_values.replace(" ", "").split(",") if part
    )
    config = TransferSweepConfig(
        buffer_bytes_values=buffer_bytes_values,
        warmup=args.warmup,
        repeats=DEFAULT_REPEATS,
        output_dir=str(args.output_dir),
        output_name=args.output_name,
    ).normalize()
    config.validate_inputs()

    cases = [
        TransferCase(buffer_bytes=size, direction=direction)
        for size in config.buffer_bytes_values
        for direction in ("h2d", "d2h")
    ]

    results = [_measure_transfer(case, config.warmup, config.repeats) for case in cases]

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [
        {
            "case_id": row["case_id"],
            "buffer_bytes": row["buffer_bytes"],
            "buffer_label": row["buffer_label"],
            "direction": row["direction"],
            "device": row["device"],
            "dtype_name": row["dtype_name"],
            "warmup": row["warmup"],
            "repeats": row["repeats"],
            "samples_ms": json.dumps(row["samples_ms"]),
            **row["stats"],
        }
        for row in results
    ]

    _write_csv(out_dir / f"{config.output_name}.csv", csv_rows)
    _atomic_write_json(
        out_dir / f"{config.output_name}.json",
        {
            "config": asdict(config),
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
        },
    )

    print(json.dumps(asdict(config), indent=2, sort_keys=True))
    for row in results[: min(len(results), 10)]:
        stats = row["stats"]
        print(
            f"{row['direction']} {row['buffer_label']} "
            f"median={stats['median_ms']:.4f}ms mean={stats['mean_ms']:.4f}ms "
            f"std={stats['std_ms']:.4f}ms"
        )


if __name__ == "__main__":
    main()
