#!/usr/bin/env python3
"""External profiler for a replaceable SAE activation boundary.

The timed region owns every kernel whose implementation changes when the SAE
activation changes.  For dense TopK this is the complete
``topk -> ReLU -> zeros_like -> scatter`` forward path and its autograd
backward.  ReLU is also supported as a second implementation.  An explicit
upstream gradient is created only after forward, so no synthetic ``sum``
reduction or leaf ``.grad`` accumulation is included.

The model key is ``(B, F_global, activation_type, activation parameters,
dtype, output_layout)``.  TP is deliberately not a single-device model
dimension: legacy ``F_local * tp`` inputs are canonicalized and deduplicated
to one global width.  Optional distributed mode may use a real all-gather to
construct the input, but the collective remains outside activation timing.
"""

from __future__ import annotations

# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================
DEFAULT_BATCH_SIZES: list[int] = [512, 1024, 2048, 4096, 8192, 16384]
# local d_sae: F_local = global d_sae / tp
DEFAULT_LOCAL_D_SAE_VALUES: list[int] = [8192, 16384, 32768, 65536, 131072]
# Optional direct global widths. Legacy F_local*tp inputs are canonicalized to these.
DEFAULT_GLOBAL_D_SAE_VALUES: list[int] = []
DEFAULT_TP_VALUES: list[int] = [1, 2]
DEFAULT_K_VALUES: list[int] = [128]
DEFAULT_ACTIVATION_TYPES: list[str] = ["topk"]
DEFAULT_DEVICES: list[str] = ["cuda:0"]
DEFAULT_DTYPES: list[str] = ["float32"]

DEFAULT_WARMUP: int = 5
DEFAULT_REPEATS: int = 8
DEFAULT_OUTPUT_DIR: str = "sae_lens/autoconfig/profile_results"
DEFAULT_OUTPUT_NAME: str = "activation_compute_profile"
# median/min above this ratio marks a case as contended. TopK on an idle A40
# reproduces to std 0.01ms (ratio 1.000), and an sgemm neighbour pushes it to
# ~2.2x, so 1.15 separates real interference from measurement jitter.
DEFAULT_INTERFERENCE_RATIO: float = 1.15
# =============================================================================

import argparse
import csv
import gc
import itertools
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


def canonical_dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    return mapping.get(dtype, str(dtype).removeprefix("torch."))


def parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in DTYPE_ALIASES:
        supported = ", ".join(sorted(DTYPE_ALIASES))
        raise ValueError(f"Unsupported dtype {name!r}. Supported aliases: {supported}")
    return DTYPE_ALIASES[key]


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


@dataclass(frozen=True)
class ActivationCase:
    """One point in the activation-compute profile space."""

    batch_size: int
    local_d_sae: int
    tp: int
    k: int
    activation_type: str
    device: str
    dtype_name: str

    @property
    def B(self) -> int:
        return self.batch_size

    @property
    def F_local(self) -> int:
        return self.local_d_sae

    @property
    def F_global(self) -> int:
        return self.local_d_sae * self.tp

    @property
    def dtype(self) -> torch.dtype:
        return parse_dtype(self.dtype_name)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def case_id(self) -> str:
        safe_device = self.device.replace(":", "_").replace("/", "_")
        return (
            f"B{self.B}_Flocal{self.F_local}_tp{self.tp}_act{self.activation_type}_k{self.k}_"
            f"{safe_device}_{self.dtype_name}"
        )


@dataclass
class ActivationSweepConfig:
    batch_sizes: list[int]
    local_d_sae_values: list[int]
    tp_values: list[int]
    global_d_sae_values: list[int]
    k_values: list[int]
    activation_types: list[str]
    devices: list[str]
    dtypes: list[str]

    warmup: int = 20
    repeats: int = 50
    seed: int = 0
    output_dir: str = "profile_results"
    output_name: str = DEFAULT_OUTPUT_NAME
    fail_fast: bool = False
    empty_cache_between_ops: bool = True
    distributed: bool = False
    profile_forward_only: bool = True
    interference_ratio: float = DEFAULT_INTERFERENCE_RATIO
    allow_shared_device: bool = False

    def normalize(self) -> "ActivationSweepConfig":
        self.batch_sizes = unique_preserve_order(int(x) for x in self.batch_sizes)
        self.local_d_sae_values = unique_preserve_order(
            int(x) for x in self.local_d_sae_values
        )
        self.tp_values = unique_preserve_order(int(x) for x in self.tp_values)
        self.global_d_sae_values = unique_preserve_order(
            int(x) for x in self.global_d_sae_values
        )
        self.k_values = unique_preserve_order(int(x) for x in self.k_values)
        self.activation_types = unique_preserve_order(
            str(x).strip().lower() for x in self.activation_types
        )
        self.devices = unique_preserve_order(str(x) for x in self.devices)
        self.dtypes = unique_preserve_order(
            canonical_dtype_name(parse_dtype(str(x))) for x in self.dtypes
        )
        return self

    def validate_inputs(self) -> None:
        arrays = {
            "batch_sizes": self.batch_sizes,
            "local_d_sae_values": self.local_d_sae_values,
            "tp_values": self.tp_values,
            "global_d_sae_values": self.global_d_sae_values,
            "k_values": self.k_values,
            "activation_types": self.activation_types,
            "devices": self.devices,
            "dtypes": self.dtypes,
        }
        for name in (
            "batch_sizes",
            "tp_values",
            "k_values",
            "activation_types",
            "devices",
            "dtypes",
        ):
            if not arrays[name]:
                raise ValueError(f"{name} must not be empty")
        if not self.local_d_sae_values and not self.global_d_sae_values:
            raise ValueError(
                "Provide local_d_sae_values or global_d_sae_values"
            )
        for name in ("batch_sizes", "local_d_sae_values", "tp_values", "k_values"):
            invalid = [x for x in arrays[name] if x <= 0]
            if invalid:
                raise ValueError(f"{name} must be positive integers; got {invalid}")
        invalid_global = [x for x in self.global_d_sae_values if x <= 0]
        if invalid_global:
            raise ValueError(
                "global_d_sae_values must be positive integers; "
                f"got {invalid_global}"
            )
        valid_activations = {"topk", "relu"}
        invalid_activations = [
            value for value in self.activation_types if value not in valid_activations
        ]
        if invalid_activations:
            raise ValueError(
                f"Unsupported activation types {invalid_activations}; "
                f"choices={sorted(valid_activations)}"
            )
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")
        for case in self.iter_cases():
            if case.activation_type == "topk" and case.k > case.F_global:
                raise ValueError(
                    f"k={case.k} exceeds global width F_local*tp={case.F_global}"
                )

    def iter_cases(self) -> Iterable[ActivationCase]:
        """Yield the activation space keyed by global width, not TP.

        Single-device profiling canonicalizes every legacy ``F_local * tp``
        combination to one ``F_global`` case, so the same activation graph is
        never re-profiled merely because it came from a different TP layout.
        Distributed mode retains the requested TP/local-shard pair because a
        real all-gather is used to construct the input before timing activation.
        """
        if self.distributed:
            shard_specs: list[tuple[int, int]] = [
                (F, tp)
                for F, tp in itertools.product(
                    self.local_d_sae_values, self.tp_values
                )
            ]
            for width in self.global_d_sae_values:
                for tp in self.tp_values:
                    if width % tp == 0:
                        shard_specs.append((width // tp, tp))
            shard_specs = unique_preserve_order(shard_specs)
        else:
            global_widths = list(self.global_d_sae_values)
            global_widths.extend(
                F * tp
                for F, tp in itertools.product(
                    self.local_d_sae_values, self.tp_values
                )
            )
            # Canonical single-device representation: F_local == F_global, tp=1.
            shard_specs = [(width, 1) for width in unique_preserve_order(global_widths)]

        for B, shard_spec, activation_type, device, dtype_name in itertools.product(
            self.batch_sizes,
            shard_specs,
            self.activation_types,
            self.devices,
            self.dtypes,
        ):
            F, tp = shard_spec
            k_values = self.k_values if activation_type == "topk" else [0]
            for k in k_values:
                yield ActivationCase(
                    batch_size=B,
                    local_d_sae=F,
                    tp=tp,
                    k=k,
                    activation_type=activation_type,
                    device=device,
                    dtype_name=dtype_name,
                )


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
    # Contention diagnostics. clean_ms == min_ms: the fastest window, which is the
    # true cost only if at least one repeat ran uncontended. interference_ratio is
    # median/min, so it catches contention that came and went but NOT contention
    # that lasted the whole run -- for that see foreign_pids_on_device.
    clean_ms: float = 0.0
    interference_ratio: float = 1.0
    interference_suspected: bool = False


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_samples(
    samples_ms: Sequence[float],
    interference_ratio: float = DEFAULT_INTERFERENCE_RATIO,
) -> TimingStats:
    if not samples_ms:
        raise ValueError("No timing samples were collected")
    ordered = sorted(float(x) for x in samples_ms)
    fastest = float(ordered[0])
    median = float(statistics.median(ordered))
    ratio = median / fastest if fastest > 0 else 1.0
    return TimingStats(
        samples=len(ordered),
        median_ms=median,
        mean_ms=float(statistics.fmean(ordered)),
        std_ms=float(statistics.pstdev(ordered) if len(ordered) > 1 else 0.0),
        min_ms=fastest,
        p10_ms=percentile(ordered, 0.10),
        p90_ms=percentile(ordered, 0.90),
        max_ms=float(ordered[-1]),
        clean_ms=fastest,
        interference_ratio=ratio,
        interference_suspected=ratio > interference_ratio,
    )


def other_process_pids_on_device(device: torch.device) -> list[int]:
    """PIDs other than ours with a compute context on ``device``, via nvidia-smi.

    A second CUDA context on the same card time-slices with ours, so a CUDA-event
    window measures our kernels *plus* the wait and the bandwidth we lost. TopK's
    zeros_like/scatter_ are bandwidth-bound and slow by ~2.2x under an sgemm
    neighbour, which is large enough to swamp the effect being profiled.
    """
    if device.type != "cuda":
        return []
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    mine = os.getpid()
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text or not text.isdigit():
            continue
        pid = int(text)
        # Our own worker subprocesses share the parent's device legitimately only
        # if they are not running kernels concurrently; the profiler runs one case
        # at a time, so any *other* pid is a foreign context.
        if pid != mine:
            pids.append(pid)
    return pids


def measure_cuda(
    operation: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeats: int,
    interference_ratio: float = DEFAULT_INTERFERENCE_RATIO,
) -> TimingStats:
    """Time an autograd op (fwd+bwd) with CUDA events. inference_mode is OFF.

    Each repeat gets its own event window, preceded by a sync so the window
    brackets that repeat alone. Interference can only make a window slower, so
    ``min_ms`` beats the median when contention is *intermittent*, and
    ``interference_ratio`` (median/min) reports how much the run drifted.

    This does NOT rescue a run that was contended throughout: if a neighbour holds
    the card for every repeat, all samples are uniformly inflated and min_ms is
    inflated with them (measured: 34.7 ms min under load vs 16.4 ms idle, with
    ratio 1.05 -- below any sane threshold). Only ``foreign_pids_on_device``
    detects that case, which is why ``require_exclusive_devices`` refuses to start
    on a shared GPU rather than relying on the statistics.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    with torch.cuda.device(device):
        for _ in range(warmup):
            output = operation()
            torch.cuda.synchronize(device)
            del output
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples_ms: list[float] = []
        for _ in range(repeats):
            # Drain anything still queued so the window brackets this repeat only.
            torch.cuda.synchronize(device)
            start.record()
            output = operation()
            end.record()
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))
            # Do not keep the previous full-width result/gradient alive while the
            # next invocation allocates its own outputs.
            del output
    return summarize_samples(samples_ms, interference_ratio=interference_ratio)


def measure_cpu(
    operation: Callable[[], Any],
    warmup: int,
    repeats: int,
    interference_ratio: float = DEFAULT_INTERFERENCE_RATIO,
) -> TimingStats:
    for _ in range(warmup):
        output = operation()
        del output
    samples_ms: list[float] = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        output = operation()
        end_ns = time.perf_counter_ns()
        samples_ms.append((end_ns - start_ns) / 1_000_000.0)
        del output
    return summarize_samples(samples_ms, interference_ratio=interference_ratio)


def measure(
    operation: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeats: int,
    interference_ratio: float = DEFAULT_INTERFERENCE_RATIO,
) -> TimingStats:
    if device.type == "cuda":
        return measure_cuda(operation, device, warmup, repeats, interference_ratio)
    return measure_cpu(operation, warmup, repeats, interference_ratio)


def cleanup_device(device: torch.device, empty_cache: bool) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        if empty_cache:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _topk_forward(x: torch.Tensor, k: int) -> torch.Tensor:
    """Dense SAELens TopK activation, including ReLU and scatter."""
    topk_values, topk_indices = torch.topk(x, k=k, dim=-1, sorted=False)
    values = topk_values.relu()
    result = torch.zeros_like(x)
    result.scatter_(-1, topk_indices, values)
    return result


def _activation_forward(
    x: torch.Tensor, activation_type: str, k: int
) -> torch.Tensor:
    if activation_type == "topk":
        return _topk_forward(x, k)
    if activation_type == "relu":
        return torch.relu(x)
    raise ValueError(f"Unsupported activation_type={activation_type!r}")


def activation_op_list(activation_type: str, *, backward: bool) -> list[str]:
    if activation_type == "topk":
        ops = ["torch.topk(sorted=False)", "relu", "zeros_like", "scatter_"]
    elif activation_type == "relu":
        ops = ["relu"]
    else:
        raise ValueError(f"Unsupported activation_type={activation_type!r}")
    if backward:
        ops.append("activation backward with explicit upstream gradient")
    return ops


def make_forward_op(
    full_width_input: torch.Tensor, activation_type: str, k: int
) -> Callable[[], torch.Tensor]:
    """Forward-only activation; no autograd graph retained."""

    def op() -> torch.Tensor:
        with torch.no_grad():
            return _activation_forward(full_width_input, activation_type, k)

    return op


def measure_forward_backward(
    full_width_input: torch.Tensor,
    activation_type: str,
    k: int,
    device: torch.device,
    warmup: int,
    repeats: int,
    interference_ratio: float = DEFAULT_INTERFERENCE_RATIO,
) -> TimingStats:
    """Measure the complete activation forward and backward boundaries.

    For TopK, this includes ``torch.topk``, ReLU, zeros/scatter and the
    corresponding backward. These kernels are deliberately excluded from all
    local-compute rows so changing the activation only replaces this table.
    """

    def one_iteration(*, timed: bool) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

            fwd_start.record()
            acts = _activation_forward(full_width_input, activation_type, k)
            fwd_end.record()
            fwd_end.synchronize()

            upstream_grad = torch.empty_like(acts)
            bwd_start.record()
            (grad_input,) = torch.autograd.grad(
                outputs=acts,
                inputs=full_width_input,
                grad_outputs=upstream_grad,
                retain_graph=False,
                create_graph=False,
            )
            bwd_end.record()
            bwd_end.synchronize()
            elapsed = float(fwd_start.elapsed_time(fwd_end)) + float(
                bwd_start.elapsed_time(bwd_end)
            )
            del grad_input, upstream_grad, acts
            return elapsed if timed else 0.0

        fwd_start_ns = time.perf_counter_ns()
        acts = _activation_forward(full_width_input, activation_type, k)
        fwd_end_ns = time.perf_counter_ns()
        upstream_grad = torch.empty_like(acts)
        bwd_start_ns = time.perf_counter_ns()
        (grad_input,) = torch.autograd.grad(
            outputs=acts,
            inputs=full_width_input,
            grad_outputs=upstream_grad,
            retain_graph=False,
            create_graph=False,
        )
        bwd_end_ns = time.perf_counter_ns()
        elapsed = (
            (fwd_end_ns - fwd_start_ns) + (bwd_end_ns - bwd_start_ns)
        ) / 1_000_000.0
        del grad_input, upstream_grad, acts
        return elapsed if timed else 0.0

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        with torch.cuda.device(device):
            scratch = torch.empty_like(full_width_input)
            del scratch
            for _ in range(warmup):
                one_iteration(timed=False)
            samples_ms = [one_iteration(timed=True) for _ in range(repeats)]
    else:
        scratch = torch.empty_like(full_width_input)
        del scratch
        for _ in range(warmup):
            one_iteration(timed=False)
        samples_ms = [one_iteration(timed=True) for _ in range(repeats)]

    return summarize_samples(samples_ms, interference_ratio=interference_ratio)


@dataclass
class PreparedActivation:
    semantic_op: str
    op_list: list[str]
    operation: Callable[[], torch.Tensor] | None
    input_shape: list[int]
    upstream_shape: list[int] | None
    requires_grad: bool


class ActivationProfiler:
    """Profiles complete replaceable activation forward/backward boundaries."""

    name = "activation_compute"

    def profile_case(
        self,
        case: ActivationCase,
        config: ActivationSweepConfig,
        *,
        full_width_input: torch.Tensor | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> list[dict[str, Any]]:
        device = case.torch_device
        dtype = case.dtype
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(f"Unsupported dtype: {dtype}")

        results: list[dict[str, Any]] = []
        builders: list[tuple[str, bool]] = []
        if config.profile_forward_only:
            builders.append(("activation_forward", False))
        builders.append(("activation_forward_backward", True))

        for semantic_op, needs_grad in builders:
            prepared: PreparedActivation | None = None
            x: torch.Tensor | None = None
            op: Callable[[], torch.Tensor] | None = None
            op_started = time.perf_counter()
            try:
                x = self._build_input(case, full_width_input, needs_grad)
                if needs_grad:
                    op = None
                    op_list = activation_op_list(
                        case.activation_type, backward=True
                    )
                    prepared = PreparedActivation(
                        semantic_op,
                        op_list,
                        None,
                        list(x.shape),
                        list(x.shape),
                        needs_grad,
                    )
                    stats = measure_forward_backward(
                        x,
                        case.activation_type,
                        case.k,
                        device,
                        config.warmup,
                        config.repeats,
                        config.interference_ratio,
                    )
                else:
                    op = make_forward_op(
                        x, case.activation_type, case.k
                    )
                    op_list = activation_op_list(
                        case.activation_type, backward=False
                    )
                    prepared = PreparedActivation(
                        semantic_op,
                        op_list,
                        op,
                        list(x.shape),
                        None,
                        needs_grad,
                    )
                    stats = measure(
                        op,
                        device,
                        config.warmup,
                        config.repeats,
                        config.interference_ratio,
                    )
                row = self._success_row(case, config, prepared, stats, rank, world_size)
            except Exception as exc:  # noqa: BLE001
                row = self._error_row(case, config, semantic_op, exc, rank, world_size)
                if config.fail_fast:
                    raise
            finally:
                op_elapsed = time.perf_counter() - op_started
                # Explicitly break closures/references before the next semantic op
                # builds another [B,F_global] input. Otherwise forward-only data
                # can survive into forward+backward and create a profiler-only OOM.
                prepared = None
                op = None
                x = None
                gc.collect()
                if full_width_input is None:
                    cleanup_device(device, config.empty_cache_between_ops)
            row["component_wall_seconds"] = op_elapsed
            results.append(row)
        return results

    def _build_input(
        self,
        case: ActivationCase,
        full_width_input: torch.Tensor | None,
        needs_grad: bool,
    ) -> torch.Tensor:
        if full_width_input is not None:
            x = full_width_input.detach()
        else:
            x = torch.randn(
                (case.B, case.F_global), device=case.torch_device, dtype=case.dtype
            )
        x.requires_grad_(needs_grad)
        return x

    def _success_row(
        self,
        case: ActivationCase,
        config: ActivationSweepConfig,
        prepared: PreparedActivation,
        stats: TimingStats,
        rank: int,
        world_size: int,
    ) -> dict[str, Any]:
        return {
            "profiler": self.name,
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "error_traceback": "",
            "case_id": case.case_id,
            "semantic_op": prepared.semantic_op,
            "phase": (
                "activation_forward_backward"
                if prepared.semantic_op == "activation_forward_backward"
                else "activation_forward"
            ),
            "shape_family": "BF_GLOBAL",
            "driving_vars": (
                "B,F_global,k,activation_type,output_layout"
                if case.activation_type == "topk"
                else "B,F_global,activation_type,output_layout"
            ),
            "activation_type": case.activation_type,
            "output_layout": "dense",
            "op_list": prepared.op_list,
            "input_shape": prepared.input_shape,
            "upstream_shape": prepared.upstream_shape or "",
            "B": case.B,
            "F_local": case.F_local,
            "tp": case.tp,
            "F_global": case.F_global,
            "k": case.k,
            "device": str(case.torch_device),
            "dtype": case.dtype_name,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "rank": rank,
            "world_size": world_size,
            "distributed": config.distributed,
            # Foreign contexts are re-checked per case: a neighbour can appear
            # mid-sweep, and only the cases it overlapped are affected.
            "foreign_pids_on_device": other_process_pids_on_device(case.torch_device),
            **asdict(stats),
        }

    def _error_row(
        self,
        case: ActivationCase,
        config: ActivationSweepConfig,
        semantic_op: str,
        exc: Exception,
        rank: int,
        world_size: int,
    ) -> dict[str, Any]:
        row = blank_row()
        row.update(
            {
                "profiler": self.name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(limit=12),
                "case_id": case.case_id,
                "semantic_op": semantic_op,
                "phase": (
                    "activation_forward_backward"
                    if semantic_op == "activation_forward_backward"
                    else "activation_forward"
                ),
                "shape_family": "BF_GLOBAL",
                "driving_vars": (
                    "B,F_global,k,activation_type,output_layout"
                    if case.activation_type == "topk"
                    else "B,F_global,activation_type,output_layout"
                ),
                "activation_type": case.activation_type,
                "output_layout": "dense",
                "B": case.B,
                "F_local": case.F_local,
                "tp": case.tp,
                "F_global": case.F_global,
                "k": case.k,
                "device": str(case.torch_device),
                "dtype": case.dtype_name,
                "warmup": config.warmup,
                "repeats": config.repeats,
                "rank": rank,
                "world_size": world_size,
                "distributed": config.distributed,
                "samples": 0,
            }
        )
        return row


CSV_FIELDS: list[str] = [
    "profiler",
    "status",
    "error_type",
    "error_message",
    "error_traceback",
    "case_id",
    "semantic_op",
    "phase",
    "shape_family",
    "driving_vars",
    "activation_type",
    "output_layout",
    "op_list",
    "input_shape",
    "upstream_shape",
    "B",
    "F_local",
    "tp",
    "F_global",
    "k",
    "device",
    "dtype",
    "warmup",
    "repeats",
    "rank",
    "world_size",
    "distributed",
    "foreign_pids_on_device",
    "component_wall_seconds",
    "samples",
    "median_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "p10_ms",
    "p90_ms",
    "max_ms",
    "clean_ms",
    "interference_ratio",
    "interference_suspected",
]


def blank_row() -> dict[str, Any]:
    return {key: "" for key in CSV_FIELDS}


def csv_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def unique_output_paths(output_dir: Path, requested_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(requested_name).stem or "activation_compute_profile"
    suffix = 0
    while True:
        candidate = stem if suffix == 0 else f"{stem}_{suffix}"
        csv_path = output_dir / f"{candidate}.csv"
        json_path = output_dir / f"{candidate}.json"
        if not csv_path.exists() and not json_path.exists():
            return csv_path, json_path
        suffix += 1


def build_metadata(config: ActivationSweepConfig, started_at: datetime) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "profiler": "activation_compute",
        "started_at_utc": started_at.isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(config),
        "semantics": {
            "profiled_width": "(B, F_global) — post-allgather full d_sae",
            "activation_boundary": "all activation-specific forward/backward kernels; local profiler excludes them",
            "backward": "explicit upstream gradient created only at backward entry; no synthetic sum reduction or leaf .grad accumulation",
            "nccl": "allgather EXCLUDED from activation timing (profiled elsewhere)",
            "replicated": "activation is applied after TP allgather; TP is not a compute-model dimension",
            "shape_family": "BF_GLOBAL",
        },
        "notes": {
            "global_width": "single-device rows are deduplicated by F_global",
            "legacy_shape_inputs": "F_local and tp are accepted only to derive F_global",
            "backward_driver": "empty_like upstream placeholder allocated after forward and outside backward timing",
            "tensor_setup": "outside timed region",
        },
    }


def require_exclusive_devices(config: ActivationSweepConfig) -> None:
    """Refuse to start if another process already holds one of our GPUs.

    Sharing a card with a second CUDA context inflates every CUDA-event window by
    time-slicing and lost memory bandwidth. Measured on an A40: TopK fwd+bwd at
    (B=8192, width=32768) is 16.36 ms with the card to itself and 36.62 ms (2.24x)
    next to an sgemm loop, with each run internally stable -- so the damage is
    invisible in the per-run spread and cannot be averaged away.
    """
    shared: dict[str, list[int]] = {}
    for device_text in config.devices:
        device = torch.device(device_text)
        pids = other_process_pids_on_device(device)
        if pids:
            shared[device_text] = pids
    if not shared:
        return
    detail = "; ".join(f"{dev}: pids {pids}" for dev, pids in shared.items())
    message = (
        f"another process is using the target GPU(s) -- {detail}. Timings would be "
        f"inflated (measured 2.24x on TopK). Wait for the card, pick a free device "
        f"with --device, or pass --allow-shared-device to profile anyway (rows will "
        f"be flagged via interference_suspected)."
    )
    if not config.allow_shared_device:
        raise RuntimeError(message)
    print(f"[activation-profiler] WARNING: {message}", file=sys.stderr)


def run_single_process(config: ActivationSweepConfig) -> tuple[Path, Path]:
    config = config.normalize()
    config.validate_inputs()
    require_exclusive_devices(config)
    csv_path, json_path = unique_output_paths(Path(config.output_dir), config.output_name)
    started_at = datetime.now(timezone.utc)
    profiler = ActivationProfiler()
    cases = list(config.iter_cases())

    print(f"[activation-profiler] cases={len(cases)} CSV={csv_path} JSON={json_path}")
    rows_all: list[dict[str, Any]] = []
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for idx, case in enumerate(cases, 1):
            print(
                f"[{idx}/{len(cases)}] B={case.B} F_local={case.F_local} tp={case.tp} "
                f"activation={case.activation_type} k={case.k} width={case.F_global} "
                f"device={case.device} dtype={case.dtype_name}"
            )
            rows = profiler.profile_case(case, config)
            for row in rows:
                rows_all.append(row)
                if row.get("interference_suspected"):
                    print(
                        f"    [warn] {row['semantic_op']}: median/min="
                        f"{float(row['interference_ratio']):.2f}x "
                        f"(median {float(row['median_ms']):.2f} vs clean "
                        f"{float(row['clean_ms']):.2f} ms) -- contention came and "
                        f"went during the run; prefer clean_ms",
                        file=sys.stderr,
                    )
                if row.get("foreign_pids_on_device"):
                    print(
                        f"    [warn] {row['semantic_op']}: foreign context on "
                        f"{row['device']} (pids {row['foreign_pids_on_device']}) -- "
                        f"ALL samples inflated, min_ms/clean_ms are NOT clean; "
                        f"discard this row and re-run on an idle GPU",
                        file=sys.stderr,
                    )
                writer.writerow({key: csv_safe(row.get(key, "")) for key in CSV_FIELDS})
            handle.flush()

    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {"metadata": build_metadata(config, started_at), "results": rows_all},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    print(f"[activation-profiler] done: {len(rows_all)} rows")
    return csv_path, json_path


def run_distributed_worker(config: ActivationSweepConfig) -> list[dict[str, Any]]:
    """One rank of a distributed run (world_size == tp). Builds a local shard,
    allgathers to full width via the real SAE TP allgather, then profiles topk.

    Requires torch.distributed to be initialized by the launcher (torchrun) with
    world_size == tp. Only tp values equal to world_size are profiled.
    """
    import torch.distributed as dist

    from sae_lens.distributed import tp_allgather

    config = config.normalize()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)))
    profiler = ActivationProfiler()

    rows_all: list[dict[str, Any]] = []
    for case in config.iter_cases():
        if case.tp != world_size:
            continue
        device = case.torch_device
        local: torch.Tensor | None = None
        full: torch.Tensor | None = None
        try:
            local = torch.randn(
                (case.B, case.F_local),
                device=device,
                dtype=case.dtype,
                requires_grad=True,
            )
            full = tp_allgather(local, group)  # real collective, excluded from TopK timing
            rows = profiler.profile_case(
                case, config, full_width_input=full, rank=rank, world_size=world_size
            )
            rows_all.extend(rows)
        finally:
            full = None
            local = None
            cleanup_device(device, config.empty_cache_between_ops)
    return rows_all


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile complete activation forward + backward over B, global width, activation type and parameters."
    )
    parser.add_argument(
        "--batch-size", "--batch-sizes", dest="batch_sizes", type=int, nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--local-d-sae", "--d-sae", dest="local_d_sae_values", type=int, nargs="+",
        default=None,
        help=(
            "Legacy LOCAL d_sae F=global_d_sae/tp. Defaults are used only "
            "when --global-d-sae is not supplied."
        ),
    )
    parser.add_argument(
        "--tp", dest="tp_values", type=int, nargs="+", default=None,
        help=(
            "Legacy shape-expansion TP values. In single-device mode only the "
            "derived F_global values matter and duplicate widths are removed."
        ),
    )
    parser.add_argument(
        "--global-d-sae",
        "--global-d-sae-values",
        dest="global_d_sae_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_GLOBAL_D_SAE_VALUES),
        help="Direct global activation widths; preferred over F_local*tp.",
    )
    parser.add_argument(
        "--k", dest="k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--activation-type",
        "--activation-types",
        dest="activation_types",
        nargs="+",
        choices=["topk", "relu"],
        default=list(DEFAULT_ACTIVATION_TYPES),
        help="Activation implementation. TopK includes topk+ReLU+zeros/scatter and backward.",
    )
    parser.add_argument(
        "--device", "--devices", dest="devices", nargs="+", default=list(DEFAULT_DEVICES),
    )
    parser.add_argument(
        "--dtype", "--dtypes", dest="dtypes", nargs="+", default=list(DEFAULT_DTYPES),
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-cache-between-ops", action="store_true")
    parser.add_argument("--skip-forward-only", action="store_true",
                        help="Only profile fwd+bwd, skip the forward-only row")
    parser.add_argument(
        "--distributed", action="store_true",
        help="Run under torchrun with world_size==tp; each rank allgathers then profiles",
    )
    parser.add_argument(
        "--allow-shared-device", action="store_true",
        help="Profile even when another process holds the GPU. Timings will be "
             "inflated (measured 2.24x); affected rows set interference_suspected",
    )
    parser.add_argument(
        "--interference-ratio", type=float, default=DEFAULT_INTERFERENCE_RATIO,
        help="median/min above this marks a case as contended "
             f"(default {DEFAULT_INTERFERENCE_RATIO})",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ActivationSweepConfig:
    global_widths = list(args.global_d_sae_values)
    if args.local_d_sae_values is None:
        local_widths = (
            [] if global_widths else list(DEFAULT_LOCAL_D_SAE_VALUES)
        )
    else:
        local_widths = list(args.local_d_sae_values)
    if args.tp_values is None:
        tp_values = (
            [1] if global_widths and not local_widths else list(DEFAULT_TP_VALUES)
        )
    else:
        tp_values = list(args.tp_values)
    return ActivationSweepConfig(
        batch_sizes=list(args.batch_sizes),
        local_d_sae_values=local_widths,
        tp_values=tp_values,
        global_d_sae_values=global_widths,
        k_values=list(args.k_values),
        activation_types=list(args.activation_types),
        devices=list(args.devices),
        dtypes=list(args.dtypes),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        output_dir=args.output_dir,
        output_name=args.output_name,
        fail_fast=args.fail_fast,
        empty_cache_between_ops=not args.keep_cache_between_ops,
        distributed=args.distributed,
        profile_forward_only=not args.skip_forward_only,
        interference_ratio=args.interference_ratio,
        allow_shared_device=args.allow_shared_device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        if config.distributed:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group()
            rows = run_distributed_worker(config)
            if dist.get_rank() == 0:
                config_n = config.normalize()
                csv_path, json_path = unique_output_paths(
                    Path(config.output_dir), config.output_name
                )
                with csv_path.open("x", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {key: csv_safe(row.get(key, "")) for key in CSV_FIELDS}
                        )
                with json_path.open("x", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "metadata": build_metadata(
                                config_n, datetime.now(timezone.utc)
                            ),
                            "results": rows,
                        },
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )
                    handle.write("\n")
                print(f"[activation-profiler] distributed done: {len(rows)} rows -> {csv_path}")
            dist.barrier()
        else:
            run_single_process(config)
        return 0
    except KeyboardInterrupt:
        print("\n[activation-profiler] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[activation-profiler] fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



