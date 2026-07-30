#!/usr/bin/env python3
"""Extensible external profiler for SAE GEMMs, local compute, and Adam.

The profiler accepts arrays of:
  * d_in (D)
  * local d_sae (F = global d_sae / TP)
  * per-rank batch size (B)
  * tensor parallel size (tp; shape-only for single-device local profiling)
  * device
  * dtype

All CLI inputs are optional. Running ``python external_gemm_optim_profiler.py`` uses
an editable default sweep declared in the ``Editable defaults`` section near
the top of this file. CLI arguments override those defaults.

Registered plugins profile only the dimensions they actually depend on:
  * GEMM:          (B, D, F, device, dtype)
  * Adam:          (D, F, implementation, device, dtype), one SAE at a time
  * Preprocess:    (B, D, device, dtype)
  * Local compute: aggregated BD/BF/FD/global-feature/TP-local shape families on one device

Error and hang isolation
------------------------
By default, every unique plugin/case task runs in its own spawned subprocess.
A Python exception, CUDA OOM, worker crash, or configured timeout is recorded as
an output row, and the parent proceeds to the next task. A timeout terminates
(and, if necessary, kills) the child process, which also releases that child's
CUDA allocations. Use --no-subprocess-isolation only when startup overhead is
more important than hang isolation.

Timing
------
CUDA operations are timed with CUDA events. Tensor/optimizer construction is
outside the timed region. The JSON summary and terminal output include the
complete profiler wall-clock runtime, including validation, worker startup,
warmup, measurements, cleanup, and result writing.

Outputs
-------
Results are written incrementally to CSV and finalized to JSON. Existing output
names are never overwritten; _1, _2, ... is appended automatically.
"""

from __future__ import annotations


# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================
# Edit ONLY this block for the usual no-argument run:
#
#     python external_gemm_optim_profiler.py
#
# Command-line options remain available and override these values.

DEFAULT_D_IN_VALUES: list[int] = [1024,4096]
# local d_sae: F = global d_sae / tensor_parallel_size
DEFAULT_LOCAL_D_SAE_VALUES: list[int] = [8192, 16384, 32768, 65536, 131072]
# Per-rank token batch sizes
DEFAULT_BATCH_SIZES: list[int] = [512, 1024, 2048, 4096, 8192, 16384]
# Local compute remains single-device. TP is a shape parameter only:
# F_global = F_local * tp for operations that run on the gathered feature width.
DEFAULT_TP_VALUES: list[int] = [1, 2]
# Match the implementations selectable by MultiSAETrainer via SAE_ADAM_IMPL.
DEFAULT_OPTIMIZER_IMPLS: list[str] = ["foreach"]
DEFAULT_STATS_SYNC_MODE: str = "immediate"
DEFAULT_STATS_SYNC_INTERVAL: int = 1
DEFAULT_NORMALIZE_ACTIVATIONS: str = "none"
DEFAULT_DEVICES: list[str] = ["cuda:0"]
DEFAULT_DTYPES: list[str] = ["float32"]

# Available defaults currently registered: "gemm", "preprocess", "local_compute", "optimizer".
# "fused_optimizer" remains accepted as a backwards-compatible CLI alias.
DEFAULT_PLUGINS: list[str] = ["gemm", "preprocess", "local_compute", "optimizer"]

DEFAULT_WARMUP: int = 5
DEFAULT_REPEATS: int = 8
DEFAULT_OUTPUT_DIR: str = "sae_lens/autoconfig/profile_results"
DEFAULT_OUTPUT_NAME: str = "sae_compute_profile"

# A single plugin/configuration is isolated in a child process.
# Timeout <= 0 disables timeout enforcement.
DEFAULT_SUBPROCESS_ISOLATION: bool = True
DEFAULT_TASK_TIMEOUT_SECONDS: float = 1800.0

# =============================================================================
# END USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================

import argparse
import csv
import gc
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import statistics
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


# -----------------------------------------------------------------------------
# Dtypes and basic utilities
# -----------------------------------------------------------------------------
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

NORMALIZE_ACTIVATION_MODES: tuple[str, ...] = (
    "none",
    "expected_average_only_in",
    "constant_norm_rescale",
    "layer_norm",
)


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


def canonical_normalize_activations(name: str) -> str:
    key = name.strip().lower()
    if key not in NORMALIZE_ACTIVATION_MODES:
        raise ValueError(
            f"Unsupported normalize_activations {name!r}. "
            f"Supported values: {', '.join(NORMALIZE_ACTIVATION_MODES)}"
    )
    return key


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# -----------------------------------------------------------------------------
# Configuration and case definitions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileCase:
    """One point in the superset Cartesian-product profile space."""

    batch_size: int
    d_in: int
    local_d_sae: int
    tp: int
    optimizer_impl: str
    device: str
    dtype_name: str
    task_variant: str = ""

    @property
    def B(self) -> int:
        return self.batch_size

    @property
    def D(self) -> int:
        return self.d_in

    @property
    def F(self) -> int:
        return self.local_d_sae

    @property
    def F_global(self) -> int:
        return self.F * self.tp

    @property
    def dtype(self) -> torch.dtype:
        return parse_dtype(self.dtype_name)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def case_id(self) -> str:
        safe_device = self.device.replace(":", "_").replace("/", "_")
        variant = f"_{self.task_variant}" if self.task_variant else ""
        return (
            f"B{self.B}_D{self.D}_Flocal{self.F}_tp{self.tp}_"
            f"opt{self.optimizer_impl}_{safe_device}_{self.dtype_name}{variant}"
        )

@dataclass
class SweepConfig:
    """External profiler configuration."""

    d_in_values: list[int]
    local_d_sae_values: list[int]
    batch_sizes: list[int]
    devices: list[str]
    dtypes: list[str]
    tp_values: list[int] = field(default_factory=lambda: [1])
    optimizer_impls: list[str] = field(default_factory=lambda: ["foreach"])
    stats_sync_mode: str = DEFAULT_STATS_SYNC_MODE
    stats_sync_interval: int = DEFAULT_STATS_SYNC_INTERVAL
    normalize_activations: str = DEFAULT_NORMALIZE_ACTIVATIONS

    warmup: int = 20
    repeats: int = 50
    seed: int = 0
    output_dir: str = "profile_results"
    output_name: str = "sae_compute_profile"
    fail_fast: bool = False
    empty_cache_between_ops: bool = True
    tf32: str = "inherit"  # inherit/on/off

    # Robust execution. A value <= 0 disables the timeout, but subprocess
    # isolation can still be used to contain crashes and CUDA error states.
    subprocess_isolation: bool = True
    task_timeout_seconds: float = 1800.0
    worker_shutdown_grace_seconds: float = 5.0

    # Adam settings.
    profile_optimizer_cold_step: bool = True
    optimizer_lr: float = 1.0e-3
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1.0e-8
    optimizer_weight_decay: float = 0.0
    optimizer_amsgrad: bool = False
    optimizer_maximize: bool = False
    optimizer_capturable: bool = False

    def normalize(self) -> "SweepConfig":
        self.d_in_values = unique_preserve_order(int(x) for x in self.d_in_values)
        self.local_d_sae_values = unique_preserve_order(
            int(x) for x in self.local_d_sae_values
        )
        self.batch_sizes = unique_preserve_order(int(x) for x in self.batch_sizes)
        self.tp_values = unique_preserve_order(int(x) for x in self.tp_values)
        self.optimizer_impls = unique_preserve_order(
            str(x).strip().lower() for x in self.optimizer_impls
        )
        self.stats_sync_mode = str(self.stats_sync_mode).strip().lower()
        self.stats_sync_interval = int(self.stats_sync_interval)
        self.normalize_activations = canonical_normalize_activations(
            self.normalize_activations
        )
        self.devices = unique_preserve_order(str(x) for x in self.devices)
        self.dtypes = unique_preserve_order(
            canonical_dtype_name(parse_dtype(str(x))) for x in self.dtypes
        )
        self.tf32 = self.tf32.lower()
        return self

    def validate_inputs(self) -> None:
        named_arrays: Mapping[str, Sequence[Any]] = {
            "d_in_values": self.d_in_values,
            "local_d_sae_values": self.local_d_sae_values,
            "batch_sizes": self.batch_sizes,
            "tp_values": self.tp_values,
            "optimizer_impls": self.optimizer_impls,
            "devices": self.devices,
            "dtypes": self.dtypes,
        }
        for name, values in named_arrays.items():
            if not values:
                raise ValueError(f"{name} must not be empty")

        for name, values in (
            ("d_in_values", self.d_in_values),
            ("local_d_sae_values", self.local_d_sae_values),
            ("batch_sizes", self.batch_sizes),
            ("tp_values", self.tp_values),
        ):
            invalid = [x for x in values if x <= 0]
            if invalid:
                raise ValueError(f"{name} must contain positive integers; got {invalid}")

        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")
        if self.tf32 not in {"inherit", "on", "off"}:
            raise ValueError("tf32 must be one of: inherit, on, off")
        valid_optimizer_impls = {"fused", "foreach", "forloop", "default"}
        invalid_impls = [
            value for value in self.optimizer_impls if value not in valid_optimizer_impls
        ]
        if invalid_impls:
            raise ValueError(
                f"optimizer_impls contains unsupported values {invalid_impls}; "
                f"choices={sorted(valid_optimizer_impls)}"
            )
        if self.stats_sync_mode not in {"immediate", "deferred", "periodic"}:
            raise ValueError(
                "stats_sync_mode must be one of: immediate, deferred, periodic"
            )
        if self.stats_sync_interval < 1:
            raise ValueError("stats_sync_interval must be >= 1")
        canonical_normalize_activations(self.normalize_activations)
        if self.worker_shutdown_grace_seconds < 0:
            raise ValueError("worker_shutdown_grace_seconds must be >= 0")
        if not (0.0 <= self.optimizer_beta1 < 1.0):
            raise ValueError("optimizer_beta1 must be in [0, 1)")
        if not (0.0 <= self.optimizer_beta2 < 1.0):
            raise ValueError("optimizer_beta2 must be in [0, 1)")
        if self.optimizer_lr < 0:
            raise ValueError("optimizer_lr must be >= 0")
        if self.optimizer_eps <= 0:
            raise ValueError("optimizer_eps must be > 0")
        if self.optimizer_weight_decay < 0:
            raise ValueError("optimizer_weight_decay must be >= 0")

        for device_text in self.devices:
            # Syntax only. Per-task checks turn unavailable devices into rows.
            torch.device(device_text)
        for dtype_text in self.dtypes:
            parse_dtype(dtype_text)

    def iter_cases(self) -> Iterable[ProfileCase]:
        for D, F, B, tp, optimizer_impl, device, dtype_name in itertools.product(
            self.d_in_values,
            self.local_d_sae_values,
            self.batch_sizes,
            self.tp_values,
            self.optimizer_impls,
            self.devices,
            self.dtypes,
        ):
            yield ProfileCase(
                batch_size=B,
                d_in=D,
                local_d_sae=F,
                tp=tp,
                optimizer_impl=optimizer_impl,
                device=device,
                dtype_name=dtype_name,
            )

# -----------------------------------------------------------------------------
# Validation extension point
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""

class CaseValidator(ABC):
    @abstractmethod
    def validate(self, case: ProfileCase) -> ValidationResult:
        raise NotImplementedError

class AlwaysValidCaseValidator(CaseValidator):
    """Placeholder. Replace later with a memory-aware validator."""

    def validate(self, case: ProfileCase) -> ValidationResult:
        del case
        return ValidationResult(valid=True)

# -----------------------------------------------------------------------------
# Timing and statistics
# -----------------------------------------------------------------------------
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

def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )

def summarize_samples(samples_ms: Sequence[float]) -> TimingStats:
    if not samples_ms:
        raise ValueError("No timing samples were collected")
    ordered = sorted(float(x) for x in samples_ms)
    return TimingStats(
        samples=len(ordered),
        median_ms=float(statistics.median(ordered)),
        mean_ms=float(statistics.fmean(ordered)),
        std_ms=float(statistics.pstdev(ordered) if len(ordered) > 1 else 0.0),
        min_ms=float(ordered[0]),
        p10_ms=percentile(ordered, 0.10),
        p90_ms=percentile(ordered, 0.90),
        max_ms=float(ordered[-1]),
    )


@dataclass(frozen=True)
class CompositeRegion:
    """One independently allocated sub-region inside an aggregated profile row.

    ``setup`` is called immediately before timing this region and may allocate
    persistent inputs/parameters for the region.  The returned mapping must
    contain either:

    * ``operation`` for a simple forward/inference region; or
    * ``forward``, ``prepare_backward`` and ``backward`` for an autograd region.

    Regions are profiled one at a time and released before the next region is
    set up.  This is essential for the aggregated local profiler: operations
    with the same driving dimensions are summed into one row without forcing
    all of their large tensors to coexist and creating profiler-only OOMs.
    """

    name: str
    setup: Callable[[], dict[str, Any]]
    staged_autograd: bool = False
    time_forward: bool = True
    inference_mode: bool = True


class OperationTimer:
    """CUDA-event timer for CUDA and perf_counter timer for CPU.
    Besides ordinary single-operation timing, ``measure_composite`` supports
    an aggregated model row made of several independently allocated semantic
    sub-regions.  Raw samples are collected for every sub-region and added by
    repeat index before summary statistics are computed.
    """

    def __init__(self, device: torch.device, warmup: int, repeats: int) -> None:
        self.device = device
        self.warmup = warmup
        self.repeats = repeats

    def measure(
        self,
        operation: Callable[[], Any],
        *,
        inference_mode: bool = True,
    ) -> TimingStats:
        return summarize_samples(
            self.measure_samples(operation, inference_mode=inference_mode)
        )

    def measure_samples(
        self,
        operation: Callable[[], Any],
        *,
        inference_mode: bool = True,
    ) -> list[float]:
        if self.device.type == "cuda":
            return self._measure_cuda_samples(
                operation, inference_mode=inference_mode
            )
        if self.device.type == "cpu":
            return self._measure_cpu_samples(
                operation, inference_mode=inference_mode
            )
        raise ValueError(
            f"Unsupported device type {self.device.type!r}; expected cuda or cpu"
        )

    def _measure_cuda_samples(
        self,
        operation: Callable[[], Any],
        *,
        inference_mode: bool,
    ) -> list[float]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")

        context = torch.inference_mode() if inference_mode else _NullContext()
        with torch.cuda.device(self.device), context:
            for _ in range(self.warmup):
                output = operation()
                torch.cuda.synchronize(self.device)
                del output

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples_ms: list[float] = []
            for _ in range(self.repeats):
                start.record()
                output = operation()
                end.record()
                end.synchronize()
                samples_ms.append(float(start.elapsed_time(end)))
                del output
        return samples_ms

    def _measure_cpu_samples(
        self,
        operation: Callable[[], Any],
        *,
        inference_mode: bool,
    ) -> list[float]:
        context = torch.inference_mode() if inference_mode else _NullContext()
        with context:
            for _ in range(self.warmup):
                output = operation()
                del output
            samples_ms: list[float] = []
            for _ in range(self.repeats):
                start_ns = time.perf_counter_ns()
                output = operation()
                end_ns = time.perf_counter_ns()
                samples_ms.append((end_ns - start_ns) / 1_000_000.0)
                del output
        return samples_ms

    def measure_staged_autograd(
        self,
        forward: Callable[[], Any],
        prepare_backward: Callable[[Any], Any],
        backward: Callable[[Any, Any], Any],
        *,
        time_forward: bool = True,
    ) -> TimingStats:
        return summarize_samples(
            self.measure_staged_samples(
                forward,
                prepare_backward,
                backward,
                time_forward=time_forward,
            )
        )

    def measure_staged_samples(
        self,
        forward: Callable[[], Any],
        prepare_backward: Callable[[Any], Any],
        backward: Callable[[Any, Any], Any],
        *,
        time_forward: bool = True,
    ) -> list[float]:
        """Measure autograd while allocating the upstream only at backward entry.

        ``time_forward=False`` is used when another plugin already owns the
        forward operation but this component still owns its backward kernels.
        The forward graph is rebuilt outside the timed interval on every repeat.
        """
        if self.device.type == "cuda":
            return self._measure_staged_cuda_samples(
                forward, prepare_backward, backward, time_forward=time_forward
            )
        if self.device.type == "cpu":
            return self._measure_staged_cpu_samples(
                forward, prepare_backward, backward, time_forward=time_forward
            )
        raise ValueError(
            f"Unsupported device type {self.device.type!r}; expected cuda or cpu"
        )

    def _measure_staged_cuda_samples(
        self,
        forward: Callable[[], Any],
        prepare_backward: Callable[[Any], Any],
        backward: Callable[[Any, Any], Any],
        *,
        time_forward: bool,
    ) -> list[float]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
        with torch.cuda.device(self.device):
            for _ in range(self.warmup):
                forward_output = forward()
                torch.cuda.synchronize(self.device)
                backward_input = prepare_backward(forward_output)
                backward_output = backward(forward_output, backward_input)
                torch.cuda.synchronize(self.device)
                del backward_output, backward_input, forward_output

            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            samples_ms: list[float] = []
            for _ in range(self.repeats):
                if time_forward:
                    fwd_start.record()
                    forward_output = forward()
                    fwd_end.record()
                    fwd_end.synchronize()
                    forward_ms = float(fwd_start.elapsed_time(fwd_end))
                else:
                    forward_output = forward()
                    torch.cuda.synchronize(self.device)
                    forward_ms = 0.0

                backward_input = prepare_backward(forward_output)
                bwd_start.record()
                backward_output = backward(forward_output, backward_input)
                bwd_end.record()
                bwd_end.synchronize()
                backward_ms = float(bwd_start.elapsed_time(bwd_end))
                samples_ms.append(forward_ms + backward_ms)
                del backward_output, backward_input, forward_output
        return samples_ms

    def _measure_staged_cpu_samples(
        self,
        forward: Callable[[], Any],
        prepare_backward: Callable[[Any], Any],
        backward: Callable[[Any, Any], Any],
        *,
        time_forward: bool,
    ) -> list[float]:
        for _ in range(self.warmup):
            forward_output = forward()
            backward_input = prepare_backward(forward_output)
            backward_output = backward(forward_output, backward_input)
            del backward_output, backward_input, forward_output

        samples_ms: list[float] = []
        for _ in range(self.repeats):
            if time_forward:
                start_ns = time.perf_counter_ns()
                forward_output = forward()
                forward_end_ns = time.perf_counter_ns()
                forward_ms = (forward_end_ns - start_ns) / 1_000_000.0
            else:
                forward_output = forward()
                forward_ms = 0.0
            backward_input = prepare_backward(forward_output)
            backward_start_ns = time.perf_counter_ns()
            backward_output = backward(forward_output, backward_input)
            end_ns = time.perf_counter_ns()
            backward_ms = (end_ns - backward_start_ns) / 1_000_000.0
            samples_ms.append(forward_ms + backward_ms)
            del backward_output, backward_input, forward_output
        return samples_ms

    def measure_composite(self, regions: Sequence[CompositeRegion]) -> TimingStats:
        """Profile one aggregate row without co-resident region inputs.

        Each region owns one real semantic subpath, but all regions share the
        same driving dimensions.  A region is fully profiled and released before
        the next is set up.  Its raw repeat samples are added elementwise to form
        one aggregate sample vector.
        """
        if not regions:
            raise ValueError("Composite profile requires at least one region")
        totals = [0.0 for _ in range(self.repeats)]
        for region in regions:
            state: dict[str, Any] | None = None
            try:
                state = region.setup()
                if region.staged_autograd:
                    samples = self.measure_staged_samples(
                        state["forward"],
                        state["prepare_backward"],
                        state["backward"],
                        time_forward=region.time_forward,
                    )
                else:
                    samples = self.measure_samples(
                        state["operation"],
                        inference_mode=region.inference_mode,
                    )
                if len(samples) != self.repeats:
                    raise RuntimeError(
                        f"Region {region.name!r} returned {len(samples)} samples; "
                        f"expected {self.repeats}"
                    )
                for idx, value in enumerate(samples):
                    totals[idx] += float(value)
            finally:
                state = None
                gc.collect()
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
        return summarize_samples(totals)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

# -----------------------------------------------------------------------------
# Shared device helpers
# -----------------------------------------------------------------------------
def check_device_and_dtype(
    device: torch.device,
    dtype: torch.dtype,
    *,
    component: str,
    cuda_required: bool = False,
) -> None:
    if cuda_required and device.type != "cuda":
        raise ValueError(f"{component} requires a CUDA device; got {device}")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested, but torch.cuda.is_available() is False"
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; "
                f"device_count={torch.cuda.device_count()}"
            )
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device type: {device.type}")

    if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(f"Unsupported {component} dtype: {dtype}")


def apply_tf32_policy(policy: str, device: torch.device) -> None:
    if device.type != "cuda" or policy == "inherit":
        return
    torch.backends.cuda.matmul.allow_tf32 = policy == "on"


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


# -----------------------------------------------------------------------------
# Profiler plugin API
# -----------------------------------------------------------------------------
class ProfilerPlugin(ABC):
    name: str

    def case_key(self, case: ProfileCase) -> tuple[Any, ...]:
        """Dimensions this plugin truly depends on; used for automatic dedup."""
        return (
            case.B,
            case.D,
            case.F,
            case.device,
            case.dtype_name,
        )

    def iter_task_cases(self, config: SweepConfig) -> Iterable[ProfileCase]:
        """Yield plugin-specific unique tasks.

        Plugins with heterogeneous semantic groups can override this method to
        deduplicate each group by its own driving dimensions rather than forcing
        all groups through one common Cartesian key.
        """
        seen: set[tuple[Any, ...]] = set()
        for case in config.iter_cases():
            key = self.case_key(case)
            if key in seen:
                continue
            seen.add(key)
            yield case

    @abstractmethod
    def profile_case(
        self,
        case: ProfileCase,
        config: SweepConfig,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class PreparedOperation:
    semantic_op: str
    shape_family: str
    equation: str
    layout_view: str
    M: int
    N: int
    K: int
    lhs: torch.Tensor
    rhs: torch.Tensor
    operation: Callable[[], torch.Tensor]

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.M, self.N)


class GemmProfiler(ProfilerPlugin):
    """Profiles six semantic dense GEMMs for one (B, D, F, device, dtype)."""

    name = "gemm"

    def case_key(self, case: ProfileCase) -> tuple[Any, ...]:
        # GEMMs are per-hook and do not depend on H.
        return (case.B, case.D, case.F, case.device, case.dtype_name)

    def profile_case(
        self,
        case: ProfileCase,
        config: SweepConfig,
    ) -> list[dict[str, Any]]:
        device = case.torch_device
        dtype = case.dtype
        check_device_and_dtype(device, dtype, component="GEMM")
        apply_tf32_policy(config.tf32, device)

        timer = OperationTimer(device, config.warmup, config.repeats)
        results: list[dict[str, Any]] = []
        builders: list[tuple[str, Callable[[], PreparedOperation]]] = [
            ("encoder_forward", lambda: self._build_encoder_forward(case)),
            ("decoder_forward", lambda: self._build_decoder_forward(case)),
            ("decoder_dgrad", lambda: self._build_decoder_dgrad(case)),
            ("decoder_wgrad", lambda: self._build_decoder_wgrad(case)),
            ("encoder_dgrad", lambda: self._build_encoder_dgrad(case)),
            ("encoder_wgrad", lambda: self._build_encoder_wgrad(case)),
        ]

        for expected_name, build in builders:
            prepared: PreparedOperation | None = None
            op_started = time.perf_counter()
            try:
                prepared = build()
                stats = timer.measure(prepared.operation)
                row = self._success_row(case, config, prepared, stats)
            except Exception as exc:
                semantic_name = prepared.semantic_op if prepared else expected_name
                row = self._error_row(case, config, semantic_name, exc)
                if config.fail_fast:
                    raise
            finally:
                op_elapsed = time.perf_counter() - op_started
                if prepared is not None:
                    del prepared
                cleanup_device(device, config.empty_cache_between_ops)

            row["component_wall_seconds"] = op_elapsed
            results.append(row)

        return results

    @staticmethod
    def _empty(shape: tuple[int, int], case: ProfileCase) -> torch.Tensor:
        return torch.empty(shape, device=case.torch_device, dtype=case.dtype)

    @staticmethod
    def _mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)

    def _build_encoder_forward(self, case: ProfileCase) -> PreparedOperation:
        lhs = self._empty((case.B, case.D), case)
        rhs = self._empty((case.D, case.F), case)
        return PreparedOperation(
            "encoder_forward", "A", "[B,D]@[D,F]->[B,F]", "NN",
            case.B, case.F, case.D, lhs, rhs,
            lambda: self._mm(lhs, rhs),
        )

    def _build_decoder_forward(self, case: ProfileCase) -> PreparedOperation:
        lhs = self._empty((case.B, case.F), case)
        rhs = self._empty((case.F, case.D), case)
        return PreparedOperation(
            "decoder_forward", "B", "[B,F]@[F,D]->[B,D]", "NN",
            case.B, case.D, case.F, lhs, rhs,
            lambda: self._mm(lhs, rhs),
        )

    def _build_decoder_dgrad(self, case: ProfileCase) -> PreparedOperation:
        lhs = self._empty((case.B, case.D), case)
        w_dec = self._empty((case.F, case.D), case)
        rhs = w_dec.t()
        return PreparedOperation(
            "decoder_dgrad", "A", "[B,D]@W_dec.T[D,F]->[B,F]", "NT",
            case.B, case.F, case.D, lhs, rhs,
            lambda: self._mm(lhs, rhs),
        )

    def _build_decoder_wgrad(self, case: ProfileCase) -> PreparedOperation:
        acts = self._empty((case.B, case.F), case)
        lhs = acts.t()
        rhs = self._empty((case.B, case.D), case)
        return PreparedOperation(
            "decoder_wgrad", "C", "acts.T[F,B]@[B,D]->[F,D]", "TN",
            case.F, case.D, case.B, lhs, rhs,
            lambda: self._mm(lhs, rhs),
        )

    def _build_encoder_dgrad(self, case: ProfileCase) -> PreparedOperation:
        lhs = self._empty((case.B, case.F), case)
        w_enc = self._empty((case.D, case.F), case)
        rhs = w_enc.t()
        return PreparedOperation(
            "encoder_dgrad", "B", "[B,F]@W_enc.T[F,D]->[B,D]", "NT",
            case.B, case.D, case.F, lhs, rhs,
            lambda: self._mm(lhs, rhs),
        )

    def _build_encoder_wgrad(self, case: ProfileCase) -> PreparedOperation:
        x = self._empty((case.B, case.D), case)
        lhs = x.t()
        rhs = self._empty((case.B, case.F), case)
        return PreparedOperation(
            "encoder_wgrad", "D", "x.T[D,B]@[B,F]->[D,F]", "TN",
            case.D, case.F, case.B, lhs, rhs,
            lambda: self._mm(lhs, rhs),
        )

    @staticmethod
    def _tensor_metadata(tensor: torch.Tensor, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_shape": list(tensor.shape),
            f"{prefix}_stride": list(tensor.stride()),
            f"{prefix}_is_contiguous": bool(tensor.is_contiguous()),
            f"{prefix}_storage_offset": int(tensor.storage_offset()),
        }

    def _success_row(
        self,
        case: ProfileCase,
        config: SweepConfig,
        prepared: PreparedOperation,
        stats: TimingStats,
    ) -> dict[str, Any]:
        flops = 2 * prepared.M * prepared.N * prepared.K
        median_tflops = (
            flops / (stats.median_ms * 1.0e9)
            if stats.median_ms > 0.0
            else math.inf
        )
        return {
            "profiler": self.name,
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "error_traceback": "",
            "case_id": case.case_id,
            "semantic_op": prepared.semantic_op,
            "shape_family": prepared.shape_family,
            "equation": prepared.equation,
            "layout_view": prepared.layout_view,
            "B": case.B,
            "D": case.D,
            "F_local_d_sae": case.F,
            "H": "",  # GEMM is per hook.
            "M": prepared.M,
            "N": prepared.N,
            "K": prepared.K,
            "output_shape": list(prepared.output_shape),
            "device": str(case.torch_device),
            "dtype": case.dtype_name,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "flops": flops,
            "median_tflops": median_tflops,
            **self._tensor_metadata(prepared.lhs, "lhs"),
            **self._tensor_metadata(prepared.rhs, "rhs"),
            **asdict(stats),
        }

    def _error_row(
        self,
        case: ProfileCase,
        config: SweepConfig,
        semantic_op: str,
        exc: Exception,
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
                "B": case.B,
                "D": case.D,
                "F_local_d_sae": case.F,
                "device": str(case.torch_device),
                "dtype": case.dtype_name,
                "warmup": config.warmup,
                "repeats": config.repeats,
                "samples": 0,
            }
        )
        return row


@dataclass
class PreparedLocalGroup:
    semantic_op: str
    phase: str
    shape_family: str
    equation: str
    driving_vars: str
    profile_dimensions: dict[str, Any]
    group_shapes: dict[str, list[int]]
    op_list: list[str]
    runner: Callable[[OperationTimer], TimingStats]
    profile_dtype: str | None = None


class _ScaleGradient(torch.autograd.Function):
    """Identity forward with a scaled incoming gradient, matching TopK SAE TP."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return grad * ctx.scale, None


def _scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _ScaleGradient.apply(x, scale)  # type: ignore[return-value]


class _ShapeFamilyPluginBase(ProfilerPlugin):
    """Shared execution and row formatting for aggregated shape-family plugins."""

    def variants_for_case(
        self, case: ProfileCase, config: SweepConfig
    ) -> list[str]:
        raise NotImplementedError

    def _build_variant(
        self, variant: str, case: ProfileCase, config: SweepConfig
    ) -> PreparedLocalGroup:
        raise NotImplementedError

    @staticmethod
    def _rand(
        shape: tuple[int, ...],
        case: ProfileCase,
        *,
        requires_grad: bool = False,
        positive: bool = False,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        tensor = torch.randn(
            shape,
            device=case.torch_device,
            dtype=case.dtype if dtype is None else dtype,
        )
        if positive:
            tensor = tensor.abs().add_(1.0e-3)
        tensor.requires_grad_(requires_grad)
        return tensor

    @staticmethod
    def _simple_runner(
        operation: Callable[[], Any], *, inference_mode: bool = True
    ) -> Callable[[OperationTimer], TimingStats]:
        return lambda timer: timer.measure(operation, inference_mode=inference_mode)

    @staticmethod
    def _staged_runner(
        forward: Callable[[], Any],
        prepare_backward: Callable[[Any], Any],
        backward: Callable[[Any, Any], Any],
        *,
        time_forward: bool = True,
    ) -> Callable[[OperationTimer], TimingStats]:
        return lambda timer: timer.measure_staged_autograd(
            forward,
            prepare_backward,
            backward,
            time_forward=time_forward,
        )

    @staticmethod
    def _composite_runner(
        regions: Sequence[CompositeRegion],
    ) -> Callable[[OperationTimer], TimingStats]:
        return lambda timer: timer.measure_composite(regions)

    def profile_case(
        self,
        case: ProfileCase,
        config: SweepConfig,
    ) -> list[dict[str, Any]]:
        device = case.torch_device
        check_device_and_dtype(device, case.dtype, component=self.name)
        apply_tf32_policy(config.tf32, device)

        variants = (
            [case.task_variant]
            if case.task_variant
            else self.variants_for_case(case, config)
        )
        timer = OperationTimer(device, config.warmup, config.repeats)
        results: list[dict[str, Any]] = []
        for variant in variants:
            prepared: PreparedLocalGroup | None = None
            op_started = time.perf_counter()
            try:
                prepared = self._build_variant(variant, case, config)
                stats = prepared.runner(timer)
                row = self._success_row(case, config, prepared, stats)
            except Exception as exc:
                row = self._error_row(case, config, variant, exc)
                if config.fail_fast:
                    raise
            finally:
                op_elapsed = time.perf_counter() - op_started
                prepared = None
                cleanup_device(device, config.empty_cache_between_ops)
            row["component_wall_seconds"] = op_elapsed
            results.append(row)
        return results

    def _success_row(
        self,
        case: ProfileCase,
        config: SweepConfig,
        prepared: PreparedLocalGroup,
        stats: TimingStats,
    ) -> dict[str, Any]:
        row = blank_row()
        row.update(
            {
                "profiler": self.name,
                "status": "ok",
                "error_type": "",
                "error_message": "",
                "error_traceback": "",
                "case_id": case.case_id,
                "semantic_op": prepared.semantic_op,
                "phase": prepared.phase,
                "shape_family": prepared.shape_family,
                "equation": prepared.equation,
                "layout_view": "single_device_aggregated_shape_family",
                "driving_vars": prepared.driving_vars,
                "group_shapes": prepared.group_shapes,
                "op_list": prepared.op_list,
                "stats_sync_mode": (
                    config.stats_sync_mode
                    if "bf_global" in prepared.semantic_op
                    or "hf_global" in prepared.semantic_op
                    else ""
                ),
                "stats_sync_interval": (
                    config.stats_sync_interval
                    if prepared.semantic_op == "local_compute_hf_global_tail"
                    and config.stats_sync_mode == "periodic"
                    else ""
                ),
                "normalize_activations": config.normalize_activations,
                "device": str(case.torch_device),
                "dtype": (
                    case.dtype_name
                    if prepared.profile_dtype is None
                    else prepared.profile_dtype
                ),
                "warmup": config.warmup,
                "repeats": config.repeats,
                **asdict(stats),
            }
        )
        row.update(prepared.profile_dimensions)
        return row

    def _error_row(
        self,
        case: ProfileCase,
        config: SweepConfig,
        semantic_op: str,
        exc: Exception,
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
                "B": case.B,
                "D": case.D,
                "F_local_d_sae": case.F,
                "F_global_d_sae": case.F_global,
                "tp": case.tp,
                "H": "",
                "stats_sync_mode": config.stats_sync_mode,
                "stats_sync_interval": config.stats_sync_interval,
                "normalize_activations": config.normalize_activations,
                "device": str(case.torch_device),
                "dtype": case.dtype_name,
                "warmup": config.warmup,
                "repeats": config.repeats,
                "samples": 0,
            }
        )
        return row


class PreprocessPlugin(_ShapeFamilyPluginBase):
    """Profile the TP- and activation-independent SAE input preprocessing path."""

    name = "preprocess"
    VARIANT = "preprocess_bd"

    def case_key(self, case: ProfileCase) -> tuple[Any, ...]:
        return (case.B, case.D, case.device, case.dtype_name)

    def iter_task_cases(self, config: SweepConfig) -> Iterable[ProfileCase]:
        seen: set[tuple[Any, ...]] = set()
        for case in config.iter_cases():
            key = (*self.case_key(case), config.normalize_activations)
            if key in seen:
                continue
            seen.add(key)
            yield replace(case, task_variant=self.VARIANT, tp=1)

    def variants_for_case(
        self, case: ProfileCase, config: SweepConfig
    ) -> list[str]:
        return [self.VARIANT]

    def _build_variant(
        self, variant: str, case: ProfileCase, config: SweepConfig
    ) -> PreparedLocalGroup:
        if variant != self.VARIANT:
            raise ValueError(f"Unknown preprocess variant {variant!r}")
        x = self._rand((case.B, case.D), case, requires_grad=True)
        b_dec = self._rand((case.D,), case, requires_grad=True)
        normalize_activations = config.normalize_activations
        expected_average_scale = 1.125

        def forward() -> torch.Tensor:
            if normalize_activations == "constant_norm_rescale":
                x_norm_coeff = (case.D**0.5) / x.norm(dim=-1, keepdim=True)
                sae_in = x * x_norm_coeff
            elif normalize_activations == "layer_norm":
                mu = x.mean(dim=-1, keepdim=True)
                centered = x - mu
                std = centered.std(dim=-1, keepdim=True)
                sae_in = centered / (std + 1.0e-5)
            elif normalize_activations == "expected_average_only_in":
                sae_in = x * expected_average_scale
            else:
                sae_in = x
            return sae_in - b_dec

        def prepare_backward(output: torch.Tensor) -> torch.Tensor:
            # The upstream gradient exists when process_sae_in backward begins;
            # allocate it only after forward to avoid a profiler-only peak.
            return torch.empty_like(output)

        def backward(
            output: torch.Tensor, upstream: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.autograd.grad(output, (x, b_dec), grad_outputs=upstream)

        group_shapes = {"x": [case.B, case.D], "b_dec": [case.D]}
        op_list = self._preprocess_op_list(normalize_activations)

        return PreparedLocalGroup(
            semantic_op=self.VARIANT,
            phase="preprocess_forward_backward",
            shape_family="BD",
            equation=(
                "T_pre(B,D,normalize): process_sae_in forward + backward"
            ),
            driving_vars="B,D,normalize_activations",
            profile_dimensions={"B": case.B, "D": case.D},
            group_shapes=group_shapes,
            op_list=op_list,
            runner=self._staged_runner(forward, prepare_backward, backward),
        )

    @staticmethod
    def _preprocess_op_list(normalize_activations: str) -> list[str]:
        if normalize_activations == "constant_norm_rescale":
            return [
                "compute x_norm_coeff = sqrt(D) / ||x||",
                "scale SAE input by x_norm_coeff",
                "subtract decoder bias from normalized SAE input",
                "preprocess backward to input and b_dec",
            ]
        if normalize_activations == "layer_norm":
            return [
                "compute per-token mean",
                "center input and compute per-token std",
                "divide by std + 1e-5",
                "subtract decoder bias from normalized SAE input",
                "preprocess backward to input and b_dec",
            ]
        if normalize_activations == "expected_average_only_in":
            return [
                "multiply batch by ActivationScaler scaling factor",
                "subtract decoder bias from scaled SAE input",
                "scalar-multiply and broadcast-subtraction backward to input and b_dec",
            ]
        return [
            "subtract decoder bias from SAE input",
            "broadcast-subtraction backward to input and b_dec",
        ]


class LocalComputePlugin(_ShapeFamilyPluginBase):
    """Profile activation-independent local work as aggregated shape families.

    The profiler intentionally produces a small number of model terms:

    * ``local_compute_bd(B,D)``
    * ``local_compute_bf_local(B,F_local)``
    * ``local_compute_fd_local(D,F_local)``
    * ``local_compute_bf_global(B,F_global,stats_mode)``
    * ``local_compute_bf_tp(B,F_local,tp)``
    * ``local_compute_bd_tp(B,D,tp)``

    TopK/ReLU/scatter and their backward kernels are excluded and belong to the
    separate activation profiler.  GEMMs, NCCL collectives and Adam are also
    excluded.  TP remains a shape-only input; every local region runs on one
    device.
    """

    name = "local_compute"

    BASE_VARIANTS: tuple[str, ...] = (
        "local_compute_bd",
        "local_compute_bf_local",
        "local_compute_fd_local",
        "local_compute_bf_global",
    )
    TP_VARIANTS: tuple[str, ...] = (
        "local_compute_bf_tp",
        "local_compute_bd_tp",
    )
    STATS_TAIL_VARIANT = "local_compute_hf_global_tail"

    def variants_for_case(
        self, case: ProfileCase, config: SweepConfig
    ) -> list[str]:
        variants = list(self.BASE_VARIANTS)
        if config.stats_sync_mode in {"deferred", "periodic"}:
            variants.append(self.STATS_TAIL_VARIANT)
        if case.tp > 1:
            variants.extend(self.TP_VARIANTS)
        return variants

    def case_key(self, case: ProfileCase) -> tuple[Any, ...]:
        return self._variant_key(case.task_variant or "all", case, None)

    def iter_task_cases(self, config: SweepConfig) -> Iterable[ProfileCase]:
        seen: set[tuple[Any, ...]] = set()
        for case in config.iter_cases():
            for variant in self.variants_for_case(case, config):
                key = self._variant_key(variant, case, config)
                if key in seen:
                    continue
                seen.add(key)
                yield replace(case, task_variant=variant)

    @staticmethod
    def _variant_key(
        variant: str,
        case: ProfileCase,
        config: SweepConfig | None,
    ) -> tuple[Any, ...]:
        dev = (case.device, case.dtype_name)
        if variant == "local_compute_bd":
            normalize = config.normalize_activations if config else ""
            return (variant, case.B, case.D, normalize, *dev)
        if variant == "local_compute_bf_local":
            return (variant, case.B, case.F, *dev)
        if variant == "local_compute_fd_local":
            return (variant, case.D, case.F, *dev)
        if variant == "local_compute_bf_global":
            mode = config.stats_sync_mode if config else ""
            return (variant, case.B, case.F_global, mode, *dev)
        if variant == LocalComputePlugin.STATS_TAIL_VARIANT:
            mode = config.stats_sync_mode if config else ""
            interval = config.stats_sync_interval if config else 1
            return (variant, case.F_global, mode, interval, case.device)
        if variant == "local_compute_bf_tp":
            return (variant, case.B, case.F, case.tp, *dev)
        if variant == "local_compute_bd_tp":
            return (variant, case.B, case.D, case.tp, *dev)
        return (variant, case.B, case.D, case.F, case.tp, *dev)

    def _build_variant(
        self, variant: str, case: ProfileCase, config: SweepConfig
    ) -> PreparedLocalGroup:
        builders: dict[str, Callable[[], PreparedLocalGroup]] = {
            "local_compute_bd": lambda: self._build_local_bd(case, config),
            "local_compute_bf_local": lambda: self._build_local_bf(case),
            "local_compute_fd_local": lambda: self._build_local_fd(case),
            "local_compute_bf_global": lambda: self._build_local_bf_global(case, config),
            self.STATS_TAIL_VARIANT: lambda: self._build_stats_tail(case, config),
            "local_compute_bf_tp": lambda: self._build_local_bf_tp(case),
            "local_compute_bd_tp": lambda: self._build_local_bd_tp(case),
        }
        try:
            return builders[variant]()
        except KeyError as exc:
            raise ValueError(f"Unknown local aggregate variant {variant!r}") from exc

    def _build_local_bd(
        self, case: ProfileCase, config: SweepConfig
    ) -> PreparedLocalGroup:
        normalize_activations = config.normalize_activations

        def setup_reconstruction_loss() -> dict[str, Any]:
            sae_in = self._rand((case.B, case.D), case, requires_grad=True)
            decoder_partial = self._rand((case.B, case.D), case, requires_grad=True)
            b_dec = self._rand((case.D,), case, requires_grad=True)
            if normalize_activations == "constant_norm_rescale":
                norm_coeff = self._rand((case.B, 1), case, requires_grad=True, positive=True)
                norm_aux = (norm_coeff,)
            elif normalize_activations == "layer_norm":
                ln_mu = self._rand((case.B, 1), case, requires_grad=True)
                ln_std = self._rand((case.B, 1), case, requires_grad=True, positive=True)
                norm_aux = (ln_mu, ln_std)
            else:
                norm_aux = ()

            def forward() -> torch.Tensor:
                sae_out = decoder_partial + b_dec
                if normalize_activations == "constant_norm_rescale":
                    sae_out = sae_out / norm_coeff
                elif normalize_activations == "layer_norm":
                    sae_out = sae_out * ln_std + ln_mu
                return (sae_out - sae_in).pow(2).sum(dim=-1).mean()

            def prepare_backward(output: torch.Tensor) -> torch.Tensor:
                return torch.ones_like(output)

            def backward(
                output: torch.Tensor, upstream: torch.Tensor
            ) -> tuple[torch.Tensor | None, ...]:
                return torch.autograd.grad(
                    output,
                    (decoder_partial, sae_in, b_dec, *norm_aux),
                    grad_outputs=upstream,
                )

            return {
                "forward": forward,
                "prepare_backward": prepare_backward,
                "backward": backward,
                "owned": (sae_in, decoder_partial, b_dec, *norm_aux),
            }

        def setup_b_dec_accumulation() -> dict[str, Any]:
            grad_from_input = self._rand((case.D,), case)
            grad_from_decode = self._rand((case.D,), case)

            def operation() -> torch.Tensor:
                return grad_from_input + grad_from_decode

            return {
                "operation": operation,
                "owned": (grad_from_input, grad_from_decode),
            }

        regions = [
            CompositeRegion(
                "reconstruction_loss_forward_backward",
                setup_reconstruction_loss,
                staged_autograd=True,
            ),
            CompositeRegion(
                "b_dec_branch_gradient_accumulation",
                setup_b_dec_accumulation,
            ),
        ]
        return PreparedLocalGroup(
            semantic_op="local_compute_bd",
            phase="forward_backward",
            shape_family="BD",
            equation="T_local_BD(B,D,normalize): reconstruction/loss + BD backward",
            driving_vars="B,D,normalize_activations",
            profile_dimensions={"B": case.B, "D": case.D},
            group_shapes={
                "sae_in": [case.B, case.D],
                "decoder_partial": [case.B, case.D],
                "b_dec": [case.D],
                **self._local_bd_norm_shapes(normalize_activations, case),
            },
            op_list=self._local_bd_op_list(normalize_activations),
            runner=self._composite_runner(regions),
        )

    @staticmethod
    def _local_bd_norm_shapes(
        normalize_activations: str, case: ProfileCase
    ) -> dict[str, list[int]]:
        if normalize_activations == "constant_norm_rescale":
            return {"x_norm_coeff": [case.B, 1]}
        if normalize_activations == "layer_norm":
            return {"ln_mu": [case.B, 1], "ln_std": [case.B, 1]}
        return {}

    @staticmethod
    def _local_bd_op_list(normalize_activations: str) -> list[str]:
        ops = ["decoder output + b_dec"]
        if normalize_activations == "constant_norm_rescale":
            ops.append("divide decoder output by stored x_norm_coeff")
        elif normalize_activations == "layer_norm":
            ops.append("restore decoder output with stored ln_std and ln_mu")
        ops.extend(
            [
                "MSE difference, square, sum(-1), mean",
                "loss backward to decoder output, sae_in and b_dec",
                "accumulate b_dec gradients from encode and decode branches",
            ]
        )
        return ops

    def _build_local_bf(self, case: ProfileCase) -> PreparedLocalGroup:
        def setup_encoder_scale() -> dict[str, Any]:
            hidden_mm = self._rand((case.B, case.F), case, requires_grad=True)
            b_enc = self._rand((case.F,), case, requires_grad=True)
            decoder_norm = self._rand(
                (case.F,), case, requires_grad=True, positive=True
            )

            def forward() -> torch.Tensor:
                return (hidden_mm + b_enc) * decoder_norm

            def prepare_backward(output: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(output)

            def backward(
                output: torch.Tensor, upstream: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return torch.autograd.grad(
                    output,
                    (hidden_mm, b_enc, decoder_norm),
                    grad_outputs=upstream,
                )

            return {
                "forward": forward,
                "prepare_backward": prepare_backward,
                "backward": backward,
                "owned": (hidden_mm, b_enc, decoder_norm),
            }

        def setup_decoder_scale() -> dict[str, Any]:
            local_acts = self._rand((case.B, case.F), case, requires_grad=True)
            inv_norm = self._rand(
                (case.F,), case, requires_grad=True, positive=True
            )

            def forward() -> torch.Tensor:
                return local_acts * inv_norm

            def prepare_backward(output: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(output)

            def backward(
                output: torch.Tensor, upstream: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.autograd.grad(
                    output, (local_acts, inv_norm), grad_outputs=upstream
                )

            return {
                "forward": forward,
                "prepare_backward": prepare_backward,
                "backward": backward,
                "owned": (local_acts, inv_norm),
            }

        regions = [
            CompositeRegion("encoder_bias_and_norm_scale", setup_encoder_scale, staged_autograd=True),
            CompositeRegion("decoder_inverse_norm_scale", setup_decoder_scale, staged_autograd=True),
        ]
        return PreparedLocalGroup(
            semantic_op="local_compute_bf_local",
            phase="forward_backward",
            shape_family="BF_LOCAL",
            equation="T_local_BF(B,F_l): encoder/decode feature-width local work",
            driving_vars="B,F_local",
            profile_dimensions={"B": case.B, "F_local_d_sae": case.F},
            group_shapes={
                "encoder_mm_output": [case.B, case.F],
                "local_feature_acts": [case.B, case.F],
                "feature_vectors": [case.F],
            },
            op_list=[
                "encoder add b_enc and multiply decoder norm + backward",
                "decoder local activation multiply inverse decoder norm + backward",
                "activation function kernels explicitly excluded",
            ],
            runner=self._composite_runner(regions),
        )

    def _build_local_fd(self, case: ProfileCase) -> PreparedLocalGroup:
        def setup_decoder_norm_paths() -> dict[str, Any]:
            w_dec = self._rand((case.F, case.D), case, requires_grad=True)

            def forward() -> tuple[torch.Tensor, torch.Tensor]:
                # The real default path computes W_dec.norm twice: once before
                # activation selection and once before decoder GEMM.
                encoder_norm = w_dec.norm(dim=-1)
                decoder_inv_norm = 1.0 / w_dec.norm(dim=-1)
                return encoder_norm, decoder_inv_norm

            def prepare_backward(
                output: tuple[torch.Tensor, torch.Tensor]
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.empty_like(output[0]), torch.empty_like(output[1])

            def backward(
                output: tuple[torch.Tensor, torch.Tensor],
                upstream: tuple[torch.Tensor, torch.Tensor],
            ) -> torch.Tensor:
                return torch.autograd.grad(
                    output,
                    w_dec,
                    grad_outputs=upstream,
                )[0]

            return {
                "forward": forward,
                "prepare_backward": prepare_backward,
                "backward": backward,
                "owned": (w_dec,),
            }

        def setup_local_grad_clip() -> dict[str, Any]:
            # Match TrainingSAE.clip_grad_norm_ exactly for the local compute
            # boundary.  The real method calls torch.nn.utils.clip_grad_norm_
            # over the SAE parameters.  Distributed scalar reductions, when a
            # TP/DP subclass adds them, remain in the NCCL model.
            parameters = (
                torch.nn.Parameter(
                    torch.empty(
                        (case.D, case.F),
                        device=case.torch_device,
                        dtype=case.dtype,
                    )
                ),
                torch.nn.Parameter(
                    torch.empty(
                        (case.F, case.D),
                        device=case.torch_device,
                        dtype=case.dtype,
                    )
                ),
                torch.nn.Parameter(
                    torch.empty(
                        (case.F,),
                        device=case.torch_device,
                        dtype=case.dtype,
                    )
                ),
                torch.nn.Parameter(
                    torch.empty(
                        (case.D,),
                        device=case.torch_device,
                        dtype=case.dtype,
                    )
                ),
            )
            # Tiny gradients keep the clipping coefficient at 1.0, so repeated
            # measurements preserve both values and the exact foreach/for-loop
            # dispatch selected by the installed PyTorch version.
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, 1.0e-8)

            def operation() -> torch.Tensor:
                return torch.nn.utils.clip_grad_norm_(parameters, 1.0)

            return {"operation": operation, "owned": parameters}

        regions = [
            CompositeRegion(
                "two_decoder_norm_paths",
                setup_decoder_norm_paths,
                staged_autograd=True,
            ),
            CompositeRegion(
                "local_gradient_norm_and_scale",
                setup_local_grad_clip,
                inference_mode=True,
            ),
        ]
        return PreparedLocalGroup(
            semantic_op="local_compute_fd_local",
            phase="forward_backward_post_backward",
            shape_family="FD_LOCAL",
            equation="T_local_FD(D,F_l): two decoder-norm paths + local grad clipping",
            driving_vars="D,F_local",
            profile_dimensions={"D": case.D, "F_local_d_sae": case.F},
            group_shapes={
                "W_dec": [case.F, case.D],
                "W_enc_grad": [case.D, case.F],
                "W_dec_grad": [case.F, case.D],
                "b_enc_grad": [case.F],
                "b_dec_grad": [case.D],
            },
            op_list=[
                "encoder W_dec row norm",
                "decoder W_dec row norm + reciprocal",
                "combined norm backward to W_dec",
                "torch.nn.utils.clip_grad_norm_ over W_enc/W_dec/b_enc/b_dec",
                "PyTorch-native foreach/for-loop dispatch preserved",
                "distributed TP/DP scalar reductions excluded",
            ],
            runner=self._composite_runner(regions),
        )

    def _build_local_bf_global(
        self, case: ProfileCase, config: SweepConfig
    ) -> PreparedLocalGroup:
        feature_acts = self._rand((case.B, case.F_global), case)
        act_freq_scores = torch.zeros(
            (case.F_global,), device=case.torch_device, dtype=torch.float32
        )
        if config.stats_sync_mode == "immediate":
            n_forward = torch.zeros_like(act_freq_scores)

            def operation() -> tuple[torch.Tensor, torch.Tensor]:
                firing_feats = feature_acts.bool().float()
                did_fire_int = (
                    firing_feats.sum(dim=-2)
                    .bool()
                    .to(torch.int32)
                    .contiguous()
                )
                act_freq_scores.add_(firing_feats.sum(dim=0))
                # DP max/sum collectives are excluded.  Use the local tensor as
                # the post-collective placeholder for the local update kernels.
                did_fire = did_fire_int.bool()
                n_forward.add_(1)
                n_forward[did_fire] = 0
                return act_freq_scores, n_forward

            shapes = {
                "feature_acts": [case.B, case.F_global],
                "feature_vectors": [case.F_global],
            }
            ops = [
                "feature_acts.bool().float()",
                "two reductions over B",
                "did_fire int32/bool/contiguous",
                "act_freq_scores update",
                "n_forward increment and masked reset",
                "stats NCCL collectives excluded",
            ]
        else:
            pending = torch.zeros(
                (case.F_global,), device=case.torch_device, dtype=torch.int32
            )

            def operation() -> tuple[torch.Tensor, torch.Tensor]:
                firing_feats = feature_acts.bool().float()
                did_fire_int = (
                    firing_feats.sum(dim=-2)
                    .bool()
                    .to(torch.int32)
                    .contiguous()
                )
                act_freq_scores.add_(firing_feats.sum(dim=0))
                return act_freq_scores, torch.maximum(pending, did_fire_int)

            shapes = {
                "feature_acts": [case.B, case.F_global],
                "feature_vectors": [case.F_global],
            }
            ops = [
                "feature_acts.bool().float()",
                "two reductions over B",
                "did_fire int32/bool/contiguous",
                "act_freq_scores update",
                "pending_did_fire maximum",
            ]

        return PreparedLocalGroup(
            semantic_op="local_compute_bf_global",
            phase="stats",
            shape_family="BF_GLOBAL",
            equation="T_local_BFg(B,F_g,mode): dense activation statistics",
            driving_vars="B,F_global,stats_sync_mode",
            profile_dimensions={
                "B": case.B,
                "F_global_d_sae": case.F_global,
            },
            group_shapes=shapes,
            op_list=ops,
            runner=self._simple_runner(operation),
        )

    def _build_stats_tail(
        self, case: ProfileCase, config: SweepConfig
    ) -> PreparedLocalGroup:
        if config.stats_sync_mode == "immediate":
            raise ValueError("local_compute_hf_global_tail is not used in immediate mode")
        pending = [
            torch.zeros(
                (case.F_global,), device=case.torch_device, dtype=torch.int32
            )
            for _ in range(1)
        ]
        sample_counts = [float(case.B)]
        n_forward = [
            torch.zeros(
                (case.F_global,), device=case.torch_device, dtype=torch.float32
            )
            for _ in range(1)
        ]
        step_increment = (
            config.stats_sync_interval
            if config.stats_sync_mode == "periodic"
            else 1
        )

        def operation() -> tuple[torch.Tensor, torch.Tensor]:
            did_fire_stack = torch.stack(pending, dim=0)
            sample_count_stack = torch.tensor(
                sample_counts,
                device=case.torch_device,
                dtype=torch.float32,
            )
            # Batched DP max/sum collectives are excluded.
            for idx in range(1):
                did_fire = did_fire_stack[idx].bool()
                n_forward[idx].add_(step_increment)
                n_forward[idx][did_fire] = 0
                pending[idx].zero_()
            return did_fire_stack, sample_count_stack

        return PreparedLocalGroup(
            semantic_op=self.STATS_TAIL_VARIANT,
            phase="stats_tail",
            shape_family="F_GLOBAL_TAIL",
            equation="T_local_Fg(F_g,mode,interval): per-SAE deferred local flush",
            driving_vars="F_global,stats_sync_mode,stats_sync_interval",
            profile_dimensions={"F_global_d_sae": case.F_global},
            group_shapes={
                "pending_stack": [1, case.F_global],
                "sample_count_stack": [1],
                "n_forward": [1, case.F_global],
            },
            op_list=[
                "stack one SAE pending vector",
                "construct one sample-count scalar",
                "per-SAE bool cast, increment, masked reset and pending zero",
                "stats NCCL collectives excluded",
            ],
            runner=self._simple_runner(operation),
            profile_dtype="",
        )

    def _build_local_bf_tp(self, case: ProfileCase) -> PreparedLocalGroup:
        if case.tp <= 1:
            raise ValueError("local_compute_bf_tp requires tp > 1")

        def setup_allgather_cat() -> dict[str, Any]:
            local = self._rand((case.B, case.F), case)

            def operation() -> torch.Tensor:
                outputs = [torch.zeros_like(local) for _ in range(case.tp)]
                return torch.cat(outputs, dim=-1)

            return {"operation": operation, "owned": (local,)}

        def setup_allgather_backward_slice() -> dict[str, Any]:
            grad_full = self._rand((case.B, case.F_global), case)

            def operation() -> torch.Tensor:
                return grad_full[..., : case.F].contiguous()

            return {"operation": operation, "owned": (grad_full,)}

        def setup_decoder_slice_backward() -> dict[str, Any]:
            feature_full = self._rand(
                (case.B, case.F_global), case, requires_grad=True
            )

            def forward() -> torch.Tensor:
                return feature_full[..., : case.F]

            def prepare_backward(output: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(output)

            def backward(
                output: torch.Tensor, upstream: torch.Tensor
            ) -> torch.Tensor:
                return torch.autograd.grad(
                    output, feature_full, grad_outputs=upstream
                )[0]

            return {
                "forward": forward,
                "prepare_backward": prepare_backward,
                "backward": backward,
                "owned": (feature_full,),
            }

        regions = [
            CompositeRegion("allgather_output_buffers_and_cat", setup_allgather_cat),
            CompositeRegion("allgather_backward_local_slice", setup_allgather_backward_slice),
            CompositeRegion(
                "decoder_feature_shard_view_and_backward",
                setup_decoder_slice_backward,
                staged_autograd=True,
            ),
        ]
        return PreparedLocalGroup(
            semantic_op="local_compute_bf_tp",
            phase="tp_layout_forward_backward",
            shape_family="BF_LOCAL_TP",
            equation="T_local_BF_TP(B,F_l,tp): allgather/decoder local layout work",
            driving_vars="B,F_local,tp",
            profile_dimensions={
                "B": case.B,
                "F_local_d_sae": case.F,
                "F_global_d_sae": case.F_global,
                "tp": case.tp,
            },
            group_shapes={
                "local_shard": [case.B, case.F],
                "full_feature": [case.B, case.F_global],
            },
            op_list=[
                "allocate tp allgather output buffers and cat to full width",
                "allgather backward full-gradient shard slice + contiguous",
                "decoder full-feature local view and backward scatter to full gradient",
                "NCCL collective kernels excluded",
            ],
            runner=self._composite_runner(regions),
        )

    def _build_local_bd_tp(self, case: ProfileCase) -> PreparedLocalGroup:
        if case.tp <= 1:
            raise ValueError("local_compute_bd_tp requires tp > 1")

        def setup_allreduce_clone() -> dict[str, Any]:
            decoder_partial = self._rand((case.B, case.D), case)

            def operation() -> torch.Tensor:
                return decoder_partial.clone()

            return {"operation": operation, "owned": (decoder_partial,)}

        def setup_decode_bias_scale_gradient() -> dict[str, Any]:
            b_dec = self._rand((case.D,), case, requires_grad=True)

            def forward() -> torch.Tensor:
                return _scale_gradient(b_dec, 1.0 / case.tp)

            def prepare_backward(output: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(output)

            def backward(
                output: torch.Tensor, upstream: torch.Tensor
            ) -> torch.Tensor:
                return torch.autograd.grad(
                    output, b_dec, grad_outputs=upstream
                )[0]

            return {
                "forward": forward,
                "prepare_backward": prepare_backward,
                "backward": backward,
                "owned": (b_dec,),
            }

        regions = [
            CompositeRegion("allreduce_input_clone", setup_allreduce_clone),
            CompositeRegion(
                "decode_bias_gradient_scaling",
                setup_decode_bias_scale_gradient,
                staged_autograd=True,
            ),
        ]
        return PreparedLocalGroup(
            semantic_op="local_compute_bd_tp",
            phase="tp_layout_forward_backward",
            shape_family="BD_TP",
            equation="T_local_BD_TP(B,D,tp): allreduce clone + decode-bias grad scale",
            driving_vars="B,D,tp",
            profile_dimensions={"B": case.B, "D": case.D, "tp": case.tp},
            group_shapes={
                "decoder_partial": [case.B, case.D],
                "b_dec": [case.D],
            },
            op_list=[
                "clone decoder partial before TP allreduce",
                "identity decode bias forward and 1/tp backward gradient scaling",
                "NCCL allreduce excluded",
            ],
            runner=self._composite_runner(regions),
        )


class AdamOptimizerProfiler(ProfilerPlugin):
    """Profile the exact Adam implementation selected by ``optimizer_impl``."""

    name = "optimizer"

    def case_key(self, case: ProfileCase) -> tuple[Any, ...]:
        # Optimizer is batch/TP-independent once local parameter shapes are known.
        return (
            case.D,
            case.F,
            case.optimizer_impl,
            case.device,
            case.dtype_name,
        )

    def profile_case(
        self,
        case: ProfileCase,
        config: SweepConfig,
    ) -> list[dict[str, Any]]:
        device = case.torch_device
        dtype = case.dtype
        check_device_and_dtype(
            device,
            dtype,
            component=f"{case.optimizer_impl} Adam",
            cuda_required=case.optimizer_impl == "fused",
        )

        task_started = time.perf_counter()
        params: list[torch.nn.Parameter] = []
        optimizer: torch.optim.Optimizer | None = None
        try:
            device_context = (
                torch.cuda.device(device) if device.type == "cuda" else _NullContext()
            )
            with device_context:
                params = self._build_parameters(case)
                for parameter in params:
                    parameter.grad = torch.ones_like(parameter)

                optimizer = self._build_optimizer(params, case, config)

                rows: list[dict[str, Any]] = []
                if config.profile_optimizer_cold_step:
                    cold_stats = self._measure_one_step(optimizer, device)
                    rows.append(
                        self._success_row(
                            case,
                            config,
                            semantic_op=f"{case.optimizer_impl}_adam_cold_step",
                            stats=cold_stats,
                            params=params,
                            optimizer=optimizer,
                        )
                    )
                else:
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                steady_timer = OperationTimer(device, config.warmup, config.repeats)
                steady_stats = steady_timer.measure(
                    optimizer.step,
                    inference_mode=False,
                )
                rows.append(
                    self._success_row(
                        case,
                        config,
                        semantic_op=f"{case.optimizer_impl}_adam_steady_step",
                        stats=steady_stats,
                        params=params,
                        optimizer=optimizer,
                    )
                )

                elapsed = time.perf_counter() - task_started
                for row in rows:
                    row["component_wall_seconds"] = elapsed
                return rows
        except Exception as exc:
            if config.fail_fast:
                raise
            return [self._error_row(case, config, exc, time.perf_counter() - task_started)]
        finally:
            if optimizer is not None:
                del optimizer
            for parameter in params:
                parameter.grad = None
            params.clear()
            cleanup_device(device, config.empty_cache_between_ops)

    @staticmethod
    def _build_optimizer(
        params: Sequence[torch.nn.Parameter],
        case: ProfileCase,
        config: SweepConfig,
    ) -> torch.optim.Optimizer:
        kwargs: dict[str, Any] = {
            "lr": config.optimizer_lr,
            "betas": (config.optimizer_beta1, config.optimizer_beta2),
            "eps": config.optimizer_eps,
            "weight_decay": config.optimizer_weight_decay,
            "amsgrad": config.optimizer_amsgrad,
            "maximize": config.optimizer_maximize,
            "capturable": config.optimizer_capturable,
        }
        if case.optimizer_impl == "fused":
            kwargs["fused"] = True
        elif case.optimizer_impl == "foreach":
            kwargs["foreach"] = True
        elif case.optimizer_impl == "forloop":
            kwargs["foreach"] = False
        elif case.optimizer_impl != "default":
            raise ValueError(f"Unsupported optimizer_impl={case.optimizer_impl!r}")
        return torch.optim.Adam(params, **kwargs)

    @staticmethod
    def _build_parameters(case: ProfileCase) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for _ in range(1):
            params.extend(
                [
                    torch.nn.Parameter(
                        torch.zeros(
                            (case.D, case.F),
                            device=case.torch_device,
                            dtype=case.dtype,
                        )
                    ),
                    torch.nn.Parameter(
                        torch.zeros(
                            (case.F, case.D),
                            device=case.torch_device,
                            dtype=case.dtype,
                        )
                    ),
                    torch.nn.Parameter(
                        torch.zeros(
                            (case.F,),
                            device=case.torch_device,
                            dtype=case.dtype,
                        )
                    ),
                    torch.nn.Parameter(
                        torch.zeros(
                            (case.D,),
                            device=case.torch_device,
                            dtype=case.dtype,
                        )
                    ),
                ]
            )
        return params

    @staticmethod
    def _measure_one_step(
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> TimingStats:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            optimizer.step()
            end.record()
            end.synchronize()
            return summarize_samples([float(start.elapsed_time(end))])
        start_ns = time.perf_counter_ns()
        optimizer.step()
        end_ns = time.perf_counter_ns()
        return summarize_samples([(end_ns - start_ns) / 1_000_000.0])

    @staticmethod
    def _optimizer_state_metadata(
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        state_shapes: list[list[int]] = []
        state_dtypes: list[str] = []
        state_numel = 0
        state_bytes = 0
        state_tensor_count = 0
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    state_tensor_count += 1
                    state_shapes.append(list(value.shape))
                    state_dtypes.append(canonical_dtype_name(value.dtype))
                    state_numel += value.numel()
                    state_bytes += value.numel() * value.element_size()
        return {
            "optimizer_state_tensor_count": state_tensor_count,
            "optimizer_state_numel": state_numel,
            "optimizer_state_bytes": state_bytes,
            "optimizer_state_shapes": state_shapes,
            "optimizer_state_dtypes": state_dtypes,
        }

    def _success_row(
        self,
        case: ProfileCase,
        config: SweepConfig,
        *,
        semantic_op: str,
        stats: TimingStats,
        params: Sequence[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        parameter_shapes = [list(p.shape) for p in params]
        parameter_numels = [p.numel() for p in params]
        parameter_numel = sum(parameter_numels)
        parameter_bytes = sum(p.numel() * p.element_size() for p in params)
        return {
            "profiler": self.name,
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "error_traceback": "",
            "case_id": case.case_id,
            "semantic_op": semantic_op,
            "shape_family": "optimizer",
            "equation": (
                f"Adam {case.optimizer_impl} for one SAE over "
                "{[D,F_local],[F_local,D],[F_local],[D]}"
            ),
            "layout_view": "parameter_tensor_list",
            "B": "",
            "D": case.D,
            "F_local_d_sae": case.F,
            "F_global_d_sae": "",
            "tp": "",
            "H": "",
            "device": str(case.torch_device),
            "dtype": case.dtype_name,
            "warmup": 0 if semantic_op.endswith("cold_step") else config.warmup,
            "repeats": 1 if semantic_op.endswith("cold_step") else config.repeats,
            "optimizer_name": "Adam",
            "optimizer_impl": case.optimizer_impl,
            "optimizer_lr": config.optimizer_lr,
            "optimizer_beta1": config.optimizer_beta1,
            "optimizer_beta2": config.optimizer_beta2,
            "optimizer_eps": config.optimizer_eps,
            "optimizer_weight_decay": config.optimizer_weight_decay,
            "optimizer_amsgrad": config.optimizer_amsgrad,
            "optimizer_maximize": config.optimizer_maximize,
            "optimizer_capturable": config.optimizer_capturable,
            "parameter_tensor_count": len(params),
            "parameter_numel": parameter_numel,
            "parameter_bytes": parameter_bytes,
            "parameter_shapes": parameter_shapes,
            "parameter_numels": parameter_numels,
            **self._optimizer_state_metadata(optimizer),
            **asdict(stats),
        }

    def _error_row(
        self,
        case: ProfileCase,
        config: SweepConfig,
        exc: Exception,
        elapsed: float,
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
                "semantic_op": f"{case.optimizer_impl}_adam",
                "B": "",
                "D": case.D,
                "F_local_d_sae": case.F,
                "F_global_d_sae": "",
                "tp": "",
                "H": "",
                "device": str(case.torch_device),
                "dtype": case.dtype_name,
                "warmup": config.warmup,
                "repeats": config.repeats,
                "optimizer_name": "Adam",
                "optimizer_impl": case.optimizer_impl,
                "component_wall_seconds": elapsed,
                "samples": 0,
            }
        )
        return row


class FusedAdamProfilerAlias(AdamOptimizerProfiler):
    """Backwards-compatible plugin name that always profiles fused Adam."""

    name = "fused_optimizer"

    def case_key(self, case: ProfileCase) -> tuple[Any, ...]:
        return (case.D, case.F, case.device, case.dtype_name)

    def profile_case(
        self,
        case: ProfileCase,
        config: SweepConfig,
    ) -> list[dict[str, Any]]:
        return super().profile_case(replace(case, optimizer_impl="fused"), config)


PLUGIN_FACTORIES: dict[str, Callable[[], ProfilerPlugin]] = {
    GemmProfiler.name: GemmProfiler,
    PreprocessPlugin.name: PreprocessPlugin,
    LocalComputePlugin.name: LocalComputePlugin,
    AdamOptimizerProfiler.name: AdamOptimizerProfiler,
    FusedAdamProfilerAlias.name: FusedAdamProfilerAlias,
}


def create_plugin(name: str) -> ProfilerPlugin:
    try:
        return PLUGIN_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown plugin {name!r}; available={sorted(PLUGIN_FACTORIES)}"
        ) from exc


# -----------------------------------------------------------------------------
# Result schema and writer
# -----------------------------------------------------------------------------


CSV_FIELDS: list[str] = [
    "task_index",
    "task_wall_seconds",
    "component_wall_seconds",
    "worker_pid",
    "worker_exitcode",
    "isolated_subprocess",
    "profiler",
    "status",
    "error_type",
    "error_message",
    "error_traceback",
    "case_id",
    "semantic_op",
    "phase",
    "shape_family",
    "equation",
    "layout_view",
    "driving_vars",
    "group_shapes",
    "op_list",
    "B",
    "D",
    "F_local_d_sae",
    "F_global_d_sae",
    "tp",
    "stats_sync_mode",
    "stats_sync_interval",
    "normalize_activations",
    "H",
    "M",
    "N",
    "K",
    "output_shape",
    "device",
    "dtype",
    "warmup",
    "repeats",
    "flops",
    "median_tflops",
    "lhs_shape",
    "lhs_stride",
    "lhs_is_contiguous",
    "lhs_storage_offset",
    "rhs_shape",
    "rhs_stride",
    "rhs_is_contiguous",
    "rhs_storage_offset",
    "optimizer_name",
    "optimizer_impl",
    "optimizer_lr",
    "optimizer_beta1",
    "optimizer_beta2",
    "optimizer_eps",
    "optimizer_weight_decay",
    "optimizer_amsgrad",
    "optimizer_maximize",
    "optimizer_capturable",
    "parameter_tensor_count",
    "parameter_numel",
    "parameter_bytes",
    "parameter_shapes",
    "parameter_numels",
    "optimizer_state_tensor_count",
    "optimizer_state_numel",
    "optimizer_state_bytes",
    "optimizer_state_shapes",
    "optimizer_state_dtypes",
    "samples",
    "median_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "p10_ms",
    "p90_ms",
    "max_ms",
]


def blank_row() -> dict[str, Any]:
    return {key: "" for key in CSV_FIELDS}


def csv_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


@dataclass(frozen=True)
class OutputPaths:
    csv_path: Path
    json_path: Path


def unique_output_paths(output_dir: Path, requested_name: str) -> OutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(requested_name).stem or "sae_compute_profile"
    suffix = 0
    while True:
        candidate_stem = stem if suffix == 0 else f"{stem}_{suffix}"
        csv_path = output_dir / f"{candidate_stem}.csv"
        json_path = output_dir / f"{candidate_stem}.json"
        if not csv_path.exists() and not json_path.exists():
            return OutputPaths(csv_path=csv_path, json_path=json_path)
        suffix += 1


class ResultWriter:
    """Writes CSV incrementally and JSON at finalization."""

    def __init__(self, paths: OutputPaths, metadata: dict[str, Any]) -> None:
        self.paths = paths
        self.metadata = metadata
        self.rows: list[dict[str, Any]] = []
        self._csv_file = paths.csv_path.open("x", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def write_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            copied = dict(row)
            self.rows.append(copied)
            self._csv_writer.writerow(
                {key: csv_safe(copied.get(key, "")) for key in CSV_FIELDS}
            )
        self._csv_file.flush()

    def finalize(self, summary: dict[str, Any]) -> None:
        payload = {
            "metadata": self.metadata,
            "summary": summary,
            "results": self.rows,
        }
        try:
            with self.paths.json_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        finally:
            self.close()

    def close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.close()


# -----------------------------------------------------------------------------
# Subprocess task execution
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileTask:
    index: int
    plugin_name: str
    case: ProfileCase


def _worker_entry(
    send_conn: Any,
    plugin_name: str,
    case_payload: dict[str, Any],
    config_payload: dict[str, Any],
    task_seed: int,
) -> None:
    """Spawn-safe worker entry point."""

    try:
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)

        case = ProfileCase(**case_payload)
        config = SweepConfig(**config_payload).normalize()
        plugin = create_plugin(plugin_name)
        rows = plugin.profile_case(case, config)
        send_conn.send(
            {
                "kind": "ok",
                "rows": rows,
                "worker_pid": os.getpid(),
            }
        )
    except BaseException as exc:
        # BaseException also reports SystemExit; KeyboardInterrupt in a child
        # should not silently disappear. The parent decides whether to continue.
        try:
            send_conn.send(
                {
                    "kind": "fatal",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_traceback": traceback.format_exc(limit=30),
                    "worker_pid": os.getpid(),
                }
            )
        except Exception:
            pass
    finally:
        try:
            send_conn.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# External framework
# -----------------------------------------------------------------------------


@dataclass
class RunCounters:
    raw_cartesian_cases: int = 0
    unique_tasks: int = 0
    completed_tasks: int = 0
    valid_tasks: int = 0
    skipped_tasks: int = 0
    result_rows: int = 0
    ok_rows: int = 0
    error_rows: int = 0
    timeout_rows: int = 0
    worker_crash_rows: int = 0


class ExternalProfilerFramework:
    """Runs registered plugins over plugin-specific unique case keys."""

    def __init__(
        self,
        config: SweepConfig,
        validator: CaseValidator | None = None,
    ) -> None:
        self.config = config.normalize()
        self.config.validate_inputs()
        self.validator = validator or AlwaysValidCaseValidator()
        self.plugins: list[ProfilerPlugin] = []

    def register(self, plugin: ProfilerPlugin) -> "ExternalProfilerFramework":
        if any(existing.name == plugin.name for existing in self.plugins):
            raise ValueError(f"Plugin {plugin.name!r} is already registered")
        self.plugins.append(plugin)
        return self

    def run(self) -> OutputPaths:
        if not self.plugins:
            raise RuntimeError("No profiler plugins were registered")

        paths = unique_output_paths(
            Path(self.config.output_dir),
            self.config.output_name,
        )
        started_at = datetime.now(timezone.utc)
        run_started_perf = time.perf_counter()
        metadata = build_run_metadata(self.config, self.plugins, started_at)
        counters = RunCounters(raw_cartesian_cases=self._raw_case_count())
        tasks = self._build_tasks()
        counters.unique_tasks = len(tasks)

        print(
            f"[external-profiler] raw cases={counters.raw_cartesian_cases}, "
            f"unique plugin tasks={len(tasks)}, plugins={len(self.plugins)}"
        )
        print(
            "[external-profiler] isolation="
            f"{self.config.subprocess_isolation}, "
            f"timeout={self.config.task_timeout_seconds}s"
        )
        print(f"[external-profiler] CSV : {paths.csv_path}")
        print(f"[external-profiler] JSON: {paths.json_path}")

        writer = ResultWriter(paths, metadata)
        completed_task_wall_seconds: list[float] = []
        fatal: BaseException | None = None
        try:
            for task in tasks:
                case = task.case
                print(
                    f"[{task.index}/{len(tasks)}] plugin={task.plugin_name} "
                    f"B={case.B} D={case.D} F={case.F} tp={case.tp} "
                    f"variant={case.task_variant or '-'} opt={case.optimizer_impl} "
                    f"device={case.device} "
                    f"dtype={case.dtype_name}"
                )

                validation = self.validator.validate(case)
                if not validation.valid:
                    rows = [self._skipped_task_row(task, validation.reason)]
                    counters.skipped_tasks += 1
                else:
                    counters.valid_tasks += 1
                    if self.config.subprocess_isolation:
                        rows = self._run_task_subprocess(task)
                    else:
                        rows = self._run_task_direct(task)

                writer.write_rows(rows)
                counters.completed_tasks += 1
                if rows and rows[0].get("task_wall_seconds") not in {"", None}:
                    completed_task_wall_seconds.append(
                        float(rows[0]["task_wall_seconds"])
                    )
                self._update_counters(counters, rows)

                if self.config.fail_fast and any(
                    row.get("status") not in {"ok", "skipped"} for row in rows
                ):
                    raise RuntimeError(
                        f"fail-fast: task {task.index} ({task.plugin_name}) failed"
                    )
        except BaseException as exc:
            fatal = exc
        finally:
            finished_at = datetime.now(timezone.utc)
            total_wall_seconds = time.perf_counter() - run_started_perf
            average_task_wall = (
                statistics.fmean(completed_task_wall_seconds)
                if completed_task_wall_seconds
                else 0.0
            )
            summary = {
                **asdict(counters),
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                # Backward-compatible name plus explicit new names.
                "elapsed_wall_seconds": total_wall_seconds,
                "profiler_total_wall_seconds": total_wall_seconds,
                "profiler_total_wall_hms": format_duration(total_wall_seconds),
                "average_completed_task_wall_seconds": average_task_wall,
                "aborted": fatal is not None,
                "abort_error_type": type(fatal).__name__ if fatal else "",
                "abort_error_message": str(fatal) if fatal else "",
                "csv_path": str(paths.csv_path),
                "json_path": str(paths.json_path),
            }
            writer.finalize(summary)

        total_wall_seconds = time.perf_counter() - run_started_perf
        print(
            "[external-profiler] complete: "
            f"ok_rows={counters.ok_rows}, error_rows={counters.error_rows}, "
            f"timeout_rows={counters.timeout_rows}, "
            f"worker_crash_rows={counters.worker_crash_rows}, "
            f"skipped_tasks={counters.skipped_tasks}"
        )
        print(
            "[external-profiler] total wall time: "
            f"{format_duration(total_wall_seconds)} "
            f"({total_wall_seconds:.3f} s)"
        )

        if fatal is not None:
            raise fatal
        return paths

    def _build_tasks(self) -> list[ProfileTask]:
        tasks: list[ProfileTask] = []
        task_index = 0
        for plugin in self.plugins:
            for case in plugin.iter_task_cases(self.config):
                task_index += 1
                tasks.append(
                    ProfileTask(
                        index=task_index,
                        plugin_name=plugin.name,
                        case=case,
                    )
                )
        return tasks

    def _run_task_direct(self, task: ProfileTask) -> list[dict[str, Any]]:
        started = time.perf_counter()
        plugin = create_plugin(task.plugin_name)
        try:
            rows = plugin.profile_case(task.case, self.config)
        except BaseException as exc:
            rows = [self._task_error_row(task, exc, status="error")]
        elapsed = time.perf_counter() - started
        self._annotate_task_rows(
            rows,
            task=task,
            task_wall_seconds=elapsed,
            worker_pid=os.getpid(),
            worker_exitcode=0,
            isolated=False,
        )
        return rows

    def _run_task_subprocess(self, task: ProfileTask) -> list[dict[str, Any]]:
        ctx = mp.get_context("spawn")
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_worker_entry,
            args=(
                send_conn,
                task.plugin_name,
                asdict(task.case),
                asdict(self.config),
                self.config.seed + task.index,
            ),
            daemon=False,
        )

        started = time.perf_counter()
        try:
            process.start()
            send_conn.close()
            timeout = (
                None
                if self.config.task_timeout_seconds <= 0
                else self.config.task_timeout_seconds
            )
            process.join(timeout=timeout)

            if process.is_alive():
                self._terminate_worker(process)
                elapsed = time.perf_counter() - started
                rows = [self._timeout_row(task, elapsed)]
                self._annotate_task_rows(
                    rows,
                    task=task,
                    task_wall_seconds=elapsed,
                    worker_pid=process.pid or "",
                    worker_exitcode=process.exitcode if process.exitcode is not None else "",
                    isolated=True,
                )
                return rows

            elapsed = time.perf_counter() - started
            message: dict[str, Any] | None = None
            try:
                if recv_conn.poll(0.25):
                    message = recv_conn.recv()
            except EOFError:
                message = None

            if message is None:
                rows = [
                    self._worker_crash_row(
                        task,
                        exitcode=process.exitcode,
                        message="Worker exited without returning a result",
                    )
                ]
                worker_pid = process.pid or ""
            elif message.get("kind") == "ok":
                rows = list(message.get("rows", []))
                worker_pid = message.get("worker_pid", process.pid or "")
            else:
                rows = [
                    self._worker_crash_row(
                        task,
                        exitcode=process.exitcode,
                        message=str(message.get("error_message", "Worker fatal error")),
                        error_type=str(message.get("error_type", "WorkerFatalError")),
                        error_traceback=str(message.get("error_traceback", "")),
                    )
                ]
                worker_pid = message.get("worker_pid", process.pid or "")

            self._annotate_task_rows(
                rows,
                task=task,
                task_wall_seconds=elapsed,
                worker_pid=worker_pid,
                worker_exitcode=process.exitcode if process.exitcode is not None else "",
                isolated=True,
            )
            return rows
        except BaseException:
            if process.is_alive():
                self._terminate_worker(process)
            raise
        finally:
            try:
                recv_conn.close()
            except Exception:
                pass
            try:
                send_conn.close()
            except Exception:
                pass
            try:
                if not process.is_alive():
                    process.close()
            except Exception:
                pass

    def _terminate_worker(self, process: mp.Process) -> None:
        try:
            process.terminate()
        except Exception:
            pass
        process.join(timeout=self.config.worker_shutdown_grace_seconds)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass
            process.join(timeout=self.config.worker_shutdown_grace_seconds)

    def _annotate_task_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        task: ProfileTask,
        task_wall_seconds: float,
        worker_pid: Any,
        worker_exitcode: Any,
        isolated: bool,
    ) -> None:
        for row in rows:
            row["task_index"] = task.index
            row["task_wall_seconds"] = task_wall_seconds
            row["worker_pid"] = worker_pid
            row["worker_exitcode"] = worker_exitcode
            row["isolated_subprocess"] = isolated
            if not row.get("normalize_activations"):
                row["normalize_activations"] = self.config.normalize_activations

    @staticmethod
    def _update_counters(counters: RunCounters, rows: Sequence[dict[str, Any]]) -> None:
        counters.result_rows += len(rows)
        for row in rows:
            status = row.get("status")
            if status == "ok":
                counters.ok_rows += 1
            elif status == "timeout":
                counters.timeout_rows += 1
                counters.error_rows += 1
            elif status == "worker_crash":
                counters.worker_crash_rows += 1
                counters.error_rows += 1
            elif status == "error":
                counters.error_rows += 1

    def _raw_case_count(self) -> int:
        return (
            len(self.config.d_in_values)
            * len(self.config.local_d_sae_values)
            * len(self.config.batch_sizes)
            * len(self.config.tp_values)
            * len(self.config.optimizer_impls)
            * len(self.config.devices)
            * len(self.config.dtypes)
        )

    def _skipped_task_row(self, task: ProfileTask, reason: str) -> dict[str, Any]:
        row = blank_row()
        row.update(
            {
                "profiler": task.plugin_name,
                "status": "skipped",
                "error_type": "ValidationRejected",
                "error_message": reason,
                "case_id": task.case.case_id,
                "B": task.case.B,
                "D": task.case.D,
                "F_local_d_sae": task.case.F,
                "F_global_d_sae": task.case.F_global,
                "tp": task.case.tp,
                "H": "",
                "normalize_activations": self.config.normalize_activations,
                "device": task.case.device,
                "dtype": task.case.dtype_name,
                "task_index": task.index,
            }
        )
        return row

    @staticmethod
    def _task_error_row(
        task: ProfileTask,
        exc: BaseException,
        *,
        status: str,
    ) -> dict[str, Any]:
        row = blank_row()
        row.update(
            {
                "profiler": task.plugin_name,
                "status": status,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(limit=20),
                "case_id": task.case.case_id,
                "B": task.case.B,
                "D": task.case.D,
                "F_local_d_sae": task.case.F,
                "H": "",
                "device": task.case.device,
                "dtype": task.case.dtype_name,
                "samples": 0,
            }
        )
        return row

    def _timeout_row(self, task: ProfileTask, elapsed: float) -> dict[str, Any]:
        row = blank_row()
        row.update(
            {
                "profiler": task.plugin_name,
                "status": "timeout",
                "error_type": "ProfileTaskTimeout",
                "error_message": (
                    f"Task exceeded timeout={self.config.task_timeout_seconds}s "
                    "and its worker process was terminated"
                ),
                "case_id": task.case.case_id,
                "B": task.case.B,
                "D": task.case.D,
                "F_local_d_sae": task.case.F,
                "H": "",
                "device": task.case.device,
                "dtype": task.case.dtype_name,
                "component_wall_seconds": elapsed,
                "samples": 0,
            }
        )
        return row

    @staticmethod
    def _worker_crash_row(
        task: ProfileTask,
        *,
        exitcode: int | None,
        message: str,
        error_type: str = "WorkerProcessError",
        error_traceback: str = "",
    ) -> dict[str, Any]:
        row = blank_row()
        row.update(
            {
                "profiler": task.plugin_name,
                "status": "worker_crash",
                "error_type": error_type,
                "error_message": f"{message}; exitcode={exitcode}",
                "error_traceback": error_traceback,
                "case_id": task.case.case_id,
                "B": task.case.B,
                "D": task.case.D,
                "F_local_d_sae": task.case.F,
                "H": "",
                "device": task.case.device,
                "dtype": task.case.dtype_name,
                "samples": 0,
            }
        )
        return row


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


def device_metadata(device_text: str) -> dict[str, Any]:
    device = torch.device(device_text)
    data: dict[str, Any] = {
        "requested": device_text,
        "type": device.type,
        "index": device.index,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        if 0 <= index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(index)
            data.update(
                {
                    "resolved_index": index,
                    "name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "total_memory_bytes": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    return data


def build_run_metadata(
    config: SweepConfig,
    plugins: Sequence[ProfilerPlugin],
    started_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "started_at_utc": started_at.isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pid": os.getpid(),
        "plugins": [plugin.name for plugin in plugins],
        "devices": [device_metadata(x) for x in config.devices],
        "config": asdict(config),
        "error_policy": {
            "python_exceptions": "record row and continue unless fail_fast",
            "cuda_oom": "record row; subprocess isolation prevents parent poisoning",
            "worker_crash": "record worker_crash row and continue",
            "task_timeout": (
                "terminate/kill worker, record timeout row, continue"
                if config.subprocess_isolation
                else "not enforceable without subprocess isolation"
            ),
            "partial_results": "CSV is flushed after every task; JSON finalized on abort",
        },
        "semantic_gemms": {
            "encoder_forward": "[B,D]@[D,F]->[B,F]",
            "decoder_forward": "[B,F]@[F,D]->[B,D]",
            "decoder_dgrad": "[B,D]@W_dec.T[D,F]->[B,F]",
            "decoder_wgrad": "acts.T[F,B]@[B,D]->[F,D]",
            "encoder_dgrad": "[B,F]@W_enc.T[F,D]->[B,D]",
            "encoder_wgrad": "x.T[D,B]@[B,F]->[D,F]",
        },
        "local_compute_partition": {
            "principle": (
                "One fitted row per driving-dimension family. Each row aggregates "
                "all activation-independent kernels in that family; activation "
                "forward/backward belongs exclusively to activation_compute."
            ),
            "preprocess_BD": ["preprocess_bd"],
            "local_BD": ["local_compute_bd"],
            "local_BF_LOCAL": ["local_compute_bf_local"],
            "local_FD_LOCAL": ["local_compute_fd_local"],
            "local_BF_GLOBAL": ["local_compute_bf_global"],
            "local_F_GLOBAL_tail_per_sae": ["local_compute_hf_global_tail"],
            "local_BF_TP": ["local_compute_bf_tp"],
            "local_BD_TP": ["local_compute_bd_tp"],
            "activation_excluded": [
                "torch.topk / ReLU / zeros / scatter",
                "activation backward gather/scatter/elementwise kernels",
            ],
        },
        "optimizer_parameter_list_per_sae": ["[D,F]", "[F,D]", "[F]", "[D]"],
        "multi_hook_policy": "not profiled; simulator sums one per-SAE estimate per hook",
        "notes": {
            "F": "local d_sae = global d_sae / TP",
            "F_global": "F_local * tp, constructed on one device for gathered-width local work",
            "tp": "shape-only for local_compute; no process group or extra GPU is used",
            "stats_sync_mode": config.stats_sync_mode,
            "stats_sync_interval": config.stats_sync_interval,
            "normalize_activations": config.normalize_activations,
            "B": "per-rank batch size",
            "H": "legacy CSV field; blank because every row profiles one SAE and the profiler has no H sweep",
            "tensor_setup": "outside timed region",
            "optimizer_steady": "state initialized before steady measurement",
            "total_runtime": "includes all framework and worker overhead",
            "validator": "AlwaysValidCaseValidator placeholder",
        },
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile semantic SAE GEMMs, single-device local compute, and Adam "
            "implementations over arrays of D, local-F, B, tp, H, device, and dtype."
        )
    )
    parser.add_argument(
        "--d-in",
        dest="d_in_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_D_IN_VALUES),
        help=(
            "One or more d_in values; default from DEFAULT_D_IN_VALUES: "
            f"{DEFAULT_D_IN_VALUES}"
        ),
    )
    parser.add_argument(
        "--local-d-sae",
        "--d-sae",
        dest="local_d_sae_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_LOCAL_D_SAE_VALUES),
        help=(
            "One or more LOCAL d_sae values F=global_d_sae/TP; default from "
            f"DEFAULT_LOCAL_D_SAE_VALUES: {DEFAULT_LOCAL_D_SAE_VALUES}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        "--batch-sizes",
        dest="batch_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help=(
            "One or more per-rank batch sizes; default from "
            f"DEFAULT_BATCH_SIZES: {DEFAULT_BATCH_SIZES}"
        ),
    )
    parser.add_argument(
        "--tp",
        dest="tp_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_TP_VALUES),
        help=(
            "Tensor-parallel sizes used only as local shape parameters; "
            "F_global=F_local*tp. No multi-GPU process group is created."
        ),
    )
    parser.add_argument(
        "--device",
        "--devices",
        dest="devices",
        nargs="+",
        default=list(DEFAULT_DEVICES),
        help=(
            "One or more devices, e.g. cuda:0 cuda:1; default from "
            f"DEFAULT_DEVICES: {DEFAULT_DEVICES}"
        ),
    )
    parser.add_argument(
        "--dtype",
        "--dtypes",
        dest="dtypes",
        nargs="+",
        default=list(DEFAULT_DTYPES),
        help=(
            "float32/fp32, float16/fp16, bfloat16/bf16; default from "
            f"DEFAULT_DTYPES: {DEFAULT_DTYPES}"
        ),
    )
    parser.add_argument(
        "--plugins",
        nargs="+",
        choices=sorted(PLUGIN_FACTORIES),
        default=list(DEFAULT_PLUGINS),
        help=(
            "Profiler plugins to run; default from DEFAULT_PLUGINS: "
            f"{DEFAULT_PLUGINS}"
        ),
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--tf32",
        choices=["inherit", "on", "off"],
        default="inherit",
        help="TF32 policy for CUDA float32 matmul",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Write the failed task, save partial results, then abort",
    )
    parser.add_argument(
        "--keep-cache-between-ops",
        action="store_true",
        help="Do not call torch.cuda.empty_cache() between components",
    )

    # Isolation/error handling.
    parser.add_argument(
        "--no-subprocess-isolation",
        action="store_true",
        help=(
            "Run tasks in the parent process. Faster startup, but a hung CUDA "
            "call cannot be timed out and a fatal CUDA error may poison the run."
        ),
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help="Per unique plugin/case timeout; <=0 disables timeout",
    )
    parser.add_argument(
        "--worker-shutdown-grace-seconds",
        type=float,
        default=5.0,
        help="Wait after terminate before kill",
    )

    parser.add_argument(
        "--stats-sync-mode",
        choices=["immediate", "deferred", "periodic"],
        default=DEFAULT_STATS_SYNC_MODE,
        help="MultiSAE statistics mode whose single-device local work is profiled",
    )
    parser.add_argument(
        "--stats-sync-interval",
        type=int,
        default=DEFAULT_STATS_SYNC_INTERVAL,
        help="Periodic statistics flush interval; metadata/occurrence driver for tail row",
    )
    parser.add_argument(
        "--normalize-activations",
        choices=list(NORMALIZE_ACTIVATION_MODES),
        default=DEFAULT_NORMALIZE_ACTIVATIONS,
        help=(
            "SAE normalize_activations mode for preprocess/local-BD kernels; "
            "default from DEFAULT_NORMALIZE_ACTIVATIONS: "
            f"{DEFAULT_NORMALIZE_ACTIVATIONS}"
        ),
    )

    # Adam implementation.
    parser.add_argument(
        "--optimizer-impl",
        dest="optimizer_impls",
        nargs="+",
        choices=["default", "fused", "foreach", "forloop"],
        default=list(DEFAULT_OPTIMIZER_IMPLS),
        help="Adam implementation(s), matching SAE_ADAM_IMPL semantics",
    )
    parser.add_argument("--optimizer-lr", type=float, default=1.0e-3)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.999)
    parser.add_argument("--optimizer-eps", type=float, default=1.0e-8)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-amsgrad", action="store_true")
    parser.add_argument("--optimizer-maximize", action="store_true")
    parser.add_argument("--optimizer-capturable", action="store_true")
    parser.add_argument(
        "--skip-optimizer-cold-step",
        action="store_true",
        help="Initialize Adam state untimed and only save steady-state timing",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        d_in_values=list(args.d_in_values),
        local_d_sae_values=list(args.local_d_sae_values),
        batch_sizes=list(args.batch_sizes),
        tp_values=list(args.tp_values),
        optimizer_impls=list(args.optimizer_impls),
        stats_sync_mode=args.stats_sync_mode,
        stats_sync_interval=args.stats_sync_interval,
        normalize_activations=args.normalize_activations,
        devices=list(args.devices),
        dtypes=list(args.dtypes),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        output_dir=args.output_dir,
        output_name=args.output_name,
        fail_fast=args.fail_fast,
        empty_cache_between_ops=not args.keep_cache_between_ops,
        tf32=args.tf32,
        subprocess_isolation=not args.no_subprocess_isolation,
        task_timeout_seconds=args.task_timeout_seconds,
        worker_shutdown_grace_seconds=args.worker_shutdown_grace_seconds,
        profile_optimizer_cold_step=not args.skip_optimizer_cold_step,
        optimizer_lr=args.optimizer_lr,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_eps=args.optimizer_eps,
        optimizer_weight_decay=args.optimizer_weight_decay,
        optimizer_amsgrad=args.optimizer_amsgrad,
        optimizer_maximize=args.optimizer_maximize,
        optimizer_capturable=args.optimizer_capturable,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        framework = ExternalProfilerFramework(
            config=config,
            validator=AlwaysValidCaseValidator(),
        )
        for plugin_name in args.plugins:
            framework.register(create_plugin(plugin_name))
        framework.run()
        return 0
    except KeyboardInterrupt:
        print("\n[external-profiler] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"[external-profiler] fatal: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
