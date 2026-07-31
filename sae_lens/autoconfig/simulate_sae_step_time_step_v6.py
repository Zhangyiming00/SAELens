#!/usr/bin/env python3
"""One-step deterministic phase-aware simulator for offline multi-hook SAE training.

There is no Random Forest, sklearn model, IDW, training stage, or saved model
bundle in this file.  Every invocation directly reads the profiler CSVs, performs
deterministic interpolation inside each natural operation family, builds the
single/TP/DDP/FSDP execution DAG, and returns the predicted step time.

Prediction rules
----------------
* d_in (D) and TP degree are exact/discrete dimensions; they are never
  interpolated across unrelated model/configuration values.
* Within one natural operation family only, B and SAE feature width may use
  deterministic rectilinear interpolation.  GEMM uses bilinear interpolation
  in (B,F_local) at exact D; BD uses linear interpolation in B at exact D;
  DF/optimizer use linear interpolation in F_local at exact D.
* NCCL is modeled independently with exact/piecewise-linear interpolation in
  message bytes for a fixed collective and communicator size.
* Whole phases are never profiled/interpolated as mixed (B,D)+(B,F)+(D,F)
  surfaces.  Phase labels only build the execution DAG.

Distributed critical path
-------------------------
The scheduler has one compute resource and one communication resource.  Their
isolated profiled durations may overlap without a contention penalty.  DDP
buckets are released when gradients become ready.  FSDP forward prefetch is
next-hook-only.  SHARD_GRAD_OP is the default and uses forward parameter
AllGather plus non-blocking backward ReduceScatter, with no backward parameter
AllGather.  FULL_SHARD remains available for controlled experiments.

Batch interface
---------------
The only exposed batch input is ``--batch-size`` (or CSV column ``batch``), and
it always means GLOBAL ``train_batch_size_tokens`` as used by
run_sae_runner_gpu.py.  The simulator derives the per-rank batch internally for
DDP/FSDP.  No local-batch input or output field is exposed.

Optimizer
---------
Only fused Adam is supported.  The compute profile must contain
``fused_adam_steady_step`` rows produced with ``--optimizer-impl fused``.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================
DEFAULT_COMPUTE_CSV = Path("sae_lens/autoconfig/profile_results/sae_compute_profile.csv")
DEFAULT_ACTIVATION_CSV = Path("sae_lens/autoconfig/profile_results/activation_compute_profile.csv")
DEFAULT_NCCL_CSV = Path("sae_lens/autoconfig/profile_results/nccl_comm_profile.csv")

DEFAULT_D_IN = 4096
DEFAULT_D_SAE = 16384
DEFAULT_BATCH_SIZE = 2048          # GLOBAL batch
DEFAULT_NUM_HOOKS = 1
DEFAULT_TP = 1
DEFAULT_DP_SIZE = 2
DEFAULT_K = 128
DEFAULT_ACTIVATION_TYPE = "topk"
DEFAULT_ACTIVATION_OUTPUT_LAYOUT = "dense"
DEFAULT_DTYPE = "float32"
DEFAULT_OPTIMIZER_IMPL = "fused"  # unified policy: fused only
DEFAULT_STATS_SYNC_MODE = "immediate"
DEFAULT_NORMALIZE_ACTIVATIONS = "none"
DEFAULT_PARALLEL_MODE = "single"  # single | tp | ddp | fsdp
DEFAULT_FSDP_SHARDING_STRATEGY = "shard_grad_op"  # default changed intentionally
DEFAULT_FSDP_FORWARD_PREFETCH = True
DEFAULT_FSDP_BACKWARD_PREFETCH = "backward_post"
DEFAULT_DDP_BUCKET_CAP_MB = 25.0
DEFAULT_BACKWARD_HOOK_ORDER = "reverse"

DEFAULT_TIME_COLUMN = "median_ms"
DEFAULT_COMPUTE_EXTRAPOLATION = "linear"  # error | linear
DEFAULT_ACTIVATION_EXTRAPOLATION = "linear"
DEFAULT_NCCL_EXTRAPOLATION = "linear"
DEFAULT_MAX_EXTRAPOLATION_FACTOR = 2.0
# =============================================================================

class SimulationError(RuntimeError):
    pass


DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
}


def canonical_dtype(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "fp32": "float32", "float": "float32",
        "fp16": "float16", "half": "float16",
        "bf16": "bfloat16",
    }
    key = aliases.get(key, key)
    if key not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype {name!r}")
    return key


def as_float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int(row: Mapping[str, str], key: str) -> int | None:
    value = as_float(row, key)
    return None if value is None else int(round(value))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SimulationError(f"Profile CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


@dataclass(frozen=True)
class InterpolationOptions:
    compute_extrapolation: str = DEFAULT_COMPUTE_EXTRAPOLATION
    activation_extrapolation: str = DEFAULT_ACTIVATION_EXTRAPOLATION
    nccl_extrapolation: str = DEFAULT_NCCL_EXTRAPOLATION
    max_extrapolation_factor: float = DEFAULT_MAX_EXTRAPOLATION_FACTOR


@dataclass(frozen=True)
class Estimate:
    ms: float
    source: str
    neighbors: int = 1


class RectilinearTable:
    """Exact + rectilinear interpolation over natural operation dimensions.

    Exact dimensions are sliced first and must match a profiled value exactly.
    Only dimensions explicitly listed as interpolation dimensions are allowed to
    interpolate.  Duplicate coordinates are collapsed with the median.
    """

    def __init__(self, rows: Sequence[Mapping[str, str]], time_column: str) -> None:
        self.rows = [dict(r) for r in rows if r.get("status", "ok") == "ok"]
        self.time_column = time_column

    @staticmethod
    def _match_text(row: Mapping[str, str], key: str, target: str | int | bool | None) -> bool:
        if target is None:
            return True
        expected = str(target).lower() if isinstance(target, bool) else str(target)
        return str(row.get(key, "")) == expected

    def _matching_rows(
        self,
        *,
        filters: Mapping[str, str | int | bool | None],
        device: str | None,
        time_column: str,
    ) -> list[dict[str, str]]:
        rows = self.rows
        for key, target in filters.items():
            if target is not None:
                rows = [r for r in rows if self._match_text(r, key, target)]
        if device:
            device_rows = [r for r in rows if r.get("device", "") == device]
            if device_rows:
                rows = device_rows
        rows = [r for r in rows if as_float(r, time_column) is not None]
        if not rows:
            raise SimulationError(f"No profile rows match filters={dict(filters)} device={device!r}")
        return rows

    @staticmethod
    def _eq(a: float, b: float) -> bool:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)

    @staticmethod
    def _available(rows: Sequence[Mapping[str, str]], dim: str) -> list[float]:
        return sorted({float(v) for r in rows if (v := as_float(r, dim)) is not None})

    @staticmethod
    def _outside_factor(q: float, lo: float, hi: float) -> float:
        if lo <= q <= hi:
            return 1.0
        if q > hi:
            return q / hi if hi > 0 else float("inf")
        return lo / q if q > 0 else float("inf")

    def _bracket(
        self,
        values: Sequence[float],
        q: float,
        *,
        dim: str,
        extrapolation: str,
        max_factor: float,
    ) -> tuple[float, float, float, bool]:
        vals = sorted(set(float(v) for v in values))
        if not vals:
            raise SimulationError(f"No profile coordinates for dimension {dim}")
        for v in vals:
            if self._eq(q, v):
                return v, v, 0.0, False
        if len(vals) < 2:
            raise SimulationError(
                f"Cannot interpolate {dim}={q:g}: only one profiled coordinate {vals[0]:g}"
            )
        if q < vals[0]:
            if extrapolation != "linear":
                raise SimulationError(f"{dim}={q:g} is below profiled minimum {vals[0]:g}")
            factor = self._outside_factor(q, vals[0], vals[-1])
            if factor > max_factor:
                raise SimulationError(
                    f"{dim}={q:g} is too far below profile domain [{vals[0]:g},{vals[-1]:g}] "
                    f"(factor={factor:.3f} > max={max_factor:g})"
                )
            lo, hi = vals[0], vals[1]
            return lo, hi, (q - lo) / (hi - lo), True
        if q > vals[-1]:
            if extrapolation != "linear":
                raise SimulationError(f"{dim}={q:g} is above profiled maximum {vals[-1]:g}")
            factor = self._outside_factor(q, vals[0], vals[-1])
            if factor > max_factor:
                raise SimulationError(
                    f"{dim}={q:g} is too far above profile domain [{vals[0]:g},{vals[-1]:g}] "
                    f"(factor={factor:.3f} > max={max_factor:g})"
                )
            lo, hi = vals[-2], vals[-1]
            return lo, hi, (q - lo) / (hi - lo), True
        for lo, hi in zip(vals, vals[1:]):
            if lo < q < hi:
                return lo, hi, (q - lo) / (hi - lo), False
        raise AssertionError((dim, q, vals))

    def estimate(
        self,
        *,
        filters: Mapping[str, str | int | bool | None],
        numeric: Mapping[str, float],
        exact_dims: Sequence[str],
        interp_dims: Sequence[str],
        device: str | None = None,
        time_column: str | None = None,
        extrapolation: str = "error",
        max_extrapolation_factor: float = DEFAULT_MAX_EXTRAPOLATION_FACTOR,
    ) -> Estimate:
        col = time_column or self.time_column
        rows = self._matching_rows(filters=filters, device=device, time_column=col)

        # Exact/discrete numeric dimensions are never interpolated.
        for dim in exact_dims:
            if dim not in numeric:
                raise SimulationError(f"Missing exact dimension {dim} in query")
            q = float(numeric[dim])
            avail = self._available(rows, dim)
            if not any(self._eq(q, v) for v in avail):
                raise SimulationError(
                    f"Exact dimension {dim}={q:g} was not profiled for filters={dict(filters)}; "
                    f"available={avail}"
                )
            rows = [r for r in rows if (v := as_float(r, dim)) is not None and self._eq(v, q)]

        # All interpolation dimensions must be present numerically.
        for dim in interp_dims:
            if dim not in numeric:
                raise SimulationError(f"Missing interpolation dimension {dim} in query")
            if not any(as_float(r, dim) is not None for r in rows):
                raise SimulationError(f"Profile rows do not expose interpolation dimension {dim}")

        # Exact full coordinate bypasses interpolation.
        exact_values: list[float] = []
        for r in rows:
            if all(
                (v := as_float(r, d)) is not None and self._eq(v, float(numeric[d]))
                for d in interp_dims
            ):
                value = as_float(r, col)
                if value is not None:
                    exact_values.append(float(value))
        if exact_values:
            return Estimate(float(statistics.median(exact_values)), "exact", len(exact_values))

        if not interp_dims:
            vals = [float(v) for r in rows if (v := as_float(r, col)) is not None]
            if not vals:
                raise SimulationError(f"No exact timing value for filters={dict(filters)} numeric={dict(numeric)}")
            return Estimate(float(statistics.median(vals)), "exact_slice", len(vals))

        brackets: list[tuple[str, float, float, float, bool]] = []
        for dim in interp_dims:
            lo, hi, w, extra = self._bracket(
                self._available(rows, dim), float(numeric[dim]), dim=dim,
                extrapolation=extrapolation, max_factor=max_extrapolation_factor,
            )
            brackets.append((dim, lo, hi, w, extra))

        # Collapse duplicate coordinates to medians, then read all required corners.
        grouped: dict[tuple[float, ...], list[float]] = defaultdict(list)
        for r in rows:
            coord_vals: list[float] = []
            ok = True
            for dim in interp_dims:
                v = as_float(r, dim)
                if v is None:
                    ok = False
                    break
                coord_vals.append(float(v))
            value = as_float(r, col)
            if ok and value is not None:
                grouped[tuple(coord_vals)].append(float(value))
        med = {coord: float(statistics.median(vals)) for coord, vals in grouped.items()}

        axis_choices: list[list[tuple[float, float]]] = []
        for _, lo, hi, w, _ in brackets:
            if self._eq(lo, hi):
                axis_choices.append([(lo, 1.0)])
            else:
                axis_choices.append([(lo, 1.0 - w), (hi, w)])

        pred = 0.0
        used = 0
        missing: list[tuple[float, ...]] = []
        for combo in itertools.product(*axis_choices):
            coord = tuple(v for v, _ in combo)
            weight = math.prod(weight for _, weight in combo)
            if coord not in med:
                missing.append(coord)
                continue
            pred += weight * med[coord]
            used += 1
        if missing:
            raise SimulationError(
                f"Rectilinear interpolation requires missing corner(s) {missing} for "
                f"filters={dict(filters)} interp_dims={list(interp_dims)}"
            )
        extrapolated = any(x[4] for x in brackets)
        kind = {1: "linear", 2: "bilinear"}.get(len(interp_dims), f"{len(interp_dims)}d_linear")
        if extrapolated:
            kind += "_extrapolated"
        return Estimate(float(pred), kind, used)


# exact_dims are discrete/structural; interp_dims are only natural continuous-ish knobs.
COMPUTE_INTERP_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # GEMM: D is model-defined and exact; interpolate only B and local feature width.
    "encoder_forward": (("D",), ("B", "F_local_d_sae")),
    "decoder_forward": (("D",), ("B", "F_local_d_sae")),
    "decoder_dgrad": (("D",), ("B", "F_local_d_sae")),
    "decoder_wgrad": (("D",), ("B", "F_local_d_sae")),
    "encoder_dgrad": (("D",), ("B", "F_local_d_sae")),
    "encoder_wgrad": (("D",), ("B", "F_local_d_sae")),
    # Local families.
    "preprocess_bd_forward": (("D",), ("B",)),
    "preprocess_bd_backward": (("D",), ("B",)),
    "local_bd_loss_forward": (("D",), ("B",)),
    "local_bd_loss_backward": (("D",), ("B",)),
    "local_d_bdec_grad_accum": (("D",), ()),
    "local_bf_encoder_forward": ((), ("B", "F_local_d_sae")),
    "local_bf_encoder_backward": ((), ("B", "F_local_d_sae")),
    "local_bf_decoder_forward": ((), ("B", "F_local_d_sae")),
    "local_bf_decoder_backward": ((), ("B", "F_local_d_sae")),
    "local_fd_norm_forward": (("D",), ("F_local_d_sae",)),
    "local_fd_norm_backward": (("D",), ("F_local_d_sae",)),
    "local_grad_clip_total": (("D",), ("F_local_d_sae",)),
    "local_grad_norm_scan": (("D",), ("F_local_d_sae",)),
    "local_grad_scale": (("D",), ("F_local_d_sae",)),
    "local_fd_grad_clip": (("D",), ("F_local_d_sae",)),
    "local_bf_global_stats": ((), ("B", "F_global_d_sae")),
    "local_bf_tp_forward": (("tp",), ("B", "F_local_d_sae")),
    "local_bf_tp_backward": (("tp",), ("B", "F_local_d_sae")),
    "local_bd_tp_forward": (("D", "tp"), ("B",)),
    "local_d_tp_backward": (("D", "tp"), ()),
}


@dataclass
class Profiles:
    compute: RectilinearTable
    activation: RectilinearTable
    nccl: RectilinearTable
    options: InterpolationOptions
    compute_device: str | None
    activation_device: str | None
    nccl_backend: str
    nccl_algo: str | None = None
    nccl_proto: str | None = None
    nccl_p2p_level: str | None = None
    prefer_clean_activation: bool = True

    @classmethod
    def load(
        cls,
        compute_csv: Path,
        activation_csv: Path,
        nccl_csv: Path,
        *,
        time_column: str,
        options: InterpolationOptions,
        compute_device: str | None,
        activation_device: str | None,
        nccl_backend: str,
        nccl_algo: str | None,
        nccl_proto: str | None,
        nccl_p2p_level: str | None,
    ) -> "Profiles":
        compute_rows = load_csv(compute_csv)
        if not any(r.get("schedule_stage", "") for r in compute_rows if r.get("status", "ok") == "ok"):
            raise SimulationError(
                "Compute CSV is not phase-aware (missing schedule_stage). Re-run the phase-aware compute profiler."
            )
        # Unified policy: optimizer must be fused; never silently substitute foreach.
        fused = [
            r for r in compute_rows
            if r.get("status", "ok") == "ok"
            and r.get("semantic_op") == "fused_adam_steady_step"
            and r.get("optimizer_impl") == "fused"
        ]
        if not fused:
            raise SimulationError(
                "Compute CSV has no fused Adam profile rows. This simulator uses fused Adam only. "
                "Re-run external_gemm_optim_profiler.py with --optimizer-impl fused."
            )
        return cls(
            compute=RectilinearTable(compute_rows, time_column),
            activation=RectilinearTable(load_csv(activation_csv), time_column),
            nccl=RectilinearTable(load_csv(nccl_csv), time_column),
            options=options,
            compute_device=compute_device,
            activation_device=activation_device,
            nccl_backend=nccl_backend,
            nccl_algo=nccl_algo or None,
            nccl_proto=nccl_proto or None,
            nccl_p2p_level=nccl_p2p_level or None,
        )

    def compute_ms(
        self,
        op: str,
        *,
        B: int,
        D: int,
        F_local: int,
        F_global: int,
        tp: int,
        dtype: str,
        optimizer_impl: str,
        normalize_activations: str,
        stats_sync_mode: str = "immediate",
    ) -> Estimate:
        if optimizer_impl != "fused":
            raise SimulationError("Only fused Adam is supported")
        if op == "fused_adam_steady_step":
            exact_dims, interp_dims = (("D",), ("F_local_d_sae",))
        else:
            spec = COMPUTE_INTERP_SPEC.get(op)
            if spec is None:
                raise SimulationError(f"Unknown compute semantic op {op!r}")
            exact_dims, interp_dims = spec
        all_values = {
            "B": float(B), "D": float(D), "F_local_d_sae": float(F_local),
            "F_global_d_sae": float(F_global), "tp": float(tp),
        }
        numeric = {d: all_values[d] for d in (*exact_dims, *interp_dims)}
        filters: dict[str, str | int | bool | None] = {"semantic_op": op, "dtype": dtype}
        candidates = [r for r in self.compute.rows if r.get("semantic_op") == op]
        if op == "fused_adam_steady_step":
            filters["optimizer_impl"] = "fused"
        if candidates and any(r.get("normalize_activations", "") for r in candidates):
            filters["normalize_activations"] = normalize_activations
        if op == "local_bf_global_stats" and candidates and any(r.get("stats_sync_mode", "") for r in candidates):
            filters["stats_sync_mode"] = stats_sync_mode
        return self.compute.estimate(
            filters=filters, numeric=numeric, exact_dims=exact_dims, interp_dims=interp_dims,
            device=self.compute_device, extrapolation=self.options.compute_extrapolation,
            max_extrapolation_factor=self.options.max_extrapolation_factor,
        )

    def activation_ms(
        self,
        semantic_op: str,
        *,
        B: int,
        F_global: int,
        k: int,
        activation_type: str,
        output_layout: str,
        dtype: str,
    ) -> Estimate:
        rows = self.activation.rows
        time_col = self.activation.time_column
        if self.prefer_clean_activation and any(r.get("clean_ms", "") for r in rows):
            time_col = "clean_ms"
        # TopK is currently a low-cost placeholder; interpolate its natural k
        # dimension as well so common k values (e.g. 128 between 64 and 256) work.
        exact_dims = ()
        interp_dims = ("B", "F_global", "k") if activation_type == "topk" else ("B", "F_global")
        numeric = {"B": float(B), "F_global": float(F_global)}
        if activation_type == "topk":
            numeric["k"] = float(k)
        return self.activation.estimate(
            filters={
                "semantic_op": semantic_op,
                "activation_type": activation_type,
                "output_layout": output_layout,
                "dtype": dtype,
            },
            numeric=numeric, exact_dims=exact_dims, interp_dims=interp_dims,
            device=self.activation_device, time_column=time_col,
            extrapolation=self.options.activation_extrapolation,
            max_extrapolation_factor=self.options.max_extrapolation_factor,
        )

    def nccl_ms(self, collective: str, *, buffer_bytes: int, group_size: int, dtype: str) -> Estimate:
        filters: dict[str, str | int | bool | None] = {
            "collective": collective,
            "dtype": dtype,
            "backend": self.nccl_backend,
        }
        if self.nccl_algo is not None:
            filters["nccl_algo"] = self.nccl_algo
        if self.nccl_proto is not None:
            filters["nccl_proto"] = self.nccl_proto
        if self.nccl_p2p_level is not None:
            filters["nccl_p2p_level"] = self.nccl_p2p_level
        rows = self.nccl.rows
        if any(r.get("group_size", "") for r in rows):
            filters["group_size"] = group_size
        else:
            filters["tp"] = group_size
        if any(r.get("row_kind") == "group_max" for r in rows):
            filters["row_kind"] = "group_max"
        # NCCL is a one-dimensional piecewise-linear message-size curve.
        return self.nccl.estimate(
            filters=filters,
            numeric={"buffer_bytes": float(max(1, buffer_bytes))},
            exact_dims=(), interp_dims=("buffer_bytes",),
            extrapolation=self.options.nccl_extrapolation,
            max_extrapolation_factor=self.options.max_extrapolation_factor,
        )



@dataclass(frozen=True)
class HookSpec:
    name: str
    d_in: int
    d_sae: int
    k: int


@dataclass
class HookTimes:
    hook: HookSpec
    F_local: int
    forward_ms: float
    bwd0_ms: float
    bwd1_ms: float
    bwd2_ms: float
    clip_total_ms: float
    clip_norm_ms: float
    clip_scale_ms: float
    stats_local_ms: float
    optimizer_ms: float
    component_sources: dict[str, str] = field(default_factory=dict)

    @property
    def backward_ms(self) -> float:
        return self.bwd0_ms + self.bwd1_ms + self.bwd2_ms


@dataclass
class TimelineEvent:
    resource: str
    name: str
    hook: str
    phase: str
    start_ms: float
    end_ms: float
    duration_ms: float
    payload_bytes: int | None = None
    collective: str | None = None
    note: str = ""


@dataclass
class SimulationResult:
    parallel_mode: str
    total_ms: float
    tokens_per_s: float
    compute_work_ms: float
    comm_work_ms: float
    exposed_comm_tail_ms: float
    hooks: list[dict[str, Any]]
    timeline: list[TimelineEvent]
    warnings: list[str]
    metadata: dict[str, Any]


class EventScheduler:
    def __init__(self) -> None:
        self.compute_available = 0.0
        self.comm_available = 0.0
        self.events: list[TimelineEvent] = []
        self.compute_work = 0.0
        self.comm_work = 0.0

    def compute(self, name: str, hook: str, phase: str, duration: float, *, earliest: float | None = None, note: str = "") -> TimelineEvent:
        start = self.compute_available
        if earliest is not None:
            start = max(start, earliest)
        end = start + max(0.0, duration)
        ev = TimelineEvent("compute", name, hook, phase, start, end, max(0.0, duration), note=note)
        self.events.append(ev)
        self.compute_available = end
        self.compute_work += max(0.0, duration)
        return ev

    def comm(self, name: str, hook: str, phase: str, duration: float, *, release: float, payload_bytes: int, collective: str, note: str = "") -> TimelineEvent:
        start = max(self.comm_available, release)
        end = start + max(0.0, duration)
        ev = TimelineEvent("comm", name, hook, phase, start, end, max(0.0, duration), payload_bytes, collective, note)
        self.events.append(ev)
        self.comm_available = end
        self.comm_work += max(0.0, duration)
        return ev

    def barrier(self, when: float) -> None:
        self.compute_available = max(self.compute_available, when)

    @property
    def wall(self) -> float:
        return max(self.compute_available, self.comm_available)


class StepTimeSimulator:
    def __init__(
        self,
        profiles: Profiles,
        *,
        parallel_mode: str,
        tp: int,
        dp_size: int,
        dtype: str,
        activation_type: str,
        activation_output_layout: str,
        optimizer_impl: str,
        stats_sync_mode: str,
        normalize_activations: str,
        fsdp_sharding_strategy: str,
        fsdp_forward_prefetch: bool,
        fsdp_backward_prefetch: str,
        ddp_bucket_cap_mb: float,
        backward_hook_order: str,
    ) -> None:
        self.p = profiles
        self.mode = parallel_mode
        self.tp = tp
        self.dp = dp_size
        self.dtype = canonical_dtype(dtype)
        self.itemsize = DTYPE_BYTES[self.dtype]
        self.activation_type = activation_type
        self.activation_output_layout = activation_output_layout
        self.optimizer_impl = optimizer_impl
        self.stats_sync_mode = stats_sync_mode
        self.normalize_activations = normalize_activations
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.fsdp_forward_prefetch = fsdp_forward_prefetch
        self.fsdp_backward_prefetch = fsdp_backward_prefetch
        self.ddp_bucket_cap_bytes = int(ddp_bucket_cap_mb * 1024 * 1024)
        self.backward_hook_order = backward_hook_order
        self.warnings: list[str] = []
        self._validate()

    def _validate(self) -> None:
        if self.mode not in {"single", "tp", "ddp", "fsdp"}:
            raise ValueError("parallel_mode must be single|tp|ddp|fsdp")
        if self.mode == "tp" and self.tp <= 1:
            raise ValueError("parallel_mode=tp requires --tp > 1")
        if self.mode in {"ddp", "fsdp"}:
            if self.dp <= 1:
                raise ValueError(f"parallel_mode={self.mode} requires --dp-size > 1")
            if self.tp != 1:
                raise ValueError(
                    "This simulator version treats TP, DDP and FSDP as separate modes; use --tp 1 for DDP/FSDP."
                )
        if self.fsdp_sharding_strategy not in {"full_shard", "shard_grad_op"}:
            raise ValueError("fsdp_sharding_strategy must be full_shard|shard_grad_op")
        if self.fsdp_backward_prefetch not in {"none", "backward_pre", "backward_post"}:
            raise ValueError("fsdp_backward_prefetch must be none|backward_pre|backward_post")
        if self.backward_hook_order not in {"reverse", "forward"}:
            raise ValueError("backward_hook_order must be reverse|forward")
        if self.optimizer_impl != "fused":
            raise ValueError("Only optimizer_impl=fused is supported by this simulator")
        if self.stats_sync_mode != "immediate" and self.mode in {"ddp", "fsdp"}:
            self.warnings.append(
                "Distributed periodic/deferred stats are not phase-modeled in v1; only local stats cost is included. "
                "Use immediate for trace-faithful DDP/FSDP validation."
            )

    def _warn_if_extrapolated(self, label: str, est: Estimate) -> None:
        if "extrapolated" in est.source:
            self.warnings.append(f"{label} used {est.source}; query is outside the sampled interpolation grid")

    def _cm(self, op: str, hook: HookSpec, F_local: int) -> Estimate:
        est = self.p.compute_ms(
            op, B=self.B, D=hook.d_in, F_local=F_local, F_global=hook.d_sae, tp=self.tp,
            dtype=self.dtype, optimizer_impl=self.optimizer_impl, normalize_activations=self.normalize_activations,
            stats_sync_mode=self.stats_sync_mode,
        )
        self._warn_if_extrapolated(f"compute:{op}", est)
        return est

    def _make_hook_times(self, hook: HookSpec) -> HookTimes:
        if hook.d_sae % self.tp != 0:
            raise SimulationError(f"Hook {hook.name}: d_sae={hook.d_sae} not divisible by tp={self.tp}")
        F = hook.d_sae // self.tp
        sources: dict[str, str] = {}

        def c(op: str) -> float:
            est = self._cm(op, hook, F); sources[op] = est.source; return est.ms

        act_f = self.p.activation_ms(
            "activation_forward", B=self.B, F_global=hook.d_sae, k=hook.k,
            activation_type=self.activation_type, output_layout=self.activation_output_layout, dtype=self.dtype,
        )
        act_fb = self.p.activation_ms(
            "activation_forward_backward", B=self.B, F_global=hook.d_sae, k=hook.k,
            activation_type=self.activation_type, output_layout=self.activation_output_layout, dtype=self.dtype,
        )
        self._warn_if_extrapolated("activation:forward", act_f)
        self._warn_if_extrapolated("activation:forward_backward", act_fb)
        act_b = max(0.0, act_fb.ms - act_f.ms)
        sources["activation_forward"] = act_f.source
        sources["activation_backward"] = f"difference({act_fb.source}-{act_f.source})"

        fwd = sum([
            c("preprocess_bd_forward"), c("encoder_forward"), c("local_bf_encoder_forward"),
            c("local_fd_norm_forward"), act_f.ms, c("local_bf_decoder_forward"),
            c("decoder_forward"), c("local_bd_loss_forward"),
        ])
        if self.mode == "tp":
            fwd += c("local_bf_tp_forward") + c("local_bd_tp_forward")

        # Stage 0 ends when the first large DDP matrix gradient (W_dec) is ready.
        b0 = sum([
            c("local_bd_loss_backward"), c("decoder_dgrad"), c("decoder_wgrad"),
            c("local_bf_decoder_backward"), act_b, c("local_bf_encoder_backward"),
            c("local_fd_norm_backward"),
        ])
        if self.mode == "tp":
            b0 += c("local_bf_tp_backward")
        # Stage 1 ends when W_enc is ready.
        b1 = c("encoder_dgrad") + c("encoder_wgrad")
        # Stage 2 is the remaining input/b_dec tail.
        b2 = c("preprocess_bd_backward") + c("local_d_bdec_grad_accum")
        if self.mode == "tp":
            b2 += c("local_d_tp_backward")

        # New schema has split norm/scale plus exact total. Fall back to old total-only row.
        try:
            clip_norm = c("local_grad_norm_scan")
            clip_scale = c("local_grad_scale")
        except SimulationError:
            clip_norm = 0.0; clip_scale = 0.0
        try:
            clip_total = c("local_grad_clip_total")
        except SimulationError:
            clip_total = c("local_fd_grad_clip")
        stats = c("local_bf_global_stats")
        opt_op = "fused_adam_steady_step"
        optimizer = c(opt_op)
        return HookTimes(hook, F, fwd, b0, b1, b2, clip_total, clip_norm, clip_scale, stats, optimizer, sources)

    def _nccl(self, collective: str, bytes_: int, group: int) -> float:
        est = self.p.nccl_ms(collective, buffer_bytes=max(self.itemsize, int(bytes_)), group_size=group, dtype=self.dtype)
        self._warn_if_extrapolated(f"nccl:{collective}:{bytes_}B:g{group}", est)
        return est.ms

    def _param_bytes(self, h: HookTimes) -> int:
        D, F = h.hook.d_in, h.F_local
        return (2 * D * F + F + D) * self.itemsize

    def _fsdp_shard_bytes(self, h: HookTimes) -> int:
        full = self._param_bytes(h)
        # Approximate FSDP flat-param padding to whole elements.
        elems = math.ceil((full / self.itemsize) / self.dp)
        return max(self.itemsize, elems * self.itemsize)

    def _stats_phase(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        for h in hooks:
            sched.compute("stats_local", h.hook.name, "stats", h.stats_local_ms)
            if self.mode in {"ddp", "fsdp"} and self.stats_sync_mode == "immediate":
                vector_bytes = h.hook.d_sae * 4  # did_fire int32
                dur = self._nccl("allreduce", vector_bytes, self.dp)
                ev = sched.comm("stats_did_fire_allreduce", h.hook.name, "stats", dur,
                                release=sched.compute_available, payload_bytes=vector_bytes, collective="allreduce")
                sched.barrier(ev.end_ms)
                dur = self._nccl("allreduce", 4, self.dp)
                ev = sched.comm("stats_sample_count_allreduce", h.hook.name, "stats", dur,
                                release=sched.compute_available, payload_bytes=4, collective="allreduce")
                sched.barrier(ev.end_ms)

    def _forward_plain(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        for h in hooks:
            sched.compute("sae_forward", h.hook.name, "forward", h.forward_ms)
            if self.mode == "tp":
                # Current TP implementation has blocking collectives in the forward graph.
                ag_bytes = self.B * h.F_local * self.itemsize
                ag = sched.comm("tp_hidden_pre_allgather", h.hook.name, "forward", self._nccl("allgather", ag_bytes, self.tp),
                                release=sched.compute_available, payload_bytes=ag_bytes, collective="allgather")
                sched.barrier(ag.end_ms)
                ar_bytes = self.B * h.hook.d_in * self.itemsize
                ar = sched.comm("tp_decoder_allreduce", h.hook.name, "forward", self._nccl("allreduce", ar_bytes, self.tp),
                                release=sched.compute_available, payload_bytes=ar_bytes, collective="allreduce")
                sched.barrier(ar.end_ms)

    def _backward_plain_or_tp(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        order = list(reversed(hooks)) if self.backward_hook_order == "reverse" else list(hooks)
        for h in order:
            sched.compute("backward_pre_wdec_ready", h.hook.name, "backward", h.bwd0_ms)
            sched.compute("backward_wdec_to_wenc_ready", h.hook.name, "backward", h.bwd1_ms)
            sched.compute("backward_post_wenc", h.hook.name, "backward", h.bwd2_ms)
            if self.mode == "tp":
                # TP backward collectives currently occur on the graph/post-backward path; keep blocking v1 semantics.
                bdec_bytes = h.hook.d_in * self.itemsize
                ev = sched.comm("tp_bdec_grad_allreduce", h.hook.name, "backward", self._nccl("allreduce", bdec_bytes, self.tp),
                                release=sched.compute_available, payload_bytes=bdec_bytes, collective="allreduce")
                sched.barrier(ev.end_ms)
        # TP global gradient norm scalar reduction happens before local clip scale.
        for h in hooks:
            if self.mode == "tp":
                ev = sched.comm("tp_grad_norm_allreduce", h.hook.name, "post_backward", self._nccl("allreduce", 4, self.tp),
                                release=sched.compute_available, payload_bytes=4, collective="allreduce")
                sched.barrier(ev.end_ms)
            sched.compute("grad_clip", h.hook.name, "post_backward", h.clip_total_ms)

    @dataclass(frozen=True)
    class _GradItem:
        name: str
        hook: str
        bytes: int
        ready: float

    @dataclass(frozen=True)
    class _Bucket:
        name: str
        bytes: int
        ready: float
        members: tuple[str, ...]

    def _pack_ddp_buckets(self, items: Sequence[_GradItem]) -> list[_Bucket]:
        """Approximate rebuilt DDP buckets from parameter sizes and ready order.

        Parameters larger than bucket_cap are indivisible standalone buckets.
        They must *not* force unrelated small bias parameters into separate tiny
        buckets; the H=4 unified Nsight trace shows eight large matrix buckets
        plus one coalesced small bucket.  Remaining sub-cap parameters are packed
        greedily in ready order.  For configurations where matrices themselves
        fit under bucket_cap this same rule naturally coalesces them.
        """
        ordered = sorted(items, key=lambda x: (x.ready, x.name))
        buckets: list[StepTimeSimulator._Bucket] = []
        small: list[StepTimeSimulator._GradItem] = []
        for item in ordered:
            if item.bytes >= self.ddp_bucket_cap_bytes:
                buckets.append(self._Bucket(
                    f"ddp_bucket{len(buckets)}", item.bytes, item.ready, (item.name,)
                ))
            else:
                small.append(item)

        pending: list[StepTimeSimulator._GradItem] = []
        pending_bytes = 0
        def flush_small() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            buckets.append(self._Bucket(
                f"ddp_bucket{len(buckets)}", pending_bytes,
                max(x.ready for x in pending), tuple(x.name for x in pending),
            ))
            pending = []; pending_bytes = 0

        for item in small:
            if pending and pending_bytes + item.bytes > self.ddp_bucket_cap_bytes:
                flush_small()
            pending.append(item); pending_bytes += item.bytes
        flush_small()
        return sorted(buckets, key=lambda x: (x.ready, x.name))

    def _backward_ddp(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        order = list(reversed(hooks)) if self.backward_hook_order == "reverse" else list(hooks)
        grad_items: list[StepTimeSimulator._GradItem] = []
        for h in order:
            ev0 = sched.compute("backward_pre_wdec_ready", h.hook.name, "backward", h.bwd0_ms)
            mat_bytes = h.hook.d_in * h.F_local * self.itemsize
            grad_items.append(self._GradItem(f"{h.hook.name}.W_dec", h.hook.name, mat_bytes, ev0.end_ms))
            # b_enc is produced in the encoder-local backward path before W_dec is finally ready.
            grad_items.append(self._GradItem(f"{h.hook.name}.b_enc", h.hook.name, h.F_local * self.itemsize, ev0.end_ms))
            ev1 = sched.compute("backward_wdec_to_wenc_ready", h.hook.name, "backward", h.bwd1_ms)
            grad_items.append(self._GradItem(f"{h.hook.name}.W_enc", h.hook.name, mat_bytes, ev1.end_ms))
            ev2 = sched.compute("backward_post_wenc", h.hook.name, "backward", h.bwd2_ms)
            grad_items.append(self._GradItem(f"{h.hook.name}.b_dec", h.hook.name, h.hook.d_in * self.itemsize, ev2.end_ms))

        compute_done = sched.compute_available
        buckets = self._pack_ddp_buckets(grad_items)
        # Buckets are issued when ready; one NCCL queue naturally creates a tail if comm is slower.
        for b in sorted(buckets, key=lambda x: (x.ready, x.name)):
            dur = self._nccl("allreduce", b.bytes, self.dp)
            sched.comm(b.name, b.members[0].split(".")[0] if len(b.members) == 1 else "multi",
                       "backward", dur, release=b.ready, payload_bytes=b.bytes,
                       collective="allreduce", note=";".join(b.members))
        backward_done = max(compute_done, sched.comm_available)
        sched.barrier(backward_done)
        for h in hooks:
            sched.compute("grad_clip", h.hook.name, "post_backward", h.clip_total_ms)

    def _forward_fsdp(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        next_ag_end: float | None = None
        for idx, h in enumerate(hooks):
            shard_bytes = self._fsdp_shard_bytes(h)
            if idx == 0:
                ag = sched.comm("fsdp_forward_allgather", h.hook.name, "forward", self._nccl("allgather", shard_bytes, self.dp),
                                release=sched.compute_available, payload_bytes=shard_bytes, collective="allgather")
                this_ag_end = ag.end_ms
            else:
                if next_ag_end is None:
                    ag = sched.comm("fsdp_forward_allgather", h.hook.name, "forward", self._nccl("allgather", shard_bytes, self.dp),
                                    release=sched.compute_available, payload_bytes=shard_bytes, collective="allgather")
                    this_ag_end = ag.end_ms
                else:
                    this_ag_end = next_ag_end
            fwd = sched.compute("sae_forward", h.hook.name, "forward", h.forward_ms, earliest=this_ag_end)
            next_ag_end = None
            if self.fsdp_forward_prefetch and idx + 1 < len(hooks):
                n = hooks[idx + 1]
                nbytes = self._fsdp_shard_bytes(n)
                ag = sched.comm("fsdp_forward_prefetch_allgather", n.hook.name, "forward_prefetch", self._nccl("allgather", nbytes, self.dp),
                                release=fwd.start_ms, payload_bytes=nbytes, collective="allgather",
                                note="prefetch_depth=1")
                next_ag_end = ag.end_ms
            elif idx + 1 < len(hooks):
                # No prefetch: next iteration issues AG after current FWD ends.
                sched.comm_available = max(sched.comm_available, fwd.end_ms)

    def _backward_fsdp_shard_grad(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        order = list(reversed(hooks)) if self.backward_hook_order == "reverse" else list(hooks)
        for h in order:
            bwd = sched.compute("sae_backward", h.hook.name, "backward", h.backward_ms)
            shard_bytes = self._fsdp_shard_bytes(h)
            sched.comm("fsdp_reduce_scatter", h.hook.name, "backward", self._nccl("reduce_scatter", shard_bytes, self.dp),
                       release=bwd.end_ms, payload_bytes=shard_bytes, collective="reduce_scatter")
            # No barrier: next hook backward may proceed and overlap this RS.
        sched.barrier(max(sched.compute_available, sched.comm_available))

    def _backward_fsdp_full(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        order = list(reversed(hooks)) if self.backward_hook_order == "reverse" else list(hooks)
        next_ag_end: float | None = None
        for idx, h in enumerate(order):
            shard_bytes = self._fsdp_shard_bytes(h)
            if idx == 0:
                ag = sched.comm("fsdp_backward_allgather", h.hook.name, "backward", self._nccl("allgather", shard_bytes, self.dp),
                                release=sched.compute_available, payload_bytes=shard_bytes, collective="allgather")
                this_ag_end = ag.end_ms
            else:
                if next_ag_end is None:
                    ag = sched.comm("fsdp_backward_allgather", h.hook.name, "backward", self._nccl("allgather", shard_bytes, self.dp),
                                    release=sched.compute_available, payload_bytes=shard_bytes, collective="allgather")
                    this_ag_end = ag.end_ms
                else:
                    this_ag_end = next_ag_end

            bwd = sched.compute("sae_backward", h.hook.name, "backward", h.backward_ms, earliest=this_ag_end)
            next_ag_end = None

            if self.fsdp_backward_prefetch == "backward_pre" and idx + 1 < len(order):
                nxt = order[idx + 1]
                nb = self._fsdp_shard_bytes(nxt)
                agn = sched.comm("fsdp_backward_prefetch_allgather", nxt.hook.name, "backward_prefetch", self._nccl("allgather", nb, self.dp),
                                 release=bwd.start_ms, payload_bytes=nb, collective="allgather", note="BACKWARD_PRE depth=1")
                next_ag_end = agn.end_ms
                # RS is queued behind the already-issued next AG. It may overlap next BWD.
                sched.comm("fsdp_reduce_scatter", h.hook.name, "backward", self._nccl("reduce_scatter", shard_bytes, self.dp),
                           release=bwd.end_ms, payload_bytes=shard_bytes, collective="reduce_scatter")
            elif self.fsdp_backward_prefetch == "backward_post" and idx + 1 < len(order):
                # Trace-faithful FSDP1 POST schedule: RS first, then next AG on the same comm queue.
                sched.comm("fsdp_reduce_scatter", h.hook.name, "backward", self._nccl("reduce_scatter", shard_bytes, self.dp),
                           release=bwd.end_ms, payload_bytes=shard_bytes, collective="reduce_scatter")
                nxt = order[idx + 1]
                nb = self._fsdp_shard_bytes(nxt)
                agn = sched.comm("fsdp_backward_post_allgather", nxt.hook.name, "backward_prefetch", self._nccl("allgather", nb, self.dp),
                                 release=bwd.end_ms, payload_bytes=nb, collective="allgather", note="BACKWARD_POST depth=1")
                next_ag_end = agn.end_ms
            else:
                # No useful prefetch (or final hook): complete RS before the next AG is issued.
                rs = sched.comm("fsdp_reduce_scatter", h.hook.name, "backward", self._nccl("reduce_scatter", shard_bytes, self.dp),
                                release=bwd.end_ms, payload_bytes=shard_bytes, collective="reduce_scatter")
                if idx + 1 < len(order):
                    sched.barrier(max(sched.compute_available, rs.end_ms))

        sched.barrier(max(sched.compute_available, sched.comm_available))

    def _post_backward_fsdp(self, sched: EventScheduler, hooks: Sequence[HookTimes]) -> None:
        for h in hooks:
            # If split FSDP-local clip rows are available, place the scalar DP AR between norm and scale.
            if h.clip_norm_ms > 0.0 and h.clip_scale_ms > 0.0:
                # Sharded gradients: local scan/scale work is approximately proportional to local numel.
                sched.compute("fsdp_grad_norm_scan", h.hook.name, "post_backward", h.clip_norm_ms / self.dp,
                              note="v1 local-shard scaling approximation")
                ar = sched.comm("fsdp_grad_norm_allreduce", h.hook.name, "post_backward", self._nccl("allreduce", 4, self.dp),
                                release=sched.compute_available, payload_bytes=4, collective="allreduce")
                sched.barrier(ar.end_ms)
                sched.compute("fsdp_grad_scale", h.hook.name, "post_backward", h.clip_scale_ms / self.dp,
                              note="v1 local-shard scaling approximation")
            else:
                self.warnings.append("FSDP clip split rows missing; using full clip/dp then scalar AllReduce approximation.")
                sched.compute("fsdp_grad_clip_local", h.hook.name, "post_backward", h.clip_total_ms / self.dp,
                              note="v1 local-shard scaling approximation")
                ar = sched.comm("fsdp_grad_norm_allreduce", h.hook.name, "post_backward", self._nccl("allreduce", 4, self.dp),
                                release=sched.compute_available, payload_bytes=4, collective="allreduce")
                sched.barrier(ar.end_ms)

    def simulate(self, hooks: Sequence[HookSpec], *, batch_size: int) -> SimulationResult:
        global_B = int(batch_size)
        if global_B <= 0:
            raise SimulationError("batch_size must be positive")
        if self.mode in {"ddp", "fsdp"}:
            if global_B % self.dp != 0:
                raise SimulationError(
                    f"Global batch {global_B} is not divisible by dp_size={self.dp}; "
                    "run_sae_runner_gpu.py uses an integer per-replica batch."
                )
            local_B = global_B // self.dp
        else:
            local_B = global_B
        self.global_B = global_B
        self.B = local_B

        hs = [self._make_hook_times(h) for h in hooks]
        sched = EventScheduler()

        if self.mode == "fsdp":
            self._forward_fsdp(sched, hs)
        else:
            self._forward_plain(sched, hs)

        self._stats_phase(sched, hs)

        if self.mode == "ddp":
            self._backward_ddp(sched, hs)
        elif self.mode == "fsdp":
            if self.fsdp_sharding_strategy == "shard_grad_op":
                self._backward_fsdp_shard_grad(sched, hs)
            else:
                self._backward_fsdp_full(sched, hs)
            self._post_backward_fsdp(sched, hs)
        else:
            self._backward_plain_or_tp(sched, hs)

        sched.barrier(max(sched.compute_available, sched.comm_available))
        for h in hs:
            opt_ms = h.optimizer_ms
            if self.mode == "fsdp":
                # FSDP optimizer state/params are local shards.  Until a dedicated
                # fused-shard profile exists, scale the full-SAE fused Adam work by dp.
                opt_ms /= self.dp
                note = "fused Adam: local-shard linear-numel approximation"
            else:
                note = "fused Adam"
            sched.compute("optimizer_step", h.hook.name, "optimizer", opt_ms, note=note)

        total = sched.wall
        # Global throughput: DDP/FSDP collectively consume global_B tokens per step.
        tokens_s = (global_B / (total / 1000.0)) if total > 0 else float("inf")
        compute_end = max((e.end_ms for e in sched.events if e.resource == "compute" and e.phase == "backward"), default=0.0)
        comm_end = max((e.end_ms for e in sched.events if e.resource == "comm" and e.phase in {"backward", "backward_prefetch"}), default=0.0)
        tail = max(0.0, comm_end - compute_end)
        return SimulationResult(
            parallel_mode=self.mode,
            total_ms=total,
            tokens_per_s=tokens_s,
            compute_work_ms=sched.compute_work,
            comm_work_ms=sched.comm_work,
            exposed_comm_tail_ms=tail,
            hooks=[{
                "name": h.hook.name, "d_in": h.hook.d_in, "d_sae": h.hook.d_sae, "k": h.hook.k,
                "F_local": h.F_local, "forward_ms": h.forward_ms, "backward_ms": h.backward_ms,
                "backward_segments_ms": [h.bwd0_ms, h.bwd1_ms, h.bwd2_ms],
                "clip_ms": h.clip_total_ms, "stats_local_ms": h.stats_local_ms, "optimizer_full_ms": h.optimizer_ms,
            } for h in hs],
            timeline=sorted(sched.events, key=lambda e: (e.start_ms, 0 if e.resource == "compute" else 1, e.end_ms)),
            warnings=list(dict.fromkeys(self.warnings)),
            metadata={
                "batch": global_B,
                "batch_semantics": "global train_batch_size_tokens; per-rank batch is derived internally",
                "tp": self.tp, "dp_size": self.dp, "dtype": self.dtype,
                "optimizer_impl": "fused",
                "fsdp_sharding_strategy": self.fsdp_sharding_strategy if self.mode == "fsdp" else None,
                "fsdp_forward_prefetch": self.fsdp_forward_prefetch if self.mode == "fsdp" else None,
                "fsdp_backward_prefetch": self.fsdp_backward_prefetch if self.mode == "fsdp" else None,
                "ddp_bucket_cap_mb": self.ddp_bucket_cap_bytes / (1024 * 1024) if self.mode == "ddp" else None,
                "overlap_contention": "ignored: compute/comm use isolated profile speed",
                "prefetch_depth": 1 if self.mode == "fsdp" else None,
                "predictor": "deterministic rectilinear compute/activation + piecewise-linear NCCL",
            },
        )


def expand(values: Sequence[int], n: int, name: str) -> list[int]:
    vals = list(values)
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"{name}: expected one value or {n} values, got {len(vals)}")
    return vals


def parse_hooks(args: argparse.Namespace) -> list[HookSpec]:
    if args.hook_spec:
        hooks: list[HookSpec] = []
        for text in args.hook_spec:
            parts = text.split(":")
            if len(parts) not in {3, 4}:
                raise ValueError(f"Invalid --hook-spec {text!r}; expected NAME:D_IN:D_SAE[:K]")
            name, d, f = parts[:3]
            k = int(parts[3]) if len(parts) == 4 else int(args.k[0])
            hooks.append(HookSpec(name, int(d), int(f), k))
        return hooks
    n = max(len(args.d_in), len(args.d_sae), len(args.k))
    if args.num_hooks is not None:
        if n == 1:
            n = args.num_hooks
        elif args.num_hooks != n:
            raise ValueError("--num-hooks conflicts with vector d-in/d-sae/k lengths")
    ds = expand(args.d_in, n, "d_in")
    fs = expand(args.d_sae, n, "d_sae")
    ks = expand(args.k, n, "k")
    return [HookSpec(f"h{i+1}", ds[i], fs[i], ks[i]) for i in range(n)]


def write_timeline_csv(path: Path, result: SimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["resource","name","hook","phase","start_ms","end_ms","duration_ms","payload_bytes","collective","note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for e in result.timeline:
            w.writerow(asdict(e))


def format_result(result: SimulationResult) -> str:
    lines = [
        f"mode={result.parallel_mode} total={result.total_ms:.4f} ms  tokens/s={result.tokens_per_s:.2f}",
        f"compute work={result.compute_work_ms:.4f} ms  comm work={result.comm_work_ms:.4f} ms  backward comm tail={result.exposed_comm_tail_ms:.4f} ms",
        "",
        "resource | phase             | hook | event                              | start ms | end ms | dur ms",
        "---------+-------------------+------+------------------------------------+----------+--------+-------",
    ]
    for e in result.timeline:
        lines.append(f"{e.resource:<8} | {e.phase:<17} | {e.hook:<4} | {e.name:<34} | {e.start_ms:8.3f} | {e.end_ms:6.3f} | {e.duration_ms:6.3f}")
    if result.warnings:
        lines.append("\nWarnings:")
        lines.extend(f"- {w}" for w in result.warnings)
    return "\n".join(lines)


def parse_bool(value: Any, default: bool | None = None) -> bool:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("Boolean value is empty")
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}")


def first_nonempty(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def parse_mode_token(token: str, tp: int, dp_size: int) -> tuple[str, int, int]:
    text = str(token).strip().lower()
    if text == "single":
        return "single", 1, 1
    if text == "tp":
        return "tp", tp, 1
    if text.startswith("tp") and text[2:].isdigit():
        return "tp", int(text[2:]), 1
    if text == "ddp":
        return "ddp", 1, dp_size
    if text.startswith("ddp") and text[3:].isdigit():
        return "ddp", 1, int(text[3:])
    if text == "fsdp":
        return "fsdp", 1, dp_size
    if text.startswith("fsdp") and text[4:].isdigit():
        return "fsdp", 1, int(text[4:])
    raise ValueError(f"Unsupported mode {token!r}; use single|tp|tp2|ddp|ddp2|fsdp|fsdp2 ...")


def parse_hook_specs_text(text: str, default_k: int) -> list[HookSpec]:
    hooks: list[HookSpec] = []
    for raw in [x.strip() for x in str(text).split(";") if x.strip()]:
        parts = raw.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(f"Invalid hook_specs entry {raw!r}; expected NAME:D_IN:D_SAE[:K]")
        name, d, f = parts[:3]
        k = int(parts[3]) if len(parts) == 4 else default_k
        hooks.append(HookSpec(name, int(d), int(f), k))
    if not hooks:
        raise ValueError("hook_specs was provided but no valid hook was parsed")
    return hooks


def add_prediction_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hook-spec", action="append", default=[], help="NAME:D_IN:D_SAE[:K], repeatable")
    p.add_argument("--d-in", nargs="+", type=int, default=[DEFAULT_D_IN])
    p.add_argument("--d-sae", nargs="+", type=int, default=[DEFAULT_D_SAE])
    p.add_argument("--k", nargs="+", type=int, default=[DEFAULT_K])
    p.add_argument("--num-hooks", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="GLOBAL train_batch_size_tokens (the only batch input).")
    p.add_argument("--parallel-mode", choices=["single","tp","ddp","fsdp"], default=DEFAULT_PARALLEL_MODE)
    p.add_argument("--tp", type=int, default=DEFAULT_TP)
    p.add_argument("--dp-size", type=int, default=DEFAULT_DP_SIZE)
    p.add_argument("--dtype", default=DEFAULT_DTYPE)
    p.add_argument("--activation-type", default=DEFAULT_ACTIVATION_TYPE)
    p.add_argument("--activation-output-layout", default=DEFAULT_ACTIVATION_OUTPUT_LAYOUT)
    # Kept only for config compatibility; no other optimizer is accepted.
    p.add_argument("--optimizer-impl", choices=["fused"], default="fused")
    p.add_argument("--stats-sync-mode", choices=["immediate","periodic","deferred"], default=DEFAULT_STATS_SYNC_MODE)
    p.add_argument("--normalize-activations", default=DEFAULT_NORMALIZE_ACTIVATIONS)
    p.add_argument("--fsdp-sharding-strategy", choices=["full_shard","shard_grad_op"], default=DEFAULT_FSDP_SHARDING_STRATEGY)
    p.add_argument("--fsdp-forward-prefetch", action=argparse.BooleanOptionalAction, default=DEFAULT_FSDP_FORWARD_PREFETCH)
    p.add_argument("--fsdp-backward-prefetch", choices=["none","backward_pre","backward_post"], default=DEFAULT_FSDP_BACKWARD_PREFETCH)
    p.add_argument("--ddp-bucket-cap-mb", type=float, default=DEFAULT_DDP_BUCKET_CAP_MB)
    p.add_argument("--backward-hook-order", choices=["reverse","forward"], default=DEFAULT_BACKWARD_HOOK_ORDER)


def add_profile_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--compute-csv", type=Path, default=DEFAULT_COMPUTE_CSV)
    p.add_argument("--activation-csv", type=Path, default=DEFAULT_ACTIVATION_CSV)
    p.add_argument("--nccl-csv", type=Path, default=DEFAULT_NCCL_CSV)
    p.add_argument("--time-column", default=DEFAULT_TIME_COLUMN)
    p.add_argument("--compute-device", default=None)
    p.add_argument("--activation-device", default=None)
    p.add_argument("--nccl-backend", default="nccl")
    p.add_argument("--nccl-algo", default=None)
    p.add_argument("--nccl-proto", default=None)
    p.add_argument("--nccl-p2p-level", default=None)
    p.add_argument("--compute-extrapolation", choices=["error","linear"], default=DEFAULT_COMPUTE_EXTRAPOLATION)
    p.add_argument("--activation-extrapolation", choices=["error","linear"], default=DEFAULT_ACTIVATION_EXTRAPOLATION)
    p.add_argument("--nccl-extrapolation", choices=["error","linear"], default=DEFAULT_NCCL_EXTRAPOLATION)
    p.add_argument("--max-extrapolation-factor", type=float, default=DEFAULT_MAX_EXTRAPOLATION_FACTOR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "One-step deterministic SAE step-time simulator: read profile CSVs, "
            "interpolate natural operation families, and build the distributed DAG."
        )
    )
    add_profile_arguments(p)
    add_prediction_arguments(p)
    p.add_argument("--config-csv", type=Path, default=None,
                   help="Optional batch prediction CSV. Column 'batch' is always GLOBAL batch.")
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--summary-jsonl", type=Path, default=None)
    p.add_argument("--timeline-dir", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None, help="Single-query result JSON")
    p.add_argument("--timeline-csv", type=Path, default=None, help="Single-query timeline CSV")
    return p


def build_simulator_from_values(
    profiles: Profiles,
    *,
    parallel_mode: str,
    tp: int,
    dp_size: int,
    dtype: str,
    activation_type: str,
    activation_output_layout: str,
    optimizer_impl: str,
    stats_sync_mode: str,
    normalize_activations: str,
    fsdp_sharding_strategy: str,
    fsdp_forward_prefetch: bool,
    fsdp_backward_prefetch: str,
    ddp_bucket_cap_mb: float,
    backward_hook_order: str,
) -> StepTimeSimulator:
    return StepTimeSimulator(
        profiles,
        parallel_mode=parallel_mode,
        tp=tp,
        dp_size=dp_size,
        dtype=dtype,
        activation_type=activation_type,
        activation_output_layout=activation_output_layout,
        optimizer_impl=optimizer_impl,
        stats_sync_mode=stats_sync_mode,
        normalize_activations=normalize_activations,
        fsdp_sharding_strategy=fsdp_sharding_strategy,
        fsdp_forward_prefetch=fsdp_forward_prefetch,
        fsdp_backward_prefetch=fsdp_backward_prefetch,
        ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        backward_hook_order=backward_hook_order,
    )



def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    hooks = parse_hooks(args)
    parallel_mode, tp, dp_size = parse_mode_token(args.parallel_mode, int(args.tp), int(args.dp_size))
    return {
        "run_id": "single_query",
        "hooks": hooks,
        "batch_size": int(args.batch_size),
        "parallel_mode": parallel_mode,
        "tp": tp,
        "dp_size": dp_size,
        "dtype": args.dtype,
        "activation_type": args.activation_type,
        "activation_output_layout": args.activation_output_layout,
        "optimizer_impl": "fused",
        "stats_sync_mode": args.stats_sync_mode,
        "normalize_activations": args.normalize_activations,
        "fsdp_sharding_strategy": args.fsdp_sharding_strategy,
        "fsdp_forward_prefetch": bool(args.fsdp_forward_prefetch),
        "fsdp_backward_prefetch": args.fsdp_backward_prefetch,
        "ddp_bucket_cap_mb": float(args.ddp_bucket_cap_mb),
        "backward_hook_order": args.backward_hook_order,
    }


def config_from_csv_row(row: Mapping[str, str], args: argparse.Namespace, index: int) -> dict[str, Any]:
    default_d = int(args.d_in[0])
    default_f = int(args.d_sae[0])
    default_k = int(args.k[0])
    batch_size = int(first_nonempty(row, "batch", default=args.batch_size))
    tp = int(first_nonempty(row, "tp", default=args.tp))
    dp_size = int(first_nonempty(row, "dp_size", "dp", default=args.dp_size))
    mode_token = str(first_nonempty(row, "mode", "parallel_mode", default=args.parallel_mode))
    parallel_mode, tp, dp_size = parse_mode_token(mode_token, tp, dp_size)

    hook_specs_text = first_nonempty(row, "hook_specs", "hook_spec", default="")
    if hook_specs_text:
        hooks = parse_hook_specs_text(str(hook_specs_text), int(first_nonempty(row, "k", default=default_k)))
    else:
        H = int(first_nonempty(row, "H", "h", "num_hooks", default=args.num_hooks or 1))
        d_in = int(first_nonempty(row, "d_in", "D", default=default_d))
        d_sae = int(first_nonempty(row, "d_sae", "F", default=default_f))
        k = int(first_nonempty(row, "k", default=default_k))
        hooks = [HookSpec(f"h{i+1}", d_in, d_sae, k) for i in range(H)]

    optimizer = str(first_nonempty(row, "optimizer_impl", "optimizer", default="fused")).strip().lower()
    if optimizer != "fused":
        raise ValueError(f"Only fused optimizer is supported; row requested {optimizer!r}")
    return {
        "run_id": str(first_nonempty(row, "run_id", "id", default=f"case_{index:04d}")),
        "hooks": hooks,
        "batch_size": batch_size,
        "parallel_mode": parallel_mode,
        "tp": tp,
        "dp_size": dp_size,
        "dtype": str(first_nonempty(row, "dtype", default=args.dtype)),
        "activation_type": str(first_nonempty(row, "activation_type", default=args.activation_type)),
        "activation_output_layout": str(first_nonempty(row, "activation_output_layout", "output_layout", default=args.activation_output_layout)),
        "optimizer_impl": "fused",
        "stats_sync_mode": str(first_nonempty(row, "stats_sync_mode", default=args.stats_sync_mode)),
        "normalize_activations": str(first_nonempty(row, "normalize_activations", default=args.normalize_activations)),
        "fsdp_sharding_strategy": str(first_nonempty(row, "fsdp_sharding_strategy", "sharding_strategy", default=args.fsdp_sharding_strategy)),
        "fsdp_forward_prefetch": parse_bool(first_nonempty(row, "fsdp_forward_prefetch", default=args.fsdp_forward_prefetch), args.fsdp_forward_prefetch),
        "fsdp_backward_prefetch": str(first_nonempty(row, "fsdp_backward_prefetch", default=args.fsdp_backward_prefetch)),
        "ddp_bucket_cap_mb": float(first_nonempty(row, "ddp_bucket_cap_mb", default=args.ddp_bucket_cap_mb)),
        "backward_hook_order": str(first_nonempty(row, "backward_hook_order", default=args.backward_hook_order)),
    }


def summarize_case(config: Mapping[str, Any], result: SimulationResult | None, error: str = "") -> dict[str, Any]:
    hooks: Sequence[HookSpec] = config["hooks"]
    base = {
        "run_id": config["run_id"],
        "status": "ok" if result is not None else "error",
        "error": error,
        "mode": config["parallel_mode"],
        "batch": config["batch_size"],
        "H": len(hooks),
        "d_in": ";".join(str(h.d_in) for h in hooks),
        "d_sae": ";".join(str(h.d_sae) for h in hooks),
        "k": ";".join(str(h.k) for h in hooks),
        "tp": config["tp"],
        "dp_size": config["dp_size"],
        "optimizer_impl": "fused",
        "fsdp_sharding_strategy": config["fsdp_sharding_strategy"],
        "fsdp_forward_prefetch": config["fsdp_forward_prefetch"],
        "fsdp_backward_prefetch": config["fsdp_backward_prefetch"],
    }
    if result is None:
        base.update({"sim_ms": "", "tokens_per_s": "", "compute_work_ms": "", "comm_work_ms": "", "exposed_comm_tail_ms": "", "warnings": ""})
    else:
        base.update({
            "sim_ms": result.total_ms,
            "tokens_per_s": result.tokens_per_s,
            "compute_work_ms": result.compute_work_ms,
            "comm_work_ms": result.comm_work_ms,
            "exposed_comm_tail_ms": result.exposed_comm_tail_ms,
            "warnings": " | ".join(result.warnings),
        })
    return base


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id","status","error","mode","batch","H","d_in","d_sae","k","tp","dp_size",
        "optimizer_impl","fsdp_sharding_strategy","fsdp_forward_prefetch","fsdp_backward_prefetch",
        "sim_ms","tokens_per_s","compute_work_ms","comm_work_ms","exposed_comm_tail_ms","warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def run_one_config(profiles: Profiles, config: Mapping[str, Any]) -> SimulationResult:
    sim = build_simulator_from_values(
        profiles,
        parallel_mode=str(config["parallel_mode"]),
        tp=int(config["tp"]),
        dp_size=int(config["dp_size"]),
        dtype=str(config["dtype"]),
        activation_type=str(config["activation_type"]),
        activation_output_layout=str(config["activation_output_layout"]),
        optimizer_impl="fused",
        stats_sync_mode=str(config["stats_sync_mode"]),
        normalize_activations=str(config["normalize_activations"]),
        fsdp_sharding_strategy=str(config["fsdp_sharding_strategy"]),
        fsdp_forward_prefetch=bool(config["fsdp_forward_prefetch"]),
        fsdp_backward_prefetch=str(config["fsdp_backward_prefetch"]),
        ddp_bucket_cap_mb=float(config["ddp_bucket_cap_mb"]),
        backward_hook_order=str(config["backward_hook_order"]),
    )
    return sim.simulate(list(config["hooks"]), batch_size=int(config["batch_size"]))


def load_profiles_from_args(args: argparse.Namespace) -> Profiles:
    opts = InterpolationOptions(
        compute_extrapolation=args.compute_extrapolation,
        activation_extrapolation=args.activation_extrapolation,
        nccl_extrapolation=args.nccl_extrapolation,
        max_extrapolation_factor=float(args.max_extrapolation_factor),
    )
    return Profiles.load(
        args.compute_csv, args.activation_csv, args.nccl_csv,
        time_column=args.time_column,
        options=opts,
        compute_device=args.compute_device,
        activation_device=args.activation_device,
        nccl_backend=args.nccl_backend,
        nccl_algo=args.nccl_algo,
        nccl_proto=args.nccl_proto,
        nccl_p2p_level=args.nccl_p2p_level,
    )


def command_run(args: argparse.Namespace) -> int:
    profiles = load_profiles_from_args(args)
    print(f"[simulate] compute={args.compute_csv}")
    print(f"[simulate] activation={args.activation_csv}")
    print(f"[simulate] nccl={args.nccl_csv}")
    print("[simulate] predictor=deterministic rectilinear compute/activation + piecewise-linear NCCL; optimizer=fused")

    if args.config_csv is None:
        config = config_from_args(args)
        try:
            result = run_one_config(profiles, config)
        except (SimulationError, ValueError, OSError) as exc:
            print(f"[simulate] ERROR: {exc}", file=sys.stderr)
            return 2
        print(format_result(result))
        print(f"batch={result.metadata.get('batch')}")
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"JSON: {args.output_json}")
        if args.timeline_csv:
            write_timeline_csv(args.timeline_csv, result)
            print(f"Timeline CSV: {args.timeline_csv}")
        if args.summary_csv:
            write_rows_csv(args.summary_csv, [summarize_case(config, result)])
            print(f"Summary CSV: {args.summary_csv}")
        return 0

    if not args.config_csv.exists():
        print(f"[simulate] ERROR: config CSV does not exist: {args.config_csv}", file=sys.stderr)
        return 2
    with args.config_csv.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    summaries: list[dict[str, Any]] = []
    jsonl_records: list[dict[str, Any]] = []
    ok = 0
    for i, row in enumerate(raw_rows, 1):
        try:
            config = config_from_csv_row(row, args, i)
            result = run_one_config(profiles, config)
            summaries.append(summarize_case(config, result))
            jsonl_records.append({
                "config": {**{k:v for k,v in config.items() if k != "hooks"}, "hooks": [asdict(h) for h in config["hooks"]]},
                "result": asdict(result),
            })
            if args.timeline_dir:
                safe_id = str(config["run_id"]).replace("/", "_").replace("\\", "_")
                write_timeline_csv(args.timeline_dir / f"{safe_id}.timeline.csv", result)
            print(f"[{i}/{len(raw_rows)}] {config['run_id']}: {result.total_ms:.4f} ms  batch={config['batch_size']}")
            ok += 1
        except (SimulationError, ValueError, OSError) as exc:
            try:
                config = config_from_csv_row(row, args, i)
                summaries.append(summarize_case(config, None, str(exc)))
            except Exception:
                summaries.append({
                    "run_id": first_nonempty(row, "run_id", "id", default=f"case_{i:04d}"),
                    "status": "error",
                    "error": str(exc),
                })
            jsonl_records.append({"row": dict(row), "status": "error", "error": str(exc)})
            print(f"[{i}/{len(raw_rows)}] ERROR: {exc}", file=sys.stderr)

    summary_path = args.summary_csv or args.config_csv.with_name(args.config_csv.stem + ".predictions.csv")
    write_rows_csv(summary_path, summaries)
    print(f"[simulate] summary={summary_path} ok={ok}/{len(raw_rows)}")
    if args.summary_jsonl:
        args.summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_jsonl.open("w", encoding="utf-8") as f:
            for record in jsonl_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[simulate] jsonl={args.summary_jsonl}")
    return 0 if ok == len(raw_rows) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return command_run(args)
    except (SimulationError, ValueError, OSError) as exc:
        print(f"[simulate-step-time] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
