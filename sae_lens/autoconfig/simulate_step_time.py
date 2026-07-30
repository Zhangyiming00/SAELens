#!/usr/bin/env python3
"""Profile-driven SAE step-time simulator for the fixed profiler CSV schemas.

This script consumes the three CSV files produced by:

* external_gemm_optim_profiler.py
* external_topk_profiler.py / activation-compute CSV
* external_nccl_profiler.py

It predicts one offline SAE training step on one TP rank by summing:

* six GEMMs per hook;
* preprocess(B,D) plus aggregated local shape-family groups per hook;
* one replaceable activation forward+backward boundary per hook;
* the TP collective call graph per hook;
* one per-SAE Adam estimate per hook, summed across heterogeneous hooks;
* an optional amortized/deferred statistics tail.

The result is an additive GPU-operation estimate. It does not model overlap,
rank-arrival skew, CPU launch gaps, data loading, checkpointing, or DP/FSDP.

Prediction policy
-----------------
Discrete implementation choices are matched exactly: dtype, TP, optimizer
implementation, statistics mode, NCCL backend/topology and (when requested)
profile device. Continuous shape variables are exact-matched first, then
interpolated from nearby measured points using inverse-distance weighting in
log-shape space. Extrapolation is rejected by default rather than silently
clamping to the profile boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
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
DEFAULT_BATCH_SIZE = 2048
DEFAULT_TP = 1
DEFAULT_K = 128
DEFAULT_ACTIVATION_TYPE = "topk"
DEFAULT_ACTIVATION_OUTPUT_LAYOUT = "dense"
DEFAULT_DTYPE = "float32"
DEFAULT_NORMALIZE_ACTIVATIONS = "none"
DEFAULT_OPTIMIZER_IMPL = "foreach"
DEFAULT_STATS_SYNC_MODE = "immediate"
DEFAULT_STATS_SYNC_INTERVAL = 1
DEFAULT_OPTIMIZER_LR = 1.0e-3
DEFAULT_OPTIMIZER_BETA1 = 0.9
DEFAULT_OPTIMIZER_BETA2 = 0.999
DEFAULT_OPTIMIZER_EPS = 1.0e-8
DEFAULT_OPTIMIZER_WEIGHT_DECAY = 0.0
DEFAULT_OPTIMIZER_AMSGRAD = False
DEFAULT_OPTIMIZER_MAXIMIZE = False
DEFAULT_OPTIMIZER_CAPTURABLE = False

DEFAULT_TIME_COLUMN = "median_ms"
DEFAULT_NEIGHBORS = 8
DEFAULT_IDW_POWER = 2.0
DEFAULT_EXTRAPOLATION = "error"
# =============================================================================
# END USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================


GEMM_OPS: tuple[str, ...] = (
    "encoder_forward",
    "decoder_forward",
    "decoder_dgrad",
    "decoder_wgrad",
    "encoder_dgrad",
    "encoder_wgrad",
)

PREPROCESS_PER_HOOK_OPS: tuple[str, ...] = (
    "preprocess_bd",
)

LOCAL_PER_HOOK_OPS: tuple[str, ...] = (
    "local_compute_bd",
    "local_compute_bf_local",
    "local_compute_fd_local",
    "local_compute_bf_global",
)

LOCAL_TP_PER_HOOK_OPS: tuple[str, ...] = (
    "local_compute_bf_tp",
    "local_compute_bd_tp",
)

ACTIVATION_OP = "activation_forward_backward"
STATS_TAIL_OP = "local_compute_hf_global_tail"


ITEMSIZE_BYTES: dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
}

NORMALIZE_ACTIVATION_MODES: tuple[str, ...] = (
    "none",
    "expected_average_only_in",
    "constant_norm_rescale",
    "layer_norm",
)


def canonical_dtype(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "float32": "float32", "fp32": "float32", "float": "float32",
        "float16": "float16", "fp16": "float16", "half": "float16",
        "bfloat16": "bfloat16", "bf16": "bfloat16",
    }
    if key not in aliases:
        raise SimulationError(
            f"Unsupported dtype {name!r}; choices={sorted(aliases)}"
        )
    return aliases[key]


def canonical_normalize_activations(name: str) -> str:
    key = name.strip().lower()
    if key not in NORMALIZE_ACTIVATION_MODES:
        raise SimulationError(
            f"Unsupported normalize_activations {name!r}; "
            f"choices={list(NORMALIZE_ACTIVATION_MODES)}"
        )
    return key


class SimulationError(RuntimeError):
    """Raised when the requested configuration is unsupported by the profiles."""


@dataclass(frozen=True)
class HookConfig:
    """One SAE hook simulated within the shared training step."""

    name: str
    d_in: int
    d_sae: int
    k: int

    def f_local(self, tp: int) -> int:
        if self.d_sae % tp != 0:
            raise SimulationError(
                f"hook {self.name!r}: d_sae={self.d_sae} must be divisible by tp={tp}"
            )
        return self.d_sae // tp

    def validate(self, *, tp: int, activation_type: str) -> None:
        if not self.name.strip():
            raise SimulationError("hook name must not be empty")
        if self.d_in <= 0 or self.d_sae <= 0:
            raise SimulationError(
                f"hook {self.name!r}: d_in and d_sae must be positive"
            )
        _ = self.f_local(tp)
        if activation_type == "topk" and not (0 < self.k <= self.d_sae):
            raise SimulationError(
                f"hook {self.name!r}: TopK k must satisfy 0 < k <= d_sae; "
                f"got k={self.k}, d_sae={self.d_sae}"
            )


@dataclass(frozen=True)
class StepConfig:
    hooks: tuple[HookConfig, ...]
    batch_size: int
    tp: int
    activation_type: str
    activation_output_layout: str
    dtype: str
    normalize_activations: str
    optimizer_impl: str
    optimizer_lr: float
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_eps: float
    optimizer_weight_decay: float
    optimizer_amsgrad: bool
    optimizer_maximize: bool
    optimizer_capturable: bool
    stats_sync_mode: str
    stats_sync_interval: int
    deferred_tail_mode: str
    deferred_flush_steps: int | None

    @property
    def num_hooks(self) -> int:
        return len(self.hooks)

    @property
    def itemsize(self) -> int:
        return ITEMSIZE_BYTES[canonical_dtype(self.dtype)]

    def validate(self) -> None:
        if not self.hooks:
            raise SimulationError("At least one hook must be configured")
        if self.batch_size <= 0 or self.tp <= 0 or self.stats_sync_interval <= 0:
            raise SimulationError("batch_size, tp and stats_sync_interval must be positive")
        names = [hook.name for hook in self.hooks]
        if len(set(names)) != len(names):
            raise SimulationError(f"hook names must be unique; got {names}")
        if self.activation_type not in {"topk", "relu"}:
            raise SimulationError("activation_type must be topk or relu")
        if self.activation_output_layout != "dense":
            raise SimulationError("Only dense activation_output_layout is currently profiled")
        if self.normalize_activations not in NORMALIZE_ACTIVATION_MODES:
            raise SimulationError(
                "normalize_activations must be one of "
                f"{list(NORMALIZE_ACTIVATION_MODES)}"
            )
        for hook in self.hooks:
            hook.validate(tp=self.tp, activation_type=self.activation_type)
        if self.optimizer_lr < 0 or self.optimizer_eps < 0 or self.optimizer_weight_decay < 0:
            raise SimulationError("optimizer lr/eps/weight_decay must be non-negative")
        if not (0.0 <= self.optimizer_beta1 < 1.0 and 0.0 <= self.optimizer_beta2 < 1.0):
            raise SimulationError("optimizer betas must satisfy 0 <= beta < 1")
        if self.stats_sync_mode not in {"immediate", "periodic", "deferred"}:
            raise SimulationError("stats_sync_mode must be immediate, periodic, or deferred")
        if self.stats_sync_mode != "immediate" and len({hook.d_sae for hook in self.hooks}) > 1:
            raise SimulationError(
                "Current MultiSAETrainer periodic/deferred stats tail stacks hook "
                "feature vectors, so heterogeneous d_sae hooks require "
                "--stats-sync-mode immediate"
            )
        if self.deferred_tail_mode not in {"exclude", "event", "amortize"}:
            raise SimulationError("deferred_tail_mode must be exclude, event, or amortize")
        if self.deferred_tail_mode == "amortize":
            if not self.deferred_flush_steps or self.deferred_flush_steps <= 0:
                raise SimulationError(
                    "--deferred-flush-steps is required and must be > 0 when "
                    "--deferred-tail-mode=amortize"
                )


@dataclass
class Estimate:
    label: str
    component: str
    semantic_op: str
    per_call_ms: float
    count: float
    total_ms: float
    source: str
    exact: bool
    extrapolated: bool
    target: dict[str, Any]
    phase: str = ""
    shape_family: str = ""
    neighbor_points: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hook_name: str = ""


@dataclass
class SimulationResult:
    config: dict[str, Any]
    components_ms: dict[str, float]
    total_ms: float
    estimates: list[Estimate]
    warnings: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InterpolationOptions:
    neighbors: int = 8
    power: float = 2.0
    extrapolation: str = "error"  # error | clamp | allow


class CsvProfiles:
    def __init__(
        self,
        compute_csv: Path,
        activation_csv: Path,
        nccl_csv: Path | None,
        *,
        time_column: str,
        prefer_clean_activation: bool,
    ) -> None:
        self.compute_csv = compute_csv
        self.activation_csv = activation_csv
        self.nccl_csv = nccl_csv
        self.time_column = time_column
        self.prefer_clean_activation = prefer_clean_activation
        self.compute_rows = self._read_ok_rows(compute_csv)
        self.activation_rows = self._read_ok_rows(activation_csv)
        self.nccl_rows = self._read_ok_rows(nccl_csv) if nccl_csv else []

    @staticmethod
    def _read_ok_rows(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            raise SimulationError(f"Profile CSV does not exist: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        ok = [row for row in rows if str(row.get("status", "")).lower() == "ok"]
        if not ok:
            raise SimulationError(f"No successful profile rows found in {path}")
        return ok

    def time_value(self, row: Mapping[str, str], *, activation: bool = False) -> float:
        candidates: list[str] = []
        if activation and self.prefer_clean_activation:
            candidates.append("clean_ms")
        candidates.append(self.time_column)
        if self.time_column != "median_ms":
            candidates.append("median_ms")
        for column in candidates:
            value = _optional_float(row.get(column))
            if value is not None and value > 0:
                return value
        raise SimulationError(
            f"Row {row.get('case_id', '<unknown>')} has no positive timing in "
            f"columns {candidates}"
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    rounded = int(round(number))
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-9):
        return None
    return rounded


def _bool_text(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "true"
    if text in {"0", "false", "no", "off"}:
        return "false"
    return text


def _median(values: Sequence[float]) -> float:
    if not values:
        raise SimulationError("Cannot compute median of an empty sequence")
    return float(statistics.median(values))


def _match_exact(row: Mapping[str, str], filters: Mapping[str, Any]) -> bool:
    for field_name, expected in filters.items():
        if expected is None:
            continue
        raw = row.get(field_name, "")
        if field_name == "normalize_activations" and str(raw).strip() == "":
            raw = "none"
        if isinstance(expected, bool):
            if _bool_text(raw) != ("true" if expected else "false"):
                return False
        elif isinstance(expected, int):
            if _optional_int(raw) != expected:
                return False
        elif isinstance(expected, float):
            parsed = _optional_float(raw)
            if parsed is None or not math.isclose(parsed, expected):
                return False
        else:
            if str(raw).strip().lower() != str(expected).strip().lower():
                return False
    return True


def _choose_unique_value(
    rows: Sequence[Mapping[str, str]],
    field_name: str,
    requested: str | None,
    *,
    label: str,
    prefer: str | None = None,
) -> str | None:
    values = sorted(
        {
            str(row.get(field_name, "")).strip()
            for row in rows
            if str(row.get(field_name, "")).strip()
        }
    )
    if requested is not None:
        wanted = requested.strip().lower()
        matches = [value for value in values if value.lower() == wanted]
        if not matches:
            raise SimulationError(
                f"{label}: requested {field_name}={requested!r}, available={values}"
            )
        return matches[0]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if prefer is not None:
        for value in values:
            if value.lower() == prefer.lower():
                return value
    raise SimulationError(
        f"{label}: multiple {field_name} values are present {values}; select one explicitly"
    )


def _profile_device_filter(
    rows: Sequence[Mapping[str, str]],
    requested: str | None,
    *,
    label: str,
) -> tuple[list[Mapping[str, str]], str | None]:
    chosen = _choose_unique_value(rows, "device", requested, label=label)
    if chosen is None:
        return list(rows), None
    return (
        [row for row in rows if str(row.get("device", "")).strip() == chosen],
        chosen,
    )


def _predict_idw(
    rows: Sequence[Mapping[str, str]],
    *,
    label: str,
    source_path: Path,
    target: Mapping[str, float],
    numeric_fields: Sequence[str],
    timing: callable,
    options: InterpolationOptions,
) -> tuple[float, bool, bool, list[dict[str, Any]], list[str]]:
    """Predict one semantic operation from measured rows.

    Duplicate rows at the same numeric coordinate are aggregated by median. The
    interpolated value is a weighted geometric mean of nearby measured times.
    """
    notes: list[str] = []
    grouped: dict[tuple[float, ...], list[float]] = {}
    for row in rows:
        coords: list[float] = []
        valid = True
        for field_name in numeric_fields:
            value = _optional_float(row.get(field_name))
            if value is None or value <= 0:
                valid = False
                break
            coords.append(value)
        if not valid:
            continue
        grouped.setdefault(tuple(coords), []).append(float(timing(row)))

    if not grouped:
        raise SimulationError(
            f"{label}: no usable measured rows after filtering in {source_path}"
        )

    points = [(coord, _median(times), len(times)) for coord, times in grouped.items()]
    target_tuple = tuple(float(target[field_name]) for field_name in numeric_fields)

    for coord, measured_ms, duplicate_count in points:
        if all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(coord, target_tuple)):
            neighbor = {
                "coordinates": dict(zip(numeric_fields, coord)),
                "measured_ms": measured_ms,
                "duplicate_rows": duplicate_count,
                "distance": 0.0,
                "weight": 1.0,
            }
            return measured_ms, True, False, [neighbor], notes

    if not numeric_fields:
        values = [time_ms for _, time_ms, _ in points]
        notes.append("No numeric interpolation dimensions; used median of matching rows")
        return _median(values), False, False, [], notes

    mins = [min(coord[i] for coord, _, _ in points) for i in range(len(numeric_fields))]
    maxs = [max(coord[i] for coord, _, _ in points) for i in range(len(numeric_fields))]
    adjusted_target = list(target_tuple)
    outside: list[str] = []
    for i, field_name in enumerate(numeric_fields):
        if target_tuple[i] < mins[i] or target_tuple[i] > maxs[i]:
            outside.append(
                f"{field_name}={target_tuple[i]:g} outside [{mins[i]:g}, {maxs[i]:g}]"
            )
            if options.extrapolation == "clamp":
                adjusted_target[i] = min(max(target_tuple[i], mins[i]), maxs[i])
    extrapolated = bool(outside)
    if outside and options.extrapolation == "error":
        raise SimulationError(f"{label}: " + "; ".join(outside))
    if outside:
        notes.append(
            ("Clamped target to measured boundary: " if options.extrapolation == "clamp" else "Extrapolated outside measured range: ")
            + "; ".join(outside)
        )

    log_points = [[math.log2(value) for value in coord] for coord, _, _ in points]
    log_target = [math.log2(value) for value in adjusted_target]
    log_mins = [min(point[i] for point in log_points) for i in range(len(numeric_fields))]
    log_maxs = [max(point[i] for point in log_points) for i in range(len(numeric_fields))]

    def distance(point: Sequence[float]) -> float:
        squared = 0.0
        for i, value in enumerate(point):
            span = log_maxs[i] - log_mins[i]
            delta = value - log_target[i]
            if span > 0:
                delta /= span
            else:
                delta = 0.0
            squared += delta * delta
        return math.sqrt(squared)

    ranked: list[tuple[float, tuple[float, ...], float, int]] = []
    for (coord, measured_ms, duplicate_count), log_coord in zip(points, log_points):
        ranked.append((distance(log_coord), coord, measured_ms, duplicate_count))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[: max(1, min(options.neighbors, len(ranked)))]

    weights: list[float] = []
    for dist, _, _, _ in selected:
        weights.append(1.0 / max(dist, 1e-12) ** options.power)
    weight_sum = sum(weights)
    predicted_log = sum(
        weight * math.log(measured_ms)
        for weight, (_, _, measured_ms, _) in zip(weights, selected)
    ) / weight_sum
    predicted = math.exp(predicted_log)

    neighbors: list[dict[str, Any]] = []
    for weight, (dist, coord, measured_ms, duplicate_count) in zip(weights, selected):
        neighbors.append(
            {
                "coordinates": dict(zip(numeric_fields, coord)),
                "measured_ms": measured_ms,
                "duplicate_rows": duplicate_count,
                "distance": dist,
                "weight": weight / weight_sum,
            }
        )
    return predicted, False, extrapolated, neighbors, notes


class StepTimeSimulator:
    def __init__(
        self,
        profiles: CsvProfiles,
        interpolation: InterpolationOptions,
        *,
        compute_device: str | None,
        activation_device: str | None,
        activation_distributed: str,
        nccl_backend: str,
        nccl_algo: str,
        nccl_proto: str,
        nccl_p2p_level: str,
    ) -> None:
        self.profiles = profiles
        self.interpolation = interpolation
        self.compute_device = compute_device
        self.activation_device = activation_device
        self.activation_distributed = activation_distributed
        self.nccl_backend = nccl_backend
        self.nccl_algo = nccl_algo
        self.nccl_proto = nccl_proto
        self.nccl_p2p_level = nccl_p2p_level
        self.warnings: list[str] = []

    def simulate(self, config: StepConfig) -> SimulationResult:
        config.validate()
        self._validate_local_profile_schema(config)
        if config.num_hooks > 1:
            self.warnings.append(
                "Multi-hook optimizer time is the sum of per-SAE H-free profiles; "
                "cross-hook foreach/fused batching effects are intentionally not modeled"
            )
        estimates: list[Estimate] = []

        for hook in config.hooks:
            for op in GEMM_OPS:
                estimates.append(self._estimate_gemm(config, hook, op))

            for op in PREPROCESS_PER_HOOK_OPS:
                estimates.append(self._estimate_preprocess(config, hook, op))

            for op in LOCAL_PER_HOOK_OPS:
                estimates.append(self._estimate_local(config, hook, op))
            if config.tp > 1:
                for op in LOCAL_TP_PER_HOOK_OPS:
                    estimates.append(self._estimate_local(config, hook, op))

            stats_tail = self._estimate_stats_tail(config, hook)
            if stats_tail is not None:
                estimates.append(stats_tail)

            estimates.append(self._estimate_activation(config, hook))

            if config.tp > 1:
                estimates.extend(self._estimate_tp_collectives(config, hook))

            # The profiler intentionally measures one SAE at a time and has no
            # H dimension.  A heterogeneous multi-hook step is therefore the
            # sum of one per-hook optimizer estimate for each parameter set.
            estimates.append(self._estimate_optimizer(config, hook))

        component_totals: dict[str, float] = {}
        hook_totals: dict[str, float] = {hook.name: 0.0 for hook in config.hooks}
        hook_components: dict[str, dict[str, float]] = {
            hook.name: {} for hook in config.hooks
        }
        for estimate in estimates:
            component_totals[estimate.component] = (
                component_totals.get(estimate.component, 0.0) + estimate.total_ms
            )
            if estimate.hook_name:
                hook_totals[estimate.hook_name] += estimate.total_ms
                per_hook = hook_components[estimate.hook_name]
                per_hook[estimate.component] = (
                    per_hook.get(estimate.component, 0.0) + estimate.total_ms
                )
        total_ms = sum(component_totals.values())

        hook_configs = []
        for hook in config.hooks:
            item = asdict(hook)
            item["f_local"] = hook.f_local(config.tp)
            hook_configs.append(item)

        return SimulationResult(
            config={
                **{k: v for k, v in asdict(config).items() if k != "hooks"},
                "hooks": hook_configs,
                "num_hooks": config.num_hooks,
                "itemsize_bytes": config.itemsize,
            },
            components_ms=component_totals,
            total_ms=total_ms,
            estimates=estimates,
            warnings=list(dict.fromkeys(self.warnings)),
            metadata={
                "compute_csv": str(self.profiles.compute_csv),
                "activation_csv": str(self.profiles.activation_csv),
                "nccl_csv": str(self.profiles.nccl_csv) if self.profiles.nccl_csv else None,
                "time_column": self.profiles.time_column,
                "prefer_clean_activation": self.profiles.prefer_clean_activation,
                "interpolation": asdict(self.interpolation),
                "hook_totals_ms": hook_totals,
                "hook_components_ms": hook_components,
                "optimizer_model": (
                    "per-SAE additive: profiler has no H dimension; heterogeneous "
                    "hooks are summed independently"
                ),
                "scope": "additive per-rank GPU-operation time; TP only; no overlap/CPU/data/checkpoint",
            },
        )

    def _validate_local_profile_schema(self, cfg: StepConfig) -> None:
        preprocess_available = {
            str(row.get("semantic_op", "")).strip()
            for row in self.profiles.compute_rows
            if str(row.get("profiler", "")).strip().lower() == "preprocess"
        }
        preprocess_missing = sorted(set(PREPROCESS_PER_HOOK_OPS) - preprocess_available)

        required_local = set(LOCAL_PER_HOOK_OPS)
        if cfg.tp > 1:
            required_local.update(LOCAL_TP_PER_HOOK_OPS)
        if cfg.stats_sync_mode != "immediate":
            required_local.add(STATS_TAIL_OP)
        local_available = {
            str(row.get("semantic_op", "")).strip()
            for row in self.profiles.compute_rows
            if str(row.get("profiler", "")).strip().lower() == "local_compute"
        }
        local_missing = sorted(required_local - local_available)
        if not preprocess_missing and not local_missing:
            return

        fine_grained_legacy = {
            "input_loss_bd_forward_backward",
            "encoder_scale_bf_local_forward_backward",
            "encoder_norm_fd_local_forward_backward",
            "decoder_scale_bf_local_forward_backward",
            "decoder_norm_fd_local_forward_backward",
            "feature_stats_reduce_bf_global",
            "feature_stats_update_f_global",
            "grad_norm_w_enc_fd",
            "grad_scale_w_enc_fd",
        }
        mixed_legacy = {
            "input_loss_forward_backward",
            "encoder_rescale_forward_backward",
            "decoder_rescale_forward_backward",
            "feature_stats_local",
            "grad_clip_local",
        }
        legacy_found = sorted(local_available & (fine_grained_legacy | mixed_legacy))
        detail = (
            " The CSV contains an older local schema "
            f"({legacy_found}); rerun the aggregated shape-family profiler."
            if legacy_found
            else ""
        )
        raise SimulationError(
            "Compute CSV is missing aggregated profiler rows: "
            f"preprocess={preprocess_missing}, local={local_missing}.{detail}"
        )
    def _estimate_from_rows(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        label: str,
        component: str,
        semantic_op: str,
        source_path: Path,
        target: Mapping[str, float],
        numeric_fields: Sequence[str],
        count: float,
        activation: bool = False,
        notes: Sequence[str] = (),
    ) -> Estimate:
        predicted, exact, extrapolated, neighbors, predict_notes = _predict_idw(
            rows,
            label=label,
            source_path=source_path,
            target=target,
            numeric_fields=numeric_fields,
            timing=lambda row: self.profiles.time_value(row, activation=activation),
            options=self.interpolation,
        )
        all_notes = list(notes) + predict_notes
        if extrapolated:
            self.warnings.append(f"{label}: extrapolated outside measured profile range")
        phases = sorted({str(row.get("phase", "")).strip() for row in rows if str(row.get("phase", "")).strip()})
        families = sorted({str(row.get("shape_family", "")).strip() for row in rows if str(row.get("shape_family", "")).strip()})
        phase = phases[0] if len(phases) == 1 else "+".join(phases)
        shape_family = families[0] if len(families) == 1 else "+".join(families)
        return Estimate(
            label=label,
            component=component,
            semantic_op=semantic_op,
            per_call_ms=predicted,
            count=count,
            total_ms=predicted * count,
            source=str(source_path),
            exact=exact,
            extrapolated=extrapolated,
            target=dict(target),
            phase=phase,
            shape_family=shape_family,
            neighbor_points=neighbors,
            notes=all_notes,
        )

    def _compute_candidates(
        self,
        *,
        profiler: str,
        semantic_op: str,
        exact_filters: Mapping[str, Any],
        label: str,
    ) -> list[Mapping[str, str]]:
        rows = [
            row
            for row in self.profiles.compute_rows
            if str(row.get("profiler", "")).lower() == profiler.lower()
            and str(row.get("semantic_op", "")).lower() == semantic_op.lower()
            and _match_exact(row, exact_filters)
        ]
        rows, _ = _profile_device_filter(rows, self.compute_device, label=label)
        if not rows:
            raise SimulationError(
                f"{label}: no matching rows in {self.profiles.compute_csv}; "
                f"filters={dict(exact_filters)}"
            )
        return rows

    @staticmethod
    def _attach_hook(estimate: Estimate, hook: HookConfig) -> Estimate:
        estimate.hook_name = hook.name
        estimate.label = f"{hook.name}/{estimate.label}"
        estimate.target = {"hook": hook.name, **estimate.target}
        return estimate

    def _estimate_gemm(
        self, cfg: StepConfig, hook: HookConfig, op: str
    ) -> Estimate:
        label = f"gemm/{op}"
        rows = self._compute_candidates(
            profiler="gemm",
            semantic_op=op,
            exact_filters={"dtype": cfg.dtype},
            label=label,
        )
        estimate = self._estimate_from_rows(
            rows,
            label=label,
            component="gemm",
            semantic_op=op,
            source_path=self.profiles.compute_csv,
            target={
                "B": cfg.batch_size,
                "D": hook.d_in,
                "F_local_d_sae": hook.f_local(cfg.tp),
            },
            numeric_fields=("B", "D", "F_local_d_sae"),
            count=1.0,
        )
        return self._attach_hook(estimate, hook)

    def _estimate_preprocess(
        self, cfg: StepConfig, hook: HookConfig, op: str
    ) -> Estimate:
        label = f"preprocess/{op}"
        rows = self._compute_candidates(
            profiler="preprocess",
            semantic_op=op,
            exact_filters={
                "dtype": cfg.dtype,
                "normalize_activations": cfg.normalize_activations,
            },
            label=label,
        )
        estimate = self._estimate_from_rows(
            rows,
            label=label,
            component="preprocess",
            semantic_op=op,
            source_path=self.profiles.compute_csv,
            target={"B": cfg.batch_size, "D": hook.d_in},
            numeric_fields=("B", "D"),
            count=1.0,
        )
        return self._attach_hook(estimate, hook)

    def _local_spec(
        self, cfg: StepConfig, hook: HookConfig, op: str
    ) -> tuple[dict[str, Any], dict[str, float], tuple[str, ...]]:
        dtype_filter = {"dtype": cfg.dtype}
        f_local = hook.f_local(cfg.tp)

        if op == "local_compute_bd":
            return (
                {
                    **dtype_filter,
                    "normalize_activations": cfg.normalize_activations,
                },
                {"B": cfg.batch_size, "D": hook.d_in},
                ("B", "D"),
            )
        if op == "local_compute_bf_local":
            return (
                dtype_filter,
                {"B": cfg.batch_size, "F_local_d_sae": f_local},
                ("B", "F_local_d_sae"),
            )
        if op == "local_compute_fd_local":
            return (
                dtype_filter,
                {"D": hook.d_in, "F_local_d_sae": f_local},
                ("D", "F_local_d_sae"),
            )
        if op == "local_compute_bf_global":
            return (
                {**dtype_filter, "stats_sync_mode": cfg.stats_sync_mode},
                {"B": cfg.batch_size, "F_global_d_sae": hook.d_sae},
                ("B", "F_global_d_sae"),
            )
        if op == STATS_TAIL_OP:
            filters: dict[str, Any] = {"stats_sync_mode": cfg.stats_sync_mode}
            if cfg.stats_sync_mode == "periodic":
                filters["stats_sync_interval"] = cfg.stats_sync_interval
            return (
                filters,
                {"F_global_d_sae": hook.d_sae},
                ("F_global_d_sae",),
            )
        if op == "local_compute_bf_tp":
            return (
                {**dtype_filter, "tp": cfg.tp},
                {"B": cfg.batch_size, "F_local_d_sae": f_local},
                ("B", "F_local_d_sae"),
            )
        if op == "local_compute_bd_tp":
            return (
                {**dtype_filter, "tp": cfg.tp},
                {"B": cfg.batch_size, "D": hook.d_in},
                ("B", "D"),
            )
        raise SimulationError(f"Unknown local semantic operation {op!r}")

    def _estimate_local(
        self, cfg: StepConfig, hook: HookConfig, op: str, *, count: float = 1.0
    ) -> Estimate:
        label = f"local/{op}"
        exact_filters, target, numeric_fields = self._local_spec(cfg, hook, op)
        rows = self._compute_candidates(
            profiler="local_compute",
            semantic_op=op,
            exact_filters=exact_filters,
            label=label,
        )
        estimate = self._estimate_from_rows(
            rows,
            label=label,
            component="local",
            semantic_op=op,
            source_path=self.profiles.compute_csv,
            target=target,
            numeric_fields=numeric_fields,
            count=count,
        )
        return self._attach_hook(estimate, hook)

    def _estimate_stats_tail(
        self, cfg: StepConfig, hook: HookConfig
    ) -> Estimate | None:
        if cfg.stats_sync_mode == "immediate":
            return None
        if cfg.stats_sync_mode == "periodic":
            count = 1.0 / float(cfg.stats_sync_interval)
            note = f"Periodic tail amortized once every {cfg.stats_sync_interval} steps"
        elif cfg.deferred_tail_mode == "exclude":
            self.warnings.append("Deferred statistics tail excluded from ordinary-step estimate")
            return None
        elif cfg.deferred_tail_mode == "event":
            count = 1.0
            note = "Deferred flush-event cost included in this simulated step"
        else:
            assert cfg.deferred_flush_steps is not None
            count = 1.0 / float(cfg.deferred_flush_steps)
            note = f"Deferred tail amortized once every {cfg.deferred_flush_steps} steps"
        estimate = self._estimate_local(cfg, hook, STATS_TAIL_OP, count=count)
        estimate.notes.append(note)
        return estimate

    def _activation_candidates(self, cfg: StepConfig) -> list[Mapping[str, str]]:
        rows = [
            row
            for row in self.profiles.activation_rows
            if str(row.get("profiler", "")).lower() == "activation_compute"
            and str(row.get("semantic_op", "")).lower() == ACTIVATION_OP
            and _match_exact(
                row,
                {
                    "activation_type": cfg.activation_type,
                    "output_layout": cfg.activation_output_layout,
                    "dtype": cfg.dtype,
                },
            )
        ]
        if self.activation_distributed != "auto":
            rows = [
                row for row in rows
                if _bool_text(row.get("distributed", "")) == self.activation_distributed
            ]
        else:
            available = {_bool_text(row.get("distributed", "")) for row in rows}
            available.discard("")
            if len(available) > 1:
                preferred = "true" if cfg.tp > 1 else "false"
                rows = [
                    row for row in rows
                    if _bool_text(row.get("distributed", "")) == preferred
                ]
                self.warnings.append(
                    "activation: both distributed and single-device data exist; "
                    f"selected {preferred}"
                )
        rows, _ = _profile_device_filter(rows, self.activation_device, label="activation")
        if not rows:
            raise SimulationError(
                "activation: no matching rows in "
                f"{self.profiles.activation_csv} for type={cfg.activation_type}, "
                f"layout={cfg.activation_output_layout}, dtype={cfg.dtype}, "
                f"distributed={self.activation_distributed}."
            )
        return rows

    def _estimate_activation(
        self, cfg: StepConfig, hook: HookConfig
    ) -> Estimate:
        rows = self._activation_candidates(cfg)
        target: dict[str, float] = {"B": cfg.batch_size, "F_global": hook.d_sae}
        numeric_fields: tuple[str, ...] = ("B", "F_global")
        if cfg.activation_type == "topk":
            target["k"] = float(hook.k)
            numeric_fields = ("B", "F_global", "k")
        estimate = self._estimate_from_rows(
            rows,
            label=f"activation/{cfg.activation_type}",
            component="activation",
            semantic_op=ACTIVATION_OP,
            source_path=self.profiles.activation_csv,
            target=target,
            numeric_fields=numeric_fields,
            count=1.0,
            activation=True,
            notes=[
                "Includes every activation-specific forward/backward kernel; "
                "local-compute rows exclude TopK/ReLU/scatter/gather work."
            ],
        )
        return self._attach_hook(estimate, hook)

    def _nccl_base_rows(
        self,
        cfg: StepConfig,
        collective: str,
        *,
        profile_dtype: str | None = None,
    ) -> list[Mapping[str, str]]:
        if not self.profiles.nccl_csv:
            raise SimulationError("TP>1 requires --nccl-csv")
        rows = [
            row
            for row in self.profiles.nccl_rows
            if _match_exact(
                row,
                {
                    "collective": collective,
                    "tp": cfg.tp,
                    "dtype": profile_dtype or cfg.dtype,
                    "backend": self.nccl_backend,
                    "nccl_algo": self.nccl_algo,
                    "nccl_proto": self.nccl_proto,
                    "nccl_p2p_level": self.nccl_p2p_level,
                },
            )
        ]
        if not rows:
            raise SimulationError(
                f"NCCL/{collective}: no matching rows for tp={cfg.tp}, "
                f"dtype={profile_dtype or cfg.dtype}, backend={self.nccl_backend}, "
                f"topology=({self.nccl_algo},{self.nccl_proto},{self.nccl_p2p_level})"
            )
        group_rows = [
            row for row in rows if str(row.get("row_kind", "")).lower() == "group_max"
        ]
        if group_rows:
            return group_rows

        # Backward-compatible fallback: aggregate per-rank medians by taking the
        # slowest rank for each measured payload size.
        by_size: dict[int, list[Mapping[str, str]]] = {}
        for row in rows:
            size = _optional_int(row.get("buffer_bytes"))
            if size is not None:
                by_size.setdefault(size, []).append(row)
        fallback: list[Mapping[str, str]] = []
        for size, rank_rows in by_size.items():
            slowest = max(
                rank_rows,
                key=lambda row: self.profiles.time_value(row),
            )
            copied = dict(slowest)
            copied["buffer_bytes"] = str(size)
            fallback.append(copied)
        self.warnings.append(
            f"NCCL/{collective}: group_max rows absent; used slowest rank median per payload"
        )
        return fallback

    def _estimate_collective(
        self,
        cfg: StepConfig,
        hook: HookConfig,
        *,
        label: str,
        collective: str,
        buffer_bytes: int,
        notes: Sequence[str],
        profile_dtype: str | None = None,
    ) -> Estimate:
        rows = self._nccl_base_rows(cfg, collective, profile_dtype=profile_dtype)
        estimate = self._estimate_from_rows(
            rows,
            label=label,
            component="nccl",
            semantic_op=collective,
            source_path=self.profiles.nccl_csv or Path("<missing>"),
            target={"buffer_bytes": float(buffer_bytes)},
            numeric_fields=("buffer_bytes",),
            count=1.0,
            notes=notes,
        )
        return self._attach_hook(estimate, hook)

    def _estimate_tp_collectives(
        self, cfg: StepConfig, hook: HookConfig
    ) -> list[Estimate]:
        f_local = hook.f_local(cfg.tp)
        return [
            self._estimate_collective(
                cfg,
                hook,
                label="nccl/tp_encode_allgather",
                collective="allgather",
                buffer_bytes=cfg.batch_size * f_local * cfg.itemsize,
                notes=("Local contribution bytes per rank",),
            ),
            self._estimate_collective(
                cfg,
                hook,
                label="nccl/tp_decode_allreduce",
                collective="allreduce",
                buffer_bytes=cfg.batch_size * hook.d_in * cfg.itemsize,
                notes=("In-place decoder partial-output payload",),
            ),
            self._estimate_collective(
                cfg,
                hook,
                label="nccl/tp_b_dec_grad_allreduce",
                collective="allreduce",
                buffer_bytes=hook.d_in * cfg.itemsize,
                notes=("Replicated b_dec gradient",),
            ),
            self._estimate_collective(
                cfg,
                hook,
                label="nccl/tp_grad_norm_allreduce",
                collective="allreduce",
                buffer_bytes=4,
                notes=("Float32 squared-gradient-norm scalar",),
                profile_dtype="float32",
            ),
        ]

    def _estimate_optimizer(
        self, cfg: StepConfig, hook: HookConfig
    ) -> Estimate:
        op = f"{cfg.optimizer_impl}_adam_steady_step"
        label = f"optimizer/{op}"
        rows = self._compute_candidates(
            profiler="optimizer",
            semantic_op=op,
            exact_filters={
                "dtype": cfg.dtype,
                "optimizer_impl": cfg.optimizer_impl,
                "optimizer_lr": cfg.optimizer_lr,
                "optimizer_beta1": cfg.optimizer_beta1,
                "optimizer_beta2": cfg.optimizer_beta2,
                "optimizer_eps": cfg.optimizer_eps,
                "optimizer_weight_decay": cfg.optimizer_weight_decay,
                "optimizer_amsgrad": cfg.optimizer_amsgrad,
                "optimizer_maximize": cfg.optimizer_maximize,
                "optimizer_capturable": cfg.optimizer_capturable,
            },
            label=label,
        )
        # Accept the new H-free schema and old rows explicitly measured at H=1;
        # never mix H=2/4 optimizer points into the per-SAE surface.
        rows = [
            row for row in rows
            if str(row.get("H", "")).strip() == "" or _optional_int(row.get("H")) == 1
        ]
        if not rows:
            raise SimulationError(
                f"{label}: no per-SAE optimizer rows (H blank or H=1) are available"
            )
        estimate = self._estimate_from_rows(
            rows,
            label=label,
            component="optimizer",
            semantic_op=op,
            source_path=self.profiles.compute_csv,
            target={
                "D": hook.d_in,
                "F_local_d_sae": hook.f_local(cfg.tp),
            },
            numeric_fields=("D", "F_local_d_sae"),
            count=1.0,
            notes=(
                "Per-SAE optimizer estimate; heterogeneous hooks are summed additively",
            ),
        )
        return self._attach_hook(estimate, hook)


def _format_table(result: SimulationResult) -> str:
    headers = (
        "hook",
        "component",
        "shape family",
        "semantic operation",
        "per-call ms",
        "count",
        "total ms",
        "source",
    )
    rows: list[tuple[str, ...]] = []
    for estimate in result.estimates:
        source = "exact" if estimate.exact else "interpolated"
        if estimate.extrapolated:
            source += "/extrapolated"
        rows.append(
            (
                estimate.hook_name or "-",
                estimate.component,
                estimate.shape_family or "-",
                estimate.semantic_op,
                f"{estimate.per_call_ms:.4f}",
                f"{estimate.count:.4g}",
                f"{estimate.total_ms:.4f}",
                source,
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    lines = [
        " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    lines.append("")
    hook_totals = result.metadata.get("hook_totals_ms", {})
    for hook_name, value in hook_totals.items():
        lines.append(f"hook/{hook_name:>12}: {float(value):10.4f} ms")
    if hook_totals:
        lines.append("")
    for component, value in result.components_ms.items():
        lines.append(f"{component:>10}: {value:10.4f} ms")
    lines.append(f"{'TOTAL':>10}: {result.total_ms:10.4f} ms")
    if result.total_ms > 0:
        batch = int(result.config["batch_size"])
        lines.append(
            f"{'tokens/s*':>10}: {batch / (result.total_ms / 1000.0):10.2f} "
            "(*serial-operation estimate, not synchronized wall time)"
        )
    if result.warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _write_breakdown_csv(path: Path, result: SimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "hook_name",
        "component",
        "phase",
        "shape_family",
        "semantic_op",
        "per_call_ms",
        "count",
        "total_ms",
        "exact",
        "extrapolated",
        "source",
        "target",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for estimate in result.estimates:
            writer.writerow(
                {
                    "hook_name": estimate.hook_name,
                    "component": estimate.component,
                    "phase": estimate.phase,
                    "shape_family": estimate.shape_family,
                    "semantic_op": estimate.semantic_op,
                    "per_call_ms": estimate.per_call_ms,
                    "count": estimate.count,
                    "total_ms": estimate.total_ms,
                    "exact": estimate.exact,
                    "extrapolated": estimate.extrapolated,
                    "source": estimate.source,
                    "target": json.dumps(estimate.target, separators=(",", ":")),
                    "notes": json.dumps(estimate.notes, separators=(",", ":")),
                }
            )


def _broadcast_hook_values(values: Sequence[Any], count: int, *, name: str) -> list[Any]:
    items = list(values)
    if len(items) == count:
        return items
    if len(items) == 1:
        return items * count
    raise SimulationError(
        f"{name} must contain either one value or exactly {count} values; got {items}"
    )


def _parse_hook_spec(text: str, *, default_k: int) -> HookConfig:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in {3, 4}:
        raise SimulationError(
            f"Invalid --hook-spec {text!r}; expected NAME:D_IN:D_SAE[:K]"
        )
    name, d_in_text, d_sae_text = parts[:3]
    k_text = parts[3] if len(parts) == 4 else str(default_k)
    try:
        return HookConfig(name=name, d_in=int(d_in_text), d_sae=int(d_sae_text), k=int(k_text))
    except ValueError as exc:
        raise SimulationError(
            f"Invalid numeric field in --hook-spec {text!r}"
        ) from exc


def _hooks_from_args(args: argparse.Namespace) -> tuple[HookConfig, ...]:
    if args.hook_specs:
        if args.hook_names:
            raise SimulationError("--hook-name cannot be combined with --hook-spec")
        default_k = int(args.k_values[0])
        hooks = tuple(_parse_hook_spec(text, default_k=default_k) for text in args.hook_specs)
        if args.num_hooks is not None and args.num_hooks != len(hooks):
            raise SimulationError(
                f"--num-hooks={args.num_hooks} disagrees with {len(hooks)} --hook-spec entries"
            )
        return hooks

    candidate_counts = [len(args.d_in_values), len(args.d_sae_values), len(args.k_values)]
    if args.hook_names:
        candidate_counts.append(len(args.hook_names))
    inferred = max(candidate_counts)
    if args.num_hooks is not None:
        if inferred > 1 and args.num_hooks != inferred:
            raise SimulationError(
                f"--num-hooks={args.num_hooks} disagrees with vector length {inferred}"
            )
        count = args.num_hooks
    else:
        count = inferred

    d_in_values = _broadcast_hook_values(args.d_in_values, count, name="--d-in")
    d_sae_values = _broadcast_hook_values(args.d_sae_values, count, name="--d-sae")
    k_values = _broadcast_hook_values(args.k_values, count, name="--k")
    names = (
        _broadcast_hook_values(args.hook_names, count, name="--hook-name")
        if args.hook_names
        else [f"h{index + 1}" for index in range(count)]
    )
    return tuple(
        HookConfig(name=str(name), d_in=int(d_in), d_sae=int(d_sae), k=int(k))
        for name, d_in, d_sae, k in zip(names, d_in_values, d_sae_values, k_values)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate one SAE training step from the fixed profiler CSVs."
    )
    parser.add_argument("--compute-csv", type=Path, default=DEFAULT_COMPUTE_CSV)
    parser.add_argument(
        "--activation-csv", "--topk-csv", dest="activation_csv",
        type=Path, default=DEFAULT_ACTIVATION_CSV
    )
    parser.add_argument("--nccl-csv", type=Path, default=DEFAULT_NCCL_CSV)

    parser.add_argument(
        "--hook-spec",
        dest="hook_specs",
        action="append",
        default=[],
        metavar="NAME:D_IN:D_SAE[:K]",
        help=(
            "Repeat once per heterogeneous hook. Example: "
            "--hook-spec h1:4096:16384:128 --hook-spec h2:5120:32768:128"
        ),
    )
    parser.add_argument("--hook-name", dest="hook_names", nargs="+", default=[])
    parser.add_argument("--d-in", dest="d_in_values", type=int, nargs="+", default=[DEFAULT_D_IN])
    parser.add_argument(
        "--d-sae", dest="d_sae_values", type=int, nargs="+",
        default=[DEFAULT_D_SAE], help="One global SAE width per hook, or one shared value"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Tokens per TP rank")
    parser.add_argument(
        "--num-hooks", type=int, default=None,
        help="Backward-compatible homogeneous repeat count; normally inferred from hook vectors"
    )
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument("--k", dest="k_values", type=int, nargs="+", default=[DEFAULT_K])
    parser.add_argument(
        "--activation-type", choices=("topk", "relu"),
        default=DEFAULT_ACTIVATION_TYPE
    )
    parser.add_argument(
        "--activation-output-layout", choices=("dense",),
        default=DEFAULT_ACTIVATION_OUTPUT_LAYOUT
    )
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument(
        "--normalize-activations",
        choices=NORMALIZE_ACTIVATION_MODES,
        default=DEFAULT_NORMALIZE_ACTIVATIONS,
        help=(
            "SAE normalize_activations mode used to select preprocess/local-BD "
            "compute rows"
        ),
    )
    parser.add_argument(
        "--optimizer-impl",
        choices=("default", "foreach", "forloop", "fused"),
        default=DEFAULT_OPTIMIZER_IMPL,
    )
    parser.add_argument("--optimizer-lr", type=float, default=DEFAULT_OPTIMIZER_LR)
    parser.add_argument("--optimizer-beta1", type=float, default=DEFAULT_OPTIMIZER_BETA1)
    parser.add_argument("--optimizer-beta2", type=float, default=DEFAULT_OPTIMIZER_BETA2)
    parser.add_argument("--optimizer-eps", type=float, default=DEFAULT_OPTIMIZER_EPS)
    parser.add_argument(
        "--optimizer-weight-decay", type=float, default=DEFAULT_OPTIMIZER_WEIGHT_DECAY
    )
    parser.add_argument(
        "--optimizer-amsgrad", action="store_true", default=DEFAULT_OPTIMIZER_AMSGRAD
    )
    parser.add_argument(
        "--optimizer-maximize", action="store_true", default=DEFAULT_OPTIMIZER_MAXIMIZE
    )
    parser.add_argument(
        "--optimizer-capturable", action="store_true", default=DEFAULT_OPTIMIZER_CAPTURABLE
    )
    parser.add_argument(
        "--stats-sync-mode",
        choices=("immediate", "periodic", "deferred"),
        default=DEFAULT_STATS_SYNC_MODE,
    )
    parser.add_argument(
        "--stats-sync-interval", type=int, default=DEFAULT_STATS_SYNC_INTERVAL
    )
    parser.add_argument(
        "--deferred-tail-mode",
        choices=("exclude", "event", "amortize"),
        default="exclude",
    )
    parser.add_argument("--deferred-flush-steps", type=int)

    parser.add_argument("--compute-device", help="Exact device field in compute CSV")
    parser.add_argument(
        "--activation-device", "--topk-device", dest="activation_device",
        help="Exact device field in activation CSV"
    )
    parser.add_argument(
        "--activation-distributed", "--topk-distributed",
        dest="activation_distributed",
        choices=("auto", "true", "false"),
        default="auto",
    )
    parser.add_argument("--nccl-backend", default="nccl")
    parser.add_argument("--nccl-algo", default="auto")
    parser.add_argument("--nccl-proto", default="auto")
    parser.add_argument("--nccl-p2p-level", default="auto")

    parser.add_argument("--time-column", default=DEFAULT_TIME_COLUMN)
    parser.add_argument(
        "--no-prefer-clean-activation", "--no-prefer-clean-topk",
        dest="no_prefer_clean_activation",
        action="store_true",
        help="Use --time-column for activation rows even when clean_ms is available",
    )
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--idw-power", type=float, default=DEFAULT_IDW_POWER)
    parser.add_argument(
        "--extrapolation",
        choices=("error", "clamp", "allow"),
        default=DEFAULT_EXTRAPOLATION,
    )

    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        hooks = _hooks_from_args(args)
        cfg = StepConfig(
            hooks=hooks,
            batch_size=args.batch_size,
            tp=args.tp,
            activation_type=args.activation_type,
            activation_output_layout=args.activation_output_layout,
            dtype=canonical_dtype(args.dtype),
            normalize_activations=canonical_normalize_activations(
                args.normalize_activations
            ),
            optimizer_impl=args.optimizer_impl,
            optimizer_lr=args.optimizer_lr,
            optimizer_beta1=args.optimizer_beta1,
            optimizer_beta2=args.optimizer_beta2,
            optimizer_eps=args.optimizer_eps,
            optimizer_weight_decay=args.optimizer_weight_decay,
            optimizer_amsgrad=args.optimizer_amsgrad,
            optimizer_maximize=args.optimizer_maximize,
            optimizer_capturable=args.optimizer_capturable,
            stats_sync_mode=args.stats_sync_mode,
            stats_sync_interval=args.stats_sync_interval,
            deferred_tail_mode=args.deferred_tail_mode,
            deferred_flush_steps=args.deferred_flush_steps,
        )
        nccl_csv = args.nccl_csv if cfg.tp > 1 else None
        profiles = CsvProfiles(
            args.compute_csv,
            args.activation_csv,
            nccl_csv,
            time_column=args.time_column,
            prefer_clean_activation=not args.no_prefer_clean_activation,
        )
        simulator = StepTimeSimulator(
            profiles,
            InterpolationOptions(
                neighbors=args.neighbors,
                power=args.idw_power,
                extrapolation=args.extrapolation,
            ),
            compute_device=args.compute_device,
            activation_device=args.activation_device,
            activation_distributed=args.activation_distributed,
            nccl_backend=args.nccl_backend,
            nccl_algo=args.nccl_algo,
            nccl_proto=args.nccl_proto,
            nccl_p2p_level=args.nccl_p2p_level,
        )
        result = simulator.simulate(cfg)
    except (SimulationError, ValueError, OSError) as exc:
        print(f"[simulate-step-time] ERROR: {exc}", file=sys.stderr)
        return 2

    print(_format_table(result))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON: {args.output_json}")
    if args.output_csv:
        _write_breakdown_csv(args.output_csv, result)
        print(f"CSV : {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
