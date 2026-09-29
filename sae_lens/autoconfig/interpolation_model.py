"""Strict measured-grid interpolation shared by runtime and sparse profiles.

Only declared numeric axes interpolate. All other configuration fields are
exact family keys (including JSON null versus the string 'none'). Missing
corners, changed execution paths and extrapolation are errors, not scaling
heuristics. This module has no Torch/CUDA dependency.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def family(config, axes):
    return canonical({k: v for k, v in config.items() if k not in axes})


def combine(values, weights=None):
    first = values[0]
    if isinstance(first, dict):
        if any(v.keys() != first.keys() for v in values):
            raise ValueError("Measurement fields differ between corners")
        return {k: combine([v[k] for v in values], weights) for k in first}
    if isinstance(first, list):
        if any(len(v) != len(first) for v in values):
            raise ValueError("Rank counts differ between corners")
        return [combine([v[i] for v in values], weights) for i in range(len(first))]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Interpolation metrics must be finite numbers")
    return statistics.median(values) if weights is None else sum(v*w for v, w in zip(values, weights))


def interpolate(config, rows, axes):
    """Multilinear interpolation, returning both metrics and contributing rows."""
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("Expected distinct interpolation axes")
    if any(type(config.get(a)) not in (int, float) or not math.isfinite(config[a]) for a in axes):
        raise ValueError("Missing or nonnumeric interpolation coordinate")
    matching = [r for r in rows if family(r["config"], axes) == family(config, axes)]
    if not matching:
        raise ValueError("Unmeasured execution family; profile this exact policy/topology first")
    bounds = []
    for axis in axes:
        coordinates = sorted(set(r["config"][axis] for r in matching))
        x = config[axis]
        if x < coordinates[0] or x > coordinates[-1]:
            raise ValueError(f"No extrapolation: {axis}={x}, measured [{coordinates[0]}, {coordinates[-1]}]")
        lo = max(v for v in coordinates if v <= x)
        hi = min(v for v in coordinates if v >= x)
        bounds.append([(lo, 1.)] if lo == hi else [(lo, (hi-x)/(hi-lo)), (hi, (x-lo)/(hi-lo))])
    metrics, weights, sources, regimes = [], [], [], set()
    for corner in itertools.product(*bounds):
        coordinate, weight = tuple(p[0] for p in corner), math.prod(p[1] for p in corner)
        found = [r for r in matching if tuple(r["config"][a] for a in axes) == coordinate]
        if not found:
            raise ValueError(f"Missing interpolation corner: {dict(zip(axes, coordinate))}")
        regimes.update(canonical(r["regime"]) for r in found)
        metrics.append(combine([r["metrics"] for r in found]))
        weights.append(weight)
        sources.append(dict(coordinate=dict(zip(axes, coordinate)), weight=weight,
                            measurements=[r["name"] for r in found]))
    if len(regimes) != 1:
        raise ValueError("Execution path changes across interpolation corners; split and reprofile the regime")
    return dict(metrics=combine(metrics, weights), corners=sources,
                regime=json.loads(next(iter(regimes))), exact=len(sources) == 1)
