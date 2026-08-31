#!/usr/bin/env python3
"""Estimate vLLM activation-generation time from exact external-profiler rows.

Selection contract
------------------
1. model_name, dtype and TP MUST match exactly (dtype aliases are canonicalized).
2. The caller specifies hook name(s), never stop_at_layer directly.
3. Requested stop_at_layer = deepest blocks.<L> hook + 1.
4. If that stop layer exists, return a measured row unchanged.
   If several rows exist, prefer the row whose profiled hook_count is closest to
   the requested hook count; ties may use the first row.
5. If the stop layer was not measured:
   * measured layers on both sides -> linear interpolation in stop_at_layer;
   * only one side -> proportional scaling from the nearest measured layer.
6. batch_size, MBT, context size and profiled hook names are NOT simulator
   dimensions.  They may be present in the CSV but are ignored except that
   hook_count is used as the tie-breaker above.
7. There is NO batch-token / throughput scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PROFILE = Path("sae_lens/autoconfig/profile_results/external_vllm_profile.csv")
HOOK_LAYER_RE = re.compile(r"\bblocks\.(\d+)\.")

DTYPE_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


@dataclass(frozen=True)
class ProfileRow:
    model_name: str
    dtype: str
    tp: int
    stop_at_layer: int
    hook_count: int
    wall_ms_median: float
    source: Mapping[str, Any]


@dataclass(frozen=True)
class Estimate:
    model_name: str
    dtype: str
    tp: int
    requested_hooks: tuple[str, ...]
    requested_hook_count: int
    stop_at_layer: int
    vllm_ms: float
    method: str
    source_stop_layers: tuple[int, ...]
    source_hook_counts: tuple[int, ...]
    source_wall_ms: tuple[float, ...]


def canonical_dtype(value: str) -> str:
    key = value.strip().lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"unsupported dtype {value!r}; choose float16/bfloat16/float32")
    return DTYPE_ALIASES[key]


def _as_int(value: Any, field: str) -> int:
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return int(value)


def _as_float(value: Any, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}")
    return result


def _infer_hook_count(raw: Mapping[str, Any]) -> int:
    value = raw.get("hook_count")
    if value not in (None, ""):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            pass
    hooks = raw.get("hook_names")
    if hooks in (None, ""):
        return 0
    if isinstance(hooks, (list, tuple)):
        return len(hooks)
    text = str(hooks)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return len(parsed)
    except json.JSONDecodeError:
        pass
    return len([part for part in text.split(",") if part.strip()])


def load_profile(path: Path) -> list[ProfileRow]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping):
            raw_rows = payload.get("rows", [])
        else:
            raw_rows = payload
        if not isinstance(raw_rows, list):
            raise ValueError("JSON profile must be a list or an object with a rows list")
    else:
        with path.open(newline="") as handle:
            raw_rows = list(csv.DictReader(handle))

    rows: list[ProfileRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("status", "ok")) != "ok":
            continue
        try:
            rows.append(
                ProfileRow(
                    model_name=str(raw.get("model_name", "")),
                    dtype=canonical_dtype(str(raw.get("dtype", ""))),
                    tp=_as_int(raw.get("tp"), "tp"),
                    stop_at_layer=_as_int(raw.get("stop_at_layer"), "stop_at_layer"),
                    hook_count=_infer_hook_count(raw),
                    wall_ms_median=_as_float(raw.get("wall_ms_median"), "wall_ms_median"),
                    source=raw,
                )
            )
        except (TypeError, ValueError):
            continue
    if not rows:
        raise ValueError(f"no usable profile rows in {path}")
    return rows


def stop_layer_from_hooks(hooks: Sequence[str]) -> int:
    if not hooks:
        raise ValueError("at least one --hook-name/--hook-names value is required")
    layers: list[int] = []
    for hook in hooks:
        match = HOOK_LAYER_RE.search(hook)
        if match is None:
            raise ValueError(
                f"cannot infer stop_at_layer from hook {hook!r}; "
                "use blocks.<N>.<hook> names"
            )
        layers.append(int(match.group(1)))
    return max(layers) + 1


def _exact_dimension_rows(
    rows: Sequence[ProfileRow], *, model_name: str, dtype: str, tp: int
) -> list[ProfileRow]:
    dtype = canonical_dtype(dtype)
    return [
        row
        for row in rows
        if row.model_name == model_name and row.dtype == dtype and row.tp == tp
    ]


def _choose_for_layer(
    rows: Sequence[ProfileRow], *, stop_at_layer: int, requested_hook_count: int
) -> ProfileRow:
    candidates = [row for row in rows if row.stop_at_layer == stop_at_layer]
    if not candidates:
        raise LookupError(f"no rows at stop_at_layer={stop_at_layer}")
    # Stable min(): ties keep the first CSV row, which satisfies "随便一条".
    return min(candidates, key=lambda row: abs(row.hook_count - requested_hook_count))


def estimate_vllm_ms(
    rows: Sequence[ProfileRow],
    *,
    model_name: str,
    dtype: str,
    tp: int,
    hooks: Sequence[str],
) -> Estimate:
    dtype = canonical_dtype(dtype)
    stop_at_layer = stop_layer_from_hooks(hooks)
    requested_hook_count = len(hooks)
    candidates = _exact_dimension_rows(
        rows, model_name=model_name, dtype=dtype, tp=tp
    )
    if not candidates:
        available = sorted({(r.model_name, r.dtype, r.tp) for r in rows})
        sample = available[:12]
        suffix = " ..." if len(available) > len(sample) else ""
        raise LookupError(
            "no exact profile group for "
            f"model_name={model_name!r}, dtype={dtype!r}, tp={tp}. "
            f"Available groups: {sample}{suffix}"
        )

    measured_layers = sorted({row.stop_at_layer for row in candidates})
    if stop_at_layer in measured_layers:
        row = _choose_for_layer(
            candidates,
            stop_at_layer=stop_at_layer,
            requested_hook_count=requested_hook_count,
        )
        return Estimate(
            model_name=model_name,
            dtype=dtype,
            tp=tp,
            requested_hooks=tuple(hooks),
            requested_hook_count=requested_hook_count,
            stop_at_layer=stop_at_layer,
            vllm_ms=row.wall_ms_median,
            method="direct",
            source_stop_layers=(row.stop_at_layer,),
            source_hook_counts=(row.hook_count,),
            source_wall_ms=(row.wall_ms_median,),
        )

    lower_layers = [layer for layer in measured_layers if layer < stop_at_layer]
    upper_layers = [layer for layer in measured_layers if layer > stop_at_layer]

    if lower_layers and upper_layers:
        lo_layer = lower_layers[-1]
        hi_layer = upper_layers[0]
        lo = _choose_for_layer(
            candidates,
            stop_at_layer=lo_layer,
            requested_hook_count=requested_hook_count,
        )
        hi = _choose_for_layer(
            candidates,
            stop_at_layer=hi_layer,
            requested_hook_count=requested_hook_count,
        )
        weight = (stop_at_layer - lo_layer) / (hi_layer - lo_layer)
        value = lo.wall_ms_median + (hi.wall_ms_median - lo.wall_ms_median) * weight
        return Estimate(
            model_name=model_name,
            dtype=dtype,
            tp=tp,
            requested_hooks=tuple(hooks),
            requested_hook_count=requested_hook_count,
            stop_at_layer=stop_at_layer,
            vllm_ms=value,
            method="interpolated",
            source_stop_layers=(lo_layer, hi_layer),
            source_hook_counts=(lo.hook_count, hi.hook_count),
            source_wall_ms=(lo.wall_ms_median, hi.wall_ms_median),
        )

    # Outside the measured layer range: proportional scaling from the nearest row.
    source_layer = lower_layers[-1] if lower_layers else upper_layers[0]
    source = _choose_for_layer(
        candidates,
        stop_at_layer=source_layer,
        requested_hook_count=requested_hook_count,
    )
    if source_layer <= 0:
        raise LookupError("cannot proportionally scale from stop_at_layer <= 0")
    value = source.wall_ms_median * stop_at_layer / source_layer
    return Estimate(
        model_name=model_name,
        dtype=dtype,
        tp=tp,
        requested_hooks=tuple(hooks),
        requested_hook_count=requested_hook_count,
        stop_at_layer=stop_at_layer,
        vllm_ms=value,
        method="proportional",
        source_stop_layers=(source_layer,),
        source_hook_counts=(source.hook_count,),
        source_wall_ms=(source.wall_ms_median,),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--hook-name", action="append", default=[])
    parser.add_argument("--hook-names", nargs="+", default=[])
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tp <= 0:
        raise SystemExit("--tp must be > 0")
    hooks = [*args.hook_name, *args.hook_names]
    if not hooks:
        raise SystemExit("provide --hook-name or --hook-names")
    rows = load_profile(args.profile)
    estimate = estimate_vllm_ms(
        rows,
        model_name=args.model_name,
        dtype=args.dtype,
        tp=args.tp,
        hooks=hooks,
    )
    if args.json:
        print(json.dumps(asdict(estimate), indent=2))
    else:
        print(
            "model={model} dtype={dtype} tp={tp} stop_at_layer={stop} "
            "hooks={hooks} method={method} vllm_ms={ms:.12g} "
            "source_layers={layers} source_hook_counts={counts}".format(
                model=estimate.model_name,
                dtype=estimate.dtype,
                tp=estimate.tp,
                stop=estimate.stop_at_layer,
                hooks=estimate.requested_hook_count,
                method=estimate.method,
                ms=estimate.vllm_ms,
                layers=list(estimate.source_stop_layers),
                counts=list(estimate.source_hook_counts),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
