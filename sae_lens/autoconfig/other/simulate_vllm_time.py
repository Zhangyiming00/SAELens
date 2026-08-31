#!/usr/bin/env python3
"""Estimate vLLM activation-generation time from measured profile JSON.

The normal path is a direct lookup in a profile produced by
``explore_vllm_activation_profile_v3_2.py``. If the requested hook depth was not
measured, the estimator uses the last hook layer as the stop layer:

* bracketed by measured stop layers for the same TP: linear interpolation
* outside the measured range: proportional scaling from the nearest measured row
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROFILE_JSON = ROOT / "results" / "vllm_runmatch" / "profile.json"

HOOK_LAYER_RE = re.compile(r"\bblocks\.(\d+)\.")


@dataclass(frozen=True)
class ProfileRow:
    tp: int
    stop_at_layer: int
    total_tokens: int
    wall_ms_median: float
    source: Mapping[str, Any]


@dataclass(frozen=True)
class Estimate:
    tp: int
    vllm_dp: int
    stop_at_layer: int
    batch_tokens: int
    tokens_per_call: int
    calls_per_step: float
    per_call_ms: float
    profiled_vllm_ms: float
    method: str
    source_stop_layers: tuple[int, ...]


def _as_int(value: object, field: str) -> int:
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return int(value)


def _as_float(value: object, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return float(value)


def _profile_rows(payload: object) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        rows = payload.get("rows", [])
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("profile JSON must be a list or an object with a rows list")
    return [row for row in rows if isinstance(row, Mapping)]


def load_profile(path: Path) -> list[ProfileRow]:
    payload = json.loads(path.read_text())
    rows: list[ProfileRow] = []
    for raw in _profile_rows(payload):
        if str(raw.get("status", "ok")) != "ok":
            continue
        stop_at_layer = raw.get("stop_at_layer")
        if stop_at_layer in {None, ""}:
            continue
        try:
            rows.append(
                ProfileRow(
                    tp=_as_int(raw.get("tp"), "tp"),
                    stop_at_layer=_as_int(stop_at_layer, "stop_at_layer"),
                    total_tokens=_as_int(raw.get("total_tokens"), "total_tokens"),
                    wall_ms_median=_as_float(
                        raw.get("wall_ms_median"), "wall_ms_median"
                    ),
                    source=raw,
                )
            )
        except (TypeError, ValueError):
            continue
    if not rows:
        raise ValueError(f"no usable profile rows in {path}")
    return rows


def load_profile_csv(path: Path) -> list[ProfileRow]:
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[ProfileRow] = []
    for raw in raw_rows:
        if str(raw.get("status", "ok")) != "ok":
            continue
        if raw.get("stop_at_layer") in {None, ""}:
            continue
        try:
            rows.append(
                ProfileRow(
                    tp=_as_int(raw.get("tp"), "tp"),
                    stop_at_layer=_as_int(raw.get("stop_at_layer"), "stop_at_layer"),
                    total_tokens=_as_int(raw.get("total_tokens"), "total_tokens"),
                    wall_ms_median=_as_float(
                        raw.get("wall_ms_median"), "wall_ms_median"
                    ),
                    source=raw,
                )
            )
        except (TypeError, ValueError):
            continue
    if not rows:
        raise ValueError(f"no usable profile rows in {path}")
    return rows


def load_profile_auto(path: Path) -> list[ProfileRow]:
    if path.suffix.lower() == ".csv":
        return load_profile_csv(path)
    return load_profile(path)


def stop_layer_from_hook_names(hook_names: Sequence[str]) -> int:
    layers: list[int] = []
    for hook_name in hook_names:
        match = HOOK_LAYER_RE.search(hook_name)
        if match:
            layers.append(int(match.group(1)))
    if not layers:
        raise ValueError("could not infer layer from --hook-name/--hook-names")
    return max(layers) + 1


def select_profile_row(
    rows: Sequence[ProfileRow],
    *,
    tp: int,
    stop_at_layer: int,
    total_tokens: int | None = None,
) -> ProfileRow:
    candidates = [row for row in rows if row.tp == tp]
    if total_tokens is not None:
        token_matches = [row for row in candidates if row.total_tokens == total_tokens]
        if token_matches:
            candidates = token_matches
    for row in candidates:
        if row.stop_at_layer == stop_at_layer:
            return row
    raise LookupError(
        f"no direct profile row for tp={tp}, stop_at_layer={stop_at_layer}"
    )


def _rows_for_tp(
    rows: Sequence[ProfileRow],
    *,
    tp: int,
    preferred_total_tokens: int | None,
) -> list[ProfileRow]:
    candidates = [row for row in rows if row.tp == tp]
    if not candidates:
        raise LookupError(f"no profile rows for tp={tp}")
    if preferred_total_tokens is not None:
        token_matches = [
            row for row in candidates if row.total_tokens == preferred_total_tokens
        ]
        if token_matches:
            candidates = token_matches
    by_layer: dict[int, ProfileRow] = {}
    for row in sorted(candidates, key=lambda item: item.wall_ms_median):
        by_layer[row.stop_at_layer] = row
    return [by_layer[layer] for layer in sorted(by_layer)]


def _estimate_per_call(
    rows: Sequence[ProfileRow],
    *,
    tp: int,
    stop_at_layer: int,
    preferred_total_tokens: int | None,
) -> tuple[float, int, str, tuple[int, ...]]:
    candidates = _rows_for_tp(
        rows,
        tp=tp,
        preferred_total_tokens=preferred_total_tokens,
    )
    layers = tuple(row.stop_at_layer for row in candidates)
    for row in candidates:
        if row.stop_at_layer == stop_at_layer:
            return row.wall_ms_median, row.total_tokens, "direct", (row.stop_at_layer,)

    lower = [row for row in candidates if row.stop_at_layer < stop_at_layer]
    upper = [row for row in candidates if row.stop_at_layer > stop_at_layer]
    if lower and upper:
        lo = lower[-1]
        hi = upper[0]
        weight = (stop_at_layer - lo.stop_at_layer) / (
            hi.stop_at_layer - lo.stop_at_layer
        )
        per_call = lo.wall_ms_median + (hi.wall_ms_median - lo.wall_ms_median) * weight
        return per_call, lo.total_tokens, "interpolated", (lo.stop_at_layer, hi.stop_at_layer)

    nearest = lower[-1] if lower else upper[0]
    per_call = nearest.wall_ms_median * stop_at_layer / nearest.stop_at_layer
    return per_call, nearest.total_tokens, "proportional", (nearest.stop_at_layer,)


def estimate_step_vllm_ms(
    rows: Sequence[ProfileRow],
    *,
    tp: int,
    vllm_dp: int,
    batch_tokens: int,
    stop_at_layer: int,
    preferred_total_tokens: int | None = None,
) -> Estimate:
    if vllm_dp < 1:
        raise ValueError("vllm_dp must be >= 1")
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be > 0")
    per_call_ms, tokens_per_call, method, source_layers = _estimate_per_call(
        rows,
        tp=tp,
        stop_at_layer=stop_at_layer,
        preferred_total_tokens=preferred_total_tokens,
    )
    calls = batch_tokens / tokens_per_call
    return Estimate(
        tp=tp,
        vllm_dp=vllm_dp,
        stop_at_layer=stop_at_layer,
        batch_tokens=batch_tokens,
        tokens_per_call=tokens_per_call,
        calls_per_step=calls,
        per_call_ms=per_call_ms,
        profiled_vllm_ms=calls / vllm_dp * per_call_ms,
        method=method,
        source_stop_layers=source_layers,
    )


def _available_tps(rows: Sequence[ProfileRow]) -> list[int]:
    return sorted({row.tp for row in rows})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--vllm-dp", type=int, default=1)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--stop-at-layer", type=int, default=None)
    parser.add_argument("--hook-name", action="append", default=[])
    parser.add_argument("--hook-names", nargs="+", default=[])
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_profile_auto(args.profile)
    hook_names = [*args.hook_name, *args.hook_names]
    stop_at_layer = args.stop_at_layer
    if stop_at_layer is None:
        if hook_names:
            stop_at_layer = stop_layer_from_hook_names(hook_names)
        else:
            profile_layers = sorted({row.stop_at_layer for row in rows})
            if len(profile_layers) != 1:
                raise SystemExit(
                    "--stop-at-layer or --hook-name is required when profile has "
                    f"multiple stop layers: {profile_layers}"
                )
            stop_at_layer = profile_layers[0]

    tps = [args.tp] if args.tp is not None else _available_tps(rows)
    estimates = [
        estimate_step_vllm_ms(
            rows,
            tp=tp,
            vllm_dp=args.vllm_dp,
            batch_tokens=args.batch_tokens,
            stop_at_layer=stop_at_layer,
            preferred_total_tokens=args.total_tokens,
        )
        for tp in tps
    ]

    if args.json:
        print(json.dumps([asdict(item) for item in estimates], indent=2))
    else:
        for item in estimates:
            print(
                "tp={tp} stop_at_layer={layer} method={method} "
                "wall_ms_median={per_call:.12g} total_tokens={tokens} "
                "profiled_vllm_ms={step:.12g}".format(
                    tp=item.tp,
                    layer=item.stop_at_layer,
                    method=item.method,
                    per_call=item.per_call_ms,
                    tokens=item.tokens_per_call,
                    step=item.profiled_vllm_ms,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
