from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_COMMAND_TEMPLATE = [
    "baseline",
    "vDP+sDP(fsdp)",
    "sDP(fsdp)+bs8",
    "vTP+sDP(fsdp)+bs8",
    "vDP+sDP(ddp)",
    "sDP(ddp)+bs8",
    "vTP+sDP(ddp)+bs8",
    "vTP",
    "sTP",
    "vTP+sTP",
    "vDP",
    "vDP+sTP",
]


@dataclass
class RunRow:
    run_id: str
    timestamp: str
    hook_count: int
    slot: int
    scenario: str
    steps: int
    final_samples: int
    elapsed_s: float
    complete: bool
    local_completion_ratio: float | None
    local_tokens_per_s: float
    normalized_tokens_per_s: float
    training_tokens: int | None
    notes: tuple[str, ...]


@dataclass
class HookGroupReport:
    hook_count: int
    rows: list[RunRow]

    @property
    def best_complete(self) -> RunRow | None:
        complete_rows = [row for row in self.rows if row.complete]
        if not complete_rows:
            return None
        return max(complete_rows, key=lambda row: row.normalized_tokens_per_s)


@dataclass
class Report:
    groups: dict[int, HookGroupReport]
    rows: list[RunRow]


@dataclass
class _RunRaw:
    run_id: str
    timestamp: str
    run_dir: Path
    steps: int
    final_samples: int
    elapsed_s: float
    cfg: dict[str, Any] | None
    hook_count: int | None
    sort_key: tuple[int, str]


def _parse_timestamp_from_run_id(run_id: str) -> str:
    prefix = "saelens_runner_gpu_"
    if run_id.startswith(prefix):
        return run_id[len(prefix) :]
    return run_id


def _timestamp_sort_key(timestamp: str) -> tuple[int, str]:
    compact = timestamp.replace("_", "")
    if compact.isdigit():
        return (int(compact), timestamp)
    return (10**20, timestamp)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_runner_cfg(run_dir: Path, checkpoints_dir: Path | None) -> dict[str, Any] | None:
    local_cfg = run_dir / "runner_cfg.json"
    if local_cfg.exists():
        with local_cfg.open("r", encoding="utf-8") as f:
            return json.load(f)

    if checkpoints_dir is None:
        return None

    timestamp = _parse_timestamp_from_run_id(run_dir.name)
    ckpt_run_dir = checkpoints_dir / timestamp
    if not ckpt_run_dir.exists():
        nearest = _find_nearest_checkpoint_dir(checkpoints_dir, timestamp)
        if nearest is None:
            return None
        ckpt_run_dir = nearest
    candidates = sorted(ckpt_run_dir.glob("final_*/runner_cfg.json"))
    if not candidates:
        return None
    with candidates[-1].open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_nearest_checkpoint_dir(checkpoints_dir: Path, timestamp: str) -> Path | None:
    try:
        target_dt = datetime.strptime(timestamp, "%y%m%d_%H%M%S")
    except ValueError:
        return None
    date_prefix = timestamp.split("_")[0]
    best: tuple[float, Path] | None = None
    for candidate in checkpoints_dir.glob(f"{date_prefix}_*"):
        if not candidate.is_dir():
            continue
        try:
            candidate_dt = datetime.strptime(candidate.name, "%y%m%d_%H%M%S")
        except ValueError:
            continue
        delta = abs((candidate_dt - target_dt).total_seconds())
        if delta > 2.0:
            continue
        if best is None or delta < best[0]:
            best = (delta, candidate)
    return best[1] if best is not None else None


def _infer_hook_count(run_dir: Path, cfg: dict[str, Any] | None) -> int | None:
    from_cfg: int | None = None
    if cfg is not None:
        hook_names = cfg.get("hook_names")
        if isinstance(hook_names, list) and len(hook_names) > 0:
            from_cfg = len(hook_names)
        elif cfg.get("hook_name"):
            from_cfg = 1

    manifest_path = run_dir / "multi_sae_manifest.json"
    from_manifest: int | None = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        hook_names = manifest.get("hook_names")
        if isinstance(hook_names, list) and len(hook_names) > 0:
            from_manifest = len(hook_names)

    hook_dirs = [
        child for child in run_dir.iterdir() if child.is_dir() and child.name.startswith("blocks_")
    ]
    from_dirs = len(hook_dirs) if hook_dirs else None

    candidates = [value for value in (from_cfg, from_manifest, from_dirs) if value is not None]
    if not candidates:
        return None
    return max(candidates)


def _fill_missing_hook_counts(runs: list[_RunRaw]) -> None:
    known_indices = [idx for idx, run in enumerate(runs) if run.hook_count is not None]
    if not known_indices:
        for run in runs:
            run.hook_count = 0
        return

    for idx, run in enumerate(runs):
        if run.hook_count is not None:
            continue

        prev_idx: int | None = None
        next_idx: int | None = None
        for kidx in known_indices:
            if kidx < idx:
                prev_idx = kidx
            elif kidx > idx:
                next_idx = kidx
                break

        if prev_idx is None and next_idx is None:
            run.hook_count = 0
            continue
        if prev_idx is None:
            run.hook_count = runs[next_idx].hook_count
            continue
        if next_idx is None:
            run.hook_count = runs[prev_idx].hook_count
            continue

        prev_distance = idx - prev_idx
        next_distance = next_idx - idx
        if prev_distance < next_distance:
            run.hook_count = runs[prev_idx].hook_count
        else:
            # Tie-break to "next" so group-boundary failed runs belong to
            # the upcoming phase rather than the previous one.
            run.hook_count = runs[next_idx].hook_count


def _collect_raw_runs(results_dir: Path, checkpoints_dir: Path | None) -> list[_RunRaw]:
    raw_runs: list[_RunRaw] = []
    for run_dir in sorted(results_dir.glob("saelens_runner_gpu_*")):
        timing_path = run_dir / "timing_history.jsonl"
        if not timing_path.exists():
            continue
        records = _read_jsonl(timing_path)
        if not records:
            continue
        last = records[-1]
        run_id = run_dir.name
        timestamp = _parse_timestamp_from_run_id(run_id)
        cfg = _load_runner_cfg(run_dir, checkpoints_dir)
        hook_count = _infer_hook_count(run_dir, cfg)
        raw_runs.append(
            _RunRaw(
                run_id=run_id,
                timestamp=timestamp,
                run_dir=run_dir,
                steps=len(records),
                final_samples=int(last.get("n_training_samples", 0)),
                elapsed_s=float(last.get("elapsed_s", 0.0)),
                cfg=cfg,
                hook_count=hook_count,
                sort_key=_timestamp_sort_key(timestamp),
            )
        )

    raw_runs.sort(key=lambda run: run.sort_key)
    _fill_missing_hook_counts(raw_runs)
    return raw_runs


def build_report(
    results_dir: Path | str,
    *,
    checkpoints_dir: Path | str | None = None,
    command_template: list[str] | None = None,
    min_steps: int = 32,
    min_completion_ratio: float = 0.98,
) -> Report:
    results_path = Path(results_dir)
    checkpoints_path = Path(checkpoints_dir) if checkpoints_dir is not None else None
    template = command_template or DEFAULT_COMMAND_TEMPLATE

    raw_runs = _collect_raw_runs(results_path, checkpoints_path)
    by_hook_count: dict[int, list[_RunRaw]] = {}
    for run in raw_runs:
        by_hook_count.setdefault(run.hook_count or 0, []).append(run)

    groups: dict[int, HookGroupReport] = {}
    all_rows: list[RunRow] = []

    for hook_count in sorted(by_hook_count):
        runs = sorted(by_hook_count[hook_count], key=lambda run: run.sort_key)
        training_token_candidates = [
            int(run.cfg["training_tokens"])
            for run in runs
            if run.cfg is not None and isinstance(run.cfg.get("training_tokens"), int)
        ]
        ref_training_tokens = max(training_token_candidates) if training_token_candidates else None

        rows: list[RunRow] = []
        for idx, run in enumerate(runs, start=1):
            scenario = template[idx - 1] if idx <= len(template) else f"extra#{idx}"
            cfg = run.cfg or {}
            training_tokens_raw = cfg.get("training_tokens")
            training_tokens = (
                int(training_tokens_raw) if isinstance(training_tokens_raw, int) and training_tokens_raw > 0 else None
            )
            local_completion_ratio = (
                run.final_samples / training_tokens if training_tokens else None
            )
            complete = run.steps >= min_steps
            notes: list[str] = []
            if run.steps < min_steps:
                notes.append("short_steps")
            if local_completion_ratio is not None and local_completion_ratio < min_completion_ratio:
                complete = False
                notes.append("under_target_tokens")
            if run.cfg is None:
                notes.append("missing_cfg")

            local_tokens_per_s = 0.0
            normalized_tokens_per_s = 0.0
            if run.elapsed_s > 0:
                local_tokens_per_s = run.final_samples / run.elapsed_s
                if (
                    training_tokens is not None
                    and ref_training_tokens is not None
                    and training_tokens > 0
                ):
                    normalized_tokens_per_s = (
                        run.final_samples * (ref_training_tokens / training_tokens)
                    ) / run.elapsed_s
                else:
                    normalized_tokens_per_s = local_tokens_per_s

            row = RunRow(
                run_id=run.run_id,
                timestamp=run.timestamp,
                hook_count=hook_count,
                slot=idx,
                scenario=scenario,
                steps=run.steps,
                final_samples=run.final_samples,
                elapsed_s=run.elapsed_s,
                complete=complete,
                local_completion_ratio=local_completion_ratio,
                local_tokens_per_s=local_tokens_per_s,
                normalized_tokens_per_s=normalized_tokens_per_s,
                training_tokens=training_tokens,
                notes=tuple(notes),
            )
            rows.append(row)
            all_rows.append(row)

        groups[hook_count] = HookGroupReport(hook_count=hook_count, rows=rows)

    return Report(groups=groups, rows=all_rows)
