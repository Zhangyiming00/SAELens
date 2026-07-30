"""Aggregate per-case KV-pool verification summaries into one CSV.

Reads every ``<output-dir>/<case>/summary.json`` and emits the results table
described in section IX: one row per case with requested/resolved config, pool
sizing, max used blocks, KV byte footprint before/max, init & run results,
preemptions, and the evidence-based dynamic-expansion verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COLUMNS = [
    "case",
    "requested_mbt",
    "resolved_mbt",
    "max_model_len",
    "batch",
    "context",
    "chunked",
    "pool_blocks",
    "max_used_blocks",
    "kv_bytes_before",
    "kv_bytes_max",
    "init_result",
    "run_result",
    "preemptions",
    "max_running",
    "kv_alloc_calls",
    "dynamic_expansion",
    "failure_stage",
    "error_type",
]


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    req = summary.get("requested", {})
    resolved = summary.get("resolved", {})
    life = summary.get("lifecycle", {})
    instr = summary.get("instrumentation", {})
    dyn = summary.get("dynamic_expansion_verdict", {})
    init_rec = life.get("after_engine_init", {})
    max_rec = life.get(
        "after_second_request", life.get("after_request_finish", {})
    )
    return {
        "case": summary.get("case_name"),
        "requested_mbt": req.get("requested_max_num_batched_tokens"),
        "resolved_mbt": resolved.get("resolved_max_num_batched_tokens"),
        "max_model_len": req.get("requested_max_model_len"),
        "batch": req.get("batch_size"),
        "context": req.get("context_len"),
        "chunked": req.get("enable_chunked_prefill"),
        "pool_blocks": resolved.get("resolved_num_gpu_blocks"),
        "max_used_blocks": instr.get("max_used_blocks"),
        "kv_bytes_before": init_rec.get("kv_storage_bytes"),
        "kv_bytes_max": max_rec.get("kv_storage_bytes"),
        "init_result": summary.get("init_result"),
        "run_result": summary.get("run_result"),
        "preemptions": instr.get("preemption_count"),
        "max_running": instr.get("max_concurrent_running_requests"),
        "kv_alloc_calls": instr.get("kv_physical_allocation_call_count"),
        "dynamic_expansion": dyn.get("dynamic_expansion"),
        "failure_stage": summary.get("failure_stage"),
        "error_type": summary.get("error_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/kv_pool_verification")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summaries = sorted(out_dir.glob("*/summary.json"))
    rows = [_row(json.loads(p.read_text())) for p in summaries]

    csv_path = out_dir / "all_cases.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to {csv_path}")
    # Pretty console table.
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLUMNS} if rows else {}
    if rows:
        print(" | ".join(c.ljust(widths[c]) for c in COLUMNS))
        for r in rows:
            print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
