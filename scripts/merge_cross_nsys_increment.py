#!/usr/bin/env python3
"""Merge an incremental nsys sweep into the canonical cross-sweep results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--increment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    shutil.copytree(args.base, args.output)

    base_manifest = {
        json.loads(line)["run_id"]: json.loads(line)
        for line in (args.base / "run_manifest.jsonl").read_text().splitlines()
        if line
    }
    base_status = {
        json.loads(line)["run_id"]: json.loads(line)
        for line in (args.base / "run_status.jsonl").read_text().splitlines()
        if line
    }
    for name in ("run_manifest.jsonl", "run_status.jsonl"):
        merged = base_manifest if name == "run_manifest.jsonl" else base_status
        for line in (args.increment / name).read_text().splitlines():
            if line:
                record = json.loads(line)
                merged[record["run_id"]] = record
        (args.output / name).write_text(
            "".join(json.dumps(record) + "\n" for record in merged.values())
        )

    for line in (args.increment / "run_manifest.jsonl").read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        run_id = record["run_id"]
        src = args.increment / "runner_outputs" / run_id
        dst = args.output / "runner_outputs" / run_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        src_log = args.increment / "logs" / f"{run_id}.log"
        dst_log = args.output / "logs" / f"{run_id}.log"
        shutil.copy2(src_log, dst_log)

        src_nsys = args.increment / "nsys" / f"{run_id}.sqlite"
        dst_nsys = args.output / "nsys" / src_nsys.name
        dst_nsys.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_nsys, dst_nsys)
        src_rep = args.increment / "nsys" / f"{run_id}.nsys-rep"
        if src_rep.exists():
            shutil.copy2(src_rep, args.output / "nsys" / src_rep.name)

    print(f"wrote merged sweep to {args.output}")


if __name__ == "__main__":
    main()
