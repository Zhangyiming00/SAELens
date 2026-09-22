#!/usr/bin/env python3
"""Check archived baseline hashes before applying the sharded TopK patch.

This check is read-only. A mismatch is not permission to overwrite newer code.
This local archive baseline is NOT a verified upstream HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/sharded_topk_baseline.json",
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read baseline manifest: {exc}")
    mismatches = 0
    for name, expected in manifest["original_file_sha256"].items():
        path = args.repo / name
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            actual = None
        ok = actual == expected
        print(f'{"MATCH" if ok else "MISMATCH"}: {name}')
        mismatches += int(not ok)
    print(
        f'Archive baseline: {manifest["baseline_snapshot_date"]}; '
        'remote HEAD was NOT verified.'
    )
    if mismatches:
        print(
            "Different/missing files found. Review/rebase the patch; do not overwrite newer files."
        )
        return 1
    print("All modified original files match. Run git apply --check before applying.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
