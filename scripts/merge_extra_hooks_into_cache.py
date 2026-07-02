"""Verify row-alignment of newly-cached hooks against an existing split cache,
then merge the new hook dirs in and extend the manifest.

Alignment check: for every row, the new hooks' token_ids must equal the
reference hook's token_ids. If any row differs we abort without touching the
existing cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from sae_lens.training.multi_sae_trainer import sanitize_hook_name_for_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--existing-cache",
        default="/home/zhangyiming/datasets/cached_activation.new_20260509_140126",
    )
    p.add_argument("--new-cache", default="/home/zhangyiming/datasets/_extra_hooks_11_26_tmp")
    p.add_argument("--ref-hook", default="blocks.21.hook_resid_post")
    p.add_argument(
        "--new-hooks",
        default="blocks.11.hook_resid_post,blocks.26.hook_resid_post",
    )
    p.add_argument("--apply", action="store_true", help="Perform the merge (else dry-run).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    existing = Path(args.existing_cache)
    new_cache = Path(args.new_cache)
    new_hooks = [h.strip() for h in args.new_hooks.split(",") if h.strip()]

    ref_ds = load_from_disk(str(existing / sanitize_hook_name_for_path(args.ref_hook)))
    ref_tok = np.asarray(ref_ds["token_ids"], dtype=np.int64)
    print(f"Reference {args.ref_hook}: {ref_tok.shape}")

    for hook in new_hooks:
        nd = load_from_disk(str(new_cache / sanitize_hook_name_for_path(hook)))
        ntok = np.asarray(nd["token_ids"], dtype=np.int64)
        if ntok.shape != ref_tok.shape:
            raise SystemExit(f"ABORT: {hook} token shape {ntok.shape} != {ref_tok.shape}")
        if not np.array_equal(ntok, ref_tok):
            n_bad = int((ntok != ref_tok).any(axis=1).sum())
            raise SystemExit(f"ABORT: {hook} token_ids misaligned in {n_bad} rows")
        if hook not in nd.column_names:
            raise SystemExit(f"ABORT: {hook} missing activation column: {nd.column_names}")
        print(f"OK  {hook}: {ntok.shape} rows aligned, columns={nd.column_names}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to merge.")
        return

    manifest_path = existing / "cache_activations_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    for hook in new_hooks:
        dst_name = sanitize_hook_name_for_path(hook)
        src = new_cache / dst_name
        dst = existing / dst_name
        if dst.exists():
            raise SystemExit(f"ABORT: destination already exists: {dst}")
        shutil.copytree(src, dst)
        manifest["hook_names"].append(hook)
        manifest["hook_to_dir"][hook] = dst_name
        print(f"merged {hook} -> {dst}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nUpdated manifest: {manifest['hook_names']}")


if __name__ == "__main__":
    main()
