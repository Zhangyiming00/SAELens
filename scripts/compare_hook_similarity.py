#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYER = 21
DEFAULT_DTYPE = "bfloat16"
DEFAULT_HOOK_TYPES = [
    "hook_resid_pre",
    "hook_resid_mid",
    "hook_resid_post",
    "hook_attn_out",
    "hook_mlp_out",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_z",
    "mlp.hook_pre",
    "mlp.hook_post",
]

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lightweight similarity check for vllm tp1/tp2 vs HookedTransformer."
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=DEFAULT_DTYPE)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--hooks", nargs="+", default=["all"])
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--skip-hooked", action="store_true")
    p.add_argument("--hooked-twice", action="store_true",
                   help="Run HookedTransformer twice (hooked_1 vs hooked_2) instead of cross-backend comparison.")
    p.add_argument("--save-json", default=None)
    p.add_argument("--_run-backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--token-batches-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--hook-types", nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def resolve_dataset_path(dataset_path: str) -> str:
    path = Path(dataset_path)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")


def is_hf_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "state.json").exists()


def build_token_batches(
    dataset_path: str,
    model_path: str,
    *,
    num_samples: int,
    batch_size: int,
    max_length: int,
) -> list[torch.Tensor]:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    path = Path(dataset_path)
    if not is_hf_dataset_dir(path):
        raise ValueError(
            f"{dataset_path} is not a datasets.load_from_disk directory. "
            "This lightweight script currently expects a tokenized HF dataset."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    dataset = load_from_disk(dataset_path)
    token_key = None
    for candidate in ("input_ids", "tokens", "token_ids"):
        if candidate in dataset.column_names:
            token_key = candidate
            break
    if token_key is None:
        raise ValueError(f"No token column found in {dataset_path}. Columns: {dataset.column_names}")

    rows: list[torch.Tensor] = []
    for row in dataset:
        tokens = row[token_key]
        if isinstance(tokens, torch.Tensor):
            seq = tokens.to(dtype=torch.long).flatten().cpu()
        else:
            seq = torch.tensor(tokens, dtype=torch.long).flatten().cpu()
        if seq.numel() == 0:
            continue
        rows.append(seq[:max_length])
        if len(rows) >= num_samples:
            break

    if not rows:
        raise ValueError(f"No usable token rows found in {dataset_path}")

    batches: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        max_len = max(int(seq.numel()) for seq in chunk)
        batch = torch.full((len(chunk), max_len), pad_token_id, dtype=torch.long)
        for idx, seq in enumerate(chunk):
            batch[idx, : seq.numel()] = seq
        batches.append(batch)
    return batches


def flatten_hook_activations(acts: torch.Tensor) -> torch.Tensor:
    if acts.ndim <= 3:
        return acts
    return acts.reshape(acts.shape[0], acts.shape[1], -1)


def capture_backend(
    backend: str,
    model_path: str,
    hooked_model_name: str,
    token_batches_path: str,
    hook_types: list[str],
    layer: int,
    device: str,
    dtype_name: str,
    out_path: str,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype(dtype_name)
    token_batches: list[torch.Tensor] = torch.load(token_batches_path, map_location="cpu")
    hook_names = [f"blocks.{layer}.{hook_type}" for hook_type in hook_types]
    stop_at_layer = layer + 1

    # hooked_1 / hooked_2 are aliases for two independent runs of the hooked backend
    effective_backend = "hooked" if backend in ("hooked_1", "hooked_2") else backend

    if effective_backend.startswith("vllm"):
        from sae_lens.vllm_model import HookedVLLMModel

        tp = 2 if effective_backend == "vllm_tp2" else 1
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        max_model_len = max(int(batch.shape[1]) for batch in token_batches) + 1
        model = HookedVLLMModel(
            model_path,
            tokenizer,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
        )
    else:
        from transformer_lens import HookedTransformer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
        )
        model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            local_files_only=True,
        )
        model.eval()

    results: dict[str, dict[str, Any]] = {}
    try:
        for hook_name in hook_names:
            outputs: list[torch.Tensor] = []
            error: str | None = None
            for batch_tokens in token_batches:
                try:
                    model_tokens = batch_tokens.to(device) if effective_backend == "hooked" else batch_tokens
                    with torch.inference_mode():
                        _, cache = model.run_with_cache(
                            model_tokens,
                            names_filter=[hook_name],
                            stop_at_layer=stop_at_layer,
                            prepend_bos=False,
                        )
                    acts = flatten_hook_activations(cache[hook_name]).float().cpu()
                    outputs.append(acts)
                except Exception as exc:
                    error = str(exc)
                    break
            if error is not None:
                results[hook_name] = {"error": error}
            else:
                tensor = torch.cat(outputs, dim=0)
                results[hook_name] = {
                    "shape": list(tensor.shape),
                    "acts": tensor,
                }
    finally:
        del model
        if effective_backend == "hooked":
            del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    torch.save({"backend": backend, "hooks": results}, out_path)


def launch_backend(
    backend: str,
    args: argparse.Namespace,
    token_batches_path: str,
    hook_types: list[str],
    out_path: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        __file__,
        "--_run-backend",
        backend,
        "--model-path",
        args.model_path,
        "--hooked-model-name",
        args.hooked_model_name,
        "--dataset-path",
        args.dataset_path,
        "--layer",
        str(args.layer),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--token-batches-path",
        token_batches_path,
        "--hook-types",
        *hook_types,
        "--out-path",
        out_path,
    ]
    print(f"[{backend}] capturing {len(hook_types)} hooks ...", flush=True)
    proc = subprocess.run(cmd, check=False, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed with exit code {proc.returncode}")
    return torch.load(out_path, map_location="cpu")


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    cosine = F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item() * 100.0
    rel_l2 = (
        (a_flat - b_flat).square().sum().sqrt()
        / b_flat.square().sum().sqrt().clamp(min=1e-12)
    ).item() * 100.0
    close_pct = torch.isclose(a_flat, b_flat, atol=5e-2, rtol=5e-2).float().mean().item() * 100.0
    return {
        "mean_cosine_pct": cosine,
        "relative_l2_error_pct": rel_l2,
        "close_element_pct": close_pct,
    }


def print_report(results: list[dict[str, Any]], hook_types: list[str], layer: int) -> dict[str, Any]:
    backend_names = [result["backend"] for result in results]
    summary: dict[str, Any] = {"layer": layer, "hooks": {}}

    print()
    print("=" * 120)
    print(f"  Hook Similarity — layer {layer}")
    print("=" * 120)

    for hook_type in hook_types:
        hook_name = f"blocks.{layer}.{hook_type}"
        print(f"\n{hook_name}")
        per_backend = [result["hooks"].get(hook_name, {}) for result in results]
        hook_summary: dict[str, Any] = {"backend_shapes": {}, "pairs": {}}

        for backend_name, info in zip(backend_names, per_backend, strict=True):
            if "error" in info:
                print(f"  {backend_name:<10} ERROR: {info['error']}")
                hook_summary["backend_shapes"][backend_name] = {"error": info["error"]}
            else:
                shape = info["shape"]
                print(f"  {backend_name:<10} shape={shape}")
                hook_summary["backend_shapes"][backend_name] = {"shape": shape}

        print("  pairwise")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                left = backend_names[i]
                right = backend_names[j]
                left_info = per_backend[i]
                right_info = per_backend[j]
                pair_name = f"{left}_vs_{right}"
                if "error" in left_info or "error" in right_info:
                    print(f"    {left:<10} vs {right:<10} skipped (backend error)")
                    hook_summary["pairs"][pair_name] = {"error": "backend error"}
                    continue
                if tuple(left_info["shape"]) != tuple(right_info["shape"]):
                    print(
                        f"    {left:<10} vs {right:<10} skipped "
                        f"(shape mismatch: {left_info['shape']} vs {right_info['shape']})"
                    )
                    hook_summary["pairs"][pair_name] = {
                        "error": "shape mismatch",
                        "left_shape": left_info["shape"],
                        "right_shape": right_info["shape"],
                    }
                    continue
                metrics = pair_metrics(left_info["acts"], right_info["acts"])
                print(
                    f"    {left:<10} vs {right:<10} "
                    f"cos={metrics['mean_cosine_pct']:.4f}%  "
                    f"close={metrics['close_element_pct']:.4f}%  "
                    f"rel_l2={metrics['relative_l2_error_pct']:.4f}%"
                )
                hook_summary["pairs"][pair_name] = metrics

        summary["hooks"][hook_name] = hook_summary

    return summary


def main() -> None:
    args = parse_args()

    if args._run_backend is not None:
        capture_backend(
            backend=args._run_backend,
            model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            token_batches_path=args.token_batches_path,
            hook_types=args.hook_types,
            layer=args.layer,
            device=args.device,
            dtype_name=args.dtype,
            out_path=args.out_path,
        )
        return

    hook_types = DEFAULT_HOOK_TYPES if args.hooks == ["all"] else args.hooks
    dataset_path = resolve_dataset_path(args.dataset_path)

    print(f"Model:   {args.model_path}")
    print(f"Layer:   {args.layer}")
    print(f"Dataset: {dataset_path}")
    print(f"Hooks:   {hook_types}")
    print(f"Samples: {args.num_samples}  |  Batch size: {args.batch_size}  |  Max length: {args.max_length}")

    token_batches = build_token_batches(
        dataset_path,
        args.model_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Prepared {len(token_batches)} token batches.")

    backends = ["vllm_tp1"]
    if not args.skip_tp2:
        backends.append("vllm_tp2")
    if not args.skip_hooked:
        backends.append("hooked")

    if args.hooked_twice:
        backends = ["hooked_1", "hooked_2"]

    with tempfile.TemporaryDirectory(prefix="hook_similarity_") as tmp_dir:
        token_batches_path = f"{tmp_dir}/token_batches.pt"
        torch.save(token_batches, token_batches_path)

        results = []
        for backend in backends:
            out_path = f"{tmp_dir}/{backend}.pt"
            results.append(
                launch_backend(
                    backend,
                    args,
                    token_batches_path,
                    hook_types,
                    out_path,
                )
            )

    summary = print_report(results, hook_types, args.layer)

    if args.save_json:
        output = {
            "model_path": args.model_path,
            "hooked_model_name": args.hooked_model_name,
            "dataset_path": dataset_path,
            "layer": args.layer,
            "hook_types": hook_types,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "backends": backends,
            "summary": summary,
        }
        Path(args.save_json).write_text(
            json.dumps(output, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
