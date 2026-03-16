#!/usr/bin/env python
"""
Validate the complete training data flow across three model backends:
  - HookedTransformer (reference)
  - HookedVLLMModel   TP=1
  - HookedVLLMModel   TP=2

Uses the exact same call signature as ActivationsStore.get_activations():
    model.run_with_cache(
        batch_tokens,
        names_filter=[hook_name],
        stop_at_layer=extract_stop_at_layer_from_tlens_hook_name(hook_name),
        prepend_bos=False,
    )

Each vLLM backend runs in its own subprocess so GPU memory is fully released
between runs (the parent process's live allocations otherwise block TP=2 init).

Pairwise comparisons reported:
  vLLM-TP1  vs HookedTransformer
  vLLM-TP2  vs HookedTransformer
  vLLM-TP1  vs vLLM-TP2
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

# ---- sub-command: capture one vLLM backend and save .pt file ----

def _capture_vllm_subprocess(
    model_path: str,
    dataset_path: str,
    tp: int,
    num_samples: int,
    batch_size: int,
    max_length: int,
    layer: int,
    hook_types: list[str],
    out_path: str,
) -> None:
    """Run inside a subprocess: capture vLLM activations, save to out_path."""
    import gc

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name
    from sae_lens.vllm_model import HookedVLLMModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    token_batches = _load_batches(dataset_path, num_samples, batch_size, max_length, pad_id)
    hook_names = [f"blocks.{layer}.{h}" for h in hook_types]

    effective_max = max(max_length, max(int(b.shape[1]) for b in token_batches)) + 1
    model = HookedVLLMModel(
        model_path, tokenizer,
        tensor_parallel_size=tp,
        max_model_len=effective_max,
    )
    try:
        all_acts: dict[str, list[torch.Tensor]] = {h: [] for h in hook_names}
        stop = extract_stop_at_layer_from_tlens_hook_name(hook_names[0])
        for batch in token_batches:
            _, cache = model.run_with_cache(
                batch,
                names_filter=hook_names,
                stop_at_layer=stop,
                prepend_bos=False,
            )
            for h in hook_names:
                all_acts[h].append(cache[h].cpu())
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    result = {h: torch.cat(all_acts[h], dim=0) for h in hook_names}
    torch.save(result, out_path)
    print(f"[vLLM TP={tp}] saved {out_path}")


def _load_batches(
    dataset_path: str,
    num_samples: int,
    batch_size: int,
    max_length: int,
    pad_id: int,
) -> list[torch.Tensor]:
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    for key in ("input_ids", "tokens", "token_ids"):
        if key in dataset.column_names:
            break
    else:
        raise ValueError(f"No token column in {dataset_path}")

    batches: list[torch.Tensor] = []
    cur: list[torch.Tensor] = []
    for i, row in enumerate(dataset):
        if i >= num_samples:
            break
        ids = torch.tensor(list(row[key]), dtype=torch.long)[:max_length]
        cur.append(ids)
        if len(cur) == batch_size:
            ml = max(t.numel() for t in cur)
            b = torch.full((len(cur), ml), pad_id, dtype=torch.long)
            for j, t in enumerate(cur):
                b[j, : t.numel()] = t
            batches.append(b)
            cur = []
    if cur:
        ml = max(t.numel() for t in cur)
        b = torch.full((len(cur), ml), pad_id, dtype=torch.long)
        for j, t in enumerate(cur):
            b[j, : t.numel()] = t
        batches.append(b)
    return batches


# ---- metrics ----

def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float(), b.float()
    fa = a.reshape(-1, a.shape[-1])
    fb = b.reshape(-1, b.shape[-1])
    cos = F.cosine_similarity(fa, fb, dim=-1)
    diff = fa - fb
    sq_err = diff.square().sum().item()
    sq_ref = fb.square().sum().item()
    n = fb.numel()
    rel_l2 = math.sqrt(sq_err / max(sq_ref, 1e-12))
    ref_rms = math.sqrt(sq_ref / max(n, 1))
    pred_rms = math.sqrt(fa.square().sum().item() / max(n, 1))
    rel_rms = abs(pred_rms - ref_rms) / max(ref_rms, 1e-12)
    return {
        "cosine_pct": cos.mean().item() * 100.0,
        "rel_l2_pct": rel_l2 * 100.0,
        "rel_rms_pct": rel_rms * 100.0,
        "max_abs_diff": diff.abs().max().item(),
    }


def print_comparison(
    label: str,
    acts_a: dict[str, torch.Tensor],
    acts_b: dict[str, torch.Tensor],
    hook_names: list[str],
) -> None:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"{'hook':<20} {'cosine%':>8} {'rel_L2%':>8} {'rel_rms%':>9} {'max_abs':>10}")
    print("-" * 65)
    for h in hook_names:
        m = compare(acts_a[h], acts_b[h])
        print(
            f"{h.split('.')[-1]:<20} {m['cosine_pct']:>8.4f} {m['rel_l2_pct']:>8.4f}"
            f" {m['rel_rms_pct']:>9.4f} {m['max_abs_diff']:>10.6f}"
        )


# ---- HookedTransformer capture (in main process, last) ----

def capture_hooked_transformer(
    token_batches: list[torch.Tensor],
    hook_names: list[str],
    *,
    local_model_path: str,
    hooked_model_name: str,
    tokenizer,
    dtype: torch.dtype,
    device: str,
) -> dict[str, torch.Tensor]:
    import gc

    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name

    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path, dtype=dtype, local_files_only=True
    )
    model = HookedTransformer.from_pretrained_no_processing(
        hooked_model_name, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=dtype, local_files_only=True,
    )
    model.eval()
    stop = extract_stop_at_layer_from_tlens_hook_name(hook_names[0])
    try:
        all_acts: dict[str, list[torch.Tensor]] = {h: [] for h in hook_names}
        for batch in token_batches:
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    batch.to(device),
                    names_filter=hook_names,
                    stop_at_layer=stop,
                    prepend_bos=False,
                    return_cache_object=False,
                )
            for h in hook_names:
                all_acts[h].append(cache[h].detach().float().cpu())
    finally:
        del model, hf_model
        gc.collect()
        torch.cuda.empty_cache()

    return {h: torch.cat(all_acts[h], dim=0) for h in hook_names}


# ---- subprocess launcher ----

def run_vllm_subprocess(tp: int, out_path: str, args: argparse.Namespace) -> None:
    """Launch this same script in a fresh subprocess to capture vLLM activations."""
    cmd = [
        sys.executable, __file__,
        "--_capture-vllm",
        "--model-path", args.model_path,
        "--dataset-path", args.dataset_path,
        "--tp", str(tp),
        "--num-samples", str(args.num_samples),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--layer", str(args.layer),
        "--out-path", out_path,
    ]
    print(f"\n[{tp} GPU(s)] Launching subprocess: vLLM TP={tp} capture ...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"vLLM TP={tp} subprocess failed (exit {result.returncode})")


# ---- arg parsing ----

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_HOOK_TYPES = ("hook_resid_pre", "hook_attn_out", "hook_mlp_out", "hook_resid_post")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--skip-tp2", action="store_true")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cuda")
    # Internal sub-command flags (used by subprocess):
    p.add_argument("--_capture-vllm", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--tp", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--out-path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---- main ----

def main() -> None:
    args = parse_args()

    # Sub-command: called by subprocess to capture one vLLM backend.
    if args._capture_vllm:
        _capture_vllm_subprocess(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            tp=args.tp,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            layer=args.layer,
            hook_types=list(DEFAULT_HOOK_TYPES),
            out_path=args.out_path,
        )
        return

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    hook_names = [f"blocks.{args.layer}.{h}" for h in DEFAULT_HOOK_TYPES]

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    token_batches = _load_batches(
        args.dataset_path, args.num_samples, args.batch_size, args.max_length, pad_id
    )

    print(f"Model:         {args.model_path}")
    print(f"Dataset:       {args.dataset_path}")
    print(f"Layer:         {args.layer}")
    print(f"Samples:       {sum(int(b.shape[0]) for b in token_batches)}")
    print(f"Batches:       {len(token_batches)}")
    print(f"stop_at_layer: {extract_stop_at_layer_from_tlens_hook_name(hook_names[0])}")

    with tempfile.TemporaryDirectory(prefix="vllm_validate_") as tmp:
        tp1_path = f"{tmp}/tp1.pt"
        tp2_path = f"{tmp}/tp2.pt"

        # Each vLLM run is in a fresh subprocess so GPU memory is fully freed.
        run_vllm_subprocess(1, tp1_path, args)
        acts_tp1: dict[str, torch.Tensor] = torch.load(tp1_path, map_location="cpu")

        acts_tp2 = None
        if not args.skip_tp2:
            run_vllm_subprocess(2, tp2_path, args)
            acts_tp2 = torch.load(tp2_path, map_location="cpu")

    # HookedTransformer runs in the main process last (GPU now fully free).
    print("\n[HookedTransformer] Capturing activations...")
    acts_hooked = capture_hooked_transformer(
        token_batches, hook_names,
        local_model_path=args.model_path,
        hooked_model_name=args.hooked_model_name,
        tokenizer=tokenizer,
        dtype=dtype,
        device=args.device,
    )

    # --- Results ---
    print_comparison("vLLM TP=1  vs  HookedTransformer", acts_tp1, acts_hooked, hook_names)
    if acts_tp2 is not None:
        print_comparison("vLLM TP=2  vs  HookedTransformer", acts_tp2, acts_hooked, hook_names)
        print_comparison("vLLM TP=1  vs  vLLM TP=2 (direct)", acts_tp1, acts_tp2, hook_names)


if __name__ == "__main__":
    main()
