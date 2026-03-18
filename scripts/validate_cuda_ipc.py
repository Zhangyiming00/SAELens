#!/usr/bin/env python3
"""
Validate correctness of CUDA IPC activation capture for TP=1 and TP=2.
Each TP config runs in its own subprocess for clean GPU memory isolation.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

MODEL_PATH = "/data/models/Llama-3.1-8B"
LAYER = 2
BS, SEQ = 3, 32

HOOKS = [
    "hook_embed",
    f"blocks.{LAYER}.hook_resid_pre",
    f"blocks.{LAYER}.hook_resid_mid",
    f"blocks.{LAYER}.hook_resid_post",
    f"blocks.{LAYER}.hook_attn_out",
    f"blocks.{LAYER}.hook_mlp_out",
]


# ---------------------------------------------------------------------------
# HF reference (run once in main process)
# ---------------------------------------------------------------------------

def collect_hf(hf_model, tokens: torch.Tensor, layer: int) -> dict[str, list]:
    caps: dict[str, torch.Tensor] = {}
    handles = []

    def reg(m, fn, pre=False):
        h = m.register_forward_pre_hook(fn) if pre else m.register_forward_hook(fn)
        handles.append(h)

    reg(hf_model.model.embed_tokens,
        lambda m, i, o: caps.__setitem__("hook_embed", o.detach().float().cpu()))
    reg(hf_model.model.layers[layer],
        lambda m, a: caps.__setitem__(f"blocks.{layer}.hook_resid_pre",
                                      a[0].detach().float().cpu()),
        pre=True)
    reg(hf_model.model.layers[layer],
        lambda m, i, o: caps.__setitem__(
            f"blocks.{layer}.hook_resid_post",
            (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
    reg(hf_model.model.layers[layer].self_attn,
        lambda m, i, o: caps.__setitem__(
            f"blocks.{layer}.hook_attn_out",
            (o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
    reg(hf_model.model.layers[layer].mlp,
        lambda m, i, o: caps.__setitem__(
            f"blocks.{layer}.hook_mlp_out", o.detach().float().cpu()))

    with torch.no_grad():
        hf_model(tokens.cuda())
    for h in handles:
        h.remove()

    pre = caps.get(f"blocks.{layer}.hook_resid_pre")
    attn = caps.get(f"blocks.{layer}.hook_attn_out")
    if pre is not None and attn is not None:
        caps[f"blocks.{layer}.hook_resid_mid"] = pre + attn

    return {k: v.tolist() for k, v in caps.items()}


# ---------------------------------------------------------------------------
# Worker: runs in subprocess, compares against serialised HF reference
# ---------------------------------------------------------------------------

def _worker(tp: int, hf_json_path: str, tokens_path: str, out_path: str) -> None:
    """Load vLLM model, run run_with_cache, compare against HF, write JSON results."""
    import json as _json
    import torch as _torch
    from sae_lens.vllm_model import HookedVLLMModel
    from transformers import AutoTokenizer as _Tok

    tok = _Tok.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokens = _torch.load(tokens_path, weights_only=True)

    with open(hf_json_path) as f:
        hf_raw = _json.load(f)
    hf_caps = {k: _torch.tensor(v) for k, v in hf_raw.items()}

    model = HookedVLLMModel(
        MODEL_PATH, tok,
        tensor_parallel_size=tp,
        max_model_len=SEQ + 8,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
    )
    _, vllm_caps = model.run_with_cache(tokens, names_filter=HOOKS)

    rows = []
    all_cuda = all(t.is_cuda for t in vllm_caps.values())
    for hook in HOOKS:
        if hook not in hf_caps:
            continue
        vt = vllm_caps[hook].float().cpu()
        ht = hf_caps[hook].float()
        shape_ok = (ht.shape == vt.shape)
        if not shape_ok:
            rows.append({"hook": hook, "ok": False,
                         "error": f"shape vllm={list(vt.shape)} hf={list(ht.shape)}"})
            continue
        mse = _torch.nn.functional.mse_loss(vt, ht).item()
        cos = _torch.nn.functional.cosine_similarity(
            vt.view(-1, vt.shape[-1]), ht.view(-1, ht.shape[-1]), dim=-1
        ).min().item()
        rows.append({"hook": hook, "mse": mse, "cos": cos, "shape": list(vt.shape),
                     "ok": cos > 0.99})

    with open(out_path, "w") as f:
        _json.dump({"all_cuda": all_cuda, "rows": rows}, f)


# ---------------------------------------------------------------------------
# Main: orchestrate, print results
# ---------------------------------------------------------------------------

def main() -> None:
    import tempfile
    import pathlib

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    torch.manual_seed(42)
    tokens = torch.randint(100, 30000, (BS, SEQ))

    print("Collecting HF reference …")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
    ).cuda().eval()
    hf_caps = collect_hf(hf, tokens, LAYER)
    del hf; gc.collect(); torch.cuda.empty_cache()
    print("HF reference done, GPU freed.\n")

    with tempfile.TemporaryDirectory() as tmp:
        hf_path = f"{tmp}/hf.json"
        tok_path = f"{tmp}/tokens.pt"
        with open(hf_path, "w") as f:
            json.dump(hf_caps, f)
        torch.save(tokens, tok_path)

        overall_ok = True
        for tp in [1, 2]:
            out_path = f"{tmp}/tp{tp}.json"
            cmd = [
                sys.executable, __file__,
                "--_worker", str(tp), hf_path, tok_path, out_path,
            ]
            print(f"{'='*70}")
            print(f"  Launching TP={tp} subprocess …")
            ret = subprocess.run(cmd, check=False)
            if ret.returncode != 0:
                print(f"  TP={tp} subprocess CRASHED (exit {ret.returncode})")
                overall_ok = False
                continue

            with open(out_path) as f:
                data = json.load(f)

            cuda_ok = data["all_cuda"]
            print(f"  TP={tp}  all tensors on CUDA: {'✓' if cuda_ok else '✗'}")
            if tp == 1:
                print("    → apply_model in-process, Python objects returned directly (zero copy)")
            else:
                print("    → ForkingPickler → CUDA IPC handle bytes (~64B) → ZMQ → pickle.loads → zero-copy CUDA")
            print(f"\n  {'Hook':<40} {'MSE':>12} {'cos_sim_min':>12} {'shape'}")
            print("  " + "-" * 76)
            tp_ok = cuda_ok
            for row in data["rows"]:
                if "error" in row:
                    print(f"  {row['hook']:<40} {row['error']}")
                    tp_ok = False
                    continue
                ok = row["ok"]
                tp_ok = tp_ok and ok
                flag = "OK" if ok else "FAIL"
                print(f"  {row['hook']:<40} {row['mse']:>12.6f} {row['cos']:>12.6f}"
                      f"  {tuple(row['shape'])}  {flag}")
            overall_ok = overall_ok and tp_ok

    print(f"\n{'='*70}")
    print(f"  Result: {'ALL PASS' if overall_ok else 'SOME FAILED'}")
    print(f"{'='*70}")
    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--_worker":
        _worker(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
