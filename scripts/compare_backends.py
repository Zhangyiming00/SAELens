#!/usr/bin/env python3
"""
Compare activations across TP=1, TP=2, and HookedTransformer.
Each backend runs in its own subprocess. Reports pairwise MSE and cosine sim.
"""
import gc
import json
import os
import subprocess
import sys

import torch
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

MODEL_PATH = "/data/models/Llama-3.1-8B"
HOOKED_NAME = "meta-llama/Llama-3.1-8B"
LAYER = 15
BS, SEQ = 4, 128
HOOKS = [
    "hook_embed",
    f"blocks.{LAYER}.hook_resid_pre",
    f"blocks.{LAYER}.hook_resid_post",
    f"blocks.{LAYER}.hook_attn_out",
    f"blocks.{LAYER}.hook_mlp_out",
]


def worker_vllm(tp: int, tokens_path: str, out_path: str) -> None:
    from sae_lens.vllm_model import HookedVLLMModel

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokens = torch.load(tokens_path, weights_only=True)

    model = HookedVLLMModel(
        MODEL_PATH, tok,
        tensor_parallel_size=tp,
        max_model_len=SEQ + 8,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
    )
    _, caps = model.run_with_cache(tokens, names_filter=HOOKS)
    out = {k: v.float().cpu().tolist() for k, v in caps.items()}
    with open(out_path, "w") as f:
        json.dump(out, f)


def worker_hooked(tokens_path: str, out_path: str) -> None:
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    tokens = torch.load(tokens_path, weights_only=True)

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = HookedTransformer.from_pretrained_no_processing(
        HOOKED_NAME, hf_model=hf, tokenizer=tok,
        device="cuda", dtype=torch.bfloat16, local_files_only=True,
    )
    model.eval()
    del hf
    gc.collect()

    stop = LAYER + 1
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens.cuda(), names_filter=HOOKS,
            stop_at_layer=stop, prepend_bos=False,
            return_cache_object=False,
        )
    out = {k: cache[k].float().cpu().tolist() for k in HOOKS if k in cache}
    with open(out_path, "w") as f:
        json.dump(out, f)


def main() -> None:
    import tempfile

    torch.manual_seed(42)
    tokens = torch.randint(100, 30000, (BS, SEQ))

    with tempfile.TemporaryDirectory() as tmp:
        tok_path = f"{tmp}/tokens.pt"
        torch.save(tokens, tok_path)

        backends = {"tp1": None, "tp2": None, "hooked": None}
        for name in backends:
            out = f"{tmp}/{name}.json"
            if name.startswith("tp"):
                tp = int(name[2:])
                cmd = [sys.executable, __file__, "--worker-vllm", str(tp), tok_path, out]
            else:
                cmd = [sys.executable, __file__, "--worker-hooked", tok_path, out]

            print(f"Running {name} ...", flush=True)
            ret = subprocess.run(cmd, check=False,
                                 capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"  {name} FAILED:\n{ret.stderr[-1000:]}")
                continue
            with open(out) as f:
                raw = json.load(f)
            backends[name] = {k: torch.tensor(v) for k, v in raw.items()}
            print(f"  {name} done.", flush=True)

    # Pairwise comparison
    pairs = [("tp1", "tp2"), ("tp1", "hooked"), ("tp2", "hooked")]
    for hook in HOOKS:
        print(f"\n{'─'*70}")
        print(f"  {hook}")
        print(f"  {'Pair':<16} {'MSE':>12} {'cos_min':>10} {'max_abs_err':>13}")
        print(f"  {'─'*14} {'─'*12} {'─'*10} {'─'*13}")
        for a_name, b_name in pairs:
            a = backends.get(a_name)
            b = backends.get(b_name)
            if a is None or b is None or hook not in a or hook not in b:
                print(f"  {a_name} vs {b_name:<8}  (missing)")
                continue
            ta = a[hook].float()
            tb = b[hook].float()
            mse = torch.nn.functional.mse_loss(ta, tb).item()
            cos = torch.nn.functional.cosine_similarity(
                ta.view(-1, ta.shape[-1]), tb.view(-1, tb.shape[-1]), dim=-1
            ).min().item()
            mae = (ta - tb).abs().max().item()
            print(f"  {a_name} vs {b_name:<8}  {mse:>12.6f} {cos:>10.6f} {mae:>13.6f}")

    print()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker-vllm"]:
        worker_vllm(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif sys.argv[1:2] == ["--worker-hooked"]:
        worker_hooked(sys.argv[2], sys.argv[3])
    else:
        main()
