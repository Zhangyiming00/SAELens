#!/usr/bin/env python
"""
Diagnose why TP=2 is slower than TP=1 by decomposing run_with_cache() into
individually-timed sub-steps, using only public vLLM APIs (no vLLM source changes).

Sub-steps timed:
  T1: pure_generate   - llm.generate() with NO hooks (compute + NCCL only)
  T2: register        - apply_model(_register_hooks) IPC overhead
  T3: generate_hooked - llm.generate() with hooks registered (T1 + hook .detach().cpu())
  T4: collect         - apply_model(_collect_and_cleanup) IPC overhead

Relationships:
  NCCL overhead      = T1(TP2) - T1(TP1)
  hook_cpu overhead  = T3 - T1  (detach+cpu copy inside worker)
  IPC collect        = T4       (scales linearly with tensor size for TP=2)
  Total run_with_cache ≈ T2 + T3 + T4

Usage:
    python scripts/diagnose_tp_overhead.py \
        --model-path /data/models/Llama-3.1-8B \
        --tp 1          # run once for tp=1, once for tp=2
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from functools import partial

import torch

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"

# Fixed diagnostic config
LAYER = 15
HOOK = "hook_resid_post"
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
CONTEXT_SIZES = [128, 256, 512, 1024, 2048]
NUM_WARMUP = 2
NUM_MEASURE = 5


def _ms(t: float) -> str:
    return f"{t * 1000:>8.1f}ms"


def make_prompts(batch_tokens: torch.Tensor) -> list[dict]:
    return [{"prompt_token_ids": batch_tokens[i].tolist()} for i in range(len(batch_tokens))]


def run_diagnosis(model_path: str, tp: int) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

    from vllm import SamplingParams
    from transformers import AutoTokenizer

    from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name
    from sae_lens.vllm_model import (
        ARCH_CONFIGS,
        HookedVLLMModel,
        _collect_and_cleanup,
        _get_arch_name,
        _register_hooks,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    max_ctx = max(CONTEXT_SIZES)
    model = HookedVLLMModel(
        model_path, tokenizer,
        tensor_parallel_size=tp,
        max_model_len=max_ctx + 1,
        enable_prefix_caching=False,
    )
    llm = model.llm

    arch = llm.apply_model(_get_arch_name)[0]
    arch_config = ARCH_CONFIGS[arch]

    hook_name = f"blocks.{LAYER}.{HOOK}"
    stop = extract_stop_at_layer_from_tlens_hook_name(hook_name)
    hook_type = HOOK
    path_tpl, extractor, is_pre, gather_fn = arch_config[hook_type]
    path = path_tpl.format(layer=LAYER)
    hook_specs = [(hook_name, path, extractor, is_pre, gather_fn)]

    sampling = SamplingParams(max_tokens=1)

    print(f"\n{'='*90}")
    print(f"  TP={tp}  layer={LAYER}  hook={HOOK}  (d_model=4096)")
    print(f"  Sub-step breakdown  [all times = mean over {NUM_MEASURE} measured batches]")
    print(f"{'='*90}")
    header = (
        f"{'bs':>4} {'ctx':>5} {'tensor_mb':>9}"
        f" {'T1:pure_gen':>12} {'T2:register':>12} {'T3:gen+hook':>12}"
        f" {'T4:collect':>12} {'hook_cpu':>10} {'NCCL_extra':>11}"
    )
    print(header)
    print("-" * 90)

    # We need a TP=1 baseline for NCCL comparison; just note it for the user.
    for ctx in CONTEXT_SIZES:
        for bs in BATCH_SIZES:
            tokens = torch.full((bs, ctx), pad_id, dtype=torch.long)
            prompts = make_prompts(tokens)
            total_tokens = bs * ctx
            tensor_mb = total_tokens * 4096 * 2 / (1024 ** 2)

            register_fn = partial(
                _register_hooks,
                hook_specs=hook_specs,
                total_tokens=total_tokens,
                stop_at_layer=stop,
            )

            def measure(fn, n_warmup=NUM_WARMUP, n_measure=NUM_MEASURE):
                for _ in range(n_warmup):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_measure):
                    fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n_measure

            # T1: pure generate, no hooks
            t1 = measure(lambda: llm.generate(prompts, sampling, use_tqdm=False))

            # T2 + T3 + T4: register → generate → collect
            # We time each piece separately over multiple rounds.

            # T2: register only (hooks removed by collect, so we need to collect after each)
            def _register_and_collect():
                llm.apply_model(register_fn)
                llm.apply_model(_collect_and_cleanup)

            t2_plus_t4_empty = measure(_register_and_collect)
            # This is T2 + T4 with an empty capture (no generate in between, so tensors are tiny).
            # Use it as an approximation for T2 alone (collect of empty dict is fast).

            # T2: register alone (approximate - use first part of above)
            # Better: time register alone, then discard by collecting
            def _just_register():
                llm.apply_model(register_fn)
                llm.apply_model(_collect_and_cleanup)  # cleanup, don't count this

            # Time only the register call
            for _ in range(NUM_WARMUP):
                llm.apply_model(register_fn)
                llm.apply_model(_collect_and_cleanup)

            torch.cuda.synchronize()
            t2_total = 0.0
            t4_total = 0.0
            t3_total = 0.0
            for _ in range(NUM_MEASURE):
                # T2
                torch.cuda.synchronize()
                t_reg_start = time.perf_counter()
                llm.apply_model(register_fn)
                torch.cuda.synchronize()
                t2_total += time.perf_counter() - t_reg_start

                # T3: generate with hooks
                torch.cuda.synchronize()
                t_gen_start = time.perf_counter()
                llm.generate(prompts, sampling, use_tqdm=False)
                torch.cuda.synchronize()
                t3_total += time.perf_counter() - t_gen_start

                # T4: collect
                torch.cuda.synchronize()
                t_col_start = time.perf_counter()
                llm.apply_model(_collect_and_cleanup)
                torch.cuda.synchronize()
                t4_total += time.perf_counter() - t_col_start

            t2 = t2_total / NUM_MEASURE
            t3 = t3_total / NUM_MEASURE
            t4 = t4_total / NUM_MEASURE

            hook_cpu = t3 - t1   # detach+cpu inside worker
            # NCCL extra is only meaningful compared to TP=1 baseline; show T1 raw instead

            print(
                f"{bs:>4} {ctx:>5} {tensor_mb:>9.1f}"
                f" {_ms(t1)} {_ms(t2)} {_ms(t3)}"
                f" {_ms(t4)} {_ms(hook_cpu)} {_ms(t1)}"
            )

    print("-" * 90)
    print("Columns:")
    print("  T1:pure_gen  = llm.generate() no hooks (pure compute + NCCL if TP=2)")
    print("  T2:register  = apply_model(_register_hooks) IPC round-trip")
    print("  T3:gen+hook  = llm.generate() with hooks (.detach().cpu() inside worker)")
    print("  T4:collect   = apply_model(_collect_and_cleanup) — activation IPC transfer")
    print("  hook_cpu     = T3 - T1  (cost of .detach().cpu() for activation tensor)")
    print("  NCCL_extra   = T1 raw value — compare across TP=1 and TP=2 runs to see NCCL cost")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--tp", type=int, default=1, choices=[1, 2])
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--batch-sizes", default=",".join(str(x) for x in BATCH_SIZES))
    p.add_argument("--context-sizes", default=",".join(str(x) for x in CONTEXT_SIZES))
    p.add_argument("--num-warmup", type=int, default=NUM_WARMUP)
    p.add_argument("--num-measure", type=int, default=NUM_MEASURE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global LAYER, BATCH_SIZES, CONTEXT_SIZES, NUM_WARMUP, NUM_MEASURE
    LAYER = args.layer
    BATCH_SIZES = [int(x) for x in args.batch_sizes.split(",")]
    CONTEXT_SIZES = [int(x) for x in args.context_sizes.split(",")]
    NUM_WARMUP = args.num_warmup
    NUM_MEASURE = args.num_measure

    run_diagnosis(args.model_path, args.tp)


if __name__ == "__main__":
    main()
