from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

from sae_lens.vllm_model import HookedVLLMModel


def _stats(x: torch.Tensor) -> dict[str, float | list[int]]:
    xf = x.detach().float()
    return {
        "shape": list(x.shape),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "abs_mean": float(xf.abs().mean().item()),
        "sq_sum_mean": float(xf.pow(2).sum(dim=-1).mean().item()),
    }


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_mismatch": 1.0,
            "a_numel": float(a.numel()),
            "b_numel": float(b.numel()),
        }
    af = a.detach().float()
    bf = b.detach().float()
    d = af - bf
    return {
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "rmse": float(d.pow(2).mean().sqrt().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Llama-3.1-8B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enable-prefix-caching", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocab_size = int(getattr(tokenizer, "vocab_size", 128000))
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.context_size),
        dtype=torch.long,
    )

    model = HookedVLLMModel(
        args.model,
        tokenizer,
        tensor_parallel_size=1,
        max_model_len=args.context_size + 1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=args.enable_prefix_caching,
        device="cuda:0",
    )

    hook16 = "blocks.16.hook_resid_post"
    hook31 = "blocks.31.hook_resid_post"

    runs: dict[str, dict[str, torch.Tensor]] = {}
    for label, hooks, stop in [
        ("single16_stop17", [hook16], 17),
        ("single16_stop32", [hook16], 32),
        ("single31_stop32", [hook31], 32),
        ("multi16_31_stop32", [hook16, hook31], 32),
    ]:
        _unused, cache = model.run_with_cache(
            tokens,
            names_filter=hooks,
            stop_at_layer=stop,
            prepend_bos=False,
        )
        runs[label] = {k: v.detach().cpu() for k, v in cache.items()}

    report = {
        label: {hook: _stats(value) for hook, value in cache.items()}
        for label, cache in runs.items()
    }
    report["comparisons"] = {
        "block16_stop17_vs_stop32": _diff(
            runs["single16_stop17"][hook16], runs["single16_stop32"][hook16]
        ),
        "block16_single_stop32_vs_multi_stop32": _diff(
            runs["single16_stop32"][hook16], runs["multi16_31_stop32"][hook16]
        ),
        "block31_single_stop32_vs_multi_stop32": _diff(
            runs["single31_stop32"][hook31], runs["multi16_31_stop32"][hook31]
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
