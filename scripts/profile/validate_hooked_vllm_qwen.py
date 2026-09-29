"""Validate config-only Qwen with vLLM dummy weights and synthetic token IDs.

No pretrained weights or tokenizer are downloaded. Run separately with TP1,
TP2 (vLLM workers), or torchrun TP2 (external launcher). GPU activations are
computed by the full-size model, never injected into the capture cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def download_configs(repo_id, directory):
    from modelscope.hub.file_download import model_file_download

    for filename in ("config.json", "generation_config.json"):
        model_file_download(repo_id, filename, revision="master", local_dir=str(directory))


def install_counters(model):
    model._validation_layer_calls = [0] * len(model.model.layers)
    handles = []
    for i, layer in enumerate(model.model.layers):
        def count(_module, _inputs, i=i):
            model._validation_layer_calls[i] += 1
        handles.append(layer.register_forward_pre_hook(count))
    model._validation_handles = handles
    return dict(architecture=type(model).__name__, layers=len(model.model.layers),
                parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()))


def reset_counters(model):
    model._validation_layer_calls[:] = [0] * len(model.model.layers)


def read_counters(model):
    import torch

    torch.cuda.synchronize()
    return dict(layer_calls=list(model._validation_layer_calls),
                capture_clean=not hasattr(model, "_sae_captures") and not hasattr(model, "_sae_handles"),
                stop_clean=not hasattr(model.model, "_sae_stop_at_layer"),
                allocated=torch.cuda.memory_allocated(), peak_allocated=torch.cuda.max_memory_allocated())


def synchronize_worker(model):
    import torch
    torch.cuda.synchronize()


def remove_counters(model):
    for handle in model._validation_handles:
        handle.remove()
    del model._validation_handles, model._validation_layer_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/models/Qwen3-8B"))
    parser.add_argument("--download-config", help="ModelScope repo ID; downloads only two config JSONs and exits")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--external-launcher", action="store_true",
                        help="Launch this probe with torchrun to exercise streaming's TP path")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/qwen_early_stop_20260928/tp1.json"))
    args = parser.parse_args()
    if args.download_config:
        download_configs(args.download_config, args.model)
        return
    if args.external_launcher and "RANK" not in os.environ:
        argv = [value for value in sys.argv[1:] if value != "--external-launcher"]
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={args.tp}", str(Path(__file__).resolve()), *argv],
                       check=True, timeout=240)
        return
    # A local config-only directory and dummy loader must work without any Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from sae_lens.vllm_model import ARCH_CONFIGS, HookedVLLMModel

    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    cfg = json.loads((args.model / "config.json").read_text())
    file_manifest = {str(p.relative_to(args.model)): dict(bytes=p.stat().st_size,
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in args.model.rglob("*") if p.is_file()}
    assert all(not name.endswith((".safetensors", ".bin", ".pt", ".pth"))
               for name in file_manifest), "Use a config-only model directory"
    tokens = torch.randint(100, min(30000, cfg["vocab_size"]), (args.batch, args.seq_len),
                          generator=torch.Generator().manual_seed(42))
    model = HookedVLLMModel(
        str(args.model), tokenizer=None, load_format="dummy", skip_tokenizer_init=True,
        generation_config="vllm", tensor_parallel_size=args.tp,
        capture_batch_size=args.batch, capture_context_size=args.seq_len,
        max_model_len=args.seq_len + 1, max_num_seqs=args.batch,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=.85, enforce_eager=True, enable_prefix_caching=False,
        disable_log_stats=True, seed=42,
    )
    depth, early = cfg["num_hidden_layers"], min(2, cfg["num_hidden_layers"] - 1)
    hooks = ["hook_embed"] + [f"blocks.{early}.{kind}" for kind in ARCH_CONFIGS[model._arch] if kind != "hook_embed"]
    report = dict(model=str(args.model), config=cfg, files=file_manifest, tp=args.tp,
                  external_launcher=model._is_external_launcher, batch=args.batch, seq_len=args.seq_len,
                  max_num_batched_tokens=args.max_num_batched_tokens, weights="vllm dummy",
                  inputs="synthetic token IDs", cases=[])
    try:
        report["workers"] = model.llm.apply_model(install_counters)
        # Warm scheduler/JIT once before comparing captures or wall-clock timing.
        model.run_with_cache(tokens, hooks, stop_at_layer=None)
        references = {}
        cases = [("full_before", hooks, None), ("explicit_early", hooks, early + 1),
                 ("auto_early", hooks, "auto"), ("default_early", hooks, "auto"),
                 ("full_after", hooks, None),
                 ("middle_auto", [f"blocks.{depth // 2}.hook_resid_post"], "auto"),
                 ("middle_full", [f"blocks.{depth // 2}.hook_resid_post"], None),
                 ("last_auto", [f"blocks.{depth - 1}.hook_resid_post"], "auto"),
                 ("embedding_auto", ["hook_embed"], "auto")]
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        widths = {"attn.hook_q": cfg["num_attention_heads"] * head_dim,
                  "attn.hook_k": cfg["num_key_value_heads"] * head_dim,
                  "attn.hook_v": cfg["num_key_value_heads"] * head_dim,
                  "attn.hook_z": cfg["num_attention_heads"] * head_dim,
                  "mlp.hook_pre": cfg["intermediate_size"], "mlp.hook_post": cfg["intermediate_size"]}
        for label, names, stop in cases:
            model.llm.apply_model(reset_counters)
            timings = []
            comparisons = {}
            for repeat in range(args.repeats):
                model.llm.apply_model(synchronize_worker)
                started = time.perf_counter()
                stop_kwargs = {} if label == "default_early" else {"stop_at_layer": stop}
                _, cache = model.run_with_cache(tokens, names, **stop_kwargs)
                model.llm.apply_model(synchronize_worker)
                timings.append((time.perf_counter() - started) * 1000)
                for name, activation in cache.items():
                    kind = name.split(".", 2)[-1] if name.startswith("blocks.") else name
                    assert activation.shape == (args.batch, args.seq_len, widths.get(kind, cfg["hidden_size"])), (name, activation.shape)
                    assert torch.isfinite(activation).all(), name
                    value = activation.detach().cpu()
                    if name not in references:
                        references[name] = value.clone()
                    error = float((value.float() - references[name].float()).abs().max())
                    comparisons[name] = max(comparisons.get(name, 0.), error)
                    torch.testing.assert_close(value, references[name], rtol=0, atol=0)
                # The loop variable also owns an IPC view in multiprocess TP.
                del activation, cache
            workers = model.llm.apply_model(read_counters)
            expected = depth if stop is None else (max([int(n.split('.')[1]) for n in names if n.startswith('blocks.')] or [0]) + 1 if stop == "auto" else stop)
            for worker in workers:
                assert worker["capture_clean"] and worker["stop_clean"], worker
                assert all(n > 0 for n in worker["layer_calls"][:expected]), worker
                assert not any(worker["layer_calls"][expected:]), worker
            report["cases"].append(dict(name=label, stop_at_layer=stop, expected_executed_layers=expected,
                                        timings_ms=timings, median_ms=statistics.median(timings),
                                        max_abs_errors=comparisons, workers=workers))
            print(f"PASS {label}: {expected}/{depth} layers, {statistics.median(timings):.2f} ms", flush=True)
        report["status"] = "PASS"
    finally:
        # Preserve partial evidence if an assertion or engine call fails.
        path = args.output
        if "RANK" in os.environ:
            path = path.with_name(f"{path.stem}_rank{os.environ['RANK']}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        model.llm.apply_model(remove_counters)
        model.close()
        # This standalone probe owns its world; the production wrapper does not.
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
