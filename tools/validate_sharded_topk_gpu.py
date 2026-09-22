#!/usr/bin/env python3
"""Native CUDA/Megatron acceptance and synthetic A/B benchmark.

No LLM or dataset download. The actual Megatron linears, runtime groups and DDP
main_grad hooks run here, with a dense torch Adam optimizer and model-level clip.
This is NOT an E2E activation-streaming or distributed-optimizer-overlap test.

Example:
 torchrun --standalone --nproc_per_node=4 tools/validate_sharded_topk_gpu.py \
   --tp 2 --dp 2 --global-batch 65 --hooks 2 --ga 2 --wavefront
"""

from __future__ import annotations

import os

os.environ.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")

import argparse
import gc
import json
import math
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from sae_lens.sae_runtime import SAERuntime
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.megatron_ddp import wrap_runtime_sae
from sae_lens.training.sae_train_unit import SAETrainUnit
from sae_lens.training.tp_wavefront import runtime_wavefront_forward


class NoFullLatent(TorchDispatchMode):
    def __init__(self, rows, width):
        self.rows, self.width = rows, width

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for x in tree_leaves(out):
            if (
                isinstance(x, torch.Tensor)
                and x.ndim >= 2
                and x.shape[-1] == self.width
                and math.prod(x.shape[:-1]) == self.rows
            ):
                raise AssertionError(
                    f"Full latent allocation: {func}, {tuple(x.shape)}"
                )
        return out


def run_mode(args, runtime, mode):
    context = runtime.require_local()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    hooks = [f"probe_{h}" for h in range(args.hooks)]
    units = {}
    for h, hook in enumerate(hooks):
        torch.manual_seed(100 + h)
        cfg = TopKTrainingSAEConfig(
            d_in=args.d_in,
            d_sae=args.d_sae,
            k=args.k,
            device=str(device),
            dtype="float32",
            topk_backend=mode,
            sparse_decoder_backend=args.decoder,
            topk_candidate_protocol=args.protocol,
            topk_key_backend=args.keys,
            rescale_acts_by_decoder_norm=not args.no_norm,
        )
        model = MegatronTopKSAE(cfg, runtime=runtime)
        wrapped = wrap_runtime_sae(
            model,
            runtime,
            gradient_accumulation_fusion=not (args.no_fusion or args.autocast),
            distributed_optimizer=False,
            bucket_cap_mb=1,
        )
        opt = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, fused=True
        )
        units[hook] = SAETrainUnit(hook, model, wrapped, opt, runtime)
    base, extra = divmod(args.global_batch, args.dp)
    local_n = base + int(context.dp_rank < extra)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    times = []
    losses = []
    peak = 0
    for step in range(args.warmup + args.steps):
        for unit in units.values():
            unit.zero_grad()
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for micro in range(args.ga):
            inputs = {}
            for h, hook in enumerate(hooks):
                gen = torch.Generator(device=device).manual_seed(
                    9000 + step * 1000 + micro * 100 + h * 10 + context.dp_rank
                )
                x = torch.randn(local_n, args.d_in, device=device, generator=gen)
                mask = (
                    (torch.arange(args.d_sae, device=device) % 3 == 0)
                    if args.aux
                    else None
                )
                inputs[hook] = TrainStepInput(
                    sae_in=x,
                    coefficients={},
                    dead_neuron_mask=mask,
                    n_training_steps=step,
                    is_logging_step=False,
                )
            guard = (
                NoFullLatent(local_n, args.d_sae)
                if mode != "legacy" and args.tp > 1 and step == 0
                else nullcontext()
            )
            autocast = torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.autocast
            )
            # Finish DP explicitly at the window boundary; early optimizer/
            # bucket overlap is left to the real training runner acceptance.
            with ExitStack() as stack, guard:
                for unit in units.values():
                    stack.enter_context(unit.no_sync())
                if args.wavefront and args.tp > 1 and args.hooks > 1:
                    outputs = runtime_wavefront_forward(
                        SimpleNamespace(autocast_if_enabled=autocast), units, inputs
                    )
                else:
                    with autocast:
                        outputs = {h: u.forward(inputs[h]) for h, u in units.items()}
                for h, u in units.items():
                    output = outputs.pop(h)
                    if mode != "legacy":
                        assert output.hidden_pre.shape[-1] == args.d_sae // args.tp
                        assert output.feature_acts.shape[-1] == args.d_sae // args.tp
                    losses.append(output.loss.detach())
                    u.backward(output.loss * local_n, scaler, sync_gradients=False)
                    del output
            del inputs, outputs
        for h, u in units.items():
            u.finish_window(args.global_batch * args.ga)
            u.clip_grad_norm(1.0)
            u.step()
        end.record()
        end.synchronize()
        if step >= args.warmup:
            times.append(start.elapsed_time(end))
            peak = max(peak, torch.cuda.max_memory_allocated())
    # No GPU reference models coexist: the old model is released before the new
    # model is constructed. CPU snapshots are disabled for large benchmark runs.
    snapshots = {}
    if not args.benchmark_only:
        for h, u in units.items():
            for name, p in u.model.named_parameters():
                snapshots[f"{h}.{name}"] = p.detach().cpu().clone()
                if p.grad is not None:
                    snapshots[f"{h}.{name}.grad"] = p.grad.detach().cpu().clone()
                for key, val in u.optimizer.state[p].items():
                    if isinstance(val, torch.Tensor):
                        snapshots[f"{h}.{name}.adam.{key}"] = val.detach().cpu().clone()
    numeric_losses = torch.stack(losses).float().cpu()
    summary = {
        "backend": mode,
        "rank": dist.get_rank(),
        "tp": args.tp,
        "dp": args.dp,
        "global_batch_per_microbatch": args.global_batch,
        "local_tokens": local_n,
        "ga": args.ga,
        "synthetic_update_ms": sum(times) / len(times),
        "peak_allocated_bytes": peak,
        "measured_windows": len(times),
        "full_latent_guard": "first warmup window"
        if mode != "legacy"
        else "reference: not guarded",
        "encoder_backward": "native_dense_local",
        "decoder": args.decoder if mode == "sharded_sparse" else "dense",
        "keys": args.keys if mode != "legacy" else "legacy",
        "protocol": args.protocol,
        "timing_scope": "synthetic window including input generation, forward/backward, normalization, clip and Adam",
    }
    del units
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=context.groups.tp_dp_cp)
    return summary, numeric_losses, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--global-batch", type=int, default=65)
    parser.add_argument("--d-in", type=int, default=128)
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--hooks", type=int, default=2)
    parser.add_argument("--ga", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--wavefront", action="store_true")
    parser.add_argument("--aux", action="store_true")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--decoder", choices=["torch", "triton"], default="torch")
    parser.add_argument("--keys", choices=["torch", "triton"], default="torch")
    parser.add_argument(
        "--protocol", choices=["auto", "candidates", "radix"], default="auto"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["legacy", "sharded_dense", "sharded_sparse"],
        default=["legacy", "sharded_dense", "sharded_sparse"],
    )
    parser.add_argument("--output", type=Path, default=Path("sharded_topk_validation"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU required; this script does not use mock Megatron modules."
        )
    if min(args.tp, args.dp, args.ga, args.hooks, args.warmup, args.steps) < 1:
        raise SystemExit("tp/dp/ga/hooks/warmup/steps must be positive")
    if args.global_batch < args.dp:
        raise SystemExit(
            "This A/B test needs at least one token per replica; CPU tests cover empty replicas."
        )
    if not args.benchmark_only and args.modes[0] != "legacy":
        raise SystemExit(
            "Numerical A/B acceptance must start with legacy; use --benchmark-only otherwise."
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl")
    if dist.get_world_size() != args.tp * args.dp:
        raise SystemExit("WORLD_SIZE must equal tp*dp for this validator")
    runtime = SAERuntime.from_layout(
        dp_size=args.dp,
        tp_size=args.tp,
        hooks=tuple(f"probe_{h}" for h in range(args.hooks)),
    )
    reports = []
    reference = None
    try:
        for mode in args.modes:
            # Previous mode's local variables/hook cycles have now left scope.
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            summary, losses, snapshots = run_mode(args, runtime, mode)
            if not args.benchmark_only:
                if reference is None:
                    reference = (losses, snapshots)
                else:
                    tol = 5e-3 if args.autocast else 2e-4
                    torch.testing.assert_close(losses, reference[0], atol=tol, rtol=tol)
                    assert snapshots.keys() == reference[1].keys()
                    worst = 0.0
                    for name, value in snapshots.items():
                        expected = reference[1][name]
                        torch.testing.assert_close(
                            value, expected, atol=tol, rtol=tol, msg=name
                        )
                        worst = max(worst, float((value - expected).abs().max()))
                    summary["max_snapshot_abs_difference"] = worst
                    summary["numerical_check"] = "passed"
            reports.append(summary)
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "rank": dist.get_rank(),
            "results": reports,
            "scope": "native Megatron TP/DDP + dense torch Adam; synthetic inputs; not E2E streaming or ZeRO overlap",
        }
        (args.output / f"rank{dist.get_rank()}.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report), flush=True)
    finally:
        runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
