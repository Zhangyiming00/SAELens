"""Train on real cached activations while changing TP without restarting.

Input safetensors: one [tokens,d_in] tensor per hook. Rows are already scaled.
Each TP peer receives the SAME rows, regardless of local feature ownership.
This runner reserves a fixed worker pool; it does not cold-start vLLM roles.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.dynamic_tp import DynamicTPSession, TPGroupPair

# This CLI emits machine-readable training records on stdout.
# ruff: noqa: T201


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--hooks", nargs="+", required=True)
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--auxk", type=int)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Microbatches per optimizer update; --batch-size is the microbatch size.")
    parser.add_argument("--cache-batches", type=int, default=4)
    parser.add_argument(
        "--activation-dtype", choices=("none", "float32", "bfloat16"),
        help="Input management dtype; omitted/none resolves from file dtype and conversion policy.",
    )
    parser.add_argument("--activation-conversion", choices=("auto", "vllm", "sae"), default="auto")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tp-schedule", default="0:1,10:2,20:3,30:4,40:3,50:2,60:1")
    parser.add_argument(
        "--schedule-json", help="JSON object: completed step -> sorted global rank list"
    )
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dead-feature-window", type=int, default=1000)
    parser.add_argument("--output", default="dynamic_tp_output")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Exit normally without any model/checkpoint export",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-final", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--tp-overlap", choices=("off", "eager", "lazy", "bounded"), default="off"
    )
    parser.add_argument("--tp-overlap-max-live-hooks", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.cache_batches, args.gradient_accumulation_steps) < 1:
        parser.error("steps, batch-size, cache-batches and gradient-accumulation-steps must be positive")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every cannot be negative")
    if args.profile_only and (args.checkpoint_every or args.checkpoint_final):
        parser.error("profile-only excludes checkpoint writing")
    if args.backend == "nccl":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    dist.init_process_group(args.backend)
    rank, size = dist.get_rank(), dist.get_world_size()
    schedule = (
        {
            int(k): tuple(v)
            for k, v in json.loads(Path(args.schedule_json).read_text()).items()
        }
        if args.schedule_json
        else {
            int(item.split(":")[0]): tuple(range(int(item.split(":")[1])))
            for item in args.tp_schedule.split(",")
        }
    )
    if 0 not in schedule:
        raise ValueError("Schedule must define initial membership at step 0")
    previous = None
    for step, ranks in sorted(schedule.items()):
        if (
            step < 0
            or not ranks
            or tuple(sorted(set(ranks))) != ranks
            or min(ranks) < 0
            or max(ranks) >= size
        ):
            raise ValueError("Invalid schedule membership")
        if len(ranks) > args.d_sae:
            raise ValueError("TP cannot exceed d_sae")
        if previous is not None and (
            abs(len(ranks) - len(previous)) != 1
            or not (set(ranks) < set(previous) or set(previous) < set(ranks))
        ):
            raise ValueError("Every event must add or remove one rank")
        previous = ranks
    start = 0
    if args.resume:
        start = json.loads((args.resume / "manifest.json").read_text())["progress"][
            "steps"
        ]
        if start >= args.steps:
            raise ValueError("steps must exceed the checkpoint's completed steps")
    initial = schedule[max(s for s in schedule if s <= start)]
    groups = TPGroupPair(
        tuple(range(size)), initial, device=device, backend=args.backend
    )
    try:
        with safe_open(args.activations, framework="pt", device="cpu") as source:
            from sae_lens.precision import resolve_activation_dtype

            shapes = {h: source.get_slice(h).get_shape() for h in args.hooks}
            dtype_names = {"F32": "float32", "BF16": "bfloat16"}
            input_dtypes = {
                h: resolve_activation_dtype(
                    dtype_names.get(source.get_slice(h).get_dtype(), "float32"), "float32",
                    args.activation_dtype, args.activation_conversion,
                ) for h in args.hooks
            }
            microbatches = args.steps * args.gradient_accumulation_steps
            if any(
                len(s) != 2 or s[0] < microbatches * args.batch_size
                for s in shapes.values()
            ):
                raise ValueError(
                    "Each activation tensor needs [steps*gradient_accumulation_steps*batch_size, d_in] rows; no cycling or discarded tokens"
                )
            configs = {
                h: TopKTrainingSAEConfig(
                    d_in=shape[1],
                    d_sae=args.d_sae,
                    k=args.k,
                    auxk=args.auxk,
                    dtype="float32",
                    device=str(device),
                    normalize_activations="none",
                    topk_backend="sharded_dense",
                    topk_tie_policy="stable_id",
                )
                for h, shape in shapes.items()
            }
            session = DynamicTPSession(
                configs,
                groups,
                lr=args.lr,
                seed=args.seed,
                dead_feature_window=args.dead_feature_window,
                adam_kwargs={"fused": args.backend == "nccl"},
                tp_overlap=args.tp_overlap,
                tp_overlap_max_live_hooks=args.tp_overlap_max_live_hooks,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            provider = None
            if args.resume or args.checkpoint_every or args.checkpoint_final:
                from sae_lens.training.dynamic_tp_checkpoint import _digest

                packet = [
                    {
                        "activation_sha256": _digest(Path(args.activations)),
                        "batch_size": args.batch_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "hooks": args.hooks,
                        "activation_dtypes": input_dtypes,
                    }
                    if rank == 0
                    else None
                ]
                dist.broadcast_object_list(packet, src=0, group=groups.control)
                provider = packet[0]
            if args.resume:
                restored_provider = session.load_checkpoint(args.resume)
                # Old checkpoints used FP32 transport unconditionally.
                old_dtype = restored_provider.pop("activation_dtype", "float32")
                restored_provider.setdefault("activation_dtypes", {h: old_dtype for h in args.hooks})
                restored_provider.setdefault("gradient_accumulation_steps", 1)
                if restored_provider != provider:
                    raise ValueError(
                        "Resume activation source/batching differs from checkpoint"
                    )
            # Preparation is outside the measured paused migration window.
            events = sorted(step for step in schedule if start < step < args.steps)
            if events:
                session.prepare(schedule[events[0]])
            for step in range(start, args.steps):
                if step in events:
                    metrics = session.switch()
                    if rank == 0:
                        print(
                            json.dumps(dict(event="switch", step=step, **metrics)),
                            flush=True,
                        )
                    following = [s for s in events if s > step]
                    if following:
                        session.prepare(schedule[following[0]])
                for micro in range(args.gradient_accumulation_steps):
                    cursor = step * args.gradient_accumulation_steps + micro
                    batches = {}
                    cache_empty = (
                        not session.state.activation_caches
                        or not next(iter(session.state.activation_caches.values())).shape[0]
                    )
                    if rank in groups.active_ranks and cache_empty:
                        rows = min(args.cache_batches, microbatches - cursor) * args.batch_size
                        for h, shape in shapes.items():
                            batch = (
                                source.get_slice(h)[
                                    cursor * args.batch_size : cursor * args.batch_size + rows
                                ].to(device, getattr(torch, input_dtypes[h]))
                                if rank == groups.active_ranks[0]
                                else torch.empty(rows, shape[1], device=device,
                                                 dtype=getattr(torch, input_dtypes[h]))
                            )
                            dist.broadcast(
                                batch, src=groups.active_ranks[0], group=groups.active_group
                            )
                            batches[h] = batch
                        session.stage_inputs(batches)
                        batches.clear()
                        del batch
                    loss = session.train_cached_microbatch(args.batch_size)
                if rank == groups.active_ranks[0]:
                    print(
                        json.dumps(
                            dict(
                                event="step",
                                step=step + 1,
                                tp=len(groups.active_ranks),
                                loss={h: float(v) for h, v in loss.items()},
                                tokens=session.state.progress["tokens"],
                                microbatches=args.gradient_accumulation_steps,
                            )
                        ),
                        flush=True,
                    )
                if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                    session.save_checkpoint(
                        Path(args.output) / f"checkpoint_{step + 1}",
                        provider_state=provider,
                    )
            if args.checkpoint_final and (
                not args.checkpoint_every or args.steps % args.checkpoint_every
            ):
                session.save_checkpoint(
                    Path(args.output) / f"checkpoint_{args.steps}",
                    provider_state=provider,
                )
            # This is an explicitly requested final export, not switch-time I/O.
            export_started = time.perf_counter()
            if not args.profile_only:
                for h, model in session.state.models.items():
                    model.save_model(Path(args.output) / h)
            if rank == 0:
                print(
                    json.dumps(
                        dict(
                            event="complete",
                            steps=args.steps,
                            model_saved=not args.profile_only,
                            export_elapsed_s=time.perf_counter() - export_started,
                            wall_time=time.time(),
                        )
                    ),
                    flush=True,
                )
    finally:
        groups.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
