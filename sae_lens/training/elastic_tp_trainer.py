"""Online SAE training and watermark-driven elastic TP transitions."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import statistics
import time

from sae_lens.training.elastic_tp_config import (
    activation_dtype,
    emit,
    open_buffer,
    write_json,
)

logger = logging.getLogger(__name__)


def setup_session(args, initial_tp=1):
    import torch
    import torch.distributed as dist

    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
    from sae_lens.training.dynamic_tp import DynamicTPSession, TPGroupPair

    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl")
    groups = TPGroupPair(
        tuple(range(dist.get_world_size())), tuple(range(initial_tp)), device=device
    )
    options = dict(
        topk_backend="sharded_ragged",
        topk_tie_policy="stable_id",
        topk_key_backend="torch",
        topk_candidate_protocol="auto",
        main_representation="sharded_ragged",
        aux_representation="sharded_ragged",
        main_compute="sparse",
        aux_compute="auto",
    )
    # A shared-runner policy is complete: do not let the probe's historical
    # ragged defaults override an explicitly requested legacy representation.
    if getattr(args, "sae_config", None):
        options = dict(args.sae_config)
    cfg = TopKTrainingSAEConfig(
        **options,
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        dtype="float32",
        device=str(device),
        normalize_activations="none",
    )
    return DynamicTPSession(
        {args.hook: cfg},
        groups,
        lr=args.lr,
        seed=getattr(args, "seed", 42),
        dead_feature_window=args.dead,
        adam_kwargs={"fused": True},
        tp_overlap=getattr(args, "tp_overlap", "off"),
        tp_overlap_max_live_hooks=getattr(args, "tp_overlap_max_live_hooks", 2),
    )


def cache_digest(session, hook, *, full=False):
    """All rows of eight columns plus first/last rows; not a full-state oracle."""
    import torch

    if hook not in session.state.activation_caches:
        return None
    tensor = session.state.activation_caches[hook]
    if full:
        return (
            tuple(tensor.shape),
            hashlib.sha256(
                tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        )
    if not tensor.shape[0]:
        return (tuple(tensor.shape), b"")
    sample = torch.cat(
        (tensor[:, :8].flatten(), tensor[:2].flatten(), tensor[-2:].flatten())
    )
    return (tuple(tensor.shape), sample.cpu().view(torch.uint8).numpy().tobytes())


def train(args):
    import torch
    import torch.distributed as dist

    from sae_lens.training.dynamic_tp_diagnostics import snapshot
    from sae_lens.training.dynamic_tp_input import DynamicTPInputLoader

    session = setup_session(args, initial_tp=args.initial_tp)
    groups = session.groups
    rank = groups.rank
    buffer = input_loader = log = None
    try:
        buffer = open_buffer(args) if rank == 0 else None
        input_loader = (
            DynamicTPInputLoader(
                buffer,
                capacity_chunks=args.cache_batches,
                device=groups.device,
                seed=args.seed,
            )
            if rank == 0
            else None
        )
        log = (args.output / f"train_rank{rank}.jsonl").open("w", buffering=1)
        control_path = args.output / "producer_control.json"
        epoch, pending = 1, None
        samples = []
        switches = []
        seen = set()
        scale = args.activation_scale
        if scale is not None:
            session.input_scale = scale
        started = time.perf_counter()
        last_poll, last_switch = -math.inf, -math.inf
        high_count = low_count = 0
        latest_fill = 0.0
        current_chunks = []
        if rank == 0:
            write_json(control_path, dict(tp=args.initial_tp, epoch=epoch, stop=False))
            write_json(
                args.output / "resolved_sae_config.json",
                session.configs[args.hook].to_dict(),
            )
            emit(
                log,
                "start",
                tp=args.initial_tp,
                min_tp=args.min_tp,
                steps=args.steps,
                batch_size=args.batch_size,
                d_sae=args.d_sae,
                dead=args.dead,
                low=args.low,
                high=args.high,
            )
        completed = 0
        while completed < args.steps:
            action = None
            if rank == 0:
                now = time.perf_counter()
                tp = len(groups.active_ranks)
                counts = buffer.queue_counts()
                latest_fill = 1 - counts["free"] / args.chunks
                cache = session.state.activation_caches.get(args.hook)
                cached = cache is not None and cache.shape[0] >= args.batch_size
                finished_producing = int(buffer._header[3]) >= args.steps
                if pending:
                    acknowledged = True
                    if pending["target"] > tp:
                        status = json.loads(
                            (args.output / f"producer{tp}_status.json").read_text()
                        )
                        acknowledged = status["state"] == "done" or (
                            status["state"] == "paused" and status["epoch"] == epoch
                        )
                    if acknowledged and (
                        completed > pending["step"]
                        or not cached
                        and not counts["ready"]
                    ):
                        action = dict(kind="switch", **pending)
                    elif now - pending["requested_at"] > args.pause_timeout:
                        raise RuntimeError(
                            "Producer did not quiesce at a chunk boundary"
                        )
                if now - last_poll >= args.poll_interval:
                    emit(
                        log,
                        "buffer",
                        step=completed,
                        tp=tp,
                        fill=latest_fill,
                        counts=counts,
                        claimed_chunks=int(buffer._header[3]),
                        loaded_chunks=len(seen),
                    )
                    if not pending and now - last_switch >= args.cooldown:
                        high_count = (
                            high_count + 1
                            if latest_fill >= args.high and tp < args.pool_size
                            else 0
                        )
                        low_count = (
                            low_count + 1
                            if latest_fill <= args.low
                            and tp > args.min_tp
                            and not finished_producing
                            else 0
                        )
                        target = (
                            tp + 1
                            if high_count >= args.watermark_samples
                            else tp - 1
                            if low_count >= args.watermark_samples
                            else None
                        )
                        if target is not None:
                            action = dict(
                                kind="prepare",
                                target=target,
                                step=completed,
                                reason="high_watermark"
                                if target > tp
                                else "low_watermark",
                                fill=latest_fill,
                                requested_at=now,
                            )
                            high_count = low_count = 0
                    last_poll = now
                if action is None:
                    action = dict(
                        kind="train" if cached or counts["ready"] else "wait",
                        refill_chunks=0
                        if cached
                        else min(
                            args.cache_batches, counts["ready"], args.steps - completed
                        ),
                        fill=latest_fill,
                    )
            packet = [action]
            dist.broadcast_object_list(packet, src=0, group=groups.control)
            action = packet[0]
            if action["kind"] == "wait":
                time.sleep(0.02)
                continue
            if action["kind"] == "prepare":
                pending = {k: v for k, v in action.items() if k != "kind"}
                epoch += 1
                if rank == 0 and action["target"] > len(groups.active_ranks):
                    write_json(
                        control_path, dict(tp=action["target"], epoch=epoch, stop=False)
                    )
                began = time.perf_counter()
                session.prepare(tuple(range(action["target"])))
                duration = time.perf_counter() - began
                pending["prepare_s"] = max(groups.agree(duration))
                if rank == 0:
                    emit(log, "prepared", **pending)
                continue
            if action["kind"] == "switch":
                torch.cuda.synchronize()
                whole_started = time.perf_counter()
                before = snapshot(session)
                expected_cache = (
                    cache_digest(session, args.hook, full=args.audit_inputs)
                    if rank == 0
                    else None
                )
                packet = [expected_cache]
                dist.broadcast_object_list(packet, src=0, group=groups.control)
                expected_cache = packet[0]
                progress = dict(session.state.progress) if rank == 0 else None
                packet = [progress]
                dist.broadcast_object_list(packet, src=0, group=groups.control)
                progress = packet[0]
                torch.cuda.reset_peak_memory_stats()
                metrics = session.switch()
                after = snapshot(session)
                for h in before:
                    for a, b in zip(before[h], after[h]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                if rank in groups.active_ranks:
                    assert (
                        cache_digest(session, args.hook, full=args.audit_inputs)
                        == expected_cache
                    )
                    assert session.state.progress == progress
                peers = groups.agree(
                    dict(
                        rank=rank,
                        pause_s=metrics["pause_s"],
                        peak_allocated=torch.cuda.max_memory_allocated(),
                    )
                )
                record = dict(
                    step=completed,
                    **metrics,
                    pause_s_max=max(p["pause_s"] for p in peers),
                    prepare_s=action["prepare_s"],
                    validation_s_included=time.perf_counter() - whole_started,
                    request_to_commit_s=time.perf_counter() - action["requested_at"],
                    reason=action["reason"],
                    trigger_fill=action["fill"],
                    validation="Exact sampled parameter/Adam coordinates, full dead counters, sampled cache, progress",
                    cache_validation="full SHA-256" if args.audit_inputs else "sampled",
                    peers=peers,
                )
                switches.append(record)
                if rank == 0:
                    write_json(
                        control_path,
                        dict(tp=len(groups.active_ranks), epoch=epoch, stop=False),
                    )
                    emit(log, "switch", **record)
                    write_json(args.output / "switches.json", switches)
                pending = None
                high_count = low_count = 0
                last_switch = time.perf_counter()
                continue

            input_started = time.perf_counter()
            count = action["refill_chunks"]
            if count:
                estimate_scale = scale is None
                if rank == 0:
                    data, current_chunks, order = input_loader.load_raw(count)
                    if scale is None:
                        norms = torch.cat(
                            [
                                batch.float().norm(dim=-1)
                                for batch in data.split(args.batch_size)
                            ]
                        )
                        scale = float(args.d_in**0.5 / norms.mean())
                        del norms
                    assert len(set(current_chunks)) == count
                    assert not seen.intersection(current_chunks)
                    seen.update(current_chunks)
                    emit(
                        log,
                        "loaded",
                        step=completed,
                        sequences=current_chunks,
                        tokens=data.shape[0],
                        activation_scale=scale,
                        input_dtype=activation_dtype(args),
                        cache_dtype=str(data.dtype),
                        cache_scaled=False,
                        shuffle_order_sha256=hashlib.sha256(
                            order.numpy().tobytes()
                        ).hexdigest(),
                    )
                elif rank in groups.active_ranks:
                    data = torch.empty(
                        count * args.batch_size,
                        args.d_in,
                        device=groups.device,
                        dtype=getattr(torch, activation_dtype(args)),
                    )
                if estimate_scale:
                    scale_packet = [scale if rank == 0 else None]
                    dist.broadcast_object_list(
                        scale_packet, src=0, group=groups.control
                    )
                    scale = session.input_scale = scale_packet[0]
                if rank in groups.active_ranks:
                    dist.broadcast(data, src=0, group=groups.active_group)
                    session.stage_inputs({args.hook: data})
                    del data
                if args.audit_inputs:
                    checks = groups.agree(cache_digest(session, args.hook, full=True))
                    assert all(checks[r] == checks[0] for r in groups.active_ranks)
                    if rank == 0:
                        emit(
                            log,
                            "input_verified",
                            step=completed,
                            ranks=list(groups.active_ranks),
                            shape=checks[0][0],
                            sha256=checks[0][1],
                        )
            if rank in groups.active_ranks:
                torch.cuda.synchronize()
                input_s = time.perf_counter() - input_started
                torch.cuda.reset_peak_memory_stats()
                dead_before = int(
                    (
                        session.state.replicated[args.hook + "/since_fired"] > args.dead
                    ).sum()
                )
                train_started = time.perf_counter()
            losses = session.train_cached_step(args.batch_size)
            completed += 1
            if rank in groups.active_ranks:
                torch.cuda.synchronize()
                step_s = time.perf_counter() - train_started
                loss = float(losses[args.hook])
                components = {
                    k: float(v)
                    for k, v in session.last_loss_components[args.hook].items()
                }
                assert math.isfinite(loss) and all(
                    math.isfinite(v) for v in components.values()
                )
                assert session.state.progress["steps"] == completed
                assert session.state.progress["tokens"] == completed * args.batch_size
                row = emit(
                    log,
                    "step",
                    step=completed,
                    tp=len(groups.active_ranks),
                    tokens=completed * args.batch_size,
                    loss=loss,
                    components=components,
                    dead_before=dead_before,
                    step_s=step_s,
                    input_s=input_s,
                    fill=action["fill"],
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    allocated=torch.cuda.memory_allocated(),
                    elapsed_s=time.perf_counter() - started,
                )
                if rank == 0:
                    samples.append(row)
                    if completed % 20 == 0 or completed == 1:
                        write_json(args.output / "progress.json", row)
                        logger.info("%s", json.dumps(row))
        if rank == 0:
            assert seen == set(range(args.steps))
            assert session.state.activation_caches[args.hook].shape[0] == 0
            write_json(
                control_path, dict(tp=args.pool_size, epoch=epoch + 1, stop=True)
            )
        if not args.profile_only:
            for model in session.state.models.values():
                model.save_model(args.output / "model")
        dist.barrier(group=groups.control)
        if rank == 0:
            write_json(
                args.output / "training_report.json",
                dict(
                    passed=True,
                    steps=completed,
                    tokens=completed * args.batch_size,
                    final_tp=len(groups.active_ranks),
                    elapsed_s=time.perf_counter() - started,
                    unique_chunks=len(seen),
                    activation_scale=scale,
                    switches=switches,
                    final_loss=samples[-1]["loss"],
                    model_saved=not args.profile_only,
                    peak_dead=max(p["dead_before"] for p in samples),
                    phase_means={
                        f"tp{tp}_{phase}": statistics.mean(
                            p["step_s"]
                            for p in samples
                            if p["tp"] == tp
                            and (p["dead_before"] > 0) == (phase == "aux")
                        )
                        for tp in range(1, args.pool_size + 1)
                        for phase in ("main", "aux")
                        if any(
                            p["tp"] == tp and (p["dead_before"] > 0) == (phase == "aux")
                            for p in samples
                        )
                    },
                ),
            )
    finally:
        if input_loader:
            input_loader.close()
        if buffer:
            buffer.close()
        if log is not None:
            log.close()
        groups.close()
        dist.destroy_process_group()
