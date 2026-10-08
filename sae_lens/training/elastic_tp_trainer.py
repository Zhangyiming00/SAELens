"""Online SAE training and watermark-driven elastic TP transitions."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import statistics
import time
import traceback
from contextlib import suppress

from sae_lens.training.elastic_tp_config import (
    activation_dtype,
    emit,
    online_hooks,
    open_buffer,
    write_json,
)
from sae_lens.training.elastic_tp_handoff import (
    cuda_memory,
    producer_quiesced,
    producer_running,
    read_status,
    release_inactive_sae,
)

logger = logging.getLogger(__name__)


def accumulation_tp_target(target, *, tp, pool_size, available, window_microbatches):
    """At an optimizer boundary, reserve inputs before pausing the last producer.

    A full-pool SAE cannot refill a partial GA window or migrate its gradients.
    Restore a producer before starting that window, even during cooldown. This
    also handles GA larger than SHM + local cache capacity without deadlocking.
    """
    if available < window_microbatches:
        if tp == pool_size:
            return tp - 1
        if target == pool_size:
            return None
    return target


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
    baseline = cuda_memory(device)["allocated"]
    configs = {args.hook: cfg}
    if len(online_hooks(args)) > 1:
        import copy

        configs = {h: copy.deepcopy(cfg) for h in online_hooks(args)}
    session = DynamicTPSession(
        configs,
        groups,
        lr=args.lr,
        seed=getattr(args, "seed", 42),
        dead_feature_window=args.dead,
        adam_kwargs={"fused": True},
        tp_overlap=getattr(args, "tp_overlap", "off"),
        tp_overlap_max_live_hooks=getattr(args, "tp_overlap_max_live_hooks", 2),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
    )
    session.handoff_baseline_allocated = baseline
    return session


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


def switch_and_validate(session, args):
    """Keep all diagnostic GPU temporaries scoped away from the release ACK."""
    import torch
    import torch.distributed as dist

    from sae_lens.training.dynamic_tp_diagnostics import snapshot

    groups = session.groups
    torch.cuda.synchronize()
    started = time.perf_counter()
    before = snapshot(session)
    digest = lambda: cache_digest(session, args.hook, full=args.audit_inputs)
    if len(online_hooks(args)) > 1:
        from sae_lens.training.elastic_tp_multihook import multi_cache_digest

        digest = lambda: multi_cache_digest(session, full=args.audit_inputs)
    packet = [digest()
              if groups.rank == 0 else None]
    dist.broadcast_object_list(packet, src=0, group=groups.control)
    expected_cache = packet[0]
    packet = [dict(session.state.progress) if groups.rank == 0 else None]
    dist.broadcast_object_list(packet, src=0, group=groups.control)
    progress = packet[0]
    torch.cuda.reset_peak_memory_stats()
    metrics = session.switch()
    after = snapshot(session)
    for h in before:
        for a, b in zip(before[h], after[h]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if groups.rank in groups.active_ranks:
        assert digest() == expected_cache
        assert session.state.progress == progress
    return metrics, time.perf_counter() - started, torch.cuda.max_memory_allocated()


def train(args):
    import torch
    import torch.distributed as dist

    from sae_lens.training.dynamic_tp_input import DynamicTPInputLoader

    groups = None
    hooks = online_hooks(args)
    multi_hook = len(hooks) > 1
    rank = int(os.environ.get("RANK", "0"))
    buffer = input_loader = log = None
    failed = False
    try:
        session = setup_session(args, initial_tp=args.initial_tp)
        groups = session.groups
        rank = groups.rank
        buffer = open_buffer(args) if rank == 0 else None
        input_loader = (
            DynamicTPInputLoader(
                buffer,
                capacity_chunks=args.cache_batches,
                device=groups.device,
                seed=args.seed,
            )
            if rank == 0 and not multi_hook
            else None
        )
        if multi_hook:
            from sae_lens.training.elastic_tp_multihook import (
                MultiHookTPInputLoader,
                hook_step_metrics,
                refill_multi_hook,
                save_multi_hook_models,
            )

            if rank == 0:
                input_loader = MultiHookTPInputLoader(
                    buffer, hooks=hooks, batch_size=args.batch_size,
                    capacity_chunks=args.cache_batches, device=groups.device, seed=args.seed,
                )
        log = (args.output / f"train_rank{rank}.jsonl").open("w", buffering=1)
        control_path = args.output / "producer_control.json"
        initial_control = [json.loads(control_path.read_text()) if rank == 0 else None]
        dist.broadcast_object_list(initial_control, src=0, group=groups.control)
        if initial_control[0]["stop"] or initial_control[0]["tp"] != args.initial_tp:
            raise RuntimeError("Supervisor has not completed the initial GPU handoff")
        epoch, pending, resuming = initial_control[0]["epoch"], None, None
        samples = []
        switches = []
        resumes = []
        seen = set()
        scale = args.activation_scale
        if scale is not None:
            session.input_scale = scale
        scales = {h: scale for h in hooks} if multi_hook and scale is not None else None
        if multi_hook:
            session.input_scales = session.validate_input_scales(scales)
        started = time.perf_counter()
        last_poll, last_switch = -math.inf, -math.inf
        high_count = low_count = 0
        latest_fill = 0.0
        current_chunks = []
        if rank == 0:
            write_json(control_path, dict(tp=args.initial_tp, epoch=epoch, stop=False, startup=False))
            write_json(
                args.output / "resolved_sae_config.json",
                {h: session.configs[h].to_dict() for h in hooks} if multi_hook
                else session.configs[args.hook].to_dict(),
            )
            emit(
                log,
                "start",
                tp=args.initial_tp,
                min_tp=args.min_tp,
                steps=args.steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                optimizer_steps=(args.steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps,
                batch_size=args.batch_size,
                d_sae=args.d_sae,
                dead=args.dead,
                low=args.low,
                high=args.high,
                vllm_residency=args.vllm_residency,
                **(dict(hook_names=hooks) if multi_hook else {}),
            )
        completed = consumed = 0
        window_step_s = window_input_s = 0.0
        window_peak = 0
        ga = args.gradient_accumulation_steps
        while consumed < args.steps:
            if consumed % ga == 0:
                window_microbatches = min(ga, args.steps - consumed)
            action = None
            if rank == 0:
                try:
                    now = time.perf_counter()
                    tp = len(groups.active_ranks)
                    if resuming is not None:
                        status = read_status(args.output / f"producer{resuming['rank']}_status.json")
                        if producer_running(status, resuming["epoch"]):
                            row = emit(log, "producer_resumed", **resuming,
                                       resume_s=now - resuming["requested_at"], status=status)
                            resumes.append(row)
                            resuming = None
                        elif now - resuming["requested_at"] > args.resume_timeout:
                            raise RuntimeError("Producer did not become ready after SAE memory release")
                    counts = buffer.queue_counts()
                    latest_fill = 1 - counts["free"] / args.chunks
                    cache = session.state.activation_caches.get(args.hook)
                    cached = cache is not None and cache.shape[0] >= args.batch_size
                    available = (
                        cache.shape[0] // args.batch_size if cache is not None else 0
                    ) + counts["ready"]
                    at_boundary = not session.state.in_step
                    needs_producer = ga > 1 and tp == args.pool_size and available < window_microbatches
                    finished_producing = int(buffer._header[3]) >= args.steps
                    if pending and at_boundary:
                        acknowledged = True
                        if pending["target"] > tp:
                            status = read_status(args.output / f"producer{tp}_status.json")
                            acknowledged = producer_quiesced(status, epoch, args.vllm_residency)
                        if acknowledged and (
                            completed > pending["step"]
                            or ga > 1
                            or not cached
                            and not counts["ready"]
                        ):
                            action = dict(kind="switch", **pending)
                            if pending["target"] > tp:
                                action["producer_ack"] = status
                        elif now - pending["requested_at"] > args.pause_timeout:
                            raise RuntimeError(
                                "Producer did not complete the requested GPU handoff"
                            )
                    if now - last_poll >= args.poll_interval:
                        emit(
                            log,
                            "buffer",
                            step=completed,
                            consumed_microbatches=consumed,
                            tp=tp,
                            fill=latest_fill,
                            counts=counts,
                            claimed_chunks=int(buffer._header[3]),
                            loaded_chunks=len(seen),
                        )
                        if not pending and not resuming and at_boundary and (
                            now - last_switch >= args.cooldown or needs_producer
                        ):
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
                            if ga > 1:
                                target = accumulation_tp_target(
                                    target, tp=tp, pool_size=args.pool_size,
                                    available=available, window_microbatches=window_microbatches,
                                )
                            if target is not None:
                                reason = (
                                    "ga_input_reserve" if needs_producer
                                    else "high_watermark" if target > tp
                                    else "low_watermark"
                                )
                                action = dict(
                                    kind="prepare",
                                    target=target,
                                    step=completed,
                                    reason=reason,
                                    fill=latest_fill,
                                    requested_at=now,
                                )
                                high_count = low_count = 0
                        last_poll = now
                    if action is None:
                        # The last producer may already be draining. Preserve the
                        # reserved full window until its pause ACK and TP commit.
                        waiting_for_last_producer = (
                            ga > 1 and pending and at_boundary
                            and pending["target"] == args.pool_size
                        )
                        can_train = (
                            (cached or counts["ready"])
                            and not waiting_for_last_producer
                            and not (at_boundary and needs_producer)
                        )
                        action = dict(
                            kind="train" if can_train else "wait",
                            refill_chunks=0
                            if cached
                            else min(
                                args.cache_batches, counts["ready"], args.steps - consumed
                            ),
                            fill=latest_fill,
                        )
                except Exception as exc:
                    # All workers must leave the control loop together; a
                    # root-only raise would strand peers in the next broadcast.
                    action = dict(kind="failed", error=repr(exc))
            packet = [action]
            dist.broadcast_object_list(packet, src=0, group=groups.control)
            action = packet[0]
            if action["kind"] == "failed":
                raise RuntimeError(action["error"])
            if action["kind"] == "wait":
                time.sleep(0.02)
                continue
            if action["kind"] == "prepare":
                pending = {k: v for k, v in action.items() if k != "kind"}
                epoch += 1
                pending["epoch"] = epoch
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
                whole_started = time.perf_counter()
                old_ranks = groups.active_ranks
                # Do not keep a view of retired input storage in this frame.
                cache = None
                metrics, validation_s, peak = switch_and_validate(session, args)
                departed = set(old_ranks) - set(groups.active_ranks)
                memory, error = None, None
                if rank in departed:
                    try:
                        memory = release_inactive_sae(
                            session, tolerance_mib=args.release_tolerance_mib
                        )
                        write_json(args.output / f"sae{rank}_status.json", dict(
                            epoch=epoch, state="released", ready=True,
                            memory_released=True, memory=memory,
                        ))
                    except Exception as exc:
                        error = repr(exc)
                        write_json(args.output / f"sae{rank}_status.json", dict(
                            epoch=epoch, state="failed", ready=False,
                            memory_released=False, error=error,
                        ))
                groups.check(error)
                peers = groups.agree(
                    dict(
                        rank=rank,
                        pause_s=metrics["pause_s"],
                        peak_allocated=peak,
                        sae_release=memory,
                    )
                )
                record = dict(
                    step=completed,
                    **metrics,
                    pause_s_max=max(p["pause_s"] for p in peers),
                    prepare_s=action["prepare_s"],
                    validation_s_included=validation_s,
                    handoff_s=time.perf_counter() - whole_started,
                    request_to_commit_s=time.perf_counter() - action["requested_at"],
                    vllm_residency=args.vllm_residency,
                    producer_ack=action.get("producer_ack"),
                    reason=action["reason"],
                    trigger_fill=action["fill"],
                    validation="Exact sampled parameter/Adam coordinates, full dead counters, sampled cache, progress",
                    cache_validation="full SHA-256" if args.audit_inputs else "sampled",
                    peers=peers,
                )
                switches.append(record)
                if rank == 0:
                    if departed:
                        resuming = dict(rank=next(iter(departed)), epoch=epoch,
                                        requested_at=time.perf_counter())
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
            if count and multi_hook:
                scales = refill_multi_hook(
                    session, args, input_loader, count, seen, scales, log, completed,
                )
            elif count:
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
                if multi_hook:
                    dead_by_hook = {h: int((session.state.replicated[h + "/since_fired"] > args.dead).sum())
                                    for h in hooks}
                    dead_before = sum(dead_by_hook.values())
                else:
                    dead_before = int(
                        (
                            session.state.replicated[args.hook + "/since_fired"] > args.dead
                        ).sum()
                    )
                train_started = time.perf_counter()
            losses = session.train_cached_microbatch(
                args.batch_size, window_microbatches=window_microbatches
            )
            consumed += 1
            update_complete = consumed % ga == 0 or consumed == args.steps
            if update_complete:
                completed += 1
            if rank in groups.active_ranks:
                torch.cuda.synchronize()
                window_step_s += time.perf_counter() - train_started
                window_input_s += input_s
                window_peak = max(window_peak, torch.cuda.max_memory_allocated())
                if not update_complete:
                    assert losses is None and session.state.in_step
                    continue
                if multi_hook:
                    hook_metrics = hook_step_metrics(session, losses, dead_by_hook)
                    loss = sum(v["loss"] for v in hook_metrics.values())
                    components = {k: sum(v["components"][k] for v in hook_metrics.values())
                                  for k in hook_metrics[hooks[0]]["components"]}
                else:
                    loss = float(losses[args.hook])
                    components = {
                        k: float(v)
                        for k, v in session.last_loss_components[args.hook].items()
                    }
                losses = None  # Release even the detached per-update GPU scalars.
                assert math.isfinite(loss) and all(
                    math.isfinite(v) for v in components.values()
                )
                assert session.state.progress["steps"] == completed
                assert session.state.progress["tokens"] == consumed * args.batch_size
                row = emit(
                    log,
                    "step",
                    step=completed,
                    tp=len(groups.active_ranks),
                    tokens=consumed * args.batch_size,
                    microbatches=window_microbatches,
                    consumed_microbatches=consumed,
                    effective_batch_size=window_microbatches * args.batch_size,
                    loss=loss,
                    components=components,
                    dead_before=dead_before,
                    step_s=window_step_s,
                    input_s=window_input_s,
                    fill=action["fill"],
                    peak_allocated=window_peak,
                    allocated=torch.cuda.memory_allocated(),
                    elapsed_s=time.perf_counter() - started,
                    **(dict(hooks=hook_metrics) if multi_hook else {}),
                )
                if rank == 0:
                    samples.append(row)
                    if completed % 20 == 0 or completed == 1:
                        write_json(args.output / "progress.json", row)
                        logger.info("%s", json.dumps(row))
                window_step_s = window_input_s = 0.0
                window_peak = 0
        if rank == 0:
            assert seen == set(range(args.steps))
            assert session.state.activation_caches[args.hook].shape[0] == 0
            if multi_hook:
                assert all(t.shape[0] == 0 for t in session.state.activation_caches.values())
            write_json(
                control_path, dict(tp=args.pool_size, epoch=epoch + 1, stop=True)
            )
        if not args.profile_only:
            if multi_hook:
                save_multi_hook_models(session, args.output / "model")
            else:
                for model in session.state.models.values():
                    model.save_model(args.output / "model")
        dist.barrier(group=groups.control)
        if rank == 0:
            write_json(
                args.output / "training_report.json",
                dict(
                    passed=True,
                    steps=completed,
                    tokens=consumed * args.batch_size,
                    microbatches=consumed,
                    gradient_accumulation_steps=ga,
                    final_tp=len(groups.active_ranks),
                    elapsed_s=time.perf_counter() - started,
                    unique_chunks=len(seen),
                    activation_scale=scale,
                    switches=switches,
                    producer_resumes=resumes,
                    vllm_residency=args.vllm_residency,
                    final_loss=samples[-1]["loss"],
                    model_saved=not args.profile_only,
                    peak_dead=max(p["dead_before"] for p in samples),
                    **(dict(hook_names=hooks, activation_scale_by_hook=scales,
                            final_metrics_by_hook=samples[-1]["hooks"],
                            peak_dead_by_hook={h: max(p["hooks"][h]["dead_before"] for p in samples) for h in hooks})
                       if multi_hook else {}),
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
    except BaseException as exc:
        failed = True
        if groups is not None:
            groups.failed = True
        # Report without CUDA or collectives: another rank can be stuck in a
        # queued NCCL operation after this rank's OOM. The supervisor observes
        # this file even if interpreter/CUDA teardown cannot finish promptly.
        try:
            write_json(
                args.output / f"train_rank{rank}_error.json",
                dict(rank=rank, pid=os.getpid(), timestamp=time.time(),
                     error_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc()),
            )
        except Exception:
            logger.exception("Could not publish elastic TP worker failure")
        logger.exception("Elastic TP rank %s failed", rank)
        raise
    finally:
        if failed:
            # No loader.close() (it waits for a CUDA event), CUDA synchronize,
            # distributed barrier, or communicator destruction on rank-local
            # failure. Process teardown reclaims device state; the supervisor
            # stops the whole training process group and owns SHM destruction.
            if log is not None:
                with suppress(Exception):
                    log.close()
        else:
            if input_loader:
                input_loader.close()
            if buffer:
                buffer.close()
            if log is not None:
                log.close()
            groups.close()
            dist.destroy_process_group()
