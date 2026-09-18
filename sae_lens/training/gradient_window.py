"""Synchronous runtime updates over a stream of unchanged provider microbatches.

Each hook accumulates token-sum gradients in its own storage, or directly
backpropagates a globally token-weighted loss in the FP32 GA1 fast path.
A known final microbatch enables native bucket-ready reduction and per-hook
updates. TP wavefront retains one microbatch's forward graphs across hooks;
graphs are never retained across accumulation microbatches.
"""

from collections.abc import Iterable, Iterator, Sized
from contextlib import nullcontext
from dataclasses import fields, replace
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from sae_lens import logger
from sae_lens.profiling import cuda_nvtx_range
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.training.megatron_optimizer import is_megatron_optimizer
from sae_lens.training.sae_train_unit import HookPhase, SAETrainUnit


def runtime_units(trainer: Any) -> dict[str, SAETrainUnit]:
    unit = getattr(trainer, "unit", None)
    return {unit.hook_name: unit} if unit is not None else trainer.units


def validate_accumulation(trainer: Any) -> int:
    steps = getattr(trainer.cfg, "gradient_accumulation_steps", 1)
    if type(steps) is not int or steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if steps > 1 and trainer.runtime is None:
        raise ValueError("Gradient accumulation requires static SAE runtime training")
    return steps


def fit_window_batches(
    trainer: Any, first: dict[str, torch.Tensor]
) -> Iterator[dict[str, torch.Tensor]]:
    """Read K batches in provider order, including a short final window.

    The caller already fetched/scaled the first batch. Progress remains in the
    existing local-budget units; TP copies never multiply the token count.
    """
    steps = validate_accumulation(trainer)
    trainer._window_extra_data_time = {"vllm_step_time_s": 0.0, "transfer_time_s": 0.0}
    yield first
    units = runtime_units(trainer)
    primary = next(iter(units))
    exhausted = False
    for _ in range(1, steps):
        progress = (
            trainer._last_global_tokens * trainer.cfg.train_batch_size_samples
            + getattr(trainer, "_token_count_remainder", 0)
        ) // trainer._progress_batch_total
        if trainer.n_training_samples + progress >= trainer.cfg.total_training_samples:
            break
        raw = None
        started = perf_counter()
        if not exhausted:
            try:
                raw = next(trainer.data_provider)
            except StopIteration:
                exhausted = True
        available = torch.tensor(int(raw is not None), device=trainer.cfg.device)
        group = trainer.runtime.require_local().dp_group
        if group.size() > 1:
            dist.all_reduce(available, group=group)
        if not available.item():
            break
        if raw is None:
            batch = {
                h: next(u.model.parameters()).new_empty((0, u.model.cfg.d_in))
                for h, u in units.items()
            }
        elif getattr(trainer, "unit", None) is not None:
            batch = {primary: trainer.activation_scaler(raw.to(trainer.cfg.device))}
        else:
            if set(raw) != set(units):
                raise ValueError(
                    "Every microbatch must contain exactly the local SAE hooks"
                )
            batch = {
                h: trainer.activation_scaler_by_hook[h](raw[h].to(trainer.cfg.device))
                for h in units
            }
        timing = trainer._consume_data_provider_timing()
        for key in trainer._window_extra_data_time:
            trainer._window_extra_data_time[key] += timing[key]
        trainer._window_data_time_s += perf_counter() - started
        yield batch


def train_runtime_window(
    trainer: Any, batches: Iterable[dict[str, torch.Tensor]]
) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
    """Run one update; ``batches`` yields already-scaled per-hook tensors."""
    units = runtime_units(trainer)
    single = getattr(trainer, "unit", None) is not None
    # A full configured window has a known final microbatch. Explicitly sized
    # input (e.g. a direct one-batch train step) may be shorter. Do not look
    # ahead into the provider: unknown-length tails use explicit window-end
    # sync, preserving streaming, buffer, failure and interruption semantics.
    final_microbatch = trainer.cfg.gradient_accumulation_steps
    if isinstance(batches, Sized):
        final_microbatch = min(final_microbatch, len(batches))
    group = trainer.runtime.require_local().dp_group
    mean_loss = getattr(trainer, "_runtime_mean_loss_fast_path", False)
    fused_amp_mean = (
        getattr(trainer, "_runtime_fused_amp_normalization", False)
        and trainer.grad_scaler.is_enabled()
    )
    ages = (
        {next(iter(units)): trainer.n_forward_passes_since_fired}
        if single
        else trainer.n_forward_passes_since_fired_by_hook
    )
    frequencies = (
        {next(iter(units)): trainer.act_freq_scores}
        if single
        else trainer.act_freq_scores_by_hook
    )
    masks = {h: (ages[h] > trainer.cfg.dead_feature_window).bool() for h in units}
    counts = {h: torch.zeros_like(frequencies[h]) for h in units}
    local_tokens = dict.fromkeys(units, 0)
    global_tokens = dict.fromkeys(units, 0)
    loss_sums = {h: {} for h in units}
    outputs = {}
    timing = dict.fromkeys(
        (
            "sae_forward_time_s",
            "sae_stats_sync_time_s",
            "sae_backward_time_s",
            "sae_post_backward_time_s",
            "sae_optimizer_time_s",
        ),
        0.0,
    )
    submitted = set()
    trainer._last_updated_hooks = []
    trainer._gradient_window_active = True
    trainer._last_global_tokens = 0
    trainer._window_data_time_s = 0.0
    trainer._last_window_microbatches = 0
    for unit in units.values():
        unit.model.train()
        unit.zero_grad()
    # Leave the active marker set on failure: partial gradients/provider state
    # must never be saved as an apparently complete, resumable update.
    for batch in batches:
        if set(batch) != set(units):
            raise ValueError(
                "Every microbatch must contain exactly the local SAE hooks"
            )
        ns = [batch[h].shape[0] for h in units]
        if group.size() > 1:
            ns = torch.tensor(ns, dtype=torch.int64, device=trainer.cfg.device)
            dist.all_reduce(ns, group=group)
            ns = ns.tolist()
        for h, n in zip(units, ns, strict=True):
            global_tokens[h] += n
        trainer._last_global_tokens = global_tokens[next(iter(units))]
        trainer._last_window_microbatches += 1
        is_final_microbatch = trainer._last_window_microbatches == final_microbatch
        inputs = {
            h: TrainStepInput(
                sae_in=acts if acts.shape[0] else acts.new_zeros((1, units[h].model.cfg.d_in)),
                dead_neuron_mask=masks[h],
                coefficients=trainer.get_coefficients() if single else {},
                n_training_steps=trainer.n_training_steps,
                is_logging_step=trainer._is_logging_step() if single else False,
            )
            for h, acts in batch.items()
        }
        wave_outputs = {}
        if getattr(trainer, "_runtime_tp_wavefront", False):
            from sae_lens.training.tp_wavefront import runtime_wavefront_forward

            t = perf_counter()
            wave_outputs = runtime_wavefront_forward(trainer, units, inputs)
            trainer._tp_phase_fence_if_needed()
            timing["sae_forward_time_s"] += perf_counter() - t
        for h, unit in units.items():
            acts = batch[h]
            n = acts.shape[0]
            local_tokens[h] += n
            native_sync = is_final_microbatch and unit.early_grad_sync
            with nullcontext() if native_sync else unit.no_sync():
                if wave_outputs:
                    output = wave_outputs.pop(h)
                else:
                    t = perf_counter()
                    with trainer.autocast_if_enabled:
                        output = unit.forward(inputs[h])
                    timing["sae_forward_time_s"] += perf_counter() - t
                with torch.no_grad():
                    if n:
                        firing = output.feature_acts.bool().float().sum(0)
                        counts[h] += firing.to_dense() if firing.is_sparse else firing
                    for key, value in {"loss": output.loss, **output.losses}.items():
                        loss_sums[h][key] = (
                            loss_sums[h].get(key, 0.0) + value.detach() * n
                        )
                t = perf_counter()
                if mean_loss and n:
                    # Counts for all replicas are known before any backward.
                    # Native DDP uses SUM, so unequal/empty replicas contribute
                    # n_local / n_global of their mean loss, with no DP divisor.
                    backward_loss = (
                        output.loss if group.size() == 1
                        else output.loss * (n / global_tokens[h])
                    )
                else:
                    backward_loss = output.loss * n
                unit.backward(
                    backward_loss,
                    trainer.grad_scaler, sync_gradients=native_sync,
                )
                del backward_loss
                timing["sae_backward_time_s"] += perf_counter() - t
            # Retain only one detached output per hook for existing logging.
            outputs[h] = replace(
                output,
                **{
                    field.name: getattr(output, field.name).detach()
                    for field in fields(output)
                    if isinstance(getattr(output, field.name), torch.Tensor)
                },
                losses={k: v.detach() for k, v in output.losses.items()},
            )
            del output
            if is_final_microbatch and getattr(
                trainer, "_runtime_optimizer_overlap", False
            ):
                if (
                    getattr(trainer, "_runtime_param_gather_schedule", "eager")
                    == "one_hook_lag"
                ):
                    # The next hook's gradient collectives have been submitted.
                    # Follow fixed local hook order, never rank-local event
                    # readiness. Each gather waits only for its own Adam event.
                    for previous in units.values():
                        if previous.hook_name in submitted:
                            _launch_param_gather(trainer, previous)
                # Record before submitting the next hook, never at window end.
                ready = torch.cuda.Event()
                ready.record()
                stream = trainer._runtime_optimizer_stream
                with torch.cuda.stream(stream), cuda_nvtx_range(f"sae:{h}:update"):
                    stream.wait_event(ready)
                    unit.finish_window(
                        global_tokens[h], gradients_are_mean=mean_loss,
                        normalization_in_unscale=fused_amp_mean,
                    )
                    unit.phase = HookPhase.UPDATING
                    if global_tokens[h]:
                        _update_hook(
                            trainer, unit,
                            normalization_tokens=global_tokens[h] if fused_amp_mean else None,
                        )
                    else:
                        unit.zero_grad()
                    done = unit.update_done if global_tokens[h] else torch.cuda.Event()
                    if not global_tokens[h]:
                        done.record(stream)
                    unit.update_done = done
                    unit.params_ready = None if unit.param_gather_pending else done
                    unit.phase = (
                        HookPhase.UPDATING
                        if unit.param_gather_pending
                        else HookPhase.PARAMS_ENQUEUED
                    )
                submitted.add(h)

    trainer._last_window_local_tokens = local_tokens[next(iter(units))]
    trainer._last_global_tokens_by_hook = global_tokens
    trainer._last_step_had_tokens = any(global_tokens.values())
    # Serial control finishes every reduction first; overlap already submitted
    # each known-final hook immediately after its final backward.
    t = perf_counter()
    for h, unit in units.items():
        if h not in submitted:
            unit.finish_window(
                global_tokens[h], gradients_are_mean=mean_loss,
                normalization_in_unscale=fused_amp_mean,
            )
    timing["sae_post_backward_time_s"] += perf_counter() - t
    for h, unit in units.items():
        n = global_tokens[h]
        if n == 0:
            if h not in submitted:
                unit.zero_grad()
            continue
        t = perf_counter()
        did_fire = counts[h].bool().to(torch.int32)
        if group.size() > 1:
            dist.all_reduce(did_fire, op=dist.ReduceOp.MAX, group=group)
        ages[h].add_(1)
        ages[h][did_fire.bool()] = 0
        frequencies[h].add_(counts[h])
        if single:
            trainer.n_frac_active_samples += local_tokens[h]
        else:
            trainer.n_frac_active_samples_by_hook[h] += n
        timing["sae_stats_sync_time_s"] += perf_counter() - t
        t = perf_counter()
        if h not in submitted:
            _update_hook(
                trainer, unit, normalization_tokens=n if fused_amp_mean else None
            )
        timing["sae_optimizer_time_s"] += perf_counter() - t
    # All consumers (scaler, statistics, checkpoint and next window) depend on
    # the updates. Waiting here does not fence B before its work was submitted.
    if any(u.param_gather_pending for u in units.values()):
        if getattr(trainer, "_runtime_param_gather_schedule", "eager") == "deferred":
            # Diagnostic: exclude parameter communication overlap with both
            # other hooks' compute and their Adam updates, still inside timing.
            torch.cuda.current_stream().wait_stream(trainer._runtime_optimizer_stream)
            for unit in units.values():
                if unit.param_gather_pending:
                    with cuda_nvtx_range(f"sae:{unit.hook_name}:deferred_param_gather"):
                        unit.ddp.start_param_sync(force_sync=True)
                    unit.param_gather_pending = False
                    unit.params_ready = torch.cuda.Event()
                    unit.params_ready.record()
                    unit.phase = HookPhase.PARAMS_ENQUEUED
        else:
            # No whole-optimizer-stream fence: A's gather may run alongside
            # later Adam kernels. Unknown-length tails also finish here.
            for unit in units.values():
                _launch_param_gather(trainer, unit)
    for unit in units.values():
        unit.wait_params()
    if trainer._last_step_had_tokens:
        trainer.grad_scaler.update()
    if single and trainer._last_updated_hooks:
        trainer.lr_scheduler.step()
        for scheduler in trainer.coefficient_schedulers.values():
            scheduler.step()
    for h, output in outputs.items():
        denom = max(local_tokens[h], 1)
        output.loss = loss_sums[h]["loss"] / denom
        output.losses = {k: loss_sums[h][k] / denom for k in output.losses}
    trainer._gradient_window_active = False
    return outputs, timing


def require_window_boundary(trainer: Any) -> None:
    if getattr(trainer, "_gradient_window_active", False) or getattr(
        trainer, "_runtime_update_pending", False
    ):
        raise RuntimeError(
            "Checkpoint requires a completed gradient accumulation window"
        )
    if getattr(trainer, "runtime", None) is not None:
        for unit in runtime_units(trainer).values():
            unit.wait_params()


def configure_update_batch(trainer: Any) -> None:
    steps = validate_accumulation(trainer)
    from sae_lens.training.hook_overlap import configure_hook_overlap
    from sae_lens.training.tp_wavefront import configure_tp_wavefront

    configure_hook_overlap(trainer, runtime_units(trainer))
    configure_tp_wavefront(trainer, runtime_units(trainer))
    trainer.global_update_batch_size = trainer._progress_batch_total * steps
    context = trainer.runtime.require_local()
    trainer._runtime_mean_loss_fast_path = (
        steps == 1
        # Changing loss scaling before BF16 backward changes its rounding.
        # Retain the accepted autocast trajectory instead of loosening resume
        # tolerances or introducing another loss-scale owner.
        and not trainer.cfg.autocast
        and getattr(trainer.cfg, "sae_ga1_loss_normalization", True)
    )
    trainer._runtime_fused_amp_normalization = (
        steps == 1 and trainer.cfg.autocast
        and getattr(trainer.cfg, "sae_ga1_loss_normalization", True)
    )
    logger.info(
        "SAE gradient objective=%s",
        "pre-backward valid-token weighting (GA=1, autocast off)" if trainer._runtime_mean_loss_fast_path
        else "token-sum loss; token normalization fused with enabled AMP unscale"
        if trainer._runtime_fused_amp_normalization
        else "token-sum loss / global valid tokens",
    )
    if context.dp_group.rank() == 0 and context.tp_group.rank() == 0:
        logger.info(
            "Static SAE global update batch: %s tokens per hook (%s global microbatch tokens x %s microbatches); actual window token counts determine normalization",
            trainer.global_update_batch_size,
            trainer._progress_batch_total,
            steps,
        )
        units = runtime_units(trainer)
        logger.info(
            "Static SAE native final-microbatch bucket reduction: %s; unknown-length "
            "short windows use explicit finish; TP+DP requires NCCL >= 2.26 and "
            "NCCL_LAUNCH_ORDER_IMPLICIT=1 before NCCL initialization",
            {h: u.early_grad_sync for h, u in units.items()},
        )


def window_metrics(trainer: Any) -> dict[str, Any]:
    if trainer.runtime is None:
        return {}
    return dict(
        global_update_batch_size=trainer.global_update_batch_size,
        window_microbatches=trainer._last_window_microbatches,
        global_valid_tokens_by_hook=trainer._last_global_tokens_by_hook,
    )


def _launch_param_gather(trainer, unit):
    if not unit.param_gather_pending:
        return
    unit.check_failure()
    stream = trainer._runtime_param_gather_stream
    with torch.cuda.stream(stream), cuda_nvtx_range(
        f"sae:{unit.hook_name}:scheduled_param_gather"
    ):
        stream.wait_event(unit.update_done)
        # Native synchronous collectives establish a dependency on the caller
        # stream; force_sync consumes the native handle, avoiding a second
        # forward-pre-hook gather. This does not wait for all hooks on the CPU.
        unit.ddp.start_param_sync(force_sync=True)
        unit.params_ready = torch.cuda.Event()
        unit.params_ready.record(stream)
    unit.param_gather_pending = False
    unit.phase = HookPhase.PARAMS_ENQUEUED


def _update_hook(trainer, unit, *, normalization_tokens=None):
    unit.phase = HookPhase.UPDATING
    finite = True
    if trainer.grad_scaler.is_enabled():
        nonfinite = torch.zeros(
            (), device=next(unit.model.parameters()).device, dtype=torch.float32
        )
        context = unit.parallel_context.require_local()
        sharded = getattr(unit.optimizer, "_sae_distributed_optimizer", False)
        stats_group = context.groups.tp_dp_cp if sharded else context.tp_group
        if normalization_tokens is None:
            trainer.grad_scaler.unscale_(unit.optimizer)
        else:
            from sae_lens.training.runtime_amp import unscale_with_token_mean

            unscale_with_token_mean(
                trainer.grad_scaler, unit.optimizer, normalization_tokens
            )
        # One external scaler owns the window. Explicitly install the agreed
        # hook overflow, also on ranks with an empty valid shard. Such ranks
        # must skip Adam and contribute the same overflow to scaler.update().
        found = trainer.grad_scaler._per_optimizer_states[id(unit.optimizer)][
            "found_inf_per_device"
        ]
        for value in found.values():
            torch.maximum(nonfinite, value, out=nonfinite)
        if stats_group.size() > 1:
            dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX, group=stats_group)
        found[nonfinite.device] = nonfinite
        finite = not bool(nonfinite.item())
    else:
        trainer.grad_scaler.unscale_(unit.optimizer)
    if not is_megatron_optimizer(unit.optimizer):
        unit.clip_grad_norm(1.0)
    unit.check_failure()
    # Native FP32Optimizer.step owns prepare_grads, per-hook clipping and
    # the Adam update. GradScaler gates the whole native step on overflow;
    # do not clip twice or bypass Megatron via its inner Adam.
    trainer.grad_scaler.step(unit.optimizer)
    if finite:
        trainer._last_updated_hooks.append(unit.hook_name)
        unit.update_count += 1
    unit.zero_grad()
    if (
        finite
        and getattr(trainer, "_runtime_defer_param_gather", False)
        and getattr(unit.optimizer, "_sae_distributed_optimizer", False)
    ):
        unit.param_gather_pending = True
    if next(unit.model.parameters()).device.type == "cuda":
        unit.update_done = torch.cuda.Event()
        unit.update_done.record()
        unit.params_ready = None if unit.param_gather_pending else unit.update_done
        unit.phase = (
            HookPhase.UPDATING
            if unit.param_gather_pending
            else HookPhase.PARAMS_ENQUEUED
        )
    else:
        unit.phase = HookPhase.PARAMS_READY
