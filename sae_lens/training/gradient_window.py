"""Synchronous runtime updates over a stream of unchanged provider microbatches.

Each hook accumulates SUM(token losses) in its own DDP buffers. A known final
microbatch enables native bucket-ready reduction; only the window boundary
waits, divides by SUM(valid tokens), clips and updates. No large activation
batch or graph list is kept.
"""

from collections.abc import Iterable, Iterator, Sized
from contextlib import nullcontext
from dataclasses import fields, replace
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from sae_lens import logger
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from sae_lens.training.sae_train_unit import SAETrainUnit


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
        ns = torch.tensor(
            [batch[h].shape[0] for h in units],
            dtype=torch.int64,
            device=trainer.cfg.device,
        )
        if group.size() > 1:
            dist.all_reduce(ns, group=group)
        for h, n in zip(units, ns.tolist(), strict=True):
            global_tokens[h] += n
        trainer._last_global_tokens = global_tokens[next(iter(units))]
        trainer._last_window_microbatches += 1
        is_final_microbatch = trainer._last_window_microbatches == final_microbatch
        for h, unit in units.items():
            acts = batch[h]
            n = acts.shape[0]
            local_tokens[h] += n
            native_sync = is_final_microbatch and unit.early_grad_sync
            with nullcontext() if native_sync else unit.no_sync():
                t = perf_counter()
                with trainer.autocast_if_enabled:
                    output = unit.forward(
                        TrainStepInput(
                            sae_in=acts
                            if n
                            else acts.new_zeros((1, unit.model.cfg.d_in)),
                            dead_neuron_mask=masks[h],
                            coefficients=trainer.get_coefficients() if single else {},
                            n_training_steps=trainer.n_training_steps,
                            is_logging_step=trainer._is_logging_step()
                            if single
                            else False,
                        )
                    )
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
                unit.backward(
                    output.loss * n, trainer.grad_scaler, sync_gradients=native_sync
                )
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

    trainer._last_window_local_tokens = local_tokens[next(iter(units))]
    trainer._last_global_tokens_by_hook = global_tokens
    trainer._last_step_had_tokens = any(global_tokens.values())
    trainer._last_updated_hooks = []
    # All hooks finish communication before any optimizer can mutate weights.
    t = perf_counter()
    for h, unit in units.items():
        unit.finish_window(global_tokens[h])
    timing["sae_post_backward_time_s"] += perf_counter() - t
    for h, unit in units.items():
        n = global_tokens[h]
        if n == 0:
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
        finite = True
        if trainer.grad_scaler.is_enabled():
            nonfinite = (
                torch.stack(
                    [
                        ~torch.isfinite(p.grad).all()
                        for p in unit.model.parameters()
                        if p.grad is not None
                    ]
                )
                .any()
                .to(torch.int32)
            )
            tp_group = unit.parallel_context.require_local().tp_group
            if tp_group.size() > 1:
                dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX, group=tp_group)
            finite = not bool(nonfinite.item())
            if not finite:
                # GradScaler inspects only local gradients. Make every TP
                # shard observe the same overflow before unscale_, including
                # shards whose own gradients are finite. All shards must skip
                # Adam and update their scaler together.
                next(
                    p for p in unit.model.parameters() if p.grad is not None
                ).grad.fill_(float("inf"))
        trainer.grad_scaler.unscale_(unit.optimizer)
        unit.clip_grad_norm(1.0)
        unit.check_failure()
        trainer.grad_scaler.step(unit.optimizer)
        if finite:
            trainer._last_updated_hooks.append(h)
        timing["sae_optimizer_time_s"] += perf_counter() - t
        unit.zero_grad()
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


def configure_update_batch(trainer: Any) -> None:
    steps = validate_accumulation(trainer)
    trainer.global_update_batch_size = trainer._progress_batch_total * steps
    context = trainer.runtime.require_local()
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
