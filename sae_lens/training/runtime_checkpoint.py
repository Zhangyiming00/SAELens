"""Rank-local progress and placement-local AMP state for static SAE checkpoints."""

import math
from pathlib import Path

import torch
import torch.distributed as dist


def _runtime(trainer):
    unit = getattr(trainer, "unit", None)
    return getattr(trainer, "runtime", None) or (
        unit.parallel_context if unit is not None else None
    )


def require_distributed_checkpoint(trainer, path):
    from sae_lens.training.gradient_window import runtime_units

    if _runtime(trainer) is None or not all(
        getattr(u.optimizer, "_sae_distributed_optimizer", False)
        or getattr(u.optimizer, "_sae_single_replica_fast_path", False)
        for u in runtime_units(trainer).values()
    ):
        raise ValueError(
            "Distributed Adam checkpoint requires the native distributed backend"
        )
    for name in (
        f"trainer_runtime_rank{dist.get_rank()}.pt",
        f"distributed_adam_rank{dist.get_rank()}.pt",
    ):
        if not (Path(path) / name).is_file():
            raise ValueError(f"Incomplete distributed Adam checkpoint: missing {name}")


def _load_single_replica_adam(unit, saved):
    """DP1's old native shards each cover a full parameter, in signature order.

    This is a same-topology conversion, not general resharding. Native FP32
    shard groups and bind_model's ranges have identical insertion order. Check
    every range/name/shape before mapping moments; never use full-model order
    to index the old shard optimizer's parameter IDs.
    """
    if unit.parallel_context.require_local().dp_group.size() != 1:
        raise ValueError("Direct-gradient checkpoint conversion requires DP=1")
    params = dict(unit.model.named_parameters())
    ranges = saved["signature"]["ranges"]
    groups = saved["adam"]["param_groups"]
    if saved["version"] != 1 or set(ranges) != set(params) or len(groups) != 1:
        raise ValueError("Unsupported DP1 distributed Adam checkpoint mapping")
    group = groups[0]
    if len(group["params"]) != len(ranges):
        raise ValueError("DP1 Adam ownership differs from its saved ranges")
    loaded = {}
    for (name, (start, end, shape)), index in zip(
        ranges.items(), group["params"], strict=True
    ):
        parameter = params[name]
        if start != 0 or end != parameter.numel() or tuple(shape) != tuple(parameter.shape):
            raise ValueError("DP1 conversion requires complete matching parameters")
        state = saved["adam"]["state"].get(index)
        if not state:
            continue  # A checkpoint before the first update has no moments.
        if not {"exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("Incomplete DP1 Adam moments")
        result = {}
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                value = state[key]
                if value.numel() != parameter.numel():
                    raise ValueError("DP1 Adam moment size differs from parameter")
                result[key] = value.reshape(parameter.shape).to(parameter).clone()
        step = state.get("step", group.get("step"))
        if step is None:
            raise ValueError("DP1 Adam checkpoint is missing its update count")
        result["step"] = torch.tensor(float(step), device=parameter.device)
        loaded[parameter] = result
    optimizer = unit.optimizer
    optimizer.state.clear()
    optimizer.state.update(loaded)
    # Named group metadata was restored earlier by the trainer. TE's saved
    # groups omit torch-only defaults (e.g. amsgrad), so backfill them before
    # the first torch Adam update without replacing the scheduler's live group.
    for key, value in optimizer.optimizer.defaults.items():
        optimizer.param_groups[0].setdefault(key, value)
    # Keep live dictionaries shared with the scheduler and torch fused flags.
    for key in ("lr", "initial_lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"):
        if key in group:
            optimizer.param_groups[0][key] = group[key]
    optimizer.param_groups[0].pop("step", None)


def uses_exact_runtime(trainer):
    return (
        _runtime(trainer) is not None
        and getattr(trainer.cfg, "routing_dp_batch_mode", "equal") == "exact"
    )


def checkpoint_token_count(trainer):
    if not uses_exact_runtime(trainer):
        return trainer.n_training_samples
    return (
        trainer.n_training_samples * trainer._progress_batch_total
        + getattr(trainer, "_token_count_remainder", 0)
    ) // trainer.cfg.train_batch_size_samples


def configure_runtime_checkpoints(trainer):
    if not uses_exact_runtime(trainer) or trainer.cfg.n_checkpoints <= 0:
        return
    total = (
        trainer.cfg.total_training_samples
        // trainer.cfg.train_batch_size_samples
        * trainer._progress_batch_total
    )
    trainer.checkpoint_thresholds = list(
        range(0, total, math.ceil(total / (trainer.cfg.n_checkpoints + 1)))
    )[1:]


def runtime_checkpoint_name(trainer, name):
    if uses_exact_runtime(trainer):
        if name.isdigit():
            return str(checkpoint_token_count(trainer))
        for prefix in ("final_", "quiesce_"):
            if name.startswith(prefix) and name[len(prefix) :].isdigit():
                return prefix + str(checkpoint_token_count(trainer))
    return name


def _identity(trainer, runtime):
    context = runtime.require_local()
    return dict(
        domain=context.domain.name,
        hooks=list(context.domain.hooks),
        tp_ranks=list(context.tp_ranks),
        dp_ranks=list(context.dp_ranks),
        local_batch_size=trainer.cfg.train_batch_size_samples,
        batch_mode=getattr(trainer.cfg, "routing_dp_batch_mode", "equal"),
        gradient_accumulation_steps=getattr(
            trainer.cfg, "gradient_accumulation_steps", 1
        ),
    )


def save_runtime_trainer_state(trainer, checkpoint_path):
    runtime = _runtime(trainer)
    if runtime is None or not dist.is_initialized():
        return
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    context = runtime.require_local()
    placement = runtime.domains.index(context.domain)
    from sae_lens.training.gradient_window import runtime_units

    units = runtime_units(trainer)
    sharded = any(
        getattr(u.optimizer, "_sae_distributed_optimizer", False)
        for u in units.values()
    )
    if sharded:
        for unit in units.values():
            unit.wait_params()
        torch.save(
            {h: u.optimizer.local_checkpoint() for h, u in units.items()},
            checkpoint_path / f"distributed_adam_rank{dist.get_rank()}.pt",
        )
    torch.save(
        dict(
            version=1,
            optimizer_backend="distributed" if sharded else "fp32",
            unit_update_counts={h: u.update_count for h, u in units.items()},
            identity=_identity(trainer, runtime),
            n_training_samples=trainer.n_training_samples,
            n_training_steps=trainer.n_training_steps,
            token_count_remainder=getattr(trainer, "_token_count_remainder", 0),
            local_statistics={
                name: getattr(trainer, name)
                for name in (
                    "act_freq_scores",
                    "n_forward_passes_since_fired",
                    "n_frac_active_samples",
                    "act_freq_scores_by_hook",
                    "n_forward_passes_since_fired_by_hook",
                    "n_frac_active_samples_by_hook",
                )
                if hasattr(trainer, name)
            },
        ),
        checkpoint_path / f"trainer_runtime_rank{dist.get_rank()}.pt",
    )
    if context.dp_rank == 0 and context.tp_rank == 0:
        torch.save(
            dict(
                version=1,
                domain=context.domain.name,
                hooks=list(context.domain.hooks),
                grad_scaler=trainer.grad_scaler.state_dict(),
            ),
            checkpoint_path / f"placement_state_{placement}.pt",
        )


def load_runtime_trainer_state(trainer, checkpoint_path):
    runtime = _runtime(trainer)
    if runtime is None or not dist.is_initialized():
        return
    path = Path(checkpoint_path)
    rank_path = path / f"trainer_runtime_rank{dist.get_rank()}.pt"
    if rank_path.exists():
        state = torch.load(rank_path, map_location="cpu", weights_only=True)
        identity = dict(state["identity"])
        identity.setdefault("gradient_accumulation_steps", 1)
        if identity != _identity(trainer, runtime):
            raise ValueError(
                "Static checkpoint topology or batch configuration differs"
            )
        if state.get("optimizer_backend") == "distributed":
            from sae_lens.training.gradient_window import runtime_units

            units = runtime_units(trainer)
            if not all(
                getattr(u.optimizer, "_sae_distributed_optimizer", False)
                or getattr(u.optimizer, "_sae_single_replica_fast_path", False)
                for u in units.values()
            ):
                raise ValueError(
                    "Distributed Adam checkpoint requires ddp_zero_optimizer=True; reverse conversion is unsupported"
                )
            shards = torch.load(
                path / f"distributed_adam_rank{dist.get_rank()}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if shards.keys() != units.keys():
                raise ValueError("Distributed Adam checkpoint hook ownership differs")
            for h, unit in units.items():
                if getattr(unit.optimizer, "_sae_distributed_optimizer", False):
                    unit.optimizer.load_local_checkpoint(shards[h])
                else:
                    _load_single_replica_adam(unit, shards[h])
        trainer.n_training_samples = state["n_training_samples"]
        trainer.n_training_steps = state["n_training_steps"]
        trainer._token_count_remainder = state["token_count_remainder"]

        def to_device(value):
            if torch.is_tensor(value):
                return value.to(trainer.cfg.device)
            if isinstance(value, dict):
                return {k: to_device(v) for k, v in value.items()}
            return value

        for name, value in state.get("local_statistics", {}).items():
            setattr(trainer, name, to_device(value))
        from sae_lens.training.gradient_window import runtime_units

        for h, unit in runtime_units(trainer).items():
            if "unit_update_counts" in state:
                unit.update_count = state["unit_update_counts"][h]
            else:
                # Legacy checkpoints retain Adam steps, including migration
                # into ranks that own no valid shard. Reconstruct at load only.
                steps = [
                    s["step"] for s in unit.optimizer.state.values() if "step" in s
                ]
                steps.extend(g["step"] for g in unit.optimizer.param_groups if "step" in g)
                count = torch.tensor(
                    max((int(s) for s in steps), default=0), device=trainer.cfg.device
                )
                if runtime.require_local().dp_group.size() > 1:
                    dist.all_reduce(
                        count,
                        op=dist.ReduceOp.MAX,
                        group=runtime.require_local().dp_group,
                    )
                unit.update_count = int(count.item())
            inner = getattr(unit.optimizer, "optimizer", unit.optimizer)
            if hasattr(inner, "adam_w_mode"):
                for group in inner.param_groups:
                    if group["params"] and int(group.get("step", 0)) != unit.update_count:
                        raise ValueError("FusedAdam step differs from the hook update count")
                    # Older torch checkpoints had no state on empty shards.
                    group["step"] = unit.update_count
    elif uses_exact_runtime(trainer):
        raise ValueError("Exact resume requires rank-local trainer progress files")
    context = runtime.require_local()
    placement = runtime.domains.index(context.domain)
    placement_path = path / f"placement_state_{placement}.pt"
    if placement_path.exists():
        state = torch.load(placement_path, map_location="cpu", weights_only=True)
        if state["domain"] != context.domain.name or state["hooks"] != list(
            context.domain.hooks
        ):
            raise ValueError("Checkpoint placement differs from the SAE runtime")
        if bool(state["grad_scaler"]) != trainer.grad_scaler.is_enabled():
            raise ValueError("Checkpoint AMP scaler enablement differs")
        if state["grad_scaler"]:
            trainer.grad_scaler.load_state_dict(state["grad_scaler"])
    elif rank_path.exists() or (
        len(runtime.domains) > 1 and trainer.grad_scaler.is_enabled()
    ):
        raise ValueError("Checkpoint is missing the placement AMP state")
    # Do not re-save already crossed thresholds immediately after resuming.
    progress = checkpoint_token_count(trainer)
    trainer.checkpoint_thresholds = [
        t for t in trainer.checkpoint_thresholds if t >= progress
    ]
    # Old PyTorch-DDP checkpoints may encode foreach/for-loop execution flags.
    # Preserve numerical hyperparameters and moments, while keeping the static
    # CUDA runtime's fused Adam implementation after loading those groups.
    for group in trainer.optimizer.param_groups:
        if group["params"] and group["params"][0].device.type == "cuda":
            group["fused"] = True
            group["foreach"] = None
            for parameter in group["params"]:
                if parameter in trainer.optimizer.state:
                    state = trainer.optimizer.state[parameter]
                    if torch.is_tensor(state.get("step")):
                        state["step"] = state["step"].to(parameter.device)
