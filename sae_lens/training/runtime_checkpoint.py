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
        gradient_accumulation_steps=getattr(trainer.cfg, "gradient_accumulation_steps", 1),
    )


def save_runtime_trainer_state(trainer, checkpoint_path):
    runtime = _runtime(trainer)
    if runtime is None or not dist.is_initialized():
        return
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    context = runtime.require_local()
    placement = runtime.domains.index(context.domain)
    torch.save(
        dict(
            version=1,
            identity=_identity(trainer, runtime),
            n_training_samples=trainer.n_training_samples,
            n_training_steps=trainer.n_training_steps,
            token_count_remainder=getattr(trainer, "_token_count_remainder", 0),
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
        trainer.n_training_samples = state["n_training_samples"]
        trainer.n_training_steps = state["n_training_steps"]
        trainer._token_count_remainder = state["token_count_remainder"]
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
