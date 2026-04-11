import contextlib
import json
import math
import os
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, Protocol

import torch
import torch.distributed as dist
import wandb
from safetensors.torch import save_file
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam
from tqdm.auto import tqdm

from sae_lens import __version__
from sae_lens.config import SAETrainerConfig
from sae_lens.constants import (
    ACTIVATION_SCALER_CFG_FILENAME,
    MSE_HISTORY_FILENAME,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
    TIMING_HISTORY_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.optim import CoefficientScheduler, get_lr_scheduler
from sae_lens.training.types import DataProvider
from sae_lens.util import path_or_tmp_dir


def _debug_prefix_tp(msg: str) -> None:
    if os.environ.get("SAELENS_DEBUG_PREFIX_TP") != "1":
        return
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    line = f"[prefix-debug rank{rank}] {msg}\n"
    with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
        f.write(line)
    print(line, end="", flush=True)


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


def _update_sae_lens_training_version(sae: TrainingSAE[Any]) -> None:
    """
    Make sure we record the version of SAELens used for the training run
    """
    sae.cfg.sae_lens_training_version = str(__version__)


class SaveCheckpointFn(Protocol):
    def __call__(
        self,
        checkpoint_path: Path | None,
    ) -> None: ...


Evaluator = Callable[[T_TRAINING_SAE, DataProvider, ActivationScaler], dict[str, Any]]


class SAETrainer(Generic[T_TRAINING_SAE, T_TRAINING_SAE_CONFIG]):
    """
    Trainer for Sparse Autoencoder (SAE) models.
    """

    data_provider: DataProvider
    activation_scaler: ActivationScaler
    evaluator: Evaluator[T_TRAINING_SAE] | None
    coefficient_schedulers: dict[str, CoefficientScheduler]

    def __init__(
        self,
        cfg: SAETrainerConfig,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        evaluator: Evaluator[T_TRAINING_SAE] | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
        dp_group: dist.ProcessGroup | None = None,
        token_count_weighted_dp: bool = False,
        base_sae: T_TRAINING_SAE | None = None,
    ) -> None:
        self.sae = sae
        # base_sae is the unwrapped module when sae is an FSDP/DDP wrapper.
        # Defaults to sae itself in manual mode.
        self._base_sae: T_TRAINING_SAE = base_sae if base_sae is not None else sae
        self._is_fsdp = isinstance(sae, FSDP)
        self.data_provider = data_provider
        self.evaluator = evaluator
        self.activation_scaler = ActivationScaler()
        self.save_checkpoint_fn = save_checkpoint_fn
        self.cfg = cfg
        self.dp_group = dp_group
        self.token_count_weighted_dp = token_count_weighted_dp

        self.n_training_steps: int = 0
        self.n_training_samples: int = 0
        self.started_fine_tuning: bool = False
        self.mse_history_path: Path | None = None
        self.timing_history_path: Path | None = None

        _update_sae_lens_training_version(self._base_sae)

        if self._should_write_training_metrics():
            assert self.cfg.output_path is not None
            output_path = Path(self.cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            self.mse_history_path = output_path / MSE_HISTORY_FILENAME
            self.mse_history_path.write_text("")
        if self._should_write_timing_metrics():
            assert self.cfg.output_path is not None
            output_path = Path(self.cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            self.timing_history_path = output_path / TIMING_HISTORY_FILENAME
            self.timing_history_path.write_text("")

        self.checkpoint_thresholds = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_samples,
                    math.ceil(
                        cfg.total_training_samples / (self.cfg.n_checkpoints + 1)
                    ),
                )
            )[1:]

        self.act_freq_scores = torch.zeros(self._base_sae.cfg.d_sae, device=cfg.device)
        self.n_forward_passes_since_fired = torch.zeros(
            self._base_sae.cfg.d_sae, device=cfg.device
        )
        self.n_frac_active_samples = 0

        # Optimizer is constructed over self.sae.parameters() so that in FSDP mode
        # it sees FSDP's managed parameter views; in manual mode self.sae == _base_sae.
        self.optimizer = Adam(
            sae.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )
        self.coefficient_schedulers = {}
        for name, coeff_cfg in self._base_sae.get_coefficients().items():
            if not isinstance(coeff_cfg, TrainCoefficientConfig):
                coeff_cfg = TrainCoefficientConfig(value=coeff_cfg, warm_up_steps=0)
            self.coefficient_schedulers[name] = CoefficientScheduler(
                warm_up_steps=coeff_cfg.warm_up_steps,
                final_value=coeff_cfg.value,
            )

        # Setup autocast if using
        self.grad_scaler = torch.amp.GradScaler(
            device=self.cfg.device, enabled=self.cfg.autocast
        )

        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type=self.cfg.device,
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_samples

    @property
    def base_sae(self) -> T_TRAINING_SAE:
        """Return the unwrapped base SAE module (same as self.sae in manual mode)."""
        return self._base_sae

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    @property
    def dead_neurons(self) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()

    def fit(self) -> T_TRAINING_SAE:
        self.sae.to(self.cfg.device)
        pbar = tqdm(total=self.cfg.total_training_samples, desc="Training SAE")

        if self._base_sae.cfg.normalize_activations == "expected_average_only_in":
            self.activation_scaler.estimate_scaling_factor(
                d_in=self._base_sae.cfg.d_in,
                data_provider=self.data_provider,
                n_batches_for_norm_estimate=int(1e3),
            )

        # Train loop
        while self.n_training_samples < self.cfg.total_training_samples:
            # Do a training step.
            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                _debug_prefix_tp(
                    f"trainer next() start samples={self.n_training_samples}"
                )
            self._maybe_synchronize_timing()
            data_t0 = time.perf_counter()
            batch = next(self.data_provider).to(self._base_sae.device)
            self._maybe_synchronize_timing()
            data_time_s = time.perf_counter() - data_t0
            data_timing = self._consume_data_provider_timing()
            vllm_step_time_s = data_timing["vllm_step_time_s"]
            transfer_time_s = data_timing["transfer_time_s"]
            residual_data_time_s = data_time_s - (vllm_step_time_s + transfer_time_s)
            if residual_data_time_s > 0:
                transfer_time_s += residual_data_time_s
            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                _debug_prefix_tp(f"trainer next() done batch={tuple(batch.shape)}")
            self.n_training_samples += batch.shape[0]
            scaled_batch = self.activation_scaler(batch)

            self._maybe_synchronize_timing()
            sae_t0 = time.perf_counter()
            step_output, dp_allreduce_time_s = self._train_step(
                sae=self.sae, sae_in=scaled_batch
            )
            self._maybe_synchronize_timing()
            sae_time_s = time.perf_counter() - sae_t0

            if self.cfg.logger.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._record_mse_if_needed(step_output)
            self._record_timing_if_needed(
                vllm_step_time_s=vllm_step_time_s,
                transfer_time_s=transfer_time_s,
                dp_allreduce_time_s=dp_allreduce_time_s,
                sae_time_s=sae_time_s,
            )
            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

        # fold the estimated norm scaling factor into the sae weights
        if self.activation_scaler.scaling_factor is not None:
            self._base_sae.fold_activation_norm_scaling_factor(
                self.activation_scaler.scaling_factor
            )
            self.activation_scaler.scaling_factor = None

        if self.cfg.save_final_checkpoint:
            self.save_checkpoint(checkpoint_name=f"final_{self.n_training_samples}")

        pbar.close()
        return self._base_sae

    def save_checkpoint(
        self,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        # With TP, all ranks in DP replica 0 must participate in the save collectives,
        # but only TP rank 0 writes files. Other DP replicas return early.
        tp_group = getattr(self._base_sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_rank = dist.get_rank(self.dp_group) if self.dp_group is not None else 0
        if dp_rank != 0:
            return
        checkpoint_path = None
        if self.cfg.checkpoint_path is not None or self.cfg.logger.log_to_wandb:
            with path_or_tmp_dir(self.cfg.checkpoint_path) as base_checkpoint_path:
                checkpoint_path = base_checkpoint_path / checkpoint_name
                checkpoint_path.mkdir(exist_ok=True, parents=True)

                weights_path, cfg_path = self._save_model(checkpoint_path)

                sparsity_path = checkpoint_path / SPARSITY_FILENAME
                save_file({"sparsity": self.log_feature_sparsity}, sparsity_path)

                self.save_trainer_state(checkpoint_path)

                if self.cfg.logger.log_to_wandb and tp_rank == 0:
                    self.cfg.logger.log(
                        self,
                        weights_path,
                        cfg_path,
                        sparsity_path=sparsity_path,
                        wandb_aliases=wandb_aliases,
                    )

        if self.save_checkpoint_fn is not None and tp_rank == 0:
            self.save_checkpoint_fn(checkpoint_path=checkpoint_path)

    def _save_model(self, checkpoint_path: Path) -> tuple[Path, Path]:
        """Save SAE weights, handling FSDP gather before the existing TP-aware export."""
        if not self._is_fsdp:
            return self._base_sae.save_model(str(checkpoint_path))

        # FSDP: gather full local-TP-shard state from DP replicas, then reuse
        # the existing TP-aware gather path in process_state_dict_for_saving().
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.sae, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            state_dict = self.sae.state_dict()
        # state_dict is populated only on dp_rank==0 (rank0_only=True).
        # process_state_dict_for_saving gathers TP shards via all_gather.
        self._base_sae.process_state_dict_for_saving(state_dict)

        model_weights_path = checkpoint_path / SAE_WEIGHTS_FILENAME
        cfg_path = checkpoint_path / SAE_CFG_FILENAME
        tp_group = getattr(self._base_sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        if tp_rank == 0:
            save_file(state_dict, model_weights_path)
            with open(cfg_path, "w") as f:
                json.dump(self._base_sae.cfg.to_dict(), f)
        if tp_group is not None:
            dist.barrier(group=tp_group)
        return model_weights_path, cfg_path

    def save_trainer_state(self, checkpoint_path: Path) -> None:
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        scheduler_state_dicts = {
            name: scheduler.state_dict()
            for name, scheduler in self.coefficient_schedulers.items()
        }
        optimizer_state_by_name = self._build_named_optimizer_state_for_save()
        tp_group = getattr(self._base_sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        if tp_rank != 0:
            return
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "optimizer_by_name": optimizer_state_by_name,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "n_training_samples": self.n_training_samples,
                "n_training_steps": self.n_training_steps,
                "act_freq_scores": self.act_freq_scores,
                "n_forward_passes_since_fired": self.n_forward_passes_since_fired,
                "n_frac_active_samples": self.n_frac_active_samples,
                "started_fine_tuning": self.started_fine_tuning,
                "coefficient_schedulers": scheduler_state_dicts,
                "sae_dp_mode": "fsdp" if self._is_fsdp else "manual",
            },
            str(checkpoint_path / TRAINER_STATE_FILENAME),
        )
        activation_scaler_path = checkpoint_path / ACTIVATION_SCALER_CFG_FILENAME
        self.activation_scaler.save(str(activation_scaler_path))

    def load_trainer_state(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.activation_scaler.load(checkpoint_path / ACTIVATION_SCALER_CFG_FILENAME)
        state_dict = torch.load(checkpoint_path / TRAINER_STATE_FILENAME)
        saved_mode = state_dict.get("sae_dp_mode", "manual")
        current_mode = "fsdp" if self._is_fsdp else "manual"
        if saved_mode != current_mode:
            raise ValueError(
                f"Cannot resume: checkpoint was saved with sae_dp_mode='{saved_mode}' "
                f"but current mode is '{current_mode}'. Cross-mode optimizer resume is "
                "not supported in v1."
            )
        if "optimizer_by_name" in state_dict:
            self._load_named_optimizer_state(state_dict["optimizer_by_name"])
        else:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.n_training_samples = state_dict["n_training_samples"]
        self.n_training_steps = state_dict["n_training_steps"]
        self.act_freq_scores = state_dict["act_freq_scores"]
        self.n_forward_passes_since_fired = state_dict["n_forward_passes_since_fired"]
        self.n_frac_active_samples = state_dict["n_frac_active_samples"]
        self.started_fine_tuning = state_dict["started_fine_tuning"]
        for name, scheduler_state_dict in state_dict["coefficient_schedulers"].items():
            self.coefficient_schedulers[name].load_state_dict(scheduler_state_dict)

    def _build_named_optimizer_state_for_save(self) -> dict[str, dict[str, Any]]:
        optimizer_state_by_name: dict[str, dict[str, Any]] = {}
        for name, param in self._base_sae.named_parameters():
            state = self.optimizer.state.get(param)
            if not state:
                continue
            optimizer_state_by_name[name] = {
                key: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
                for key, value in state.items()
            }
        self._base_sae.process_named_optimizer_state_for_saving(optimizer_state_by_name)
        return optimizer_state_by_name

    def _load_named_optimizer_state(
        self, optimizer_state_by_name: dict[str, dict[str, Any]]
    ) -> None:
        optimizer_state_by_name = deepcopy(optimizer_state_by_name)
        self._base_sae.process_named_optimizer_state_for_loading(optimizer_state_by_name)
        self.optimizer.state.clear()

        named_params = dict(self._base_sae.named_parameters())
        for name, state in optimizer_state_by_name.items():
            if name not in named_params:
                continue
            param = named_params[name]
            loaded_state: dict[str, Any] = {}
            for key, value in state.items():
                if torch.is_tensor(value):
                    target_dtype = (
                        param.dtype if value.is_floating_point() and value.ndim > 0 else value.dtype
                    )
                    loaded_state[key] = value.to(device=param.device, dtype=target_dtype)
                else:
                    loaded_state[key] = deepcopy(value)
            self.optimizer.state[param] = loaded_state

    def _train_step(
        self,
        sae: T_TRAINING_SAE,
        sae_in: torch.Tensor,
    ) -> tuple[TrainStepOutput, float]:
        sae.train()
        dp_allreduce_time_s = 0.0
        if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
            _debug_prefix_tp("train_step forward start")

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            # Sync act_freq_scores across DP replicas before logging/reset.
            if self.dp_group is not None:
                dist.all_reduce(self.act_freq_scores, op=dist.ReduceOp.SUM, group=self.dp_group)
                dp_size = dist.get_world_size(self.dp_group)
                self.n_frac_active_samples *= dp_size
            if self.cfg.logger.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        step_input = TrainStepInput(
            sae_in=sae_in,
            dead_neuron_mask=self.dead_neurons,
            coefficients=self.get_coefficients(),
            n_training_steps=self.n_training_steps,
            is_logging_step=self._is_logging_step(),
        )
        # In FSDP mode self.sae is the FSDP wrapper; calling self.sae(step_input)
        # routes through forward() which triggers FSDP's parameter-gather lifecycle.
        # In manual mode self.sae == self._base_sae and the dispatch in
        # TrainingSAE.forward() calls training_forward_pass identically.
        with self.autocast_if_enabled:
            train_step_output = self.sae(step_input)  # type: ignore[arg-type]

            with torch.no_grad():
                # calling .bool() should be equivalent to .abs() > 0, and work with coo tensors
                firing_feats = train_step_output.feature_acts.bool().float()
                did_fire = firing_feats.sum(-2).bool()
                if did_fire.is_sparse:
                    did_fire = did_fire.to_dense()
                # Sync did_fire across DP replicas: a feature counts as fired if any replica saw it.
                if self.dp_group is not None:
                    did_fire_int = did_fire.to(torch.int32)
                    dist.all_reduce(did_fire_int, op=dist.ReduceOp.MAX, group=self.dp_group)
                    did_fire = did_fire_int.bool()
                self.n_forward_passes_since_fired += 1
                self.n_forward_passes_since_fired[did_fire] = 0
                self.act_freq_scores += firing_feats.sum(0)
                self.n_frac_active_samples += self.cfg.train_batch_size_samples
        if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
            _debug_prefix_tp("train_step forward done")

        # Grad scaler will rescale gradients if autocast is enabled
        self.grad_scaler.scale(
            train_step_output.loss
        ).backward()  # loss.backward() if not autocasting
        if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
            _debug_prefix_tp("backward done")
        self.grad_scaler.unscale_(self.optimizer)  # needed to clip correctly

        if self._is_fsdp:
            # FSDP handles gradient reduction across dp_group during backward().
            # No manual flat all-reduce needed.
            pass
        elif self.dp_group is not None:
            # Manual mode: AllReduce gradients across DP group.
            # Flatten all gradients into one tensor to issue a single allreduce.
            grads = [p.grad for p in self._base_sae.parameters() if p.grad is not None]
            if grads:
                flat = torch.cat([g.view(-1) for g in grads])
                if self.token_count_weighted_dp:
                    local_tokens = float(sae_in.shape[0])
                    flat.mul_(local_tokens)
                    dp_t0 = time.perf_counter()
                    dist.all_reduce(flat, group=self.dp_group)
                    tokens_t = torch.tensor(local_tokens, device=flat.device)
                    dist.all_reduce(tokens_t, op=dist.ReduceOp.SUM, group=self.dp_group)
                    dp_allreduce_time_s += time.perf_counter() - dp_t0
                    flat.div_(tokens_t.item())
                else:
                    dp_size = dist.get_world_size(self.dp_group)
                    dp_t0 = time.perf_counter()
                    dist.all_reduce(flat, group=self.dp_group)
                    dp_allreduce_time_s += time.perf_counter() - dp_t0
                    flat /= dp_size
                offset = 0
                for g in grads:
                    numel = g.numel()
                    g.copy_(flat[offset : offset + numel].view_as(g))
                    offset += numel

        self._base_sae.sync_tensor_parallel_gradients()
        if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
            _debug_prefix_tp("grad sync done")
        # In FSDP mode pass dp_group so the norm all-reduce covers the DP shard dimension.
        dp_group_for_clip = self.dp_group if self._is_fsdp else None
        self._base_sae.clip_grad_norm_(1.0, dp_group=dp_group_for_clip)
        self.grad_scaler.step(
            self.optimizer
        )  # just ctx.optimizer.step() if not autocasting
        self.grad_scaler.update()
        if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
            _debug_prefix_tp("optimizer step done")

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        for scheduler in self.coefficient_schedulers.values():
            scheduler.step()

        return train_step_output, dp_allreduce_time_s

    def _is_logging_step(self) -> bool:
        return (
            self.cfg.logger.log_to_wandb
            and (self.n_training_steps + 1) % self.cfg.logger.wandb_log_frequency == 0
        )

    def _should_write_training_metrics(self) -> bool:
        if self.cfg.output_path is None or self.cfg.save_mse_every_n_steps <= 0:
            return False
        tp_group = getattr(self._base_sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_rank = dist.get_rank(self.dp_group) if self.dp_group is not None else 0
        return tp_rank == 0 and dp_rank == 0

    def _should_write_timing_metrics(self) -> bool:
        if self.cfg.output_path is None or self.cfg.save_timing_every_n_steps <= 0:
            return False
        tp_group = getattr(self._base_sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_rank = dist.get_rank(self.dp_group) if self.dp_group is not None else 0
        return tp_rank == 0 and dp_rank == 0

    def _maybe_synchronize_timing(self) -> None:
        if not self.cfg.synchronize_timing:
            return
        if not self._base_sae.device.type.startswith("cuda"):
            return
        torch.cuda.synchronize(self._base_sae.device)

    def _consume_data_provider_timing(self) -> dict[str, float]:
        consume_timing = getattr(self.data_provider, "consume_last_data_timing", None)
        if consume_timing is None:
            return {
                "vllm_step_time_s": 0.0,
                "transfer_time_s": 0.0,
            }
        timing = consume_timing()
        return {
            "vllm_step_time_s": float(timing.get("vllm_step_time_s", 0.0)),
            "transfer_time_s": float(timing.get("transfer_time_s", 0.0)),
        }

    @torch.no_grad()
    def _record_mse_if_needed(self, step_output: TrainStepOutput) -> None:
        if self.mse_history_path is None:
            return
        if (self.n_training_steps + 1) % self.cfg.save_mse_every_n_steps != 0:
            return

        mse_loss = step_output.losses.get("mse_loss")
        if mse_loss is None:
            mse_loss = (step_output.sae_out - step_output.sae_in).pow(2).mean()

        record = {
            "step": self.n_training_steps + 1,
            "n_training_samples": self.n_training_samples,
            "mse_loss": _unwrap_item(mse_loss),
            "overall_loss": _unwrap_item(step_output.loss),
        }
        if "auxiliary_reconstruction_loss" in step_output.losses:
            record["auxiliary_reconstruction_loss"] = _unwrap_item(
                step_output.losses["auxiliary_reconstruction_loss"]
            )

        with open(self.mse_history_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    @torch.no_grad()
    def _record_timing_if_needed(
        self,
        vllm_step_time_s: float,
        transfer_time_s: float,
        dp_allreduce_time_s: float,
        sae_time_s: float,
    ) -> None:
        if self.timing_history_path is None:
            return
        if (self.n_training_steps + 1) % self.cfg.save_timing_every_n_steps != 0:
            return

        data_time_s = vllm_step_time_s + transfer_time_s
        record = {
            "step": self.n_training_steps + 1,
            "n_training_samples": self.n_training_samples,
            "vllm_step_time_s": vllm_step_time_s,
            "transfer_time_s": transfer_time_s,
            "data_time_s": data_time_s,
            "vllm_time_s": data_time_s,
            "dp_allreduce_time_s": dp_allreduce_time_s,
            "sae_time_s": sae_time_s,
            "step_time_s": data_time_s + sae_time_s,
        }
        with open(self.timing_history_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput):
        if self._is_logging_step():
            wandb.log(
                self._build_train_step_log_dict(
                    output=step_output,
                    n_training_samples=self.n_training_samples,
                ),
                step=self.n_training_steps,
            )

    @torch.no_grad()
    def get_coefficients(self) -> dict[str, float]:
        return {
            name: scheduler.value
            for name, scheduler in self.coefficient_schedulers.items()
        }

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_samples: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        # metrics for currents acts
        l0 = feature_acts.bool().float().sum(-1).to_dense().mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance_legacy = 1 - per_token_l2_loss / total_variance
        explained_variance = 1 - per_token_l2_loss.mean() / total_variance.mean()

        log_dict = {
            # losses
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance_legacy": explained_variance_legacy.mean().item(),
            "metrics/explained_variance_legacy_std": explained_variance_legacy.std().item(),
            "metrics/explained_variance": explained_variance.item(),
            "metrics/l0": l0.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "details/current_learning_rate": current_learning_rate,
            "details/n_training_samples": n_training_samples,
            **{
                f"details/{name}_coefficient": scheduler.value
                for name, scheduler in self.coefficient_schedulers.items()
            },
        }
        for loss_name, loss_value in output.losses.items():
            log_dict[f"losses/{loss_name}"] = _unwrap_item(loss_value)

        for metric_name, metric_value in output.metrics.items():
            log_dict[f"metrics/{metric_name}"] = _unwrap_item(metric_value)

        return log_dict

    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        if (self.n_training_steps + 1) % (
            self.cfg.logger.wandb_log_frequency
            * self.cfg.logger.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            eval_metrics = (
                self.evaluator(self._base_sae, self.data_provider, self.activation_scaler)
                if self.evaluator is not None
                else {}
            )
            for key, value in self._base_sae.log_histograms().items():
                eval_metrics[key] = wandb.Histogram(value)  # type: ignore

            wandb.log(
                eval_metrics,
                step=self.n_training_steps,
            )
            self.sae.train()

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        log_feature_sparsity = _log_feature_sparsity(self.feature_sparsity)
        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())  # type: ignore
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        self.act_freq_scores = torch.zeros(
            self._base_sae.cfg.d_sae,
            device=self.cfg.device,
        )
        self.n_frac_active_samples = 0

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_samples > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(checkpoint_name=str(self.n_training_samples))
            self.checkpoint_thresholds.pop(0)

    @torch.no_grad()
    def _update_pbar(
        self,
        step_output: TrainStepOutput,
        pbar: tqdm,  # type: ignore
        update_interval: int = 8,
    ):
        if self.n_training_steps % update_interval == 0:
            loss_strs = " | ".join(
                f"{loss_name}: {_unwrap_item(loss_value):.5f}"
                for loss_name, loss_value in step_output.losses.items()
            )
            pbar.set_description(f"{self.n_training_steps}| {loss_strs}")
            pbar.update(update_interval * self.cfg.train_batch_size_samples)


def _unwrap_item(
    item: float | int | torch.Tensor | Callable[[], float | int | torch.Tensor],
) -> float | int:
    if callable(item):
        item = item()
    return item.item() if isinstance(item, torch.Tensor) else item
