from __future__ import annotations

import contextlib
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm.auto import tqdm

from sae_lens.constants import (
    MSE_HISTORY_FILENAME,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
    TIMING_HISTORY_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.profiling import cuda_nvtx_range, nccl_nvtx_range
from sae_lens.saes.sae import TrainingSAE, TrainStepInput, TrainStepOutput
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.optim import get_lr_scheduler
from sae_lens.training.sae_trainer import (
    SaveCheckpointFn,
    _log_feature_sparsity,
    _unwrap_item,
)
from sae_lens.training.types import DataProvider

MULTI_SAE_MANIFEST_FILENAME = "multi_sae_manifest.json"
MULTI_SAE_FSDP_OPTIMIZER_STATE_FILENAME_TEMPLATE = (
    "multi_fsdp_optimizer_state_rank{rank}.pt"
)
MULTI_SAE_FSDP_OPTIMIZER_STATE_FORMAT = "multi_fsdp_raw_rank_sharded_v1"


def sanitize_hook_name_for_path(hook_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in hook_name).strip("_")


class MultiSAETrainer:
    def __init__(
        self,
        *,
        hook_names: list[str],
        sae_by_hook: dict[str, Any],
        base_sae_by_hook: dict[str, TrainingSAE[Any]],
        data_provider: DataProvider,
        save_checkpoint_fn: SaveCheckpointFn | None,
        cfg: Any,
        dp_group: dist.ProcessGroup | None,
        token_count_weighted_dp: bool,
        sae_dp_mode: str,
        backward_mode: str = "combined",
        seed_mode: str = "same",
    ) -> None:
        self.hook_names = hook_names
        self.sae_by_hook = sae_by_hook
        self.base_sae_by_hook = base_sae_by_hook
        self.data_provider = data_provider
        self.save_checkpoint_fn = save_checkpoint_fn
        self.cfg = cfg
        self.dp_group = dp_group
        self.token_count_weighted_dp = token_count_weighted_dp
        self.sae_dp_mode = sae_dp_mode
        self.backward_mode = backward_mode
        self.seed_mode = seed_mode
        self.backward_order: Literal["forward", "reverse", "largest_first"] = (
            getattr(cfg, "multi_sae_backward_order", "forward")
        )
        self.stats_sync_mode: Literal["immediate", "deferred", "periodic"] = getattr(
            cfg, "multi_sae_stats_sync_mode", "immediate"
        )
        self.stats_sync_interval: int = int(
            getattr(cfg, "multi_sae_stats_sync_interval", 1)
        )
        self._is_fsdp = sae_dp_mode == "fsdp"
        self._is_ddp = sae_dp_mode == "ddp"
        if not (self._is_fsdp or self._is_ddp):
            raise ValueError("MultiSAETrainer requires sae_dp_mode='ddp' or 'fsdp'")
        if self._is_fsdp and self.dp_group is None:
            raise ValueError("MultiSAETrainer with FSDP requires a DP process group")
        if self.backward_mode not in ("combined", "sequential"):
            raise ValueError("backward_mode must be 'combined' or 'sequential'")
        if self.backward_order not in ("forward", "reverse", "largest_first"):
            raise ValueError(
                "multi_sae_backward_order must be 'forward', 'reverse', or 'largest_first'"
            )
        if self.stats_sync_mode not in ("immediate", "deferred", "periodic"):
            raise ValueError(
                "multi_sae_stats_sync_mode must be 'immediate', 'deferred', or 'periodic'"
            )
        if self.stats_sync_interval < 1:
            raise ValueError("multi_sae_stats_sync_interval must be >= 1")

        params: list[torch.nn.Parameter] = []
        for hook_name in self.hook_names:
            params.extend(list(self.sae_by_hook[hook_name].parameters()))
        self.optimizer = Adam(
            params,
            lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        self.lr_scheduler = get_lr_scheduler(
            scheduler_name=cfg.lr_scheduler_name,
            optimizer=self.optimizer,
            training_steps=cfg.total_training_steps,
            lr=cfg.lr,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=cfg.autocast and torch.cuda.is_available(),
        )
        self.autocast_if_enabled = torch.autocast(
            device_type=torch.device(cfg.device).type,
            dtype=torch.bfloat16,
            enabled=cfg.autocast,
        )

        self.activation_scaler_by_hook = {
            hook_name: ActivationScaler() for hook_name in self.hook_names
        }
        self.act_freq_scores_by_hook = {
            hook_name: torch.zeros(
                self.base_sae_by_hook[hook_name].cfg.d_sae,
                device=cfg.device,
            )
            for hook_name in self.hook_names
        }
        self.n_forward_passes_since_fired_by_hook = {
            hook_name: torch.zeros(
                self.base_sae_by_hook[hook_name].cfg.d_sae,
                device=cfg.device,
            )
            for hook_name in self.hook_names
        }
        self.n_frac_active_samples_by_hook = {
            hook_name: 0 for hook_name in self.hook_names
        }
        self._pending_did_fire_max_by_hook = {
            hook_name: torch.zeros(
                self.base_sae_by_hook[hook_name].cfg.d_sae,
                device=cfg.device,
                dtype=torch.int32,
            )
            for hook_name in self.hook_names
        }
        self._pending_sample_count_by_hook = {
            hook_name: 0.0 for hook_name in self.hook_names
        }
        self._pending_step_count_by_hook = {
            hook_name: 0 for hook_name in self.hook_names
        }
        self._trainable_param_bytes_by_hook = {
            hook_name: sum(
                p.numel() * p.element_size()
                for p in self.base_sae_by_hook[hook_name].parameters()
                if p.requires_grad
            )
            for hook_name in self.hook_names
        }

        self.n_training_steps = 0
        self.n_training_samples = 0
        self._t_ready: float = time.time()
        self.mse_history_path: Path | None = None
        self.timing_history_path: Path | None = None
        self.memory_history_path: Path | None = None
        self.checkpoint_thresholds: list[int] = []
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

        should_write_logs = self._is_metric_writer_rank()
        if (
            should_write_logs
            and cfg.output_path is not None
            and cfg.save_mse_every_n_steps > 0
        ):
            output_path = Path(cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            self.mse_history_path = output_path / MSE_HISTORY_FILENAME
            self.mse_history_path.write_text("")
        if (
            should_write_logs
            and cfg.output_path is not None
            and cfg.save_timing_every_n_steps > 0
        ):
            output_path = Path(cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            self.timing_history_path = output_path / TIMING_HISTORY_FILENAME
            self.timing_history_path.write_text("")

        # Memory profiling: each rank writes its own file (global_rank for multi-node safety).
        if dist.is_available() and dist.is_initialized():
            _global_rank = dist.get_rank()
        else:
            _global_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._memory_rank: int = _global_rank
        self._profile_memory: bool = (
            cfg.output_path is not None
            and getattr(cfg, "save_memory_every_n_steps", 0) > 0
        )
        if self._profile_memory:
            output_path = Path(cfg.output_path)  # type: ignore[arg-type]
            output_path.mkdir(exist_ok=True, parents=True)
            self.memory_history_path = (
                output_path / f"memory_history_rank{_global_rank}.jsonl"
            )
            self.memory_history_path.write_text("")

    def _dp_world_size(self) -> int:
        if (
            self.dp_group is None
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return 1
        return dist.get_world_size(self.dp_group)

    def _dp_rank(self) -> int:
        if (
            self.dp_group is None
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return 0
        return dist.get_rank(self.dp_group)

    def _tp_group(self) -> dist.ProcessGroup | None:
        if not self.hook_names:
            return None
        first_sae = self.base_sae_by_hook[self.hook_names[0]]
        return getattr(first_sae, "_tp_group", None)

    def _tp_rank(self) -> int:
        tp_group = self._tp_group()
        if dist.is_available() and dist.is_initialized() and tp_group is not None:
            return dist.get_rank(tp_group)

        # Fallback for runs where SAE TP metadata is available from distributed
        # topology, but the SAE instance itself does not expose _tp_group.
        if dist.is_available() and dist.is_initialized():
            try:
                import sae_lens.distributed_v2 as v2_mod

                if getattr(v2_mod, "_initialized", False) and v2_mod.is_consumer():
                    sae_tp_rank = int(v2_mod.get_sae_tp_rank())
                    if sae_tp_rank >= 0:
                        return sae_tp_rank
            except ImportError:
                pass

            try:
                from sae_lens.distributed import get_sae_tp_group

                fallback_group = get_sae_tp_group()
                if fallback_group is not None:
                    return dist.get_rank(fallback_group)
            except ImportError:
                pass
        return 0

    def _is_metric_writer_rank(self) -> bool:
        return self._dp_rank() == 0 and self._tp_rank() == 0

    def _tp_barrier(self) -> None:
        tp_group = self._tp_group()
        if tp_group is not None:
            dist.barrier(group=tp_group)

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._dp_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group)
        return tensor

    def _all_reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._dp_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.dp_group)
        return tensor

    @property
    def log_feature_sparsity_by_hook(self) -> dict[str, torch.Tensor]:
        return {
            hook_name: _log_feature_sparsity(
                self.act_freq_scores_by_hook[hook_name]
                / max(1, self.n_frac_active_samples_by_hook[hook_name])
            )
            for hook_name in self.hook_names
        }

    def fit(
        self,
        quiesce_request_path: Path | str | None = None,
        quiesce_ack_path: Path | str | None = None,
    ) -> dict[str, TrainingSAE[Any]]:
        pbar = tqdm(total=self.cfg.total_training_samples, desc="Training Multi SAE")
        while self.n_training_samples < self.cfg.total_training_samples:
            step_wall_t0 = time.perf_counter()
            self._maybe_synchronize_timing()
            with cuda_nvtx_range("multi_sae:data_fetch"):
                try:
                    batch_by_hook = next(self.data_provider)
                except StopIteration:
                    break
            if not isinstance(batch_by_hook, dict):
                raise TypeError(
                    "MultiSAETrainer expected data_provider to yield dict batches"
                )
            local_ns = {hook: acts.shape[0] for hook, acts in batch_by_hook.items()}
            if len(set(local_ns.values())) != 1:
                raise RuntimeError(f"Multi-layer activation sizes diverged: {local_ns}")
            local_n = next(iter(local_ns.values()))
            self._maybe_synchronize_timing()
            data_timing = self._consume_data_provider_timing()

            scaled_batch_by_hook = {
                hook_name: self.activation_scaler_by_hook[hook_name](
                    batch_by_hook[hook_name].to(self.cfg.device)
                )
                for hook_name in self.hook_names
            }
            self.n_training_samples += local_n

            self._maybe_synchronize_timing()
            sae_t0 = time.perf_counter()
            if self._profile_memory:
                torch.cuda.reset_peak_memory_stats(self.cfg.device)
            with cuda_nvtx_range("multi_sae:train_step"):
                outputs, sae_phase_timing = self._train_step(scaled_batch_by_hook, local_n)
            self._maybe_synchronize_timing()
            sae_time_s = time.perf_counter() - sae_t0

            if self._profile_memory:
                memory_stats: dict[str, float] = {
                    "peak_step_allocated_mb": torch.cuda.max_memory_allocated(self.cfg.device) / 1024**2,
                    "peak_step_reserved_mb": torch.cuda.max_memory_reserved(self.cfg.device) / 1024**2,
                }
            else:
                memory_stats = {}

            self._record_mse_if_needed(outputs, local_n)
            vllm_step_time_s = data_timing["vllm_step_time_s"]
            transfer_time_s = data_timing["transfer_time_s"]
            timing = self._global_timing_if_needed(
                vllm_step_time_s=vllm_step_time_s,
                transfer_time_s=transfer_time_s,
                sae_time_s=sae_time_s,
                wall_time_s=time.perf_counter() - step_wall_t0,
            )
            self._record_timing_if_needed(
                **timing,
                **sae_phase_timing,
            )
            self._record_memory_if_needed(memory_stats)
            self.n_training_steps += 1
            self.lr_scheduler.step()
            self._checkpoint_if_needed()
            pbar.update(local_n)
            if self.n_training_steps % 8 == 0 and outputs:
                avg_loss = sum(_unwrap_item(o.loss) for o in outputs.values()) / len(
                    outputs
                )
                pbar.set_description(
                    f"{self.n_training_steps}| avg_loss: {avg_loss:.5f}"
                )

            if quiesce_request_path is not None and Path(quiesce_request_path).exists():
                self.save_checkpoint(checkpoint_name=f"quiesce_{self.n_training_samples}")
                if quiesce_ack_path is not None and self._is_metric_writer_rank():
                    Path(quiesce_ack_path).touch()
                break

        pbar.close()
        # Ensure periodic/deferred stats are flushed before final save/logging.
        self._sync_deferred_stats_if_needed(force=True)
        if self.cfg.save_final_checkpoint:
            self.save_checkpoint(checkpoint_name=f"final_{self.n_training_samples}")
        return self.base_sae_by_hook

    def _train_step(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        for sae in self.sae_by_hook.values():
            sae.train()

        self.optimizer.zero_grad(set_to_none=True)
        outputs: dict[str, TrainStepOutput] = {}
        phase_timing = {
            "sae_forward_time_s": 0.0,
            "sae_stats_sync_time_s": 0.0,
            "sae_backward_time_s": 0.0,
            "sae_post_backward_time_s": 0.0,
            "sae_optimizer_time_s": 0.0,
        }
        loss_scale = 1.0
        if self._is_ddp and self.token_count_weighted_dp:
            local_n_t = torch.tensor(float(local_n), device=self.cfg.device)
            global_n_t = local_n_t.clone()
            self._all_reduce_sum(global_n_t)
            global_n = float(global_n_t.item())
            if global_n == 0:
                for hook_name in self.hook_names:
                    dummy = torch.zeros(
                        1,
                        self.base_sae_by_hook[hook_name].cfg.d_in,
                        device=self.cfg.device,
                        dtype=batch_by_hook[hook_name].dtype,
                    )
                    output = self._forward_one(hook_name, dummy)
                    self.grad_scaler.scale(output.loss * 0.0).backward()
                self.grad_scaler.unscale_(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                return outputs, phase_timing
            dp_world_size = self._dp_world_size()
            loss_scale = dp_world_size * float(local_n) / global_n
        if self.backward_mode == "combined":
            return self._train_step_combined_backward(
                batch_by_hook, local_n, loss_scale, phase_timing
            )
        return self._train_step_sequential_backward(
            batch_by_hook, local_n, loss_scale, phase_timing
        )

    def _train_step_sequential_backward(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
        loss_scale: float,
        phase_timing: dict[str, float],
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        outputs: dict[str, TrainStepOutput] = {}
        scaled_loss_by_hook: dict[str, torch.Tensor] = {}
        # Phase A: run forward for all hooks first.
        for hook_name in self.hook_names:
            acts = batch_by_hook[hook_name]
            if local_n == 0:
                dummy = torch.zeros(
                    1,
                    self.base_sae_by_hook[hook_name].cfg.d_in,
                    device=self.cfg.device,
                    dtype=acts.dtype,
                )
                t_fwd = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:forward"):
                    output = self._forward_one(hook_name, dummy)
                phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
                outputs[hook_name] = output
                scaled_loss_by_hook[hook_name] = output.loss * 0.0
            else:
                t_fwd = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:forward"):
                    output = self._forward_one(hook_name, acts)
                phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
                outputs[hook_name] = output
                t_stats = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:stats_sync"):
                    self._update_stats(hook_name, output, local_n)
                phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
                # SAE parameters are disjoint across hooks, so summing per-layer
                # losses gives the same per-parameter gradients as independent
                # single-layer training. Dividing by num_layers changes gradient
                # clipping behavior and breaks equivalence.
                scaled_loss_by_hook[hook_name] = output.loss * loss_scale

        # Phase B: run backward in configured order.
        for hook_name in self._ordered_hook_names_for_backward():
            t_bwd = time.perf_counter()
            with nccl_nvtx_range(
                f"nccl:multi_sae_{self.sae_dp_mode}_backward", self.dp_group
            ):
                with cuda_nvtx_range(f"multi_sae:{hook_name}:backward"):
                    self.grad_scaler.scale(scaled_loss_by_hook[hook_name]).backward()
            phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd

        t_post = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_unscale"):
            self.grad_scaler.unscale_(self.optimizer)
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            with cuda_nvtx_range(f"multi_sae:{hook_name}:tp_sync"):
                base_sae.sync_tensor_parallel_gradients()
            with cuda_nvtx_range(f"multi_sae:{hook_name}:clip_grad"):
                base_sae.clip_grad_norm_(
                    1.0,
                    dp_group=self.dp_group if self._is_fsdp else None,
                )
        phase_timing["sae_post_backward_time_s"] += time.perf_counter() - t_post
        t_opt = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_step"):
            self.grad_scaler.step(self.optimizer)
        with cuda_nvtx_range("multi_sae:scaler_update"):
            self.grad_scaler.update()
        t_stats = time.perf_counter()
        with cuda_nvtx_range("multi_sae:stats_sync_tail"):
            self._sync_deferred_stats_if_needed(force=False)
        phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
        phase_timing["sae_optimizer_time_s"] += time.perf_counter() - t_opt
        return outputs, phase_timing

    def _train_step_combined_backward(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
        loss_scale: float,
        phase_timing: dict[str, float],
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        outputs: dict[str, TrainStepOutput] = {}
        scaled_losses: list[torch.Tensor] = []
        for hook_name in self.hook_names:
            acts = batch_by_hook[hook_name]
            if local_n == 0:
                dummy = torch.zeros(
                    1,
                    self.base_sae_by_hook[hook_name].cfg.d_in,
                    device=self.cfg.device,
                    dtype=acts.dtype,
                )
                t_fwd = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:forward"):
                    output = self._forward_one(hook_name, dummy)
                phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
                outputs[hook_name] = output
                scaled_losses.append(output.loss * 0.0)
            else:
                t_fwd = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:forward"):
                    output = self._forward_one(hook_name, acts)
                phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
                outputs[hook_name] = output
                t_stats = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:stats_sync"):
                    self._update_stats(hook_name, output, local_n)
                phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
                scaled_losses.append(output.loss * loss_scale)

        total_loss = sum(scaled_losses)
        t_bwd = time.perf_counter()
        with nccl_nvtx_range(
            f"nccl:multi_sae_{self.sae_dp_mode}_combined_backward", self.dp_group
        ):
            with cuda_nvtx_range("multi_sae:combined_backward"):
                self.grad_scaler.scale(total_loss).backward()
        phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd

        t_post = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_unscale"):
            self.grad_scaler.unscale_(self.optimizer)
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            with cuda_nvtx_range(f"multi_sae:{hook_name}:tp_sync"):
                base_sae.sync_tensor_parallel_gradients()
            with cuda_nvtx_range(f"multi_sae:{hook_name}:clip_grad"):
                base_sae.clip_grad_norm_(
                    1.0,
                    dp_group=self.dp_group if self._is_fsdp else None,
                )
        phase_timing["sae_post_backward_time_s"] += time.perf_counter() - t_post
        t_opt = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_step"):
            self.grad_scaler.step(self.optimizer)
        with cuda_nvtx_range("multi_sae:scaler_update"):
            self.grad_scaler.update()
        t_stats = time.perf_counter()
        with cuda_nvtx_range("multi_sae:stats_sync_tail"):
            self._sync_deferred_stats_if_needed(force=False)
        phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
        phase_timing["sae_optimizer_time_s"] += time.perf_counter() - t_opt
        return outputs, phase_timing

    def _forward_one(self, hook_name: str, acts: torch.Tensor) -> TrainStepOutput:
        step_input = TrainStepInput(
            sae_in=acts,
            dead_neuron_mask=(
                self.n_forward_passes_since_fired_by_hook[hook_name]
                > self.cfg.dead_feature_window
            ).bool(),
            coefficients={},
            n_training_steps=self.n_training_steps,
            is_logging_step=False,
        )
        with self.autocast_if_enabled:
            context = (
                nccl_nvtx_range(
                    "nccl:multi_sae_fsdp_forward_param_all_gather", self.dp_group
                )
                if self._is_fsdp
                else contextlib.nullcontext()
            )
            with context:
                return self.sae_by_hook[hook_name](step_input)

    def _ordered_hook_names_for_backward(self) -> list[str]:
        if self.backward_order == "forward":
            return list(self.hook_names)
        if self.backward_order == "reverse":
            return list(reversed(self.hook_names))
        if self.backward_order == "largest_first":
            idx_by_hook = {hook_name: idx for idx, hook_name in enumerate(self.hook_names)}
            return sorted(
                self.hook_names,
                key=lambda hook_name: (
                    -self._trainable_param_bytes_by_hook[hook_name],
                    idx_by_hook[hook_name],
                ),
            )
        return list(self.hook_names)

    @torch.no_grad()
    def _update_stats(
        self, hook_name: str, output: TrainStepOutput, local_n: int
    ) -> None:
        firing_feats = output.feature_acts.bool().float()
        did_fire = firing_feats.sum(-2).bool()
        if did_fire.is_sparse:
            did_fire = did_fire.to_dense()
        did_fire_int = did_fire.to(torch.int32).contiguous()
        self.act_freq_scores_by_hook[hook_name] += firing_feats.sum(0)
        if self.stats_sync_mode == "immediate":
            self._apply_stats_from_global(
                hook_name=hook_name,
                global_did_fire_int=self._all_reduce_max(did_fire_int),
                global_sample_count=float(
                    self._all_reduce_sum(
                        torch.tensor(float(local_n), device=self.cfg.device)
                    ).item()
                ),
                step_increment=1,
            )
            return

        # Deferred/periodic path: accumulate locally and reduce later.
        self._pending_did_fire_max_by_hook[hook_name] = torch.maximum(
            self._pending_did_fire_max_by_hook[hook_name],
            did_fire_int,
        )
        self._pending_sample_count_by_hook[hook_name] += float(local_n)
        self._pending_step_count_by_hook[hook_name] += 1

    @torch.no_grad()
    def _apply_stats_from_global(
        self,
        *,
        hook_name: str,
        global_did_fire_int: torch.Tensor,
        global_sample_count: float,
        step_increment: int,
    ) -> None:
        did_fire = global_did_fire_int.bool()
        self.n_forward_passes_since_fired_by_hook[hook_name] += step_increment
        self.n_forward_passes_since_fired_by_hook[hook_name][did_fire] = 0
        self.n_frac_active_samples_by_hook[hook_name] += int(global_sample_count)

    @torch.no_grad()
    def _sync_deferred_stats_if_needed(self, *, force: bool) -> None:
        if self.stats_sync_mode == "immediate":
            return

        if self.stats_sync_mode == "periodic" and not force:
            if (self.n_training_steps + 1) % self.stats_sync_interval != 0:
                return

        hooks_to_sync = [
            hook_name
            for hook_name in self.hook_names
            if self._pending_step_count_by_hook[hook_name] > 0
        ]
        if not hooks_to_sync:
            return

        did_fire_stack = torch.stack(
            [self._pending_did_fire_max_by_hook[hook_name] for hook_name in hooks_to_sync],
            dim=0,
        )
        sample_count_stack = torch.tensor(
            [self._pending_sample_count_by_hook[hook_name] for hook_name in hooks_to_sync],
            device=self.cfg.device,
            dtype=torch.float32,
        )
        step_count_by_hook = {
            hook_name: self._pending_step_count_by_hook[hook_name]
            for hook_name in hooks_to_sync
        }

        with nccl_nvtx_range("nccl:multi_sae_stats_batched_max", self.dp_group):
            self._all_reduce_max(did_fire_stack)
        with nccl_nvtx_range("nccl:multi_sae_stats_batched_sum", self.dp_group):
            self._all_reduce_sum(sample_count_stack)

        for idx, hook_name in enumerate(hooks_to_sync):
            self._apply_stats_from_global(
                hook_name=hook_name,
                global_did_fire_int=did_fire_stack[idx],
                global_sample_count=float(sample_count_stack[idx].item()),
                step_increment=step_count_by_hook[hook_name],
            )
            self._pending_did_fire_max_by_hook[hook_name].zero_()
            self._pending_sample_count_by_hook[hook_name] = 0.0
            self._pending_step_count_by_hook[hook_name] = 0

    def save_final(self, output_path: str) -> None:
        base_output = Path(output_path)
        base_output.mkdir(exist_ok=True, parents=True)
        manifest = self._manifest()
        if self._is_metric_writer_rank():
            with open(base_output / MULTI_SAE_MANIFEST_FILENAME, "w") as f:
                json.dump(manifest, f)

        for hook_name in self.hook_names:
            self._save_one_final(base_output, hook_name)

    def _manifest(self) -> dict[str, Any]:
        return {
            "format": "multi_independent_sae_v1",
            "hook_names": self.hook_names,
            "hook_to_dir": {
                hook_name: sanitize_hook_name_for_path(hook_name)
                for hook_name in self.hook_names
            },
            "shared_hyperparams": True,
            "sae_dp_mode": self.sae_dp_mode,
            "backward_mode": self.backward_mode,
            "backward_order": self.backward_order,
            "stats_sync_mode": self.stats_sync_mode,
            "stats_sync_interval": self.stats_sync_interval,
            "seed_mode": self.seed_mode,
        }

    def _save_one_final(self, base_output: Path, hook_name: str) -> None:
        sae = self.sae_by_hook[hook_name]
        base_sae = self.base_sae_by_hook[hook_name]
        out_dir = base_output / sanitize_hook_name_for_path(hook_name)
        out_dir.mkdir(exist_ok=True, parents=True)

        dp_rank = self._dp_rank()
        tp_rank = self._tp_rank()
        if self._is_fsdp:
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                state_dict = sae.state_dict()
            if dp_rank != 0:
                return
            base_sae.process_state_dict_for_saving_inference(state_dict)
        else:
            if dp_rank != 0:
                return
            state_dict = base_sae.state_dict()
            base_sae.process_state_dict_for_saving_inference(state_dict)
            if tp_rank != 0:
                self._tp_barrier()
                return

        save_file(state_dict, out_dir / SAE_WEIGHTS_FILENAME)
        with open(out_dir / SAE_CFG_FILENAME, "w") as f:
            json.dump(base_sae.cfg.get_inference_sae_cfg_dict(), f)
        save_file(
            {"sparsity": self.log_feature_sparsity_by_hook[hook_name]},
            out_dir / SPARSITY_FILENAME,
        )
        if not self._is_fsdp:
            self._tp_barrier()

    def save_checkpoint(self, checkpoint_name: str) -> None:
        if self.cfg.checkpoint_path is None:
            return
        checkpoint_path = Path(self.cfg.checkpoint_path) / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True, parents=True)

        if self._is_metric_writer_rank():
            with open(checkpoint_path / MULTI_SAE_MANIFEST_FILENAME, "w") as f:
                json.dump(self._manifest(), f)

        for hook_name in self.hook_names:
            self._save_one_checkpoint_model(checkpoint_path, hook_name)

        self.save_trainer_state(checkpoint_path)

        if self.save_checkpoint_fn is not None and self._is_metric_writer_rank():
            self.save_checkpoint_fn(checkpoint_path=checkpoint_path)

    def _save_one_checkpoint_model(self, checkpoint_path: Path, hook_name: str) -> None:
        sae = self.sae_by_hook[hook_name]
        base_sae = self.base_sae_by_hook[hook_name]
        out_dir = checkpoint_path / sanitize_hook_name_for_path(hook_name)
        out_dir.mkdir(exist_ok=True, parents=True)

        dp_rank = self._dp_rank()
        tp_rank = self._tp_rank()
        if self._is_fsdp:
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                state_dict = sae.state_dict()
            if dp_rank != 0:
                return
            base_sae.process_state_dict_for_saving(state_dict)
        else:
            if dp_rank != 0:
                return
            state_dict = base_sae.state_dict()
            base_sae.process_state_dict_for_saving(state_dict)
            if tp_rank != 0:
                self._tp_barrier()
                return

        save_file(state_dict, out_dir / SAE_WEIGHTS_FILENAME)
        with open(out_dir / SAE_CFG_FILENAME, "w") as f:
            json.dump(base_sae.cfg.to_dict(), f)
        save_file(
            {"sparsity": self.log_feature_sparsity_by_hook[hook_name]},
            out_dir / SPARSITY_FILENAME,
        )
        del state_dict
        torch.cuda.empty_cache()
        if not self._is_fsdp:
            self._tp_barrier()

    def save_trainer_state(self, checkpoint_path: Path) -> None:
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        dp_rank = self._dp_rank()
        tp_rank = self._tp_rank()
        dp_size = self._dp_world_size()
        if self._is_fsdp:
            self._save_fsdp_raw_optimizer_state(checkpoint_path)
            optimizer_state: dict[str, Any] = {
                "optimizer_state_format": MULTI_SAE_FSDP_OPTIMIZER_STATE_FORMAT,
                "fsdp_dp_size": dp_size,
            }
            if dp_rank != 0:
                return
        else:
            if dp_rank != 0:
                return
            optimizer_state = {
                "optimizer_by_hook_by_name": self._build_named_optimizer_state_for_save()
            }
            if tp_rank != 0:
                return
        torch.save(
            {
                **optimizer_state,
                "format": "multi_independent_sae_v1",
                "hook_names": self.hook_names,
                "n_training_samples": self.n_training_samples,
                "n_training_steps": self.n_training_steps,
                "act_freq_scores_by_hook": self.act_freq_scores_by_hook,
                "n_forward_passes_since_fired_by_hook": self.n_forward_passes_since_fired_by_hook,
                "n_frac_active_samples_by_hook": self.n_frac_active_samples_by_hook,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "sae_dp_mode": self.sae_dp_mode,
                "backward_mode": self.backward_mode,
                "backward_order": self.backward_order,
                "stats_sync_mode": self.stats_sync_mode,
                "stats_sync_interval": self.stats_sync_interval,
                "seed_mode": self.seed_mode,
            },
            checkpoint_path / TRAINER_STATE_FILENAME,
        )

    def load_trainer_state(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)
        self._load_checkpoint_models(checkpoint_path)
        state = torch.load(checkpoint_path / TRAINER_STATE_FILENAME, map_location="cpu")
        if state["hook_names"] != self.hook_names:
            raise ValueError(
                "Cannot resume multi-SAE checkpoint with different hook_names"
            )
        self.n_training_samples = int(state["n_training_samples"])
        self.n_training_steps = int(state["n_training_steps"])
        saved_mode = state.get("sae_dp_mode", "ddp")
        if saved_mode != self.sae_dp_mode:
            raise ValueError(
                f"Cannot resume multi-SAE checkpoint saved with sae_dp_mode='{saved_mode}' "
                f"using current sae_dp_mode='{self.sae_dp_mode}'."
            )
        if self._is_fsdp:
            if (
                state.get("optimizer_state_format")
                != MULTI_SAE_FSDP_OPTIMIZER_STATE_FORMAT
            ):
                raise ValueError(
                    "Cannot resume multi-SAE FSDP checkpoint: missing "
                    f"optimizer_state_format='{MULTI_SAE_FSDP_OPTIMIZER_STATE_FORMAT}'."
                )
            expected_dp_size = state.get("fsdp_dp_size")
            if expected_dp_size != self._dp_world_size():
                raise ValueError(
                    "Cannot resume multi-SAE FSDP checkpoint with a different "
                    f"sae_dp_size: checkpoint has {expected_dp_size}, current run "
                    f"has {self._dp_world_size()}."
                )
            self._load_fsdp_raw_optimizer_state(checkpoint_path)
        elif "optimizer_by_hook_by_name" in state:
            self._load_named_optimizer_state(state["optimizer_by_hook_by_name"])
        else:
            self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        for hook_name in self.hook_names:
            self.act_freq_scores_by_hook[hook_name] = state["act_freq_scores_by_hook"][
                hook_name
            ].to(self.cfg.device)
            self.n_forward_passes_since_fired_by_hook[hook_name] = state[
                "n_forward_passes_since_fired_by_hook"
            ][hook_name].to(self.cfg.device)
            self.n_frac_active_samples_by_hook[hook_name] = state[
                "n_frac_active_samples_by_hook"
            ][hook_name]

    def _checkpoint_if_needed(self) -> None:
        if (
            self.checkpoint_thresholds
            and self.n_training_samples > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(checkpoint_name=str(self.n_training_samples))
            self.checkpoint_thresholds.pop(0)

    def _fsdp_raw_optimizer_state_path(self, checkpoint_path: Path) -> Path:
        return (
            checkpoint_path
            / MULTI_SAE_FSDP_OPTIMIZER_STATE_FILENAME_TEMPLATE.format(
                rank=self._dp_rank()
            )
        )

    def _save_fsdp_raw_optimizer_state(self, checkpoint_path: Path) -> None:
        torch.save(
            self.optimizer.state_dict(),
            self._fsdp_raw_optimizer_state_path(checkpoint_path),
        )

    def _load_fsdp_raw_optimizer_state(self, checkpoint_path: Path) -> None:
        self.optimizer.load_state_dict(
            torch.load(
                self._fsdp_raw_optimizer_state_path(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
        )

    def _build_named_optimizer_state_for_save(self) -> dict[str, dict[str, dict[str, Any]]]:
        optimizer_state_by_hook: dict[str, dict[str, dict[str, Any]]] = {}
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            hook_state: dict[str, dict[str, Any]] = {}
            for name, param in base_sae.named_parameters():
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                hook_state[name] = {
                    key: value.detach().clone()
                    if torch.is_tensor(value)
                    else deepcopy(value)
                    for key, value in state.items()
                }
            base_sae.process_named_optimizer_state_for_saving(hook_state)
            optimizer_state_by_hook[hook_name] = hook_state
        return optimizer_state_by_hook

    def _load_named_optimizer_state(
        self, optimizer_state_by_hook: dict[str, dict[str, dict[str, Any]]]
    ) -> None:
        self.optimizer.state.clear()
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            hook_state = deepcopy(optimizer_state_by_hook.get(hook_name, {}))
            base_sae.process_named_optimizer_state_for_loading(hook_state)
            named_params = dict(base_sae.named_parameters())
            for name, state in hook_state.items():
                if name not in named_params:
                    continue
                param = named_params[name]
                loaded_state: dict[str, Any] = {}
                for key, value in state.items():
                    if torch.is_tensor(value):
                        target_dtype = (
                            param.dtype
                            if value.is_floating_point() and value.ndim > 0
                            else value.dtype
                        )
                        loaded_state[key] = value.to(
                            device=param.device,
                            dtype=target_dtype,
                        )
                    else:
                        loaded_state[key] = deepcopy(value)
                self.optimizer.state[param] = loaded_state

    def _load_checkpoint_models(self, checkpoint_path: Path) -> None:
        for hook_name in self.hook_names:
            self._load_one_checkpoint_model(checkpoint_path, hook_name)

    def _load_one_checkpoint_model(self, checkpoint_path: Path, hook_name: str) -> None:
        sae = self.sae_by_hook[hook_name]
        base_sae = self.base_sae_by_hook[hook_name]
        hook_dir = checkpoint_path / sanitize_hook_name_for_path(hook_name)
        filepath = hook_dir / SAE_WEIGHTS_FILENAME

        tp_group = getattr(base_sae, "_tp_group", None)
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            state_dict = _load_tp_sharded_state_dict(filepath, base_sae, tp_group)
        else:
            state_dict = load_file(filepath)
            base_sae.process_state_dict_for_loading(state_dict)

        if self._is_fsdp:
            with FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT):
                sae.load_state_dict(state_dict)
        elif isinstance(sae, DDP):
            sae.module.load_state_dict(state_dict)
        else:
            base_sae.load_state_dict(state_dict)
        del state_dict

    def _maybe_synchronize_timing(self) -> None:
        if not self.cfg.synchronize_timing:
            return
        first_sae = self.base_sae_by_hook[self.hook_names[0]]
        if first_sae.device.type.startswith("cuda"):
            torch.cuda.synchronize(first_sae.device)

    def _consume_data_provider_timing(self) -> dict[str, float]:
        consume_timing = getattr(self.data_provider, "consume_last_data_timing", None)
        if consume_timing is None:
            return {"vllm_step_time_s": 0.0, "transfer_time_s": 0.0}
        timing = consume_timing()
        return {
            "vllm_step_time_s": float(timing.get("vllm_step_time_s", 0.0)),
            "transfer_time_s": float(timing.get("transfer_time_s", 0.0)),
        }

    def _global_timing_if_needed(
        self,
        *,
        vllm_step_time_s: float,
        transfer_time_s: float,
        sae_time_s: float,
        wall_time_s: float = 0.0,
    ) -> dict[str, float]:
        data_time_s = vllm_step_time_s + transfer_time_s
        step_time_s = data_time_s + sae_time_s
        # Single-card-style timing: keep local writer-rank timing without DP
        # cross-rank aggregation. In multi-layer mode, sae_time_s already
        # measures the full local SAE stage (all hooks/layers for this step).
        return {
            "vllm_step_time_s": vllm_step_time_s,
            "transfer_time_s": transfer_time_s,
            "data_time_s": data_time_s,
            "vllm_time_s": data_time_s,
            "dp_allreduce_time_s": 0.0,
            "sae_time_s": sae_time_s,
            "step_time_s": step_time_s,
            "wall_time_s": wall_time_s,
        }

    def _should_record_mse_step(self) -> bool:
        return (
            self.cfg.output_path is not None
            and self.cfg.save_mse_every_n_steps > 0
            and (self.n_training_steps + 1) % self.cfg.save_mse_every_n_steps == 0
        )

    def _should_record_timing_step(self) -> bool:
        return (
            self.cfg.output_path is not None
            and self.cfg.save_timing_every_n_steps > 0
            and (self.n_training_steps + 1) % self.cfg.save_timing_every_n_steps == 0
        )

    def _global_weighted_metric(
        self,
        value: torch.Tensor | float,
        local_n: int,
    ) -> float:
        value_t = torch.as_tensor(value, device=self.cfg.device, dtype=torch.float32)
        local_n_t = torch.tensor(float(local_n), device=self.cfg.device)
        metric_t = torch.stack([value_t.detach() * local_n_t, local_n_t])
        if self._dp_world_size() > 1:
            dist.all_reduce(metric_t, op=dist.ReduceOp.SUM, group=self.dp_group)
        if metric_t[1].item() == 0:
            return 0.0
        return float((metric_t[0] / metric_t[1]).detach().cpu().item())

    @torch.no_grad()
    def _record_mse_if_needed(
        self,
        outputs: dict[str, TrainStepOutput],
        local_n: int,
    ) -> None:
        if not self._should_record_mse_step():
            return
        record: dict[str, Any] = {
            "step": self.n_training_steps + 1,
            "n_training_samples": self.n_training_samples,
            "hooks": {},
        }
        for hook_name in self.hook_names:
            output = outputs.get(hook_name)
            if output is None:
                continue
            mse_loss = output.losses.get("mse_loss")
            if mse_loss is None:
                mse_loss = (output.sae_out - output.sae_in).pow(2).mean()
            hook_record = {
                # Match single-SAE logging semantics: record writer-rank local
                # metrics without DP aggregation so single vs multi traces are
                # directly comparable step-by-step.
                "mse_loss": _unwrap_item(mse_loss),
                "overall_loss": _unwrap_item(output.loss),
            }
            for loss_name, loss_value in output.losses.items():
                if loss_name == "mse_loss":
                    continue
                hook_record[loss_name] = _unwrap_item(loss_value)
            record["hooks"][hook_name] = hook_record
        if self.mse_history_path is None:
            return
        with open(self.mse_history_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    @torch.no_grad()
    def _record_timing_if_needed(
        self,
        *,
        vllm_step_time_s: float,
        transfer_time_s: float,
        data_time_s: float,
        vllm_time_s: float,
        sae_time_s: float,
        step_time_s: float,
        wall_time_s: float = 0.0,
        dp_allreduce_time_s: float = 0.0,
        sae_forward_time_s: float = 0.0,
        sae_stats_sync_time_s: float = 0.0,
        sae_backward_time_s: float = 0.0,
        sae_post_backward_time_s: float = 0.0,
        sae_optimizer_time_s: float = 0.0,
    ) -> None:
        if self.timing_history_path is None or not self._should_record_timing_step():
            return
        record = {
            "step": self.n_training_steps + 1,
            "n_training_samples": self.n_training_samples,
            "elapsed_s": time.time() - self._t_ready,
            "wall_time_s": wall_time_s,
            "vllm_step_time_s": vllm_step_time_s,
            "transfer_time_s": transfer_time_s,
            "data_time_s": data_time_s,
            "vllm_time_s": vllm_time_s,
            "dp_allreduce_time_s": dp_allreduce_time_s,
            "sae_time_s": sae_time_s,
            "step_time_s": step_time_s,
            "sae_forward_time_s": sae_forward_time_s,
            "sae_stats_sync_time_s": sae_stats_sync_time_s,
            "sae_backward_time_s": sae_backward_time_s,
            "sae_post_backward_time_s": sae_post_backward_time_s,
            "sae_optimizer_time_s": sae_optimizer_time_s,
        }
        with open(self.timing_history_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def _record_memory_if_needed(self, memory_stats: dict[str, float]) -> None:
        if self.memory_history_path is None:
            return
        save_every = getattr(self.cfg, "save_memory_every_n_steps", 0)
        if (self.n_training_steps + 1) % save_every != 0:
            return
        record: dict[str, object] = {
            "step": self.n_training_steps + 1,
            "n_training_samples": self.n_training_samples,
            "rank": self._memory_rank,
            **memory_stats,
        }
        with open(self.memory_history_path, "a") as f:
            json.dump(record, f)
            f.write("\n")


def _load_tp_sharded_state_dict(
    filepath: Path,
    base_sae: Any,
    tp_group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    """Load a checkpoint such that only tp_rank=0 reads the full file.

    tp_rank=0 loads the full state_dict, then scatters each parameter's shard
    to the corresponding rank. Non-sharded parameters (shard_dim=None) are
    broadcast from tp_rank=0. Returns a state_dict containing only this rank's
    shard — callers must NOT call process_state_dict_for_loading afterwards.
    """
    from safetensors import safe_open
    from sae_lens.util import str_to_dtype

    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    shard_dims: dict[str, int | None] = base_sae._tp_param_shard_dims()

    # tp_rank=0 loads full tensors; others only need metadata for pre-allocation
    full_state: dict[str, torch.Tensor] | None = None
    if tp_rank == 0:
        full_state = load_file(str(filepath))

    _safetensors_dtype_map = {
        "F32": "float32", "BF16": "bfloat16", "F16": "float16",
        "F64": "float64", "I32": "int32", "I64": "int64",
    }

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(filepath), framework="pt", device="cpu") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            shape = list(sl.get_shape())
            dtype_str = str(sl.get_dtype())
            dtype = str_to_dtype(_safetensors_dtype_map.get(dtype_str, dtype_str.lower()))
            shard_dim = shard_dims.get(k)

            if shard_dim is None:
                # Non-sharded parameter (e.g. b_dec): broadcast from rank 0
                if tp_rank == 0:
                    tensor = full_state[k]  # type: ignore[index]
                else:
                    tensor = torch.empty(shape, dtype=dtype)
                dist.broadcast(tensor, src=dist.get_global_rank(tp_group, 0), group=tp_group)
                state_dict[k] = tensor
            else:
                # Sharded parameter: scatter shard_dim slices to each rank
                full_size = shape[shard_dim]
                assert full_size % tp_size == 0, (
                    f"Checkpoint tensor '{k}' size {full_size} on dim {shard_dim} "
                    f"not divisible by tp_size={tp_size}"
                )
                shard_size = full_size // tp_size
                shard_shape = shape[:]
                shard_shape[shard_dim] = shard_size
                recv_buf = torch.empty(shard_shape, dtype=dtype)
                if tp_rank == 0:
                    chunks = full_state[k].split(shard_size, dim=shard_dim)  # type: ignore[index]
                    scatter_list = [c.contiguous() for c in chunks]
                else:
                    scatter_list = None
                dist.scatter(
                    recv_buf,
                    scatter_list,
                    src=dist.get_global_rank(tp_group, 0),
                    group=tp_group,
                )
                state_dict[k] = recv_buf

    if tp_rank == 0:
        del full_state

    return state_dict
