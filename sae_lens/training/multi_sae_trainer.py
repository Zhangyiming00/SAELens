from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

if os.environ.get("SAE_DUMP_ALLOC_SNAPSHOT") == "1":
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=200000,
        )
    except Exception:
        pass
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
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.ddp_overlap_v2 import (
    DDPOptimizerOverlapState,
    TPPostSharedMemory,
    apply_clip_coef_,
    tp_post_cpu_shm_prepare_clip,
)
from sae_lens.training.multi_hook_sae import MultiHookSAE
from sae_lens.training.optim import get_lr_scheduler
from sae_lens.training.sae_trainer import (
    SaveCheckpointFn,
    _log_feature_sparsity,
    _unwrap_item,
    _write_checkpoint_complete_marker,
)
from sae_lens.training.step_window_profiler import StepWindowProfiler
from sae_lens.training.tp_checkpoint import (
    gather_tp_state_dict_to_root_cpu,
    get_current_sae_tp_cpu_group,
)
from sae_lens.training.types import DataProvider

MULTI_SAE_MANIFEST_FILENAME = "multi_sae_manifest.json"
MULTI_SAE_FSDP_OPTIMIZER_STATE_FILENAME_TEMPLATE = (
    "multi_fsdp_optimizer_state_rank{rank}.pt"
)
MULTI_SAE_FSDP_OPTIMIZER_STATE_PP_FILENAME_TEMPLATE = (
    "multi_fsdp_optimizer_state_pp{pp_rank}_rank{rank}.pt"
)
MULTI_SAE_FSDP_OPTIMIZER_STATE_FORMAT = "multi_fsdp_raw_rank_sharded_v1"
MULTI_SAE_OPTIMIZER_STATE_FORMAT = "multi_per_hook_safetensors_v1"
PP_TRAINER_STATE_FILENAME_TEMPLATE = "trainer_state_pp{pp_rank}.pt"
# v2 layout: per-hook state file inside each hook's directory (PP-symmetric).
HOOK_STATE_FILENAME = "hook_state.pt"
HOOK_OPTIMIZER_STATE_FILENAME = "optimizer_state.safetensors"
HOOK_OPTIMIZER_META_FILENAME = "optimizer_state_meta.json"
_OPTIMIZER_STATE_KEY_SEP = "::"


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
        append_logs: bool = False,
        multi_hook_sae: Any | None = None,
    ) -> None:
        self.hook_names = hook_names
        self.multi_hook_sae = multi_hook_sae
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
        self.multi_sae_distributed_architecture: Literal[
            "legacy_per_hook_wrapper", "unified_multi_hook"
        ] = getattr(
            cfg,
            "multi_sae_distributed_architecture",
            "legacy_per_hook_wrapper",
        )
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
        if self.multi_sae_distributed_architecture not in (
            "legacy_per_hook_wrapper",
            "unified_multi_hook",
        ):
            raise ValueError(
                "multi_sae_distributed_architecture must be "
                "'legacy_per_hook_wrapper' or 'unified_multi_hook'"
            )
        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            if self.multi_hook_sae is None:
                raise ValueError("unified_multi_hook requires multi_hook_sae")
            if self.backward_mode != "combined":
                raise ValueError(
                    "unified_multi_hook only supports combined multi-SAE backward"
                )
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

        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            assert self.multi_hook_sae is not None
            params = list(self.multi_hook_sae.parameters())
        else:
            params: list[torch.nn.Parameter] = []
            for hook_name in self.hook_names:
                params.extend(list(self.sae_by_hook[hook_name].parameters()))
        _adam_kwargs: dict[str, Any] = {
            "lr": cfg.lr,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
        }
        # Adam implementation selector. SAE_ADAM_IMPL takes precedence:
        #   "fused"   -> fused=True       (single fused CUDA kernel)
        #   "foreach" -> foreach=True     (multi-tensor apply; CUDA default)
        #   "forloop" -> foreach=False    (single-tensor Python for-loop)
        # SAE_FUSED_ADAM=1 is kept for backward compatibility (= "fused").
        _adam_impl = os.environ.get("SAE_ADAM_IMPL", "").lower()
        if not _adam_impl and os.environ.get("SAE_FUSED_ADAM") == "1":
            _adam_impl = "fused"
        if _adam_impl == "fused":
            _adam_kwargs["fused"] = True
        elif _adam_impl == "foreach":
            _adam_kwargs["foreach"] = True
        elif _adam_impl == "forloop":
            _adam_kwargs["foreach"] = False
        self.optimizer = Adam(params, **_adam_kwargs)

        # Experimental per-hook DP-reduction -> optimizer overlap.  The raw
        # MultiHookSAE remains the state-dict owner, while each child executes
        # through an independent bucket-view DDP reducer.
        requested_overlap_mode = getattr(cfg, "multi_sae_optimizer_overlap", "off")
        if os.environ.get("SAE_DDP_OPT_OVERLAP_V2", "0") == "1":
            requested_overlap_mode = "on"
        if requested_overlap_mode not in ("off", "on", "non_tp_only"):
            raise ValueError(
                "multi_sae_optimizer_overlap must be 'off', 'on', or 'non_tp_only'"
            )
        tp_mode = self._tp_world_size() > 1
        overlap_requested_here = requested_overlap_mode == "on" or (
            requested_overlap_mode == "non_tp_only" and not tp_mode
        )
        # With one local hook or one DP rank there is no cross-hook DDP/optimizer
        # overlap to exploit.  This is important for uneven PP assignment (e.g.
        # H=3, PP=2): the H=2 stage takes the new path, while the H=1 stage safely
        # falls back without changing its optimizer behavior.
        if (
            overlap_requested_here
            and len(self.hook_names) > 1
            and self._dp_world_size() > 1
            and not self._is_ddp
        ):
            raise ValueError("multi_sae_optimizer_overlap requires sae_dp_mode='ddp'")
        self._ddp_opt_overlap_v2 = (
            overlap_requested_here
            and len(self.hook_names) > 1
            and self._is_ddp
            and self._dp_world_size() > 1
        )
        self._overlap_optimizer_by_hook: dict[str, Adam] = {}
        self._overlap_optimizer_done_by_hook: dict[str, torch.cuda.Event] = {}
        self._overlap_post_ready_by_hook: dict[str, torch.cuda.Event] = {}
        self._overlap_ddp_state: DDPOptimizerOverlapState | None = None
        self._overlap_tp_post: TPPostSharedMemory | None = None
        self._overlap_optimizer_stream: torch.cuda.Stream | None = None
        self._overlap_backward_done: torch.cuda.Event | None = None
        self._tp_phase_fence_mode = os.environ.get(
            "SAE_TP_PHASE_FENCE",
            getattr(cfg, "multi_sae_tp_phase_fence", "auto"),
        ).lower()
        if self._tp_phase_fence_mode not in ("auto", "always", "off"):
            raise ValueError(
                "multi_sae_tp_phase_fence/SAE_TP_PHASE_FENCE must be auto|always|off"
            )
        self._tp_phase_event: torch.cuda.Event | None = None
        if torch.cuda.is_available():
            self._tp_phase_event = torch.cuda.Event(blocking=False, interprocess=False)
        if self._ddp_opt_overlap_v2:
            if not self._is_ddp or self._is_fsdp:
                raise ValueError(
                    "multi_sae_optimizer_overlap requires sae_dp_mode='ddp'"
                )
            if self.multi_sae_distributed_architecture != "unified_multi_hook":
                raise ValueError(
                    "multi_sae_optimizer_overlap requires unified_multi_hook so all "
                    "hooks share one combined DDP backward"
                )
            if self.backward_mode != "combined":
                raise ValueError(
                    "multi_sae_optimizer_overlap requires combined backward"
                )
            if cfg.autocast:
                raise ValueError(
                    "multi_sae_optimizer_overlap currently requires autocast=False"
                )
            if not torch.cuda.is_available():
                raise ValueError(
                    "multi_sae_optimizer_overlap currently requires CUDA"
                )

            if not isinstance(self.multi_hook_sae, MultiHookSAE):
                raise ValueError(
                    "multi_sae_optimizer_overlap expected a MultiHookSAE root"
                )
            if self.dp_group is None:
                raise ValueError("DDP optimizer overlap requires a DP process group")
            first_device = self.base_sae_by_hook[self.hook_names[0]].device
            self._overlap_optimizer_stream = torch.cuda.Stream(device=first_device)
            self._overlap_backward_done = torch.cuda.Event(
                blocking=False, interprocess=False
            )

            for hook_name in self.hook_names:
                base_sae = self.base_sae_by_hook[hook_name]
                self._overlap_optimizer_by_hook[hook_name] = Adam(
                    list(base_sae.parameters()), **_adam_kwargs
                )
                self._overlap_optimizer_done_by_hook[hook_name] = torch.cuda.Event(
                    blocking=False, interprocess=False
                )
                self._overlap_post_ready_by_hook[hook_name] = torch.cuda.Event(
                    blocking=False, interprocess=False
                )

            ddp_by_hook = self.multi_hook_sae.ddp_forward_modules()
            if set(ddp_by_hook) != set(self.hook_names):
                raise ValueError(
                    "multi_sae_optimizer_overlap requires one DDP reducer per hook"
                )
            self._overlap_ddp_state = DDPOptimizerOverlapState(
                list(self.hook_names), ddp_by_hook, self.dp_group
            )

            tp_group = self._tp_group()
            if tp_group is not None and dist.get_world_size(tp_group) > 1:
                if os.environ.get("SAE_TP_POST_TRANSPORT", "cpu_shm") != "cpu_shm":
                    raise ValueError(
                        "TP optimizer-overlap currently supports "
                        "SAE_TP_POST_TRANSPORT=cpu_shm only"
                    )
                replicated_sizes: list[int] = []
                for hook_name in self.hook_names:
                    hook_sae = self.base_sae_by_hook[hook_name]
                    if not isinstance(hook_sae, TopKTrainingSAE):
                        raise TypeError(
                            "TP optimizer-overlap CPU-post currently supports "
                            "TopKTrainingSAE only"
                        )
                    shard_dims = hook_sae._tp_param_shard_dims()
                    replicated_sizes.extend(
                        param.numel()
                        for name, param in hook_sae.named_parameters()
                        if shard_dims.get(name) is None
                    )
                self._overlap_tp_post = TPPostSharedMemory(
                    tp_group=tp_group,
                    max_numel=max(replicated_sizes, default=1),
                    output_path=getattr(cfg, "output_path", None),
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
        self.rank_local_mse_history_path: Path | None = None
        self.debug_mse_history_path: Path | None = None
        self.timing_history_path: Path | None = None
        self.memory_history_path: Path | None = None
        self.memory_phase_history_path: Path | None = None
        self._memory_phase_records: list[dict[str, object]] = []
        self._memory_current_raw_batch_by_hook: dict[str, torch.Tensor] | None = None
        self._memory_current_scaled_batch_by_hook: dict[str, torch.Tensor] | None = None
        self._memory_current_outputs: dict[str, TrainStepOutput] | None = None
        self._memory_retained_outputs: dict[str, TrainStepOutput] | None = None
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
            if not (append_logs or getattr(cfg, "append_history_logs", False)):
                self.mse_history_path.write_text("")
        if (
            not should_write_logs
            and cfg.output_path is not None
            and cfg.save_mse_every_n_steps > 0
            and self._dp_rank() == 0
            and self._tp_rank() == 0
            and self._pp_rank() > 0
        ):
            output_path = Path(cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.rank_local_mse_history_path = (
                output_path / f"mse_history_pp{self._pp_rank()}_rank{rank}.jsonl"
            )
            if not (append_logs or getattr(cfg, "append_history_logs", False)):
                self.rank_local_mse_history_path.write_text("")
        if (
            os.environ.get("SAELENS_DEBUG_ALL_RANK_MSE") == "1"
            and cfg.output_path is not None
            and cfg.save_mse_every_n_steps > 0
        ):
            output_path = Path(cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.debug_mse_history_path = (
                output_path / f"debug_mse_rank{rank}_pid{os.getpid()}.jsonl"
            )
            self.debug_mse_history_path.write_text("")
        if (
            should_write_logs
            and cfg.output_path is not None
            and cfg.save_timing_every_n_steps > 0
        ):
            output_path = Path(cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            self.timing_history_path = output_path / TIMING_HISTORY_FILENAME
            if not (append_logs or getattr(cfg, "append_history_logs", False)):
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
        self._memory_phase_step_active = False
        if self._profile_memory:
            output_path = Path(cfg.output_path)  # type: ignore[arg-type]
            output_path.mkdir(exist_ok=True, parents=True)
            self.memory_history_path = (
                output_path / f"memory_history_rank{_global_rank}.jsonl"
            )
            self.memory_phase_history_path = (
                output_path / f"memory_phase_history_rank{_global_rank}.jsonl"
            )
            self.memory_history_path.write_text("")
            self.memory_phase_history_path.write_text("")
            self.device_history_path = (
                output_path / f"device_history_rank{_global_rank}.jsonl"
            )
            self.device_history_path.write_text("")
            if os.environ.get("SAE_DUMP_ALLOC_SNAPSHOT") == "1":
                # Already enabled at module load time; no-op here.
                pass
        # Background ~1 Hz device-memory sampler (see _device_sampler_loop).
        self.device_history_path: Path | None = getattr(
            self, "device_history_path", None
        )
        self._device_sampler_thread: threading.Thread | None = None
        self._device_sampler_stop = threading.Event()
        self._device_sampler_interval_s: float = float(
            os.environ.get("SAE_DEVICE_SAMPLE_INTERVAL_S", "1.0")
        )

        # Full alloc/free timeline + peak-moment snapshot for a single step. The
        # recorder logs every allocation/free with its Python stack; the dumped
        # pickle opens at https://pytorch.org/memory_viz. Scoped to one step
        # because the recorder grows unboundedly and adds noticeable overhead.
        self.memory_timeline_path: Path | None = None
        self._memory_timeline_step: int = getattr(
            cfg, "record_memory_timeline_step", -1
        )
        self._memory_timeline_active: bool = False
        if self._memory_timeline_step >= 0 and cfg.output_path is not None:
            output_path = Path(cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            self.memory_timeline_path = (
                output_path / f"memory_timeline_rank{_global_rank}.pickle"
            )

        # Fixed-window end-to-end step timing. Every rank writes its own file:
        # with TP/PP the ranks do different work, so a single writer would hide
        # skew. See StepWindowProfiler for the sync placement.
        self.step_window_profiler = StepWindowProfiler.maybe_create(
            start_step=getattr(cfg, "step_window_profile_start_step", 0),
            window_steps=getattr(cfg, "step_window_profile_window_steps", 0),
            window_count=getattr(cfg, "step_window_profile_window_count", 0),
            output_dir=cfg.output_path,
            role="sae",
            step_unit="sae_step",
            rank=_global_rank,
            device=cfg.device,
            context={
                "num_hooks": len(self.hook_names),
                "train_batch_size_samples": cfg.train_batch_size_samples,
                "dp_world_size": self._dp_world_size(),
                "synchronize_timing": cfg.synchronize_timing,
                "save_memory_every_n_steps": getattr(
                    cfg, "save_memory_every_n_steps", 0
                ),
            },
        )

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
        pp_rank = self._pp_rank()
        return self._dp_rank() == 0 and self._tp_rank() == 0 and pp_rank == 0

    def _pp_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            try:
                import sae_lens.distributed_v2 as v2_mod
                if getattr(v2_mod, "_initialized", False) and v2_mod.is_consumer():
                    return v2_mod.get_sae_pp_rank()
            except ImportError:
                pass
        return 0

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
        quiesce_drain_ack_path: Path | str | None = None,
        quiesce_finished_ack_path: Path | str | None = None,
        stop_after_step_check: Callable[[], bool] | None = None,
    ) -> dict[str, TrainingSAE[Any]]:
        pbar = tqdm(total=self.cfg.total_training_samples, desc="Training Multi SAE")
        self._start_device_sampler()
        quiesce_draining = False
        quiesce_checkpoint_now = False
        quiesce_checkpoint_saved = False
        quiesce_drain_acked = False
        stopped_for_reconfigure = False

        def _touch_if_metric_writer(path: Path | str | None) -> None:
            if path is not None and self._is_metric_writer_rank():
                ack_path = Path(path)
                ack_path.parent.mkdir(parents=True, exist_ok=True)
                ack_path.touch()

        def _maybe_start_quiesce_drain() -> None:
            nonlocal quiesce_draining, quiesce_checkpoint_now
            if (
                quiesce_request_path is None
                or not Path(quiesce_request_path).exists()
                or quiesce_draining
            ):
                return
            drain_local_pool = getattr(self.data_provider, "request_drain_local_pool", None)
            if callable(drain_local_pool):
                drain_local_pool()
            else:
                quiesce_checkpoint_now = True
            quiesce_draining = True

        def _ack_drain_done() -> None:
            nonlocal quiesce_drain_acked
            if quiesce_drain_acked:
                return
            _touch_if_metric_writer(quiesce_drain_ack_path)
            quiesce_drain_acked = True

        def _save_quiesce_checkpoint() -> None:
            nonlocal quiesce_checkpoint_saved
            if quiesce_checkpoint_saved:
                return
            self.save_checkpoint(checkpoint_name=f"quiesce_{self.n_training_samples}")
            _touch_if_metric_writer(quiesce_finished_ack_path or quiesce_ack_path)
            quiesce_checkpoint_saved = True

        while (
            self.n_training_samples < self.cfg.total_training_samples
            or quiesce_draining
        ):
            if (
                not quiesce_draining
                and stop_after_step_check is not None
                and stop_after_step_check()
            ):
                stopped_for_reconfigure = True
                break
            _maybe_start_quiesce_drain()
            if quiesce_checkpoint_now:
                _ack_drain_done()
                _save_quiesce_checkpoint()
                break
            step_number = self.n_training_steps + 1
            if self.step_window_profiler is not None:
                self.step_window_profiler.on_step_start(step_number)
            step_wall_t0 = time.perf_counter()
            self._start_memory_phase_step()
            self._reset_memory_phase_peak()
            self._maybe_start_memory_timeline()
            self._maybe_synchronize_timing()
            with cuda_nvtx_range("multi_sae:data_fetch"):
                try:
                    batch_by_hook = next(self.data_provider)
                except StopIteration:
                    break
            self._memory_current_raw_batch_by_hook = batch_by_hook
            self._record_memory_phase("after_data_fetch")
            if not isinstance(batch_by_hook, dict):
                raise TypeError(
                    "MultiSAETrainer expected data_provider to yield dict batches"
                )
            self._validate_unified_hook_set(batch_by_hook)
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
            self._memory_current_scaled_batch_by_hook = scaled_batch_by_hook
            self._record_memory_phase("after_scale_to_device")
            previous_samples = self.n_training_samples
            if getattr(self.data_provider, "tracks_global_progress", False):
                self.n_training_samples = int(
                    self.data_provider.global_tokens_consumed
                )
            else:
                self.n_training_samples += local_n
            progress_samples = self.n_training_samples - previous_samples

            self._maybe_synchronize_timing()
            sae_t0 = time.perf_counter()
            self._memory_retained_outputs = self._memory_current_outputs
            self._memory_current_outputs = None
            if self._memory_phase_step_active:
                torch.cuda.reset_peak_memory_stats(self.cfg.device)
            with cuda_nvtx_range("multi_sae:train_step"):
                outputs, sae_phase_timing = self._train_step(scaled_batch_by_hook, local_n)
            self._memory_current_outputs = outputs
            self._maybe_synchronize_timing()
            sae_time_s = time.perf_counter() - sae_t0

            if self._memory_phase_step_active:
                memory_stats = self._aggregate_memory_phase_stats()
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
            self._maybe_stop_memory_timeline()
            self.n_training_steps += 1
            self.lr_scheduler.step()
            self._checkpoint_if_needed()
            pbar.update(progress_samples)
            if self.n_training_steps % 8 == 0 and outputs:
                avg_loss = sum(_unwrap_item(o.loss) for o in outputs.values()) / len(
                    outputs
                )
                pbar.set_description(
                    f"{self.n_training_steps}| avg_loss: {avg_loss:.5f}"
                )

            # Closed after per-step logging/checkpointing so the window covers
            # everything a real step costs, not just the compute.
            if self.step_window_profiler is not None:
                self.step_window_profiler.on_step_end(
                    step_number,
                    samples=local_n,
                    components={**timing, **sae_phase_timing},
                )

            # The next data fetch can run a large vLLM capture/routing refill.
            # Nothing after this point consumes tensor-valued step outputs, so
            # release them before that refill instead of retaining two steps of
            # TrainStepOutput graphs and activation batches concurrently.
            self._memory_retained_outputs = None
            self._memory_current_outputs = None
            self._memory_current_raw_batch_by_hook = None
            self._memory_current_scaled_batch_by_hook = None
            del outputs, scaled_batch_by_hook, batch_by_hook

            _maybe_start_quiesce_drain()
            if quiesce_checkpoint_now:
                _ack_drain_done()
                _save_quiesce_checkpoint()
                break

        if self.step_window_profiler is not None:
            self.step_window_profiler.close()
        pbar.close()
        self._stop_device_sampler()
        # Ensure periodic/deferred stats are flushed before final save/logging.
        self._sync_deferred_stats_if_needed(force=True)
        if quiesce_draining:
            _ack_drain_done()
            _save_quiesce_checkpoint()
        self.last_fit_stopped_for_reconfigure = stopped_for_reconfigure
        if (
            self.cfg.save_final_checkpoint
            and not quiesce_checkpoint_saved
            and not stopped_for_reconfigure
        ):
            self.save_checkpoint(checkpoint_name=f"final_{self.n_training_samples}")
        if self._overlap_tp_post is not None:
            self._overlap_tp_post.close()
            self._overlap_tp_post = None
        if self._overlap_ddp_state is not None:
            self._overlap_ddp_state.close()
        return self.base_sae_by_hook

    def _train_step(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        self._validate_unified_hook_set(batch_by_hook)
        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            assert self.multi_hook_sae is not None
            self.multi_hook_sae.train()
        else:
            for sae in self.sae_by_hook.values():
                sae.train()

        if self._ddp_opt_overlap_v2:
            # The scheduler is intentionally attached to the legacy aggregate
            # optimizer; mirror its current LR into the actual per-hook optimizers.
            lr = float(self.optimizer.param_groups[0]["lr"])
            for optimizer in self._overlap_optimizer_by_hook.values():
                for group in optimizer.param_groups:
                    group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        for optimizer in self._overlap_optimizer_by_hook.values():
            optimizer.zero_grad(set_to_none=True)
        self._record_memory_phase("after_zero_grad_start")
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
                if self.multi_sae_distributed_architecture == "unified_multi_hook":
                    self._train_step_unified_zero_backward(batch_by_hook, phase_timing)
                    return outputs, phase_timing
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
        if self._ddp_opt_overlap_v2:
            return self._train_step_ddp_optimizer_overlap_v2(
                batch_by_hook,
                local_n,
                loss_scale,
                phase_timing,
            )
        if self.backward_mode == "combined":
            return self._train_step_combined_backward(
                batch_by_hook, local_n, loss_scale, phase_timing
            )
        return self._train_step_sequential_backward(
            batch_by_hook, local_n, loss_scale, phase_timing
        )

    def _tp_world_size(self) -> int:
        group = self._tp_group()
        if group is None or not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size(group)

    def _tp_phase_fence_if_needed(self) -> None:
        """Fence a completed TP phase before traffic on a different NCCL group.

        ``auto`` is intentionally narrow: it is active only for the new local
        cross-hook TP forward when either (a) this SAE also has a distinct,
        multi-rank DDP communicator or (b) producer and SAE roles share this rank.
        ``always`` is a debug/safety override and ``off`` preserves the fully
        asynchronous behavior.
        The fence is placed *after* the whole cross-hook TP wavefront, never
        between hooks, so it does not destroy H_i/H_(i+1) forward overlap.
        """

        mode = self._tp_phase_fence_mode
        if mode == "off" or self._tp_world_size() <= 1 or self._tp_phase_event is None:
            return
        if mode == "auto":
            # Auto is for the *new cross-hook TP forward* only.  Legacy per-hook
            # TP remains untouched unless the user explicitly chooses always.
            cross_hook_tp_forward = len(self.hook_names) > 1 and (
                self.multi_sae_distributed_architecture == "unified_multi_hook"
            )
            producer_sae_overlap = bool(
                getattr(
                    self.cfg,
                    "multi_sae_tp_phase_fence_runtime_hazard",
                    False,
                )
            )
            distinct_dp_group = (
                self._is_ddp
                and self._dp_world_size() > 1
                and self.dp_group is not None
                and self.dp_group is not self._tp_group()
            )
            if not cross_hook_tp_forward or not (
                producer_sae_overlap or distinct_dp_group
            ):
                return
        current = torch.cuda.current_stream(torch.device(self.cfg.device))
        self._tp_phase_event.record(current)
        # Host-visible by design.  This is a phase fence, not a per-collective
        # synchronization; it prevents rank-dependent TP/DP enqueue interleaving.
        self._tp_phase_event.synchronize()

    def _train_step_ddp_optimizer_overlap_v2(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
        loss_scale: float,
        phase_timing: dict[str, float],
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        """Run one combined backward and overlap per-hook post/optimizer work.

        Per-hook DDP buckets launch as soon as each bucket becomes ready in the
        normal combined autograd traversal.  The comm hook does not add the real
        Work to the backward stream, so this returns once every hook's compute
        has been enqueued.  Only then may the optimizer stream wait for one
        hook's buckets and update it.  TP post-processing stays on CPU shared
        memory and cannot reorder the still-running DP/NCCL collectives.
        """

        state = self._overlap_ddp_state
        opt_stream = self._overlap_optimizer_stream
        backward_done = self._overlap_backward_done
        if state is None or opt_stream is None or backward_done is None:
            raise RuntimeError("DDP optimizer overlap V2 was not initialized")
        if self.multi_hook_sae is None:
            raise RuntimeError("DDP optimizer overlap requires a MultiHookSAE owner")

        state.begin_step()

        inputs: dict[str, TrainStepInput] = {}
        for hook_name in self.hook_names:
            acts = batch_by_hook[hook_name]
            if local_n == 0:
                acts = torch.zeros(
                    1,
                    self.base_sae_by_hook[hook_name].cfg.d_in,
                    device=self.cfg.device,
                    dtype=acts.dtype,
                )
            inputs[hook_name] = self._build_step_input(hook_name, acts)

        t_fwd = time.perf_counter()
        with cuda_nvtx_range("multi_sae:overlap_v4_forward"):
            with self.autocast_if_enabled:
                outputs = self.multi_hook_sae(inputs)
        phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
        self._memory_current_outputs = outputs
        self._record_memory_phase("after_forward_all")

        # One boundary after the entire TP wavefront.  auto is a no-op when
        # there is no multi-rank DDP communicator to interleave with TP.
        self._tp_phase_fence_if_needed()

        scaled_losses: list[torch.Tensor] = []
        for hook_name in self.hook_names:
            output = outputs[hook_name]
            if local_n != 0:
                t_stats = time.perf_counter()
                with cuda_nvtx_range(f"multi_sae:{hook_name}:stats_sync"):
                    self._update_stats(hook_name, output, local_n)
                phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
            scaled_losses.append(
                output.loss * (loss_scale if local_n != 0 else 0.0)
            )

        t_bwd = time.perf_counter()
        with nccl_nvtx_range(
            "nccl:multi_sae_dp_overlap_v4_combined_backward",
            self.dp_group,
        ):
            with cuda_nvtx_range("multi_sae:dp_overlap_v4_combined_backward"):
                sum(scaled_losses).backward()
        state.end_backward()
        backward_done.record(torch.cuda.current_stream(torch.device(self.cfg.device)))
        phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd
        # A device-wide synchronization here would wait for every outstanding
        # DDP reduction and erase the overlap this path is designed to create.
        # CUDA allocator counters are host-side, so an enqueue-boundary snapshot
        # remains useful without draining the device.
        self._record_memory_phase("after_combined_backward", synchronize=False)

        t_post = time.perf_counter()
        tp_mode = self._tp_world_size() > 1
        order = state.completion_order(list(reversed(self.hook_names)))
        previous_opt_done: torch.cuda.Event | None = None
        if tp_mode:
            for hook_name in order:
                base_sae = self.base_sae_by_hook[hook_name]
                if (
                    not isinstance(base_sae, TopKTrainingSAE)
                    or self._overlap_tp_post is None
                ):
                    raise RuntimeError(
                        "TP optimizer-overlap requires TopKTrainingSAE + cpu_shm reducer"
                    )

                # .cpu() below is the host-visible wait for this hook only.  The
                # remaining DP reductions keep progressing on their NCCL streams.
                current_stream = torch.cuda.current_stream(base_sae.device)
                current_stream.wait_event(backward_done)
                state.wait_for_hook(hook_name)
                with cuda_nvtx_range(f"multi_sae:{hook_name}:tp_post_cpu_shm"):
                    coef = tp_post_cpu_shm_prepare_clip(
                        base_sae,
                        self._overlap_tp_post,
                    )
                post_ready = self._overlap_post_ready_by_hook[hook_name]
                post_ready.record(current_stream)
                optimizer = self._overlap_optimizer_by_hook[hook_name]
                opt_done = self._overlap_optimizer_done_by_hook[hook_name]
                with torch.cuda.stream(opt_stream):
                    opt_stream.wait_event(post_ready)
                    with cuda_nvtx_range(f"multi_sae:{hook_name}:clip_scale_v4"):
                        apply_clip_coef_(base_sae, coef)
                    with cuda_nvtx_range(f"multi_sae:{hook_name}:optimizer_step_v4"):
                        optimizer.step()
                    opt_done.record(opt_stream)
                previous_opt_done = opt_done
        else:
            # A single stream preserves optimizer ordering.  Each wait is scoped
            # to this hook's parameters, so the first ready hook can update while
            # later hook reductions are still in flight.
            with torch.cuda.stream(opt_stream):
                opt_stream.wait_event(backward_done)
                for hook_name in order:
                    base_sae = self.base_sae_by_hook[hook_name]
                    optimizer = self._overlap_optimizer_by_hook[hook_name]
                    opt_done = self._overlap_optimizer_done_by_hook[hook_name]
                    state.wait_for_hook(hook_name)
                    with cuda_nvtx_range(f"multi_sae:{hook_name}:clip_grad_v4"):
                        base_sae.clip_grad_norm_(1.0)
                    with cuda_nvtx_range(f"multi_sae:{hook_name}:optimizer_step_v4"):
                        optimizer.step()
                    opt_done.record(opt_stream)
                    previous_opt_done = opt_done

        phase_timing["sae_post_backward_time_s"] += time.perf_counter() - t_post
        t_opt = time.perf_counter()
        # There is no useful cross-hook communication left after the last hook.
        # A host wait here also makes set_to_none zero_grad on the next step safe
        # for gradients consumed by the side optimizer stream.
        if previous_opt_done is not None:
            previous_opt_done.synchronize()
        state.finish_step()
        self._record_memory_phase("after_optimizer_step")
        t_stats = time.perf_counter()
        with cuda_nvtx_range("multi_sae:stats_sync_tail"):
            self._sync_deferred_stats_if_needed(force=False)
        phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
        phase_timing["sae_optimizer_time_s"] += time.perf_counter() - t_opt
        self._record_memory_phase("after_stats_tail")
        return outputs, phase_timing

    def _train_step_unified_zero_backward(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        phase_timing: dict[str, float],
    ) -> None:
        assert self.multi_hook_sae is not None
        overlap_state = self._overlap_ddp_state
        if overlap_state is not None:
            overlap_state.begin_step()
        step_inputs_by_hook: dict[str, TrainStepInput] = {}
        for hook_name in self.hook_names:
            acts = batch_by_hook[hook_name]
            dummy = torch.zeros(
                1,
                self.base_sae_by_hook[hook_name].cfg.d_in,
                device=self.cfg.device,
                dtype=acts.dtype,
            )
            step_inputs_by_hook[hook_name] = self._build_step_input(hook_name, dummy)
        t_fwd = time.perf_counter()
        with cuda_nvtx_range("multi_sae:unified_forward"):
            with self.autocast_if_enabled:
                outputs = self.multi_hook_sae(step_inputs_by_hook)
        phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
        self._tp_phase_fence_if_needed()
        total_loss = sum(output.loss * 0.0 for output in outputs.values())
        t_bwd = time.perf_counter()
        with cuda_nvtx_range("multi_sae:combined_backward"):
            self.grad_scaler.scale(total_loss).backward()
        phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd
        if overlap_state is not None:
            overlap_state.end_backward()
            for hook_name in overlap_state.completion_order(
                list(reversed(self.hook_names))
            ):
                overlap_state.wait_for_hook(hook_name)
            torch.cuda.current_stream(torch.device(self.cfg.device)).synchronize()
            overlap_state.finish_step()
        self.grad_scaler.unscale_(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        for optimizer in self._overlap_optimizer_by_hook.values():
            optimizer.zero_grad(set_to_none=True)

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
        self._memory_current_outputs = outputs
        self._record_memory_phase("after_forward_all")

        # Phase B: run backward in configured order.
        for hook_name in self._ordered_hook_names_for_backward():
            t_bwd = time.perf_counter()
            with nccl_nvtx_range(
                f"nccl:multi_sae_{self.sae_dp_mode}_backward", self.dp_group
            ):
                with cuda_nvtx_range(f"multi_sae:{hook_name}:backward"):
                    self.grad_scaler.scale(scaled_loss_by_hook[hook_name]).backward()
            phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd
            self._record_memory_phase(f"after_backward_{sanitize_hook_name_for_path(hook_name)}")

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
        self._record_memory_phase("after_post_backward")
        t_opt = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_step"):
            self.grad_scaler.step(self.optimizer)
        with cuda_nvtx_range("multi_sae:scaler_update"):
            self.grad_scaler.update()
        self._record_memory_phase("after_optimizer_step")
        t_stats = time.perf_counter()
        with cuda_nvtx_range("multi_sae:stats_sync_tail"):
            self._sync_deferred_stats_if_needed(force=False)
        phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
        phase_timing["sae_optimizer_time_s"] += time.perf_counter() - t_opt
        self._record_memory_phase("after_stats_tail")
        return outputs, phase_timing

    def _train_step_combined_backward(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
        loss_scale: float,
        phase_timing: dict[str, float],
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            return self._train_step_unified_combined_backward(
                batch_by_hook, local_n, loss_scale, phase_timing
            )

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
        self._memory_current_outputs = outputs
        self._record_memory_phase("after_forward_all")

        total_loss = sum(scaled_losses)
        t_bwd = time.perf_counter()
        with nccl_nvtx_range(
            f"nccl:multi_sae_{self.sae_dp_mode}_combined_backward", self.dp_group
        ):
            with cuda_nvtx_range("multi_sae:combined_backward"):
                self.grad_scaler.scale(total_loss).backward()
        phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd
        self._record_memory_phase("after_combined_backward")

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
        self._record_memory_phase("after_post_backward")
        t_opt = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_step"):
            self.grad_scaler.step(self.optimizer)
        with cuda_nvtx_range("multi_sae:scaler_update"):
            self.grad_scaler.update()
        self._record_memory_phase("after_optimizer_step")
        t_stats = time.perf_counter()
        with cuda_nvtx_range("multi_sae:stats_sync_tail"):
            self._sync_deferred_stats_if_needed(force=False)
        phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
        phase_timing["sae_optimizer_time_s"] += time.perf_counter() - t_opt
        self._record_memory_phase("after_stats_tail")
        return outputs, phase_timing

    def _train_step_unified_combined_backward(
        self,
        batch_by_hook: dict[str, torch.Tensor],
        local_n: int,
        loss_scale: float,
        phase_timing: dict[str, float],
    ) -> tuple[dict[str, TrainStepOutput], dict[str, float]]:
        assert self.multi_hook_sae is not None
        step_inputs_by_hook: dict[str, TrainStepInput] = {}
        for hook_name in self.hook_names:
            acts = batch_by_hook[hook_name]
            if local_n == 0:
                acts = torch.zeros(
                    1,
                    self.base_sae_by_hook[hook_name].cfg.d_in,
                    device=self.cfg.device,
                    dtype=acts.dtype,
                )
            step_inputs_by_hook[hook_name] = self._build_step_input(hook_name, acts)

        t_fwd = time.perf_counter()
        with cuda_nvtx_range("multi_sae:unified_forward"):
            context = (
                nccl_nvtx_range(
                    "nccl:multi_sae_fsdp_forward_param_all_gather", self.dp_group
                )
                if self._is_fsdp
                else contextlib.nullcontext()
            )
            with context:
                with self.autocast_if_enabled:
                    outputs = self.multi_hook_sae(step_inputs_by_hook)
        phase_timing["sae_forward_time_s"] += time.perf_counter() - t_fwd
        # Default TP x DDP path: MultiHookSAE already executed the cross-hook TP
        # wavefront above.  Fence only at the TP->DP phase boundary when policy
        # requests it; never fence between local hooks.
        self._tp_phase_fence_if_needed()

        scaled_losses: list[torch.Tensor] = []
        for hook_name in self.hook_names:
            output = outputs[hook_name]
            if local_n == 0:
                scaled_losses.append(output.loss * 0.0)
                continue
            t_stats = time.perf_counter()
            with cuda_nvtx_range(f"multi_sae:{hook_name}:stats_sync"):
                self._update_stats(hook_name, output, local_n)
            phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
            scaled_losses.append(output.loss * loss_scale)
        self._memory_current_outputs = outputs
        self._record_memory_phase("after_forward_all")

        total_loss = sum(scaled_losses)
        t_bwd = time.perf_counter()
        with nccl_nvtx_range(
            f"nccl:multi_sae_{self.sae_dp_mode}_combined_backward", self.dp_group
        ):
            with cuda_nvtx_range("multi_sae:combined_backward"):
                self.grad_scaler.scale(total_loss).backward()
        phase_timing["sae_backward_time_s"] += time.perf_counter() - t_bwd
        self._record_memory_phase("after_combined_backward")

        self._post_backward_and_optimizer_step(phase_timing)
        return outputs, phase_timing

    def _post_backward_and_optimizer_step(self, phase_timing: dict[str, float]) -> None:
        t_post = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_unscale"):
            self.grad_scaler.unscale_(self.optimizer)
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            with cuda_nvtx_range(f"multi_sae:{hook_name}:tp_sync"):
                base_sae.sync_tensor_parallel_gradients()
            with cuda_nvtx_range(f"multi_sae:{hook_name}:clip_grad"):
                self._clip_hook_grad_norm_(hook_name, max_norm=1.0)
        phase_timing["sae_post_backward_time_s"] += time.perf_counter() - t_post
        self._record_memory_phase("after_post_backward")
        t_opt = time.perf_counter()
        with cuda_nvtx_range("multi_sae:optimizer_step"):
            self.grad_scaler.step(self.optimizer)
        with cuda_nvtx_range("multi_sae:scaler_update"):
            self.grad_scaler.update()
        self._record_memory_phase("after_optimizer_step")
        t_stats = time.perf_counter()
        with cuda_nvtx_range("multi_sae:stats_sync_tail"):
            self._sync_deferred_stats_if_needed(force=False)
        phase_timing["sae_stats_sync_time_s"] += time.perf_counter() - t_stats
        phase_timing["sae_optimizer_time_s"] += time.perf_counter() - t_opt
        self._record_memory_phase("after_stats_tail")

    def _clip_hook_grad_norm_(self, hook_name: str, max_norm: float) -> torch.Tensor:
        base_sae = self.base_sae_by_hook[hook_name]
        return base_sae.clip_grad_norm_(
            max_norm,
            dp_group=self.dp_group if self._is_fsdp else None,
        )

    def _validate_unified_hook_set(
        self,
        batch_by_hook: dict[str, torch.Tensor],
    ) -> None:
        if self.multi_sae_distributed_architecture != "unified_multi_hook":
            return
        actual_hooks = set(batch_by_hook)
        expected_hooks = set(self.hook_names)
        if actual_hooks != expected_hooks:
            raise ValueError(
                "unified_multi_hook requires every step to contain exactly "
                f"{self.hook_names}; got {sorted(actual_hooks)}"
            )

    def _multi_hook_root_module(self) -> MultiHookSAE:
        if self.multi_hook_sae is None:
            raise ValueError("multi_hook_sae is not configured")
        module = self.multi_hook_sae
        if isinstance(module, DDP):
            module = module.module
        elif isinstance(module, FSDP):
            module = module.module
        if not isinstance(module, MultiHookSAE):
            raise TypeError(
                f"Expected MultiHookSAE owner, got {type(module).__name__}"
            )
        return module

    def _build_step_input(self, hook_name: str, acts: torch.Tensor) -> TrainStepInput:
        return TrainStepInput(
            sae_in=acts,
            dead_neuron_mask=(
                self.n_forward_passes_since_fired_by_hook[hook_name]
                > self.cfg.dead_feature_window
            ).bool(),
            coefficients={},
            n_training_steps=self.n_training_steps,
            is_logging_step=False,
        )

    def _forward_one(self, hook_name: str, acts: torch.Tensor) -> TrainStepOutput:
        step_input = self._build_step_input(hook_name, acts)
        with self.autocast_if_enabled:
            context = (
                nccl_nvtx_range(
                    "nccl:multi_sae_fsdp_forward_param_all_gather", self.dp_group
                )
                if self._is_fsdp
                else contextlib.nullcontext()
            )
            with context:
                output = self.sae_by_hook[hook_name](step_input)
        self._tp_phase_fence_if_needed()
        return output

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
        pp_rank = self._pp_rank()
        global_rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        if self._is_metric_writer_rank():
            with open(base_output / MULTI_SAE_MANIFEST_FILENAME, "w") as f:
                json.dump(manifest, f)
        if pp_rank > 0:
            local_manifest_path = (
                base_output / f"multi_sae_manifest_pp{pp_rank}_rank{global_rank}.json"
            )
            with open(local_manifest_path, "w") as f:
                json.dump(self._manifest(self.hook_names), f)

        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            self._save_unified_models(base_output, inference=True)
            return

        for hook_name in self.hook_names:
            self._save_one_final(base_output, hook_name)

    def _manifest(self, hook_names: list[str] | None = None) -> dict[str, Any]:
        hook_names = self.hook_names if hook_names is None else hook_names
        return {
            "format": "multi_independent_sae_v1",
            "hook_names": hook_names,
            "hook_to_dir": {
                hook_name: sanitize_hook_name_for_path(hook_name)
                for hook_name in hook_names
            },
            "shared_hyperparams": True,
            "sae_dp_mode": self.sae_dp_mode,
            "backward_mode": self.backward_mode,
            "backward_order": self.backward_order,
            "stats_sync_mode": self.stats_sync_mode,
            "stats_sync_interval": self.stats_sync_interval,
            "seed_mode": self.seed_mode,
            "multi_sae_distributed_architecture": self.multi_sae_distributed_architecture,
        }

    def _unified_state_dict_by_hook(
        self,
    ) -> dict[str, dict[str, Any]] | None:
        if self.multi_hook_sae is None:
            raise ValueError("unified_multi_hook requires multi_hook_sae")
        root = self._multi_hook_root_module()
        dp_rank = self._dp_rank()
        if self._is_fsdp:
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.multi_hook_sae,
                StateDictType.FULL_STATE_DICT,
                fsdp_cfg,
            ):
                root_state_dict = self.multi_hook_sae.state_dict()
            if dp_rank != 0:
                return None
        else:
            if dp_rank != 0:
                return None
            root_state_dict = self.multi_hook_sae.state_dict()
        return root.split_state_dict_by_hook(root_state_dict)

    def _save_unified_models(self, base_path: Path, *, inference: bool) -> None:
        state_dict_by_hook = self._unified_state_dict_by_hook()
        if state_dict_by_hook is None:
            return
        tp_rank = self._tp_rank()
        tp_cpu_group = get_current_sae_tp_cpu_group() if self._is_fsdp else None
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            state_dict = state_dict_by_hook[hook_name]
            if self._is_fsdp:
                state_dict = gather_tp_state_dict_to_root_cpu(
                    state_dict, base_sae, tp_cpu_group
                )
                if inference and tp_rank == 0:
                    assert state_dict is not None
                    base_sae.postprocess_full_state_dict_for_inference(state_dict)
            elif inference:
                base_sae.process_state_dict_for_saving_inference(state_dict)
            else:
                base_sae.process_state_dict_for_saving(state_dict)
            out_dir = base_path / sanitize_hook_name_for_path(hook_name)
            out_dir.mkdir(exist_ok=True, parents=True)
            if tp_rank == 0:
                save_file(state_dict, out_dir / SAE_WEIGHTS_FILENAME)
                cfg_dict = (
                    base_sae.cfg.get_inference_sae_cfg_dict()
                    if inference
                    else base_sae.cfg.to_dict()
                )
                with open(out_dir / SAE_CFG_FILENAME, "w") as f:
                    json.dump(cfg_dict, f)
                save_file(
                    {"sparsity": self.log_feature_sparsity_by_hook[hook_name]},
                    out_dir / SPARSITY_FILENAME,
                )
            if self._is_fsdp:
                if tp_cpu_group is not None and dist.get_world_size(tp_cpu_group) > 1:
                    dist.barrier(group=tp_cpu_group)
            else:
                self._tp_barrier()

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
                local_tp_state = sae.state_dict()
            if dp_rank != 0:
                return
            tp_cpu_group = get_current_sae_tp_cpu_group()
            state_dict = gather_tp_state_dict_to_root_cpu(
                local_tp_state, base_sae, tp_cpu_group
            )
            if tp_rank == 0:
                assert state_dict is not None
                base_sae.postprocess_full_state_dict_for_inference(state_dict)
        else:
            if dp_rank != 0:
                return
            state_dict = base_sae.state_dict()
            base_sae.process_state_dict_for_saving_inference(state_dict)
            if tp_rank != 0:
                self._tp_barrier()
                return

        if tp_rank == 0:
            assert state_dict is not None
            save_file(state_dict, out_dir / SAE_WEIGHTS_FILENAME)
            with open(out_dir / SAE_CFG_FILENAME, "w") as f:
                json.dump(base_sae.cfg.get_inference_sae_cfg_dict(), f)
            save_file(
                {"sparsity": self.log_feature_sparsity_by_hook[hook_name]},
                out_dir / SPARSITY_FILENAME,
            )
        if self._is_fsdp:
            if tp_cpu_group is not None and dist.get_world_size(tp_cpu_group) > 1:
                dist.barrier(group=tp_cpu_group)
        else:
            self._tp_barrier()

    def save_checkpoint(self, checkpoint_name: str) -> None:
        checkpoint_base_path = self._checkpoint_base_path(checkpoint_name)
        if checkpoint_base_path is None:
            return
        checkpoint_path = Path(checkpoint_base_path) / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True, parents=True)

        if self._is_metric_writer_rank():
            with open(checkpoint_path / MULTI_SAE_MANIFEST_FILENAME, "w") as f:
                json.dump(self._manifest(), f)

        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            self._save_unified_models(checkpoint_path, inference=False)
        else:
            for hook_name in self.hook_names:
                self._save_one_checkpoint_model(checkpoint_path, hook_name)

        self.save_trainer_state(checkpoint_path)
        if self._is_metric_writer_rank():
            _write_checkpoint_complete_marker(checkpoint_path)

        if self.save_checkpoint_fn is not None and self._is_metric_writer_rank():
            self.save_checkpoint_fn(checkpoint_path=checkpoint_path)

    def _checkpoint_base_path(self, checkpoint_name: str) -> str | None:
        if (
            checkpoint_name.startswith("quiesce_")
            and self.cfg.quiesce_checkpoint_path is not None
        ):
            return self.cfg.quiesce_checkpoint_path
        return self.cfg.checkpoint_path

    def _save_one_checkpoint_model(self, checkpoint_path: Path, hook_name: str) -> None:
        sae = self.sae_by_hook[hook_name]
        base_sae = self.base_sae_by_hook[hook_name]
        out_dir = checkpoint_path / sanitize_hook_name_for_path(hook_name)
        out_dir.mkdir(exist_ok=True, parents=True)

        dp_rank = self._dp_rank()
        tp_rank = self._tp_rank()
        tp_cpu_group = None
        if self._is_fsdp:
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                local_tp_state = sae.state_dict()
            if dp_rank != 0:
                return
            tp_cpu_group = get_current_sae_tp_cpu_group()
            state_dict = gather_tp_state_dict_to_root_cpu(
                local_tp_state, base_sae, tp_cpu_group
            )
        else:
            if dp_rank != 0:
                return
            state_dict = base_sae.state_dict()
            base_sae.process_state_dict_for_saving(state_dict)
            if tp_rank != 0:
                self._tp_barrier()
                return

        if tp_rank == 0:
            assert state_dict is not None
            save_file(state_dict, out_dir / SAE_WEIGHTS_FILENAME)
            with open(out_dir / SAE_CFG_FILENAME, "w") as f:
                json.dump(base_sae.cfg.to_dict(), f)
            save_file(
                {"sparsity": self.log_feature_sparsity_by_hook[hook_name]},
                out_dir / SPARSITY_FILENAME,
            )
        if state_dict is not None:
            del state_dict
        torch.cuda.empty_cache()
        if self._is_fsdp:
            if tp_cpu_group is not None and dist.get_world_size(tp_cpu_group) > 1:
                dist.barrier(group=tp_cpu_group)
        else:
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
            optimizer_by_hook_by_name = self._build_named_optimizer_state_for_save()
            optimizer_state = {
                "optimizer_state_format": MULTI_SAE_OPTIMIZER_STATE_FORMAT,
            }
            if tp_rank != 0:
                return
        state = {
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
            "multi_sae_distributed_architecture": self.multi_sae_distributed_architecture,
        }
        if not self._is_fsdp:
            for hook_name in self.hook_names:
                hook_dir = checkpoint_path / sanitize_hook_name_for_path(hook_name)
                hook_dir.mkdir(exist_ok=True, parents=True)
                _save_hook_optimizer_state_safetensors(
                    hook_dir,
                    optimizer_by_hook_by_name[hook_name],
                )
                torch.save(
                    {
                        "hook_name": hook_name,
                        "optimizer_state_format": MULTI_SAE_OPTIMIZER_STATE_FORMAT,
                        "act_freq_scores": self.act_freq_scores_by_hook[hook_name],
                        "n_forward_passes_since_fired": self.n_forward_passes_since_fired_by_hook[hook_name],
                        "n_frac_active_samples": self.n_frac_active_samples_by_hook[hook_name],
                    },
                    hook_dir / HOOK_STATE_FILENAME,
                )
        if self._pp_rank() != 0:
            return
        torch.save(state, checkpoint_path / TRAINER_STATE_FILENAME)

    def load_trainer_state(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path / TRAINER_STATE_FILENAME, map_location="cpu")
        saved_architecture = state.get(
            "multi_sae_distributed_architecture",
            "legacy_per_hook_wrapper",
        )
        if saved_architecture != self.multi_sae_distributed_architecture:
            raise ValueError(
                "Cannot resume multi-SAE checkpoint saved with "
                f"multi_sae_distributed_architecture='{saved_architecture}' "
                "using current "
                f"multi_sae_distributed_architecture='{self.multi_sae_distributed_architecture}'."
            )
        self._load_checkpoint_models(checkpoint_path)
        hook_state_paths = {
            hook_name: checkpoint_path
            / sanitize_hook_name_for_path(hook_name)
            / HOOK_STATE_FILENAME
            for hook_name in self.hook_names
        }
        has_local_hook_states = all(path.exists() for path in hook_state_paths.values())
        hook_optimizer_paths = {
            hook_name: checkpoint_path
            / sanitize_hook_name_for_path(hook_name)
            / HOOK_OPTIMIZER_STATE_FILENAME
            for hook_name in self.hook_names
        }
        has_hook_optimizer_safetensors = all(
            path.exists() for path in hook_optimizer_paths.values()
        )
        if state["hook_names"] != self.hook_names and not has_local_hook_states:
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
        elif has_hook_optimizer_safetensors:
            optimizer_by_hook_by_name = {
                hook_name: _load_hook_optimizer_state_safetensors(
                    checkpoint_path / sanitize_hook_name_for_path(hook_name),
                    self.base_sae_by_hook[hook_name],
                    getattr(self.base_sae_by_hook[hook_name], "_tp_group", None),
                )
                for hook_name in self.hook_names
            }
            self._load_named_optimizer_state(
                optimizer_by_hook_by_name,
                already_processed=True,
            )
        elif has_local_hook_states:
            optimizer_by_hook_by_name = {}
            for hook_name, path in hook_state_paths.items():
                hook_state = torch.load(path, map_location="cpu")
                optimizer_by_hook_by_name[hook_name] = hook_state["optimizer_state"]
            self._load_named_optimizer_state(optimizer_by_hook_by_name)
        elif "optimizer_by_hook_by_name" in state:
            self._load_named_optimizer_state(state["optimizer_by_hook_by_name"])
        else:
            self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        for hook_name in self.hook_names:
            if has_local_hook_states:
                hook_state = torch.load(hook_state_paths[hook_name], map_location="cpu")
                self.act_freq_scores_by_hook[hook_name] = hook_state[
                    "act_freq_scores"
                ].to(self.cfg.device)
                self.n_forward_passes_since_fired_by_hook[hook_name] = hook_state[
                    "n_forward_passes_since_fired"
                ].to(self.cfg.device)
                self.n_frac_active_samples_by_hook[hook_name] = hook_state[
                    "n_frac_active_samples"
                ]
            else:
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
            state_optimizer = self._overlap_optimizer_by_hook.get(
                hook_name,
                self.optimizer,
            )
            for name, param in base_sae.named_parameters():
                state = state_optimizer.state.get(param)
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
        self,
        optimizer_state_by_hook: dict[str, dict[str, dict[str, Any]]],
        *,
        already_processed: bool = False,
    ) -> None:
        self.optimizer.state.clear()
        for optimizer in self._overlap_optimizer_by_hook.values():
            optimizer.state.clear()
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            state_optimizer = self._overlap_optimizer_by_hook.get(
                hook_name,
                self.optimizer,
            )
            hook_state = deepcopy(optimizer_state_by_hook.get(hook_name, {}))
            if not already_processed:
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
                state_optimizer.state[param] = loaded_state

    def _load_checkpoint_models(self, checkpoint_path: Path) -> None:
        if self.multi_sae_distributed_architecture == "unified_multi_hook":
            self._load_unified_checkpoint_models(checkpoint_path)
            return
        for hook_name in self.hook_names:
            self._load_one_checkpoint_model(checkpoint_path, hook_name)

    def _load_unified_checkpoint_models(self, checkpoint_path: Path) -> None:
        if self.multi_hook_sae is None:
            raise ValueError("unified_multi_hook requires multi_hook_sae")
        root = self._multi_hook_root_module()
        state_dict_by_hook: dict[str, dict[str, Any]] = {}
        for hook_name in self.hook_names:
            base_sae = self.base_sae_by_hook[hook_name]
            hook_dir = checkpoint_path / sanitize_hook_name_for_path(hook_name)
            filepath = hook_dir / SAE_WEIGHTS_FILENAME
            tp_group = getattr(base_sae, "_tp_group", None)
            if tp_group is not None and dist.get_world_size(tp_group) > 1:
                state_dict = _load_tp_sharded_state_dict(filepath, base_sae, tp_group)
            else:
                state_dict = load_file(filepath)
                base_sae.process_state_dict_for_loading(state_dict)
            state_dict_by_hook[hook_name] = state_dict

        root_state_dict = root.merge_state_dict_by_hook(state_dict_by_hook)
        if self._is_fsdp:
            with FSDP.state_dict_type(
                self.multi_hook_sae,
                StateDictType.FULL_STATE_DICT,
            ):
                self.multi_hook_sae.load_state_dict(root_state_dict)
        elif isinstance(self.multi_hook_sae, DDP):
            self.multi_hook_sae.module.load_state_dict(root_state_dict)
        else:
            self.multi_hook_sae.load_state_dict(root_state_dict)
        for hook_name in self.hook_names:
            self._debug_log_loaded_model_state(
                checkpoint_path,
                hook_name,
                self.base_sae_by_hook[hook_name],
            )

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
        self._debug_log_loaded_model_state(checkpoint_path, hook_name, base_sae)

    def _debug_log_loaded_model_state(
        self,
        checkpoint_path: Path,
        hook_name: str,
        base_sae: TrainingSAE[Any],
    ) -> None:
        if (
            os.environ.get("SAELENS_DEBUG_CHECKPOINT_LOAD") != "1"
            or self.cfg.output_path is None
        ):
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        path = (
            Path(self.cfg.output_path)
            / f"debug_checkpoint_load_rank{rank}_pid{os.getpid()}.jsonl"
        )
        tensors = {}
        for name, value in base_sae.state_dict().items():
            if torch.is_tensor(value):
                value_f = value.detach().float()
                tensors[name] = {
                    "shape": list(value.shape),
                    "norm": float(value_f.norm().cpu().item()),
                    "mean": float(value_f.mean().cpu().item()),
                    "std": float(value_f.std(unbiased=False).cpu().item()),
                }
        record = {
            "event": "checkpoint_model_loaded",
            "checkpoint_path": str(checkpoint_path),
            "hook_name": hook_name,
            "rank": rank,
            "tp_rank": self._tp_rank(),
            "pp_rank": self._pp_rank(),
            "tensors": tensors,
        }
        with open(path, "a") as f:
            json.dump(record, f)
            f.write("\n")

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
        pp_rank = self._pp_rank()
        if (
            self.rank_local_mse_history_path is None
            and self.cfg.output_path is not None
            and self._dp_rank() == 0
            and self._tp_rank() == 0
            and pp_rank > 0
        ):
            output_path = Path(self.cfg.output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            self.rank_local_mse_history_path = (
                output_path / f"mse_history_pp{pp_rank}_rank{rank}.jsonl"
            )
            if not getattr(self.cfg, "append_history_logs", False):
                self.rank_local_mse_history_path.write_text("")
        if self.mse_history_path is None and self.rank_local_mse_history_path is None:
            if self.debug_mse_history_path is None:
                return
        if self.mse_history_path is not None and pp_rank == 0:
            with open(self.mse_history_path, "a") as f:
                json.dump(record, f)
                f.write("\n")
        if self.rank_local_mse_history_path is not None:
            rank_record = {
                **record,
                "rank": dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0,
                "tp_rank": self._tp_rank(),
                "pp_rank": self._pp_rank(),
                "dp_rank": self._dp_rank(),
            }
            with open(self.rank_local_mse_history_path, "a") as f:
                json.dump(rank_record, f)
                f.write("\n")
        if self.debug_mse_history_path is not None:
            debug_record = {
                **record,
                "rank": dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0,
                "tp_rank": self._tp_rank(),
                "pp_rank": self._pp_rank(),
                "dp_rank": self._dp_rank(),
            }
            with open(self.debug_mse_history_path, "a") as f:
                json.dump(debug_record, f)
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

    def _start_memory_phase_step(self) -> None:
        if not self._profile_memory:
            self._memory_phase_step_active = False
            return
        save_every = int(getattr(self.cfg, "save_memory_every_n_steps", 0))
        self._memory_phase_step_active = (
            save_every > 0 and (self.n_training_steps + 1) % save_every == 0
        )
        self._memory_phase_records = []

    def _device_sampler_loop(self, wall_t0: float) -> None:
        # Runs in a daemon thread, sampling device-memory at a fixed wall-clock
        # interval (default 1 Hz). Each line records the device-used watermark
        # (mem_get_info: total - free, includes the CUDA context and any other
        # process on the device) alongside this process' allocator view, so the
        # timeline can be compared against the per-phase snapshots.
        device = self.cfg.device
        with open(self.device_history_path, "a") as f:  # type: ignore[arg-type]
            while not self._device_sampler_stop.is_set():
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                record = {
                    "t_s": time.perf_counter() - wall_t0,
                    "rank": self._memory_rank,
                    "step": self.n_training_steps + 1,
                    "n_training_samples": self.n_training_samples,
                    "device_used_mb": (total_bytes - free_bytes) / 1024**2,
                    "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                self._device_sampler_stop.wait(self._device_sampler_interval_s)

    def _start_device_sampler(self) -> None:
        if not self._profile_memory or self.device_history_path is None:
            return
        if torch.device(self.cfg.device).type != "cuda":
            return
        self._device_sampler_stop.clear()
        self._device_sampler_thread = threading.Thread(
            target=self._device_sampler_loop,
            args=(time.perf_counter(),),
            name=f"sae-device-sampler-rank{self._memory_rank}",
            daemon=True,
        )
        self._device_sampler_thread.start()

    def _stop_device_sampler(self) -> None:
        if self._device_sampler_thread is None:
            return
        self._device_sampler_stop.set()
        self._device_sampler_thread.join(timeout=5.0)
        self._device_sampler_thread = None

    def _reset_memory_phase_peak(self) -> None:
        if not self._memory_phase_step_active:
            return
        torch.cuda.reset_peak_memory_stats(self.cfg.device)

    def _maybe_start_memory_timeline(self) -> None:
        """Begin recording the full alloc/free history for the target step.

        Captures every allocation/free event together with the Python call
        stack (``stacks="all"``, ``context="all"``) and the peak-moment block
        layout. In memory_viz the forward/backward/optimizer phases are
        separable both along the time axis (they run in sequence) and by the
        captured stack (each allocation shows whether it came from an SAE
        forward, autograd, or ``optimizer.step``).
        """
        if (
            self.memory_timeline_path is None
            or self.n_training_steps != self._memory_timeline_step
            or self._memory_timeline_active
        ):
            return
        if torch.device(self.cfg.device).type != "cuda":
            return
        torch.cuda.synchronize(self.cfg.device)
        torch.cuda.memory._record_memory_history(
            max_entries=1_000_000,
            stacks="all",
            context="all",
        )
        self._memory_timeline_active = True

    def _maybe_stop_memory_timeline(self) -> None:
        """Dump the recorded history for the target step and stop recording."""
        if not self._memory_timeline_active:
            return
        assert self.memory_timeline_path is not None
        torch.cuda.synchronize(self.cfg.device)
        torch.cuda.memory._dump_snapshot(str(self.memory_timeline_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        self._memory_timeline_active = False


    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _tensor_tree_bytes(self, value: Any, seen: set[int]) -> int:
        if torch.is_tensor(value):
            if value.device.type != "cuda":
                return 0
            ident = id(value)
            if ident in seen:
                return 0
            seen.add(ident)
            return self._tensor_bytes(value)
        if isinstance(value, dict):
            return sum(self._tensor_tree_bytes(v, seen) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._tensor_tree_bytes(v, seen) for v in value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return sum(
                self._tensor_tree_bytes(getattr(value, field.name), seen)
                for field in dataclasses.fields(value)
            )
        return 0

    def _data_provider_buffer_bytes(self, seen: set[int]) -> int:
        """Sum GPU bytes held by the data provider's internal buffers.

        Different providers expose different buffer attributes:

        - GpuStreamingActivationProvider: ``_pool_by_hook`` and
          ``_serving_by_hook`` (dicts) plus ``_chunk_buffer`` (list of dicts)
        - StreamingActivationProvider: ``_mixing_pool`` (a single tensor)
        - GpuDirectDataProvider: wraps another provider in ``_inner``

        Reuses the shared ``seen`` set, so a buffer tensor that is the *same
        object* as an already-counted batch tensor is not double counted; the
        pool therefore contributes only the portion not yet handed out as a
        batch. Cross-process pools owned by another process are not
        Python-reachable here, so they contribute nothing, keeping this stat
        consistent with this process's ``memory_allocated``.
        """
        provider: Any = self.data_provider
        total = 0
        visited: set[int] = set()
        while provider is not None and id(provider) not in visited:
            visited.add(id(provider))
            for attr in (
                "_pool_by_hook",
                "_serving_by_hook",
                "_chunk_buffer",
                "_mixing_pool",
            ):
                value = getattr(provider, attr, None)
                if value is not None:
                    total += self._tensor_tree_bytes(value, seen)
            provider = getattr(provider, "_inner", None)
        return total

    def _component_memory_stats_mb(self) -> dict[str, float]:
        seen: set[int] = set()
        param_bytes = 0
        grad_bytes = 0
        optimizer_state_bytes = 0

        for hook_name, sae in self.sae_by_hook.items():
            state_optimizer = self._overlap_optimizer_by_hook.get(
                hook_name,
                self.optimizer,
            )
            for param in sae.parameters():
                if param.device.type == "cuda":
                    param_bytes += self._tensor_tree_bytes(param, seen)
                if param.grad is not None and param.grad.device.type == "cuda":
                    grad_bytes += self._tensor_tree_bytes(param.grad, seen)
                state = state_optimizer.state.get(param, {})
                optimizer_state_bytes += self._tensor_tree_bytes(state, seen)

        trainer_buffer_values: list[Any] = [
            self.act_freq_scores_by_hook,
            self.n_forward_passes_since_fired_by_hook,
            self._pending_did_fire_max_by_hook,
        ]
        trainer_buffer_bytes = sum(
            self._tensor_tree_bytes(value, seen) for value in trainer_buffer_values
        )
        raw_batch_bytes = self._tensor_tree_bytes(
            self._memory_current_raw_batch_by_hook, seen
        )
        scaled_batch_bytes = self._tensor_tree_bytes(
            self._memory_current_scaled_batch_by_hook, seen
        )
        retained_outputs_bytes = self._tensor_tree_bytes(
            self._memory_retained_outputs, seen
        )
        current_outputs_bytes = self._tensor_tree_bytes(
            self._memory_current_outputs, seen
        )
        # Walk the data provider's buffers last, so any pool tensor that is the
        # same object as a batch tensor already counted above is deduped to 0;
        # the pool then contributes only its not-yet-served residual.
        data_provider_buffer_bytes = self._data_provider_buffer_bytes(seen)
        known_live_bytes = (
            param_bytes
            + grad_bytes
            + optimizer_state_bytes
            + trainer_buffer_bytes
            + raw_batch_bytes
            + scaled_batch_bytes
            + retained_outputs_bytes
            + current_outputs_bytes
            + data_provider_buffer_bytes
        )

        allocated_bytes = torch.cuda.memory_allocated(self.cfg.device)
        to_mb = 1 / 1024**2
        return {
            "params_mb": param_bytes * to_mb,
            "grads_mb": grad_bytes * to_mb,
            "optimizer_state_mb": optimizer_state_bytes * to_mb,
            "trainer_buffers_mb": trainer_buffer_bytes * to_mb,
            "raw_batch_mb": raw_batch_bytes * to_mb,
            "scaled_batch_mb": scaled_batch_bytes * to_mb,
            "retained_outputs_mb": retained_outputs_bytes * to_mb,
            "current_outputs_mb": current_outputs_bytes * to_mb,
            "outputs_mb": (retained_outputs_bytes + current_outputs_bytes) * to_mb,
            "data_provider_buffers_mb": data_provider_buffer_bytes * to_mb,
            "known_live_mb": known_live_bytes * to_mb,
            "unattributed_allocated_mb": (allocated_bytes - known_live_bytes) * to_mb,
        }

    def _record_memory_phase(self, phase: str, *, synchronize: bool = True) -> None:
        if not self._memory_phase_step_active:
            return
        # Optional: force the caching allocator to release empty blocks back
        # to the CUDA driver before the snapshot, so driver_used_mb reflects
        # *current* live tensors rather than the historical watermark. Off by
        # default — empty_cache stalls the device and adds tens of ms per
        # phase, so we only flip it on for memory-profiling runs. Enabled via
        # cfg.record_memory_empty_cache (or the legacy SAE_RECORD_EMPTY_CACHE=1).
        if synchronize:
            if (
                getattr(self.cfg, "record_memory_empty_cache", False)
                or os.environ.get("SAE_RECORD_EMPTY_CACHE") == "1"
            ):
                torch.cuda.synchronize(self.cfg.device)
                torch.cuda.empty_cache()
            torch.cuda.synchronize(self.cfg.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.cfg.device)
        record = {
            "step": self.n_training_steps + 1,
            "n_training_samples": self.n_training_samples,
            "rank": self._memory_rank,
            "phase": phase,
            "allocated_mb": torch.cuda.memory_allocated(self.cfg.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.cfg.device) / 1024**2,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.cfg.device) / 1024**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.cfg.device) / 1024**2,
            "driver_used_mb": (total_bytes - free_bytes) / 1024**2,
        }
        record.update(self._component_memory_stats_mb())
        self._memory_phase_records.append(record)

        if os.environ.get("SAE_DUMP_LIVE_TENSORS") == "1" and phase in {
            "after_data_fetch", "after_optimizer_step"
        } and self.n_training_steps + 1 in {15, 16}:
            self._dump_live_tensors(phase, self.n_training_steps + 1)
            if os.environ.get("SAE_DUMP_ALLOC_SNAPSHOT") == "1":
                self._dump_alloc_snapshot(phase, self.n_training_steps + 1)
            if os.environ.get("SAE_DUMP_SAVED_TENSORS") == "1":
                self._dump_saved_tensors_via_grad_fn(phase, self.n_training_steps + 1)

        torch.cuda.reset_peak_memory_stats(self.cfg.device)

    def _dump_live_tensors(self, phase: str, step: int) -> None:
        import gc as _gc
        path = self.memory_phase_history_path
        if path is None:
            return
        out = path.parent / f"live_tensors_rank{self._memory_rank}.jsonl"
        by_shape: dict[tuple, dict] = {}
        for obj in _gc.get_objects():
            try:
                if not torch.is_tensor(obj):
                    continue
                if obj.device.type != "cuda":
                    continue
                key = (tuple(obj.shape), str(obj.dtype))
                d = by_shape.setdefault(key, {"count": 0, "bytes": 0})
                d["count"] += 1
                d["bytes"] += obj.numel() * obj.element_size()
            except Exception:
                continue
        rows = sorted(by_shape.items(), key=lambda kv: -kv[1]["bytes"])
        snap = {
            "step": step, "phase": phase, "rank": self._memory_rank,
            "allocated_mb": torch.cuda.memory_allocated(self.cfg.device) / 1024**2,
            "rows": [
                {"shape": list(k[0]), "dtype": k[1],
                 "count": v["count"], "MB": v["bytes"]/1024**2}
                for k, v in rows[:40]
            ],
        }
        with open(out, "a") as f:
            f.write(json.dumps(snap) + "\n")

    def _dump_alloc_snapshot(self, phase: str, step: int) -> None:
        """Use torch.cuda.memory._snapshot() to enumerate ALL live storages,
        not just gc-visible Python tensors."""
        path = self.memory_phase_history_path
        if path is None:
            return
        out = path.parent / f"alloc_snapshot_rank{self._memory_rank}_{phase}_step{step}.json"
        snap = torch.cuda.memory._snapshot()
        # Reduce: aggregate live segments by allocator size
        live_blocks = []
        for seg in snap.get("segments", []):
            for blk in seg.get("blocks", []):
                if blk.get("state") in ("active_allocated", "active_pending_free", "inactive"):
                    live_blocks.append({
                        "size_MB": blk["size"] / 1024**2,
                        "state": blk.get("state"),
                        "frames": [
                            {"name": f.get("name"), "filename": f.get("filename", "")[-60:],
                             "line": f.get("line")}
                            for f in (blk.get("frames") or [])[:16]
                        ],
                    })
        live_blocks.sort(key=lambda b: -b["size_MB"])
        with open(out, "w") as f:
            json.dump({"phase": phase, "step": step, "rank": self._memory_rank,
                       "n_blocks": len(live_blocks),
                       "total_MB": sum(b["size_MB"] for b in live_blocks),
                       "blocks": live_blocks[:60]}, f, indent=2)

    def _dump_saved_tensors_via_grad_fn(self, phase: str, step: int) -> None:
        """Walk grad_fn graphs of retained_outputs.* to enumerate autograd-saved tensors."""
        path = self.memory_phase_history_path
        if path is None:
            return
        out = path.parent / f"saved_via_gradfn_rank{self._memory_rank}_{phase}_step{step}.json"
        records = []

        def walk(grad_fn, depth=0, visited=None):
            if visited is None:
                visited = set()
            if grad_fn is None or id(grad_fn) in visited or depth > 20:
                return
            visited.add(id(grad_fn))
            entry = {"depth": depth, "name": type(grad_fn).__name__, "saved_tensors": []}
            try:
                # Inspect any 'saved_tensors' attribute (some custom Functions expose it)
                for attr in dir(grad_fn):
                    if attr.startswith("_saved_"):
                        val = getattr(grad_fn, attr)
                        if torch.is_tensor(val) and val.device.type == "cuda":
                            entry["saved_tensors"].append({
                                "attr": attr,
                                "shape": list(val.shape),
                                "dtype": str(val.dtype),
                                "MB": val.numel() * val.element_size() / 1024**2,
                                "data_ptr": val.data_ptr(),
                            })
            except Exception as e:
                entry["error"] = str(e)
            records.append(entry)
            try:
                for nf in grad_fn.next_functions:
                    if nf[0] is not None:
                        walk(nf[0], depth + 1, visited)
            except Exception:
                pass

        if self._memory_retained_outputs is not None:
            for hook_name, output in self._memory_retained_outputs.items():
                for field_name in ["hidden_pre", "feature_acts", "sae_out", "loss"]:
                    t = getattr(output, field_name, None)
                    if torch.is_tensor(t) and t.grad_fn is not None:
                        records.append({"depth": -1, "name": f"=== {hook_name}.{field_name} ===",
                                        "saved_tensors": []})
                        walk(t.grad_fn)

        with open(out, "w") as f:
            json.dump({"phase": phase, "step": step, "rank": self._memory_rank,
                       "records": records}, f, indent=2)

    def _aggregate_memory_phase_stats(self) -> dict[str, float]:
        if not self._memory_phase_records:
            return {
                "peak_step_allocated_mb": torch.cuda.max_memory_allocated(self.cfg.device) / 1024**2,
                "peak_step_reserved_mb": torch.cuda.max_memory_reserved(self.cfg.device) / 1024**2,
            }
        return {
            "peak_step_allocated_mb": max(
                float(record["peak_allocated_mb"])
                for record in self._memory_phase_records
            ),
            "peak_step_reserved_mb": max(
                float(record["peak_reserved_mb"])
                for record in self._memory_phase_records
            ),
            "end_step_allocated_mb": float(
                self._memory_phase_records[-1]["allocated_mb"]
            ),
            "end_step_reserved_mb": float(
                self._memory_phase_records[-1]["reserved_mb"]
            ),
        }

    def _record_memory_if_needed(self, memory_stats: dict[str, float]) -> None:
        if self.memory_history_path is None or not self._memory_phase_step_active:
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
        if self.memory_phase_history_path is None:
            return
        with open(self.memory_phase_history_path, "a") as f:
            for phase_record in self._memory_phase_records:
                json.dump(phase_record, f)
                f.write("\n")


def _load_tp_sharded_state_dict(
    filepath: Path,
    base_sae: Any,
    tp_group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    """Load only this TP rank's checkpoint tensor slices from safetensors.

    Thin wrapper around :func:`sae_lens.training.tp_checkpoint.load_tp_sharded_state_dict`
    kept here so existing callers within ``multi_sae_trainer`` need not be
    rewritten. New call sites should import from ``tp_checkpoint`` directly.
    """
    from sae_lens.training.tp_checkpoint import load_tp_sharded_state_dict

    return load_tp_sharded_state_dict(filepath, base_sae, tp_group)


def _tp_param_shard_dims(base_sae: Any) -> dict[str, int | None]:
    if hasattr(base_sae, "_tp_param_shard_dims"):
        return base_sae._tp_param_shard_dims()
    return {}


def _optimizer_state_shard_dim(
    param_name: str,
    state_value: torch.Tensor,
    base_sae: Any,
) -> int | None:
    if state_value.ndim == 0:
        return None
    return _tp_param_shard_dims(base_sae).get(param_name)


def _flatten_optimizer_tensor_state(
    optimizer_state_by_name: dict[str, dict[str, Any]]
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, dict[str, Any]] = {}
    for param_name, param_state in optimizer_state_by_name.items():
        for state_name, value in param_state.items():
            flat_key = f"{param_name}{_OPTIMIZER_STATE_KEY_SEP}{state_name}"
            if torch.is_tensor(value):
                tensors[flat_key] = value.detach().cpu().contiguous()
            else:
                meta[flat_key] = {
                    "param_name": param_name,
                    "state_name": state_name,
                    "value": value,
                }
    return tensors, meta


def _save_hook_optimizer_state_safetensors(
    hook_dir: Path,
    optimizer_state_by_name: dict[str, dict[str, Any]],
) -> None:
    tensors, meta = _flatten_optimizer_tensor_state(optimizer_state_by_name)
    if tensors:
        save_file(tensors, hook_dir / HOOK_OPTIMIZER_STATE_FILENAME)
    else:
        (hook_dir / HOOK_OPTIMIZER_STATE_FILENAME).unlink(missing_ok=True)
    with open(hook_dir / HOOK_OPTIMIZER_META_FILENAME, "w") as f:
        json.dump(meta, f)


def _load_hook_optimizer_state_safetensors(
    hook_dir: Path,
    base_sae: Any,
    tp_group: dist.ProcessGroup | None,
) -> dict[str, dict[str, Any]]:
    from safetensors import safe_open
    from sae_lens.util import str_to_dtype

    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    path = hook_dir / HOOK_OPTIMIZER_STATE_FILENAME
    meta_path = hook_dir / HOOK_OPTIMIZER_META_FILENAME
    optimizer_state: dict[str, dict[str, Any]] = {}
    dtype_map = {
        "F32": "float32", "BF16": "bfloat16", "F16": "float16",
        "F64": "float64", "I32": "int32", "I64": "int64",
    }

    with safe_open(str(path), framework="pt", device="cpu") as f:
        for flat_key in f.keys():
            param_name, state_name = flat_key.split(_OPTIMIZER_STATE_KEY_SEP, 1)
            sl = f.get_slice(flat_key)
            shape = list(sl.get_shape())
            dtype_str = str(sl.get_dtype())
            dtype = str_to_dtype(dtype_map.get(dtype_str, dtype_str.lower()))
            shard_dim = (
                _optimizer_state_shard_dim(param_name, torch.empty(shape), base_sae)
                if tp_size > 1
                else None
            )
            if shard_dim is None:
                value = f.get_tensor(flat_key)
            else:
                full_size = shape[shard_dim]
                assert full_size % tp_size == 0, (
                    f"Optimizer tensor '{flat_key}' size {full_size} on dim {shard_dim} "
                    f"not divisible by tp_size={tp_size}"
                )
                shard_size = full_size // tp_size
                slices: list[slice] = [slice(None)] * len(shape)
                slices[shard_dim] = slice(
                    tp_rank * shard_size,
                    (tp_rank + 1) * shard_size,
                )
                value = sl[tuple(slices)].to(dtype=dtype)
            optimizer_state.setdefault(param_name, {})[state_name] = value

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for item in meta.values():
            optimizer_state.setdefault(item["param_name"], {})[
                item["state_name"]
            ] = item["value"]

    return optimizer_state
