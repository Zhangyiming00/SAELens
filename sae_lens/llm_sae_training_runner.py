import json
import math
import os
import signal
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic

import torch
import torch.distributed as dist
import wandb
from safetensors.torch import save_file
from simple_parsing import ArgumentParser
from transformer_lens.hook_points import HookedRootModule
from typing_extensions import deprecated

from sae_lens import logger
from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.constants import (
    RUNNER_CFG_FILENAME,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.distributed import (
    get_dp_group,
    get_sae_dp_size,
    get_sae_root_rank,
    get_tp_group,
    get_vllm_dp_p2p_group,
    get_vllm_dp_rank,
    get_vllm_dp_size,
    get_vllm_world_ranks,
    init_distributed,
    is_sae_active,
    is_vllm_active,
    is_vllm_dp_root,
    preinit_vllm_distributed,
)
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import load_model, load_tokenizer_only_model
from sae_lens.registry import SAE_TRAINING_CLASS_REGISTRY
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainingSAE,
    TrainingSAEConfig,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.types import DataProvider
from sae_lens.util import temporary_seed


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


@dataclass
class LLMSaeEvaluator(Generic[T_TRAINING_SAE]):
    model: HookedRootModule
    activations_store: ActivationsStore
    eval_batch_size_prompts: int | None
    n_eval_batches: int
    model_kwargs: dict[str, Any]

    def __call__(
        self,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        activation_scaler: ActivationScaler,
    ) -> dict[str, Any]:
        exclude_special_tokens = False
        if self.activations_store.exclude_special_tokens is not None:
            exclude_special_tokens = (
                self.activations_store.exclude_special_tokens.tolist()
            )

        eval_config = EvalConfig(
            batch_size_prompts=self.eval_batch_size_prompts,
            n_eval_reconstruction_batches=self.n_eval_batches,
            n_eval_sparsity_variance_batches=self.n_eval_batches,
            compute_ce_loss=True,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
        )

        eval_metrics, _ = run_evals(
            sae=sae,
            activation_store=self.activations_store,
            model=self.model,
            activation_scaler=activation_scaler,
            eval_config=eval_config,
            exclude_special_tokens=exclude_special_tokens,
            model_kwargs=self.model_kwargs,
        )  # not calculating featurwise metrics here.

        # Remove eval metrics that are already logged during training
        eval_metrics.pop("metrics/explained_variance", None)
        eval_metrics.pop("metrics/explained_variance_std", None)
        eval_metrics.pop("metrics/l0", None)
        eval_metrics.pop("metrics/l1", None)
        eval_metrics.pop("metrics/mse", None)

        # Remove metrics that are not useful for wandb logging
        eval_metrics.pop("metrics/total_tokens_evaluated", None)

        return eval_metrics


class LanguageModelSAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig[Any]
    model: HookedRootModule
    sae: TrainingSAE[Any] | None
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG],
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE[Any] | None = None,
        resume_from_checkpoint: Path | str | None = None,
        tp_size: int = 1,
        shared_tp_size: int | None = None,
        vllm_tp_size: int | None = None,
        sae_tp_size: int | None = None,
        sae_dp_size: int = 1,
        vllm_dp_size: int = 1,
        use_shard_routing: bool = True,
        streaming_mode: bool = False,
    ):
        if override_dataset is not None:
            logger.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg
        self.hook_names = (
            list(cfg.hook_names)
            if cfg.hook_names is not None and len(cfg.hook_names) > 0
            else [cfg.hook_name]
        )
        self.is_multi_sae = len(self.hook_names) > 1
        if resume_from_checkpoint is not None:
            self.cfg.resume_from_checkpoint = str(resume_from_checkpoint)
        self.sae_dp_size = sae_dp_size
        self.vllm_dp_size = vllm_dp_size
        self.use_shard_routing = use_shard_routing
        inferred_cfg_vllm_tp_size = int(
            self.cfg.model_from_pretrained_kwargs.get("tensor_parallel_size", 1)
        )
        self.shared_tp_size = shared_tp_size
        if (
            self.shared_tp_size is None
            and sae_tp_size is None
            and vllm_tp_size is None
            and tp_size > 1
            and vllm_dp_size == 1
        ):
            # Backward-compatible shared-TP semantics.
            self.shared_tp_size = tp_size

        if self.shared_tp_size is not None:
            self.sae_tp_size = self.shared_tp_size
            self.vllm_tp_size = self.shared_tp_size
            if (
                "tensor_parallel_size" in self.cfg.model_from_pretrained_kwargs
                and inferred_cfg_vllm_tp_size != self.shared_tp_size
            ):
                raise ValueError(
                    "shared_tp_size does not match "
                    "cfg.model_from_pretrained_kwargs['tensor_parallel_size']"
                )
        else:
            self.sae_tp_size = tp_size if sae_tp_size is None else sae_tp_size
            self.vllm_tp_size = (
                inferred_cfg_vllm_tp_size if vllm_tp_size is None else vllm_tp_size
            )
            if (
                "tensor_parallel_size" in self.cfg.model_from_pretrained_kwargs
                and inferred_cfg_vllm_tp_size != self.vllm_tp_size
            ):
                raise ValueError(
                    "vllm_tp_size does not match "
                    "cfg.model_from_pretrained_kwargs['tensor_parallel_size']"
                )

        self.streaming_mode = streaming_mode or cfg.streaming_mode
        if self.streaming_mode:
            self._streaming_init(cfg)
            return

        # Initialize distributed process groups for SAE TP/DP.
        # With torchrun, dist.init_process_group is already called; skip it.
        if (
            self.vllm_tp_size > 1
            or self.sae_tp_size > 1
            or sae_dp_size > 1
            or vllm_dp_size > 1
        ):
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            if use_shard_routing:
                from sae_lens.distributed_v2 import init_distributed_v2
                batch_size = cfg.store_batch_size_prompts * len(
                    range(cfg.context_size)[slice(*cfg.seqpos_slice)]
                )
                init_distributed_v2(
                    P=vllm_dp_size,
                    Q=sae_dp_size,
                    vllm_tp_size=self.vllm_tp_size,
                    sae_tp_size=self.sae_tp_size,
                    batch_size=batch_size,
                )
            elif self.shared_tp_size is not None:
                init_distributed(
                    shared_tp_size=self.shared_tp_size,
                    sae_dp_size=sae_dp_size,
                )
            else:
                init_distributed(
                    sae_tp_size=self.sae_tp_size,
                    vllm_tp_size=self.vllm_tp_size,
                    vllm_dp_size=vllm_dp_size,
                    sae_dp_size=sae_dp_size,
                )
        self._sync_run_paths_across_ranks()

        if use_shard_routing and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod
            self.sae_active = v2_mod.is_consumer()
            self.vllm_active = v2_mod.is_producer()
            self.uses_split_roles = v2_mod.is_producer() != v2_mod.is_consumer()
        else:
            self.sae_active = is_sae_active() if dist.is_initialized() else True
            self.vllm_active = is_vllm_active() if dist.is_initialized() else True
            self.uses_split_roles = (
                self.vllm_tp_size != self.sae_tp_size
                or self.vllm_dp_size != self.sae_dp_size
            )
        self.uses_vllm_dp_fan_in = self.vllm_dp_size > self.sae_dp_size
        self.uses_matched_dp = self.vllm_dp_size == self.sae_dp_size and self.vllm_dp_size > 1

        if dist.is_initialized():
            if use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                vllm_world_ranks = sorted(
                    r for ranks in v2_mod._producer_world_ranks.values() for r in ranks
                )
            else:
                vllm_world_ranks = get_vllm_world_ranks()
            os.environ["SAELENS_VLLM_WORLD_RANKS"] = ",".join(
                str(rank) for rank in vllm_world_ranks
            )

        # Pre-initialize vLLM parallel state when needed to avoid deadlock
        # on dist.new_group() calls that non-vLLM ranks would never enter.
        if dist.is_initialized() and (
            use_shard_routing or self.sae_tp_size > self.vllm_tp_size or vllm_dp_size > 1
        ):
            if use_shard_routing:
                preinit_vllm_distributed(vllm_world_ranks, self.vllm_tp_size)
            else:
                preinit_vllm_distributed(get_vllm_world_ranks(), self.vllm_tp_size)

        if self.uses_split_roles:
            if self.cfg.logger.log_to_wandb:
                raise ValueError(
                    "Prefix-overlap training currently requires log_to_wandb=False."
                )
            if self.cfg.n_eval_batches > 0:
                raise ValueError(
                    "Prefix-overlap training currently requires n_eval_batches=0."
                )
            if self.cfg.sae.normalize_activations == "expected_average_only_in":
                raise ValueError(
                    "Prefix-overlap training does not yet support "
                    "normalize_activations='expected_average_only_in'."
                )
        if self.is_multi_sae:
            if self.cfg.sae_dp_mode not in ("ddp", "fsdp"):
                raise ValueError("Multi-layer SAE training requires sae_dp_mode='ddp' or 'fsdp'.")
            if self.cfg.sae.normalize_activations == "expected_average_only_in":
                raise ValueError(
                    "Multi-layer SAE training does not yet support "
                    "normalize_activations='expected_average_only_in'."
                )
            if self.cfg.sae_dp_mode == "fsdp" and not dist.is_initialized():
                raise ValueError(
                    "Multi-layer SAE training with sae_dp_mode='fsdp' requires "
                    "torch distributed. Use the default/ddp mode for sae_dp_size=1."
                )
            if self.cfg.n_eval_batches > 0:
                raise ValueError("Multi-layer SAE training requires n_eval_batches=0 in v1.")
            if self.cfg.use_cached_activations:
                raise ValueError("Multi-layer SAE training does not support cached activations in v1.")
            if override_sae is not None or self.cfg.from_pretrained_path is not None:
                raise ValueError("Multi-layer SAE training does not support override/pretrained SAE in v1.")

        if (
            vllm_dp_size > 1
            and self.cfg.resume_from_checkpoint is not None
            and self.cfg.sae_dp_mode not in ("ddp", "fsdp")
        ):
            raise ValueError(
                "resume_from_checkpoint with vllm_dp_size > 1 is only "
                "supported with sae_dp_mode='ddp' or 'fsdp'"
            )
        if vllm_dp_size > 1 and self.cfg.use_cached_activations:
            raise ValueError(
                "use_cached_activations is not supported with vllm_dp_size > 1"
            )

        if override_model is None:
            if self.vllm_active:
                self.model = load_model(
                    self.cfg.model_class_name,
                    self.cfg.model_name,
                    device=self.cfg.device,
                    model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
                )
            else:
                self.model = load_tokenizer_only_model(
                    self.cfg.model_name,
                    self.cfg.device,
                )
        else:
            self.model = override_model

        # Determine dataset shard params for vLLM DP.
        ds_shard_index = 0
        ds_shard_count = 1
        if dist.is_initialized() and vllm_dp_size > 1:
            if use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                if v2_mod.is_producer():
                    ds_shard_index = v2_mod.get_producer_idx()
                    ds_shard_count = vllm_dp_size
            else:
                ds_shard_index = get_vllm_dp_rank()
                ds_shard_count = vllm_dp_size

        consumer_only = use_shard_routing and self.sae_active and not self.vllm_active
        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
            dataset_shard_index=ds_shard_index,
            dataset_shard_count=ds_shard_count,
            consumer_only=consumer_only,
        )

        self.sae_by_hook: dict[str, Any] = {}
        self.base_sae_by_hook: dict[str, TrainingSAE[Any]] = {}
        if self.is_multi_sae:
            if self.sae_active:
                self._init_multi_saes()
            self.sae = None
            self._base_sae = None
            return

        if self.sae_active:
            if override_sae is None:
                with temporary_seed(self.cfg.seed):
                    if self.cfg.from_pretrained_path is not None:
                        self.sae = TrainingSAE.load_from_disk(
                            self.cfg.from_pretrained_path, self.cfg.device
                        )
                    else:
                        self.sae = TrainingSAE.from_dict(
                            TrainingSAEConfig.from_dict(
                                self.cfg.get_training_sae_cfg_dict(),
                            ).to_dict()
                        )
            else:
                self.sae = override_sae
            self.sae.to(self.cfg.device)
        else:
            self.sae = None

        # Shard SAE weights across TP ranks if applicable.
        if self.sae is not None and self.sae_tp_size > 1:
            if self.use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                tp_group = v2_mod.get_sae_tp_group()
            else:
                tp_group = get_tp_group()
            if tp_group is not None and hasattr(self.sae, "shard_weights"):
                self.sae.shard_weights(tp_group)

        # _base_sae is always the raw module before any torch DP wrapper.
        # SAE compilation must happen on this module before FSDP wrapping; training
        # still enters through the wrapper so FSDP owns parameter all-gather/reshard.
        self._base_sae = self.sae
        if (
            self._base_sae is not None
            and self.cfg.sae_dp_mode == "fsdp"
            and self.cfg.resume_from_checkpoint is not None
        ):
            self._base_sae.load_weights_from_checkpoint(
                self.cfg.resume_from_checkpoint
            )

        # Wrap after TP sharding and optional raw-module compile when sae_dp_mode
        # requests a torch DP wrapper. self.sae may become FSDP/DDP.
        if self.sae is not None and self.cfg.sae_dp_mode in ("ddp", "fsdp"):
            if not dist.is_initialized():
                raise ValueError(
                    f"sae_dp_mode='{self.cfg.sae_dp_mode}' requires an initialized "
                    "distributed process group."
                )
            sae_dp_group = get_dp_group()
            if self.use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                sae_dp_group = v2_mod.get_sae_dp_group()
            if sae_dp_group is None or dist.get_world_size(sae_dp_group) <= 1:
                raise ValueError(
                    f"sae_dp_mode='{self.cfg.sae_dp_mode}' requires sae_dp_size > 1."
                )
            if self.cfg.sae_dp_mode == "fsdp":
                if sae_tp_size > 1:
                    raise ValueError(
                        "sae_dp_mode='fsdp' with sae_tp_size > 1 is not supported. "
                        "Use sae_tp_size=1 with FSDP."
                    )
                self._compile_sae_if_needed()
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.api import ShardingStrategy
                self.sae = FSDP(
                    self._base_sae,
                    process_group=sae_dp_group,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True,
                )
            else:
                from torch.nn.parallel import DistributedDataParallel as DDP
                device_ids = None
                output_device = None
                device = torch.device(self.cfg.device)
                if device.type == "cuda":
                    device_index = (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                    device_ids = [device_index]
                    output_device = device_index
                self._compile_sae_if_needed()
                ddp_kwargs = self._resolve_ddp_kwargs()
                self.sae = DDP(
                    self._base_sae,
                    process_group=sae_dp_group,
                    device_ids=device_ids,
                    output_device=output_device,
                    **ddp_kwargs,
                )
        else:
            self._compile_sae_if_needed()

    def _init_multi_saes(self) -> None:
        if self.cfg.sae_dp_mode == "fsdp" and not dist.is_initialized():
            raise ValueError("Multi-layer SAE training with FSDP requires torch distributed.")

        sae_dp_group = get_dp_group() if dist.is_initialized() else None
        if self.use_shard_routing and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod

            sae_dp_group = v2_mod.get_sae_dp_group()
        sae_dp_world_size = (
            dist.get_world_size(sae_dp_group)
            if sae_dp_group is not None and dist.is_initialized()
            else 1
        )
        if self.cfg.sae_dp_mode == "fsdp" and self.sae_tp_size > 1:
            raise ValueError(
                "sae_dp_mode='fsdp' with sae_tp_size > 1 is not supported. "
                "Use sae_tp_size=1 with FSDP."
            )
        if self.cfg.sae_dp_mode == "fsdp" and sae_dp_group is None:
            raise ValueError("Multi-layer SAE training with FSDP requires an SAE DP group.")
        ddp_kwargs_multi = (
            self._resolve_ddp_kwargs()
            if self.cfg.sae_dp_mode == "ddp" and sae_dp_world_size > 1
            else {}
        )

        for idx, hook_name in enumerate(self.hook_names):
            seed = (
                self.cfg.seed
                if self.cfg.multi_sae_seed_mode == "same"
                else self.cfg.seed + idx
            )
            with temporary_seed(seed):
                sae = TrainingSAE.from_dict(
                    TrainingSAEConfig.from_dict(
                        self.cfg.get_training_sae_cfg_dict(),
                    ).to_dict()
                )
            sae.to(self.cfg.device)
            sae.cfg.metadata.hook_name = hook_name
            sae.cfg.metadata.hook_head_index = self.cfg.hook_head_index
            sae.cfg.metadata.dataset_path = self.cfg.dataset_path
            sae.cfg.metadata.model_name = self.cfg.model_name
            sae.cfg.metadata.model_class_name = self.cfg.model_class_name
            sae.cfg.metadata.context_size = self.cfg.context_size
            sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
            sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
            sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
            self.base_sae_by_hook[hook_name] = sae

            if self.sae_tp_size > 1:
                if self.use_shard_routing:
                    import sae_lens.distributed_v2 as v2_mod

                    tp_group = v2_mod.get_sae_tp_group()
                else:
                    tp_group = get_tp_group()
                if tp_group is not None and hasattr(sae, "shard_weights"):
                    sae.shard_weights(tp_group)

            wrapped: Any
            if self.cfg.sae_dp_mode == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.api import ShardingStrategy

                wrapped = FSDP(
                    sae,
                    process_group=sae_dp_group,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True,
                )
            elif sae_dp_world_size > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                device_ids = None
                output_device = None
                device = torch.device(self.cfg.device)
                if device.type == "cuda":
                    device_index = (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                    device_ids = [device_index]
                    output_device = device_index
                wrapped = DDP(
                    sae,
                    process_group=sae_dp_group,
                    device_ids=device_ids,
                    output_device=output_device,
                    **ddp_kwargs_multi,
                )
            else:
                wrapped = sae
            self.sae_by_hook[hook_name] = wrapped

    def _resolve_ddp_kwargs(self) -> dict[str, Any]:
        ddp_kwargs: dict[str, Any] = {}
        if self.cfg.ddp_broadcast_buffers is not None:
            ddp_kwargs["broadcast_buffers"] = self.cfg.ddp_broadcast_buffers
        if self.cfg.ddp_find_unused_parameters is not None:
            ddp_kwargs["find_unused_parameters"] = self.cfg.ddp_find_unused_parameters
        if self.cfg.ddp_gradient_as_bucket_view is not None:
            ddp_kwargs["gradient_as_bucket_view"] = self.cfg.ddp_gradient_as_bucket_view
        if self.cfg.ddp_static_graph is not None:
            ddp_kwargs["static_graph"] = self.cfg.ddp_static_graph
        if self.cfg.ddp_bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb

        if (
            ddp_kwargs.get("static_graph") is True
            and ddp_kwargs.get("find_unused_parameters") is True
        ):
            message = (
                "DDP config conflict: static_graph=True with "
                "find_unused_parameters=True is not recommended."
            )
            if self.cfg.ddp_config_strict:
                raise ValueError(message)
            logger.warning(message + " Overriding find_unused_parameters=False.")
            ddp_kwargs["find_unused_parameters"] = False

        effective = {
            "broadcast_buffers": ddp_kwargs.get("broadcast_buffers", "torch_default"),
            "find_unused_parameters": ddp_kwargs.get(
                "find_unused_parameters", "torch_default"
            ),
            "gradient_as_bucket_view": ddp_kwargs.get(
                "gradient_as_bucket_view", "torch_default"
            ),
            "static_graph": ddp_kwargs.get("static_graph", "torch_default"),
            "bucket_cap_mb": ddp_kwargs.get("bucket_cap_mb", "torch_default"),
        }
        logger.info(f"Effective DDP config: {effective}")
        return ddp_kwargs

    def _sync_run_paths_across_ranks(self) -> None:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        run_paths = [self.cfg.checkpoint_path, self.cfg.output_path]
        dist.broadcast_object_list(run_paths, src=0)
        self.cfg.checkpoint_path = run_paths[0]
        self.cfg.output_path = run_paths[1]

    def run(self):
        """
        Run the training of the SAE.
        """
        if self.streaming_mode:
            import sae_lens.distributed_streaming as ds
            if ds.is_producer():
                self._run_streaming_producer_loop()
                return None
            else:
                return self._run_streaming_consumer_loop()

        if self.use_shard_routing and self.vllm_active and not self.sae_active:
            self._load_producer_resume_state_if_needed()
            self._run_producer_helper_loop_v2()
            return None

        if not self.sae_active:
            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = f"[prefix-debug rank{rank}] entering helper loop\n"
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)
            self._run_vllm_helper_loop()
            return None

        if self.is_multi_sae:
            return self._run_multi_sae()

        assert self.sae is not None
        self._set_sae_metadata()
        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=self.cfg.to_dict(),
                name=self.cfg.logger.run_name,
                id=self.cfg.logger.wandb_id,
            )

        evaluator = LLMSaeEvaluator(
            model=self.model,
            activations_store=self.activations_store,
            eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_batches=self.cfg.n_eval_batches,
            model_kwargs=self.cfg.model_kwargs,
        )

        sae_dp_group = get_dp_group()
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod
            sae_dp_group = v2_mod.get_sae_dp_group()
        # In FSDP mode the dp_group is already embedded in the FSDP wrapper; we
        # still pass it so the trainer can use it for sparsity/firing sync and rank checks.
        trainer = SAETrainer(
            sae=self.sae,
            base_sae=self._base_sae,
            data_provider=self.activations_store,
            evaluator=evaluator,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=(
                sae_dp_group
                if sae_dp_group is not None and dist.get_world_size(sae_dp_group) > 1
                else None
            ),
            token_count_weighted_dp=self.use_shard_routing,
        )

        if self.cfg.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.cfg.resume_from_checkpoint}")
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            if self.cfg.sae_dp_mode != "fsdp":
                self._base_sae.load_weights_from_checkpoint(
                    self.cfg.resume_from_checkpoint
                )
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.output_path is not None:
            self.save_final_sae(
                sae=sae,
                output_path=self.cfg.output_path,
                log_feature_sparsity=trainer.log_feature_sparsity,
            )

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return sae

    def _run_multi_sae(self) -> dict[str, TrainingSAE[Any]]:
        sae_dp_group = get_dp_group() if dist.is_initialized() else None
        if self.use_shard_routing and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod

            sae_dp_group = v2_mod.get_sae_dp_group()

        trainer = MultiSAETrainer(
            hook_names=self.hook_names,
            sae_by_hook=self.sae_by_hook,
            base_sae_by_hook=self.base_sae_by_hook,
            data_provider=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=sae_dp_group,
            token_count_weighted_dp=self.use_shard_routing,
            sae_dp_mode=self.cfg.sae_dp_mode,
            backward_mode=self.cfg.multi_sae_backward_mode,
            seed_mode=self.cfg.multi_sae_seed_mode,
        )
        if self.cfg.resume_from_checkpoint is not None:
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        result = self.run_multi_trainer_with_interruption_handling(trainer)
        if self.cfg.output_path is not None:
            trainer.save_final(self.cfg.output_path)
        return result

    def _run_vllm_helper_loop(self) -> None:
        """Pump activations for helper-only ranks (vllm_active and not sae_active).

        With m:1 fan-in, helper DP root ranks (vllm_dp_rank > 0, vllm_tp_rank == 0)
        also send their raw batch to the cluster SAE root via Gloo P2P.
        """
        import math

        vllm_dp_size = get_vllm_dp_size() if dist.is_initialized() else 1
        sae_dp_size = get_sae_dp_size() if dist.is_initialized() else 1
        is_dp_root = is_vllm_dp_root() if dist.is_initialized() else False
        vllm_dp_rank = get_vllm_dp_rank() if dist.is_initialized() else 0
        batch_size = self.cfg.store_batch_size_prompts
        ctx_size = self.activations_store.training_context_size

        # n = vLLM replicas per cluster; each cluster feeds one SAE replica.
        n = vllm_dp_size // sae_dp_size if sae_dp_size > 0 else vllm_dp_size

        # Approximate number of raw batches this helper should produce.
        if self.uses_vllm_dp_fan_in:
            tokens_per_batch = batch_size * ctx_size
            # Each helper produces total_tokens / (n * tokens_per_batch) batches,
            # because the cluster's SAE root will yield n batches per outer iteration.
            target_batches = math.ceil(
                self.cfg.total_training_tokens / (tokens_per_batch * n)
            )
        else:
            target_batches = None  # Legacy: use token count

        n_batches_done = 0
        n_training_samples = 0
        while True:
            if target_batches is not None and n_batches_done >= target_batches:
                break
            if target_batches is None and n_training_samples >= self.cfg.total_training_tokens:
                break

            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = (
                    f"[prefix-debug rank{rank}] helper batch start n={n_batches_done}\n"
                )
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)

            if self.uses_vllm_dp_fan_in:
                # Directly call get_raw_llm_batch to participate in intra-group
                # broadcasts. Do NOT go through mixing buffer.
                raw_acts, raw_tokens = self.activations_store.get_raw_llm_batch()
                n_batches_done += 1

                # Non-root helper DP root: send raw batch to cluster SAE root.
                # The cluster SAE root is the rank of the first vLLM DP replica
                # in this cluster (vllm_dp_rank % n == 0 → cluster root).
                cluster_first_vllm_dp = (vllm_dp_rank // n) * n
                is_cluster_sae_root_vllm = vllm_dp_rank == cluster_first_vllm_dp
                if is_dp_root and not is_cluster_sae_root_vllm:
                    p2p_group = get_vllm_dp_p2p_group()
                    if p2p_group is None:
                        raise RuntimeError(
                            "vLLM DP P2P group is not initialized; refusing to fall back to the default process group."
                        )
                    cluster_sae_root = get_sae_root_rank()
                    raw_acts = raw_acts.to("cpu").contiguous()
                    dist.send(raw_acts, dst=cluster_sae_root, group=p2p_group)
                    if raw_tokens is not None:
                        raw_tokens = raw_tokens.to("cpu").contiguous()
                        dist.send(raw_tokens, dst=cluster_sae_root, group=p2p_group)
            else:
                # Legacy: go through mixing buffer to stay in sync with
                # existing split-role broadcasts.
                batch = next(self.activations_store)
                n_training_samples += batch.shape[0]

            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = (
                    f"[prefix-debug rank{rank}] helper batch done n={n_batches_done}\n"
                )
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)

    def _run_producer_helper_loop_v2(self) -> None:
        """Producer-only loop for shard-routing mode (use_shard_routing=True).

        All producer TP ranks participate in get_raw_llm_batch() each step.
        Producer TP roots then participate in the same per-consumer NCCL P2P
        exchange phase as consumer ranks.
        Exits after total_producer_steps steps to stay in lockstep with consumers.
        """
        ctx_size = self.activations_store.training_context_size
        remaining_training_tokens = max(
            self.cfg.total_training_tokens - self._resume_training_samples(),
            0,
        )
        rows_per_consumer_step = self._v2_rows_per_consumer_step(ctx_size)
        buffer_size = self.cfg.n_batches_in_buffer * ctx_size
        total_producer_steps = max(
            self._mixing_buffer_source_steps_needed(
                target_samples=remaining_training_tokens,
                source_batch_size=rows_per_step,
                buffer_size=buffer_size,
                train_batch_size=self.cfg.train_batch_size_tokens,
                mix_fraction=self.cfg.activations_mixing_fraction,
            )
            for rows_per_step in rows_per_consumer_step
        )

        for _ in range(total_producer_steps):
            _local_slices, outgoing = self.activations_store._run_producer_phase2_v2()
            self.activations_store._run_nccl_p2p_exchange_v2(outgoing)

    def _v2_rows_per_consumer_step(self, ctx_size: int) -> list[int]:
        try:
            import sae_lens.distributed_v2 as v2_mod

            if v2_mod._initialized:
                from sae_lens.shard_routing import routes_for_consumer

                return [
                    sum(
                        route.row_end - route.row_start
                        for route in routes_for_consumer(v2_mod.get_routing_table(), c)
                    )
                    for c in range(v2_mod.get_sae_dp_size())
                ]
        except ImportError:
            pass

        total_rows = self.cfg.store_batch_size_prompts * ctx_size * self.vllm_dp_size
        rows_per_consumer = (total_rows + self.sae_dp_size - 1) // self.sae_dp_size
        return [rows_per_consumer]

    @staticmethod
    def _mixing_buffer_source_steps_needed(
        *,
        target_samples: int,
        source_batch_size: int,
        buffer_size: int,
        train_batch_size: int,
        mix_fraction: float,
    ) -> int:
        if target_samples <= 0:
            return 0
        if source_batch_size <= 0:
            raise ValueError("source_batch_size must be > 0")
        if buffer_size < train_batch_size:
            raise ValueError(
                "buffer_size must be greater than or equal to train_batch_size"
            )
        if not 0 <= mix_fraction <= 1:
            raise ValueError("mix_fraction must be in [0, 1]")

        source_steps = 0
        yielded_samples = 0
        storage_samples = 0
        while yielded_samples < target_samples:
            source_steps += 1
            storage_samples += source_batch_size
            if storage_samples < buffer_size:
                continue

            keep_for_mixing = int(buffer_size * mix_fraction)
            num_to_serve = storage_samples - keep_for_mixing
            num_serving_batches = max(1, num_to_serve // train_batch_size)
            serving_cutoff = num_serving_batches * train_batch_size
            yielded_samples += serving_cutoff
            storage_samples -= serving_cutoff

        return source_steps

    def _resume_checkpoint_path(self) -> Path | None:
        if self.cfg.resume_from_checkpoint is None:
            return None
        return Path(self.cfg.resume_from_checkpoint)

    def _resume_training_samples(self) -> int:
        checkpoint_path = self._resume_checkpoint_path()
        if checkpoint_path is None:
            return 0
        state_dict = torch.load(
            checkpoint_path / TRAINER_STATE_FILENAME,
            map_location="cpu",
        )
        return int(state_dict.get("n_training_samples", 0))

    def _load_producer_resume_state_if_needed(self) -> None:
        checkpoint_path = self._resume_checkpoint_path()
        if checkpoint_path is None:
            return
        self.activations_store.load_from_checkpoint(checkpoint_path)

    def save_final_sae(
        self,
        sae: TrainingSAE[Any],
        output_path: str,
        log_feature_sparsity: torch.Tensor | None = None,
    ):
        tp_group = getattr(sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_group = get_dp_group()
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod
            dp_group = v2_mod.get_sae_dp_group()
        dp_rank = (
            dist.get_rank(dp_group)
            if dp_group is not None and dist.get_world_size(dp_group) > 1
            else 0
        )

        base_output_path = Path(output_path)
        base_output_path.mkdir(exist_ok=True, parents=True)

        if self.cfg.sae_dp_mode == "fsdp":
            # FSDP state dict gather is a collective — all DP ranks must call it.
            # rank0_only=True means only dp_rank 0 receives a non-empty result.
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.sae, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                state_dict = self.sae.state_dict()
            # Non-rank-0 processes have an empty dict; nothing to save.
            if dp_rank != 0:
                return
            sae.process_state_dict_for_saving_inference(state_dict)
            weights_path = base_output_path / SAE_WEIGHTS_FILENAME
            cfg_path = base_output_path / SAE_CFG_FILENAME
            if tp_rank == 0:
                save_file(state_dict, weights_path)
                config = sae.cfg.get_inference_sae_cfg_dict()
                with open(cfg_path, "w") as f:
                    json.dump(config, f)
            if tp_group is not None:
                dist.barrier(group=tp_group)
        else:
            if dp_rank != 0:
                return
            weights_path, cfg_path = sae.save_inference_model(str(base_output_path))
            if tp_rank != 0:
                return

        sparsity_path = None
        if log_feature_sparsity is not None:
            sparsity_path = base_output_path / SPARSITY_FILENAME
            save_file({"sparsity": log_feature_sparsity}, sparsity_path)

        runner_config = self.cfg.to_dict()
        with open(base_output_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)

        if self.cfg.logger.log_to_wandb:
            self.cfg.logger.log(
                self,
                weights_path,
                cfg_path,
                sparsity_path=sparsity_path,
                wandb_aliases=["final_model"],
            )

    def _set_sae_metadata(self):
        assert self._base_sae is not None
        self._base_sae.cfg.metadata.dataset_path = self.cfg.dataset_path
        self._base_sae.cfg.metadata.hook_name = self.cfg.hook_name
        self._base_sae.cfg.metadata.model_name = self.cfg.model_name
        self._base_sae.cfg.metadata.model_class_name = self.cfg.model_class_name
        self._base_sae.cfg.metadata.hook_head_index = self.cfg.hook_head_index
        self._base_sae.cfg.metadata.context_size = self.cfg.context_size
        self._base_sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
        self._base_sae.cfg.metadata.model_from_pretrained_kwargs = (
            self.cfg.model_from_pretrained_kwargs
        )
        self._base_sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
        self._base_sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
        self._base_sae.cfg.metadata.sequence_separator_token = (
            self.cfg.sequence_separator_token
        )
        self._base_sae.cfg.metadata.disable_concat_sequences = (
            self.cfg.disable_concat_sequences
        )

    def _compile_sae_if_needed(self):
        if not self.cfg.compile_sae or self._base_sae is None:
            return

        backend = "aot_eager" if self.cfg.device == "mps" else "inductor"
        compiled_training_forward = torch.compile(
            self._base_sae.training_forward_pass,
            mode=self.cfg.sae_compilation_mode,
            backend=backend,
        )
        self._base_sae.training_forward_pass = (  # type: ignore[method-assign]
            compiled_training_forward
        )

    def _compile_if_needed(self):
        # Compile model. SAE compilation is done before FSDP/DDP wrapping so it
        # targets the raw module, not the distributed wrapper.
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

    def run_trainer_with_interruption_handling(
        self, trainer: SAETrainer[TrainingSAE[TrainingSAEConfig], TrainingSAEConfig]
    ):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                logger.warning("interrupted, saving progress")
                checkpoint_path = Path(self.cfg.checkpoint_path) / str(
                    trainer.n_training_samples
                )
                self.save_checkpoint(checkpoint_path)
                logger.info("done saving")
            raise

        return sae

    def run_multi_trainer_with_interruption_handling(
        self, trainer: MultiSAETrainer
    ) -> dict[str, TrainingSAE[Any]]:
        try:
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)
            return trainer.fit()
        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                logger.warning("interrupted, saving multi-SAE progress")
                trainer.save_checkpoint(checkpoint_name=str(trainer.n_training_samples))
                logger.info("done saving")
            raise

    # ------------------------------------------------------------------
    # Streaming mode (v1) — sae_dp=1 only
    # ------------------------------------------------------------------

    def _streaming_init(self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]) -> None:
        import sae_lens.distributed_streaming as ds
        from sae_lens.training.shared_activation_buffer import SharedActivationBuffer
        from sae_lens.util import str_to_dtype

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        ds.init_distributed_streaming(
            vllm_tp=self.vllm_tp_size,
            vllm_dp=self.vllm_dp_size,
            sae_tp=self.sae_tp_size,
            sae_dp=1,
        )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")

        # Buffer name: rank 0 generates, all ranks receive via CUDA tensor broadcast (NCCL)
        buffer_name = cfg.streaming_buffer_name
        if not buffer_name:
            name_buf = torch.zeros(32, dtype=torch.int32, device=self.device)
            if dist.get_rank() == 0:
                uid = uuid.uuid4().hex[:24]
                for i, c in enumerate(uid):
                    name_buf[i] = ord(c)
            dist.broadcast(name_buf, src=0)
            buffer_name = "sae_buf_" + "".join(
                chr(int(c)) for c in name_buf.tolist() if c > 0
            )

        self._streaming_buffer_name = buffer_name
        target_chunks = math.ceil(cfg.training_tokens / cfg.streaming_chunk_size_tokens)

        # Rank 0 creates the shared buffer; all other ranks attach after barrier
        if dist.get_rank() == 0:
            self._streaming_buffer = SharedActivationBuffer(
                name=buffer_name,
                num_chunks=cfg.streaming_num_chunks,
                chunk_size_tokens=cfg.streaming_chunk_size_tokens,
                d_model=cfg.sae.d_in,
                num_producers=self.vllm_dp_size,
                target_chunks=target_chunks,
                create=True,
                dtype=str_to_dtype(cfg.dtype),
            )
        dist.barrier()
        if dist.get_rank() != 0:
            self._streaming_buffer = SharedActivationBuffer(
                name=buffer_name,
                num_chunks=cfg.streaming_num_chunks,
                chunk_size_tokens=cfg.streaming_chunk_size_tokens,
                d_model=cfg.sae.d_in,
                num_producers=self.vllm_dp_size,
                target_chunks=target_chunks,
                create=False,
            )

        self.sae_active = ds.is_consumer()
        self.vllm_active = ds.is_producer()
        self.uses_split_roles = True
        self.uses_vllm_dp_fan_in = False
        self.uses_matched_dp = False

        # Pre-initialize vLLM parallel state on ALL ranks before any rank calls LLM().
        # vLLM creates 5+ process groups (WORLD, TP, DCP, PCP, PP, DP) even with tp=1;
        # these are collective, so all ranks must participate — even non-vLLM consumers.
        # SAELENS_VLLM_WORLD_RANKS tells vLLM's init_distributed_environment the subset
        # of world ranks that belong to vLLM, so its world-size check passes when the
        # process group was pre-initialized with fewer ranks than torch world_size.
        vllm_world_ranks = list(range(self.vllm_dp_size * self.vllm_tp_size))
        os.environ["SAELENS_VLLM_WORLD_RANKS"] = ",".join(str(r) for r in vllm_world_ranks)
        preinit_vllm_distributed(vllm_world_ranks, self.vllm_tp_size)

        self._sync_run_paths_across_ranks()
        self._init_streaming_logger()

        if ds.is_producer():
            self._streaming_init_producer(cfg)
        else:
            self._streaming_init_consumer(cfg)

    def _streaming_init_producer(
        self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
    ) -> None:
        import sae_lens.distributed_streaming as ds

        self.model = load_model(  # type: ignore[assignment]
            cfg.model_class_name,
            cfg.model_name,
            device=str(self.device),
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )
        self.activations_store = ActivationsStore.from_config(
            self.model,
            cfg,
            dataset_shard_index=ds.get_producer_idx(),
            dataset_shard_count=ds.get_vllm_dp_size(),
        )
        self.sae = None
        self._base_sae = None

    def _streaming_init_consumer(
        self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
    ) -> None:
        import sae_lens.distributed_streaming as ds

        self.model = None  # type: ignore[assignment]
        self.activations_store = None  # type: ignore[assignment]

        with temporary_seed(cfg.seed):
            sae = TrainingSAE.from_dict(
                TrainingSAEConfig.from_dict(
                    cfg.get_training_sae_cfg_dict(),
                ).to_dict()
            )
        sae.to(self.device)

        if self.sae_tp_size > 1:
            tp_group = ds.get_sae_tp_group()
            if tp_group is not None and hasattr(sae, "shard_weights"):
                sae.shard_weights(tp_group)

        self._base_sae = sae
        self.sae = sae

    def _init_streaming_logger(self) -> None:
        import logging
        self._logger = logging.getLogger("saelens.streaming.null")
        self._logger.addHandler(logging.NullHandler())
        self._logger.propagate = False

    def _run_streaming_producer_loop(self) -> None:
        import sae_lens.distributed_streaming as ds

        vllm_tp_group = ds.get_vllm_tp_group()
        is_tp_root = ds.is_vllm_tp_root()
        tp_root_world = ds.get_producer_tp_root()
        chunk_size = self.cfg.streaming_chunk_size_tokens
        total_tokens = self.cfg.training_tokens
        buf = self._streaming_buffer
        store = self.activations_store

        # Set up per-chunk timing and shm management logs (TP root only)
        timing_path: Path | None = None
        shm_log_path: Path | None = None
        t_ready = time.time()
        if is_tp_root and self.cfg.output_path is not None:
            out_dir = Path(self.cfg.output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.save_timing_every_n_steps > 0:
                timing_path = out_dir / "timing_history_vllm.jsonl"
                timing_path.write_text("")
            shm_log_path = out_dir / "shm_log_vllm.jsonl"
            shm_log_path.write_text("")

        def _shm_log(record: dict) -> None:
            if shm_log_path is None:
                return
            record["elapsed_s"] = time.time() - t_ready
            with open(shm_log_path, "a") as f:
                json.dump(record, f)
                f.write("\n")

        ctrl = torch.zeros(1, dtype=torch.int32, device=self.device)
        chunk_idx = -1
        seq_no = -1
        chunk_step = 0

        while True:
            # Outer Phase 1: quota check — root tries to allocate a chunk slot
            if is_tp_root:
                result = buf.allocate_write_chunk()
                ctrl[0] = 0 if result is None else 1
                if result is not None:
                    chunk_idx, seq_no = result
                    _shm_log({"event": "chunk_allocated", "chunk_idx": chunk_idx, "seq_no": seq_no})
            if vllm_tp_group is not None:
                dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
            if int(ctrl[0]) == 0:
                if is_tp_root:
                    _shm_log({"event": "quota_exhausted", "total_chunks": chunk_step})
                break  # global quota exhausted; all TP ranks exit together

            # Compute exact token limit for this chunk (root knows seq_no; non-root uses full chunk_size)
            if is_tp_root:
                max_this_chunk = max(1, min(chunk_size, total_tokens - seq_no * chunk_size))
            else:
                max_this_chunk = chunk_size  # non-root: ignored, root drives the internal loop

            # All TP ranks participate in inference (required by vLLM external_launcher semantics)
            t_infer_start = time.perf_counter()
            acts_cpu, valid_tokens = store.get_streaming_activations(max_this_chunk)
            t_infer_end = time.perf_counter()

            # Outer Phase 2: EOF check — did dataset run dry?
            if is_tp_root:
                ctrl[0] = 0 if acts_cpu is None else 1
            if vllm_tp_group is not None:
                dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
            if int(ctrl[0]) == 0:
                if is_tp_root:
                    buf.abort_write_chunk(chunk_idx)  # WRITING → FREE
                    _shm_log({"event": "dataset_exhausted", "chunk_idx": chunk_idx, "total_chunks": chunk_step})
                break  # dataset exhausted; all TP ranks exit together

            # Write to buffer (TP root only)
            if is_tp_root:
                t_write_start = time.perf_counter()
                buf.write_chunk(
                    chunk_idx,
                    acts_cpu,
                    valid_tokens,
                    producer_id=ds.get_producer_idx(),
                )
                buf.mark_ready(chunk_idx)
                t_write_end = time.perf_counter()

                chunk_step += 1
                inference_time_s = t_infer_end - t_infer_start
                write_time_s = t_write_end - t_write_start
                _shm_log({
                    "event": "chunk_written",
                    "chunk_idx": chunk_idx,
                    "seq_no": seq_no,
                    "step": chunk_step,
                    "valid_tokens": valid_tokens,
                    "total_tokens": seq_no * chunk_size + valid_tokens,
                    "inference_time_s": inference_time_s,
                    "write_time_s": write_time_s,
                    "buffer_state": buf.queue_counts(),
                })

                if timing_path is not None and chunk_step % self.cfg.save_timing_every_n_steps == 0:
                    record = {
                        "step": chunk_step,
                        "elapsed_s": time.time() - t_ready,
                        "inference_time_s": inference_time_s,
                        "write_time_s": write_time_s,
                        "chunk_time_s": inference_time_s + write_time_s,
                        "valid_tokens": valid_tokens,
                        "total_tokens": seq_no * chunk_size + valid_tokens,
                    }
                    with open(timing_path, "a") as f:
                        json.dump(record, f)
                        f.write("\n")

        if is_tp_root:
            buf.signal_done()
            _shm_log({"event": "producer_done", "total_chunks": chunk_step})
        buf.close()

    def _run_streaming_consumer_loop(self) -> TrainingSAE[Any]:
        import sae_lens.distributed_streaming as ds
        from sae_lens.training.streaming_activation_provider import StreamingActivationProvider

        sae_tp_group = ds.get_sae_tp_group()
        sae_tp_size = ds.get_sae_tp_size()

        # shm log and buffer monitor written by TP root only
        shm_log_path: Path | None = None
        buffer_monitor_path: Path | None = None
        if ds.is_sae_tp_root() and self.cfg.output_path is not None:
            out_dir = Path(self.cfg.output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            shm_log_path = out_dir / "shm_log_sae.jsonl"
            buffer_monitor_path = out_dir / "buffer_monitor.jsonl"

        from sae_lens.util import str_to_dtype
        provider = StreamingActivationProvider(
            buffer=self._streaming_buffer,
            train_batch_size_tokens=self.cfg.train_batch_size_tokens,
            prefetch_chunks=self.cfg.streaming_prefetch_chunks,
            device=self.device,
            sae_tp_group=sae_tp_group if sae_tp_size > 1 else None,
            sae_tp_rank=ds.get_sae_tp_rank(),
            sae_tp_root_global_rank=ds.get_consumer_tp_root(),
            d_model=self.cfg.sae.d_in,
            dtype=str_to_dtype(self.cfg.dtype),
            shm_log_path=shm_log_path,
            shuffle=self.cfg.streaming_shuffle,
            random_chunks=self.cfg.streaming_random_chunks,
            buffer_monitor_path=buffer_monitor_path,
        )

        trainer = SAETrainer(
            sae=self.sae,
            base_sae=self._base_sae,
            data_provider=provider,
            evaluator=None,
            save_checkpoint_fn=self._streaming_save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=None,
            token_count_weighted_dp=False,
        )

        try:
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)
            sae = trainer.fit()
        except StopIteration:
            # Data provider exhausted before training_tokens reached (e.g. dataset
            # shorter than configured).  Treat as normal end-of-data termination.
            sae = trainer.sae
        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                checkpoint_path = Path(self.cfg.checkpoint_path) / str(
                    trainer.n_training_samples
                )
                self._streaming_save_checkpoint(checkpoint_path)
            raise

        self._streaming_buffer.close()

        if self.cfg.output_path is not None:
            self._streaming_save_final(sae, self.cfg.output_path, trainer.log_feature_sparsity)

        return sae

    def _streaming_save_checkpoint(self, checkpoint_path: Path | None) -> None:
        """Called by ALL SAE TP ranks — save_inference_model is collective when sae_tp > 1."""
        if checkpoint_path is None:
            return
        import sae_lens.distributed_streaming as ds
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        self._base_sae.save_inference_model(str(checkpoint_path))  # type: ignore[union-attr]
        if ds.is_sae_tp_root():
            runner_config = self.cfg.to_dict()
            with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
                json.dump(runner_config, f)

    def _streaming_save_final(
        self,
        sae: TrainingSAE[Any],
        output_path: str,
        log_feature_sparsity: torch.Tensor | None,
    ) -> None:
        """Called by ALL SAE TP ranks — save_inference_model is collective when sae_tp > 1."""
        import sae_lens.distributed_streaming as ds
        base = Path(output_path)
        base.mkdir(exist_ok=True, parents=True)
        sae.save_inference_model(str(base))
        if ds.is_sae_tp_root():
            if log_feature_sparsity is not None:
                save_file({"sparsity": log_feature_sparsity}, base / SPARSITY_FILENAME)
            runner_config = self.cfg.to_dict()
            with open(base / RUNNER_CFG_FILENAME, "w") as f:
                json.dump(runner_config, f)

    def save_checkpoint(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        if checkpoint_path is None:
            return
        sae = self._base_sae
        tp_group = getattr(sae, "_tp_group", None) if sae is not None else None
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_group = get_dp_group()
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod
            dp_group = v2_mod.get_sae_dp_group()
        dp_rank = (
            dist.get_rank(dp_group)
            if dp_group is not None and dist.get_world_size(dp_group) > 1
            else 0
        )
        if tp_rank != 0 or dp_rank != 0:
            return

        self.activations_store.save_to_checkpoint(checkpoint_path)

        runner_config = self.cfg.to_dict()
        with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)


def _parse_cfg_args(
    args: Sequence[str],
) -> LanguageModelSAERunnerConfig[TrainingSAEConfig]:
    """
    Parse command line arguments into a LanguageModelSAERunnerConfig.

    This function first parses the architecture argument to determine which
    concrete SAE config class to use, then parses the full configuration
    with that concrete type.
    """
    if len(args) == 0:
        args = ["--help"]

    # First, parse only the architecture to determine which concrete class to use
    architecture_parser = ArgumentParser(
        description="Parse architecture to determine SAE config class",
        exit_on_error=False,
        add_help=False,  # Don't add help to avoid conflicts
    )
    architecture_parser.add_argument(
        "--architecture",
        type=str,
        choices=["standard", "gated", "jumprelu", "topk", "batchtopk"],
        default="standard",
        help="SAE architecture to use",
    )

    # Parse known args to extract architecture, ignore unknown args for now
    arch_args, remaining_args = architecture_parser.parse_known_args(args)
    architecture = arch_args.architecture

    # Remove architecture from remaining args if it exists
    filtered_args = []
    skip_next = False
    for arg in remaining_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--architecture":
            skip_next = True  # Skip the next argument (the architecture value)
            continue
        filtered_args.append(arg)

    # Create a custom wrapper class that simple_parsing can handle
    def create_config_class(
        sae_config_type: type[TrainingSAEConfig],
    ) -> type[LanguageModelSAERunnerConfig[TrainingSAEConfig]]:
        """Create a concrete config class for the given SAE config type."""

        # Create the base config without the sae field
        from dataclasses import field as dataclass_field
        from dataclasses import fields, make_dataclass

        # Get all fields from LanguageModelSAERunnerConfig except the generic sae field
        base_fields = []
        for field_obj in fields(LanguageModelSAERunnerConfig):
            if field_obj.name != "sae":
                base_fields.append((field_obj.name, field_obj.type, field_obj))

        # Add the concrete sae field
        base_fields.append(
            (
                "sae",
                sae_config_type,
                dataclass_field(
                    default_factory=lambda: sae_config_type(d_in=512, d_sae=1024)
                ),
            )
        )

        # Create the concrete class
        return make_dataclass(
            f"{sae_config_type.__name__}RunnerConfig",
            base_fields,
            bases=(LanguageModelSAERunnerConfig,),
        )

    # Map architecture to concrete config class
    sae_config_map: dict[str, type[TrainingSAEConfig]] = {
        name: cfg for name, (_, cfg) in SAE_TRAINING_CLASS_REGISTRY.items()
    }

    sae_config_type = sae_config_map[architecture]
    concrete_config_class = create_config_class(sae_config_type)

    # Now parse the full configuration with the concrete type
    parser = ArgumentParser(exit_on_error=False)
    parser.add_arguments(concrete_config_class, dest="cfg")

    # Parse the filtered arguments (without --architecture)
    parsed_args = parser.parse_args(filtered_args)

    # Return the parsed configuration
    return parsed_args.cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    LanguageModelSAETrainingRunner(cfg=cfg).run()


if __name__ == "__main__":
    _run_cli(args=sys.argv[1:])


@deprecated("Use LanguageModelSAETrainingRunner instead")
class SAETrainingRunner(LanguageModelSAETrainingRunner):
    pass
