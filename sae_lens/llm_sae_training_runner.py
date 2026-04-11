import json
import os
import signal
import sys
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

        if vllm_dp_size > 1:
            if self.cfg.resume_from_checkpoint is not None:
                raise ValueError(
                    "resume_from_checkpoint is not supported with vllm_dp_size > 1"
                )
            if self.cfg.use_cached_activations:
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

        # Wrap with FSDP after TP sharding when sae_dp_mode='fsdp'.
        # base_sae always points to the raw module; wrapped_sae may be an FSDP wrapper.
        self._base_sae = self.sae
        if self.sae is not None and self.cfg.sae_dp_mode == "fsdp":
            if not dist.is_initialized():
                raise ValueError(
                    "sae_dp_mode='fsdp' requires an initialized distributed process group."
                )
            if sae_tp_size > 1:
                raise ValueError(
                    "sae_dp_mode='fsdp' with sae_tp_size > 1 is not supported. "
                    "Use sae_tp_size=1 with FSDP."
                )
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import ShardingStrategy
            sae_dp_group = get_dp_group()
            if self.use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                sae_dp_group = v2_mod.get_sae_dp_group()
            if sae_dp_group is None or dist.get_world_size(sae_dp_group) <= 1:
                raise ValueError(
                    "sae_dp_mode='fsdp' requires sae_dp_size > 1."
                )
            self.sae = FSDP(
                self.sae,
                process_group=sae_dp_group,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True,
            )

    def run(self):
        """
        Run the training of the SAE.
        """
        if self.use_shard_routing and self.vllm_active and not self.sae_active:
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
            self._base_sae.load_weights_from_checkpoint(self.cfg.resume_from_checkpoint)
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
        import math

        P = self.vllm_dp_size
        batch_size = self.cfg.store_batch_size_prompts
        ctx_size = self.activations_store.training_context_size
        total_producer_steps = math.ceil(
            self.cfg.total_training_tokens / (batch_size * ctx_size * P)
        )

        for _ in range(total_producer_steps):
            _local_slices, outgoing = self.activations_store._run_producer_phase2_v2()
            self.activations_store._run_nccl_p2p_exchange_v2(outgoing)

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
        assert self.sae is not None
        self.sae.cfg.metadata.dataset_path = self.cfg.dataset_path
        self.sae.cfg.metadata.hook_name = self.cfg.hook_name
        self.sae.cfg.metadata.model_name = self.cfg.model_name
        self.sae.cfg.metadata.model_class_name = self.cfg.model_class_name
        self.sae.cfg.metadata.hook_head_index = self.cfg.hook_head_index
        self.sae.cfg.metadata.context_size = self.cfg.context_size
        self.sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
        self.sae.cfg.metadata.model_from_pretrained_kwargs = (
            self.cfg.model_from_pretrained_kwargs
        )
        self.sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
        self.sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
        self.sae.cfg.metadata.sequence_separator_token = (
            self.cfg.sequence_separator_token
        )
        self.sae.cfg.metadata.disable_concat_sequences = (
            self.cfg.disable_concat_sequences
        )

    def _compile_if_needed(self):
        # Compile model and SAE
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

        if self.cfg.compile_sae and self.sae is not None:
            backend = "aot_eager" if self.cfg.device == "mps" else "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
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

    def save_checkpoint(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        if checkpoint_path is None:
            return
        sae = self.sae
        tp_group = getattr(sae, "_tp_group", None) if sae is not None else None
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_group = get_dp_group()
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
