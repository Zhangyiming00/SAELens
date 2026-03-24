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
    SPARSITY_FILENAME,
)
from sae_lens.distributed import (
    get_dp_group,
    get_tp_group,
    get_vllm_world_ranks,
    init_distributed,
    is_sae_active,
    is_vllm_active,
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
        dp_size: int = 1,
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
        self.dp_size = dp_size
        inferred_cfg_vllm_tp_size = int(
            self.cfg.model_from_pretrained_kwargs.get("tensor_parallel_size", 1)
        )
        self.shared_tp_size = shared_tp_size
        if (
            self.shared_tp_size is None
            and sae_tp_size is None
            and vllm_tp_size is None
            and tp_size > 1
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
        if self.vllm_tp_size > 1 or self.sae_tp_size > 1 or dp_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            if self.shared_tp_size is not None:
                init_distributed(
                    shared_tp_size=self.shared_tp_size,
                    dp_size=dp_size,
                )
            else:
                init_distributed(
                    sae_tp_size=self.sae_tp_size,
                    vllm_tp_size=self.vllm_tp_size,
                    dp_size=dp_size,
                )

        self.sae_active = is_sae_active() if dist.is_initialized() else True
        self.vllm_active = is_vllm_active() if dist.is_initialized() else True
        # uses_split_roles is a system-wide property: some ranks are vLLM-only
        # or SAE-only.  It is True whenever the two TP sizes differ.
        self.uses_split_roles = self.vllm_tp_size != self.sae_tp_size

        if dist.is_initialized():
            os.environ["SAELENS_VLLM_WORLD_RANKS"] = ",".join(
                str(rank) for rank in get_vllm_world_ranks()
            )

        # When sae_tp > vllm_tp, some ranks will never create an LLM() but
        # PyTorch requires every rank to participate in dist.new_group() calls.
        # Pre-initialize vLLM parallel state on ALL ranks so LLM() construction
        # on vLLM-active ranks skips group creation (finding it already done).
        if dist.is_initialized() and self.sae_tp_size > self.vllm_tp_size:
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

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
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
            tp_group = get_tp_group()
            if tp_group is not None and hasattr(self.sae, "shard_weights"):
                self.sae.shard_weights(tp_group)

    def run(self):
        """
        Run the training of the SAE.
        """
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
        trainer = SAETrainer(
            sae=self.sae,
            data_provider=self.activations_store,
            evaluator=evaluator,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=(
                sae_dp_group
                if sae_dp_group is not None and dist.get_world_size(sae_dp_group) > 1
                else None
            ),
        )

        if self.cfg.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.cfg.resume_from_checkpoint}")
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            self.sae.load_weights_from_checkpoint(self.cfg.resume_from_checkpoint)
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
        n_training_samples = 0
        while n_training_samples < self.cfg.total_training_tokens:
            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = (
                    f"[prefix-debug rank{rank}] helper next() start samples={n_training_samples}\n"
                )
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)
            batch = next(self.activations_store)
            n_training_samples += batch.shape[0]
            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = (
                    f"[prefix-debug rank{rank}] helper next() done samples={n_training_samples}\n"
                )
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)

    def save_final_sae(
        self,
        sae: TrainingSAE[Any],
        output_path: str,
        log_feature_sparsity: torch.Tensor | None = None,
    ):
        tp_group = getattr(sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_group = get_dp_group()
        dp_rank = (
            dist.get_rank(dp_group)
            if dp_group is not None and dist.get_world_size(dp_group) > 1
            else 0
        )
        if dp_rank != 0:
            return

        base_output_path = Path(output_path)
        base_output_path.mkdir(exist_ok=True, parents=True)

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
