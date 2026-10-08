"""Adapt the shared SAE runner CLI to the online elastic TP runtime."""

from __future__ import annotations

import argparse

from sae_lens.training.elastic_tp_config import ElasticTPConfig
from sae_lens.v5_cli import execution_config_kwargs, option_specs, supplied


def add_elastic_tp_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Online elastic tensor parallel training")
    group.add_argument(
        "--elastic-tp",
        action="store_true",
        help=(
            "Launch online watermark-controlled TP training with plain python. "
            "Owns the GPU worker pool; one hook, FP32 SAE, DP=PP=1."
        ),
    )
    group.add_argument(
        "--elastic-tp-size",
        "--elastic-tp-pool-size",
        dest="elastic_tp_pool_size",
        type=int,
        default=None,
        help="GPU pool size; also enables elastic TP. --elastic-tp alone uses 4. Initial SAE TP uses --sae-tp-size, otherwise --elastic-tp-min-size.",
    )
    group.add_argument(
        "--elastic-tp-min-size",
        type=int,
        default=1,
        help="Minimum SAE TP during training; must be smaller than the GPU pool so a vLLM producer can resume. Also the default initial SAE TP.",
    )
    group.add_argument("--elastic-tp-low-watermark", type=float, default=0.05)
    group.add_argument("--elastic-tp-high-watermark", type=float, default=0.85)
    group.add_argument("--elastic-tp-cooldown", type=float, default=12.0)
    group.add_argument("--elastic-tp-poll-interval", type=float, default=0.5)
    group.add_argument("--elastic-tp-watermark-samples", type=int, default=3)
    group.add_argument("--elastic-tp-activation-scale", type=float)
    group.add_argument("--elastic-tp-startup-timeout", type=float, default=300.0)
    group.add_argument("--elastic-tp-pause-timeout", type=float, default=120.0)
    group.add_argument("--elastic-tp-resume-timeout", type=float, default=300.0)
    group.add_argument(
        "--elastic-tp-vllm-residency", choices=("release", "resident"), default="release",
        help="release (default): close vLLM and release GPU memory before SAE joins; reload on return. resident: retain warm weights while paused for faster switching.",
    )
    group.add_argument(
        "--elastic-tp-release-tolerance-mib", type=int, default=64,
        help="Maximum residual live PyTorch memory above the pre-model baseline at a release handoff. Context/NCCL allocations are separate.",
    )
    group.add_argument("--elastic-tp-audit-inputs", action="store_true")
    group.add_argument("--elastic-tp-validate-activations", action="store_true")


def validate_elastic_tp_arguments(
    args: argparse.Namespace, argv: list[str], parser: argparse.ArgumentParser
) -> None:
    """Reject unsupported combinations before loading models or spawning children."""
    if args.elastic_streaming:
        raise ValueError(
            "--elastic-tp and --elastic-streaming (elastic DP) are mutually exclusive"
        )
    if (
        args.sae_dp_size != 1
        or args.sae_pp_size != 1
        or args.sae_dp_mode != "ddp"
        or args.ddp_zero_optimizer
    ):
        raise ValueError("--elastic-tp requires SAE DP=PP=1 without FSDP/ZeRO")
    if args.vllm_tp_size not in (None, 1) or args.tp_size != 1:
        raise ValueError(
            "--elastic-tp requires vLLM TP=1; use --sae-tp-size for initial SAE TP"
        )
    if args.vllm_dp_size != 1:
        raise ValueError(
            "--elastic-tp derives active producer replicas from --elastic-tp-size and --sae-tp-size; do not set --vllm-dp-size"
        )
    if args.dtype != "float32" or args.autocast or args.autocast_lm:
        raise ValueError(
            "--elastic-tp requires FP32 SAE without autocast; --vllm-dtype controls the LLM"
        )
    if type(args.gradient_accumulation_steps) is not int or args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be a positive integer")
    if args.use_cached_activations or not args.is_dataset_tokenized:
        raise ValueError(
            "--elastic-tp requires a local tokenized dataset for online vLLM capture"
        )
    if (
        args.resume_from_checkpoint
        or args.control_state_path
        or args.quiesce_checkpoint_path
    ):
        raise ValueError(
            "--elastic-tp does not support checkpoint resume or the DP topology supervisor"
        )
    if args.n_checkpoints or args.save_final_checkpoint:
        raise ValueError(
            "--elastic-tp exports the final SAE; online training checkpoints are not supported"
        )
    if (
        args.train_batch_size_tokens < 1
        or args.training_tokens < 1
        or args.training_tokens % args.train_batch_size_tokens
    ):
        raise ValueError(
            "--elastic-tp requires training-tokens to be a positive multiple of train-batch-size-tokens"
        )
    # The shared CLI has a multi-hook default. Elastic TP preserves its existing
    # single-hook online contract and uses --hook-name unless hooks are explicit.
    if any(supplied(argv, flag) for flag in ("--hook-names", "--hooks")):
        hooks = [hook.strip() for hook in args.hook_names.split(",") if hook.strip()]
        if len(hooks) != 1:
            raise ValueError("--elastic-tp currently supports exactly one online hook")
        if (
            any(supplied(argv, flag) for flag in ("--hook-name", "--hook"))
            and args.hook_name != hooks[0]
        ):
            raise ValueError("Conflicting --hook-name and --hook-names")
        args.hook_name = hooks[0]
    args.hook_names = None
    # Validate numerical controls now, even when d_in will be inferred later.
    config_from_runner_args(args, d_in=1)
    # Shared parser acceptance must not silently turn unsupported options into
    # no-ops. Defaults of other modes are harmless; reject explicit requests.
    shared = {
        "model_name",
        "dataset_path",
        "context_size",
        "max_model_len",
        "max_num_batched_tokens",
        "hook_name",
        "hook_names",
        "d_sae",
        "k",
        "lr",
        "training_tokens",
        "train_batch_size_tokens",
        "gradient_accumulation_steps",
        "dead_feature_window",
        "output_path",
        "vllm_tp_size",
        "sae_tp_size",
        "vllm_dp_size",
        "sae_dp_size",
        "sae_pp_size",
        "sae_dp_mode",
        "ddp",
        "fsdp",
        "sae_ddp_size",
        "sae_fsdp_size",
        "ddp_zero_optimizer",
        "tp_size",
        "elastic_streaming",
        "streaming_mode",
        "streaming_num_chunks",
        "streaming_prefetch_chunks",
        "streaming_chunk_size_tokens",
        "routing_transport",
        "store_batch_size_prompts",
        "dtype",
        "vllm_dtype",
        "activation_dtype",
        "activation_conversion",
        "autocast",
        "autocast_lm",
        "is_dataset_tokenized",
        "seed",
        "performance_only",
        "no_save_final",
        "no_save_final_sae",
        "save_final_checkpoint",
        "n_checkpoints",
        "use_cached_activations",
        "resume_from_checkpoint",
        "control_state_path",
        "quiesce_checkpoint_path",
        "use_sparse_activations",
        "sae_topk_backend",
        "sae_topk_keys",
        "sae_topk_protocol",
        "sae_sparse_decoder",
        "rescale_acts_by_decoder_norm",
        "vllm_text_only",
        "sae_tp_overlap",
        "sae_tp_overlap_max_live_hooks",
    }
    shared.update(spec[1] for spec in option_specs())
    shared.update(
        f"sae_{branch}_{part}"
        for branch in ("main", "aux")
        for part in (
            "representation",
            "compute",
            "forward",
            "dvalues",
            "dweight",
            "backward",
        )
    )
    explicit = {}
    for token in argv:
        flag = token.partition("=")[0]
        action = parser._option_string_actions.get(flag)
        if action is not None:
            explicit[action.dest] = flag
    unsupported = [
        flag
        for dest, flag in explicit.items()
        if dest not in shared and not dest.startswith("elastic_tp")
    ]
    if unsupported:
        raise ValueError(
            f"These runner options are not supported in elastic TP: {', '.join(unsupported)}"
        )
    if (
        "streaming_chunk_size_tokens" in explicit
        and args.streaming_chunk_size_tokens != args.train_batch_size_tokens
    ):
        raise ValueError(
            "Elastic TP requires streaming-chunk-size-tokens == train-batch-size-tokens"
        )
    if "routing_transport" in explicit and args.routing_transport != "shm_async":
        raise ValueError(
            "Elastic TP input transport currently requires --routing-transport shm_async"
        )


def config_from_runner_args(args: argparse.Namespace, *, d_in: int) -> ElasticTPConfig:
    sae_config = execution_config_kwargs(args)
    sae_config.update(
        use_sparse_activations=args.use_sparse_activations,
        topk_backend=args.sae_topk_backend,
        topk_key_backend=args.sae_topk_keys,
        topk_candidate_protocol=args.sae_topk_protocol,
        sparse_decoder_backend=args.sae_sparse_decoder,
        rescale_acts_by_decoder_norm=args.rescale_acts_by_decoder_norm,
    )
    return ElasticTPConfig(
        model=args.model_name,
        dataset=args.dataset_path,
        hook=args.hook_name,
        output=args.output_path,
        d_in=d_in,
        d_sae=args.d_sae,
        k=args.k,
        batch_size=args.train_batch_size_tokens,
        steps=args.training_tokens // args.train_batch_size_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        context=args.context_size,
        prompts=args.store_batch_size_prompts,
        dead=args.dead_feature_window,
        seed=args.seed,
        lr=args.lr,
        pool_size=args.elastic_tp_pool_size,
        initial_tp=(
            args.sae_tp_size
            if args.sae_tp_size is not None
            else args.elastic_tp_min_size
        ),
        min_tp=args.elastic_tp_min_size,
        chunks=args.streaming_num_chunks,
        cache_batches=args.streaming_prefetch_chunks,
        low=args.elastic_tp_low_watermark,
        high=args.elastic_tp_high_watermark,
        cooldown=args.elastic_tp_cooldown,
        poll_interval=args.elastic_tp_poll_interval,
        watermark_samples=args.elastic_tp_watermark_samples,
        startup_timeout=args.elastic_tp_startup_timeout,
        pause_timeout=args.elastic_tp_pause_timeout,
        resume_timeout=args.elastic_tp_resume_timeout,
        vllm_residency=args.elastic_tp_vllm_residency,
        release_tolerance_mib=args.elastic_tp_release_tolerance_mib,
        activation_dtype=args.activation_dtype,
        vllm_dtype=args.vllm_dtype,
        activation_conversion=args.activation_conversion,
        activation_scale=args.elastic_tp_activation_scale,
        profile_only=args.no_save_final_sae,
        audit_inputs=args.elastic_tp_audit_inputs,
        validate_activations=args.elastic_tp_validate_activations,
        tp_overlap=args.sae_tp_overlap or "off",
        tp_overlap_max_live_hooks=args.sae_tp_overlap_max_live_hooks,
        sae_config=sae_config,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        vllm_text_only=args.vllm_text_only,
    )
