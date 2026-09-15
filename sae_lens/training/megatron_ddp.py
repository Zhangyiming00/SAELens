"""Static, per-hook Megatron DDP construction (no global parallel state)."""

import os

import torch
from torch.nn.parallel import DistributedDataParallel as TorchDDP


def is_megatron_ddp(module):
    # Keep importing SAELens possible without the optional Megatron dependency.
    return getattr(module, "_sae_megatron_ddp", False)


def supports_early_grad_sync(module, runtime):
    """Whether native DP bucket reductions may cross subsequent TP work.

    TP1 has only one communicating group. For TP+DP, use NCCL's explicit
    multi-communicator ordering contract (2.26+), enabled by the launcher
    BEFORE any NCCL initialization. Do not set the environment here: callers
    may already own initialized communicators. Otherwise retain late sync.
    See NCCL's "Using multiple NCCL communicators concurrently" documentation.
    """
    if not is_megatron_ddp(module):
        return False
    context = runtime.require_local()
    if context.tp_group.size() == 1 or context.dp_group.size() == 1:
        return True
    return (
        os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") == "1"
        and torch.cuda.nccl.version() >= (2, 26, 0)
    )


def wrap_runtime_sae(model, runtime, *, bucket_cap_mb=None):
    runtime.validate_model(model)
    context = runtime.require_local()
    if next(model.parameters()).device.type == "cpu":
        # Megatron's buffers require CUDA. CPU is a development/test fallback.
        return (
            TorchDDP(model, process_group=context.dp_group)
            if context.dp_group.size() > 1
            else model
        )

    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError(
            "Static Megatron DDP requires FP32 parameters; use autocast for BF16 compute"
        )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=model.cfg.d_in,
        num_attention_heads=context.tp_group.size(),
        tensor_model_parallel_size=context.tp_group.size(),
        calculate_per_token_loss=True,
        gradient_accumulation_fusion=False,
    )
    ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            # Megatron disables buckets when this flag is false. The final
            # known microbatch uses native bucket-ready reduction; all earlier
            # microbatches use no_sync. Gradient access/updates wait at window end.
            overlap_grad_reduce=True,
            bucket_size=None
            if bucket_cap_mb is None
            else max(1, int(bucket_cap_mb * 1024**2 / 4)),
            average_in_collective=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,  # GradScaler owns overflow handling.
        ),
        module=model,
        pg_collection=context.groups,
    )
    ddp._sae_megatron_ddp = True
    ddp.broadcast_params()
    return ddp
