"""Memory estimates for SAELens autoconfiguration.

The VLLM component is intentionally data-driven: run a VLLM probe once, save the
JSON profile, then use that profile as the source of truth for VLLM memory.
The SAE and buffer components are estimated from task/config dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal


MB = 1024**2

DTypeName = Literal["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"]
ModeName = Literal["normal", "streaming", "offline_cache", "topology_switch"]
SaeDpMode = Literal["manual", "ddp", "fsdp"]
ActStoreDevice = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class TaskSpec:
    model_name: str
    d_in: int
    d_sae: int
    hook_names: list[str]
    dtype: DTypeName = "float32"
    train_batch_size_tokens: int = 2048
    context_size: int = 2048
    n_batches_in_buffer: int | None = None
    act_store_device: ActStoreDevice = "cuda"
    mix_fraction: float = 0.5
    store_batch_size_prompts: int = 4


@dataclass(frozen=True)
class CandidateConfig:
    mode: ModeName = "normal"
    vllm_tp_size: int = 1
    vllm_dp_size: int = 1
    vllm_dtype: DTypeName | None = None
    vllm_context_size: int | None = None
    vllm_max_model_len: int | None = None
    vllm_gpu_memory_utilization: float | None = None
    sae_tp_size: int = 1
    sae_dp_size: int = 1
    sae_pp_size: int = 1
    sae_dp_mode: SaeDpMode = "manual"
    disjoint: bool = False
    streaming_chunk_size_tokens: int = 4096
    streaming_num_chunks: int = 32
    streaming_prefetch_chunks: int = 2
    streaming_mix_chunks: int = 8
    streaming_mix_fraction: float = 0.5
    checkpoint_storage: Literal["memory", "disk"] = "disk"
    vllm_profile_stage: str = "after_hooked_vllm_load"


@dataclass(frozen=True)
class RankVllmProfile:
    rank: int
    stage: str
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    driver_used_mb: float
    driver_total_mb: float


@dataclass(frozen=True)
class VllmMemoryProfile:
    model_name: str
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    context_size: int
    max_model_len: int
    rank_profiles: list[RankVllmProfile]
    peak_allocated_mb: float
    peak_reserved_mb: float
    peak_driver_used_mb: float


@dataclass(frozen=True)
class RankMemoryEstimate:
    rank: int
    role: str
    local_num_hooks: int
    predicted_allocated_mb: float
    predicted_reserved_mb: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateMemoryEstimate:
    task: TaskSpec
    config: CandidateConfig
    ranks: list[RankMemoryEstimate]
    shared_memory_bytes: int = 0

    @property
    def max_reserved_mb(self) -> float:
        if not self.ranks:
            return 0.0
        return max(rank.predicted_reserved_mb for rank in self.ranks)


def dtype_nbytes(dtype: str) -> int:
    normalized = canonical_dtype_name(dtype)
    if normalized in {"float32", "fp32"}:
        return 4
    if normalized in {"bfloat16", "bf16", "float16", "fp16"}:
        return 2
    raise ValueError(f"Unsupported dtype for memory model: {dtype!r}")


def canonical_dtype_name(dtype: str) -> str:
    normalized = dtype.lower()
    if normalized == "fp32":
        return "float32"
    if normalized == "bf16":
        return "bfloat16"
    if normalized == "fp16":
        return "float16"
    return normalized


def validate_vllm_profile_compatibility(
    *,
    task: TaskSpec,
    config: CandidateConfig,
    profile: VllmMemoryProfile,
) -> None:
    mismatches: list[str] = []
    if profile.model_name != task.model_name:
        mismatches.append(
            f"model_name profile={profile.model_name!r} candidate={task.model_name!r}"
        )
    if profile.tensor_parallel_size != config.vllm_tp_size:
        mismatches.append(
            "tensor_parallel_size "
            f"profile={profile.tensor_parallel_size} candidate={config.vllm_tp_size}"
        )
    if config.vllm_dtype is not None and (
        canonical_dtype_name(profile.dtype) != canonical_dtype_name(config.vllm_dtype)
    ):
        mismatches.append(
            f"dtype profile={profile.dtype!r} candidate={config.vllm_dtype!r}"
        )
    if (
        config.vllm_context_size is not None
        and profile.context_size != config.vllm_context_size
    ):
        mismatches.append(
            f"context_size profile={profile.context_size} "
            f"candidate={config.vllm_context_size}"
        )
    if (
        config.vllm_max_model_len is not None
        and profile.max_model_len != config.vllm_max_model_len
    ):
        mismatches.append(
            f"max_model_len profile={profile.max_model_len} "
            f"candidate={config.vllm_max_model_len}"
        )
    if config.vllm_gpu_memory_utilization is not None and not math.isclose(
        profile.gpu_memory_utilization,
        config.vllm_gpu_memory_utilization,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        mismatches.append(
            "gpu_memory_utilization "
            f"profile={profile.gpu_memory_utilization} "
            f"candidate={config.vllm_gpu_memory_utilization}"
        )
    if mismatches:
        raise ValueError(
            "VLLM profile does not match candidate configuration; "
            "rerun scripts/probe_vllm_memory.py for this VLLM setting. "
            + "; ".join(mismatches)
        )


def derive_n_batches_in_buffer(
    train_batch_size_tokens: int,
    context_size: int,
) -> int:
    return max(2, math.ceil(train_batch_size_tokens / context_size))


def local_hooks_for_pp_rank(
    pp_rank: int,
    pp_size: int,
    hook_names: list[str],
) -> list[str]:
    if pp_size < 1:
        raise ValueError("pp_size must be >= 1")
    if not 0 <= pp_rank < pp_size:
        raise ValueError(
            f"pp_rank must satisfy 0 <= pp_rank < pp_size; got {pp_rank}, {pp_size}"
        )
    n_hooks = len(hook_names)
    base = n_hooks // pp_size
    extra = n_hooks % pp_size
    start = pp_rank * base + min(pp_rank, extra)
    count = base + (1 if pp_rank < extra else 0)
    return hook_names[start : start + count]


def activation_buffer_mb(
    *,
    d_in: int,
    dtype_bytes: int,
    n_batches_in_buffer: int,
    context_size: int,
    local_num_hooks: int,
    act_store_device: str,
    mix_fraction: float = 0.5,
    store_batch_size_prompts: int = 4,
) -> float:
    if act_store_device == "cpu" or local_num_hooks <= 0:
        return 0.0
    if not 0 <= mix_fraction <= 1:
        raise ValueError("mix_fraction must be in [0, 1]")
    # mixing_buffer peak analysis. During the permute step of a steady-state
    # refill the following are concurrently live on GPU:
    #   * prior iteration's serving_buffer              (buffer_size rows)
    #   * storage_buffer holding the cat output          (buffer_size + chunk)
    #   * permute advanced-index output                  (buffer_size + chunk)
    #   * the current `new_activations` loop variable    (chunk rows)
    # Peak = 3*buffer_size + 3*chunk_rows. mix_fraction == 0 skips permute
    # so the peak drops to 2*buffer_size + 2*chunk_rows.
    buffer_rows = n_batches_in_buffer * context_size
    chunk_rows = store_batch_size_prompts * context_size
    multiplier = 3 if mix_fraction > 0 else 2
    peak_rows = multiplier * (buffer_rows + chunk_rows)
    return peak_rows * d_in * dtype_bytes * local_num_hooks / MB


def streaming_local_buffer_mb(
    *,
    d_in: int,
    dtype_bytes: int,
    local_num_hooks: int,
    train_batch_size_tokens: int,
    streaming_chunk_size_tokens: int,
    streaming_prefetch_chunks: int,
    streaming_mix_chunks: int,
) -> float:
    if local_num_hooks <= 0:
        return 0.0
    take_rows = train_batch_size_tokens * local_num_hooks
    chunk_rows = streaming_chunk_size_tokens * local_num_hooks
    prefetch_rows = max(1, streaming_prefetch_chunks) * chunk_rows
    if streaming_mix_chunks > 0:
        pool_rows = max(take_rows, streaming_mix_chunks * chunk_rows)
    else:
        pool_rows = max(take_rows, prefetch_rows)
    # Refill can transiently hold existing pool/mixing rows and newly read rows.
    return (pool_rows + prefetch_rows) * d_in * dtype_bytes / MB


def cached_loader_buffer_mb(
    *,
    d_in: int,
    dtype_bytes: int,
    local_num_hooks: int,
    n_batches_in_buffer: int,
    context_size: int,
    train_batch_size_tokens: int,
    mix_fraction: float,
    act_store_device: str,
    store_batch_size_prompts: int = 4,
) -> float:
    if act_store_device == "cpu" or local_num_hooks <= 0:
        return 0.0
    buffer_rows = n_batches_in_buffer * context_size
    chunk_rows = store_batch_size_prompts * context_size
    new_rows = max(train_batch_size_tokens, context_size)
    # Shares the mixing_buffer peak with activation_buffer_mb, plus a freshly
    # loaded cached batch that may briefly coexist before _cat_batches absorbs
    # it.
    multiplier = 3 if mix_fraction > 0 else 2
    rows = multiplier * (buffer_rows + chunk_rows) + new_rows
    return rows * d_in * dtype_bytes * local_num_hooks / MB


def estimate_shared_memory_bytes(
    *,
    d_in: int,
    dtype_bytes: int,
    num_hooks: int,
    streaming_chunk_size_tokens: int,
    streaming_num_chunks: int,
) -> int:
    return int(
        d_in
        * dtype_bytes
        * num_hooks
        * streaming_chunk_size_tokens
        * streaming_num_chunks
    )


def load_vllm_profile(path: str | Path) -> VllmMemoryProfile:
    data = json.loads(Path(path).read_text())
    profiles: list[RankVllmProfile] = []
    for raw in data.get("profiles", []):
        profiles.append(
            RankVllmProfile(
                rank=int(raw.get("rank", 0)),
                stage=str(raw.get("stage", "")),
                allocated_mb=float(raw.get("allocated_mb", 0.0)),
                reserved_mb=float(raw.get("reserved_mb", 0.0)),
                peak_allocated_mb=float(raw.get("peak_allocated_mb", 0.0)),
                peak_reserved_mb=float(raw.get("peak_reserved_mb", 0.0)),
                driver_used_mb=float(raw.get("driver_used_mb", 0.0)),
                driver_total_mb=float(raw.get("driver_total_mb", 0.0)),
            )
        )
    if not profiles:
        raise ValueError(f"VLLM profile has no rank profiles: {path}")
    return VllmMemoryProfile(
        model_name=str(data["model_name"]),
        dtype=str(data["dtype"]),
        tensor_parallel_size=int(data["tensor_parallel_size"]),
        gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.0)),
        context_size=int(data.get("context_size", 0)),
        max_model_len=int(data.get("max_model_len", 0)),
        rank_profiles=profiles,
        peak_allocated_mb=max(p.peak_allocated_mb for p in profiles),
        peak_reserved_mb=max(p.peak_reserved_mb for p in profiles),
        peak_driver_used_mb=max(p.driver_used_mb for p in profiles),
    )


def save_vllm_profile(
    *,
    path: str | Path,
    model_name: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    context_size: int,
    max_model_len: int,
    rank_profiles: list[dict[str, Any]],
) -> None:
    out = {
        "schema_version": 1,
        "model_name": model_name,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "context_size": context_size,
        "max_model_len": max_model_len,
        "profiles": rank_profiles,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")


def _sae_component_mb(
    *,
    d_in: int,
    d_sae: int,
    dtype_bytes: int,
    train_batch_tokens: int,
    sae_tp_size: int,
    sae_dp_size: int,
    sae_dp_mode: str,
    local_num_hooks: int,
    is_sae_rank: bool,
    is_vllm_rank: bool,
    sae_tp_rank: int,
) -> dict[str, float]:
    if not is_sae_rank or local_num_hooks <= 0:
        return {
            "sae_params": 0.0,
            "adam": 0.0,
            "grad": 0.0,
            "fwd": 0.0,
            "dp_comm": 0.0,
            "fsdp_bwd_temp": 0.0,
            "extra_hooks": 0.0,
        }

    param_elems = 2 * d_in * d_sae + d_sae + d_in
    full_param_mb = param_elems * dtype_bytes / MB
    sae_unsharded_mb = full_param_mb / sae_tp_size
    adam_divisor = sae_tp_size * (sae_dp_size if sae_dp_mode == "fsdp" else 1)
    grad_divisor = sae_tp_size * (sae_dp_size if sae_dp_mode == "fsdp" else 1)

    hidden_pre_local_mb = train_batch_tokens * (d_sae // sae_tp_size) * dtype_bytes / MB
    feature_acts_mb = train_batch_tokens * d_sae * dtype_bytes / MB
    sae_in_out_mb = train_batch_tokens * d_in * dtype_bytes * 2 / MB
    fwd_one_hook_mb = hidden_pre_local_mb + feature_acts_mb + sae_in_out_mb

    dp_comm_mb = 0.0
    if sae_dp_mode == "ddp":
        dp_comm_mb += 798.0 if sae_tp_rank == 0 else 865.0
    if sae_dp_mode == "fsdp":
        dp_comm_mb += 798.0 if is_vllm_rank else 463.0

    fsdp_bwd_temp_mb = 0.0
    if sae_dp_mode == "fsdp" and not is_vllm_rank:
        fsdp_bwd_temp_mb = 2.0 * sae_unsharded_mb

    return {
        "sae_params": sae_unsharded_mb * local_num_hooks,
        "adam": full_param_mb * 2 / adam_divisor * local_num_hooks,
        "grad": full_param_mb / grad_divisor * local_num_hooks,
        "fwd": fwd_one_hook_mb * local_num_hooks,
        "dp_comm": dp_comm_mb,
        "fsdp_bwd_temp": fsdp_bwd_temp_mb,
        "extra_hooks": max(0, local_num_hooks - 1) * 1470.0,
    }


def _vllm_reserved_for_rank(
    *,
    profile: VllmMemoryProfile,
    local_vllm_rank: int,
    stage: str,
) -> float:
    stage_matches = [
        rank_profile
        for rank_profile in profile.rank_profiles
        if rank_profile.stage == stage
    ]
    for rank_profile in stage_matches:
        if rank_profile.rank == local_vllm_rank:
            return rank_profile.peak_reserved_mb
    if stage_matches:
        return max(rank_profile.peak_reserved_mb for rank_profile in stage_matches)
    for rank_profile in profile.rank_profiles:
        if rank_profile.rank == local_vllm_rank:
            return rank_profile.peak_reserved_mb
    return profile.peak_reserved_mb


def _vllm_profile_for_rank(
    *,
    profile: VllmMemoryProfile,
    local_vllm_rank: int,
    stage: str,
) -> RankVllmProfile:
    stage_matches = [
        rank_profile
        for rank_profile in profile.rank_profiles
        if rank_profile.stage == stage
    ]
    for rank_profile in stage_matches:
        if rank_profile.rank == local_vllm_rank:
            return rank_profile
    if stage_matches:
        return max(stage_matches, key=lambda rank_profile: rank_profile.driver_used_mb)
    for rank_profile in profile.rank_profiles:
        if rank_profile.rank == local_vllm_rank:
            return rank_profile
    if not profile.rank_profiles:
        return RankVllmProfile(
            rank=local_vllm_rank,
            stage=stage,
            allocated_mb=profile.peak_allocated_mb,
            reserved_mb=profile.peak_reserved_mb,
            peak_allocated_mb=profile.peak_allocated_mb,
            peak_reserved_mb=profile.peak_reserved_mb,
            driver_used_mb=profile.peak_driver_used_mb,
            driver_total_mb=0.0,
        )
    return max(profile.rank_profiles, key=lambda rank_profile: rank_profile.driver_used_mb)


def _batch_overhead_mb(train_batch_tokens: int, context_size: int, is_vllm_rank: bool) -> float:
    if not is_vllm_rank:
        return 0.0
    b_eff = max(train_batch_tokens, context_size)
    return max(0.0, 0.1997 * b_eff * math.log2(b_eff) - 1.9010 * b_eff - 501.7)


def estimate_candidate_memory(
    task: TaskSpec,
    config: CandidateConfig,
    vllm_profile: VllmMemoryProfile,
) -> CandidateMemoryEstimate:
    validate_vllm_profile_compatibility(
        task=task,
        config=config,
        profile=vllm_profile,
    )
    dtype_bytes = dtype_nbytes(task.dtype)
    train_batch_tokens = task.train_batch_size_tokens // max(1, config.sae_dp_size)
    n_batches = task.n_batches_in_buffer or derive_n_batches_in_buffer(
        train_batch_tokens, task.context_size
    )

    if config.disjoint or config.mode in {"streaming", "topology_switch"}:
        world_size = config.vllm_dp_size * config.vllm_tp_size + (
            config.sae_dp_size * config.sae_pp_size * config.sae_tp_size
        )
        producer_start = 0
        sae_start = config.vllm_dp_size * config.vllm_tp_size
    else:
        world_size = max(
            config.vllm_dp_size * config.vllm_tp_size,
            config.sae_dp_size * config.sae_pp_size * config.sae_tp_size,
        )
        producer_start = 0
        sae_start = 0

    ranks: list[RankMemoryEstimate] = []
    num_vllm_ranks = config.vllm_dp_size * config.vllm_tp_size
    num_sae_ranks = config.sae_dp_size * config.sae_pp_size * config.sae_tp_size
    for rank in range(world_size):
        is_vllm = producer_start <= rank < producer_start + num_vllm_ranks
        is_sae = sae_start <= rank < sae_start + num_sae_ranks
        if is_sae:
            sae_linear_rank = rank - sae_start
            endpoint_idx = sae_linear_rank // config.sae_tp_size
            sae_tp_rank = sae_linear_rank % config.sae_tp_size
            pp_rank = endpoint_idx % config.sae_pp_size
            local_hooks = local_hooks_for_pp_rank(
                pp_rank, config.sae_pp_size, task.hook_names
            )
        else:
            sae_tp_rank = 0
            local_hooks = []

        components: dict[str, float] = {}
        if is_vllm:
            local_vllm_rank = (rank - producer_start) % config.vllm_tp_size
            rank_profile = _vllm_profile_for_rank(
                profile=vllm_profile,
                local_vllm_rank=local_vllm_rank,
                stage=config.vllm_profile_stage,
            )
            components["vllm_profile"] = rank_profile.peak_reserved_mb
            components["vllm_driver"] = rank_profile.driver_used_mb
            components["vllm_runtime_slack"] = 1000.0
            if config.vllm_dp_size > 1 and config.sae_dp_size == 1:
                components["vllm_dp_comm"] = 577.0
        else:
            components["vllm_profile"] = 0.0
            components["vllm_driver"] = 0.0
            components["vllm_runtime_slack"] = 0.0

        sae_components = _sae_component_mb(
            d_in=task.d_in,
            d_sae=task.d_sae,
            dtype_bytes=dtype_bytes,
            train_batch_tokens=train_batch_tokens,
            sae_tp_size=config.sae_tp_size,
            sae_dp_size=config.sae_dp_size,
            sae_dp_mode=config.sae_dp_mode,
            local_num_hooks=len(local_hooks),
            is_sae_rank=is_sae,
            is_vllm_rank=is_vllm,
            sae_tp_rank=sae_tp_rank,
        )
        components.update(sae_components)
        if config.mode in {"streaming", "topology_switch"}:
            activation_buffer = 0.0
        else:
            activation_buffer = activation_buffer_mb(
                d_in=task.d_in,
                dtype_bytes=dtype_bytes,
                n_batches_in_buffer=n_batches,
                context_size=task.context_size,
                local_num_hooks=len(local_hooks),
                act_store_device=task.act_store_device,
                mix_fraction=task.mix_fraction,
                store_batch_size_prompts=task.store_batch_size_prompts,
            )
        components["activation_buffer"] = activation_buffer
        components["streaming_local_buffer"] = (
            streaming_local_buffer_mb(
                d_in=task.d_in,
                dtype_bytes=dtype_bytes,
                local_num_hooks=len(local_hooks),
                train_batch_size_tokens=train_batch_tokens,
                streaming_chunk_size_tokens=config.streaming_chunk_size_tokens,
                streaming_prefetch_chunks=config.streaming_prefetch_chunks,
                streaming_mix_chunks=config.streaming_mix_chunks,
            )
            if config.mode in {"streaming", "topology_switch"} and is_sae
            else 0.0
        )
        components["cached_loader_buffer"] = (
            cached_loader_buffer_mb(
                d_in=task.d_in,
                dtype_bytes=dtype_bytes,
                local_num_hooks=len(local_hooks),
                n_batches_in_buffer=n_batches,
                context_size=task.context_size,
                train_batch_size_tokens=train_batch_tokens,
                mix_fraction=task.mix_fraction,
                act_store_device=task.act_store_device,
                store_batch_size_prompts=task.store_batch_size_prompts,
            )
            if config.mode == "offline_cache" and is_sae
            else 0.0
        )
        components["batch_overhead"] = _batch_overhead_mb(
            train_batch_tokens, task.context_size, is_vllm and is_sae
        )
        components["split_role_overhead"] = (
            1500.0 if config.mode == "normal" and is_sae and not is_vllm else 0.0
        )
        components["allocator_slack"] = (
            1500.0 if is_vllm or (is_sae and config.sae_dp_mode != "fsdp") else 500.0
        )
        reserved_only_components = {
            "vllm_profile",
            "activation_buffer",
            "streaming_local_buffer",
            "cached_loader_buffer",
            "allocator_slack",
        }
        live_components = {
            name: value
            for name, value in components.items()
            if name not in reserved_only_components
        }
        live_components["vllm_driver"] = components["vllm_driver"]
        predicted_allocated = sum(live_components.values())
        predicted_allocated += components["activation_buffer"]
        predicted_allocated += components["streaming_local_buffer"]
        predicted_allocated += components["cached_loader_buffer"]

        predicted_reserved = predicted_allocated + components["allocator_slack"]
        if is_vllm:
            predicted_reserved = max(
                predicted_reserved,
                components["vllm_profile"]
                + sum(
                    value
                    for name, value in components.items()
                    if name
                    not in {
                        "vllm_profile",
                        "vllm_driver",
                        "allocator_slack",
                    }
                ),
            )
        if is_vllm or is_sae:
            components["base_overhead"] = 63.0 if is_vllm else 2000.0
            if config.sae_dp_mode == "fsdp" and is_sae and not is_vllm:
                components["base_overhead"] = 370.0
            predicted_allocated += components["base_overhead"]
            predicted_reserved += components["base_overhead"]
        role = "+".join(part for part, active in [("vllm", is_vllm), ("sae", is_sae)] if active)
        ranks.append(
            RankMemoryEstimate(
                rank=rank,
                role=role or "idle",
                local_num_hooks=len(local_hooks),
                predicted_allocated_mb=predicted_allocated,
                predicted_reserved_mb=predicted_reserved,
                components=components,
            )
        )

    shm_bytes = 0
    if config.mode in {"streaming", "topology_switch"}:
        shm_bytes = estimate_shared_memory_bytes(
            d_in=task.d_in,
            dtype_bytes=dtype_bytes,
            num_hooks=len(task.hook_names),
            streaming_chunk_size_tokens=config.streaming_chunk_size_tokens,
            streaming_num_chunks=config.streaming_num_chunks,
        )
    return CandidateMemoryEstimate(
        task=task,
        config=config,
        ranks=ranks,
        shared_memory_bytes=shm_bytes,
    )


def estimate_to_dict(estimate: CandidateMemoryEstimate) -> dict[str, Any]:
    return {
        "max_reserved_mb": estimate.max_reserved_mb,
        "shared_memory_bytes": estimate.shared_memory_bytes,
        "ranks": [
            {
                "rank": rank.rank,
                "role": rank.role,
                "local_num_hooks": rank.local_num_hooks,
                "predicted_allocated_mb": rank.predicted_allocated_mb,
                "predicted_reserved_mb": rank.predicted_reserved_mb,
                "components": rank.components,
            }
            for rank in estimate.ranks
        ],
    }
