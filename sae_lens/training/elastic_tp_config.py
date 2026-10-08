"""Configuration and shared I/O for online elastic tensor parallel training."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

OFFLINE_ENV = dict.fromkeys(
    (
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "VLLM_NO_USAGE_STATS",
    ),
    "1",
)


def configure_offline_environment() -> None:
    """Elastic TP consumes prepared local artifacts and never uses the Hub."""
    os.environ.update(OFFLINE_ENV)


def validate_local_sources(model: str, dataset: str) -> None:
    """Reject remote identifiers and missing paths before any model loading."""
    for name, value in (("model", model), ("dataset", dataset)):
        if not Path(value).is_dir():
            raise ValueError(
                f"Elastic TP requires an existing local {name} directory: {value}"
            )


@dataclass
class ElasticTPConfig:
    """Single-node, single-hook FP32 SAE training over a fixed GPU pool.

    Rank zero always trains. Other GPUs host vLLM producers and join or
    leave the SAE TP group at optimizer boundaries as the SHM buffer fills.
    ``min_tp`` bounds training TP; it must leave room to restore a producer.
    ``steps`` counts provider microbatches of ``batch_size`` tokens. One update
    consumes up to ``gradient_accumulation_steps`` microbatches; the final
    window may be shorter. The dataset cycles without increasing the budget.
    """

    model: str
    dataset: str
    hook: str
    d_in: int
    output: Path
    d_sae: int = 65536
    batch_size: int = 4096
    steps: int = 4000
    gradient_accumulation_steps: int = 1
    dead: int = 1500
    k: int = 128
    lr: float = 3e-4
    seed: int = 42
    context: int = 1024
    prompts: int = 1
    pool_size: int = 4
    initial_tp: int = 1
    min_tp: int = 1
    low: float = 0.05
    high: float = 0.85
    chunks: int = 256
    cache_batches: int = 4
    activation_dtype: str | None = None
    vllm_dtype: str = "bfloat16"
    activation_conversion: str = "auto"
    activation_scale: float | None = None
    cooldown: float = 12.0
    poll_interval: float = 0.5
    watermark_samples: int = 3
    startup_timeout: float = 300.0
    pause_timeout: float = 120.0
    resume_timeout: float = 300.0
    vllm_residency: str = "release"
    release_tolerance_mib: int = 64
    validate_activations: bool = False
    profile_only: bool = False
    audit_inputs: bool = False
    producer_id: int = 1
    tp_overlap: str = "off"
    tp_overlap_max_live_hooks: int = 2
    sae_config: dict[str, Any] = field(default_factory=dict)
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float = 0.55
    vllm_text_only: bool = False

    def __post_init__(self) -> None:
        self.output = Path(self.output).resolve()
        if type(self.gradient_accumulation_steps) is not int or self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        for name in (
            "d_in",
            "d_sae",
            "batch_size",
            "steps",
            "k",
            "context",
            "prompts",
            "chunks",
            "cache_batches",
            "watermark_samples",
            "tp_overlap_max_live_hooks",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.vllm_residency not in ("release", "resident"):
            raise ValueError("vllm_residency must be release or resident")
        if type(self.release_tolerance_mib) is not int or self.release_tolerance_mib < 0:
            raise ValueError("release_tolerance_mib must be a nonnegative integer")
        if (
            self.pool_size < 2
            or not 1 <= self.initial_tp <= self.pool_size <= self.d_sae
        ):
            raise ValueError(
                "Require pool_size >= 2 and 1 <= initial_tp <= pool_size <= d_sae; SAE TP0 is not supported"
            )
        if not 1 <= self.min_tp <= self.initial_tp or self.min_tp >= self.pool_size:
            raise ValueError(
                "Require 1 <= min_tp <= initial_tp and min_tp < pool_size; "
                "online training must be able to restore at least one vLLM producer"
            )
        if self.k > self.d_sae:
            raise ValueError("k cannot exceed d_sae")
        if self.batch_size % self.context:
            raise ValueError("batch_size must contain whole context windows")
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("watermarks must satisfy 0 <= low < high <= 1")
        for name in ("lr", "poll_interval", "startup_timeout", "pause_timeout", "resume_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.cooldown) or self.cooldown < 0:
            raise ValueError("cooldown must be finite and non-negative")
        if self.activation_scale is not None and (
            not math.isfinite(self.activation_scale) or self.activation_scale <= 0
        ):
            raise ValueError("activation_scale must be finite and positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if self.max_model_len is not None and self.max_model_len <= self.context:
            raise ValueError("max_model_len must exceed context")
        if (
            self.max_num_batched_tokens is not None
            and self.max_num_batched_tokens < self.context
        ):
            raise ValueError("max_num_batched_tokens must cover one context window")
        if self.tp_overlap not in ("off", "eager", "lazy", "bounded"):
            raise ValueError("Invalid TP overlap mode")
        if (
            self.sae_config.get("topk_tie_policy", "stable_id") != "stable_id"
            or self.sae_config.get("topk_backend") == "legacy"
        ):
            raise ValueError(
                "Elastic TP requires stable_id ties and a sharded TopK backend"
            )
        managed = {"d_in", "d_sae", "k", "dtype", "device", "normalize_activations"}
        if managed.intersection(self.sae_config):
            raise ValueError(
                "sae_config cannot override managed SAE dimensions, dtype, device or normalization"
            )
        activation_dtype(self)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output"] = str(self.output)
        return data


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(handle, event, **data):
    record = dict(event=event, timestamp=time.time(), **data)
    handle.write(json.dumps(record, allow_nan=False) + "\n")
    handle.flush()
    return record


def activation_dtype(args):
    from sae_lens.precision import resolve_activation_dtype

    return resolve_activation_dtype(
        args.vllm_dtype, "float32", args.activation_dtype, args.activation_conversion
    )


def open_buffer(args, *, create=False, name=None):
    import torch

    from sae_lens.training.shared_activation_buffer import SharedActivationBuffer

    if name is None:
        name = json.loads((args.output / "buffer.json").read_text())["name"]
    buffer = SharedActivationBuffer(
        name,
        args.chunks,
        args.batch_size,
        args.d_in,
        num_producers=getattr(args, "pool_size", 4) - 1,
        target_chunks=args.steps,
        create=create,
        dtype=getattr(torch, activation_dtype(args)),
    )
    if buffer._dtype != getattr(torch, activation_dtype(args)):
        buffer.close()
        raise ValueError("Existing SHM dtype differs from --activation-dtype")
    return buffer
