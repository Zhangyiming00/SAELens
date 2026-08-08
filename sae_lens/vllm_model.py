"""
HookedVLLMModel: wraps vLLM's LLM to provide a run_with_cache() interface
compatible with ActivationsStore (activations_store.py:491).

Key design points:
- Registers PyTorch forward hooks on the underlying nn.Module inside the vLLM
  worker to capture activations during prefill.
- enforce_eager=True disables CUDA graphs so per-layer hooks fire correctly.
- VLLM_ACTIVATION_CAPTURE_MODE=1 tells vLLM to allocate only the minimum KV
  cache needed for one batch (no large block pool), since decode is never used.
- VLLM_ENABLE_V1_MULTIPROCESSING=0 forces in-process scheduler so hook
  closures do not need cross-process serialisation.
- For TP, hooks that are post-allreduce (hook_resid_*, hook_attn_out,
  hook_mlp_out) capture full tensors on every rank; rank-0's capture is used
  directly (gather_fn=None).
- Hooks that capture sharded tensors (hook_q/k/v/z, mlp.hook_pre/post) carry
  a gather_fn that concatenates shards from all TP workers into a full tensor.

Supported hook points (TransformerLens naming convention):
  Global:
    hook_embed              – embedding output
  Per-layer (blocks.{L}.*):
    hook_resid_pre          – residual stream entering the block
    hook_resid_mid          – residual stream after attention, before MLP
    hook_resid_post         – residual stream leaving the block
    hook_attn_out           – attention output (after o_proj allreduce)
    hook_mlp_out            – MLP output (after down_proj allreduce)
  Per-layer, attention internals (blocks.{L}.attn.*):
    attn.hook_q             – query vectors before rotary embedding (sharded)
    attn.hook_k             – key vectors before rotary embedding   (sharded)
    attn.hook_v             – value vectors before attention        (sharded)
    attn.hook_z             – attention output before o_proj        (sharded)
  Per-layer, MLP internals (blocks.{L}.mlp.*):
    mlp.hook_pre            – gate projection output before activation (sharded, gate-only to match TL)
    mlp.hook_post           – activation output before down_proj    (sharded)

Not supported (inside FlashAttention kernel, unreachable by forward hooks):
    attn.hook_attn_scores, attn.hook_pattern
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import re
import threading
from contextlib import contextmanager
from functools import partial
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from transformer_lens.utils import USE_DEFAULT_VALUE, get_tokens_with_bos_removed
from transformers import PreTrainedTokenizerBase

from sae_lens.profiling import nccl_nvtx_range

logger = logging.getLogger(__name__)

# Force in-process vLLM scheduler.  Must be set before vllm is imported.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Allow cloudpickle serialisation as a fallback for any worker-to-worker comms.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
# Activation capture mode: allocate only the minimum KV cache for one batch.
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")
# Use spawn (not fork) for TP worker processes.  vLLM's MultiprocExecutor
# initialises CUDA in the parent before forking, which makes forked children
# crash with "Cannot re-initialize CUDA in forked subprocess".  Spawn avoids
# this.  For TP=1 (UniProcExecutor) this env var has no effect.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]


def _get_vllm_tp_device_group() -> dist.ProcessGroup | None:
    """Return vLLM's tensor-parallel device group when available."""
    try:
        from vllm.distributed.parallel_state import get_tp_group as get_vllm_tp_group
    except ImportError:
        return None

    try:
        return get_vllm_tp_group().device_group
    except AssertionError:
        return None


def _get_vllm_tp_rank() -> int | None:
    """Return vLLM's tensor-parallel rank when available."""
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    except ImportError:
        return None

    try:
        return int(get_tensor_model_parallel_rank())
    except AssertionError:
        return None


def _in_torchrun() -> bool:
    """Return True if this process was launched by torchrun / torch.distributed.launch."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"))

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


# ---------------------------------------------------------------------------
# Architecture hook registry
# ---------------------------------------------------------------------------
# Each entry:
#   hook_type → (path_template, extractor_fn, is_pre_hook, gather_fn)
#
# path_template: dotted path for nn.Module.get_submodule(); use {layer} for
#   the layer index.
# extractor_fn:
#   - post-hook (is_pre_hook=False): receives the module and its output, returns
#     the desired activation tensor.
#   - pre-hook  (is_pre_hook=True):  receives the module and the args tuple passed to
#     forward() and returns the desired activation tensor.
# is_pre_hook: True → register_forward_pre_hook; False → register_forward_hook
# gather_fn:
#   None       → post-allreduce; rank-0's capture is the full tensor.
#   callable   → sharded; called as gather_fn(shards: list[Tensor]) → Tensor
#                where shards[i] is the capture from TP worker i.


# ---------------------------------------------------------------------------
# Extractor functions
# ---------------------------------------------------------------------------

def _resid_post_extractor(_module: nn.Module, output: Any) -> torch.Tensor:
    # LlamaDecoderLayer returns (hidden_states, residual).
    # The full residual stream is their sum.
    return output[0] + output[1]


def _resid_pre_extractor(_module: nn.Module, args: tuple) -> torch.Tensor:
    # Pre-hook on LlamaDecoderLayer: args = (positions, hidden_states, residual).
    # For layer 0, residual is None and the stream is just hidden_states.
    _positions, hidden_states, residual = args
    if residual is None:
        return hidden_states
    return hidden_states + residual


def _resid_mid_extractor(_module: nn.Module, args: tuple) -> torch.Tensor:
    # Pre-hook on post_attention_layernorm: args = (hidden_states, residual)
    # = (attention_output, accumulated_residual_before_attention).
    # hook_resid_mid = attn_out + pre_attn_residual.
    hidden_states, residual = args[0], args[1]
    return hidden_states + residual


def _identity_extractor(_module: nn.Module, output: Any) -> torch.Tensor:
    return output


def _first_extractor(_module: nn.Module, output: Any) -> torch.Tensor:
    # For modules that return (tensor, bias_or_None), e.g. MergedColumnParallelLinear.
    return output[0]


def _gate_extractor(_module: nn.Module, output: Any) -> torch.Tensor:
    # gate_up_proj returns [gate, up] concatenated along dim=-1.
    # mlp.hook_pre should match TransformerLens semantics: gate only (first half).
    out = output[0] if isinstance(output, tuple) else output
    return out[..., : out.shape[-1] // 2]


def _qkv_output(module: nn.Module, output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported qkv_proj output type: {type(output)}")


def _q_proj_extractor(module: nn.Module, output: Any) -> torch.Tensor:
    qkv = _qkv_output(module, output)
    q_size = getattr(module, "num_heads") * getattr(module, "head_size")
    return qkv[..., :q_size]


def _k_proj_extractor(module: nn.Module, output: Any) -> torch.Tensor:
    qkv = _qkv_output(module, output)
    q_size = getattr(module, "num_heads") * getattr(module, "head_size")
    kv_size = getattr(module, "num_kv_heads") * getattr(module, "head_size")
    return qkv[..., q_size : q_size + kv_size]


def _v_proj_extractor(module: nn.Module, output: Any) -> torch.Tensor:
    qkv = _qkv_output(module, output)
    q_size = getattr(module, "num_heads") * getattr(module, "head_size")
    k_size = getattr(module, "num_kv_heads") * getattr(module, "head_size")
    v_size = getattr(module, "num_kv_heads") * getattr(module, "v_head_size")
    return qkv[..., q_size + k_size : q_size + k_size + v_size]


# ---------------------------------------------------------------------------
# Gather functions (used in the main process after apply_model returns)
# ---------------------------------------------------------------------------

def _gather_cat(shards: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate shards along the feature dimension (dim=-1)."""
    dev = shards[0].device
    return torch.cat([s.to(dev) for s in shards], dim=-1)


def _gather_gate_up(shards: list[torch.Tensor]) -> torch.Tensor:
    """
    Gather MergedColumnParallelLinear output.

    Each TP rank produces [gate_local, up_local] concatenated.  After simple
    cat we'd get [g0, u0, g1, u1, ...] which is wrong; we need [gate_all, up_all].
    """
    dev = shards[0].device
    half = shards[0].shape[-1] // 2
    gate = torch.cat([s[..., :half].to(dev) for s in shards], dim=-1)
    up = torch.cat([s[..., half:].to(dev) for s in shards], dim=-1)
    return torch.cat([gate, up], dim=-1)


# ---------------------------------------------------------------------------
# Hook registry
# ---------------------------------------------------------------------------

_LLAMA_LIKE_HOOKS: dict[str, tuple[str, Callable, bool, Callable | None]] = {
    # ---- post-allreduce: gather_fn=None, use rank-0's capture ----
    "hook_embed": (
        "model.embed_tokens",
        _identity_extractor,
        False,
        None,
    ),
    "hook_resid_pre": (
        "model.layers.{layer}",
        _resid_pre_extractor,
        True,
        None,
    ),
    "hook_resid_mid": (
        "model.layers.{layer}.post_attention_layernorm",
        _resid_mid_extractor,
        True,
        None,
    ),
    "hook_resid_post": (
        "model.layers.{layer}",
        _resid_post_extractor,
        False,
        None,
    ),
    "hook_attn_out": (
        "model.layers.{layer}.self_attn",
        _identity_extractor,
        False,
        None,
    ),
    "hook_mlp_out": (
        "model.layers.{layer}.mlp",
        _identity_extractor,
        False,
        None,
    ),
    # ---- sharded (ColumnParallel): gather across TP workers ----
    # Attention internals:
    # - q/k/v hook on self_attn.qkv_proj so they match TransformerLens
    #   semantics (pre-RoPE q/k, pre-attention v).
    # - z hook on self_attn.attn output before o_proj.
    # Each rank holds num_heads/tp heads; gather_fn=_gather_cat concatenates.
    "attn.hook_q": (
        "model.layers.{layer}.self_attn.qkv_proj",
        _q_proj_extractor,
        False,
        _gather_cat,
    ),
    "attn.hook_k": (
        "model.layers.{layer}.self_attn.qkv_proj",
        _k_proj_extractor,
        False,
        _gather_cat,
    ),
    "attn.hook_v": (
        "model.layers.{layer}.self_attn.qkv_proj",
        _v_proj_extractor,
        False,
        _gather_cat,
    ),
    "attn.hook_z": (
        "model.layers.{layer}.self_attn.attn",
        _identity_extractor,
        False,
        _gather_cat,
    ),
    # MLP internals.
    # hook_pre: gate_up_proj output, gate half only (first d_mlp_local columns per rank).
    #   Matches TransformerLens mlp.hook_pre semantics. Simple _gather_cat gives correct layout.
    # hook_post: act_fn (SiluAndMul) output, shape (B*S, inter_local) per rank.
    #   Simple cat gives correct layout.
    "mlp.hook_pre": (
        "model.layers.{layer}.mlp.gate_up_proj",
        _gate_extractor,
        False,
        _gather_cat,
    ),
    "mlp.hook_post": (
        "model.layers.{layer}.mlp.act_fn",
        _identity_extractor,
        False,
        _gather_cat,
    ),
}

ARCH_CONFIGS: dict[str, dict[str, tuple[str, Callable, bool, Callable | None]]] = {
    arch: _LLAMA_LIKE_HOOKS
    for arch in [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    ]
}

# ---------------------------------------------------------------------------
# Hook name parsing
# ---------------------------------------------------------------------------

# Matches:
#   "blocks.5.hook_resid_post"      → layer=5, submodule=None,   hook_type="hook_resid_post"
#   "blocks.5.attn.hook_q"          → layer=5, submodule="attn", hook_type="attn.hook_q"
#   "blocks.5.mlp.hook_pre"         → layer=5, submodule="mlp",  hook_type="mlp.hook_pre"
_BLOCKS_PATTERN = re.compile(r"^blocks\.(\d+)\.(?:(attn|mlp)\.)?(hook_\w+)$")
_GLOBAL_PATTERN = re.compile(r"^(hook_\w+)$")


def _parse_hook_name(hook_name: str) -> tuple[str, int | None]:
    """Parse a TransformerLens-style hook name into (hook_type, layer_or_None)."""
    m = _BLOCKS_PATTERN.match(hook_name)
    if m:
        layer = int(m.group(1))
        submodule = m.group(2)  # "attn", "mlp", or None
        base = m.group(3)       # e.g., "hook_q" or "hook_resid_post"
        hook_type = f"{submodule}.{base}" if submodule else base
        return hook_type, layer
    m = _GLOBAL_PATTERN.match(hook_name)
    if m:
        return m.group(1), None
    raise ValueError(
        f"Cannot parse hook name {hook_name!r}. "
        "Expected 'blocks.{{L}}.hook_{{type}}', "
        "'blocks.{{L}}.attn.hook_{{type}}', "
        "'blocks.{{L}}.mlp.hook_{{type}}', "
        "or 'hook_{{type}}'."
    )


# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for multiprocessing with TP>1)
#
# With TP>1, vLLM spawns one worker process per GPU rank.  Functions passed
# as args to LLM.apply_model() are serialised by MessageQueue.enqueue() with
# standard pickle (not cloudpickle).  Module-level functions + functools.partial
# with plain-Python captured values are standard-picklable, so all helpers that
# will be passed to apply_model() are defined at module level.
# ---------------------------------------------------------------------------


def _get_arch_name(model: nn.Module) -> str:
    """Return the class name of the top-level model module."""
    return type(model).__name__


# (substage label, submodule path relative to the decoder layer). A clean
# NON-OVERLAPPING leaf sequence covering the whole decoder layer: attention and
# MLP each split into their projection / kernel steps instead of one opaque box.
# Non-overlapping matters because each pre-hook resets the peak-memory counter,
# so a parent box wrapping its children would only ever see the last child's
# peak. The whole-attn / whole-mlp peak is recoverable in analysis as the max
# over the relevant leaves. Paths not present on a given architecture are
# skipped at install time, so the probe degrades to whatever submodules exist
# rather than crashing.
_VLLM_MEMORY_SUBSTAGES: tuple[tuple[str, str], ...] = (
    ("ln1", "input_layernorm"),
    ("attn_qkv", "self_attn.qkv_proj"),
    ("attn_core", "self_attn.attn"),
    ("attn_o", "self_attn.o_proj"),
    ("ln2", "post_attention_layernorm"),
    ("mlp_gate_up", "mlp.gate_up_proj"),
    ("mlp_act", "mlp.act_fn"),
    ("mlp_down", "mlp.down_proj"),
)


def _tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = _tensor_shape(item)
            if shape is not None:
                return shape
    return None


def _install_vllm_memory_probe(model: nn.Module, layer_idx: int) -> int:
    """Install per-submodule CUDA memory probes on one vLLM decoder layer."""
    model._sae_vllm_memory_records = []  # type: ignore[attr-defined]
    model._sae_vllm_memory_handles = []  # type: ignore[attr-defined]

    if not torch.cuda.is_available():
        return 0

    layer_path = f"model.layers.{layer_idx}"

    def make_pre_hook() -> Callable[[nn.Module, tuple], None]:
        def pre_hook(_module: nn.Module, _args: tuple) -> None:
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        return pre_hook

    def make_post_hook(
        substage: str,
    ) -> Callable[[nn.Module, Any, Any], None]:
        def post_hook(_module: nn.Module, _args: Any, output: Any) -> None:
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            model._sae_vllm_memory_records.append(  # type: ignore[attr-defined]
                {
                    "layer": layer_idx,
                    "substage": substage,
                    "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
                    "peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                    / 1024**2,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device)
                    / 1024**2,
                    "device_mb": (total_bytes - free_bytes) / 1024**2,
                    "out_shape": _tensor_shape(output),
                }
            )

        return post_hook

    installed = 0
    for substage, module_name in _VLLM_MEMORY_SUBSTAGES:
        try:
            module = model.get_submodule(f"{layer_path}.{module_name}")
        except AttributeError:
            # Architecture does not expose this submodule (e.g. fused qkv with a
            # different name). Skip it rather than failing the whole probe.
            continue
        model._sae_vllm_memory_handles.append(  # type: ignore[attr-defined]
            module.register_forward_pre_hook(make_pre_hook())
        )
        model._sae_vllm_memory_handles.append(  # type: ignore[attr-defined]
            module.register_forward_hook(make_post_hook(substage))
        )
        installed += 1
    return installed


def _collect_and_clear_vllm_memory_probe(model: nn.Module) -> list[dict[str, Any]]:
    records = list(getattr(model, "_sae_vllm_memory_records", []))
    if hasattr(model, "_sae_vllm_memory_records"):
        model._sae_vllm_memory_records.clear()  # type: ignore[attr-defined]
    return records


def _remove_vllm_memory_probe(model: nn.Module) -> bool:
    for handle in getattr(model, "_sae_vllm_memory_handles", []):
        handle.remove()
    for attr in ("_sae_vllm_memory_records", "_sae_vllm_memory_handles"):
        if hasattr(model, attr):
            delattr(model, attr)
    return True


def _collect_vllm_static_memory(worker: Any) -> dict[str, Any]:
    """Static (resting) memory breakdown for a vLLM worker.

    Runs inside the worker process via ``LLM.collective_rpc`` (so ``self`` is
    the worker, which owns ``model_runner``). Decomposes the resting GPU
    footprint into the big static blocks that do NOT move during capture:

      * ``weights_mb``      — model parameters (``model_runner.model_memory_usage``,
                              the value vLLM itself measured at load).
      * ``kv_cache_mb``     — sum of the pre-allocated KV cache tensors
                              (``model_runner.kv_caches``); the pool vLLM sized
                              from leftover VRAM at profile_run.
      * ``non_torch_mb``    — CUDA/NCCL context + any non-torch allocations vLLM
                              accounts separately (``worker.non_torch_memory``).
      * ``allocated_mb`` / ``reserved_mb`` / ``device_mb`` — the same three
                              torch/driver totals the per-substage records carry,
                              so the static record can be reconciled against them.

    Everything is best-effort: a missing attribute (different vLLM version or
    architecture) yields 0 for that field rather than raising, so the probe
    degrades gracefully.
    """
    to_mb = 1 / 1024**2
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)

    runner = getattr(worker, "model_runner", None)

    weights_bytes = int(getattr(runner, "model_memory_usage", 0) or 0)

    kv_cache_bytes = 0
    seen: set[int] = set()
    for cache in getattr(runner, "kv_caches", []) or []:
        if torch.is_tensor(cache) and cache.device.type == "cuda":
            ident = id(cache)
            if ident in seen:
                continue
            seen.add(ident)
            kv_cache_bytes += cache.numel() * cache.element_size()

    non_torch_bytes = int(getattr(worker, "non_torch_memory", 0) or 0)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "weights_mb": weights_bytes * to_mb,
        "kv_cache_mb": kv_cache_bytes * to_mb,
        "non_torch_mb": non_torch_bytes * to_mb,
        "allocated_mb": torch.cuda.memory_allocated(device) * to_mb,
        "reserved_mb": torch.cuda.memory_reserved(device) * to_mb,
        "device_mb": (total_bytes - free_bytes) * to_mb,
    }


def write_vllm_static_memory_record(
    path: str | Path,
    *,
    record: dict[str, Any],
    step: int,
    n_training_samples: int | None,
    rank: int,
    producer_idx: int | None = None,
    vllm_tp_rank: int | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "n_training_samples": n_training_samples,
        "rank": rank,
        "producer_idx": producer_idx,
        "vllm_tp_rank": vllm_tp_rank,
        **record,
    }
    with out_path.open("a") as f:
        json.dump(payload, f)
        f.write("\n")


def write_vllm_memory_records(
    path: str | Path,
    *,
    records: list[dict[str, Any]],
    step: int,
    n_training_samples: int | None,
    rank: int,
    producer_idx: int | None = None,
    vllm_tp_rank: int | None = None,
) -> None:
    if not records:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for record in records:
            payload: dict[str, Any] = {
                "step": step,
                "n_training_samples": n_training_samples,
                "rank": rank,
                "producer_idx": producer_idx,
                "vllm_tp_rank": vllm_tp_rank,
                **record,
            }
            json.dump(payload, f)
            f.write("\n")


@contextmanager
def _maybe_record_vllm_memory_timeline(
    *,
    enabled: bool,
    target_step: int,
    current_step: int,
    path: str | Path | None,
) -> Iterator[None]:
    """Record one vLLM ``run_with_cache`` allocator timeline as a pickle."""
    if (
        not enabled
        or target_step < 0
        or current_step != target_step
        or path is None
        or not torch.cuda.is_available()
    ):
        yield
        return

    device = torch.cuda.current_device()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize(device)
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000,
        stacks="all",
        context="all",
    )
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        torch.cuda.memory._dump_snapshot(str(path))
        torch.cuda.memory._record_memory_history(enabled=None)


def _vllm_timeline_worker_path(
    path: str | Path,
    *,
    suffix_tp_rank: bool,
) -> Path:
    out_path = Path(path)
    if not suffix_tp_rank:
        return out_path
    tp_rank = _get_vllm_tp_rank()
    if tp_rank is None:
        tp_rank = torch.cuda.current_device()
    return out_path.with_name(f"{out_path.stem}_vllm_tp{tp_rank}{out_path.suffix}")


def _start_vllm_memory_timeline(
    model: nn.Module,
    *,
    path: str | Path,
    suffix_tp_rank: bool = False,
) -> str | None:
    """Start allocator history inside a vLLM worker process."""
    if not torch.cuda.is_available():
        return None
    out_path = _vllm_timeline_worker_path(path, suffix_tp_rank=suffix_tp_rank)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000,
        stacks="all",
        context="all",
    )
    model._sae_vllm_memory_timeline_path = str(out_path)  # type: ignore[attr-defined]
    model._sae_vllm_memory_timeline_active = True  # type: ignore[attr-defined]
    return str(out_path)


def _stop_vllm_memory_timeline(model: nn.Module) -> str | None:
    """Dump and stop allocator history inside a vLLM worker process."""
    if not getattr(model, "_sae_vllm_memory_timeline_active", False):
        return None
    path = getattr(model, "_sae_vllm_memory_timeline_path")
    device = torch.cuda.current_device()
    try:
        torch.cuda.synchronize(device)
        torch.cuda.memory._dump_snapshot(str(path))
        return str(path)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        for attr in (
            "_sae_vllm_memory_timeline_path",
            "_sae_vllm_memory_timeline_active",
        ):
            if hasattr(model, attr):
                delattr(model, attr)


def _register_hooks(
    model: nn.Module,
    hook_specs: list[tuple[str, str, Callable, bool, Callable | None]],
    total_tokens: int,
    stop_at_layer: int | None,
) -> None:
    """Register SAE capture hooks on the worker's model."""
    # Use lists to accumulate chunks from chunked prefill.
    model._sae_captures: dict[str, list[torch.Tensor]] = {}  # type: ignore[attr-defined]
    model._sae_handles: list = []  # type: ignore[attr-defined]
    if stop_at_layer is not None:
        model.model._sae_stop_at_layer = stop_at_layer  # type: ignore[attr-defined]

    for hook_name, path, extractor, is_pre, _gather_fn in hook_specs:
        module = model.get_submodule(path)

        if is_pre:

            def make_pre_hook(
                name: str, ext: Callable
            ) -> Callable[[nn.Module, tuple], None]:
                def hook_fn(m: nn.Module, args: tuple) -> None:
                    act = ext(m, args)
                    # Accumulate all chunks (chunked prefill splits the batch).
                    # We slice to total_tokens in run_with_cache.
                    if name not in model._sae_captures:  # type: ignore[attr-defined]
                        model._sae_captures[name] = []  # type: ignore[attr-defined]
                    model._sae_captures[name].append(act.detach().clone())  # type: ignore[attr-defined]

                return hook_fn

            handle = module.register_forward_pre_hook(make_pre_hook(hook_name, extractor))
        else:

            def make_post_hook(
                name: str, ext: Callable
            ) -> Callable[[nn.Module, Any, Any], None]:
                def hook_fn(m: nn.Module, inp: Any, out: Any) -> None:
                    act = ext(m, out)
                    if name not in model._sae_captures:  # type: ignore[attr-defined]
                        model._sae_captures[name] = []  # type: ignore[attr-defined]
                    model._sae_captures[name].append(act.detach().clone())  # type: ignore[attr-defined]

                return hook_fn

            handle = module.register_forward_hook(make_post_hook(hook_name, extractor))

        model._sae_handles.append(handle)  # type: ignore[attr-defined]


def _finalize_captures(
    capture_lists: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Finalize chunked captures without copying when only one chunk exists."""
    captures: dict[str, torch.Tensor] = {}
    for name, chunks in capture_lists.items():
        if len(chunks) == 1:
            captures[name] = chunks[0]
        else:
            captures[name] = torch.cat(chunks, dim=0)
    return captures


def _collect_and_cleanup(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect captured activations and remove all hooks from the model."""
    captures = _finalize_captures(model._sae_captures)  # type: ignore[attr-defined]
    for handle in model._sae_handles:  # type: ignore[attr-defined]
        handle.remove()
    del model._sae_captures, model._sae_handles  # type: ignore[attr-defined]
    if hasattr(model.model, "_sae_stop_at_layer"):
        del model.model._sae_stop_at_layer  # type: ignore[attr-defined]
    return captures


def _reshape_captured_activation(
    raw: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    total_tokens: int,
    hook_name: str,
) -> torch.Tensor:
    """Validate and reshape vLLM hook captures to (B, S, d_model)."""
    if raw.ndim < 2:
        raise RuntimeError(
            f"vLLM hook {hook_name!r} returned activation with invalid shape "
            f"{tuple(raw.shape)}"
        )
    flat = raw.reshape(-1, raw.shape[-1])
    if flat.shape[0] < total_tokens:
        raise RuntimeError(
            f"vLLM hook {hook_name!r} captured fewer activation rows than "
            f"requested tokens: captured={flat.shape[0]}, expected={total_tokens}. "
            "This usually means prefix caching reused tokens, so forward hooks "
            "did not fire for the full prompt. Disable vLLM prefix caching for "
            "activation capture."
        )
    return flat[:total_tokens].view(batch_size, seq_len, -1)


# ---------------------------------------------------------------------------
# CUDA IPC helpers for TP>1
#
# With TP>1, vLLM's MultiprocExecutor uses ZMQ to transfer apply_model()
# return values from worker to main process.  Standard pickle cannot handle
# CUDA tensors, and serialising large activations (~128 MB) as CPU bytes
# takes ~800 ms at the ZMQ bandwidth (~160 MB/s).
#
# Instead, we:
#   1. Keep activations on GPU in hooks (no .cpu()).
#   2. In _collect_and_pin, cat on GPU, store in the worker's module-level
#      _CUDA_IPC_PINNED dict to prevent GC, then use ForkingPickler to
#      serialise the dict to CUDA IPC handle bytes (~64 bytes per tensor).
#   3. Return those tiny bytes through ZMQ (microseconds).
#   4. Main process calls pickle.loads — torch's registered reducers
#      reconstruct tensors pointing to the same GPU memory (zero copy).
#   5. After main is done, _release_pinned deletes the worker's reference;
#      CUDA frees the memory once all IPC handles are closed.
#
# For TP=1 (UniProcExecutor), apply_model runs in-process and Python objects
# are returned directly — no serialisation at all, so _collect_and_cleanup
# already returns CUDA tensors with zero overhead.
# ---------------------------------------------------------------------------

# Worker-side storage that keeps GPU tensors alive while main process holds
# CUDA IPC handles to the same memory.
_CUDA_IPC_PINNED: dict[str, torch.Tensor] = {}


def _collect_and_pin(model: nn.Module) -> bytes:
    """
    Collect GPU activations, pin them, and return CUDA IPC handle bytes.

    The bytes are created by ForkingPickler (~64 B per tensor, not the tensor
    data), so ZMQ transfer is microseconds regardless of activation size.
    Main process reconstructs zero-copy CUDA tensors via pickle.loads.
    """
    global _CUDA_IPC_PINNED
    captures = _finalize_captures(model._sae_captures)  # type: ignore[attr-defined]
    for handle in model._sae_handles:  # type: ignore[attr-defined]
        handle.remove()
    del model._sae_captures, model._sae_handles  # type: ignore[attr-defined]
    if hasattr(model.model, "_sae_stop_at_layer"):
        del model.model._sae_stop_at_layer  # type: ignore[attr-defined]
    _CUDA_IPC_PINNED = captures  # prevent GC until _release_pinned is called
    buf = io.BytesIO()
    ForkingPickler(buf, 2).dump(captures)
    return buf.getvalue()


def _collect_and_pin_selective(
    model: nn.Module,
    rank0_only_hooks: tuple[str, ...],
) -> bytes:
    """
    Collect GPU activations and only return rank-0 copies for full-tensor hooks.

    Hooks in ``rank0_only_hooks`` are post-allreduce tensors that are identical on
    every TP rank. Non-zero TP ranks drop them before serialisation so the main
    process does not redundantly reconstruct duplicate full tensors.
    """
    tp_rank = _get_vllm_tp_rank()
    captures = _finalize_captures(model._sae_captures)  # type: ignore[attr-defined]
    if tp_rank not in (None, 0):
        for hook_name in rank0_only_hooks:
            captures.pop(hook_name, None)

    for handle in model._sae_handles:  # type: ignore[attr-defined]
        handle.remove()
    del model._sae_captures, model._sae_handles  # type: ignore[attr-defined]
    if hasattr(model.model, "_sae_stop_at_layer"):
        del model.model._sae_stop_at_layer  # type: ignore[attr-defined]

    global _CUDA_IPC_PINNED
    _CUDA_IPC_PINNED = captures
    buf = io.BytesIO()
    ForkingPickler(buf, 2).dump(captures)
    return buf.getvalue()


def _release_pinned(model: nn.Module) -> None:
    """Release worker-side pinned activations after main process is done."""
    global _CUDA_IPC_PINNED
    _CUDA_IPC_PINNED = {}


# ---------------------------------------------------------------------------
# HookedVLLMModel
# ---------------------------------------------------------------------------


class HookedVLLMModel:
    """
    Wraps vLLM's LLM to provide a run_with_cache() interface compatible with
    ActivationsStore.get_activations() (activations_store.py:491).

    Supports tensor parallelism via vLLM's tensor_parallel_size kwarg.
    Post-allreduce hooks (hook_resid_*, hook_attn_out, hook_mlp_out) use
    rank-0's capture directly.  Sharded hooks (attn.hook_q/k/v/z,
    mlp.hook_pre/post) concatenate captures from all TP workers.

    Supported architectures:

    - LlamaForCausalLM
    - MistralForCausalLM
    - GemmaForCausalLM
    - Gemma2ForCausalLM
    - Qwen2ForCausalLM
    - Qwen3ForCausalLM
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype = torch.bfloat16,
        capture_batch_size: int | None = None,
        capture_context_size: int | None = None,
        allow_cold_reconfigure: bool = False,
        cold_reconfigure_mbt_capacity: int | None = None,
        cold_reconfigure_kv_pool_capacity_tokens: int | None = None,
        **llm_kwargs: Any,
    ) -> None:
        if LLM is None:
            raise ImportError(
                "vllm must be installed to use HookedVLLMModel. "
                "Install with `pip install vllm`."
            )
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._generation_lock = threading.RLock()
        self.allow_cold_reconfigure = bool(allow_cold_reconfigure)
        self._cold_reconfigure_mbt_capacity: int | None = None
        self._cold_reconfigure_kv_pool_capacity_tokens: int | None = None

        # In activation-capture mode, size the minimal KV cache pool to the
        # actual per-batch workload (batch_size * context_size) rather than
        # vLLM's max_num_batched_tokens, so the whole batch's KV can reside at
        # once. These are read by gpu_worker.determine_available_memory via
        # vllm_config.additional_config; if either is None the worker falls
        # back to max_num_batched_tokens.
        if self.allow_cold_reconfigure:
            mbt_capacity = (
                cold_reconfigure_mbt_capacity
                if cold_reconfigure_mbt_capacity is not None
                else llm_kwargs.get("max_num_batched_tokens")
            )
            if mbt_capacity is None:
                raise ValueError(
                    "allow_cold_reconfigure=True requires "
                    "cold_reconfigure_mbt_capacity or max_num_batched_tokens."
                )
            mbt_capacity = int(mbt_capacity)
            if mbt_capacity < 1:
                raise ValueError(
                    "cold_reconfigure_mbt_capacity must be >= 1; "
                    f"got {mbt_capacity}."
                )

            if cold_reconfigure_kv_pool_capacity_tokens is not None:
                kv_pool_capacity_tokens = int(
                    cold_reconfigure_kv_pool_capacity_tokens
                )
            elif capture_batch_size is not None and capture_context_size is not None:
                kv_pool_capacity_tokens = int(capture_batch_size) * int(
                    capture_context_size
                )
            else:
                kv_pool_capacity_tokens = mbt_capacity
            if kv_pool_capacity_tokens < 1:
                raise ValueError(
                    "cold_reconfigure_kv_pool_capacity_tokens must be >= 1; "
                    f"got {kv_pool_capacity_tokens}."
                )

            llm_kwargs["max_num_batched_tokens"] = mbt_capacity
            additional_config = dict(llm_kwargs.get("additional_config") or {})
            additional_config["sae_allow_cold_reconfigure"] = True
            additional_config["sae_capture_kv_pool_capacity_tokens"] = (
                kv_pool_capacity_tokens
            )
            if capture_batch_size is not None and capture_context_size is not None:
                additional_config.setdefault(
                    "sae_capture_batch_size", int(capture_batch_size)
                )
                additional_config.setdefault(
                    "sae_capture_context_size", int(capture_context_size)
                )
            llm_kwargs["additional_config"] = additional_config
            self._cold_reconfigure_mbt_capacity = mbt_capacity
            self._cold_reconfigure_kv_pool_capacity_tokens = kv_pool_capacity_tokens
        elif capture_batch_size is not None and capture_context_size is not None:
            additional_config = dict(llm_kwargs.get("additional_config") or {})
            additional_config.setdefault(
                "sae_capture_batch_size", int(capture_batch_size)
            )
            additional_config.setdefault(
                "sae_capture_context_size", int(capture_context_size)
            )
            llm_kwargs["additional_config"] = additional_config

        # enforce_eager=True: disables CUDA graphs so per-layer hooks fire.
        # VLLM_ACTIVATION_CAPTURE_MODE=1 (set at module load above) ensures
        # vLLM allocates only the minimal KV cache needed for one batch.
        llm_kwargs.setdefault("enforce_eager", True)
        # Prefix caching skips forward execution for cached prompt tokens, so
        # activation hooks only see the uncached suffix. That silently corrupts
        # captured activation order/shape unless disabled.
        llm_kwargs.setdefault("enable_prefix_caching", False)
        dtype_str = _DTYPE_TO_STR.get(dtype, "bfloat16")
        llm_kwargs.setdefault("dtype", dtype_str)
        explicit_device = llm_kwargs.get("device")

        # When launched under torchrun (RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT
        # are in the environment), vLLM must NOT spawn its own TP worker
        # processes — that creates nested multiprocessing which deadlocks
        # waiting for inner workers to become ready.  Instead, use
        # "external_launcher": each torchrun rank creates its own LLM
        # instance backed by ExecutorWithExternalLauncher (a UniProcExecutor)
        # that runs the model inline.  All torchrun ranks call generate()
        # simultaneously and TP communication happens via the already-
        # initialised torch.distributed process group.
        tp = llm_kwargs.get("tensor_parallel_size", 1)
        if tp > 1 and _in_torchrun():
            llm_kwargs.setdefault("distributed_executor_backend", "external_launcher")

        self._normalized_llm_kwargs = dict(llm_kwargs)
        self.llm = LLM(model_name, **llm_kwargs)
        self._is_external_launcher = (
            llm_kwargs.get("distributed_executor_backend") == "external_launcher"
        )

        if self.allow_cold_reconfigure:
            status = self.get_cold_reconfigure_status()
            if (
                int(status["max_num_batched_tokens_capacity"])
                != self._cold_reconfigure_mbt_capacity
            ):
                raise RuntimeError(
                    "vLLM reported max_num_batched_tokens_capacity="
                    f"{status['max_num_batched_tokens_capacity']}, expected "
                    f"{self._cold_reconfigure_mbt_capacity}."
                )
            if (
                int(status["active_max_num_batched_tokens"])
                != self._cold_reconfigure_mbt_capacity
            ):
                raise RuntimeError(
                    "vLLM initial active_max_num_batched_tokens="
                    f"{status['active_max_num_batched_tokens']}, expected "
                    f"{self._cold_reconfigure_mbt_capacity}."
                )
            if (
                int(status["kv_pool_capacity_tokens"])
                != self._cold_reconfigure_kv_pool_capacity_tokens
            ):
                raise RuntimeError(
                    "vLLM reported kv_pool_capacity_tokens="
                    f"{status['kv_pool_capacity_tokens']}, expected "
                    f"{self._cold_reconfigure_kv_pool_capacity_tokens}."
                )

        arch: str = self.llm.apply_model(_get_arch_name)[0]
        if arch not in ARCH_CONFIGS:
            raise ValueError(
                f"Architecture {arch!r} is not supported by HookedVLLMModel. "
                f"Supported: {sorted(ARCH_CONFIGS)}"
            )
        self._arch = arch
        self._tp: int = llm_kwargs.get("tensor_parallel_size", 1)
        self.dtype = dtype
        # ActivationsStore calls _get_model_device(model) which falls back to
        # next(model.parameters()).device.  vLLM always runs on CUDA so we
        # expose a device property to satisfy that check without needing
        # model.parameters().
        self.device = (
            torch.device(explicit_device)
            if explicit_device is not None
            else torch.device("cuda")
        )

    def _external_reconfigure_group(self) -> dist.ProcessGroup | None:
        if not self._is_external_launcher:
            return None
        if not dist.is_available() or not dist.is_initialized():
            if self._tp > 1:
                raise RuntimeError(
                    "vLLM external_launcher cold reconfigure requires an "
                    "initialized torch.distributed process group."
                )
            return None
        tp_group = _get_vllm_tp_device_group()
        if tp_group is None and self._tp > 1:
            raise RuntimeError(
                "vLLM TP group is not initialized under external_launcher."
            )
        return tp_group

    def _external_reconfigure_device(self) -> torch.device:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return self.device
        return torch.device("cpu")

    def _check_external_reconfigure_value(self, value: int) -> None:
        group = self._external_reconfigure_group()
        if group is None:
            return
        device = self._external_reconfigure_device()
        min_value = torch.tensor([value], device=device, dtype=torch.int64)
        max_value = torch.tensor([value], device=device, dtype=torch.int64)
        dist.all_reduce(min_value, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX, group=group)
        if int(min_value.item()) != int(max_value.item()):
            raise RuntimeError(
                "All vLLM external_launcher ranks must request the same "
                "active max_num_batched_tokens; got distributed min/max "
                f"{int(min_value.item())}/{int(max_value.item())}."
            )
        dist.barrier(group=group)

    def _external_reconfigure_success_allreduce(self, success: bool) -> bool:
        group = self._external_reconfigure_group()
        if group is None:
            return success
        device = self._external_reconfigure_device()
        success_tensor = torch.tensor(
            [1 if success else 0], device=device, dtype=torch.int64
        )
        dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN, group=group)
        return bool(success_tensor.item())

    def _external_reconfigure_barrier(self) -> None:
        group = self._external_reconfigure_group()
        if group is not None:
            dist.barrier(group=group)

    def get_cold_reconfigure_status(self) -> dict[str, Any]:
        if not hasattr(self.llm, "get_cold_reconfigure_status"):
            raise RuntimeError(
                "The installed vLLM LLM object does not expose "
                "get_cold_reconfigure_status()."
            )
        return dict(self.llm.get_cold_reconfigure_status())

    def cold_reconfigure(
        self,
        *,
        max_num_batched_tokens: int,
    ) -> dict[str, Any]:
        if not self.allow_cold_reconfigure:
            raise RuntimeError(
                "cold_reconfigure() requires allow_cold_reconfigure=True."
            )

        value = int(max_num_batched_tokens)

        with self._generation_lock:
            self._check_external_reconfigure_value(value)

            local_updated = False
            local_error: BaseException | None = None
            old_value: int | None = None
            result: dict[str, Any] | None = None
            try:
                capacity = self._cold_reconfigure_mbt_capacity
                if capacity is None:
                    raise RuntimeError(
                        "Cold reconfigure capacity was not initialized."
                    )
                if value < 1 or value > capacity:
                    raise ValueError(
                        "Requested active max_num_batched_tokens must satisfy "
                        f"1 <= value <= capacity ({capacity}); got {value}."
                    )

                before = self.get_cold_reconfigure_status()
                old_value = int(before["active_max_num_batched_tokens"])
                result = dict(self.llm.set_active_max_num_batched_tokens(value))
                local_updated = True
                after = self.get_cold_reconfigure_status()
                if int(after["active_max_num_batched_tokens"]) != value:
                    raise RuntimeError(
                        "vLLM cold reconfigure did not take effect: "
                        f"requested {value}, status={after}."
                    )
                result.update(after)
                result.setdefault("old_active_max_num_batched_tokens", old_value)
            except BaseException as exc:
                local_error = exc

            globally_successful = self._external_reconfigure_success_allreduce(
                local_error is None
            )
            if not globally_successful:
                if local_updated and old_value is not None and old_value != value:
                    try:
                        self.llm.set_active_max_num_batched_tokens(old_value)
                    except BaseException as rollback_exc:
                        logger.error(
                            "Failed to roll back active MBT after distributed "
                            "cold reconfigure failure: %s",
                            rollback_exc,
                        )
                self._external_reconfigure_barrier()
                if local_error is not None:
                    if self._external_reconfigure_group() is None:
                        raise local_error
                    raise RuntimeError(
                        "Cold reconfigure failed on this rank; all ranks "
                        "aborted the update."
                    ) from local_error
                raise RuntimeError(
                    "Cold reconfigure failed on another external_launcher "
                    "rank; this rank rolled back to the previous active MBT."
                )

            self._external_reconfigure_barrier()
            assert result is not None
            tp_rank = _get_vllm_tp_rank()
            if tp_rank in (None, 0):
                logger.info(
                    "[COLD RECONFIGURE] active MBT %s -> %s, capacity=%s, "
                    "pool_capacity=%s",
                    result["old_active_max_num_batched_tokens"],
                    result["active_max_num_batched_tokens"],
                    result["max_num_batched_tokens_capacity"],
                    result["kv_pool_capacity_tokens"],
                )
            return result

    def run_with_cache(
        self,
        batch_tokens: torch.Tensor,  # (B, S)
        names_filter: list[str],
        **kwargs: Any,  # stop_at_layer, prepend_bos, etc. accepted but ignored
    ) -> tuple[None, dict[str, torch.Tensor]]:
        """
        Run prefill on batch_tokens and return captured activations.

        Args:
            batch_tokens: integer token ids of shape (B, S).
            names_filter: list of TransformerLens-style hook names, e.g.
                ``["blocks.21.hook_resid_post", "blocks.21.attn.hook_q"]``.

        Returns:
            (None, {hook_name: tensor of shape (B, S, d)})
        """
        with self._generation_lock:
            return self._run_with_cache_unlocked(batch_tokens, names_filter, **kwargs)

    def _run_with_cache_unlocked(
        self,
        batch_tokens: torch.Tensor,
        names_filter: list[str],
        **kwargs: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        B, S = batch_tokens.shape
        arch_config = ARCH_CONFIGS[self._arch]
        total_tokens = B * S
        stop_at_layer: int | None = kwargs.get("stop_at_layer", None)
        memory_probe_layer: int | None = kwargs.get("vllm_memory_probe_layer", None)
        memory_probe_path: str | Path | None = kwargs.get("vllm_memory_history_path", None)
        memory_probe_step: int = int(kwargs.get("vllm_memory_step", 0))
        memory_probe_samples = kwargs.get("vllm_memory_n_training_samples", None)
        memory_probe_rank: int = int(kwargs.get("vllm_memory_rank", 0))
        memory_probe_producer_idx = kwargs.get("vllm_memory_producer_idx", None)
        memory_probe_tp_rank = kwargs.get("vllm_memory_tp_rank", None)
        memory_timeline_step: int = int(
            kwargs.get("vllm_memory_timeline_step", -1)
        )
        memory_timeline_current_step: int = int(
            kwargs.get("vllm_memory_timeline_current_step", memory_probe_step)
        )
        memory_timeline_path: str | Path | None = kwargs.get(
            "vllm_memory_timeline_path", None
        )
        memory_timeline_enabled = (
            memory_timeline_step >= 0 and memory_timeline_path is not None
        )
        memory_probe_enabled = (
            memory_probe_layer is not None
            and memory_probe_path is not None
            and memory_probe_step > 0
        )

        # Resolve hook names → (name, module_path, extractor, is_pre_hook, gather_fn).
        hook_specs: list[tuple[str, str, Callable, bool, Callable | None]] = []
        for hook_name in names_filter:
            hook_type, layer = _parse_hook_name(hook_name)
            if hook_type not in arch_config:
                raise ValueError(
                    f"Hook type {hook_type!r} not supported for {self._arch}. "
                    f"Supported: {sorted(arch_config)}"
                )
            path_tpl, extractor, is_pre, gather_fn = arch_config[hook_type]
            path = path_tpl if layer is None else path_tpl.format(layer=layer)
            hook_specs.append((hook_name, path, extractor, is_pre, gather_fn))

        prompts = [{"prompt_token_ids": batch_tokens[i].tolist()} for i in range(B)]
        register = partial(
            _register_hooks,
            hook_specs=hook_specs,
            total_tokens=total_tokens,
            stop_at_layer=stop_at_layer,
        )
        install_memory_probe = (
            partial(_install_vllm_memory_probe, layer_idx=int(memory_probe_layer))
            if memory_probe_enabled
            else None
        )

        def collect_memory_probe() -> list[dict[str, Any]]:
            if not memory_probe_enabled:
                return []
            records_by_rank = self.llm.apply_model(_collect_and_clear_vllm_memory_probe)
            records: list[dict[str, Any]] = []
            for worker_records in records_by_rank:
                records.extend(worker_records)
            return records

        def remove_memory_probe() -> None:
            if memory_probe_enabled:
                self.llm.apply_model(_remove_vllm_memory_probe)

        def collect_and_remove_memory_probe() -> list[dict[str, Any]]:
            if not memory_probe_enabled:
                return []
            try:
                return collect_memory_probe()
            finally:
                remove_memory_probe()

        def write_static_memory() -> None:
            """Collect the resting weights/kv/non_torch breakdown once (per
            probed step) and append it to vllm_static_memory_rank{N}.jsonl next
            to the per-substage history. Best-effort: never fail the run."""
            if not memory_probe_enabled or memory_probe_path is None:
                return
            try:
                records = self.llm.collective_rpc(_collect_vllm_static_memory)
            except Exception as exc:  # pragma: no cover - version/runtime guard
                logger.warning("vLLM static memory collection failed: %s", exc)
                return
            static_path = Path(memory_probe_path).with_name(
                Path(memory_probe_path).name.replace(
                    "vllm_memory_history", "vllm_static_memory"
                )
            )
            # rank-0 worker's view is representative; under TP each rank holds
            # the same weights footprint and an equal KV shard, so record rank 0.
            if records:
                write_vllm_static_memory_record(
                    static_path,
                    record=records[0],
                    step=memory_probe_step,
                    n_training_samples=memory_probe_samples,
                    rank=memory_probe_rank,
                    producer_idx=memory_probe_producer_idx,
                    vllm_tp_rank=memory_probe_tp_rank,
                )

        if self._tp == 1 or self._is_external_launcher:
            # UniProcExecutor (tp=1) or external_launcher: apply_model runs
            # inline in the current process; results[0] is the local captures.
            #
            # For external_launcher with TP>1 all torchrun ranks call
            # generate() simultaneously — TP communication happens via NCCL.
            # Post-allreduce hooks already hold the full tensor on every rank.
            # Sharded hooks need dist.all_gather across the TP ranks.
            self.llm.apply_model(register)
            if install_memory_probe is not None:
                self.llm.apply_model(install_memory_probe)
            try:
                with _maybe_record_vllm_memory_timeline(
                    enabled=memory_timeline_enabled,
                    target_step=memory_timeline_step,
                    current_step=memory_timeline_current_step,
                    path=memory_timeline_path,
                ):
                    self.llm.generate(
                        prompts, SamplingParams(max_tokens=1), use_tqdm=False
                    )
            finally:
                memory_records = collect_and_remove_memory_probe()
                results = self.llm.apply_model(_collect_and_cleanup)
            if memory_probe_enabled and memory_probe_path is not None:
                write_vllm_memory_records(
                    memory_probe_path,
                    records=memory_records,
                    step=memory_probe_step,
                    n_training_samples=memory_probe_samples,
                    rank=memory_probe_rank,
                    producer_idx=memory_probe_producer_idx,
                    vllm_tp_rank=memory_probe_tp_rank,
                )
                write_static_memory()

            local_caps = results[0]
            activations: dict[str, torch.Tensor] = {}
            for hook_name, _path, _extractor, _is_pre, gather_fn in hook_specs:
                raw = local_caps[hook_name]
                if gather_fn is not None and self._is_external_launcher and dist.is_initialized():
                    tp_group = _get_vllm_tp_device_group()
                    if tp_group is None:
                        raise RuntimeError(
                            "vLLM TP group is not initialized under external_launcher"
                        )
                    world_size = dist.get_world_size(tp_group)
                    shards = [torch.zeros_like(raw) for _ in range(world_size)]
                    with nccl_nvtx_range("nccl:vllm_hook_shard_all_gather", tp_group):
                        dist.all_gather(shards, raw.contiguous(), group=tp_group)
                    raw = gather_fn(shards)
                activations[hook_name] = _reshape_captured_activation(
                    raw,
                    batch_size=B,
                    seq_len=S,
                    total_tokens=total_tokens,
                    hook_name=hook_name,
                )
        else:
            # TP>1: MultiprocExecutor uses ZMQ to transfer apply_model returns.
            # Serialising large CUDA tensors as CPU bytes is ~800 ms for 128 MB.
            # Instead, _collect_and_pin uses ForkingPickler to create CUDA IPC
            # handle bytes (~64 B per tensor, μs over ZMQ).  Main process
            # reconstructs zero-copy CUDA tensors via pickle.loads, then calls
            # _release_pinned so the worker can free its GPU reference.
            rank0_only_hooks = tuple(
                hook_name
                for hook_name, _path, _extractor, _is_pre, gather_fn in hook_specs
                if gather_fn is None
            )
            collect = partial(
                _collect_and_pin_selective,
                rank0_only_hooks=rank0_only_hooks,
            )

            self.llm.apply_model(register)
            if install_memory_probe is not None:
                self.llm.apply_model(install_memory_probe)
            timeline_active = False
            if (
                memory_timeline_enabled
                and memory_timeline_current_step == memory_timeline_step
                and memory_timeline_path is not None
            ):
                start_timeline = partial(
                    _start_vllm_memory_timeline,
                    path=memory_timeline_path,
                    suffix_tp_rank=True,
                )
                self.llm.apply_model(start_timeline)
                timeline_active = True
            try:
                self.llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
            finally:
                if timeline_active:
                    self.llm.apply_model(_stop_vllm_memory_timeline)
                memory_records = collect_and_remove_memory_probe()
                ipc_bytes_per_rank: list[bytes] = self.llm.apply_model(collect)
            if memory_probe_enabled and memory_probe_path is not None:
                write_vllm_memory_records(
                    memory_probe_path,
                    records=memory_records,
                    step=memory_probe_step,
                    n_training_samples=memory_probe_samples,
                    rank=memory_probe_rank,
                    producer_idx=memory_probe_producer_idx,
                    vllm_tp_rank=memory_probe_tp_rank,
                )
                write_static_memory()

            try:
                activations = {}
                needs_all_ranks = any(gather_fn is not None for *_rest, gather_fn in hook_specs)
                rank0_caps = pickle.loads(ipc_bytes_per_rank[0])
                all_caps: list[dict[str, torch.Tensor]] | None = None
                if needs_all_ranks:
                    all_caps = [rank0_caps] + [
                        pickle.loads(b) for b in ipc_bytes_per_rank[1:]
                    ]

                for hook_name, _path, _extractor, _is_pre, gather_fn in hook_specs:
                    if gather_fn is None:
                        raw = rank0_caps[hook_name]
                    else:
                        assert all_caps is not None
                        shards = [c[hook_name] for c in all_caps if hook_name in c]
                        raw = gather_fn(shards)
                    activations[hook_name] = _reshape_captured_activation(
                        raw,
                        batch_size=B,
                        seq_len=S,
                        total_tokens=total_tokens,
                        hook_name=hook_name,
                    )
            finally:
                self.llm.apply_model(_release_pinned)

        return None, activations

    def to_tokens(
        self,
        input: str | list[str],
        prepend_bos: bool | None = USE_DEFAULT_VALUE,
        padding_side: Any = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        # Matches HookedProxyLM.to_tokens() contract (load_model.py:154).
        if prepend_bos is not False:
            raise ValueError(
                "Only works with prepend_bos=False, to match ActivationsStore usage"
            )
        if padding_side is not None:
            raise ValueError(
                "Only works with padding_side=None, to match ActivationsStore usage"
            )
        if truncate is not False:
            raise ValueError(
                "Only works with truncate=False, to match ActivationsStore usage"
            )
        if move_to_device is not False:
            raise ValueError(
                "Only works with move_to_device=False, to match ActivationsStore usage"
            )
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )["input_ids"]
        if hasattr(self.tokenizer, "add_bos_token") and self.tokenizer.add_bos_token:
            tokens = get_tokens_with_bos_removed(self.tokenizer, tokens)  # type: ignore[arg-type]
        return tokens  # type: ignore[return-value]
