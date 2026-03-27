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
import os
import pickle
import re
from functools import partial
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from transformer_lens.utils import USE_DEFAULT_VALUE, get_tokens_with_bos_removed
from transformers import PreTrainedTokenizerBase

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
        **llm_kwargs: Any,
    ) -> None:
        if LLM is None:
            raise ImportError(
                "vllm must be installed to use HookedVLLMModel. "
                "Install with `pip install vllm`."
            )
        self.tokenizer = tokenizer
        # enforce_eager=True: disables CUDA graphs so per-layer hooks fire.
        # VLLM_ACTIVATION_CAPTURE_MODE=1 (set at module load above) ensures
        # vLLM allocates only the minimal KV cache needed for one batch.
        llm_kwargs.setdefault("enforce_eager", True)
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

        self.llm = LLM(model_name, **llm_kwargs)
        self._is_external_launcher = (
            llm_kwargs.get("distributed_executor_backend") == "external_launcher"
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
        B, S = batch_tokens.shape
        arch_config = ARCH_CONFIGS[self._arch]
        total_tokens = B * S
        stop_at_layer: int | None = kwargs.get("stop_at_layer", None)

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

        if self._tp == 1 or self._is_external_launcher:
            # UniProcExecutor (tp=1) or external_launcher: apply_model runs
            # inline in the current process; results[0] is the local captures.
            #
            # For external_launcher with TP>1 all torchrun ranks call
            # generate() simultaneously — TP communication happens via NCCL.
            # Post-allreduce hooks already hold the full tensor on every rank.
            # Sharded hooks need dist.all_gather across the TP ranks.
            self.llm.apply_model(register)
            try:
                self.llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
            finally:
                results = self.llm.apply_model(_collect_and_cleanup)

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
                    dist.all_gather(shards, raw.contiguous(), group=tp_group)
                    raw = gather_fn(shards)
                activations[hook_name] = raw[:total_tokens].view(B, S, -1)
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
            try:
                self.llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
            finally:
                ipc_bytes_per_rank: list[bytes] = self.llm.apply_model(collect)

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
                    activations[hook_name] = raw[:total_tokens].view(B, S, -1)
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
