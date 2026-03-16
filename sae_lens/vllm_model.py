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

import os
import re
from functools import partial
from typing import Any, Callable

import torch
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
    return torch.cat(shards, dim=-1)


def _gather_gate_up(shards: list[torch.Tensor]) -> torch.Tensor:
    """
    Gather MergedColumnParallelLinear output.

    Each TP rank produces [gate_local, up_local] concatenated.  After simple
    cat we'd get [g0, u0, g1, u1, ...] which is wrong; we need [gate_all, up_all].
    """
    half = shards[0].shape[-1] // 2
    gate = torch.cat([s[..., :half] for s in shards], dim=-1)
    up = torch.cat([s[..., half:] for s in shards], dim=-1)
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
    model._sae_captures: dict[str, torch.Tensor] = {}  # type: ignore[attr-defined]
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
                    # Only capture prefill pass: shape (B*S, d).
                    # Decode pass has shape (B, d).
                    if act.shape[0] == total_tokens:
                        model._sae_captures[name] = act.detach().cpu()  # type: ignore[attr-defined]

                return hook_fn

            handle = module.register_forward_pre_hook(make_pre_hook(hook_name, extractor))
        else:

            def make_post_hook(
                name: str, ext: Callable
            ) -> Callable[[nn.Module, Any, Any], None]:
                def hook_fn(m: nn.Module, inp: Any, out: Any) -> None:
                    act = ext(m, out)
                    if act.shape[0] == total_tokens:
                        model._sae_captures[name] = act.detach().cpu()  # type: ignore[attr-defined]

                return hook_fn

            handle = module.register_forward_hook(make_post_hook(hook_name, extractor))

        model._sae_handles.append(handle)  # type: ignore[attr-defined]


def _collect_and_cleanup(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect captured activations and remove all hooks from the model."""
    captures = dict(model._sae_captures)  # type: ignore[attr-defined]
    for handle in model._sae_handles:  # type: ignore[attr-defined]
        handle.remove()
    del model._sae_captures, model._sae_handles  # type: ignore[attr-defined]
    if hasattr(model.model, "_sae_stop_at_layer"):
        del model.model._sae_stop_at_layer  # type: ignore[attr-defined]
    return captures


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
        self.llm = LLM(model_name, **llm_kwargs)

        arch: str = self.llm.apply_model(_get_arch_name)[0]
        if arch not in ARCH_CONFIGS:
            raise ValueError(
                f"Architecture {arch!r} is not supported by HookedVLLMModel. "
                f"Supported: {sorted(ARCH_CONFIGS)}"
            )
        self._arch = arch
        # ActivationsStore calls _get_model_device(model) which falls back to
        # next(model.parameters()).device.  vLLM always runs on CUDA so we
        # expose a device property to satisfy that check without needing
        # model.parameters().
        self.device = torch.device("cuda")

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

        # Register hooks inside the worker process(es).
        self.llm.apply_model(
            partial(
                _register_hooks,
                hook_specs=hook_specs,
                total_tokens=total_tokens,
                stop_at_layer=stop_at_layer,
            )
        )

        # Run inference.  max_tokens=1 keeps decode overhead minimal.
        prompts = [{"prompt_token_ids": batch_tokens[i].tolist()} for i in range(B)]
        try:
            self.llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
        finally:
            results = self.llm.apply_model(_collect_and_cleanup)

        # Build per-hook activations.
        # - gather_fn=None: post-allreduce; rank-0's capture is already full.
        # - gather_fn callable: sharded; call gather_fn(shards) to reconstruct.
        activations: dict[str, torch.Tensor] = {}
        for hook_name, _path, _extractor, _is_pre, gather_fn in hook_specs:
            if gather_fn is None or len(results) == 1:
                raw = results[0][hook_name]
            else:
                shards = [r[hook_name] for r in results if hook_name in r]
                raw = gather_fn(shards)
            activations[hook_name] = raw.view(B, S, -1)
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
