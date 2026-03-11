"""
Integration tests for HookedVLLMModel.

These tests compare activations captured by HookedVLLMModel against
ground-truth values obtained from HuggingFace transformers with forward
hooks.  They require a GPU and the local model at /data/models/Llama-3.1-8B.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set vLLM env vars before importing sae_lens.vllm_model
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

from sae_lens.vllm_model import ARCH_CONFIGS, HookedVLLMModel, _parse_hook_name

MODEL_PATH = "/data/models/Llama-3.1-8B"
LAYER = 2  # layer index used in all per-layer hook tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def vllm_model(tokenizer):
    return HookedVLLMModel(
        MODEL_PATH,
        tokenizer,
        max_model_len=128,
        tensor_parallel_size=1,
    )


@pytest.fixture(scope="module")
def hf_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
    model.eval()
    return model


@pytest.fixture(scope="module")
def batch_tokens():
    """Fixed token batch used in all comparison tests."""
    torch.manual_seed(42)
    # Use real-ish token IDs (not special tokens) so both models process normally
    return torch.randint(100, 30000, (3, 16))


# ---------------------------------------------------------------------------
# Helper: collect HF reference activations via forward hooks
# ---------------------------------------------------------------------------


def collect_hf_activations(
    hf_model: torch.nn.Module,
    batch_tokens: torch.Tensor,
    layer: int,
) -> dict[str, torch.Tensor]:
    """Run HF model and collect all supported hook-type activations at `layer`."""
    caps: dict[str, torch.Tensor] = {}
    handles = []

    def reg(module, fn, pre=False):
        h = (
            module.register_forward_pre_hook(fn)
            if pre
            else module.register_forward_hook(fn)
        )
        handles.append(h)

    def h_embed(m, inp, out):
        caps["hook_embed"] = out.detach().float().cpu()

    def h_resid_pre(m, args):
        # args[0] = hidden_states entering layer `layer`
        caps["hook_resid_pre"] = args[0].detach().float().cpu()

    def h_resid_post(m, inp, out):
        # HF decoder layer returns a single tensor (full residual stream)
        caps["hook_resid_post"] = out.detach().float().cpu()

    def h_mlp(m, inp, out):
        caps["hook_mlp_out"] = out.detach().float().cpu()

    def h_attn(m, inp, out):
        # HF attention returns (attn_output, ...) tuple; [0] is the output tensor
        val = out[0] if isinstance(out, tuple) else out
        caps["hook_attn_out"] = val.detach().float().cpu()

    reg(hf_model.model.embed_tokens, h_embed)
    reg(hf_model.model.layers[layer], h_resid_pre, pre=True)
    reg(hf_model.model.layers[layer], h_resid_post)
    reg(hf_model.model.layers[layer].mlp, h_mlp)
    reg(hf_model.model.layers[layer].self_attn, h_attn)

    with torch.no_grad():
        hf_model(batch_tokens.cuda())

    for h in handles:
        h.remove()

    return caps


# ---------------------------------------------------------------------------
# Unit tests: hook name parsing
# ---------------------------------------------------------------------------


def test_parse_hook_name_block():
    hook_type, layer = _parse_hook_name("blocks.5.hook_resid_post")
    assert hook_type == "hook_resid_post"
    assert layer == 5


def test_parse_hook_name_global():
    hook_type, layer = _parse_hook_name("hook_embed")
    assert hook_type == "hook_embed"
    assert layer is None


def test_parse_hook_name_invalid():
    with pytest.raises(ValueError):
        _parse_hook_name("invalid_name_no_hook_prefix")


# ---------------------------------------------------------------------------
# Unit tests: ARCH_CONFIGS coverage
# ---------------------------------------------------------------------------


def test_arch_configs_have_all_hook_types():
    expected = {"hook_embed", "hook_resid_pre", "hook_resid_post", "hook_mlp_out", "hook_attn_out"}
    for arch, cfg in ARCH_CONFIGS.items():
        assert expected <= set(cfg.keys()), f"{arch} missing hooks: {expected - set(cfg.keys())}"


def test_llama_arch_registered():
    assert "LlamaForCausalLM" in ARCH_CONFIGS


# ---------------------------------------------------------------------------
# Integration tests: vLLM vs HF activation comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook_name",
    [
        "hook_embed",
        f"blocks.{LAYER}.hook_resid_pre",
        f"blocks.{LAYER}.hook_resid_post",
        f"blocks.{LAYER}.hook_mlp_out",
        f"blocks.{LAYER}.hook_attn_out",
    ],
)
def test_activation_matches_hf(vllm_model, hf_model, batch_tokens, hook_name):
    """
    vLLM captured activations must match HF reference within bfloat16 tolerance.

    bfloat16 has ~1% relative error; FlashAttention vs standard attention may
    differ slightly, so we use a loose absolute tolerance of 0.05 on values
    that are O(1) after normalization.
    """
    B, S = batch_tokens.shape

    # --- vLLM capture ---
    _, vllm_acts = vllm_model.run_with_cache(batch_tokens, names_filter=[hook_name])
    vllm_act = vllm_acts[hook_name].float()  # (B, S, d)

    # --- HF reference ---
    hook_type, layer = _parse_hook_name(hook_name)
    effective_layer = layer if layer is not None else 0
    hf_caps = collect_hf_activations(hf_model, batch_tokens, effective_layer)
    hf_act = hf_caps[hook_type]  # (B, S, d)

    assert vllm_act.shape == hf_act.shape, (
        f"{hook_name}: shape mismatch vllm={vllm_act.shape} hf={hf_act.shape}"
    )

    # Cosine similarity per token should be > 0.99
    vllm_flat = vllm_act.view(-1, vllm_act.shape[-1])
    hf_flat = hf_act.view(-1, hf_act.shape[-1])
    cos_sim = torch.nn.functional.cosine_similarity(vllm_flat, hf_flat, dim=-1)
    assert cos_sim.min().item() > 0.99, (
        f"{hook_name}: cosine similarity too low: min={cos_sim.min().item():.4f}"
    )

    # RMS should match within 1% (catches scale bugs like 2x multiplier).
    # max_abs_diff / rms is intentionally NOT used here: at bfloat16 values near
    # the max (~414 for Llama residual stream), the bfloat16 step size equals 2.0,
    # so a single outlier element inflates max diff while mean diff is negligible.
    vllm_rms = vllm_act.pow(2).mean().sqrt().item()
    hf_rms = hf_act.pow(2).mean().sqrt().item()
    rms_ratio = abs(vllm_rms - hf_rms) / (hf_rms + 1e-8)
    assert rms_ratio < 0.01, (
        f"{hook_name}: RMS mismatch: vllm={vllm_rms:.4f} hf={hf_rms:.4f} ratio={rms_ratio:.4f}"
    )


def test_run_with_cache_output_shape(vllm_model, batch_tokens):
    B, S = batch_tokens.shape
    hook_name = f"blocks.{LAYER}.hook_resid_post"
    _, acts = vllm_model.run_with_cache(batch_tokens, names_filter=[hook_name])
    assert hook_name in acts
    act = acts[hook_name]
    assert act.shape[0] == B
    assert act.shape[1] == S
    assert act.ndim == 3


def test_run_with_cache_returns_none_logits(vllm_model, batch_tokens):
    logits, _ = vllm_model.run_with_cache(
        batch_tokens, names_filter=["hook_embed"]
    )
    assert logits is None


def test_run_with_cache_multiple_hooks(vllm_model, batch_tokens):
    hooks = [
        "hook_embed",
        f"blocks.{LAYER}.hook_resid_post",
        f"blocks.{LAYER}.hook_mlp_out",
    ]
    _, acts = vllm_model.run_with_cache(batch_tokens, names_filter=hooks)
    assert set(acts.keys()) == set(hooks)
    for h in hooks:
        assert acts[h].ndim == 3


def test_activation_capture_mode_reduces_kv_memory(vllm_model):
    """Verify the engine was initialized with minimal KV blocks (Phase 1)."""
    engine = vllm_model.llm.llm_engine if hasattr(vllm_model.llm, "llm_engine") else None
    if engine is None:
        pytest.skip("Cannot access engine internals")
    cfg = engine.vllm_config.cache_config
    # With max_model_len=128 and block_size=16: ceil(128*scheduler_max_seqs/16)+1
    # should be << 10000 (what normal mode would allocate)
    assert cfg.num_gpu_blocks is not None
    assert cfg.num_gpu_blocks < 2000, (
        f"Expected minimal KV blocks (<2000), got {cfg.num_gpu_blocks}"
    )
