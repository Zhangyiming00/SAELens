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

from sae_lens.vllm_model import (
    ARCH_CONFIGS,
    HookedVLLMModel,
    _gather_gate_up,
    _k_proj_extractor,
    _parse_hook_name,
    _q_proj_extractor,
    _v_proj_extractor,
)

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


def test_parse_hook_name_attn_submodule():
    hook_type, layer = _parse_hook_name("blocks.5.attn.hook_q")
    assert hook_type == "attn.hook_q"
    assert layer == 5


def test_parse_hook_name_mlp_submodule():
    hook_type, layer = _parse_hook_name("blocks.5.mlp.hook_pre")
    assert hook_type == "mlp.hook_pre"
    assert layer == 5


def test_parse_hook_name_invalid():
    with pytest.raises(ValueError):
        _parse_hook_name("invalid_name_no_hook_prefix")


# ---------------------------------------------------------------------------
# Unit tests: ARCH_CONFIGS coverage
# ---------------------------------------------------------------------------


def test_arch_configs_have_all_hook_types():
    expected = {
        "hook_embed",
        "hook_resid_pre",
        "hook_resid_mid",
        "hook_resid_post",
        "hook_mlp_out",
        "hook_attn_out",
        "attn.hook_q",
        "attn.hook_k",
        "attn.hook_v",
        "attn.hook_z",
        "mlp.hook_pre",
        "mlp.hook_post",
    }
    for arch, cfg in ARCH_CONFIGS.items():
        assert expected <= set(cfg.keys()), f"{arch} missing hooks: {expected - set(cfg.keys())}"


def test_llama_arch_registered():
    assert "LlamaForCausalLM" in ARCH_CONFIGS


# ---------------------------------------------------------------------------
# Unit tests: gather functions
# ---------------------------------------------------------------------------


def test_gather_gate_up_single_shard():
    # Single shard (TP=1): should just return the tensor as-is.
    t = torch.randn(4, 10)
    result = _gather_gate_up([t])
    assert result.shape == t.shape
    assert torch.allclose(result, t)


def test_gather_gate_up_two_shards():
    # TP=2: rank 0 has [g0, u0], rank 1 has [g1, u1].
    # Result should be [g0, g1, u0, u1].
    g0 = torch.ones(4, 3) * 1
    u0 = torch.ones(4, 3) * 2
    g1 = torch.ones(4, 3) * 3
    u1 = torch.ones(4, 3) * 4

    shard0 = torch.cat([g0, u0], dim=-1)  # (4, 6)
    shard1 = torch.cat([g1, u1], dim=-1)  # (4, 6)

    result = _gather_gate_up([shard0, shard1])
    assert result.shape == (4, 12)
    # gate portion = [g0, g1]
    assert torch.allclose(result[..., :6], torch.cat([g0, g1], dim=-1))
    # up portion = [u0, u1]
    assert torch.allclose(result[..., 6:], torch.cat([u0, u1], dim=-1))


class _DummyQKVModule:
    def __init__(self, num_heads: int, num_kv_heads: int, head_size: int, v_head_size: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.v_head_size = v_head_size


def test_qkv_extractors_split_pre_rope_outputs_correctly():
    module = _DummyQKVModule(num_heads=2, num_kv_heads=1, head_size=2, v_head_size=2)
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    k = torch.tensor([[5.0, 6.0]])
    v = torch.tensor([[7.0, 8.0]])
    qkv = torch.cat([q, k, v], dim=-1)
    output = (qkv, None)

    assert torch.equal(_q_proj_extractor(module, output), q)
    assert torch.equal(_k_proj_extractor(module, output), k)
    assert torch.equal(_v_proj_extractor(module, output), v)


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


# ---------------------------------------------------------------------------
# Integration tests: new hook points
# ---------------------------------------------------------------------------


def test_hook_resid_mid_shape_and_consistency(vllm_model, batch_tokens):
    """hook_resid_mid = hook_resid_pre + hook_attn_out (residual addition identity)."""
    B, S = batch_tokens.shape
    hooks = [
        f"blocks.{LAYER}.hook_resid_pre",
        f"blocks.{LAYER}.hook_attn_out",
        f"blocks.{LAYER}.hook_resid_mid",
    ]
    _, acts = vllm_model.run_with_cache(batch_tokens, names_filter=hooks)

    mid = acts[f"blocks.{LAYER}.hook_resid_mid"].float()
    pre = acts[f"blocks.{LAYER}.hook_resid_pre"].float()
    attn_out = acts[f"blocks.{LAYER}.hook_attn_out"].float()

    assert mid.shape == (B, S, pre.shape[-1])

    # Identity: hook_resid_mid = hook_resid_pre + hook_attn_out
    expected = pre + attn_out
    cos = torch.nn.functional.cosine_similarity(
        mid.view(-1, mid.shape[-1]), expected.view(-1, expected.shape[-1]), dim=-1
    )
    assert cos.min().item() > 0.999, f"hook_resid_mid consistency failed: cos={cos.min():.4f}"


def test_hook_resid_post_consistency(vllm_model, batch_tokens):
    """hook_resid_post = hook_resid_mid + hook_mlp_out."""
    B, S = batch_tokens.shape
    hooks = [
        f"blocks.{LAYER}.hook_resid_mid",
        f"blocks.{LAYER}.hook_mlp_out",
        f"blocks.{LAYER}.hook_resid_post",
    ]
    _, acts = vllm_model.run_with_cache(batch_tokens, names_filter=hooks)

    post = acts[f"blocks.{LAYER}.hook_resid_post"].float()
    mid = acts[f"blocks.{LAYER}.hook_resid_mid"].float()
    mlp_out = acts[f"blocks.{LAYER}.hook_mlp_out"].float()

    expected = mid + mlp_out
    cos = torch.nn.functional.cosine_similarity(
        post.view(-1, post.shape[-1]), expected.view(-1, expected.shape[-1]), dim=-1
    )
    assert cos.min().item() > 0.999, f"hook_resid_post consistency failed: cos={cos.min():.4f}"


def test_attn_hooks_shapes(vllm_model, batch_tokens):
    """Attention internal hooks produce correct shapes."""
    B, S = batch_tokens.shape
    hooks = [
        f"blocks.{LAYER}.attn.hook_q",
        f"blocks.{LAYER}.attn.hook_k",
        f"blocks.{LAYER}.attn.hook_v",
        f"blocks.{LAYER}.attn.hook_z",
    ]
    _, acts = vllm_model.run_with_cache(batch_tokens, names_filter=hooks)

    q = acts[f"blocks.{LAYER}.attn.hook_q"]
    k = acts[f"blocks.{LAYER}.attn.hook_k"]
    v = acts[f"blocks.{LAYER}.attn.hook_v"]
    z = acts[f"blocks.{LAYER}.attn.hook_z"]

    # All must be 3D (B, S, d)
    assert q.shape[:2] == (B, S)
    assert k.shape[:2] == (B, S)
    assert v.shape[:2] == (B, S)
    assert z.shape[:2] == (B, S)

    # hook_z has the same last dim as hook_q (both are query-sized)
    assert z.shape[-1] == q.shape[-1], "hook_z should have same d as hook_q"

    # GQA: k/v have fewer heads than q; their last dim must be smaller
    assert k.shape[-1] <= q.shape[-1], "GQA: k dim should be <= q dim"
    assert k.shape == v.shape, "k and v must have the same shape"


def test_mlp_hooks_shapes(vllm_model, batch_tokens):
    """MLP internal hooks produce correct shapes."""
    B, S = batch_tokens.shape
    hooks = [
        f"blocks.{LAYER}.mlp.hook_pre",
        f"blocks.{LAYER}.mlp.hook_post",
    ]
    _, acts = vllm_model.run_with_cache(batch_tokens, names_filter=hooks)

    pre = acts[f"blocks.{LAYER}.mlp.hook_pre"]
    post = acts[f"blocks.{LAYER}.mlp.hook_post"]

    assert pre.shape[:2] == (B, S)
    assert post.shape[:2] == (B, S)

    # hook_pre = gate only (first half of gate_up_proj) → same size as hook_post
    # hook_post = silu(gate) * up → intermediate_size
    assert pre.shape[-1] == post.shape[-1], (
        f"hook_pre.d={pre.shape[-1]} should equal hook_post.d={post.shape[-1]}"
    )
