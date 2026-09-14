"""CPU layout/checkpoint tests. A singleton fake group performs no communication.

Real distributed correctness belongs to the NCCL acceptance tests. TP1 uses
the actual Megatron modules; every TP collective has a size-one early return.
"""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from sae_lens.megatron_tp import _shard_init_topk_cpu, require_megatron_core
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.tp_checkpoint import load_tp_sharded_state_dict

REFERENCE = Path(__file__).resolve().parents[1] / "native_reference"


@pytest.fixture
def sae():
    pytest.importorskip("megatron.core")
    import torch.testing._internal.distributed.fake_pg  # noqa: F401

    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield MegatronTopKSAE(
            TopKTrainingSAEConfig(d_in=16, d_sae=32, k=4, decoder_init_norm=0.7),
            tp_group=dist.group.WORLD,
        )
    finally:
        dist.destroy_process_group()


def test_torchrun_tp1_initializes_worker_group(monkeypatch):
    from types import SimpleNamespace

    from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

    runner = object.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = SimpleNamespace(device="cpu")
    for key, value in {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(key, value)
    with (
        patch.object(dist, "is_initialized", return_value=False),
        patch.object(dist, "init_process_group") as initialize,
        patch.object(dist, "get_world_size", return_value=1),
    ):
        runner._resolve_megatron_tp_group(None)
        initialize.assert_called_once_with(backend="gloo")


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_initialization_rng_endpoint(tp_size, dtype):
    cfg = TopKTrainingSAEConfig(d_in=16, d_sae=32, k=4, dtype=dtype)
    torch.manual_seed(7291)
    _shard_init_topk_cpu(cfg, 1, 0)
    expected = torch.get_rng_state().clone()
    for rank in range(tp_size):
        torch.manual_seed(7291)
        _shard_init_topk_cpu(cfg, tp_size, rank)
        torch.testing.assert_close(torch.get_rng_state(), expected, atol=0, rtol=0)


def test_adam_safetensors_loads_all_moments_in_runtime_layout(sae, tmp_path):
    from sae_lens.training.multi_sae_trainer import (
        _load_hook_optimizer_state_safetensors,
        _save_hook_optimizer_state_safetensors,
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    sae(torch.randn(11, 16)).square().sum().backward()
    optimizer.step()
    expected = {n: copy.deepcopy(optimizer.state[p]) for n, p in sae.named_parameters()}
    saved = copy.deepcopy(expected)
    sae.process_named_optimizer_state_for_saving(saved)
    _save_hook_optimizer_state_safetensors(tmp_path, saved)
    loaded = _load_hook_optimizer_state_safetensors(tmp_path, sae, sae._tp_group)
    assert set(loaded) == set(expected)
    for name, state in loaded.items():
        for key, value in state.items():
            torch.testing.assert_close(value, expected[name][key], rtol=0, atol=0)


def test_missing_megatron_has_install_instructions():
    require_megatron_core.cache_clear()
    try:
        with (
            patch(
                "sae_lens.megatron_tp.import_module", side_effect=ModuleNotFoundError
            ),
            pytest.raises(ImportError, match=r"sae-lens\[megatron\]"),
        ):
            require_megatron_core()
    finally:
        require_megatron_core.cache_clear()


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_optimizer_slice_loader_reads_only_local_native_moments(
    tmp_path, tp_size, tp_rank
):
    """No collectives: fake ranks exercise real file slicing and allocation."""
    pytest.importorskip("megatron.core")
    import torch.testing._internal.distributed.fake_pg  # noqa: F401

    from sae_lens.training.multi_sae_trainer import (
        _load_hook_optimizer_state_safetensors,
        _save_hook_optimizer_state_safetensors,
    )

    golden = load_file(REFERENCE / "golden.safetensors")
    state = {
        name: {
            key: golden[f"case3.step2.adam.{name}.{key}"]
            for key in ("step", "exp_avg", "exp_avg_sq")
        }
        for name in ("W_enc", "W_dec", "b_enc", "b_dec")
    }
    _save_hook_optimizer_state_safetensors(tmp_path, state)
    dist.init_process_group(
        "fake", store=dist.HashStore(), rank=tp_rank, world_size=tp_size
    )
    try:
        sae = MegatronTopKSAE(
            TopKTrainingSAEConfig(d_in=16, d_sae=32, k=4), tp_group=dist.group.WORLD
        )
        loaded = _load_hook_optimizer_state_safetensors(tmp_path, sae, dist.group.WORLD)
        width = 32 // tp_size
        sl = slice(tp_rank * width, (tp_rank + 1) * width)
        for key in ("exp_avg", "exp_avg_sq"):
            expected = {
                "encoder.weight": state["W_enc"][key][:, sl].T,
                "decoder.weight": state["W_dec"][key][sl].T,
                "encoder.bias": state["b_enc"][key][sl],
                "b_dec": state["b_dec"][key],
            }
            assert set(loaded) == set(expected)
            for name, value in expected.items():
                torch.testing.assert_close(loaded[name][key], value, rtol=0, atol=0)
                assert loaded[name][key].is_contiguous()
        for param in sae.parameters():
            assert (
                param.untyped_storage().nbytes() == param.numel() * param.element_size()
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_sharded_initialization_matches_pinned_native_weights(tp_size):
    inputs = load_file(REFERENCE / "inputs.safetensors")
    cfg = TopKTrainingSAEConfig(d_in=16, d_sae=32, k=4, decoder_init_norm=0.7)
    for rank in range(tp_size):
        torch.manual_seed(7291)
        dec, enc, enc_bias, dec_bias = _shard_init_topk_cpu(cfg, tp_size, rank)
        width = 32 // tp_size
        sl = slice(rank * width, (rank + 1) * width)
        torch.testing.assert_close(dec, inputs["native_init.W_dec"][sl], atol=0, rtol=0)
        torch.testing.assert_close(
            enc, inputs["native_init.W_enc"][:, sl], atol=0, rtol=0
        )
        torch.testing.assert_close(
            enc_bias, inputs["native_init.b_enc"][sl], atol=0, rtol=0
        )
        torch.testing.assert_close(
            dec_bias, inputs["native_init.b_dec"], atol=0, rtol=0
        )


def test_native_import_is_explicit_strict_and_does_not_alias(sae, tmp_path):
    inputs = load_file(REFERENCE / "inputs.safetensors")
    state = {
        k.removeprefix("initial."): v
        for k, v in inputs.items()
        if k.startswith("initial.")
    }
    sae.import_saelens_state_dict(state)
    for name, value in sae.export_saelens_state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert sae.encoder.weight.is_contiguous()
    assert sae.decoder.weight.is_contiguous()
    assert (
        sae.encoder.weight.untyped_storage().nbytes() == sae.encoder.weight.numel() * 4
    )
    malformed = copy.deepcopy(state)
    malformed["W_enc"] = malformed["W_enc"].T
    with pytest.raises(ValueError, match="W_enc: expected shape"):
        sae.import_saelens_state_dict(malformed)
    with pytest.raises(ValueError, match="Unexpected"):
        sae.import_saelens_state_dict({**state, "extra": torch.ones(1)})
    path = tmp_path / "weights.safetensors"
    save_file(state, path)
    loaded = load_tp_sharded_state_dict(path, sae, dist.group.WORLD)
    assert set(loaded) == set(dict(sae.named_parameters()))
    sae.load_state_dict(loaded)
    state["W_enc"].zero_()
    assert sae.encoder.weight.count_nonzero() > 0


def test_sparse_and_dense_megatron_decode_match(sae):
    x = torch.linspace(-1.0, 2.0, 176).reshape(11, 16)
    acts = sae.encode(x)
    dense = sae.decode(acts)
    grads = torch.autograd.grad(dense.square().sum(), tuple(sae.parameters()))
    sparse = sae.decode(sae.encode(x).to_sparse())
    sparse_grads = torch.autograd.grad(sparse.square().sum(), tuple(sae.parameters()))
    torch.testing.assert_close(dense, sparse)
    for left, right in zip(grads, sparse_grads):
        torch.testing.assert_close(left, right)


def test_runner_constructs_megatron_for_tp1_and_checkpoint(sae, tmp_path):
    from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

    class Config:
        def get_training_sae_cfg_dict(self):
            return sae.cfg.to_dict()

    runner = object.__new__(LanguageModelSAETrainingRunner)
    runner.cfg = Config()
    checkpoint = tmp_path / "model"
    sae.save_model(checkpoint)
    model = runner._create_training_sae(
        seed=198, tp_group=None, device="cpu", from_pretrained_path=str(checkpoint)
    )
    assert isinstance(model, MegatronTopKSAE)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, sae.get_parameter(name), atol=0, rtol=0)
