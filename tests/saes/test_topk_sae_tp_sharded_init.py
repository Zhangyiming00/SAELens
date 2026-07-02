"""Bit-equivalence and RNG-endpoint tests for ``TopKTrainingSAE.from_config_sharded``.

The shard-init path must produce parameters that are byte-identical to the
legacy ``from_dict + shard_weights`` path under the same torch RNG state, and
must leave the CPU MT19937 in the same state as a single full-init would.
Both invariants run on CPU + Gloo so they don't need GPUs.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.saes.topk_sae import TopKTrainingSAE, _shard_init_topk_cpu
from sae_lens.util import temporary_seed
from tests.helpers import build_topk_sae_training_cfg


def _make_full_sae(cfg, seed: int) -> TopKTrainingSAE:
    with temporary_seed(seed):
        return TopKTrainingSAE(cfg)


def _save_reference(path: Path, cfg, seed: int) -> None:
    sae = _make_full_sae(cfg, seed)
    payload = {
        "W_dec": sae.W_dec.data.cpu().clone(),
        "W_enc": sae.W_enc.data.cpu().clone(),
        "b_enc": sae.b_enc.data.cpu().clone(),
        "b_dec": sae.b_dec.data.cpu().clone(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_reference(path: Path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _worker_bit_equal(
    rank: int,
    world_size: int,
    cfg_kwargs: dict,
    seed: int,
    ref_path: str,
    port: int,
    result_list: list,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_group = dist.new_group(list(range(world_size)), backend="gloo")

    cfg = build_topk_sae_training_cfg(**cfg_kwargs)
    with temporary_seed(seed):
        sae = TopKTrainingSAE.from_config_sharded(cfg, tp_group)

    ref = _load_reference(Path(ref_path))
    s = cfg.d_sae // world_size
    ok = {
        "W_dec": torch.equal(
            sae.W_dec.data,
            ref["W_dec"][rank * s : (rank + 1) * s, :].contiguous(),
        ),
        "W_enc": torch.equal(
            sae.W_enc.data,
            ref["W_enc"][:, rank * s : (rank + 1) * s].contiguous(),
        ),
        "b_enc": torch.equal(
            sae.b_enc.data,
            ref["b_enc"][rank * s : (rank + 1) * s].contiguous(),
        ),
        "b_dec": torch.equal(sae.b_dec.data, ref["b_dec"]),
    }
    shapes = {
        "W_dec": tuple(sae.W_dec.data.shape),
        "W_enc": tuple(sae.W_enc.data.shape),
        "b_enc": tuple(sae.b_enc.data.shape),
        "b_dec": tuple(sae.b_dec.data.shape),
    }
    result_list.append((rank, ok, shapes))
    dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("decoder_init_norm", [None, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_from_config_sharded_bit_equal_to_full_init_then_shard(
    tmp_path: Path, tp_size: int, decoder_init_norm: float | None, dtype: str
):
    d_in, d_sae = 32, 64
    seed = 42
    cfg_kwargs = dict(
        d_in=d_in,
        d_sae=d_sae,
        k=4,
        dtype=dtype,
        decoder_init_norm=decoder_init_norm,
    )
    cfg = build_topk_sae_training_cfg(**cfg_kwargs)
    ref_path = tmp_path / "ref.pkl"
    _save_reference(ref_path, cfg, seed)

    port = 29800 + tp_size * 10 + (1 if decoder_init_norm else 0) + (
        2 if dtype == "bfloat16" else 0
    )
    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _worker_bit_equal,
        args=(tp_size, cfg_kwargs, seed, str(ref_path), port, result_list),
        nprocs=tp_size,
        join=True,
    )
    results = sorted(list(result_list), key=lambda r: r[0])
    assert len(results) == tp_size
    s = d_sae // tp_size
    for rank, ok, shapes in results:
        assert all(ok.values()), f"rank {rank}: bit-equality failed: {ok}"
        assert shapes == {
            "W_dec": (s, d_in),
            "W_enc": (d_in, s),
            "b_enc": (s,),
            "b_dec": (d_in,),
        }, f"rank {rank}: wrong shapes: {shapes}"


def test_shard_init_rng_endpoint_matches_full_init():
    """After shard-init, the CPU RNG must be at the same state as after a
    full-init, so subsequent code in the same temporary_seed scope is unaffected.
    """
    d_in, d_sae = 64, 128
    cfg = build_topk_sae_training_cfg(d_in=d_in, d_sae=d_sae, k=4, dtype="float32")
    seed = 7

    with temporary_seed(seed):
        TopKTrainingSAE(cfg)
        full_tail = torch.rand(8, dtype=torch.float32)

    for tp_size in (2, 4, 8):
        for tp_rank in range(tp_size):
            with temporary_seed(seed):
                _shard_init_topk_cpu(cfg, tp_size, tp_rank)
                shard_tail = torch.rand(8, dtype=torch.float32)
            torch.testing.assert_close(shard_tail, full_tail, rtol=0, atol=0)
