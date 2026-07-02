"""End-to-end test: build a full TopKTrainingSAE, save it, and verify that
each TP rank can reconstruct it via ``from_config_sharded`` +
``load_weights_from_checkpoint``. The TP forward output (with allgather/
allreduce internally) must match the single-process reference forward.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAE
from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import build_jumprelu_sae_training_cfg, build_topk_sae_training_cfg


def _save_full_and_reference(tmp_dir: Path, dtype: str) -> tuple[Path, torch.Tensor, torch.Tensor]:
    cfg = build_topk_sae_training_cfg(d_in=32, d_sae=64, k=8, dtype=dtype)
    sae = TopKTrainingSAE(cfg)
    # Make params non-trivial.
    with torch.no_grad():
        for p in sae.parameters():
            p.data.normal_()
    sae.save_model(str(tmp_dir))

    torch.manual_seed(123)
    x = torch.randn(4, cfg.d_in, dtype=sae.dtype)
    with torch.no_grad():
        y_ref = sae(x)
    return tmp_dir, x, y_ref


def _worker(
    rank: int,
    world_size: int,
    ckpt_dir: str,
    cfg_kwargs: dict,
    x_path: str,
    y_ref_path: str,
    port: int,
    result_list: list,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_group = dist.new_group(list(range(world_size)), backend="gloo")

    cfg = build_topk_sae_training_cfg(**cfg_kwargs)
    sae = TopKTrainingSAE.from_config_sharded(cfg, tp_group)

    # Local W_dec/W_enc/b_enc must be shard-shaped before load.
    pre_shapes = (
        tuple(sae.W_dec.data.shape),
        tuple(sae.W_enc.data.shape),
        tuple(sae.b_enc.data.shape),
    )

    sae.load_weights_from_checkpoint(ckpt_dir)

    post_shapes = (
        tuple(sae.W_dec.data.shape),
        tuple(sae.W_enc.data.shape),
        tuple(sae.b_enc.data.shape),
    )

    x = torch.load(x_path)
    y_ref = torch.load(y_ref_path)
    with torch.no_grad():
        y = sae(x)

    result_list.append(
        (rank, pre_shapes, post_shapes, torch.allclose(y, y_ref, rtol=1e-5, atol=1e-5))
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_load_weights_tp_round_trip(tmp_path: Path, tp_size: int):
    cfg_kwargs = dict(d_in=32, d_sae=64, k=8, dtype="float32")
    ckpt_dir, x, y_ref = _save_full_and_reference(tmp_path, dtype="float32")
    x_path = tmp_path / "x.pt"
    y_path = tmp_path / "y.pt"
    torch.save(x, x_path)
    torch.save(y_ref, y_path)

    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _worker,
        args=(
            tp_size,
            str(ckpt_dir),
            cfg_kwargs,
            str(x_path),
            str(y_path),
            29950 + tp_size,
            result_list,
        ),
        nprocs=tp_size,
        join=True,
    )
    results = sorted(list(result_list), key=lambda r: r[0])
    assert len(results) == tp_size

    s = cfg_kwargs["d_sae"] // tp_size
    expected_pre = ((s, 32), (32, s), (s,))
    for rank, pre, post, allclose in results:
        assert pre == expected_pre, f"rank {rank}: pre-load shapes {pre} != {expected_pre}"
        assert post == expected_pre, f"rank {rank}: post-load shapes {post} != {expected_pre}"
        assert allclose, f"rank {rank}: TP forward did not match full reference"


def test_load_weights_tp_rejects_non_allowlisted_arch(tmp_path: Path):
    """``_assert_tp_slice_load_safe`` should refuse for non-TopK training SAEs."""
    cfg = build_jumprelu_sae_training_cfg(d_in=8, d_sae=16, dtype="float32")
    sae = JumpReLUTrainingSAE(cfg)
    sae.save_model(str(tmp_path))

    # Force the TP path by stubbing _tp_group as if a real TP=2 group existed.
    class _StubGroup:
        pass

    orig_initialized = dist.is_initialized
    orig_world_size = dist.get_world_size
    dist.is_initialized = lambda: True  # type: ignore[assignment]
    dist.get_world_size = lambda _group=None: 2  # type: ignore[assignment]
    try:
        sae._tp_group = _StubGroup()  # type: ignore[assignment]
        with pytest.raises(NotImplementedError, match="TP slice load is only supported"):
            sae.load_weights_from_checkpoint(str(tmp_path))
    finally:
        dist.is_initialized = orig_initialized  # type: ignore[assignment]
        dist.get_world_size = orig_world_size  # type: ignore[assignment]
        sae._tp_group = None
