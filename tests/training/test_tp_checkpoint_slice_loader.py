"""``load_tp_sharded_state_dict`` reads exactly the local TP shard from a
full-tensor safetensors checkpoint and gathers back to a bit-identical
full tensor — same as ``safetensors.torch.load_file``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.tp_checkpoint import load_tp_sharded_state_dict
from tests.helpers import build_topk_sae_training_cfg


def _save_full_checkpoint(tmp_dir: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    cfg = build_topk_sae_training_cfg(d_in=32, d_sae=64, k=4, dtype="float32")
    sae = TopKTrainingSAE(cfg)
    sae.save_model(str(tmp_dir))
    weights_path = tmp_dir / "sae_weights.safetensors"
    full = load_file(weights_path)
    return weights_path, full


def _worker(
    rank: int, world_size: int, ckpt_path: str, port: int, result_list: list
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_group = dist.new_group(list(range(world_size)), backend="gloo")

    cfg = build_topk_sae_training_cfg(d_in=32, d_sae=64, k=4, dtype="float32")
    base_sae = TopKTrainingSAE(cfg)

    sd = load_tp_sharded_state_dict(ckpt_path, base_sae, tp_group)
    shapes = {k: tuple(v.shape) for k, v in sd.items()}

    # Gather W_dec/W_enc/b_enc to compare against full file.
    gathered: dict[str, torch.Tensor] = {}
    shard_dims = base_sae._tp_param_shard_dims()
    for name, t in sd.items():
        shard_dim = shard_dims.get(name)
        if shard_dim is None:
            gathered[name] = t
            continue
        parts = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(parts, t.contiguous(), group=tp_group)
        gathered[name] = torch.cat(parts, dim=shard_dim)

    result_list.append((rank, shapes, gathered))
    dist.destroy_process_group()


def test_slice_loader_local_shape_and_full_equality(tmp_path: Path):
    weights_path, full = _save_full_checkpoint(tmp_path)
    world_size = 2
    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _worker,
        args=(world_size, str(weights_path), 29900, result_list),
        nprocs=world_size,
        join=True,
    )
    results = sorted(list(result_list), key=lambda r: r[0])
    assert len(results) == world_size

    expected_shard_shapes = {
        "W_enc": (32, 32),  # d_in, d_sae // 2
        "W_dec": (32, 32),  # d_sae // 2, d_in
        "b_enc": (32,),
        "b_dec": (32,),  # replicated
    }
    for rank, shapes, gathered in results:
        for name, expected in expected_shard_shapes.items():
            assert shapes[name] == expected, (
                f"rank {rank}: {name} shard shape {shapes[name]} != {expected}"
            )
            torch.testing.assert_close(
                gathered[name], full[name], rtol=0, atol=0
            )
