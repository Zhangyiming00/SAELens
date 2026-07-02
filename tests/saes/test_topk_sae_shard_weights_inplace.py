"""``TopKTrainingSAE.shard_weights`` must release the full-size storage of
W_dec, W_enc, and b_enc so the legacy "full init then shard" path stops
leaving 2x parameter memory live after the call.
"""

from __future__ import annotations

import os

import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import build_topk_sae_training_cfg


def _worker(rank: int, world_size: int, port: int, result_list: list) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_group = dist.new_group(list(range(world_size)), backend="gloo")

    d_in, d_sae = 32, 64
    cfg = build_topk_sae_training_cfg(d_in=d_in, d_sae=d_sae, k=4, dtype="float32")
    sae = TopKTrainingSAE(cfg)

    pre_W_dec_storage = sae.W_dec.data.untyped_storage().nbytes()
    pre_W_dec_ptr = sae.W_dec.data.data_ptr()

    sae.shard_weights(tp_group)

    post_W_dec_storage = sae.W_dec.data.untyped_storage().nbytes()
    post_W_dec_ptr = sae.W_dec.data.data_ptr()
    expected_shard_bytes = (d_sae // world_size) * d_in * sae.W_dec.data.element_size()

    result_list.append(
        (
            rank,
            pre_W_dec_storage,
            post_W_dec_storage,
            expected_shard_bytes,
            pre_W_dec_ptr != post_W_dec_ptr,
            tuple(sae.W_dec.data.shape),
            tuple(sae.W_enc.data.shape),
            tuple(sae.b_enc.data.shape),
            tuple(sae.b_dec.data.shape),
            sae.W_dec.grad,
            sae.W_enc.grad,
            sae.b_enc.grad,
        )
    )
    dist.destroy_process_group()


def test_shard_weights_releases_full_storage():
    world_size = 2
    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(_worker, args=(world_size, 29870, result_list), nprocs=world_size, join=True)
    results = sorted(list(result_list), key=lambda r: r[0])
    assert len(results) == world_size

    for (
        rank,
        pre_storage,
        post_storage,
        expected_shard_bytes,
        ptr_changed,
        sh_W_dec,
        sh_W_enc,
        sh_b_enc,
        sh_b_dec,
        g_W_dec,
        g_W_enc,
        g_b_enc,
    ) in results:
        assert pre_storage > post_storage, (
            f"rank {rank}: shard_weights must release full storage; "
            f"pre={pre_storage} post={post_storage}"
        )
        assert post_storage == expected_shard_bytes, (
            f"rank {rank}: post-shard W_dec storage {post_storage} "
            f"!= expected shard bytes {expected_shard_bytes}"
        )
        assert ptr_changed, f"rank {rank}: storage pointer must change after shard"
        assert sh_W_dec == (32, 32)  # d_sae // 2, d_in
        assert sh_W_enc == (32, 32)  # d_in, d_sae // 2
        assert sh_b_enc == (32,)
        assert sh_b_dec == (32,)
        assert g_W_dec is None and g_W_enc is None and g_b_enc is None
