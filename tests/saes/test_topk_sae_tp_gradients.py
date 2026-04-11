"""Tests that TopKTrainingSAE gradient sync is correct under TP=2.

Strategy: run the same forward+backward on identical inputs and params,
once with TP=1 (reference) and once with TP=2 (two real processes via
mp.spawn + Gloo). After sync_tensor_parallel_gradients(), all gradients
on every TP rank must match the TP=1 reference exactly.

No GPU required — Gloo backend runs on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import build_topk_sae_training_cfg, random_params


def _make_sae(d_in: int, d_sae: int, k: int, apply_b_dec_to_input: bool) -> TopKTrainingSAE:
    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        apply_b_dec_to_input=apply_b_dec_to_input,
        rescale_acts_by_decoder_norm=False,
    )
    return TopKTrainingSAE(cfg)


def _forward_backward(sae: TopKTrainingSAE, x: torch.Tensor, d_sae: int) -> None:
    out = sae.training_forward_pass(
        TrainStepInput(
            sae_in=x,
            dead_neuron_mask=torch.zeros(d_sae, dtype=torch.bool),
            coefficients={},
            n_training_steps=0,
            is_logging_step=False,
        )
    )
    out.loss.backward()


def _grads_tp1(
    d_in: int,
    d_sae: int,
    k: int,
    apply_b_dec_to_input: bool,
    state_dict: dict,
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sae = _make_sae(d_in, d_sae, k, apply_b_dec_to_input)
    sae.load_state_dict(state_dict)
    _forward_backward(sae, x, d_sae)
    return {name: param.grad.clone() for name, param in sae.named_parameters() if param.grad is not None}


def _worker_tp2(
    rank: int,
    world_size: int,
    d_in: int,
    d_sae: int,
    k: int,
    apply_b_dec_to_input: bool,
    state_dict: dict,
    x: torch.Tensor,
    result_list: list,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_group = dist.new_group(list(range(world_size)), backend="gloo")

    sae = _make_sae(d_in, d_sae, k, apply_b_dec_to_input)
    sae.load_state_dict(state_dict)
    sae.shard_weights(tp_group)

    _forward_backward(sae, x, d_sae)
    sae.sync_tensor_parallel_gradients()

    # Gather sharded grads back to full shape for comparison with TP=1
    shard_dims = sae._tp_param_shard_dims()
    full_grads: dict[str, torch.Tensor] = {}
    for name, param in sae.named_parameters():
        if param.grad is None:
            continue
        shard_dim = shard_dims.get(name)
        if shard_dim is not None:
            tp_size = dist.get_world_size(tp_group)
            parts = [torch.zeros_like(param.grad) for _ in range(tp_size)]
            dist.all_gather(parts, param.grad.contiguous(), group=tp_group)
            full_grads[name] = torch.cat(parts, dim=shard_dim)
        else:
            full_grads[name] = param.grad.clone()

    dist.destroy_process_group()
    result_list.append((rank, full_grads))


def _run_tp2(
    d_in: int,
    d_sae: int,
    k: int,
    apply_b_dec_to_input: bool,
    state_dict: dict,
    x: torch.Tensor,
    port: int,
) -> list[dict[str, torch.Tensor]]:
    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _worker_tp2,
        args=(2, d_in, d_sae, k, apply_b_dec_to_input, state_dict, x, result_list, port),
        nprocs=2,
        join=True,
    )
    results = sorted(result_list, key=lambda t: t[0])
    return [grads for _, grads in results]


def _worker_tp2_dp2(
    rank: int,
    world_size: int,
    d_in: int,
    d_sae: int,
    k: int,
    state_dict: dict,
    x_per_dp_rank: list[torch.Tensor],
    result_list: list,
    port: int,
) -> None:
    """Worker for TP=2, DP=2 (4 ranks total).

    Rank layout:
        rank 0: tp_rank=0, dp_rank=0
        rank 1: tp_rank=1, dp_rank=0
        rank 2: tp_rank=0, dp_rank=1
        rank 3: tp_rank=1, dp_rank=1

    tp_group:  [0,1] and [2,3]
    dp_group:  [0,2] and [1,3]
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    tp_size = 2
    dp_size = 2
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    tp_ranks = [dp_rank * tp_size + i for i in range(tp_size)]
    dp_ranks = [i * tp_size + tp_rank for i in range(dp_size)]
    tp_group = dist.new_group(tp_ranks, backend="gloo")
    dp_group = dist.new_group(dp_ranks, backend="gloo")

    sae = _make_sae(d_in, d_sae, k, apply_b_dec_to_input=True)
    sae.load_state_dict(state_dict)
    sae.shard_weights(tp_group)

    x = x_per_dp_rank[dp_rank]
    _forward_backward(sae, x, d_sae)

    # Replicate sae_trainer.py order: DP all_reduce first, then TP sync
    grads = [p.grad for p in sae.parameters() if p.grad is not None]
    flat = torch.cat([g.view(-1) for g in grads])
    dist.all_reduce(flat, group=dp_group)
    flat /= dp_size
    offset = 0
    for g in grads:
        numel = g.numel()
        g.copy_(flat[offset : offset + numel].view_as(g))
        offset += numel

    sae.sync_tensor_parallel_gradients()

    # Gather sharded grads to full shape
    shard_dims = sae._tp_param_shard_dims()
    full_grads: dict[str, torch.Tensor] = {}
    for name, param in sae.named_parameters():
        if param.grad is None:
            continue
        shard_dim = shard_dims.get(name)
        if shard_dim is not None:
            parts = [torch.zeros_like(param.grad) for _ in range(tp_size)]
            dist.all_gather(parts, param.grad.contiguous(), group=tp_group)
            full_grads[name] = torch.cat(parts, dim=shard_dim)
        else:
            full_grads[name] = param.grad.clone()

    dist.destroy_process_group()
    result_list.append((rank, full_grads))


@pytest.mark.parametrize("apply_b_dec_to_input", [False, True])
def test_tp2_gradients_match_tp1(apply_b_dec_to_input: bool):
    d_in, d_sae, k = 16, 32, 4
    port = 29700 + int(apply_b_dec_to_input)

    sae_ref = _make_sae(d_in, d_sae, k, apply_b_dec_to_input)
    random_params(sae_ref)
    state_dict = {name: v.clone() for name, v in sae_ref.state_dict().items()}
    x = torch.randn(8, d_in)

    ref_grads = _grads_tp1(d_in, d_sae, k, apply_b_dec_to_input, state_dict, x)
    tp2_grads_per_rank = _run_tp2(d_in, d_sae, k, apply_b_dec_to_input, state_dict, x, port)

    assert len(tp2_grads_per_rank) == 2
    for rank_idx, tp2_grads in enumerate(tp2_grads_per_rank):
        for name, ref_g in ref_grads.items():
            assert name in tp2_grads, f"rank {rank_idx}: missing grad for {name}"
            torch.testing.assert_close(
                tp2_grads[name],
                ref_g,
                atol=1e-5,
                rtol=1e-4,
                msg=f"rank {rank_idx}, param {name}: TP=2 grad differs from TP=1",
            )


def test_tp2_both_ranks_have_identical_grads_after_sync():
    d_in, d_sae, k = 16, 32, 4
    port = 29702

    sae_ref = _make_sae(d_in, d_sae, k, apply_b_dec_to_input=True)
    random_params(sae_ref)
    state_dict = {name: v.clone() for name, v in sae_ref.state_dict().items()}
    x = torch.randn(8, d_in)

    tp2_grads_per_rank = _run_tp2(d_in, d_sae, k, True, state_dict, x, port)

    assert len(tp2_grads_per_rank) == 2
    grads_r0, grads_r1 = tp2_grads_per_rank
    for name in grads_r0:
        torch.testing.assert_close(
            grads_r0[name],
            grads_r1[name],
            atol=0.0,
            rtol=0.0,
            msg=f"rank 0 and rank 1 disagree on grad for {name}",
        )


def _worker_clip_norm(
    rank: int,
    world_size: int,
    d_in: int,
    d_sae: int,
    k: int,
    state_dict: dict,
    x: torch.Tensor,
    result_list: list,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    tp_group = dist.new_group(list(range(world_size)), backend="gloo")

    sae = _make_sae(d_in, d_sae, k, apply_b_dec_to_input=True)
    sae.load_state_dict(state_dict)
    sae.shard_weights(tp_group)

    _forward_backward(sae, x, d_sae)
    sae.sync_tensor_parallel_gradients()
    norm = sae.clip_grad_norm_(1.0)

    dist.destroy_process_group()
    result_list.append((rank, norm.item()))


def test_tp2_clip_grad_norm_matches_tp1():
    """clip_grad_norm_ under TP=2 must return the same norm as TP=1."""
    d_in, d_sae, k = 16, 32, 4
    port = 29703

    sae_ref = _make_sae(d_in, d_sae, k, apply_b_dec_to_input=True)
    random_params(sae_ref)
    state_dict = {name: v.clone() for name, v in sae_ref.state_dict().items()}
    x = torch.randn(8, d_in)

    # TP=1 reference norm
    sae_tp1 = _make_sae(d_in, d_sae, k, apply_b_dec_to_input=True)
    sae_tp1.load_state_dict(state_dict)
    _forward_backward(sae_tp1, x, d_sae)
    ref_norm = sae_tp1.clip_grad_norm_(1.0).item()

    # TP=2 norm (both ranks must return the same value, equal to TP=1)
    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _worker_clip_norm,
        args=(2, d_in, d_sae, k, state_dict, x, result_list, port),
        nprocs=2,
        join=True,
    )
    results = sorted(result_list, key=lambda t: t[0])
    norms = [n for _, n in results]

    assert norms[0] == pytest.approx(norms[1], rel=1e-5), "Two TP ranks returned different norms"
    assert norms[0] == pytest.approx(ref_norm, rel=1e-4), "TP=2 norm differs from TP=1 norm"


def test_tp2_dp2_all_ranks_identical_grads():
    """TP=2, DP=2: after DP all_reduce + TP sync, all 4 ranks must agree on full gradients."""
    d_in, d_sae, k = 16, 32, 4
    port = 29704

    sae_ref = _make_sae(d_in, d_sae, k, apply_b_dec_to_input=True)
    random_params(sae_ref)
    state_dict = {name: v.clone() for name, v in sae_ref.state_dict().items()}

    # Two DP replicas get different data — this is the whole point of DP
    x_dp0 = torch.randn(8, d_in)
    x_dp1 = torch.randn(8, d_in)
    x_per_dp_rank = [x_dp0, x_dp1]

    manager = mp.Manager()
    result_list = manager.list()
    mp.spawn(
        _worker_tp2_dp2,
        args=(4, d_in, d_sae, k, state_dict, x_per_dp_rank, result_list, port),
        nprocs=4,
        join=True,
    )
    results = sorted(result_list, key=lambda t: t[0])
    all_grads = [grads for _, grads in results]

    # All 4 ranks must agree on every gradient (after full gather to full shape)
    ref = all_grads[0]
    for rank_idx, grads in enumerate(all_grads[1:], start=1):
        for name in ref:
            torch.testing.assert_close(
                grads[name],
                ref[name],
                atol=1e-5,
                rtol=1e-4,
                msg=f"rank {rank_idx} disagrees with rank 0 on grad for {name}",
            )

    # Sanity: DP must have actually mixed the two batches — gradients must differ
    # from what either replica would compute alone.
    ref_grads_dp0 = _grads_tp1(d_in, d_sae, k, True, state_dict, x_dp0)
    ref_grads_dp1 = _grads_tp1(d_in, d_sae, k, True, state_dict, x_dp1)
    expected_avg = {
        name: (ref_grads_dp0[name] + ref_grads_dp1[name]) / 2
        for name in ref_grads_dp0
    }
    for name, exp in expected_avg.items():
        torch.testing.assert_close(
            ref[name],
            exp,
            atol=1e-5,
            rtol=1e-4,
            msg=f"TP=2+DP=2 grad for {name} doesn't match average of two single-replica grads",
        )
