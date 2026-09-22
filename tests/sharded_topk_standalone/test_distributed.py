from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from helpers import (
    Input,
    NoGlobalLatent,
    dense_reference,
    make_harness,
    reference_forward,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from sae_lens.sharded_topk import (
    sharded_firing_counts,
    sharded_topk,
)


def init(rank, size, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=60),
    )


def selection_worker(rank, size, path):
    init(rank, size, path)
    try:
        group = dist.new_group(list(range(size)))
        width, rows = 32, 5
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for k in (1, 3, 15, 33, size * width):
                for protocol in ("auto", "radix"):
                    for use_mask in (False, True):
                        torch.manual_seed(19)
                        full = torch.randint(-4, 5, (rows, size * width)).to(dtype)
                        # All largest features can live on rank0; test nonuniform ownership.
                        full[0, :width] += 10
                        mask = (
                            (torch.arange(size * width) % 2 == 0) if use_mask else None
                        )
                        budget = min(k, int(mask.sum())) if use_mask else k
                        full.requires_grad_()
                        local = (
                            full.detach()[:, rank * width : (rank + 1) * width]
                            .clone()
                            .requires_grad_()
                        )
                        expected = dense_reference(
                            full, budget, eligible=mask, relu=not use_mask
                        )
                        ref_grad = torch.autograd.grad(expected.sum(), full)[0]
                        with NoGlobalLatent(rows, size * width):
                            out = sharded_topk(
                                local,
                                budget,
                                group,
                                eligible=None
                                if mask is None
                                else mask[rank * width : (rank + 1) * width],
                                relu=not use_mask,
                                sparse=True,
                                protocol=protocol,
                            )
                            out.sum().backward()
                            summary = sharded_firing_counts(out, group)
                        torch.testing.assert_close(
                            out.to_dense(),
                            expected.detach()[:, rank * width : (rank + 1) * width],
                        )
                        torch.testing.assert_close(
                            local.grad, ref_grad[:, rank * width : (rank + 1) * width]
                        )
                        torch.testing.assert_close(
                            summary, (expected.detach() != 0).float().sum(0)
                        )
        # Forced candidates may not violate the shard-sized workspace budget.
        with pytest.raises(ValueError, match="exceeds one shard"):
            sharded_topk(torch.randn(rows, width), width, group, protocol="candidates")
        # group=None must remain local even with WORLD initialized.
        x = torch.randn(rows, width)
        torch.testing.assert_close(
            sharded_topk(x, 2, None, sparse=False), dense_reference(x, 2)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_distributed_selection(size, tmp_path):
    mp.start_processes(
        selection_worker,
        args=(size, str(tmp_path / "store")),
        nprocs=size,
        start_method="spawn",
    )


def make_data(update, micro, dp_rank, dp_size):
    # Arbitrary unequal token counts. A zero-token rank on the last microbatch
    # exercises the DDP bucket readiness of sparse decoder zero gradients.
    sizes = [5, 3, 4, 2]
    n = sizes[dp_rank]
    if micro == 1 and dp_rank == dp_size - 1 and dp_size > 1:
        n = 0
    g = torch.Generator().manual_seed(500 + update * 100 + micro * 10 + dp_rank)
    return torch.randn(n, 7, generator=g)


def training_worker(rank, size, path, tp, dp, backend, wavefront):
    init(rank, size, path)
    try:
        tp_groups = [
            dist.new_group(list(range(j * tp, (j + 1) * tp))) for j in range(dp)
        ]
        dp_groups = [dist.new_group([j * tp + i for j in range(dp)]) for i in range(tp)]
        # The last world rank is a spectator/vLLM role. It must never enter a
        # selector, gradient, or summary collective for the SAE role.
        if rank >= tp * dp:
            dist.barrier()
            return
        ti, di = rank % tp, rank // tp
        group = tp_groups[di]
        width = 16
        torch.manual_seed(108)
        initial = [
            torch.randn(tp * width, 7) * 0.2,
            torch.randn(tp * width) * 0.1,
            torch.randn(7, tp * width) * 0.2,
            torch.randn(7) * 0.1,
        ]
        model = make_harness(initial, group, backend, rescale=True)
        wrapped = DDP(model, process_group=dp_groups[ti]) if dp > 1 else model
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        ref_weights = [torch.nn.Parameter(t.clone()) for t in initial]
        ref_opt = torch.optim.Adam(ref_weights, lr=3e-4)
        mask = torch.arange(tp * width) % 3 == 0
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            ref_opt.zero_grad(set_to_none=True)
            total = sum(
                make_data(update, m, j, dp).shape[0]
                for m in range(2)
                for j in range(dp)
            )
            # Reference full tensors are created OUTSIDE the allocation guard.
            expected_local = []
            for m in range(2):
                for j in range(dp):
                    x = make_data(update, m, j, dp)
                    out, acts, loss = reference_forward(ref_weights, x, 2, mask)
                    (loss * (x.shape[0] / total)).backward()
                    if j == di:
                        expected_local.append(
                            (out.detach(), (acts.detach() != 0).float().sum(0))
                        )
            for m in range(2):
                x = make_data(update, m, di, dp)
                context = wrapped.no_sync() if dp > 1 and m == 0 else nullcontext()
                with context, NoGlobalLatent(
                    x.shape[0], tp * width
                ) if tp > 1 and x.shape[0] else nullcontext():
                    if wavefront:
                        assert (
                            dp == 1
                        )  # this harness bypasses DDP forward; GPU script covers both
                        state = model.tp_wavefront_encode_launch(Input(x, mask))
                        model.tp_wavefront_decode_launch(state)
                        result = model.tp_wavefront_finish(state)
                    else:
                        result = wrapped(Input(x, mask))
                    assert result.hidden_pre.shape[-1] == width
                    assert result.feature_acts.shape[-1] == width
                    torch.testing.assert_close(
                        result.sae_out, expected_local[m][0], atol=3e-6, rtol=3e-5
                    )
                    torch.testing.assert_close(
                        result.feature_firing_counts, expected_local[m][1]
                    )
                    # Torch DDP averages. This compensates it to test the same
                    # token-weighted SUM semantics used by the Megatron runner.
                    (result.loss * (x.shape[0] / total) * dp).backward()
            actual_params = [
                model.encoder.weight,
                model.encoder.bias,
                model.decoder.weight,
                model.b_dec,
            ]
            slices = [
                ref_weights[0].grad[ti * width : (ti + 1) * width],
                ref_weights[1].grad[ti * width : (ti + 1) * width],
                ref_weights[2].grad[:, ti * width : (ti + 1) * width],
                ref_weights[3].grad,
            ]
            for p, expected in zip(actual_params, slices):
                assert p.grad is not None and p.grad.layout == torch.strided
                torch.testing.assert_close(p.grad, expected, atol=4e-6, rtol=8e-5)
            optimizer.step()
            ref_opt.step()
            expected_params = [
                ref_weights[0][ti * width : (ti + 1) * width],
                ref_weights[1][ti * width : (ti + 1) * width],
                ref_weights[2][:, ti * width : (ti + 1) * width],
                ref_weights[3],
            ]
            for p, expected in zip(actual_params, expected_params):
                torch.testing.assert_close(p, expected, atol=5e-6, rtol=5e-5)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "tp,dp,wavefront",
    [(1, 3, False), (2, 1, False), (4, 1, False), (2, 2, False), (2, 1, True)],
)
@pytest.mark.parametrize("backend", ["sharded_dense", "sharded_sparse"])
def test_tp_dp_ga_updates(tp, dp, wavefront, backend, tmp_path):
    size = tp * dp + 1
    mp.start_processes(
        training_worker,
        args=(size, str(tmp_path / "store"), tp, dp, backend, wavefront),
        nprocs=size,
        start_method="spawn",
    )
