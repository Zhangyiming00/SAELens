"""Real process groups, including domains smaller than the routing world."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.sae_runtime import SAERuntime, SAETrainingDomain


def _runtime_worker(rank, rendezvous, output, backend):
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=90),
    )
    try:
        from megatron.core import parallel_state
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.process_groups_config import ProcessGroupCollection

        cases = [
            (SAETrainingDomain("all", (0, 1, 2, 3), 2, ("h1", "h2")),),
            (SAETrainingDomain("prefix", (0, 1), 2, ("h1",)),),
            (SAETrainingDomain("noncontiguous", (1, 3), 1, ("h1",)),),
            (
                SAETrainingDomain("h1", (0, 1), 2, ("h1",)),
                SAETrainingDomain("h2", (2, 3), 2, ("h2",)),
            ),
        ]
        reports = []
        for domains in cases:
            with_runtime = SAERuntime(domains, backend=backend)
            context = with_runtime.local
            assert dist.get_world_size() == 4
            assert not parallel_state.model_parallel_is_initialized()
            if context is not None:
                groups = ProcessGroupCollection.setup_process_groups_for_ddp(
                    context.groups,
                    ModelParallelConfig(),
                    DistributedDataParallelConfig(),
                )
                assert groups["tp_group"] is context.tp_group
                assert groups["dp_group"] is context.dp_group
                optimizer_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
                    context.groups, [SimpleNamespace(ddp_config=DistributedDataParallelConfig())],
                    use_gloo_process_groups=False,
                )
                assert optimizer_groups["dp_group"] is context.dp_group
                assert optimizer_groups["mp_group"] is context.tp_group
                device = f"cuda:{rank}" if backend == "nccl" else "cpu"
                value = torch.tensor(float(rank), device=device)
                dist.all_reduce(value, group=context.dp_group)
                assert value.item() == sum(context.dp_ranks)
                dist.broadcast(value, src=context.receive_rank, group=context.tp_group)
            reports.append(
                [
                    {
                        "domain": e.domain,
                        "replica": e.replica_index,
                        "ranks": e.tp_ranks,
                        "root": e.receive_rank,
                    }
                    for e in with_runtime.endpoints
                ]
            )
            dist.barrier()
            with_runtime.close()
            with_runtime.close()
            dist.barrier()
        (Path(output) / f"runtime_rank{rank}.json").write_text(
            json.dumps(reports, indent=2)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_runtime_domains_on_four_gpus(tmp_path):
    mp.spawn(
        _runtime_worker,
        args=(f"file://{tmp_path / 'rdzv'}", str(tmp_path), "nccl"),
        nprocs=4,
    )
    reports = [
        json.loads((tmp_path / f"runtime_rank{rank}.json").read_text())
        for rank in range(4)
    ]
    assert all(r == reports[0] for r in reports)
    assert [r["ranks"] for r in reports[0][0]] == [[0, 1], [2, 3]]
    assert [r["ranks"] for r in reports[0][1]] == [[0, 1]]
    assert [r["root"] for r in reports[0][2]] == [1, 3]
