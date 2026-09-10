from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from run_sae_runner_gpu import parse_args
from sae_lens.training.ddp_zero_optimizer import (
    build_adam_optimizer,
    consolidate_optimizer_state,
    is_zero_optimizer,
)
from sae_lens.training.elastic_trainer_state import broadcast_optimizer_state


def test_ddp_zero_optimizer_cli_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_sae_runner_gpu.py"])
    assert parse_args().ddp_zero_optimizer is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_sae_runner_gpu.py", "--ddp-zero-optimizer"],
    )
    assert parse_args().ddp_zero_optimizer is True


def test_ddp_zero_optimizer_cli_rejects_fsdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_sae_runner_gpu.py", "--fsdp", "--ddp-zero-optimizer"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def _zero_optimizer_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Linear(4, 4, bias=False),
        )
        ddp = DDP(model)
        optimizer = build_adam_optimizer(
            ddp.parameters(),
            adam_kwargs={"lr": 0.01},
            zero_redundancy=True,
            ddp_enabled=True,
            dp_group=dist.group.WORLD,
        )
        assert is_zero_optimizer(optimizer)

        inputs = torch.full((2, 4), float(rank + 1))
        ddp(inputs).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        local_parameters = [
            parameter
            for group in optimizer.optim.param_groups  # type: ignore[attr-defined]
            for parameter in group["params"]
        ]
        local_state_count = len(optimizer.optim.state)  # type: ignore[attr-defined]
        consolidate_optimizer_state(optimizer, to=0)
        if rank == 0:
            consolidated = getattr(
                optimizer, "_saelens_consolidated_state_by_parameter"
            )
            assert len(consolidated) == len(list(model.parameters()))
            assert all(
                value.device.type == "cpu"
                for state in consolidated.values()
                for value in state.values()
                if torch.is_tensor(value)
            )

        target_model = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Linear(4, 4, bias=False),
        )
        target_model.load_state_dict(model.state_dict())
        target_ddp = DDP(target_model)
        target_optimizer = build_adam_optimizer(
            target_ddp.parameters(),
            adam_kwargs={"lr": 1.0},
            zero_redundancy=True,
            ddp_enabled=True,
            dp_group=dist.group.WORLD,
        )
        broadcast_optimizer_state(
            source=optimizer if rank == 0 else None,
            target=target_optimizer,
            group=dist.group.WORLD,
            source_global_rank=0,
            device=torch.device("cpu"),
            source_parameters=list(model.parameters()) if rank == 0 else None,
            target_parameters=list(target_model.parameters()),
        )

        # The restored optimizer should produce the same second Adam update.
        ddp(inputs).sum().backward()
        optimizer.step()
        target_ddp(inputs).sum().backward()
        target_optimizer.step()
        for source_parameter, target_parameter in zip(
            model.parameters(), target_model.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)

        torch.save(
            {
                "local_parameter_count": len(local_parameters),
                "local_state_count": local_state_count,
                "target_local_state_count": len(
                    target_optimizer.optim.state  # type: ignore[attr-defined]
                ),
                "weights": [parameter.detach() for parameter in model.parameters()],
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_zero_optimizer_shards_updates_and_restores_state(tmp_path: Path) -> None:
    world_size = 2
    mp.start_processes(
        _zero_optimizer_worker,
        args=(world_size, str(tmp_path / "gloo_init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )

    results = [
        torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]
    assert sum(result["local_parameter_count"] for result in results) == 2
    assert all(result["local_parameter_count"] == 1 for result in results)
    assert all(result["local_state_count"] == 1 for result in results)
    assert all(result["target_local_state_count"] == 1 for result in results)
    for left, right in zip(results[0]["weights"], results[1]["weights"]):
        torch.testing.assert_close(left, right)
