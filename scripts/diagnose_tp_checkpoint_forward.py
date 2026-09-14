"""Compare serial and TP-sharded SAE forward from the same checkpoint.

Run with torchrun, for example:
  torchrun --nproc_per_node=2 scripts/diagnose_tp_checkpoint_forward.py \
    --checkpoint /path/to/quiesce_661696 --batch-tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from sae_lens.config import DTYPE_MAP
from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig, TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.multi_sae_trainer import _load_tp_sharded_state_dict


def _init_dist() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world_size, device


def _make_sae(hook_dir: Path, device: torch.device, *, tp_group=None) -> TrainingSAE:
    cfg_dict = json.loads((hook_dir / "cfg.json").read_text())
    cfg = TrainingSAEConfig.from_dict(cfg_dict)
    if tp_group is not None:
        if type(cfg) is not TopKTrainingSAEConfig:
            raise ValueError("Megatron TP checkpoint diagnostic requires a TopK SAE")
        cfg.device = str(device)
        return MegatronTopKSAE(cfg, tp_group=tp_group)
    sae = TrainingSAE.from_dict(cfg.to_dict())
    sae.to(device)
    return sae


def _train_input(x: torch.Tensor) -> TrainStepInput:
    return TrainStepInput(
        sae_in=x,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rank, world_size, device = _init_dist()
    checkpoint = Path(args.checkpoint)
    hooks = [
        p.name
        for p in checkpoint.iterdir()
        if p.is_dir() and (p / "sae_weights.safetensors").exists()
    ]
    hooks = sorted(hooks)

    if rank == 0:
        print(f"checkpoint={checkpoint}")
        print(f"world_size={world_size} device={device}")

    for hook_path_name in hooks:
        hook_dir = checkpoint / hook_path_name
        cfg_dict = json.loads((hook_dir / "cfg.json").read_text())
        d_in = int(cfg_dict["d_in"])
        dtype = DTYPE_MAP[cfg_dict.get("dtype", "float32")]

        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed)
        x_cpu = torch.randn(args.batch_tokens, d_in, generator=gen, dtype=torch.float32)
        x = x_cpu.to(device=device, dtype=dtype)

        tp_sae = _make_sae(hook_dir, device, tp_group=dist.group.WORLD)
        state_dict = _load_tp_sharded_state_dict(
            hook_dir / "sae_weights.safetensors",
            tp_sae,
            dist.group.WORLD,
        )
        tp_sae.load_state_dict(state_dict)
        tp_sae.eval()
        tp_out = tp_sae.training_forward_pass(_train_input(x))
        tp_mse = tp_out.losses["mse_loss"].detach().float()

        mse_values = [torch.zeros_like(tp_mse) for _ in range(world_size)]
        dist.all_gather(mse_values, tp_mse)

        if rank == 0:
            serial_sae = _make_sae(hook_dir, device)
            serial_sae.load_weights_from_checkpoint(hook_dir)
            serial_sae.eval()
            serial_out = serial_sae.training_forward_pass(_train_input(x))
            serial_mse = serial_out.losses["mse_loss"].detach().float()
            max_abs = (
                (serial_out.sae_out.detach().float() - tp_out.sae_out.detach().float())
                .abs()
                .max()
            )
            mean_abs = (
                (serial_out.sae_out.detach().float() - tp_out.sae_out.detach().float())
                .abs()
                .mean()
            )
            print(
                json.dumps(
                    {
                        "hook": hook_path_name,
                        "serial_mse": float(serial_mse.cpu()),
                        "tp_mse_by_rank": [float(v.cpu()) for v in mse_values],
                        "rank0_output_max_abs_diff": float(max_abs.cpu()),
                        "rank0_output_mean_abs_diff": float(mean_abs.cpu()),
                    },
                    sort_keys=True,
                )
            )

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
