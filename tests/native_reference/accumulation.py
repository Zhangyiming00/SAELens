"""Isolated upstream large-batch oracle for runtime accumulation acceptance."""

import argparse
import hashlib
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SHA256 = "651789edfb1c905291aaec39fab3e3709d8cb09caa1e1c9f2c139b1a939075e1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wheel = ROOT / "vendor/sae_lens-6.37.6-py3-none-any.whl"
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == SHA256
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    plan = torch.load(args.input, weights_only=True)
    with tempfile.TemporaryDirectory(prefix="native-accumulation-") as extracted:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(extracted)
        sys.path.insert(0, extracted)
        import sae_lens
        from sae_lens.saes.sae import TrainStepInput
        from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
        from sae_lens.training.optim import get_lr_scheduler

        assert Path(sae_lens.__file__).is_relative_to(extracted)
        assert sae_lens.__version__ == "6.37.6"
        golden = {}
        for accumulation in (1, 2, 3):
            hooks = {}
            for h, cfg in plan["configs"].items():
                model = TopKTrainingSAE(
                    TopKTrainingSAEConfig.from_dict({**cfg, "device": "cuda:0"})
                )
                model.load_state_dict(plan["initial"])
                # The reference optimizer intentionally uses ordinary Adam.
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=3e-4, foreach=False, fused=False
                )
                scheduler = get_lr_scheduler(
                    "cosineannealing",
                    optimizer,
                    math.ceil(7 / accumulation),
                    3e-4,
                    0,
                    0,
                    3e-5,
                    1,
                )
                ages = torch.zeros(32, device="cuda:0")
                age_history = []
                snapshots = []
                for step, start in enumerate(range(0, 7, accumulation)):
                    batch = torch.cat(
                        plan["batches"][h][start : start + accumulation]
                    ).cuda()
                    snapshot = {}
                    if len(batch):
                        optimizer.zero_grad(set_to_none=True)
                        output = model.training_forward_pass(
                            TrainStepInput(
                                sae_in=batch,
                                coefficients={},
                                dead_neuron_mask=ages > 0,
                                n_training_steps=step,
                                is_logging_step=False,
                            )
                        )
                        output.loss.backward()
                        snapshot["loss"] = output.loss.detach().cpu()
                        snapshot["grad"] = {
                            n: p.grad.detach().cpu().clone()
                            for n, p in model.named_parameters()
                        }
                        snapshot["norm"] = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0, foreach=False
                        ).cpu()
                        snapshot["clipped"] = {
                            n: p.grad.detach().cpu().clone()
                            for n, p in model.named_parameters()
                        }
                        optimizer.step()
                        scheduler.step()
                        ages += 1
                        ages[output.feature_acts.bool().any(0)] = 0
                    snapshot["parameters"] = {
                        n: p.detach().cpu().clone() for n, p in model.named_parameters()
                    }
                    snapshot["adam"] = {
                        n: {
                            k: v.detach().cpu().clone()
                            for k, v in optimizer.state[p].items()
                        }
                        for n, p in model.named_parameters()
                    }
                    snapshot["lr"] = optimizer.param_groups[0]["lr"]
                    snapshots.append(snapshot)
                    age_history.append(ages.cpu().clone())
                hooks[h] = dict(snapshots=snapshots, ages=age_history)
            golden[accumulation] = hooks
        torch.save(golden, args.output)


if __name__ == "__main__":
    main()
