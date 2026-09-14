"""Generate/replay the oracle using ONLY the pinned upstream wheel.

Run: .venv/bin/python -I tests/native_reference/generate.py --device cuda:0
No imports from the checkout are allowed in this process.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
WHEEL = ROOT / "vendor/sae_lens-6.37.6-py3-none-any.whl"
SHA256 = "651789edfb1c905291aaec39fab3e3709d8cb09caa1e1c9f2c139b1a939075e1"
COMMIT = "69c4c62b0dc24e5ba23fc773a0286149514b4a23"
ADAM = dict(
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    amsgrad=False,
    foreach=False,
    fused=False,
)


def snapshot(sae, optimizer, output):
    result = {
        "reconstruction": output.sae_out,
        "hidden_pre": output.hidden_pre,
        "feature_acts": output.feature_acts,
        "loss": output.loss,
        **{f"losses.{k}": v for k, v in output.losses.items()},
        **{f"grad.{k}": p.grad for k, p in sae.named_parameters()},
    }
    result = {k: v.detach().cpu().clone() for k, v in result.items()}
    norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0, foreach=False)
    result["grad_norm"] = norm.detach().cpu().clone()
    result.update(
        {
            f"clipped_grad.{k}": p.grad.detach().cpu().clone()
            for k, p in sae.named_parameters()
        }
    )
    optimizer.step()
    for name, param in sae.named_parameters():
        result[f"parameter.{name}"] = param.detach().cpu().clone()
        for key, value in optimizer.state[param].items():
            result[f"adam.{name}.{key}"] = value.detach().cpu().clone()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Replay stored weights/batches and check the committed oracle",
    )
    args = parser.parse_args()
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest() == SHA256
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    with tempfile.TemporaryDirectory(prefix="saelens-native-") as extracted:
        with zipfile.ZipFile(WHEEL) as wheel:
            wheel.extractall(extracted)
        sys.path.insert(0, extracted)
        import sae_lens
        from sae_lens.saes.sae import TrainStepInput
        from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

        assert Path(sae_lens.__file__).is_relative_to(extracted), sae_lens.__file__
        assert sae_lens.__version__ == "6.37.6"
        fixture_path = ROOT / "inputs.safetensors"
        if args.check:
            inputs = load_file(fixture_path)
        else:
            torch.manual_seed(7291)
            init = TopKTrainingSAE(
                TopKTrainingSAEConfig(d_in=16, d_sae=32, k=4, decoder_init_norm=0.7)
            )
            inputs = {
                f"initial.{k}": v.detach().clone() for k, v in init.state_dict().items()
            }
            inputs.update(
                {
                    f"native_init.{k}": v.detach().clone()
                    for k, v in init.state_dict().items()
                }
            )
            # Unequal decoder norms and nonzero biases exercise both norm axes
            # and both b_dec gradient contributions.
            inputs["initial.W_dec"] *= torch.linspace(0.4, 1.7, 32)[:, None]
            inputs["initial.b_enc"] = torch.linspace(-0.23, 0.37, 32)
            inputs["initial.b_dec"] = torch.linspace(-0.4, 0.3, 16)
            inputs["batches"] = torch.randn(6, 11, 16)
            masks = torch.zeros(6, 32, dtype=torch.bool)
            masks[2, [1, 7, 29]] = True
            masks[3, ::2] = True
            masks[4] = True
            masks[5, 1::2] = True
            inputs["dead_masks"] = masks
            save_file(inputs, fixture_path)
        configs, golden = [], {}
        for rescale in (False, True):
            for apply_bias in (False, True):
                cfg = TopKTrainingSAEConfig(
                    d_in=16,
                    d_sae=32,
                    k=4,
                    dtype="float32",
                    device=args.device,
                    decoder_init_norm=0.7,
                    use_sparse_activations=False,
                    rescale_acts_by_decoder_norm=rescale,
                    apply_b_dec_to_input=apply_bias,
                    aux_loss_coefficient=0.13,
                    normalize_activations="none",
                )
                configs.append(cfg.to_dict())
                sae = TopKTrainingSAE(cfg)
                # Explicit load, independent of the constructor's RNG state.
                sae.load_state_dict(
                    {
                        k.removeprefix("initial."): v
                        for k, v in inputs.items()
                        if k.startswith("initial.")
                    }
                )
                optimizer = torch.optim.Adam(sae.parameters(), **ADAM)
                for step, batch in enumerate(inputs["batches"]):
                    optimizer.zero_grad(set_to_none=True)
                    output = sae.training_forward_pass(
                        TrainStepInput(
                            sae_in=batch.to(args.device),
                            coefficients={},
                            dead_neuron_mask=None
                            if step == 0
                            else inputs["dead_masks"][step].to(args.device),
                            n_training_steps=step,
                            is_logging_step=False,
                        )
                    )
                    output.loss.backward()
                    for key, value in snapshot(sae, optimizer, output).items():
                        golden[f"case{len(configs) - 1}.step{step}.{key}"] = value
        if args.check:
            expected = load_file(ROOT / "golden.safetensors")
            assert expected.keys() == golden.keys()
            for key in golden:
                torch.testing.assert_close(
                    golden[key], expected[key], atol=2e-6, rtol=2e-5, msg=key
                )
            print(f"Native oracle replay passed: {len(configs)} cases x 6 steps")  # noqa: T201
        else:
            save_file(golden, ROOT / "golden.safetensors")
            metadata = {
                "version": "6.37.6",
                "commit": COMMIT,
                "wheel_sha256": SHA256,
                "torch_version": torch.__version__,
                "device": args.device,
                "tf32": False,
                "adam": ADAM,
                "clip_norm": 1.0,
                "clip_scope": "one SAE",
                "configs": configs,
                "inputs_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "golden_sha256": hashlib.sha256(
                    (ROOT / "golden.safetensors").read_bytes()
                ).hexdigest(),
            }
            (ROOT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Saved native oracle: {len(configs)} cases x 6 steps")  # noqa: T201


if __name__ == "__main__":
    main()
