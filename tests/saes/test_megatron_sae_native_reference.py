"""Static TP acceptance against an independently generated upstream oracle."""

import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig

REFERENCE = Path(__file__).resolve().parents[1] / "native_reference"
# FP32 GEMM and collective addition order differ from the unsharded oracle.
# These are absolute tolerances; tiny/zero Adam moments must also pass.
TOLERANCE = dict(atol=3e-6, rtol=3e-5)


def test_oracle_integrity():
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    assert manifest["commit"] == "69c4c62b0dc24e5ba23fc773a0286149514b4a23"
    for path, key in [
        ("vendor/sae_lens-6.37.6-py3-none-any.whl", "wheel_sha256"),
        ("inputs.safetensors", "inputs_sha256"),
        ("golden.safetensors", "golden_sha256"),
    ]:
        assert (
            hashlib.sha256((REFERENCE / path).read_bytes()).hexdigest() == manifest[key]
        )


@pytest.mark.parametrize("case", range(4))
def test_tp1_math_against_native_oracle(case):
    """Real Megatron TP1 math; the singleton test group has no communication."""
    pytest.importorskip("megatron.core")
    import torch.testing._internal.distributed.fake_pg  # noqa: F401

    inputs = load_file(REFERENCE / "inputs.safetensors")
    golden = load_file(REFERENCE / "golden.safetensors")
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        cfg = TopKTrainingSAEConfig.from_dict(manifest["configs"][case])
        cfg.device = "cpu"
        sae = MegatronTopKSAE(cfg, tp_group=dist.group.WORLD)
        sae.import_saelens_state_dict(
            {
                k.removeprefix("initial."): v
                for k, v in inputs.items()
                if k.startswith("initial.")
            }
        )
        optimizer = torch.optim.Adam(sae.parameters(), **manifest["adam"])
        for step, batch in enumerate(inputs["batches"]):
            prefix = f"case{case}.step{step}."
            optimizer.zero_grad(set_to_none=True)
            output = sae(
                TrainStepInput(
                    batch,
                    {},
                    None if step == 0 else inputs["dead_masks"][step],
                    step,
                    False,
                )
            )
            for name in ("hidden_pre", "feature_acts", "loss"):
                torch.testing.assert_close(
                    getattr(output, name), golden[prefix + name], **TOLERANCE
                )
            torch.testing.assert_close(
                output.sae_out, golden[prefix + "reconstruction"], **TOLERANCE
            )
            for name, value in output.losses.items():
                torch.testing.assert_close(
                    value, golden[prefix + "losses." + name], **TOLERANCE
                )
            output.loss.backward()
            for phase in ("grad", "clipped_grad"):
                if phase == "clipped_grad":
                    torch.testing.assert_close(
                        sae.clip_grad_norm_(1.0),
                        golden[prefix + "grad_norm"],
                        **TOLERANCE,
                    )
                grads = {n: p.grad for n, p in sae.named_parameters()}
                sae.process_state_dict_for_saving(grads)
                for name, value in grads.items():
                    torch.testing.assert_close(
                        value, golden[prefix + phase + "." + name], **TOLERANCE
                    )
            optimizer.step()
            for name, value in sae.export_saelens_state_dict().items():
                torch.testing.assert_close(
                    value, golden[prefix + "parameter." + name], **TOLERANCE
                )
    finally:
        dist.destroy_process_group()


def _worker(rank, rendezvous, work_dir):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=180),
    )
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    inputs = load_file(REFERENCE / "inputs.safetensors")
    golden = load_file(REFERENCE / "golden.safetensors")
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    initial = {
        k.removeprefix("initial."): v
        for k, v in inputs.items()
        if k.startswith("initial.")
    }
    report = []
    try:
        for tp_size in (1, 2, 4):
            groups = [
                dist.new_group(list(range(start, start + tp_size)))
                for start in range(0, 4, tp_size)
            ]
            group = groups[rank // tp_size]
            errors = {}

            def check(actual, expected, label):
                actual = actual.detach().cpu()
                delta = (actual - expected).abs().max().item()
                category = label.split(".")[0]
                errors[category] = max(errors.get(category, 0.0), delta)
                torch.testing.assert_close(
                    actual,
                    expected,
                    **TOLERANCE,
                    msg=f"TP{tp_size} rank{rank}: {label}",
                )

            for case, config in enumerate(manifest["configs"]):
                cfg = TopKTrainingSAEConfig.from_dict(config)
                cfg.device = f"cuda:{rank}"
                # Deliberately unrelated per-rank constructor RNG, then explicit load.
                torch.manual_seed(914 + rank + case * 17)
                sae = MegatronTopKSAE(cfg, tp_group=group)
                sae.import_saelens_state_dict(initial)
                assert isinstance(sae.encoder, ColumnParallelLinear)
                assert isinstance(sae.decoder, RowParallelLinear)
                assert sae.encoder.allreduce_dgrad == (tp_size > 1)
                assert sae.encoder.weight.shape == (32 // tp_size, 16)
                assert sae.decoder.weight.shape == (16, 32 // tp_size)
                assert (
                    sae.encoder.weight.is_contiguous()
                    and sae.decoder.weight.is_contiguous()
                )
                assert set(dict(sae.named_parameters())) == {
                    "encoder.weight",
                    "decoder.weight",
                    "encoder.bias",
                    "b_dec",
                }
                optimizer = torch.optim.Adam(sae.parameters(), **manifest["adam"])
                for step, batch in enumerate(inputs["batches"]):
                    prefix = f"case{case}.step{step}."
                    optimizer.zero_grad(set_to_none=True)
                    step_input = TrainStepInput(
                        batch.cuda(rank),
                        {},
                        None if step == 0 else inputs["dead_masks"][step].cuda(rank),
                        step,
                        False,
                    )
                    output = sae(step_input)
                    for name in ("hidden_pre", "feature_acts", "loss"):
                        check(getattr(output, name), golden[prefix + name], name)
                    check(
                        output.sae_out,
                        golden[prefix + "reconstruction"],
                        "reconstruction",
                    )
                    for name, value in output.losses.items():
                        check(
                            value, golden[prefix + "losses." + name], "losses." + name
                        )
                    output.loss.backward()
                    # No TP sync helper is called: standard module backward must
                    # already produce a complete b_dec gradient on EVERY rank.
                    grads = {n: p.grad for n, p in sae.named_parameters()}
                    sae.process_state_dict_for_saving(grads)
                    for name, grad in grads.items():
                        check(grad, golden[prefix + "grad." + name], "grad." + name)
                    check(
                        sae.clip_grad_norm_(manifest["clip_norm"]),
                        golden[prefix + "grad_norm"],
                        "grad_norm",
                    )
                    clipped = {n: p.grad for n, p in sae.named_parameters()}
                    sae.process_state_dict_for_saving(clipped)
                    for name, grad in clipped.items():
                        check(
                            grad,
                            golden[prefix + "clipped_grad." + name],
                            "clipped_grad." + name,
                        )
                    optimizer.step()
                    state = sae.export_saelens_state_dict()
                    for name, value in state.items():
                        check(
                            value,
                            golden[prefix + "parameter." + name],
                            "parameter." + name,
                        )
                    adam = {
                        n: copy.deepcopy(optimizer.state[p])
                        for n, p in sae.named_parameters()
                    }
                    sae.process_named_optimizer_state_for_saving(adam)
                    for name, values in adam.items():
                        for key, value in values.items():
                            check(
                                value,
                                golden[prefix + f"adam.{name}.{key}"],
                                f"adam.{key}",
                            )
                    if step == 2:
                        # Checkpoint conversion and continuation; optimizer state
                        # remains native Megatron layout in memory.
                        checkpoint = (
                            Path(work_dir)
                            / f"tp{tp_size}_group{rank // tp_size}_case{case}"
                        )
                        sae.save_model(checkpoint)
                        restored = MegatronTopKSAE(cfg, tp_group=group)
                        restored.load_weights_from_checkpoint(checkpoint)
                        restored.process_named_optimizer_state_for_loading(adam)
                        optimizer = torch.optim.Adam(
                            restored.parameters(), **manifest["adam"]
                        )
                        for name, param in restored.named_parameters():
                            optimizer.state[param] = {
                                key: value.to(param.device) if value.ndim > 0 else value
                                for key, value in adam[name].items()
                            }
                        sae = restored
                checkpoint = (
                    Path(work_dir)
                    / f"inference_tp{tp_size}_group{rank // tp_size}_case{case}"
                )
                sae.save_inference_model(checkpoint)
                inference = SAE.load_from_disk(checkpoint, device=f"cuda:{rank}")
                with torch.no_grad():
                    probe = inputs["batches"][0].cuda(rank)
                    check(inference(probe), sae(probe).cpu(), "inference_roundtrip")
                    before = sae(probe)
                    if cfg.rescale_acts_by_decoder_norm:
                        sae.fold_W_dec_norm()
                        check(sae(probe), before.cpu(), "norm_fold")
            assert not parallel_state.model_parallel_is_initialized()
            report.append(
                {
                    "tp_size": tp_size,
                    "cases": 4,
                    "steps_per_case": 6,
                    "max_abs_error": errors,
                }
            )
            dist.barrier()
            for g in groups:
                if g != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(g)
        (Path(work_dir) / f"rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Static TP1/2/4 acceptance requires four CUDA GPUs",
)
def test_tp1_tp2_tp4_against_native_saelens(tmp_path):
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=4,
        join=True,
    )
