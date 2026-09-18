"""SAE bias-edge TP reduction, including differentiable-input fallback."""

import json
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig


def _worker(rank, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=4,
        init_method=f"file://{directory}/world",
        timeout=timedelta(seconds=120),
    )
    fixture = load_file(
        Path(__file__).parents[1] / "native_reference/inputs.safetensors"
    )
    initial = {
        k.removeprefix("initial."): v
        for k, v in fixture.items()
        if k.startswith("initial.")
    }
    records = []
    try:
        for tp_size in (1, 2, 4):
            groups = [
                dist.new_group(list(range(i, i + tp_size)))
                for i in range(0, 4, tp_size)
            ]
            group = groups[rank // tp_size]
            for norm in ("none", "layer_norm", "constant_norm_rescale"):
                for apply_bias in (False, True):
                    for mode in ("detached", "input", "hook"):
                        cfg = TopKTrainingSAEConfig(
                            d_in=16,
                            d_sae=32,
                            k=4,
                            device=f"cuda:{rank}",
                            normalize_activations=norm,
                            apply_b_dec_to_input=apply_bias,
                            use_sparse_activations=False,
                        )
                        sae = MegatronTopKSAE(cfg, tp_group=group)
                        sae.import_saelens_state_dict(initial)
                        reference = TopKTrainingSAE(cfg)
                        reference.load_state_dict(initial)
                        x = (
                            fixture["batches"][0]
                            .cuda(rank)
                            .detach()
                            .requires_grad_(mode == "input")
                        )
                        y = x.detach().clone().requires_grad_(mode == "input")
                        gate = torch.tensor(0.9, device=x.device, requires_grad=True)
                        ref_gate = gate.detach().clone().requires_grad_()
                        if mode == "hook":
                            sae.hook_sae_input.add_hook(
                                lambda value, hook: value * gate  # noqa: ARG005 -- HookPoint keyword
                            )
                            reference.hook_sae_input.add_hook(
                                lambda value, hook: value * ref_gate  # noqa: ARG005 -- HookPoint keyword
                            )
                        actual = sae(TrainStepInput(x, {}, None, 0, False))
                        expected = reference(TrainStepInput(y, {}, None, 0, False))
                        torch.testing.assert_close(
                            actual.loss, expected.loss, atol=3e-6, rtol=3e-5
                        )
                        calls = []
                        original = dist.all_reduce

                        def record(tensor, *args, **kwargs):
                            calls.append(tuple(tensor.shape))
                            return original(tensor, *args, **kwargs)

                        with patch.object(dist, "all_reduce", record):
                            actual.loss.backward()
                        expected.loss.backward()
                        target = []
                        if tp_size > 1:
                            if apply_bias:
                                target.append((16,))
                            if mode != "detached":
                                target.append(tuple(x.shape))
                        assert Counter(calls) == Counter(target), (
                            tp_size,
                            norm,
                            mode,
                            calls,
                            target,
                        )
                        grads = {name: p.grad for name, p in sae.named_parameters()}
                        sae.process_state_dict_for_saving(grads)
                        errors = {}
                        for name, param in reference.named_parameters():
                            torch.testing.assert_close(
                                grads[name], param.grad, atol=3e-6, rtol=3e-5
                            )
                            errors[name] = (grads[name] - param.grad).abs().max().item()
                        if mode == "input":
                            torch.testing.assert_close(
                                x.grad, y.grad, atol=3e-6, rtol=3e-5
                            )
                        if mode == "hook":
                            torch.testing.assert_close(
                                gate.grad, ref_gate.grad, atol=3e-6, rtol=3e-5
                            )
                        records.append(
                            dict(
                                tp=tp_size,
                                norm=norm,
                                apply_bias=apply_bias,
                                mode=mode,
                                backward_all_reduce_shapes=calls,
                                max_abs_grad_error=errors,
                            )
                        )
            dist.barrier()
            for g in groups:
                if g != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(g)
        (Path(directory) / f"rank{rank}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_bias_edge_and_differentiable_input_gradients(tmp_path):
    directory = Path(os.environ.get("SAE_TP_DGRAD_REPORT_DIR", tmp_path)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(str(directory),), nprocs=4, join=True)
