"""Native fused wgrad must preserve norm branches, aux reuse, and GA windows."""

import functools
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.sae_runtime import SAERuntime
from sae_lens.training.gradient_window import train_runtime_window
from sae_lens.training.megatron_ddp import wrap_runtime_sae
from tests.training.test_hook_optimizer_overlap import build, equal, snapshot


def _worker(rank, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", rank=rank, world_size=4, init_method=f"file://{directory}/world",
        timeout=timedelta(seconds=90),
    )
    reports = []
    try:
        for tp, dp in ((1, 2), (2, 2), (2, 1)):
            runtime = SAERuntime.from_layout(
                tp_size=tp, dp_size=dp, placement_size=1, hooks=("h0", "h1", "h2")
            )
            try:
                if runtime.local is None:
                    continue
                for ga, amp, rescale, norm_probe in (
                    (1, False, True, False), (3, False, True, False),
                    (3, True, True, False), (1, False, False, False),
                    (1, False, True, True),
                ):
                    trainers, captures, flags = [], [], []
                    for enabled in (False, True):
                        with patch(
                            "tests.training.test_hook_optimizer_overlap.wrap_runtime_sae",
                            functools.partial(wrap_runtime_sae, gradient_accumulation_fusion=enabled),
                        ):
                            trainer = build(runtime, Path(directory), "on", ga, amp,
                                            architecture="unified_multi_hook")
                        trainer.cfg.dead_feature_window = -1
                        captured, marked = {}, {}
                        for h, unit in trainer.units.items():
                            assert unit.model.gradient_accumulation_fusion == (
                                enabled and dp > 1 and not amp
                            )
                            unit.model.cfg.rescale_acts_by_decoder_norm = rescale
                            active = unit.model.configure_gradient_accumulation_fusion(enabled and not amp)
                            assert active == (enabled and dp > 1 and not amp)
                            if norm_probe:
                                # A non-cancelling ordinary gradient proves the
                                # native zero_out_wgrad safeguard is necessary.
                                original_aux = unit.model.calculate_aux_loss
                                def auxiliary(*a, model=unit.model, original=original_aux, **kw):
                                    losses = original(*a, **kw)
                                    losses["norm_probe"] = 0.03 * model.decoder.weight.norm(dim=0).sum()
                                    return losses
                                unit.model.calculate_aux_loss = auxiliary
                            original_clip = unit.optimizer.clip_grad_norm
                            def clip(value, unit=unit, hook=h, original=original_clip,
                                     captured=captured, marked=marked):
                                captured[hook] = {
                                    n: p.grad.detach().clone()
                                    for n, p in unit.model.named_parameters()
                                }
                                marked[hook] = {
                                    n: getattr(p, "grad_added_to_main_grad", False)
                                    for n, p in unit.model.named_parameters()
                                }
                                return original(value)
                            unit.optimizer.clip_grad_norm = clip
                        trainers.append(trainer)
                        captures.append(captured)
                        flags.append(marked)
                    max_gradient_error = 0.0
                    for step in range(4):
                        batches = []
                        for micro in range(ga):
                            torch.manual_seed(900 + step * 17 + micro + runtime.local.dp_rank)
                            batches.append({h: torch.randn(7 + runtime.local.dp_rank, 16,
                                                           device=f"cuda:{rank}")
                                            for h in trainers[0].units})
                        for trainer in trainers:
                            train_runtime_window(trainer, batches)
                            trainer.n_training_steps += 1
                        equal(captures[0], captures[1])
                        equal(snapshot(trainers[0]), snapshot(trainers[1]))
                        for h in trainers[0].units:
                            for name in ("encoder.weight", "decoder.weight"):
                                assert flags[1][h][name] == (dp > 1 and not amp)
                                assert not flags[0][h][name]
                            for name in captures[0][h]:
                                max_gradient_error = max(max_gradient_error, float(
                                    (captures[0][h][name] - captures[1][h][name]).abs().max()))
                    reports.append(dict(tp=tp, dp=dp, ga=ga, amp=amp, rescale=rescale,
                                        norm_probe=norm_probe,
                                        wavefront=trainers[1]._runtime_tp_wavefront,
                                        max_gradient_error=max_gradient_error, flags=flags[1]))
            finally:
                dist.barrier(group=runtime.control_group)
                runtime.close()
    finally:
        Path(directory, f"rank{rank}.json").write_text(json.dumps(reports, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_native_gradient_accumulation_fusion(tmp_path, monkeypatch):
    pytest.importorskip("fused_weight_gradient_mlp_cuda")
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    directory = Path(os.environ.get("SAE_GRAD_FUSION_REPORT_DIR", tmp_path)).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    processes = mp.spawn(_worker, args=(str(directory),), nprocs=4, join=False)
    deadline = time.monotonic() + 120
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Native gradient fusion validation timed out")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
