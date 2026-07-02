"""Controlled three-layer memory case study for one TopK SAE training step.

Goal: for ONE representative config, walk the real combined-backward training
loop phase-by-phase and decompose GPU memory into the three semantically
distinct layers, plus the two gaps between them:

    allocated            live PyTorch tensors (params, grads, Adam state, batch,
                         autograd-saved activations) RIGHT NOW
    reserved             bytes the caching allocator holds from the CUDA driver
    device_used          whole-process footprint as the driver / nvidia-smi sees
                         it (== total - free from cudaMemGetInfo)

    reserved  - allocated   allocator cache: freed-but-not-returned blocks +
                            internal fragmentation (live inside torch's pool)
    device    - reserved    everything OUTSIDE torch's allocator: CUDA context,
                            kernels/cuBLAS/cuDNN workspaces, NCCL buffers, the
                            runtime itself

Measurement discipline (matches multi_sae_trainer._record_memory_phase):
  * torch.cuda.synchronize() BEFORE every snapshot so async kernels have landed
    and the allocator/driver counters are settled.
  * peak_* at each phase = the high-water mark over the interval SINCE THE
    PREVIOUS phase record, because we call reset_peak_memory_stats() right after
    every snapshot. So "peak_allocated at after_combined_backward" means "the
    tallest the live set got while backward was running", not a global peak.
  * device_watermark = the max device_used seen across all phases of the step
    (the driver footprint never shrinks unless empty_cache hands blocks back).

The phase names mirror the real loop's combined-backward path
(multi_sae_trainer.py:670-744):
    after_data_fetch -> after_scale_to_device -> after_zero_grad_start
    -> after_forward_all -> after_combined_backward -> after_post_backward
    -> after_optimizer_step -> [optional] after_empty_cache

Run as::

    python3 -m scripts.case_study_three_layer_memory \\
        --output results/memory_model/case_study/three_layer.json
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import Adam

from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

MB = 1024**2
DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}
DTYPE_NAME = {"bf16": "bfloat16", "fp32": "float32"}
DTYPE_BYTES = {"bf16": 2, "fp32": 4}


@dataclass
class PhaseRecord:
    phase: str
    allocated_mb: float
    reserved_mb: float
    device_used_mb: float
    # peak over the interval since the previous phase (peak stats reset each snap)
    peak_allocated_mb: float
    peak_reserved_mb: float
    # running max of device_used across the whole step so far
    device_watermark_mb: float
    # the two gaps (current)
    reserved_minus_allocated_mb: float
    device_minus_reserved_mb: float


class StepProfiler:
    """Snapshot the three layers at named phases with sync + peak reset.

    Each ``record(phase)`` call:
      1. synchronizes the device (kernels land, counters settle),
      2. reads allocated/reserved (torch) and device_used (driver),
      3. reads peak_allocated/peak_reserved accumulated SINCE the last record,
      4. resets the peak stats so the next interval starts clean.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.idx = device.index if device.index is not None else 0
        self.records: list[PhaseRecord] = []
        self._device_watermark = 0.0
        torch.cuda.synchronize(self.idx)
        torch.cuda.reset_peak_memory_stats(self.idx)

    def record(self, phase: str) -> None:
        torch.cuda.synchronize(self.idx)
        allocated = torch.cuda.memory_allocated(self.idx) / MB
        reserved = torch.cuda.memory_reserved(self.idx) / MB
        peak_alloc = torch.cuda.max_memory_allocated(self.idx) / MB
        peak_reserved = torch.cuda.max_memory_reserved(self.idx) / MB
        free_b, total_b = torch.cuda.mem_get_info(self.idx)
        device_used = (total_b - free_b) / MB
        self._device_watermark = max(self._device_watermark, device_used)
        self.records.append(
            PhaseRecord(
                phase=phase,
                allocated_mb=round(allocated, 2),
                reserved_mb=round(reserved, 2),
                device_used_mb=round(device_used, 2),
                peak_allocated_mb=round(peak_alloc, 2),
                peak_reserved_mb=round(peak_reserved, 2),
                device_watermark_mb=round(self._device_watermark, 2),
                reserved_minus_allocated_mb=round(reserved - allocated, 2),
                device_minus_reserved_mb=round(device_used - reserved, 2),
            )
        )
        torch.cuda.reset_peak_memory_stats(self.idx)


def _build_saes(
    *, n_hooks: int, d_in: int, d_sae: int, k: int, dtype: str, device: torch.device
) -> list[TopKTrainingSAE]:
    saes: list[TopKTrainingSAE] = []
    for _ in range(n_hooks):
        cfg = TopKTrainingSAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            dtype=DTYPE_NAME[dtype],
            device=str(device),
            normalize_activations="none",
            apply_b_dec_to_input=False,
            decoder_init_norm=0.1,
            rescale_acts_by_decoder_norm=True,
        )
        sae = TopKTrainingSAE(cfg)
        sae.to(device=device, dtype=DTYPE_MAP[dtype])
        sae.train()
        saes.append(sae)
    return saes


def _make_optimizer(saes: list[TopKTrainingSAE], optim_mode: str) -> Adam:
    params = [p for sae in saes for p in sae.parameters()]
    kwargs: dict = dict(lr=1e-4)
    if optim_mode == "for_loop":
        kwargs.update(foreach=False, fused=False)
    elif optim_mode == "foreach":
        kwargs.update(foreach=True, fused=False)
    elif optim_mode == "fused":
        kwargs.update(fused=True)
    else:
        raise ValueError(f"unknown optim_mode {optim_mode!r}")
    return Adam(params, **kwargs)


def _forward_one(sae: TopKTrainingSAE, sae_in: torch.Tensor):
    return sae.training_forward_pass(
        TrainStepInput(
            sae_in=sae_in,
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )


def run_step(
    *,
    saes: list[TopKTrainingSAE],
    optimizer: Adam,
    raw_batch: torch.Tensor,
    device: torch.device,
    do_empty_cache: bool,
    prof: StepProfiler,
) -> None:
    """One combined-backward training step, mirroring multi_sae_trainer.

    ``raw_batch`` stands in for the on-device activation batch yielded by the
    streaming data provider. We keep a per-hook ``.to(device)`` copy so
    after_scale_to_device reflects the scaled on-device batch the trainer holds.
    """
    # after_data_fetch: the raw batch is the new live tensor this step.
    prof.record("after_data_fetch")

    # after_scale_to_device: scaled on-device copy of each hook's batch.
    # The scalar multiply mimics ActivationScaler and forces a real new
    # allocation (a bare .to(device) on an already-on-device tensor is a no-op).
    scaled_batch = [raw_batch.to(device) * 1.0 for _ in saes]
    prof.record("after_scale_to_device")

    # after_zero_grad_start: grads dropped to None before forward.
    optimizer.zero_grad(set_to_none=True)
    prof.record("after_zero_grad_start")

    # after_forward_all: forward every hook, retaining the loss graph.
    outputs = [_forward_one(sae, scaled_batch[i]) for i, sae in enumerate(saes)]
    prof.record("after_forward_all")

    # after_combined_backward: backward over the summed loss.
    total_loss = sum(o.loss for o in outputs)
    total_loss.backward()
    prof.record("after_combined_backward")

    # after_post_backward: grads now dense & persistent; the graph is freed.
    del outputs, total_loss
    prof.record("after_post_backward")

    # after_optimizer_step: Adam state materializes (exp_avg, exp_avg_sq).
    optimizer.step()
    prof.record("after_optimizer_step")

    # after_empty_cache: optionally hand empty blocks back to the driver.
    if do_empty_cache:
        gc.collect()
        torch.cuda.empty_cache()
        prof.record("after_empty_cache")


def _print_table(records: list[PhaseRecord]) -> None:
    hdr = (
        f"{'phase':<24}{'alloc':>9}{'reserved':>10}{'device':>9}"
        f"{'res-alloc':>11}{'dev-res':>9}{'pk_alloc':>10}{'dev_wm':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        print(
            f"{r.phase:<24}{r.allocated_mb:>9.0f}{r.reserved_mb:>10.0f}"
            f"{r.device_used_mb:>9.0f}{r.reserved_minus_allocated_mb:>11.0f}"
            f"{r.device_minus_reserved_mb:>9.0f}{r.peak_allocated_mb:>10.0f}"
            f"{r.device_watermark_mb:>9.0f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-in", type=int, default=4096)
    ap.add_argument("--d-sae", type=int, default=65536)
    ap.add_argument("--n-hooks", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPE_MAP))
    ap.add_argument("--optim-mode", default="for_loop",
                    choices=["fused", "foreach", "for_loop"])
    ap.add_argument("--warmup-steps", type=int, default=2,
                    help="steps run before profiling (warm cuBLAS/allocator, "
                         "materialize Adam state) so the profiled step is steady-state")
    ap.add_argument("--empty-cache", action="store_true",
                    help="add an after_empty_cache phase that returns blocks to driver")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/memory_model/case_study/three_layer.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This case study requires a CUDA device.")
    torch.cuda.set_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    db = DTYPE_BYTES[args.dtype]
    closed_form = {
        "param_total_mb": round(
            (2 * args.d_in * args.d_sae + args.d_sae + args.d_in)
            * db / MB * args.n_hooks, 1),
        "param_biggest_mb": round(args.d_in * args.d_sae * db / MB, 1),
        "batch_per_hook_mb": round(args.batch * args.d_in * db / MB, 1),
        "hidden_acts_per_hook_mb": round(args.batch * args.d_sae * db / MB, 1),
    }

    # Capture the CUDA-context floor: the device_used after the first tensor
    # touches the GPU but before any model is built. This is the dev-res term's
    # irreducible base (context + primary kernels), independent of the SAE.
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device.index or 0)
    free_b, total_b = torch.cuda.mem_get_info(device.index or 0)
    ctx_floor_mb = round((total_b - free_b) / MB, 1)
    del _
    gc.collect()
    torch.cuda.empty_cache()

    saes = _build_saes(
        n_hooks=args.n_hooks, d_in=args.d_in, d_sae=args.d_sae,
        k=args.k, dtype=args.dtype, device=device,
    )
    optimizer = _make_optimizer(saes, args.optim_mode)
    raw_batch = torch.randn(args.batch, args.d_in, device=device,
                            dtype=DTYPE_MAP[args.dtype])

    # Warm-up: run full steps so cuBLAS workspaces, the allocator pool, and Adam
    # state all exist before we profile. The profiled step is then steady-state
    # (state-create transients won't masquerade as step cost).
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = [_forward_one(sae, raw_batch.to(device))
                   for sae in saes]
        total = sum(o.loss for o in outputs)
        total.backward()
        optimizer.step()
        del outputs, total
    gc.collect()
    torch.cuda.synchronize(device.index or 0)

    prof = StepProfiler(device)
    run_step(
        saes=saes,
        optimizer=optimizer,
        raw_batch=raw_batch,
        device=device,
        do_empty_cache=args.empty_cache,
        prof=prof,
    )

    config = {
        "d_in": args.d_in,
        "d_sae": args.d_sae,
        "n_hooks": args.n_hooks,
        "batch": args.batch,
        "k": args.k,
        "dtype": args.dtype,
        "optim_mode": args.optim_mode,
        "tp": 1,
        "warmup_steps": args.warmup_steps,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_context_floor_mb": ctx_floor_mb,
    }
    print("\n=== config ===")
    print(json.dumps(config, indent=2))
    print("\n=== closed-form tensor sizes (MB) ===")
    print(json.dumps(closed_form, indent=2))
    print(f"\nCUDA-context floor (device_used before model build): "
          f"{ctx_floor_mb:.0f} MB\n")
    print("=== per-phase three-layer table (MB) ===")
    _print_table(prof.records)

    args.output.write_text(json.dumps({
        "config": config,
        "closed_form_mb": closed_form,
        "phases": [asdict(r) for r in prof.records],
    }, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
