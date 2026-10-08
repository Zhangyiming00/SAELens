"""Epoch-qualified GPU handoffs between separate vLLM and SAE processes.

Quiescence, tensor release and allocator release are different operations.
No peer may infer memory ownership from ``done`` or a process being idle.
CUDA contexts and the fixed SAE communication groups deliberately survive.
"""

from __future__ import annotations

import gc
import json
import time


def read_status(path):
    try:
        status = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if status.get("state") == "failed":
        raise RuntimeError(f"GPU handoff failed ({path.name}): {status.get('error')}")
    return status


def released(status, epoch):
    return bool(
        status
        and status.get("epoch") == epoch
        and status.get("ready") is True
        and status.get("state") in ("released", "done")
        and status.get("memory_released") is True
    )


def producer_quiesced(status, epoch, mode):
    if released(status, epoch):
        return True
    return bool(
        mode == "resident"
        and status
        and status.get("epoch") == epoch
        and status.get("ready") is True
        and status.get("state") == "paused"
    )


def producer_running(status, epoch):
    # A terminal producer remains alive as a CPU control endpoint and ACKs
    # subsequent epochs, but never reloads after the input budget is exhausted.
    return (released(status, epoch) and status["state"] == "done") or bool(
        status
        and status.get("epoch") == epoch
        and status.get("ready") is True
        and status.get("state") == "running"
    )


def cuda_memory(device):
    import torch

    if device.type != "cuda":
        return dict(allocated=0, reserved=0)
    return dict(
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
    )


def release_cuda_memory(device, *, baseline_allocated=0, tolerance_mib=64):
    """Return unused blocks to the driver before another process allocates.

    The allocated-memory guard detects retained PyTorch tensors. It is not a
    claim that NCCL/context/third-party allocations have disappeared from NVML.
    Call only AFTER dropping all model/cache/diagnostic tensor references.
    """
    import torch

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    before = cuda_memory(device)
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
    after = cuda_memory(device)
    limit = baseline_allocated + tolerance_mib * (1 << 20)
    if after["allocated"] > limit:
        raise RuntimeError(
            "GPU handoff retained live tensors: "
            f"allocated={after['allocated']} bytes, allowed={limit}; "
            "memory ownership was not released"
        )
    return dict(
        before=before, after=after, baseline_allocated=baseline_allocated,
        tolerance_mib=tolerance_mib, release_s=time.perf_counter() - started,
    )


def release_inactive_sae(session, *, tolerance_mib=64):
    """Called on a departing worker after migration/validation have returned."""
    state = session.state
    if (
        session.groups.rank in session.groups.active_ranks
        or state.in_step
        or getattr(session, "_window", None) is not None
        or getattr(session, "_microbatch_active", False)
        or any((state.models, state.optimizers, state.replicated, state.activation_caches))
    ):
        raise RuntimeError("SAE release requires an empty, inactive optimizer boundary")
    session.last_loss_components = {}
    return release_cuda_memory(
        session.groups.device,
        baseline_allocated=getattr(session, "handoff_baseline_allocated", 0),
        tolerance_mib=tolerance_mib,
    )
