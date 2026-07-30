"""Env-gated, non-invasive instrumentation of the vLLM v1 KV block manager.

This module monkeypatches a handful of vLLM v1 scheduler / model-runner methods
at runtime so we can log the *logical* KV block state per scheduler iteration
and count *physical* KV tensor allocations — WITHOUT editing the vendored vLLM
fork under ``third_party/vllm``.

It is entirely disabled unless ``SAELENS_KV_POOL_INSTRUMENT=1`` is set in the
environment; :func:`maybe_install` is a no-op otherwise. Nothing here changes
vLLM scheduling behaviour: the patches only read state and append JSONL records
around the original (unmodified) method calls.

What it captures per ``Scheduler.schedule()`` iteration, for each scheduled
request:

* request status, ``num_computed_tokens``, ``num_scheduled_tokens``
* the request's block count before/after allocation, per KV cache group
* pool totals: ``num_gpu_blocks``, free blocks, used blocks
* whether any request was preempted this iteration
* physical KV storage bytes + deduped storage data_ptrs
* ``torch.cuda.memory_allocated / reserved``

And a monotonically increasing ``kv_physical_allocation_call_count`` incremented
each time vLLM allocates KV cache tensors (``initialize_kv_cache_tensors``),
which is the decisive evidence for whether the pool is ever expanded at runtime.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

ENV_FLAG = "SAELENS_KV_POOL_INSTRUMENT"
ENV_LOG_BLOCK_IDS = "SAELENS_KV_LOG_BLOCK_IDS"


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def _log_block_ids() -> bool:
    return os.environ.get(ENV_LOG_BLOCK_IDS, "0") == "1"


# ---------------------------------------------------------------------------
# Shared, process-global recorder state. A single recorder is used because the
# patched vLLM methods are unbound and have no other channel to reach it.
# ---------------------------------------------------------------------------


class KVPoolRecorder:
    """Collects per-iteration events and physical-allocation counts."""

    def __init__(self, events_path: str | Path | None) -> None:
        self.events_path = Path(events_path) if events_path is not None else None
        if self.events_path is not None:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any stale file so each run starts clean.
            self.events_path.write_text("")
        self.iteration = 0
        self.kv_physical_allocation_call_count = 0
        self.preemption_count = 0
        self.max_used_blocks = 0
        self.max_running_requests = 0
        self.events: list[dict[str, Any]] = []

    def write_event(self, record: dict[str, Any]) -> None:
        self.events.append(record)
        if self.events_path is not None:
            with self.events_path.open("a") as f:
                json.dump(record, f)
                f.write("\n")


_RECORDER: KVPoolRecorder | None = None


def get_recorder() -> KVPoolRecorder | None:
    return _RECORDER


def reset_recorder(events_path: str | Path | None) -> KVPoolRecorder:
    global _RECORDER
    _RECORDER = KVPoolRecorder(events_path)
    return _RECORDER


# ---------------------------------------------------------------------------
# Physical KV storage accounting (deduped by underlying CUDA storage)
# ---------------------------------------------------------------------------


def kv_storage_stats(kv_caches: Any) -> dict[str, Any]:
    """Deduped ``(data_ptr, nbytes)`` accounting over a list of KV tensors.

    ``kv_caches`` is ``model_runner.kv_caches`` (a list of per-layer CUDA
    tensors, many of which may be views into a small number of shared
    storages). Deduping by storage — not tensor ``id`` — gives the true
    physical byte footprint and the distinct backing pointers.
    """
    seen: set[tuple[int, int]] = set()
    total_bytes = 0
    data_ptrs: list[int] = []
    for tensor in kv_caches or []:
        if not torch.is_tensor(tensor) or tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        total_bytes += storage.nbytes()
        data_ptrs.append(storage.data_ptr())
    return {
        "kv_storage_bytes": total_bytes,
        "kv_storage_data_ptrs": sorted(data_ptrs),
        "kv_storage_count": len(data_ptrs),
    }


def cuda_mem_stats() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"cuda_allocated": 0, "cuda_reserved": 0, "cuda_max_allocated": 0}
    device = torch.cuda.current_device()
    return {
        "cuda_allocated": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved": int(torch.cuda.memory_reserved(device)),
        "cuda_max_allocated": int(torch.cuda.max_memory_allocated(device)),
    }


def _pool_state(scheduler: Any) -> dict[str, int]:
    """Read total/free/used blocks from the scheduler's block pool."""
    try:
        pool = scheduler.kv_cache_manager.block_pool
        total = int(pool.num_gpu_blocks)
        free = int(pool.get_num_free_blocks())
        return {
            "total_blocks": total,
            "free_blocks": free,
            "used_blocks": total - free,
        }
    except Exception:
        return {"total_blocks": 0, "free_blocks": 0, "used_blocks": 0}


def _request_block_counts(scheduler: Any, request_id: str) -> list[int]:
    """Per-KV-cache-group block-table lengths for one request."""
    try:
        groups = scheduler.kv_cache_manager.coordinator.get_blocks(request_id)
        return [len(group) for group in groups]
    except Exception:
        return []


def _request_block_ids(scheduler: Any, request_id: str) -> list[list[int]]:
    """Per-group block IDs for one request (for reuse analysis)."""
    try:
        groups = scheduler.kv_cache_manager.coordinator.get_blocks(request_id)
        return [[blk.block_id for blk in group] for group in groups]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Monkeypatch installation
#
# The KV *tensors* live on the GPUModelRunner (worker side); the scheduler owns
# the *logical* block pool. For TP1 / external_launcher both live in this
# process, so we stash a weak-ish module-global handle to the model_runner when
# its KV cache is initialized, and read kv_storage_stats from it during
# scheduling.
# ---------------------------------------------------------------------------

# Set by the patched initialize_kv_cache_tensors so schedule() can read the
# physical KV tensors. A plain reference is fine — the runner outlives the run.
_MODEL_RUNNER: Any = None


def _kv_storage_stats_from_runner() -> dict[str, Any]:
    if _MODEL_RUNNER is None:
        return {"kv_storage_bytes": 0, "kv_storage_data_ptrs": [], "kv_storage_count": 0}
    return kv_storage_stats(getattr(_MODEL_RUNNER, "kv_caches", None))


def maybe_install() -> bool:
    """Install the instrumentation patches if the env flag is set.

    Idempotent and safe to call multiple times. Returns True if patches are
    (now or already) installed, False if instrumentation is disabled.
    """
    if not is_enabled():
        return False

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("kv_pool_instrument: could not import vLLM classes: %r", exc)
        return False

    if getattr(Scheduler, "_sae_kv_instrumented", False):
        return True

    _patch_scheduler_schedule(Scheduler)
    _patch_scheduler_preempt(Scheduler)
    _patch_runner_kv_alloc(GPUModelRunner)

    Scheduler._sae_kv_instrumented = True  # type: ignore[attr-defined]
    return True


def _patch_runner_kv_alloc(GPUModelRunner: Any) -> None:
    orig = GPUModelRunner.initialize_kv_cache_tensors

    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _MODEL_RUNNER
        result = orig(self, *args, **kwargs)
        _MODEL_RUNNER = self
        recorder = get_recorder()
        if recorder is not None:
            recorder.kv_physical_allocation_call_count += 1
            stats = kv_storage_stats(getattr(self, "kv_caches", None))
            logger.info(
                "kv_instrument: initialize_kv_cache_tensors call #%d: "
                "%d storages, %d bytes",
                recorder.kv_physical_allocation_call_count,
                stats["kv_storage_count"],
                stats["kv_storage_bytes"],
            )
        return result

    GPUModelRunner.initialize_kv_cache_tensors = patched  # type: ignore[method-assign]


def _patch_scheduler_preempt(Scheduler: Any) -> None:
    orig = Scheduler._preempt_request

    def patched(self: Any, request: Any, timestamp: float) -> Any:
        recorder = get_recorder()
        if recorder is not None:
            recorder.preemption_count += 1
        return orig(self, request, timestamp)

    Scheduler._preempt_request = patched  # type: ignore[method-assign]


def _patch_scheduler_schedule(Scheduler: Any) -> None:
    orig = Scheduler.schedule

    def patched(self: Any) -> Any:
        recorder = get_recorder()
        if recorder is None:
            return orig(self)

        # Snapshot per-request block counts BEFORE this iteration's allocation.
        before_counts = {
            rid: sum(_request_block_counts(self, rid))
            for rid in list(self.requests.keys())
        }
        preemptions_before = recorder.preemption_count

        output = orig(self)

        num_scheduled = getattr(output, "num_scheduled_tokens", {}) or {}
        pool = _pool_state(self)
        kv_stats = _kv_storage_stats_from_runner()
        mem = cuda_mem_stats()
        preempted_this_iter = recorder.preemption_count > preemptions_before

        recorder.max_used_blocks = max(
            recorder.max_used_blocks, pool["used_blocks"]
        )
        recorder.max_running_requests = max(
            recorder.max_running_requests, len(self.running)
        )

        scheduled_ids = list(num_scheduled.keys())
        # Emit one record per scheduled request; if nothing scheduled, emit a
        # single pool-level record so idle/blocked iterations are still visible.
        targets = scheduled_ids if scheduled_ids else [None]
        for rid in targets:
            request = self.requests.get(rid) if rid is not None else None
            after_group_counts = (
                _request_block_counts(self, rid) if rid is not None else []
            )
            after_count = sum(after_group_counts)
            before_count = before_counts.get(rid, 0)
            record = {
                "iteration": recorder.iteration,
                "request_id": rid,
                "request_status": (
                    str(getattr(request, "status", None)) if request else None
                ),
                "num_computed_tokens": (
                    int(getattr(request, "num_computed_tokens", 0))
                    if request
                    else 0
                ),
                "num_scheduled_tokens": int(num_scheduled.get(rid, 0)),
                "num_new_blocks_requested": max(0, after_count - before_count),
                "request_block_count_before": before_count,
                "request_block_count_after": after_count,
                "request_block_counts_per_group": after_group_counts,
                "request_block_ids": (
                    _request_block_ids(self, rid)
                    if (rid is not None and _log_block_ids())
                    else None
                ),
                "num_running_requests": len(self.running),
                "num_waiting_requests": len(self.waiting),
                "preempted": preempted_this_iter,
                "preemption_count": recorder.preemption_count,
                "kv_physical_allocation_call_count": (
                    recorder.kv_physical_allocation_call_count
                ),
                **pool,
                **kv_stats,
                **mem,
            }
            recorder.write_event(record)

        recorder.iteration += 1
        return output

    Scheduler.schedule = patched  # type: ignore[method-assign]
