"""Helpers for capturing PyTorch CUDA memory snapshots of the Hooked vLLM
activation-generation path.

The goal of this module is to make it possible to record a *real* CUDA memory
snapshot (the format consumed by https://pytorch.org/memory_viz) that covers:

1. vLLM model weights,
2. the minimized KV cache, and
3. the activation capture buffers that ``HookedVLLMModel.run_with_cache``
   returns and keeps alive.

Recording MUST start before the model is loaded so the allocation stacks for
weights and KV cache are present in the snapshot. These helpers only wrap the
``torch.cuda.memory._record_memory_history`` / ``_dump_snapshot`` API and do
byte-accounting over live tensors; the orchestration lives in
``scripts/profile_hooked_vllm_memory.py``.

None of this is SAE-training specific — it deliberately does not import the SAE
trainer, activations store, or cache runner.
"""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

# Top-level keys a genuine PyTorch CUDA memory snapshot must contain. Used by
# the validator and the tests.
REQUIRED_SNAPSHOT_KEYS: tuple[str, ...] = (
    "segments",
    "device_traces",
    "allocator_settings",
    "external_annotations",
)

# Stage names emitted via torch.profiler.record_function. They land in the
# snapshot's ``external_annotations`` when recording is started with
# ``global_record_annotations=True``.
STAGE_MODEL_LOAD = "vllm_model_load"
STAGE_KV_CACHE_INIT = "vllm_kv_cache_init"
STAGE_RUN_WITH_CACHE = "vllm_run_with_cache"
STAGE_CAPTURE_MERGE = "vllm_capture_merge"
STAGE_SNAPSHOT_DUMP = "vllm_snapshot_dump"


# ---------------------------------------------------------------------------
# Recording lifecycle
# ---------------------------------------------------------------------------


def start_memory_history(max_entries: int = 1_000_000) -> None:
    """Start recording the full CUDA allocation history.

    Must be called *before* any model weights or KV cache are allocated so the
    allocation stacks for those blocks appear in the dumped snapshot. Passes
    ``global_record_annotations=True`` when the installed PyTorch supports it so
    ``torch.profiler.record_function`` stage markers reach the snapshot's
    ``external_annotations``; older PyTorch silently records without them.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("start_memory_history requires CUDA")

    kwargs: dict[str, Any] = {
        "enabled": "all",
        "context": "all",
        "stacks": "all",
        "max_entries": max_entries,
    }
    params = inspect.signature(
        torch.cuda.memory._record_memory_history
    ).parameters
    if "global_record_annotations" in params:
        kwargs["global_record_annotations"] = True
    torch.cuda.memory._record_memory_history(**kwargs)


def dump_memory_snapshot(path: str | Path) -> None:
    """Synchronize and dump the current CUDA allocation history to ``path``.

    Recording is intentionally left running so subsequent cumulative snapshots
    (after weight load, after KV init, after capture) share one lifecycle. Call
    :func:`stop_memory_history` once, at the very end.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(str(out_path))


def stop_memory_history() -> None:
    """Stop CUDA allocation-history recording."""
    torch.cuda.memory._record_memory_history(enabled=None)


# ---------------------------------------------------------------------------
# Live-tensor byte accounting
# ---------------------------------------------------------------------------


def iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Recursively yield every ``torch.Tensor`` found in a nested structure.

    Handles tensors, lists, tuples, sets, and dicts (values only). Anything
    else is ignored. Used to extract activation tensors from whatever nested
    container ``run_with_cache`` returns.
    """
    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from iter_tensors(item)
        return


def unique_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Sum the bytes of distinct CUDA storages backing ``tensors``.

    Deduplicates by the underlying storage ``(data_ptr, nbytes)`` rather than
    Python object ``id`` or tensor ``id``, so views/slices sharing one storage
    are counted once. Non-CUDA tensors are skipped — this models GPU memory
    only, per the profiling boundary (CPU cache buffers are out of scope).
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def sum_unique_cuda_storage_bytes(obj: Any) -> int:
    """Total distinct CUDA storage bytes over every tensor nested in ``obj``."""
    return unique_storage_bytes(iter_tensors(obj))


# ---------------------------------------------------------------------------
# Worker-side collectors (run inside the vLLM worker via collective_rpc)
#
# These are module-level so they are standard-picklable for TP>1 spawn workers.
# ---------------------------------------------------------------------------


def collect_worker_weight_kv_bytes(worker: Any) -> dict[str, int]:
    """Return weight / KV-cache byte totals for one vLLM worker.

    Runs inside the worker process (``self`` is the worker, which owns
    ``model_runner``). ``weight_bytes`` prefers vLLM's own measured value
    (``model_runner.model_memory_usage``); ``weight_bytes_param_scan`` is an
    independent storage-deduped scan of parameters + persistent buffers for
    cross-validation. ``kv_cache_bytes`` sums the distinct CUDA storages in
    ``model_runner.kv_caches`` (deduped by storage, not tensor ``id``).
    Best-effort: a missing attribute yields 0 rather than raising.
    """
    runner = getattr(worker, "model_runner", None)

    weight_bytes = int(getattr(runner, "model_memory_usage", 0) or 0)

    model = getattr(runner, "model", None)
    param_tensors: list[torch.Tensor] = []
    if isinstance(model, torch.nn.Module):
        param_tensors.extend(model.parameters())
        param_tensors.extend(
            b for b in model.buffers() if isinstance(b, torch.Tensor)
        )
    weight_bytes_param_scan = unique_storage_bytes(param_tensors)

    kv_caches = getattr(runner, "kv_caches", None) or []
    kv_cache_bytes = unique_storage_bytes(
        t for t in kv_caches if torch.is_tensor(t)
    )

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    return {
        "weight_bytes": weight_bytes,
        "weight_bytes_param_scan": weight_bytes_param_scan,
        "kv_cache_bytes": kv_cache_bytes,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


# ---------------------------------------------------------------------------
# Snapshot validation
# ---------------------------------------------------------------------------

# Substrings that indicate an allocation stack belongs to the vLLM
# activation-generation path (weights / KV cache / capture).
VLLM_STACK_MARKERS: tuple[str, ...] = (
    "vllm",
    "gpu_worker",
    "gpu_model_runner",
    "load_model",
    "initialize_kv_cache",
    "initialize_from_config",
    "vllm_model.py",
    "run_with_cache",
)

# Substrings that would indicate the WRONG path (SAE training) dominates.
SAE_TRAINER_STACK_MARKERS: tuple[str, ...] = (
    "multi_sae_trainer.py",
    "llm_sae_training_runner.py",
    "activations_store.py",
    "sae_trainer.py",
)


def _iter_frame_strings(snapshot: dict[str, Any]) -> Iterator[str]:
    """Yield frame filenames AND function names across all block stacks.

    Both are yielded because markers of interest span both fields: ``vllm`` /
    ``gpu_worker`` / ``vllm_model.py`` are filenames, while ``load_model`` /
    ``run_with_cache`` / ``initialize_kv_cache`` are function names.
    """
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            frames = block.get("frames") or []
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                filename = frame.get("filename")
                if filename:
                    yield str(filename)
                name = frame.get("name")
                if name:
                    yield str(name)


def validate_snapshot(path: str | Path) -> dict[str, Any]:
    """Load and validate a dumped CUDA memory snapshot.

    Checks that the pickle loads, has the required top-level keys, and that its
    allocation stacks are dominated by vLLM paths rather than SAE-training
    paths. Returns a report dict; ``report["ok"]`` is the overall verdict.
    """
    path = Path(path)
    report: dict[str, Any] = {
        "path": str(path),
        "loaded": False,
        "missing_keys": [],
        "num_segments": 0,
        "vllm_marker_hits": {},
        "sae_marker_hits": {},
        "has_vllm_stacks": False,
        "sae_dominates": False,
        "ok": False,
    }

    with path.open("rb") as f:
        snapshot = pickle.load(f)
    report["loaded"] = True

    report["missing_keys"] = [
        key for key in REQUIRED_SNAPSHOT_KEYS if key not in snapshot
    ]
    report["num_segments"] = len(snapshot.get("segments", []))

    vllm_hits: dict[str, int] = {m: 0 for m in VLLM_STACK_MARKERS}
    sae_hits: dict[str, int] = {m: 0 for m in SAE_TRAINER_STACK_MARKERS}
    for frame_str in _iter_frame_strings(snapshot):
        lowered = frame_str.lower()
        for marker in VLLM_STACK_MARKERS:
            if marker in lowered:
                vllm_hits[marker] += 1
        for marker in SAE_TRAINER_STACK_MARKERS:
            if marker in lowered:
                sae_hits[marker] += 1

    report["vllm_marker_hits"] = vllm_hits
    report["sae_marker_hits"] = sae_hits
    total_vllm = sum(vllm_hits.values())
    total_sae = sum(sae_hits.values())
    report["has_vllm_stacks"] = total_vllm > 0
    report["sae_dominates"] = total_sae > total_vllm and total_sae > 0

    report["ok"] = (
        not report["missing_keys"]
        and report["has_vllm_stacks"]
        and not report["sae_dominates"]
    )
    return report
