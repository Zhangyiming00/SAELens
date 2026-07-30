import pytest
import torch

from sae_lens import kv_pool_instrument
from sae_lens.kv_pool_instrument import (
    KVPoolRecorder,
    cuda_mem_stats,
    is_enabled,
    kv_storage_stats,
    maybe_install,
)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(kv_pool_instrument.ENV_FLAG, raising=False)
    assert is_enabled() is False
    # maybe_install must be a no-op (return False) when disabled.
    assert maybe_install() is False


def test_enabled_flag(monkeypatch):
    monkeypatch.setenv(kv_pool_instrument.ENV_FLAG, "1")
    assert is_enabled() is True
    monkeypatch.setenv(kv_pool_instrument.ENV_FLAG, "0")
    assert is_enabled() is False


def test_recorder_writes_jsonl(tmp_path):
    events = tmp_path / "events.jsonl"
    recorder = KVPoolRecorder(events)
    recorder.write_event({"iteration": 0, "used_blocks": 3})
    recorder.write_event({"iteration": 1, "used_blocks": 5})
    lines = events.read_text().strip().splitlines()
    assert len(lines) == 2
    import json

    assert json.loads(lines[0])["used_blocks"] == 3
    assert json.loads(lines[1])["iteration"] == 1
    # in-memory copy mirrors the file
    assert len(recorder.events) == 2


def test_recorder_truncates_stale_file(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text("STALE\nDATA\n")
    KVPoolRecorder(events)
    assert events.read_text() == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_storage_stats_dedups_shared_storage():
    # Two tensors that are views of one storage must count once; a third,
    # distinct storage adds its own bytes. This mirrors how vLLM's kv_caches
    # list holds many per-layer views over few backing tensors.
    base = torch.zeros(4096, dtype=torch.int8, device="cuda")
    view_a = base[:1024]
    view_b = base[1024:2048]
    distinct = torch.zeros(512, dtype=torch.int8, device="cuda")

    stats = kv_storage_stats([view_a, view_b])
    assert stats["kv_storage_bytes"] == base.untyped_storage().nbytes()
    assert stats["kv_storage_count"] == 1

    stats2 = kv_storage_stats([view_a, view_b, distinct])
    assert stats2["kv_storage_count"] == 2
    assert stats2["kv_storage_bytes"] == (
        base.untyped_storage().nbytes() + distinct.untyped_storage().nbytes()
    )
    # data_ptrs are returned sorted and distinct
    assert stats2["kv_storage_data_ptrs"] == sorted(
        set(stats2["kv_storage_data_ptrs"])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_storage_stats_skips_cpu():
    cpu_t = torch.zeros(1000, dtype=torch.int8)
    assert kv_storage_stats([cpu_t]) == {
        "kv_storage_bytes": 0,
        "kv_storage_data_ptrs": [],
        "kv_storage_count": 0,
    }


def test_kv_storage_stats_empty():
    assert kv_storage_stats(None)["kv_storage_bytes"] == 0
    assert kv_storage_stats([])["kv_storage_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_mem_stats_keys():
    stats = cuda_mem_stats()
    assert set(stats) == {"cuda_allocated", "cuda_reserved", "cuda_max_allocated"}
    assert all(isinstance(v, int) for v in stats.values())
