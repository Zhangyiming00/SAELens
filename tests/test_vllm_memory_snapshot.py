import pickle

import pytest
import torch

from sae_lens.vllm_memory_snapshot import (
    REQUIRED_SNAPSHOT_KEYS,
    iter_tensors,
    sum_unique_cuda_storage_bytes,
    unique_storage_bytes,
    validate_snapshot,
)


def test_iter_tensors_recurses_nested_containers():
    a = torch.zeros(2)
    b = torch.ones(3)
    c = torch.arange(4)
    nested = {"x": a, "y": [b, {"z": (c,)}], "ignored": "not a tensor"}
    found = list(iter_tensors(nested))
    assert len(found) == 3
    # identity, not just count: the exact tensors are yielded
    assert any(t is a for t in found)
    assert any(t is b for t in found)
    assert any(t is c for t in found)


def test_iter_tensors_ignores_non_tensor_scalars():
    assert list(iter_tensors(5)) == []
    assert list(iter_tensors("hello")) == []
    assert list(iter_tensors(None)) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unique_storage_bytes_dedups_by_storage():
    # A base tensor and a view share one storage -> counted once.
    base = torch.zeros(1024, dtype=torch.float32, device="cuda")  # 4096 bytes
    view = base[:512]
    other = torch.zeros(256, dtype=torch.float32, device="cuda")  # 1024 bytes

    # base + view share storage; only base's 4096 bytes should count for them.
    assert unique_storage_bytes([base, view]) == base.untyped_storage().nbytes()
    # Adding a distinct storage adds its bytes.
    total = unique_storage_bytes([base, view, other])
    assert total == (
        base.untyped_storage().nbytes() + other.untyped_storage().nbytes()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unique_storage_bytes_skips_cpu_tensors():
    cpu_t = torch.zeros(1000, dtype=torch.float32)
    cuda_t = torch.zeros(1000, dtype=torch.float32, device="cuda")
    assert unique_storage_bytes([cpu_t]) == 0
    assert unique_storage_bytes([cpu_t, cuda_t]) == cuda_t.untyped_storage().nbytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sum_unique_cuda_storage_bytes_over_nested_dict():
    t1 = torch.zeros(1024, dtype=torch.float32, device="cuda")
    t2 = torch.zeros(2048, dtype=torch.float32, device="cuda")
    acts = {"hook_a": t1, "hook_b": [t2]}
    expected = (
        t1.untyped_storage().nbytes() + t2.untyped_storage().nbytes()
    )
    assert sum_unique_cuda_storage_bytes(acts) == expected


def _make_snapshot(frame_filenames: list[str]) -> dict:
    """Build a minimal but structurally-real memory snapshot dict."""
    return {
        "segments": [
            {
                "blocks": [
                    {
                        "frames": [
                            {"filename": fn, "name": "some_fn", "line": 1}
                        ]
                    }
                    for fn in frame_filenames
                ]
            }
        ],
        "device_traces": [],
        "allocator_settings": {},
        "external_annotations": [],
    }


def test_validate_snapshot_accepts_vllm_dominated(tmp_path):
    snap = _make_snapshot(
        [
            "/x/vllm/v1/worker/gpu_worker.py",
            "/x/vllm/v1/worker/gpu_model_runner.py",
            "/x/sae_lens/vllm_model.py",
        ]
    )
    path = tmp_path / "good.pickle"
    path.write_bytes(pickle.dumps(snap))

    report = validate_snapshot(path)
    assert report["loaded"] is True
    assert report["missing_keys"] == []
    assert report["has_vllm_stacks"] is True
    assert report["sae_dominates"] is False
    assert report["ok"] is True


def test_validate_snapshot_rejects_sae_dominated(tmp_path):
    snap = _make_snapshot(
        [
            "/x/sae_lens/training/multi_sae_trainer.py",
            "/x/sae_lens/llm_sae_training_runner.py",
            "/x/sae_lens/training/activations_store.py",
        ]
    )
    path = tmp_path / "bad.pickle"
    path.write_bytes(pickle.dumps(snap))

    report = validate_snapshot(path)
    assert report["has_vllm_stacks"] is False
    assert report["sae_dominates"] is True
    assert report["ok"] is False


def test_validate_snapshot_rejects_missing_keys(tmp_path):
    snap = _make_snapshot(["/x/vllm/v1/worker/gpu_worker.py"])
    del snap["device_traces"]
    del snap["allocator_settings"]
    path = tmp_path / "incomplete.pickle"
    path.write_bytes(pickle.dumps(snap))

    report = validate_snapshot(path)
    assert set(report["missing_keys"]) == {"device_traces", "allocator_settings"}
    assert report["ok"] is False


def test_required_snapshot_keys_match_memory_viz_contract():
    assert set(REQUIRED_SNAPSHOT_KEYS) == {
        "segments",
        "device_traces",
        "allocator_settings",
        "external_annotations",
    }
