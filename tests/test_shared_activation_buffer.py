"""Tests for SharedActivationBuffer.

Each test uses tmp_path as base_dir so nothing touches /dev/shm.
Tests verify functional correctness of the state machine, quota, and data round-trip.
"""

import numpy as np
import pytest
import shutil
import torch

from sae_lens.training.shared_activation_buffer import ChunkState, SharedActivationBuffer


def _make_buffer(
    tmp_path,
    num_chunks: int = 8,
    chunk_size_tokens: int = 100,
    d_model: int = 16,
    num_producers: int = 1,
    target_chunks: int = 10,
    create: bool = True,
) -> SharedActivationBuffer:
    return SharedActivationBuffer(
        name="test_buf",
        num_chunks=num_chunks,
        chunk_size_tokens=chunk_size_tokens,
        d_model=d_model,
        num_producers=num_producers,
        target_chunks=target_chunks,
        create=create,
        base_dir=str(tmp_path),
    )


def test_single_producer_consumer_roundtrip(tmp_path):
    buf = _make_buffer(tmp_path)
    d_model = 16
    rows = 64

    acts = torch.randn(rows, d_model, dtype=torch.bfloat16)
    result = buf.allocate_write_chunk()
    assert result is not None
    chunk_idx, seq_no = result
    assert seq_no == 0

    buf.write_chunk(chunk_idx, acts, valid_tokens=rows)
    buf.mark_ready(chunk_idx)

    indices, _ = buf.acquire_up_to(1)
    assert indices == [chunk_idx]

    recovered, valid = buf.read_chunk(chunk_idx)
    assert valid == rows
    assert recovered.shape == (rows, d_model)
    assert recovered.dtype == torch.bfloat16
    # Exact bfloat16 round-trip via uint16 storage
    assert torch.equal(recovered, acts)

    buf.release_chunk(chunk_idx)
    counts = buf.queue_counts()
    assert counts["free"] >= 1
    buf.close()


def test_partial_chunk_valid_tokens(tmp_path):
    buf = _make_buffer(tmp_path, chunk_size_tokens=100)
    rows = 50
    acts = torch.randn(rows, 16, dtype=torch.bfloat16)

    result = buf.allocate_write_chunk()
    assert result is not None
    chunk_idx, _ = result
    buf.write_chunk(chunk_idx, acts, valid_tokens=rows)
    buf.mark_ready(chunk_idx)

    indices, _ = buf.acquire_up_to(1)
    recovered, valid = buf.read_chunk(indices[0])
    assert valid == rows
    assert recovered.shape == (rows, 16)
    assert torch.equal(recovered, acts)
    buf.release_chunk(indices[0])
    buf.close()


def test_global_quota_stops_at_target(tmp_path):
    # target_chunks=3; allocating 5 times should return None after 3
    buf = _make_buffer(tmp_path, num_chunks=8, target_chunks=3)

    allocated = []
    for _ in range(5):
        result = buf.allocate_write_chunk()
        if result is not None:
            allocated.append(result)

    assert len(allocated) == 3
    seq_nos = [r[1] for r in allocated]
    assert seq_nos == [0, 1, 2]
    buf.close()


def test_abort_write_chunk_frees_slot(tmp_path):
    buf = _make_buffer(tmp_path, num_chunks=2, target_chunks=4)

    r1 = buf.allocate_write_chunk()
    r2 = buf.allocate_write_chunk()
    assert r1 is not None and r2 is not None
    chunk_idx1, _ = r1
    chunk_idx2, _ = r2

    # Abort the first chunk (WRITING → FREE)
    buf.abort_write_chunk(chunk_idx1)

    # Now a third allocation should succeed (reuses freed slot)
    r3 = buf.allocate_write_chunk()
    assert r3 is not None
    chunk_idx3, seq_no3 = r3
    assert seq_no3 == 2  # global seq continues from 2
    assert chunk_idx3 == chunk_idx1  # same physical slot reused

    buf.close()


def test_acquire_up_to_returns_partial_when_less_than_n_ready(tmp_path):
    # Only 1 chunk is READY; acquire_up_to(3) must return immediately with 1
    buf = _make_buffer(tmp_path, num_chunks=8, target_chunks=4)

    r = buf.allocate_write_chunk()
    assert r is not None
    chunk_idx, _ = r
    acts = torch.randn(10, 16, dtype=torch.bfloat16)
    buf.write_chunk(chunk_idx, acts, valid_tokens=10)
    buf.mark_ready(chunk_idx)

    indices, _ = buf.acquire_up_to(3)
    # Must return the 1 available chunk immediately, not block waiting for 3
    assert len(indices) == 1
    assert indices[0] == chunk_idx
    buf.release_chunk(indices[0])
    buf.close()


def test_acquire_stops_when_done_and_no_ready(tmp_path):
    buf = _make_buffer(tmp_path, num_chunks=4, target_chunks=2, num_producers=1)
    buf.signal_done()

    with pytest.raises(StopIteration):
        buf.acquire_up_to(1)

    buf.close()


def test_acquire_drains_tail_before_stop_iteration(tmp_path):
    # 2 READY chunks, then signal_done(). First acquire should return both; second raises.
    buf = _make_buffer(tmp_path, num_chunks=8, target_chunks=4, num_producers=1)

    acts = torch.randn(10, 16, dtype=torch.bfloat16)
    for _ in range(2):
        r = buf.allocate_write_chunk()
        assert r is not None
        chunk_idx, _ = r
        buf.write_chunk(chunk_idx, acts, valid_tokens=10)
        buf.mark_ready(chunk_idx)

    buf.signal_done()

    # First acquire should get both READY chunks
    indices, _ = buf.acquire_up_to(3)
    assert len(indices) == 2
    for i in indices:
        buf.release_chunk(i)

    # Second acquire: no READY and done → StopIteration
    with pytest.raises(StopIteration):
        buf.acquire_up_to(3)

    buf.close()


def test_bf16_bit_pattern_preserved(tmp_path):
    # Write specific bfloat16 values and verify exact bit-pattern preservation
    buf = _make_buffer(tmp_path, chunk_size_tokens=16, d_model=8)

    # Use values that are exactly representable in bfloat16
    acts = torch.tensor(
        [[1.0, -2.0, 0.5, 3.5, -1.5, 0.25, 100.0, -0.125]] * 16,
        dtype=torch.bfloat16,
    )
    r = buf.allocate_write_chunk()
    assert r is not None
    chunk_idx, _ = r
    buf.write_chunk(chunk_idx, acts, valid_tokens=16)
    buf.mark_ready(chunk_idx)

    indices, _ = buf.acquire_up_to(1)
    recovered, valid = buf.read_chunk(indices[0])

    assert valid == 16
    # int16 view compares the raw bits
    assert torch.equal(
        recovered.view(torch.int16),
        acts.view(torch.int16),
    )
    buf.release_chunk(indices[0])
    buf.close()


def test_destroy_removes_all_backing_files(tmp_path):
    buf = _make_buffer(tmp_path)
    expected = [
        tmp_path / "test_buf_state.bin",
        tmp_path / "test_buf_meta.bin",
        tmp_path / "test_buf_header.bin",
        tmp_path / "test_buf_data.bin",
        tmp_path / "test_buf.lock",
    ]
    assert all(path.exists() for path in expected)

    buf.destroy()

    assert not any(path.exists() for path in expected)


def test_cleanup_files_removes_stale_buffer_without_open_handle(tmp_path):
    buf = _make_buffer(tmp_path)
    buf.close()
    expected = [
        tmp_path / "test_buf_state.bin",
        tmp_path / "test_buf_meta.bin",
        tmp_path / "test_buf_header.bin",
        tmp_path / "test_buf_data.bin",
        tmp_path / "test_buf.lock",
    ]
    assert all(path.exists() for path in expected)

    SharedActivationBuffer.cleanup_files("test_buf", base_dir=str(tmp_path))

    assert not any(path.exists() for path in expected)


def test_create_failure_removes_partial_backing_files(tmp_path, monkeypatch):
    original_create_file = SharedActivationBuffer._create_file
    created: list[str] = []

    def fail_on_data(path, n_elements, dtype):
        created.append(path.name)
        if path.name.endswith("_data.bin"):
            raise RuntimeError("simulated data allocation failure")
        return original_create_file(path, n_elements, dtype)

    monkeypatch.setattr(SharedActivationBuffer, "_create_file", staticmethod(fail_on_data))

    with pytest.raises(RuntimeError, match="simulated data allocation failure"):
        SharedActivationBuffer(
            name="test_buf",
            num_chunks=2,
            chunk_size_tokens=4,
            d_model=4,
            num_producers=1,
            create=True,
            base_dir=str(tmp_path),
        )

    assert any(name.endswith("_data.bin") for name in created)
    assert not any(tmp_path.iterdir())


def test_space_check_includes_metadata_before_creating_files(tmp_path, monkeypatch):
    data_bytes = 1 * 1 * 1 * 2
    enough_for_data_only = data_bytes + 1024 * 1024
    usage = shutil._ntuple_diskusage(10 * 1024 * 1024, 0, enough_for_data_only)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage)

    with pytest.raises(RuntimeError, match="Insufficient space"):
        SharedActivationBuffer(
            name="test_buf",
            num_chunks=1,
            chunk_size_tokens=1,
            d_model=1,
            num_producers=1,
            create=True,
            base_dir=str(tmp_path),
        )

    assert not any(tmp_path.iterdir())


def test_preflight_estimate_matches_backing_file_sizes():
    estimate = SharedActivationBuffer.estimate_required_bytes(
        num_chunks=3,
        chunk_size_tokens=5,
        d_model=7,
        dtype=torch.bfloat16,
    )

    assert estimate["data"] == 3 * 5 * 7 * 2
    assert estimate["metadata"] == 3 * np.dtype(np.int8).itemsize + 3 * 6 * np.dtype(np.int32).itemsize + 8 * np.dtype(np.int32).itemsize
    assert estimate["headroom"] == 1024 * 1024
    assert estimate["required"] == estimate["data"] + estimate["metadata"] + estimate["headroom"]
