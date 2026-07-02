from __future__ import annotations

from pathlib import Path

import torch

from sae_lens.vllm_model import (
    _collect_and_clear_vllm_memory_probe,
    _collect_vllm_static_memory,
    _install_vllm_memory_probe,
    _maybe_record_vllm_memory_timeline,
    _start_vllm_memory_timeline,
    _stop_vllm_memory_timeline,
    write_vllm_memory_records,
    write_vllm_static_memory_record,
)


class _Handle:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _Module:
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        self.output = output
        self.pre_hooks = []
        self.post_hooks = []
        self.handles: list[_Handle] = []

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)
        handle = _Handle()
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, hook):
        self.post_hooks.append(hook)
        handle = _Handle()
        self.handles.append(handle)
        return handle

    def run(self) -> None:
        for hook in self.pre_hooks:
            hook(self, ())
        for hook in self.post_hooks:
            hook(self, (), self.output)


class _Worker:
    def __init__(self, *, omit: set[str] | None = None) -> None:
        output = torch.zeros((2, 3, 4))
        omit = omit or set()
        all_modules = {
            "model.layers.7.input_layernorm": _Module(output),
            "model.layers.7.self_attn.qkv_proj": _Module(output),
            "model.layers.7.self_attn.attn": _Module((output, None)),
            "model.layers.7.self_attn.o_proj": _Module(output),
            "model.layers.7.post_attention_layernorm": _Module(output),
            "model.layers.7.mlp.gate_up_proj": _Module(output),
            "model.layers.7.mlp.act_fn": _Module(output),
            "model.layers.7.mlp.down_proj": _Module(output),
        }
        self.modules = {k: v for k, v in all_modules.items() if k not in omit}

    def get_submodule(self, path: str) -> _Module:
        if path not in self.modules:
            raise AttributeError(path)
        return self.modules[path]


def test_vllm_memory_probe_collects_substage_records(monkeypatch) -> None:
    worker = _Worker()
    calls = {"reset": 0, "sync": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.__setitem__("sync", calls["sync"] + 1))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.__setitem__("reset", calls["reset"] + 1))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 100 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 120 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 130 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 150 * 1024**2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (850 * 1024**2, 1000 * 1024**2))

    assert _install_vllm_memory_probe(worker, layer_idx=7) == 8

    for module in worker.modules.values():
        module.run()

    records = _collect_and_clear_vllm_memory_probe(worker)

    assert [r["substage"] for r in records] == [
        "ln1",
        "attn_qkv",
        "attn_core",
        "attn_o",
        "ln2",
        "mlp_gate_up",
        "mlp_act",
        "mlp_down",
    ]
    assert records[0] == {
        "layer": 7,
        "substage": "ln1",
        "allocated_mb": 100.0,
        "reserved_mb": 120.0,
        "peak_allocated_mb": 130.0,
        "peak_reserved_mb": 150.0,
        "device_mb": 150.0,
        "out_shape": [2, 3, 4],
    }
    assert calls["reset"] == 8
    assert calls["sync"] >= 16


def test_vllm_memory_probe_skips_missing_submodules(monkeypatch) -> None:
    # An architecture that does not expose split qkv/o or the act_fn module:
    # the probe should install hooks only on the submodules that exist.
    worker = _Worker(
        omit={
            "model.layers.7.self_attn.qkv_proj",
            "model.layers.7.self_attn.o_proj",
            "model.layers.7.mlp.act_fn",
        }
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (0, 0))

    assert _install_vllm_memory_probe(worker, layer_idx=7) == 5

    for module in worker.modules.values():
        module.run()

    records = _collect_and_clear_vllm_memory_probe(worker)
    assert [r["substage"] for r in records] == [
        "ln1",
        "attn_core",
        "ln2",
        "mlp_gate_up",
        "mlp_down",
    ]


def test_write_vllm_memory_records_adds_rank_and_step(tmp_path: Path) -> None:
    path = tmp_path / "vllm_memory_history_rank3.jsonl"

    write_vllm_memory_records(
        path,
        records=[
            {
                "layer": 7,
                "substage": "attn",
                "allocated_mb": 1.0,
                "reserved_mb": 2.0,
            }
        ],
        step=5,
        n_training_samples=4096,
        rank=3,
        producer_idx=1,
        vllm_tp_rank=0,
    )

    assert path.read_text() == (
        '{"step": 5, "n_training_samples": 4096, "rank": 3, '
        '"producer_idx": 1, "vllm_tp_rank": 0, "layer": 7, '
        '"substage": "attn", "allocated_mb": 1.0, "reserved_mb": 2.0}\n'
    )


class _Runner:
    def __init__(self, *, model_memory_usage: int, kv_caches: list) -> None:
        self.model_memory_usage = model_memory_usage
        self.kv_caches = kv_caches


class _StaticWorker:
    def __init__(self, *, runner, non_torch_memory: int) -> None:
        self.model_runner = runner
        self.non_torch_memory = non_torch_memory


def _patch_cuda_for_static(monkeypatch, *, allocated, reserved, free, total) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: reserved)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (free, total))


def test_collect_vllm_static_memory_breakdown(monkeypatch) -> None:
    mb = 1024**2

    class _FakeCudaTensor:
        """Stand-in for a KV cache tensor living on CUDA."""

        def __init__(self, nbytes: int) -> None:
            self._nbytes = nbytes
            self.device = torch.device("cuda", 0)

        def numel(self) -> int:
            return self._nbytes // 4

        def element_size(self) -> int:
            return 4

    monkeypatch.setattr(torch, "is_tensor", lambda x: isinstance(x, _FakeCudaTensor))

    kv = [_FakeCudaTensor(int(10 * mb)), _FakeCudaTensor(int(10 * mb))]
    runner = _Runner(model_memory_usage=int(100 * mb), kv_caches=kv)
    worker = _StaticWorker(runner=runner, non_torch_memory=int(5 * mb))

    _patch_cuda_for_static(
        monkeypatch,
        allocated=int(130 * mb),
        reserved=int(140 * mb),
        free=int(800 * mb),
        total=int(1000 * mb),
    )

    rec = _collect_vllm_static_memory(worker)
    assert rec["weights_mb"] == 100.0
    assert rec["kv_cache_mb"] == 20.0
    assert rec["non_torch_mb"] == 5.0
    assert rec["allocated_mb"] == 130.0
    assert rec["reserved_mb"] == 140.0
    assert rec["device_mb"] == 200.0  # total - free


def test_collect_vllm_static_memory_missing_attrs_degrades(monkeypatch) -> None:
    # A worker whose runner lacks kv_caches / model_memory_usage entirely.
    class _Bare:
        pass

    monkeypatch.setattr(torch, "is_tensor", lambda x: False)
    worker = _Bare()
    _patch_cuda_for_static(
        monkeypatch, allocated=0, reserved=0, free=0, total=0
    )
    rec = _collect_vllm_static_memory(worker)
    assert rec["weights_mb"] == 0.0
    assert rec["kv_cache_mb"] == 0.0
    assert rec["non_torch_mb"] == 0.0


def test_write_vllm_static_memory_record(tmp_path: Path) -> None:
    path = tmp_path / "vllm_static_memory_rank0.jsonl"
    write_vllm_static_memory_record(
        path,
        record={"weights_mb": 100.0, "kv_cache_mb": 20.0, "non_torch_mb": 5.0},
        step=3,
        n_training_samples=2048,
        rank=0,
        producer_idx=None,
        vllm_tp_rank=None,
    )
    assert path.read_text() == (
        '{"step": 3, "n_training_samples": 2048, "rank": 0, '
        '"producer_idx": null, "vllm_tp_rank": null, '
        '"weights_mb": 100.0, "kv_cache_mb": 20.0, "non_torch_mb": 5.0}\n'
    )


def test_vllm_memory_timeline_records_target_step(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "vllm_cache_memory_timeline_rank0_step3.pickle"
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: calls.append(("sync", device)),
    )

    class _Memory:
        def _record_memory_history(self, **kwargs):
            calls.append(("record", kwargs))

        def _dump_snapshot(self, snapshot_path: str) -> None:
            calls.append(("dump", snapshot_path))

    monkeypatch.setattr(torch.cuda, "memory", _Memory())

    with _maybe_record_vllm_memory_timeline(
        enabled=True,
        target_step=3,
        current_step=3,
        path=path,
    ):
        calls.append(("body", None))

    assert calls == [
        ("sync", 0),
        (
            "record",
            {"max_entries": 1_000_000, "stacks": "all", "context": "all"},
        ),
        ("body", None),
        ("sync", 0),
        ("dump", str(path)),
        ("record", {"enabled": None}),
    ]


def test_vllm_memory_timeline_ignores_non_target_step(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class _Memory:
        def _record_memory_history(self, **kwargs):
            calls.append(("record", kwargs))

        def _dump_snapshot(self, snapshot_path: str) -> None:
            calls.append(("dump", snapshot_path))

    monkeypatch.setattr(torch.cuda, "memory", _Memory())

    with _maybe_record_vllm_memory_timeline(
        enabled=True,
        target_step=3,
        current_step=2,
        path=tmp_path / "timeline.pickle",
    ):
        calls.append(("body", None))

    assert calls == [("body", None)]


def test_vllm_memory_timeline_worker_uses_tp_rank_suffix(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: calls.append(("sync", device)),
    )
    monkeypatch.setattr(
        "sae_lens.vllm_model._get_vllm_tp_rank",
        lambda: 2,
    )

    class _Memory:
        def _record_memory_history(self, **kwargs):
            calls.append(("record", kwargs))

        def _dump_snapshot(self, snapshot_path: str) -> None:
            calls.append(("dump", snapshot_path))

    monkeypatch.setattr(torch.cuda, "memory", _Memory())

    class _Model:
        pass

    model = _Model()
    base_path = tmp_path / "vllm_cache_memory_timeline_rank0.pickle"

    assert _start_vllm_memory_timeline(
        model,
        path=base_path,
        suffix_tp_rank=True,
    ) == str(tmp_path / "vllm_cache_memory_timeline_rank0_vllm_tp2.pickle")
    assert _stop_vllm_memory_timeline(model) == str(
        tmp_path / "vllm_cache_memory_timeline_rank0_vllm_tp2.pickle"
    )

    assert calls == [
        ("sync", 0),
        (
            "record",
            {"max_entries": 1_000_000, "stacks": "all", "context": "all"},
        ),
        ("sync", 0),
        (
            "dump",
            str(tmp_path / "vllm_cache_memory_timeline_rank0_vllm_tp2.pickle"),
        ),
        ("record", {"enabled": None}),
    ]
