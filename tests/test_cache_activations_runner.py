import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset
from tqdm import trange
from transformer_lens import HookedTransformer

from sae_lens.cache_activations_runner import (
    CacheActivationsRunner,
    _is_consolidation_artifact,
)
from sae_lens.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)
from sae_lens.constants import DTYPE_MAP
from sae_lens.load_model import load_model
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import sanitize_hook_name_for_path
from tests.helpers import assert_close


def _default_cfg(
    tmp_path: Path,
    batch_size: int = 16,
    context_size: int = 8,
    dataset_num_rows: int = 128,
    n_buffers: int = 4,
    shuffle: bool = False,
    **kwargs: Any,
) -> CacheActivationsRunnerConfig:
    d_in = 512
    dtype = "float32"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    sliced_context_size = kwargs.get("seqpos_slice")
    if sliced_context_size is not None:
        sliced_context_size = len(range(context_size)[slice(*sliced_context_size)])
    else:
        sliced_context_size = context_size

    # Calculate buffer_size_gb to achieve desired n_buffers
    bytes_per_token = d_in * DTYPE_MAP[dtype].itemsize
    tokens_per_buffer = math.ceil(dataset_num_rows * sliced_context_size / n_buffers)
    buffer_size_gb = (tokens_per_buffer * bytes_per_token) / 1_000_000_000
    total_training_tokens = dataset_num_rows * sliced_context_size

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        ### Parameters
        training_tokens=total_training_tokens,
        model_batch_size=batch_size,
        buffer_size_gb=buffer_size_gb,
        context_size=context_size,
        ###
        d_in=d_in,
        shuffle=shuffle,
        prepend_bos=False,
        device=device,
        seed=42,
        dtype=dtype,
        **kwargs,
    )
    assert cfg.n_buffers == n_buffers
    assert cfg.n_seq_in_dataset == dataset_num_rows
    assert (
        cfg.n_tokens_in_buffer
        == cfg.n_batches_in_buffer * batch_size * sliced_context_size
    )
    return cfg


def test_vllm_memory_sidecars_are_consolidation_artifacts(tmp_path: Path) -> None:
    artifact_names = [
        ".tmp_shards",
        "vllm_memory_history_rank0.jsonl",
        "vllm_static_memory_rank0.jsonl",
        "vllm_cache_memory_timeline_rank0.pickle",
        "vllm_cache_memory_timeline_rank0_vllm_tp1.pickle",
    ]

    for name in artifact_names:
        path = tmp_path / name
        if name == ".tmp_shards":
            path.mkdir()
        else:
            path.write_text("")
        assert _is_consolidation_artifact(path)

    non_artifact = tmp_path / "data-00000-of-00001.arrow"
    non_artifact.write_text("")
    assert not _is_consolidation_artifact(non_artifact)


# The way to run this with this command:
# poetry run py.test tests/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    dataset = runner.run()

    assert cfg.n_seq_in_dataset == len(dataset)
    assert dataset.column_names == [cfg.hook_name, "token_ids"]

    features = dataset.features
    assert isinstance(features[cfg.hook_name], datasets.Array2D)
    assert features[cfg.hook_name].shape == (cfg.context_size, cfg.d_in)
    assert isinstance(features["token_ids"], datasets.Sequence)
    assert features["token_ids"].length == cfg.context_size


def test_load_cached_activations(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    runner.run()

    model = HookedTransformer.from_pretrained(cfg.model_name)

    activations_store = ActivationsStore.from_config(model, cfg)

    # get_raw_llm_batch returns one batch at a time
    # Total batches = n_buffers * n_batches_in_buffer
    for _ in range(cfg.n_buffers * cfg.n_batches_in_buffer):
        buffer = activations_store.get_raw_llm_batch()
        assert buffer[0].shape == (
            activations_store.store_batch_size_prompts * cfg.context_size,
            cfg.d_in,
        )
        assert buffer[1] is not None
        assert buffer[1].shape == (
            activations_store.store_batch_size_prompts * cfg.context_size,
        )


def test_cache_activations_runner_to_string():
    cfg = _default_cfg(Path("tmp_path"))
    runner = CacheActivationsRunner(cfg)
    result = str(runner)

    # Check that the string contains the expected summary format
    assert "Activation Cache Runner:" in result
    assert "Total training tokens: 1024" in result
    assert "Number of buffers: 4" in result
    assert "Tokens per buffer: 256" in result
    assert "Disk space required: 0.00 GB" in result
    assert "Configuration:" in result

    # Check that the config contains expected fields
    assert "dataset_path='chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests'" in result
    assert "model_name='gelu-1l'" in result
    assert "hook_name='blocks.0.hook_mlp_out'" in result
    assert "training_tokens=1024" in result
    assert "context_size=8" in result
    assert "d_in=512" in result


def test_activations_store_refreshes_dataset_when_it_runs_out(tmp_path: Path):
    context_size = 8
    n_batches_in_buffer = 4
    store_batch_size = 1
    total_training_steps = 4
    batch_size = 4
    total_training_tokens = total_training_steps * batch_size

    cache_cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cache_cfg)
    runner.run()

    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_in=512, d_sae=768),
        cached_activations_path=str(tmp_path),
        use_cached_activations=True,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,
        training_tokens=total_training_tokens // 2,
        train_batch_size_tokens=8,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        device="cpu",
        seed=42,
        dtype="float16",
    )

    class MockModel:
        def to_tokens(self, *args: tuple[Any, ...], **kwargs: Any) -> torch.Tensor:
            return torch.ones(context_size)

        @property
        def W_E(self) -> torch.Tensor:
            return torch.ones(16, 16)

        @property
        def cfg(self) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
            return cfg

    dataset = Dataset.from_list([{"text": "hello world1"}] * 64)

    model = MockModel()
    activations_store = ActivationsStore.from_config(
        model,  # type: ignore
        cfg,
        override_dataset=dataset,
    )
    for _ in range(16):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=True)

    # assert a stop iteration is raised when we do one more get_batch_tokens

    pytest.raises(
        StopIteration,
        activations_store.get_batch_tokens,
        batch_size,
        raise_at_epoch_end=True,
    )

    # no errors are ever raised if we do not ask for raise_at_epoch_end
    for _ in range(32):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=False)


def test_compare_cached_activations_end_to_end_with_ground_truth(tmp_path: Path):
    """
    Creates activations using CacheActivationsRunner and compares them with ground truth
    model.run_with_cache
    """

    torch.manual_seed(42)
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    activation_dataset = runner.run()
    activation_dataset.set_format("torch")
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
    token_dataset: Dataset = load_dataset(
        cfg.dataset_path, split=f"train[:{cfg.n_seq_in_dataset}]"
    )  # type: ignore
    token_dataset.set_format("torch", device=cfg.device)

    ground_truth_acts = []
    for i in trange(0, cfg.n_seq_in_dataset, cfg.model_batch_size):
        tokens = token_dataset[i : i + cfg.model_batch_size]["input_ids"][
            :, : cfg.context_size
        ]
        _, layerwise_activations = model.run_with_cache(
            tokens,
            names_filter=[cfg.hook_name],
        )
        acts = layerwise_activations[cfg.hook_name]
        ground_truth_acts.append(acts)

    ground_truth_acts = torch.cat(ground_truth_acts, dim=0).cpu()

    dataset_acts_tensor = torch.tensor(np.array(dataset_acts))
    assert_close(ground_truth_acts, dataset_acts_tensor, rtol=1e-3, atol=5e-2)


def test_load_activations_store_with_nonexistent_dataset(tmp_path: Path):
    cfg = _default_cfg(tmp_path)

    model = load_model(
        model_class_name=cfg.model_class_name,
        model_name=cfg.model_name,
        device=cfg.device,
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

    # Attempt to load from a non-existent dataset
    with pytest.raises(
        FileNotFoundError,
        match="is neither a `Dataset` directory nor a `DatasetDict` directory.",
    ):
        ActivationsStore.from_config(model, cfg)


def test_cache_activations_runner_with_nonempty_directory(tmp_path: Path):
    # Create a file to make the directory non-empty
    with open(tmp_path / "some_file.txt", "w") as f:
        f.write("test")

    with pytest.raises(
        Exception, match="is not empty. Please delete it or specify a different path."
    ):
        cfg = _default_cfg(tmp_path)
        runner = CacheActivationsRunner(cfg)
        runner.run()


def test_cache_activations_runner_with_incorrect_d_in(tmp_path: Path):
    correct_cfg = _default_cfg(tmp_path)

    # d_in different from hook
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513

    runner = CacheActivationsRunner(wrong_d_in_cfg)
    with pytest.raises(
        RuntimeError,
        match=r"The expanded size of the tensor \(513\) must match the existing size \(512\) at non-singleton dimension 2.",
    ):
        runner.run()


def test_cache_activations_runner_load_dataset_with_incorrect_config(tmp_path: Path):
    correct_cfg = _default_cfg(tmp_path, context_size=16)
    runner = CacheActivationsRunner(correct_cfg)
    runner.run()
    model = runner.model

    # Context size different from dataset
    wrong_context_size_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_context_size_cfg.context_size = 13

    with pytest.raises(
        ValueError,
        match=r"Given dataset of shape \(16, 512\) does not match context_size \(13\) and d_in \(512\)",
    ):
        ActivationsStore.from_config(model, wrong_context_size_cfg)

    # d_in different from dataset
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513

    with pytest.raises(
        ValueError,
        match=r"Given dataset of shape \(16, 512\) does not match context_size \(16\) and d_in \(513\)",
    ):
        ActivationsStore.from_config(model, wrong_d_in_cfg)

    # Incorrect hook_name
    wrong_hook_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_hook_cfg.hook_name = "blocks.1.hook_mlp_out"

    with pytest.raises(
        ValueError,
        match=r"Columns \['blocks.1.hook_mlp_out'\] not in the dataset. Current columns in the dataset: \['blocks.0.hook_mlp_out'\, 'token_ids'\]",
    ):
        ActivationsStore.from_config(model, wrong_hook_cfg)


def test_cache_activations_runner_with_valid_seqpos(tmp_path: Path):
    cfg = _default_cfg(
        tmp_path,
        batch_size=1,
        context_size=16,
        n_buffers=3,
        dataset_num_rows=12,
        seqpos_slice=(3, -3),
    )
    runner = CacheActivationsRunner(cfg)

    activation_dataset = runner.run()
    activation_dataset.set_format("torch", device=cfg.device)
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    assert os.path.exists(tmp_path)

    # assert that there are n_buffer files in the directory.
    buffer_files = [
        f
        for f in os.listdir(tmp_path)
        if f.startswith("data-") and f.endswith(".arrow")
    ]
    assert len(buffer_files) == cfg.n_buffers

    for act in dataset_acts:
        # should be 16 - 3 - 3 = 10
        assert act.shape == (10, cfg.d_in)


def test_cache_activations_runner_stores_token_ids(tmp_path: Path):
    cfg = _default_cfg(tmp_path)
    runner = CacheActivationsRunner(cfg)
    dataset = runner.run()
    dataset.set_format("torch")

    assert "token_ids" in dataset.features
    token_ids_array = np.array(dataset["token_ids"])
    mlp_out_array = np.array(dataset["blocks.0.hook_mlp_out"])
    assert token_ids_array.shape[1] == cfg.context_size
    assert mlp_out_array.shape[:2] == token_ids_array.shape


def test_cache_activations_runner_multi_hook_writes_split_cache(tmp_path: Path):
    hook_names = ["blocks.0.hook_resid_pre", "blocks.0.hook_mlp_out"]
    cfg = _default_cfg(
        tmp_path,
        batch_size=2,
        context_size=8,
        dataset_num_rows=16,
        n_buffers=2,
        shuffle=True,
    )
    cfg.hook_name = hook_names[0]
    cfg.hook_names = hook_names

    result = CacheActivationsRunner(cfg).run()

    assert isinstance(result, dict)
    assert set(result) == set(hook_names)
    manifest_path = tmp_path / "cache_activations_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "split_hook_cached_activations_v1"
    assert manifest["hook_names"] == hook_names
    for hook_name in hook_names:
        hook_dir = tmp_path / sanitize_hook_name_for_path(hook_name)
        assert hook_dir.exists()
        ds = datasets.load_from_disk(str(hook_dir))
        assert ds.column_names == [hook_name, "token_ids"]
        assert len(ds) == cfg.n_seq_in_dataset


def test_cache_activations_runner_does_not_pad_to_full_buffer(tmp_path: Path):
    context_size = 8
    dataset_num_rows = 21
    batch_size = 16
    d_in = 512
    dtype = "float32"
    buffer_size_gb = (
        batch_size * context_size * d_in * DTYPE_MAP[dtype].itemsize
    ) / 1_000_000_000
    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path / "partial"),
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        training_tokens=dataset_num_rows * context_size,
        model_batch_size=batch_size,
        buffer_size_gb=buffer_size_gb,
        context_size=context_size,
        d_in=d_in,
        shuffle=False,
        prepend_bos=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        dtype=dtype,
    )
    assert cfg.n_seq_in_buffer == 16
    assert cfg.n_buffers == 2

    dataset = CacheActivationsRunner(cfg).run()

    assert len(dataset) == 21
    state = json.loads((tmp_path / "partial" / "state.json").read_text())
    assert len(state["_data_files"]) == 2
    assert state["_data_files"][-1]["filename"] == "data-00001-of-00002.arrow"


def test_cache_activations_runner_dp_does_not_pad_to_full_buffer(
    tmp_path: Path,
):
    context_size = 8
    dataset_num_rows = 22
    batch_size = 8
    d_in = 512
    dtype = "float32"
    buffer_size_gb = (
        batch_size * 2 * context_size * d_in * DTYPE_MAP[dtype].itemsize
    ) / 1_000_000_000
    override_dataset = Dataset.from_dict(
        {
            "tokens": [
                list(range(i * context_size, (i + 1) * context_size))
                for i in range(dataset_num_rows)
            ]
        }
    )
    override_dataset.set_format("torch")

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path / "base"),
        dataset_path="unused",
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        training_tokens=dataset_num_rows * context_size,
        model_batch_size=batch_size,
        buffer_size_gb=buffer_size_gb,
        context_size=context_size,
        d_in=d_in,
        shuffle=False,
        prepend_bos=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        dtype=dtype,
    )
    assert cfg.n_seq_in_buffer == 16

    shard_dirs = []
    for shard_idx in range(2):
        shard_cfg = CacheActivationsRunnerConfig(**dataclasses.asdict(cfg))
        shard_cfg.new_cached_activations_path = str(tmp_path / f"dp{shard_idx}")
        shard_cfg.dataset_shard_index = shard_idx
        shard_cfg.dataset_shard_count = 2
        shard = CacheActivationsRunner(
            shard_cfg,
            override_dataset=override_dataset,
        ).run()
        assert len(shard) == 11
        shard_dirs.append(Path(shard_cfg.new_cached_activations_path))

    merged = CacheActivationsRunner.consolidate_dp_shards(
        shard_dirs,
        tmp_path / "merged",
        shuffle=False,
    )

    assert len(merged) == dataset_num_rows
    state = json.loads((tmp_path / "merged" / "state.json").read_text())
    assert len(state["_data_files"]) == 2
    assert state["_data_files"][-1]["filename"] == "data-00001-of-00002.arrow"


def test_cache_activations_runner_dp_consolidates_split_multi_hook_cache(
    tmp_path: Path,
):
    hook_names = ["blocks.0.hook_resid_pre", "blocks.0.hook_mlp_out"]
    base_cfg = _default_cfg(
        tmp_path / "base",
        batch_size=2,
        context_size=8,
        dataset_num_rows=16,
        n_buffers=2,
        shuffle=False,
    )
    base_cfg.hook_name = hook_names[0]
    base_cfg.hook_names = hook_names

    shard_dirs = []
    for shard_idx in range(2):
        shard_cfg = CacheActivationsRunnerConfig(**dataclasses.asdict(base_cfg))
        shard_cfg.new_cached_activations_path = str(tmp_path / f"dp{shard_idx}")
        shard_cfg.dataset_shard_index = shard_idx
        shard_cfg.dataset_shard_count = 2
        CacheActivationsRunner(shard_cfg).run()
        shard_dirs.append(Path(shard_cfg.new_cached_activations_path))

    merged = CacheActivationsRunner.consolidate_dp_shards(
        shard_dirs,
        tmp_path / "merged",
        shuffle=False,
    )

    assert isinstance(merged, dict)
    assert set(merged) == set(hook_names)
    manifest = json.loads((tmp_path / "merged" / "cache_activations_manifest.json").read_text())
    assert manifest["hook_names"] == hook_names
    for hook_name in hook_names:
        hook_dir = tmp_path / "merged" / sanitize_hook_name_for_path(hook_name)
        ds = datasets.load_from_disk(str(hook_dir))
        assert ds.column_names == [hook_name, "token_ids"]
        assert len(ds) == base_cfg.n_seq_in_dataset


def test_cache_activations_runner_dp_no_shuffle_split_cache_consolidates_without_resaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    hook_names = ["blocks.0.hook_resid_pre", "blocks.0.hook_mlp_out"]
    base_cfg = _default_cfg(
        tmp_path / "base",
        batch_size=2,
        context_size=8,
        dataset_num_rows=16,
        n_buffers=2,
        shuffle=False,
    )
    base_cfg.hook_name = hook_names[0]
    base_cfg.hook_names = hook_names

    shard_dirs = []
    for shard_idx in range(2):
        shard_cfg = CacheActivationsRunnerConfig(**dataclasses.asdict(base_cfg))
        shard_cfg.new_cached_activations_path = str(tmp_path / f"dp{shard_idx}")
        shard_cfg.dataset_shard_index = shard_idx
        shard_cfg.dataset_shard_count = 2
        CacheActivationsRunner(shard_cfg).run()
        shard_dirs.append(Path(shard_cfg.new_cached_activations_path))

    def fail_save_to_disk(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("shuffle=False split DP consolidation must not rewrite")

    monkeypatch.setattr(Dataset, "save_to_disk", fail_save_to_disk)

    merged_dir = tmp_path / "merged"
    merged = CacheActivationsRunner.consolidate_dp_shards(
        shard_dirs,
        merged_dir,
        shuffle=False,
    )

    assert isinstance(merged, dict)
    assert set(merged) == set(hook_names)
    for hook_name in hook_names:
        hook_dir = merged_dir / sanitize_hook_name_for_path(hook_name)
        assert datasets.load_from_disk(str(hook_dir)).num_rows == base_cfg.n_seq_in_dataset


def test_cache_activations_runner_dp_shards_are_disjoint_and_complete(tmp_path: Path):
    context_size = 16
    dataset_num_rows = 16
    override_dataset = Dataset.from_dict(
        {
            "tokens": [
                list(range(i * context_size, (i + 1) * context_size))
                for i in range(dataset_num_rows)
            ]
        }
    )
    override_dataset.set_format("torch")

    full_cfg = _default_cfg(
        tmp_path / "full",
        batch_size=2,
        context_size=context_size,
        dataset_num_rows=dataset_num_rows,
        n_buffers=2,
        shuffle=False,
    )
    full = CacheActivationsRunner(full_cfg, override_dataset=override_dataset).run()
    full.set_format("torch")
    full_tokens = np.array(full["token_ids"])

    shard_dirs = []
    shard_token_sets = []
    for shard_idx in range(2):
        shard_cfg = CacheActivationsRunnerConfig(
            **dataclasses.asdict(full_cfg),
        )
        shard_cfg.new_cached_activations_path = str(tmp_path / f"dp{shard_idx}")
        shard_cfg.dataset_shard_index = shard_idx
        shard_cfg.dataset_shard_count = 2
        shard_runner = CacheActivationsRunner(shard_cfg, override_dataset=override_dataset)
        shard = shard_runner.run()
        shard.set_format("torch")
        shard_dirs.append(Path(shard_cfg.new_cached_activations_path))
        shard_token_sets.append({tuple(row.tolist()) for row in shard["token_ids"]})

    assert shard_token_sets[0].isdisjoint(shard_token_sets[1])

    merged_dir = tmp_path / "merged"
    merged = CacheActivationsRunner.consolidate_dp_shards(shard_dirs, merged_dir, shuffle=False)
    merged.set_format("torch")
    merged_tokens = np.array(merged["token_ids"])

    assert {tuple(row.tolist()) for row in merged_tokens} == {
        tuple(row.tolist()) for row in full_tokens
    }
    assert len(merged_tokens) == len(full_tokens)


def test_cache_activations_runner_dp_no_shuffle_consolidates_without_resaving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context_size = 16
    dataset_num_rows = 16
    override_dataset = Dataset.from_dict(
        {
            "tokens": [
                list(range(i * context_size, (i + 1) * context_size))
                for i in range(dataset_num_rows)
            ]
        }
    )
    override_dataset.set_format("torch")

    cfg = _default_cfg(
        tmp_path / "base",
        batch_size=2,
        context_size=context_size,
        dataset_num_rows=dataset_num_rows,
        n_buffers=2,
        shuffle=False,
    )

    shard_dirs = []
    for shard_idx in range(2):
        shard_cfg = CacheActivationsRunnerConfig(**dataclasses.asdict(cfg))
        shard_cfg.new_cached_activations_path = str(tmp_path / f"dp{shard_idx}")
        shard_cfg.dataset_shard_index = shard_idx
        shard_cfg.dataset_shard_count = 2
        CacheActivationsRunner(shard_cfg, override_dataset=override_dataset).run()
        shard_dirs.append(Path(shard_cfg.new_cached_activations_path))

    def fail_save_to_disk(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("shuffle=False DP consolidation must not rewrite the dataset")

    monkeypatch.setattr(Dataset, "save_to_disk", fail_save_to_disk)

    merged_dir = tmp_path / "merged"
    merged = CacheActivationsRunner.consolidate_dp_shards(
        shard_dirs,
        merged_dir,
        shuffle=False,
    )

    merged.set_format("torch")
    merged_tokens = np.array(merged["token_ids"])
    assert len(merged_tokens) == dataset_num_rows
    assert datasets.load_from_disk(str(merged_dir)).num_rows == dataset_num_rows


def test_cache_activations_runner_dp_handles_partial_final_batch(tmp_path: Path):
    context_size = 16
    dataset_num_rows = 12
    override_dataset = Dataset.from_dict(
        {
            "tokens": [
                list(range(i * context_size, (i + 1) * context_size))
                for i in range(dataset_num_rows)
            ]
        }
    )
    override_dataset.set_format("torch")
    full_cfg = _default_cfg(
        tmp_path / "full",
        batch_size=4,
        context_size=context_size,
        dataset_num_rows=dataset_num_rows,
        n_buffers=3,
        shuffle=False,
    )
    full = CacheActivationsRunner(full_cfg, override_dataset=override_dataset).run()
    expected_local_rows = len(full) // 2

    shard_dirs = []
    for shard_idx in range(2):
        shard_cfg = CacheActivationsRunnerConfig(**dataclasses.asdict(full_cfg))
        shard_cfg.new_cached_activations_path = str(tmp_path / f"partial_dp{shard_idx}")
        shard_cfg.dataset_shard_index = shard_idx
        shard_cfg.dataset_shard_count = 2
        shard = CacheActivationsRunner(
            shard_cfg,
            override_dataset=override_dataset,
        ).run()
        assert len(shard) == expected_local_rows
        shard_dirs.append(Path(shard_cfg.new_cached_activations_path))

    merged = CacheActivationsRunner.consolidate_dp_shards(
        shard_dirs,
        tmp_path / "partial_merged",
        shuffle=False,
    )

    assert len(merged) == len(full)


def test_cache_activations_runner_dp_does_not_shuffle_rank_local_shards(
    tmp_path: Path,
):
    context_size = 16
    dataset_num_rows = 16
    override_dataset = Dataset.from_dict(
        {
            "tokens": [
                list(range(i * context_size, (i + 1) * context_size))
                for i in range(dataset_num_rows)
            ]
        }
    )
    override_dataset.set_format("torch")

    cfg = _default_cfg(
        tmp_path / "dp0",
        batch_size=2,
        context_size=context_size,
        dataset_num_rows=dataset_num_rows,
        n_buffers=2,
        shuffle=True,
    )
    cfg.dataset_shard_index = 0
    cfg.dataset_shard_count = 2

    shard = CacheActivationsRunner(cfg, override_dataset=override_dataset).run()
    shard.set_format("torch")

    assert [tuple(row.tolist()) for row in shard["token_ids"]] == [
        tuple(range(i * context_size, (i + 1) * context_size))
        for i in range(0, dataset_num_rows, 2)
    ]


def test_cache_activations_runner_shuffling(tmp_path: Path):
    """Test that when shuffle=True, activations and token IDs remain aligned after shuffling."""
    # Create test dataset with arbitrary unique tokens
    tokenizer = HookedTransformer.from_pretrained("gelu-1l").tokenizer
    text = "".join(
        [
            " " + word[1:]
            for word in tokenizer.vocab  # type: ignore
            if word[0] == "Ġ" and word[1:].isascii() and word.isalnum()
        ]
    )
    dataset = Dataset.from_list([{"text": text}])

    # Create configs for unshuffled and shuffled versions
    base_cfg = _default_cfg(
        tmp_path / "base",
        context_size=3,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=False,
    )
    shuffle_cfg = _default_cfg(
        tmp_path / "shuffled",
        context_size=3,
        batch_size=2,
        dataset_num_rows=8,
        shuffle=True,
    )

    # Get unshuffled dataset
    unshuffled_runner = CacheActivationsRunner(base_cfg, override_dataset=dataset)
    unshuffled_ds = unshuffled_runner.run()
    unshuffled_ds.set_format("torch")

    # Get shuffled dataset
    shuffled_runner = CacheActivationsRunner(shuffle_cfg, override_dataset=dataset)
    shuffled_ds = shuffled_runner.run()
    shuffled_ds.set_format("torch")

    # Get activations and tokens
    hook_name = base_cfg.hook_name
    unshuffled_acts: torch.Tensor = unshuffled_ds[hook_name]  # type: ignore
    unshuffled_tokens: torch.Tensor = unshuffled_ds["token_ids"]  # type: ignore
    shuffled_acts: torch.Tensor = shuffled_ds[hook_name]  # type: ignore
    shuffled_tokens: torch.Tensor = shuffled_ds["token_ids"]  # type: ignore

    # Verify shapes are preserved
    unshuffled_acts_array = np.array(unshuffled_acts)
    shuffled_acts_array = np.array(shuffled_acts)
    unshuffled_tokens_array = np.array(unshuffled_tokens)
    shuffled_tokens_array = np.array(shuffled_tokens)
    assert unshuffled_acts_array.shape == shuffled_acts_array.shape
    assert unshuffled_tokens_array.shape == shuffled_tokens_array.shape

    # Verify data is actually shuffled
    assert not np.array_equal(unshuffled_acts_array, shuffled_acts_array)
    assert not np.array_equal(unshuffled_tokens_array, shuffled_tokens_array)

    # For each token in unshuffled, find its position in shuffled
    # and verify the activations were moved together
    for i in range(len(unshuffled_tokens_array)):
        token = unshuffled_tokens_array[i]
        # Find where this token went in shuffled version
        shuffled_idx = np.where(shuffled_tokens_array == token)[0][0]
        # Verify activations moved with it
        assert_close(
            torch.from_numpy(unshuffled_acts_array[i]),
            torch.from_numpy(shuffled_acts_array[shuffled_idx]),
        )
