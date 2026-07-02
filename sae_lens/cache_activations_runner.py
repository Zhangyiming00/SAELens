import io
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import einops
import torch
from datasets import Array2D, Dataset, Features, Sequence, Value, concatenate_datasets
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformer_lens.HookedTransformer import HookedRootModule

from sae_lens import logger
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import sanitize_hook_name_for_path
from sae_lens.util import str_to_dtype


def _is_consolidation_artifact(path: Path) -> bool:
    """Items in the output dir that are NOT cached activations and so must not
    block consolidation: the temp shard dir and any profiling sidecar files
    (the memory probes write ``vllm_memory_history_rank*.jsonl``,
    ``vllm_static_memory_rank*.jsonl``, and
    ``vllm_cache_memory_timeline_rank*.pickle``).
    """
    return (
        path.name == ".tmp_shards"
        or path.name.startswith("vllm_memory_history_rank")
        or path.name.startswith("vllm_static_memory_rank")
        or path.name.startswith("vllm_cache_memory_timeline_rank")
    )


def _mk_activations_store(
    model: HookedRootModule,
    cfg: CacheActivationsRunnerConfig,
    override_dataset: Dataset | None = None,
) -> ActivationsStore:
    """
    Internal method used in CacheActivationsRunner. Used to create a cached dataset
    from a ActivationsStore.
    """
    return ActivationsStore(
        model=model,
        dataset=override_dataset or cfg.dataset_path,
        streaming=cfg.streaming,
        hook_name=cfg.hook_name,
        hook_names=cfg.hook_names,
        hook_head_index=None,
        context_size=cfg.context_size,
        d_in=cfg.d_in,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        total_training_tokens=cfg.training_tokens,
        store_batch_size_prompts=cfg.model_batch_size,
        train_batch_size_tokens=-1,
        prepend_bos=cfg.prepend_bos,
        normalize_activations="none",
        device=torch.device("cpu"),  # since we're saving to disk
        dtype=cfg.dtype,
        cached_activations_path=None,
        model_kwargs=cfg.model_kwargs,
        autocast_lm=cfg.autocast_lm,
        dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        seqpos_slice=cfg.seqpos_slice,
        dataset_shard_index=cfg.dataset_shard_index,
        dataset_shard_count=cfg.dataset_shard_count,
    )


class CacheActivationsRunner:
    def __init__(
        self,
        cfg: CacheActivationsRunnerConfig,
        override_dataset: Dataset | None = None,
    ):
        self.cfg = cfg
        self.model: HookedRootModule = load_model(
            model_class_name=self.cfg.model_class_name,
            model_name=self.cfg.model_name,
            device=self.cfg.device,
            model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
        )
        if self.cfg.compile_llm:
            self.model = torch.compile(self.model, mode=self.cfg.llm_compilation_mode)  # type: ignore
        self.activations_store = _mk_activations_store(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
        )
        self.context_size = self._get_sliced_context_size(
            self.cfg.context_size, self.cfg.seqpos_slice
        )
        self.hook_names = list(self.cfg.hook_names or [self.cfg.hook_name])
        features_dict: dict[str, Array2D | Sequence] = {
            hook_name: Array2D(
                shape=(self.context_size, self.cfg.d_in), dtype=self.cfg.dtype
            )
            for hook_name in self.hook_names
        }
        features_dict["token_ids"] = Sequence(  # type: ignore
            Value(dtype="int32"), length=self.context_size
        )
        self.features = Features(features_dict)

    def __str__(self):
        """
        Print the number of tokens to be cached.
        Print the number of buffers, and the number of tokens per buffer.
        Print the disk space required to store the activations.

        """

        bytes_per_token = (
            self.cfg.d_in * self.cfg.dtype.itemsize
            if isinstance(self.cfg.dtype, torch.dtype)
            else str_to_dtype(self.cfg.dtype).itemsize
        )
        total_training_tokens = self.cfg.n_seq_in_dataset * self.context_size
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {self.cfg.n_buffers}\n"
            f"Tokens per buffer: {self.cfg.n_tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        if not source_dir.exists() or not source_dir.is_dir():
            raise NotADirectoryError(
                f"source_dir is not an existing directory: {source_dir}"
            )

        if not output_dir.exists() or not output_dir.is_dir():
            raise NotADirectoryError(
                f"output_dir is not an existing directory: {output_dir}"
            )

        other_items = [
            p for p in output_dir.iterdir() if not _is_consolidation_artifact(p)
        ]
        if other_items:
            raise FileExistsError(
                f"output_dir must be empty (besides .tmp_shards). Found: {other_items}"
            )

        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        shard_dirs = [
            shard_dir
            for shard_dir in sorted(source_dir.iterdir())
            if shard_dir.name.startswith("shard_")
        ]

        for shard_dir in shard_dirs:
            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(shard_dirs):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not includeing _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @staticmethod
    def consolidate_dp_shards(
        shard_dirs: list[Path],
        output_dir: Path,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Dataset | dict[str, Dataset]:
        """Merge cache-time DP rank outputs into one cached activation dataset."""
        if not shard_dirs:
            raise ValueError("shard_dirs must not be empty")

        output_dir.mkdir(exist_ok=True, parents=True)
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output_dir must be empty. Found: {output_dir}")

        manifest_path = shard_dirs[0] / "cache_activations_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("format") != "split_hook_cached_activations_v1":
                raise ValueError(
                    f"Unsupported split cached activations format in {manifest_path}: "
                    f"{manifest.get('format')}"
                )
            hook_names = manifest["hook_names"]
            hook_to_dir = manifest["hook_to_dir"]
            output_dir.mkdir(exist_ok=True, parents=True)
            if any(output_dir.iterdir()):
                raise FileExistsError(f"output_dir must be empty. Found: {output_dir}")

            merged: dict[str, Dataset] = {}
            out_hook_to_dir: dict[str, str] = {}
            for hook_name in hook_names:
                hook_dir_name = hook_to_dir.get(
                    hook_name, sanitize_hook_name_for_path(hook_name)
                )
                out_hook_dir_name = sanitize_hook_name_for_path(hook_name)
                out_hook_to_dir[hook_name] = out_hook_dir_name
                hook_output_dir = output_dir / out_hook_dir_name
                if shuffle:
                    datasets = [
                        Dataset.load_from_disk(str(shard_dir / hook_dir_name))
                        for shard_dir in shard_dirs
                    ]
                    dataset = concatenate_datasets(datasets).shuffle(seed=seed)
                    dataset.save_to_disk(str(hook_output_dir))
                    merged[hook_name] = Dataset.load_from_disk(str(hook_output_dir))
                else:
                    hook_output_dir.mkdir(parents=True, exist_ok=False)
                    tmp_source_dir = hook_output_dir / ".tmp_shards"
                    tmp_source_dir.mkdir(exist_ok=False, parents=False)
                    for shard_idx, shard_dir in enumerate(shard_dirs):
                        shutil.move(
                            str(shard_dir / hook_dir_name),
                            str(tmp_source_dir / f"shard_{shard_idx:05d}"),
                        )
                    merged[hook_name] = CacheActivationsRunner._consolidate_shards(
                        tmp_source_dir,
                        hook_output_dir,
                        copy_files=False,
                    )

            out_manifest = {
                "format": "split_hook_cached_activations_v1",
                "hook_names": hook_names,
                "hook_to_dir": out_hook_to_dir,
                "token_ids_column": manifest.get("token_ids_column", "token_ids"),
                "dataset_format": manifest.get(
                    "dataset_format", "huggingface_dataset_per_hook"
                ),
            }
            (output_dir / "cache_activations_manifest.json").write_text(
                json.dumps(out_manifest, indent=2) + "\n"
            )
            return merged

        if not shuffle:
            tmp_source_dir = output_dir / ".tmp_shards"
            tmp_source_dir.mkdir(exist_ok=False, parents=False)
            moved_shards: list[tuple[Path, Path]] = []
            try:
                for shard_idx, shard_dir in enumerate(shard_dirs):
                    moved_shard_dir = tmp_source_dir / f"shard_{shard_idx:05d}"
                    shutil.move(
                        str(shard_dir),
                        str(moved_shard_dir),
                    )
                    moved_shards.append((moved_shard_dir, shard_dir))
                return CacheActivationsRunner._consolidate_shards(
                    tmp_source_dir,
                    output_dir,
                    copy_files=False,
                )
            except Exception:
                for moved_shard_dir, original_shard_dir in reversed(moved_shards):
                    if moved_shard_dir.exists() and not original_shard_dir.exists():
                        shutil.move(str(moved_shard_dir), str(original_shard_dir))
                if tmp_source_dir.exists() and not any(tmp_source_dir.iterdir()):
                    tmp_source_dir.rmdir()
                raise

        datasets = [Dataset.load_from_disk(str(shard_dir)) for shard_dir in shard_dirs]
        dataset = concatenate_datasets(datasets)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        dataset.save_to_disk(str(output_dir))
        return Dataset.load_from_disk(str(output_dir))

    @torch.no_grad()
    def run(self) -> Dataset:
        activation_save_path = self.cfg.new_cached_activations_path
        assert activation_save_path is not None

        ### Paths setup
        final_cached_activation_path = Path(activation_save_path)
        final_cached_activation_path.mkdir(exist_ok=True, parents=True)
        if any(final_cached_activation_path.iterdir()):
            raise Exception(
                f"Activations directory ({final_cached_activation_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )

        tmp_cached_activation_path = final_cached_activation_path / ".tmp_shards/"
        tmp_cached_activation_path.mkdir(exist_ok=False, parents=False)

        ### Create temporary sharded datasets

        logger.info(f"Started caching activations for {self.cfg.dataset_path}")

        for i in tqdm(range(self.cfg.n_buffers), desc="Caching activations"):
            try:
                batch_sizes_in_shard = [
                    self.cfg.model_batch_size
                ] * self.cfg.n_batches_in_buffer
                remaining_sequences = (
                    self.cfg.n_seq_in_dataset - i * self.cfg.n_seq_in_buffer
                )
                sequences_in_shard = min(
                    self.cfg.n_seq_in_buffer, remaining_sequences
                )
                if sequences_in_shard <= 0:
                    break
                batch_sizes_in_shard = []
                while sequences_in_shard > 0:
                    batch_size = min(self.cfg.model_batch_size, sequences_in_shard)
                    batch_sizes_in_shard.append(batch_size)
                    sequences_in_shard -= batch_size
                # Accumulate n_batches_in_buffer batches into one shard
                buffers: list[
                    tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None]
                ] = []
                for batch_size in batch_sizes_in_shard:
                    buffers.append(
                        self.activations_store.get_raw_llm_batch(
                            batch_size=batch_size
                        )
                    )
                # Concatenate all batches
                first_acts = buffers[0][0]
                if isinstance(first_acts, dict):
                    acts = {
                        hook_name: torch.cat(
                            [
                                b[0][hook_name]  # type: ignore[index]
                                for b in buffers
                            ],
                            dim=0,
                        )
                        for hook_name in self.hook_names
                    }
                else:
                    acts = torch.cat([b[0] for b in buffers], dim=0)  # type: ignore[list-item]
                token_ids: torch.Tensor | None = None
                if buffers[0][1] is not None:
                    # All batches have token_ids if the first one does
                    token_ids = torch.cat([b[1] for b in buffers], dim=0)  # type: ignore[arg-type]
                shard = self._create_shard((acts, token_ids))
                shard_path = tmp_cached_activation_path / f"shard_{i:05d}"
                if isinstance(shard, dict):
                    shard_path.mkdir(exist_ok=False, parents=False)
                    for hook_name, hook_shard in shard.items():
                        hook_shard.save_to_disk(
                            str(shard_path / sanitize_hook_name_for_path(hook_name)),
                            num_shards=1,
                        )
                else:
                    shard.save_to_disk(str(shard_path), num_shards=1)
                del buffers, acts, token_ids, shard
            except StopIteration:
                logger.warning(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {self.cfg.n_buffers} batches."
                )
                break

        ### Concatenate shards and push to Huggingface Hub

        if self.is_multi_hook:
            dataset = self._consolidate_multi_hook_shards(
                tmp_cached_activation_path,
                final_cached_activation_path,
                copy_files=False,
            )
        else:
            dataset = self._consolidate_shards(
                tmp_cached_activation_path, final_cached_activation_path, copy_files=False
            )

        if self.cfg.shuffle and self.cfg.dataset_shard_count == 1:
            logger.info("Shuffling...")
            if isinstance(dataset, dict):
                shuffled: dict[str, Dataset] = {}
                for hook_name, hook_dataset in dataset.items():
                    hook_dataset = hook_dataset.shuffle(seed=self.cfg.seed)
                    hook_dir = (
                        final_cached_activation_path
                        / sanitize_hook_name_for_path(hook_name)
                    )
                    tmp_hook_dir = final_cached_activation_path / (
                        f".{sanitize_hook_name_for_path(hook_name)}.shuffle_tmp"
                    )
                    if tmp_hook_dir.exists():
                        shutil.rmtree(tmp_hook_dir)
                    hook_dataset.save_to_disk(str(tmp_hook_dir))
                    shutil.rmtree(hook_dir)
                    shutil.move(str(tmp_hook_dir), str(hook_dir))
                    shuffled[hook_name] = Dataset.load_from_disk(str(hook_dir))
                dataset = shuffled
            else:
                dataset = dataset.shuffle(seed=self.cfg.seed)

        if self.cfg.hf_repo_id:
            if isinstance(dataset, dict):
                raise NotImplementedError(
                    "Pushing split multi-hook cached activations to Hugging Face "
                    "Hub is not implemented."
                )
            logger.info("Pushing to Huggingface Hub...")
            dataset.push_to_hub(
                repo_id=self.cfg.hf_repo_id,
                num_shards=self.cfg.hf_num_shards,
                private=self.cfg.hf_is_private_repo,
                revision=self.cfg.hf_revision,
            )

            meta_io = io.BytesIO()
            meta_contents = json.dumps(
                asdict(self.cfg), indent=2, ensure_ascii=False
            ).encode("utf-8")
            meta_io.write(meta_contents)
            meta_io.seek(0)

            api = HfApi()
            api.upload_file(
                path_or_fileobj=meta_io,
                path_in_repo="cache_activations_runner_cfg.json",
                repo_id=self.cfg.hf_repo_id,
                repo_type="dataset",
                commit_message="Add cache_activations_runner metadata",
            )

        return dataset

    def _consolidate_multi_hook_shards(
        self,
        source_dir: Path,
        output_dir: Path,
        copy_files: bool = True,
    ) -> dict[str, Dataset]:
        output_dir.mkdir(exist_ok=True, parents=True)
        other_items = [
            p for p in output_dir.iterdir() if not _is_consolidation_artifact(p)
        ]
        if other_items:
            raise FileExistsError(
                f"output_dir must be empty (besides .tmp_shards). Found: {other_items}"
            )

        datasets_by_hook: dict[str, Dataset] = {}
        hook_to_dir: dict[str, str] = {}
        for hook_name in self.hook_names:
            hook_tmp_dir = source_dir / sanitize_hook_name_for_path(hook_name)
            hook_tmp_dir.mkdir(exist_ok=False, parents=False)
            for shard_dir in sorted(source_dir.iterdir()):
                if not shard_dir.name.startswith("shard_"):
                    continue
                src = shard_dir / sanitize_hook_name_for_path(hook_name)
                dst = hook_tmp_dir / shard_dir.name
                if copy_files:
                    shutil.copytree(src, dst)
                else:
                    shutil.move(str(src), str(dst))

            hook_output_dir_name = sanitize_hook_name_for_path(hook_name)
            hook_to_dir[hook_name] = hook_output_dir_name
            hook_output_dir = output_dir / hook_output_dir_name
            hook_output_dir.mkdir(exist_ok=True, parents=False)
            datasets_by_hook[hook_name] = self._consolidate_shards(
                hook_tmp_dir,
                hook_output_dir,
                copy_files=False,
            )

        manifest = {
            "format": "split_hook_cached_activations_v1",
            "hook_names": self.hook_names,
            "hook_to_dir": hook_to_dir,
            "token_ids_column": "token_ids",
            "dataset_format": "huggingface_dataset_per_hook",
        }
        (output_dir / "cache_activations_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        if not copy_files:
            shutil.rmtree(source_dir)

        return datasets_by_hook

    @property
    def is_multi_hook(self) -> bool:
        return len(self.hook_names) > 1

    def _create_shard(
        self,
        buffer: tuple[
            torch.Tensor | dict[str, torch.Tensor],  # shape: (bs context_size) d_in
            torch.Tensor | None,  # shape: (bs context_size) or None
        ],
    ) -> Dataset | dict[str, Dataset]:
        acts, token_ids = buffer
        if isinstance(acts, dict):
            return {
                hook_name: self._create_single_hook_shard(
                    hook_name,
                    hook_acts,
                    token_ids,
                )
                for hook_name, hook_acts in acts.items()
            }

        return self._create_single_hook_shard(self.cfg.hook_name, acts, token_ids)

    def _create_single_hook_shard(
        self,
        hook_name: str,
        acts: torch.Tensor,
        token_ids: torch.Tensor | None,
    ) -> Dataset:
        n_seq_in_shard = acts.shape[0] // self.context_size
        acts = einops.rearrange(
            acts,
            "(bs context_size) d_in -> bs context_size d_in",
            bs=n_seq_in_shard,
            context_size=self.context_size,
            d_in=self.cfg.d_in,
        )
        shard_dict: dict[str, object] = {hook_name: acts}
        features = Features(
            {
                hook_name: self.features[hook_name],
                "token_ids": self.features["token_ids"],
            }
        )

        if token_ids is not None:
            token_ids = einops.rearrange(
                token_ids,
                "(bs context_size) -> bs context_size",
                bs=n_seq_in_shard,
                context_size=self.context_size,
            )
            shard_dict["token_ids"] = token_ids.to(torch.int32)
        return Dataset.from_dict(
            shard_dict,
            features=features,
        )

    @staticmethod
    def _get_sliced_context_size(
        context_size: int, seqpos_slice: tuple[int | None, ...] | None
    ) -> int:
        if seqpos_slice is not None:
            context_size = len(range(context_size)[slice(*seqpos_slice)])
        return context_size
