from __future__ import annotations

import json
import os
import warnings
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from requests import HTTPError
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from sae_lens import logger
from sae_lens.config import (
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.constants import ACTIVATIONS_STORE_STATE_FILENAME
from sae_lens.distributed import (
    get_sae_tp_group,
    get_sae_tp_size,
    get_vllm_dp_p2p_group,
    get_vllm_dp_rank,
    get_vllm_dp_size,
    get_vllm_root_rank,
    get_vllm_tp_group,
    get_vllm_tp_size,
    get_worker_cpu_group,
    get_worker_group,
    is_vllm_active,
    is_vllm_dp_root,
)
from sae_lens.pretokenize_runner import get_special_token_from_cfg
from sae_lens.saes.sae import SAE, T_SAE_CONFIG, T_TRAINING_SAE_CONFIG
from sae_lens.tokenization_and_batching import concat_and_batch_sequences
from sae_lens.training.mixing_buffer import mixing_buffer
from sae_lens.util import (
    extract_stop_at_layer_from_tlens_hook_name,
    get_special_token_ids,
    str_to_dtype,
)


# TODO: Make an activation store config class to be consistent with the rest of the code.
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    cached_activation_dataset: Dataset | None = None
    tokens_column: Literal["tokens", "input_ids", "text", "problem"]
    hook_name: str
    hook_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    exclude_special_tokens: torch.Tensor | None = None
    device: torch.device

    @classmethod
    def from_cache_activations(
        cls,
        model: HookedRootModule,
        cfg: CacheActivationsRunnerConfig,
    ) -> ActivationsStore:
        """
        Public api to create an ActivationsStore from a cached activations dataset.
        """
        return cls(
            cached_activations_path=cfg.new_cached_activations_path,
            dtype=cfg.dtype,
            hook_name=cfg.hook_name,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.model_batch_size,  # get_buffer
            train_batch_size_tokens=cfg.model_batch_size,  # dataloader
            seqpos_slice=(None,),
            device=torch.device(cfg.device),  # since we're sending these to SAE
            # NOOP
            prepend_bos=False,
            hook_head_index=None,
            dataset=cfg.dataset_path,
            streaming=False,
            model=model,
            normalize_activations="none",
            model_kwargs=None,
            autocast_lm=False,
            dataset_trust_remote_code=None,
            exclude_special_tokens=None,
        )

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
        | CacheActivationsRunnerConfig,
        override_dataset: HfDataset | None = None,
        dataset_shard_index: int = 0,
        dataset_shard_count: int = 1,
    ) -> ActivationsStore:
        if isinstance(cfg, CacheActivationsRunnerConfig):
            return cls.from_cache_activations(model, cfg)

        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )

        device = torch.device(cfg.act_store_device)
        exclude_special_tokens = cfg.exclude_special_tokens
        if exclude_special_tokens is False:
            exclude_special_tokens = None
        if exclude_special_tokens is True:
            exclude_special_tokens = get_special_token_ids(model.tokenizer)  # type: ignore
        if exclude_special_tokens is not None:
            exclude_special_tokens = torch.tensor(
                exclude_special_tokens, dtype=torch.long, device=device
            )
        return cls(
            model=model,
            dataset=override_dataset or cfg.dataset_path,
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in
            if isinstance(cfg, CacheActivationsRunnerConfig)
            else cfg.sae.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.sae.normalize_activations,
            device=device,
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice,
            exclude_special_tokens=exclude_special_tokens,
            disable_concat_sequences=cfg.disable_concat_sequences,
            sequence_separator_token=cfg.sequence_separator_token,
            activations_mixing_fraction=cfg.activations_mixing_fraction,
            dataset_shard_index=dataset_shard_index,
            dataset_shard_count=dataset_shard_count,
        )

    @classmethod
    def from_sae(
        cls,
        model: HookedRootModule,
        sae: SAE[T_SAE_CONFIG],
        dataset: HfDataset | str,
        dataset_trust_remote_code: bool = False,
        context_size: int | None = None,
        streaming: bool = True,
        store_batch_size_prompts: int = 8,
        n_batches_in_buffer: int = 8,
        train_batch_size_tokens: int = 4096,
        total_tokens: int = 10**9,
        device: str = "cpu",
        disable_concat_sequences: bool = False,
        sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos",
    ) -> ActivationsStore:
        if context_size is None:
            context_size = sae.cfg.metadata.context_size
        if sae.cfg.metadata.hook_name is None:
            raise ValueError("hook_name is required")
        if context_size is None:
            raise ValueError("context_size is required")
        if sae.cfg.metadata.prepend_bos is None:
            raise ValueError("prepend_bos is required")
        return cls(
            model=model,
            dataset=dataset,
            d_in=sae.cfg.d_in,
            hook_name=sae.cfg.metadata.hook_name,
            hook_head_index=sae.cfg.metadata.hook_head_index,
            context_size=context_size,
            prepend_bos=sae.cfg.metadata.prepend_bos,
            streaming=streaming,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            n_batches_in_buffer=n_batches_in_buffer,
            total_training_tokens=total_tokens,
            normalize_activations=sae.cfg.normalize_activations,
            dataset_trust_remote_code=dataset_trust_remote_code,
            dtype=sae.cfg.dtype,
            device=torch.device(device),
            seqpos_slice=sae.cfg.metadata.seqpos_slice or (None,),
            disable_concat_sequences=disable_concat_sequences,
            sequence_separator_token=sequence_separator_token,
        )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        streaming: bool,
        hook_name: str,
        hook_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        dataset_trust_remote_code: bool | None = None,
        seqpos_slice: tuple[int | None, ...] = (None,),
        exclude_special_tokens: torch.Tensor | None = None,
        disable_concat_sequences: bool = False,
        sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos",
        activations_mixing_fraction: float = 0.5,
        dataset_shard_index: int = 0,
        dataset_shard_count: int = 1,
    ):
        self.model = model
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        if isinstance(dataset, str):
            dataset_path = Path(dataset)
            if dataset_path.exists() and dataset_path.is_dir():
                self.dataset = datasets.load_from_disk(str(dataset_path))
            else:
                self.dataset = load_dataset(
                    dataset,
                    split="train",
                    streaming=streaming,  # type: ignore
                    trust_remote_code=dataset_trust_remote_code,  # type: ignore
                )
        else:
            self.dataset = dataset

        if isinstance(dataset, (Dataset, DatasetDict)):
            self.dataset = cast(Dataset | DatasetDict, self.dataset)
            n_samples = len(self.dataset)

            if n_samples < total_training_tokens:
                warnings.warn(
                    f"The training dataset contains fewer samples ({n_samples}) than the number of samples required by your training configuration ({total_training_tokens}). This will result in multiple training epochs and some samples being used more than once."
                )

        self.hook_name = hook_name
        self.hook_head_index = hook_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = str_to_dtype(dtype)
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm
        self.seqpos_slice = seqpos_slice
        self.training_context_size = len(range(context_size)[slice(*seqpos_slice)])
        self.exclude_special_tokens = exclude_special_tokens
        self.disable_concat_sequences = disable_concat_sequences
        self.sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = (
            sequence_separator_token
        )
        self.activations_mixing_fraction = activations_mixing_fraction
        self._dataset_shard_index = dataset_shard_index
        self._dataset_shard_count = dataset_shard_count

        self.n_dataset_processed = 0

        # Check if dataset is tokenized
        dataset_sample = next(iter(self.dataset))

        # check if it's tokenized
        if "tokens" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "text"
        elif "problem" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "problem"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', 'text', or 'problem' column."
            )
        if self.is_dataset_tokenized:
            ds_context_size = len(dataset_sample[self.tokens_column])  # type: ignore
            if ds_context_size < self.context_size:
                raise ValueError(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}.
                    The context_size {ds_context_size} is expected to be larger than or equal to the provided context size {self.context_size}."""
                )
            if self.context_size != ds_context_size:
                warnings.warn(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}. Some data will be discarded in this case.""",
                    RuntimeWarning,
                )
            # TODO: investigate if this can work for iterable datasets, or if this is even worthwhile as a perf improvement
            if hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])  # type: ignore

            if (
                isinstance(dataset, str)
                and hasattr(model, "tokenizer")
                and model.tokenizer is not None
            ):
                validate_pretokenized_dataset_tokenizer(
                    dataset_path=dataset,
                    model_tokenizer=model.tokenizer,  # type: ignore
                )
        else:
            warnings.warn(
                "Dataset is not tokenized. Pre-tokenizing will improve performance and allows for more control over special tokens. See https://decoderesearch.github.io/SAELens/training_saes/#pretokenizing-datasets for more info."
            )

        self.iterable_sequences = self._iterate_tokenized_sequences()

        self.cached_activation_dataset = self.load_cached_activation_dataset()

        # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=False,  # we move to device below
                    prepend_bos=False,
                )  # type: ignore
                .squeeze(0)
                .to(self.device)
            )
            if len(tokens.shape) != 1:
                raise ValueError(f"tokens.shape should be 1D but was {tokens.shape}")
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """Generator which iterates over full sequence of context_size tokens.

        When dataset_shard_count > 1, yields every shard_count-th sequence
        starting at shard_index (strided sharding for vLLM DP).
        """
        base_iter = self._iterate_tokenized_sequences_unsharded()
        shard_idx = self._dataset_shard_index
        shard_count = self._dataset_shard_count

        if shard_count <= 1:
            yield from base_iter
            return

        # Strided sharding: skip to our starting offset, then stride.
        for i, seq in enumerate(base_iter):
            if i % shard_count == shard_idx:
                yield seq

    def _iterate_tokenized_sequences_unsharded(
        self,
    ) -> Generator[torch.Tensor, None, None]:
        """Base iterator over tokenized sequences (no sharding)."""
        # If the datset is pretokenized, we will slice the dataset to the length of the context window if needed. Otherwise, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield (
                    row[: self.context_size]  # If self.context_size = None, this line simply returns the whole row
                    .detach()
                    .clone()
                    .to(dtype=torch.long, device=self.device)
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id

            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=get_special_token_from_cfg(
                    self.sequence_separator_token, tokenizer
                )
                if tokenizer is not None
                else None,
                disable_concat_sequences=self.disable_concat_sequences,
            )

    def load_cached_activation_dataset(self) -> Dataset | None:
        """
        Load the cached activation dataset from disk.

        - If cached_activations_path is set, returns Huggingface Dataset else None
        - Checks that the loaded dataset has current has activations for hooks in config and that shapes match.
        """
        if self.cached_activations_path is None:
            return None

        assert self.cached_activations_path is not None  # keep pyright happy
        # Sanity check: does the cache directory exist?
        if not os.path.exists(self.cached_activations_path):
            raise FileNotFoundError(
                f"Cache directory {self.cached_activations_path} does not exist. "
                "Consider double-checking your dataset, model, and hook names."
            )

        # ---
        # Actual code
        activations_dataset = datasets.load_from_disk(self.cached_activations_path)
        columns = [self.hook_name]
        if "token_ids" in activations_dataset.column_names:
            columns.append("token_ids")
        activations_dataset.set_format(
            type="torch", columns=columns, device=self.device, dtype=self.dtype
        )
        self.current_row_idx = 0  # idx to load next batch from
        # ---

        assert isinstance(activations_dataset, Dataset)

        # multiple in hooks future
        if not set([self.hook_name]).issubset(activations_dataset.column_names):
            raise ValueError(
                f"loaded dataset does not include hook activations, got {activations_dataset.column_names}"
            )

        if activations_dataset.features[self.hook_name].shape != (
            self.context_size,
            self.d_in,
        ):
            raise ValueError(
                f"Given dataset of shape {activations_dataset.features[self.hook_name].shape} does not match context_size ({self.context_size}) and d_in ({self.d_in})"
            )

        return activations_dataset

    def shuffle_input_dataset(self, seed: int, buffer_size: int = 1):
        """
        This applies a shuffle to the huggingface dataset that is the input to the activations store. This
        also shuffles the shards of the dataset, which is especially useful for evaluating on different
        sections of very large streaming datasets. Buffer size is only relevant for streaming datasets.
        The default buffer_size of 1 means that only the shard will be shuffled; larger buffer sizes will
        additionally shuffle individual elements within the shard.
        """
        if isinstance(self.dataset, IterableDataset):
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.iterable_dataset = iter(self.dataset)

    def reset_input_dataset(self):
        """
        Resets the input dataset iterator to the beginning.
        """
        self.iterable_dataset = iter(self.dataset)

    def get_batch_tokens(
        self, batch_size: int | None = None, raise_at_epoch_end: bool = False
    ):
        """
        Streams a batch of tokens from a dataset.

        If raise_at_epoch_end is true we will reset the dataset at the end of each epoch and raise a StopIteration. Otherwise we will reset silently.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        worker_group = get_worker_group() if dist.is_initialized() else None
        worker_cpu_group = get_worker_cpu_group() if dist.is_initialized() else None
        if worker_group is not None and worker_cpu_group is not None:
            src_rank = get_vllm_root_rank()
            _debug_prefix_tp(f"get_batch_tokens start batch_size={batch_size} src={src_rank}")
            epoch_end = torch.zeros(1, dtype=torch.int32)
            if dist.get_rank() == src_rank:
                try:
                    batch_tokens = self._get_batch_tokens_local(
                        batch_size=batch_size,
                        raise_at_epoch_end=raise_at_epoch_end,
                    ).cpu()
                except StopIteration:
                    epoch_end.fill_(1)
                    batch_tokens = torch.empty(
                        (batch_size, self.context_size),
                        dtype=torch.long,
                    )
            else:
                batch_tokens = torch.empty(
                    (batch_size, self.context_size),
                    dtype=torch.long,
                )
            dist.broadcast(epoch_end, src=src_rank, group=worker_cpu_group)
            if epoch_end.item():
                _debug_prefix_tp("get_batch_tokens epoch_end")
                raise StopIteration
            dist.broadcast(batch_tokens, src=src_rank, group=worker_cpu_group)
            _debug_prefix_tp("get_batch_tokens done")
            return batch_tokens.to(self._get_runtime_device())

        return self._get_batch_tokens_local(
            batch_size=batch_size,
            raise_at_epoch_end=raise_at_epoch_end,
        ).to(_get_model_device(self.model))

    def _get_batch_tokens_local(
        self, batch_size: int, raise_at_epoch_end: bool = False
    ) -> torch.Tensor:
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                self.iterable_sequences = self._iterate_tokenized_sequences()
                if raise_at_epoch_end:
                    raise StopIteration(
                        f"Ran out of tokens in dataset after {self.n_dataset_processed} samples, beginning the next epoch."
                    )
                sequences.append(next(self.iterable_sequences))

        return torch.stack(sequences, dim=0)

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """
        worker_group = get_worker_group() if dist.is_initialized() else None
        worker_cpu_group = get_worker_cpu_group() if dist.is_initialized() else None
        if worker_group is None or worker_cpu_group is None:
            return self._get_activations_local(batch_tokens)

        src_rank = get_vllm_root_rank()
        n_context = self.training_context_size
        if is_vllm_active():
            _debug_prefix_tp("get_activations local run_with_cache start")
            # Normalize to the activation-store dtype before the CPU broadcast so
            # split-role SAE-only ranks allocate a matching receive buffer.
            stacked_activations = self._get_activations_local(batch_tokens).to(
                device="cpu", dtype=self.dtype
            )
            _debug_prefix_tp("get_activations local run_with_cache done")
        else:
            stacked_activations = torch.empty(
                (batch_tokens.shape[0], n_context, self.d_in),
                dtype=self.dtype,
            )
            _debug_prefix_tp("get_activations waiting for broadcast")
        dist.broadcast(stacked_activations, src=src_rank, group=worker_cpu_group)
        _debug_prefix_tp("get_activations broadcast done")
        return stacked_activations.to(self.device)

    def _get_activations_local(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        model_device = _get_model_device(self.model)
        with torch.autocast(
            device_type=model_device.type,
            dtype=torch.bfloat16,
            enabled=self.autocast_lm and model_device.type != "cpu",
        ):
            layerwise_activations_cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_name],
                stop_at_layer=extract_stop_at_layer_from_tlens_hook_name(
                    self.hook_name
                ),
                prepend_bos=False,
                **self.model_kwargs,
            )[1]

        layerwise_activations = layerwise_activations_cache[self.hook_name][
            :, slice(*self.seqpos_slice)
        ]

        if (
            dist.is_initialized()
            and is_vllm_active()
            and layerwise_activations.shape[-1] != self.d_in
        ):
            vllm_tp_group = get_vllm_tp_group()
            if vllm_tp_group is None:
                raise RuntimeError(
                    "vLLM TP group is not initialized but activation gathering is required."
                )
            world_size = dist.get_world_size(vllm_tp_group)
            if layerwise_activations.shape[-1] * world_size != self.d_in:
                raise RuntimeError(
                    "Activation width does not match SAE input width after vLLM TP gather. "
                    f"Got local width={layerwise_activations.shape[-1]}, "
                    f"tp_world_size={world_size}, expected d_in={self.d_in}."
                )
            shards = [torch.empty_like(layerwise_activations) for _ in range(world_size)]
            dist.all_gather(
                shards,
                layerwise_activations.contiguous(),
                group=vllm_tp_group,
            )
            layerwise_activations = torch.cat(shards, dim=-1)

        n_batches, n_context = layerwise_activations.shape[:2]
        stacked_activations = torch.zeros(
            (n_batches, n_context, self.d_in),
            dtype=layerwise_activations.dtype,
            device=layerwise_activations.device,
        )

        if self.hook_head_index is not None:
            stacked_activations[:, :] = layerwise_activations[
                :, :, self.hook_head_index
            ]
        elif layerwise_activations.ndim > 3:  # if we have a head dimension
            stacked_activations[:, :] = layerwise_activations.reshape(
                n_batches, n_context, -1
            )
        else:
            stacked_activations[:, :] = layerwise_activations

        return stacked_activations

    def _get_runtime_device(self) -> torch.device:
        if is_vllm_active():
            return _get_model_device(self.model)
        return self.device

    def _load_raw_llm_batch_from_cached(
        self,
        raise_on_epoch_end: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Loads a batch of activations from `cached_activation_dataset`

        The dataset has columns for each hook_name,
        each containing activations of shape (context_size, d_in).

        raises StopIteration
        """
        assert self.cached_activation_dataset is not None
        context_size = self.context_size
        batch_size = self.store_batch_size_prompts
        d_in = self.d_in

        # In future, could be a list of multiple hook names
        if self.hook_name not in self.cached_activation_dataset.column_names:
            raise ValueError(
                f"Missing columns in dataset. Expected {self.hook_name}, "
                f"got {self.cached_activation_dataset.column_names}."
            )

        if self.current_row_idx > len(self.cached_activation_dataset) - batch_size:
            self.current_row_idx = 0
            if raise_on_epoch_end:
                raise StopIteration

        ds_slice = self.cached_activation_dataset[
            self.current_row_idx : self.current_row_idx + batch_size
        ]
        # Load activations for each hook.
        # Usually faster to first slice dataset then pick column
        acts_buffer = ds_slice[self.hook_name]
        if acts_buffer.shape != (batch_size, context_size, d_in):
            raise ValueError(
                f"acts_buffer has shape {acts_buffer.shape}, "
                f"but expected ({batch_size}, {context_size}, {d_in})."
            )

        self.current_row_idx += batch_size
        acts_buffer = acts_buffer.reshape(batch_size * context_size, d_in)

        if "token_ids" not in self.cached_activation_dataset.column_names:
            return acts_buffer, None

        token_ids_buffer = ds_slice["token_ids"]
        if token_ids_buffer.shape != (batch_size, context_size):
            raise ValueError(
                f"token_ids_buffer has shape {token_ids_buffer.shape}, "
                f"but expected ({batch_size}, {context_size})."
            )
        token_ids_buffer = token_ids_buffer.reshape(batch_size * context_size)
        return acts_buffer, token_ids_buffer

    @torch.no_grad()
    def get_raw_llm_batch(
        self,
        raise_on_epoch_end: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Loads the next batch of activations from the LLM and returns it.

        If raise_on_epoch_end is True, when the dataset is exhausted it will
        automatically refill the dataset and then raise a StopIteration so that
        the caller has a chance to react.

        Returns:
            Tuple of (activations, token_ids) where activations has shape
            (batch_size * context_size, d_in) and token_ids has shape
            (batch_size * context_size,).
        """
        d_in = self.d_in

        if self.cached_activation_dataset is not None:
            return self._load_raw_llm_batch_from_cached(raise_on_epoch_end)

        # move batch toks to gpu for model
        batch_tokens = self.get_batch_tokens(raise_at_epoch_end=raise_on_epoch_end).to(
            _get_model_device(self.model)
        )
        activations = self.get_activations(batch_tokens).to(
            device=self.device, dtype=self.dtype
        )

        # handle seqpos_slice, this is done for activations in get_activations
        batch_tokens = batch_tokens[:, slice(*self.seqpos_slice)]

        # reshape from (batch, context, d_in) to (batch * context, d_in)
        activations = activations.reshape(-1, d_in)
        token_ids = batch_tokens.reshape(-1)

        return activations, token_ids

    def get_filtered_llm_batch(
        self,
        raise_on_epoch_end: bool = False,
    ) -> torch.Tensor:
        """
        Get a batch of LLM activations with special tokens filtered out.
        """
        return _filter_buffer_acts(
            self.get_raw_llm_batch(raise_on_epoch_end=raise_on_epoch_end),
            self.exclude_special_tokens,
        )

    def _iterate_filtered_activations(self) -> Generator[torch.Tensor, None, None]:
        """Iterate over filtered LLM activation batches.

        When vllm_dp_size > 1 and this rank is sae_active, gathers raw batches
        from all vLLM DP groups via P2P send/recv and yields them one producer
        at a time.

        This keeps rank 0 from materializing a single large concatenated batch
        before handing data to the mixing buffer. The mixing buffer still sees
        the same total stream of activations, just in smaller chunks.
        """
        vllm_dp_size = get_vllm_dp_size() if dist.is_initialized() else 1
        if vllm_dp_size <= 1:
            yield from self._iterate_filtered_activations_single()
            return

        # vLLM DP mode: gather raw batches from all DP groups.
        from sae_lens.distributed import is_sae_active

        rank = dist.get_rank()
        vllm_dp_rank = get_vllm_dp_rank()
        vllm_tp_size = get_vllm_tp_size()
        sae_tp_size = get_sae_tp_size()
        sae_tp_group = get_sae_tp_group()
        p2p_group = get_vllm_dp_p2p_group()
        if p2p_group is None:
            raise RuntimeError(
                "vLLM DP P2P group is not initialized; refusing to fall back to the default process group."
            )

        while True:
            try:
                raw_acts, raw_tokens = self.get_raw_llm_batch(
                    raise_on_epoch_end=True
                )
            except StopIteration:
                warnings.warn(
                    "All samples in the training dataset have been exhausted, beginning new epoch."
                )
                raw_acts, raw_tokens = self.get_raw_llm_batch()
            _debug_prefix_tp(
                f"vllm_dp raw batch acts={tuple(raw_acts.shape)} "
                f"tokens={None if raw_tokens is None else tuple(raw_tokens.shape)}"
            )
            # Use CPU tensors for the dedicated Gloo P2P path.
            runtime_device = self.device
            tp_runtime_device = _get_model_device(self.model)
            raw_acts = raw_acts.to("cpu").contiguous()
            if raw_tokens is not None:
                raw_tokens = raw_tokens.to("cpu").contiguous()

            # Step 1: Non-zero DP roots send their raw batch to SAE root (rank 0)
            # via P2P. This must complete BEFORE any SAE TP broadcasts, because
            # rank 0 needs to recv before it can broadcast, and SAE TP follower
            # ranks that are also vLLM DP helpers must finish sending before they
            # can participate in SAE TP broadcasts.
            if is_vllm_dp_root() and vllm_dp_rank > 0:
                _debug_prefix_tp("vllm_dp helper send start")
                dist.send(raw_acts, dst=0, group=p2p_group)
                if raw_tokens is not None:
                    dist.send(raw_tokens, dst=0, group=p2p_group)
                _debug_prefix_tp("vllm_dp helper send done")

            # Step 2: Rank 0 collects all helper batches via P2P recv, then
            # broadcasts each batch (own + helpers) to the SAE TP group.
            # SAE TP follower ranks participate in the broadcasts.
            #
            # The ordering is: rank 0 recvs ALL helper batches first, then
            # broadcasts them in sequence. This avoids deadlock when a rank is
            # both a vLLM DP helper (needs to send before rank 0 can proceed)
            # and an SAE TP follower (needs to participate in broadcasts).
            if rank == 0:
                # Collect helper batches first so P2P sends can complete.
                helper_batches: list[tuple[torch.Tensor, torch.Tensor | None]] = []
                for k in range(1, vllm_dp_size):
                    src = k * vllm_tp_size
                    recv_acts = torch.empty_like(raw_acts)
                    _debug_prefix_tp(f"vllm_dp root recv start src={src}")
                    dist.recv(recv_acts, src=src, group=p2p_group)
                    recv_tokens = None
                    if raw_tokens is not None:
                        recv_tokens = torch.empty_like(raw_tokens)
                        dist.recv(recv_tokens, src=src, group=p2p_group)
                    _debug_prefix_tp(f"vllm_dp root recv done src={src}")
                    helper_batches.append((recv_acts, recv_tokens))

                # Now broadcast own batch + helper batches to SAE TP group.
                all_batches = [(raw_acts, raw_tokens)] + helper_batches
                for batch_acts, batch_tokens in all_batches:
                    if sae_tp_size > 1 and sae_tp_group is not None:
                        batch_acts = batch_acts.to(tp_runtime_device)
                        batch_tokens = (
                            batch_tokens.to(tp_runtime_device)
                            if batch_tokens is not None
                            else None
                        )
                        _debug_prefix_tp("vllm_dp root tp broadcast start")
                        dist.broadcast(batch_acts, src=0, group=sae_tp_group)
                        if batch_tokens is not None:
                            dist.broadcast(batch_tokens, src=0, group=sae_tp_group)
                        _debug_prefix_tp("vllm_dp root tp broadcast done")
                        yield _filter_buffer_acts(
                            (batch_acts, batch_tokens),
                            self.exclude_special_tokens,
                        )
                    else:
                        yield _filter_buffer_acts(
                            (
                                batch_acts.to(runtime_device),
                                batch_tokens.to(runtime_device)
                                if batch_tokens is not None
                                else None,
                            ),
                            self.exclude_special_tokens,
                        )
            elif is_sae_active() and sae_tp_size > 1 and sae_tp_group is not None:
                # SAE TP follower: participate in all vllm_dp_size broadcasts
                # (own batch from rank 0 + each helper batch).
                combined_shape_acts = list(raw_acts.shape)
                combined_shape_tokens = list(raw_tokens.shape) if raw_tokens is not None else None
                for _ in range(vllm_dp_size):
                    helper_acts = torch.empty(
                        combined_shape_acts,
                        dtype=raw_acts.dtype,
                        device=tp_runtime_device,
                    )
                    _debug_prefix_tp("vllm_dp tp follower broadcast wait")
                    dist.broadcast(helper_acts, src=0, group=sae_tp_group)
                    helper_tokens = None
                    if combined_shape_tokens is not None:
                        helper_tokens = torch.empty(
                            combined_shape_tokens,
                            dtype=raw_tokens.dtype,  # type: ignore[union-attr]
                            device=tp_runtime_device,
                        )
                        dist.broadcast(helper_tokens, src=0, group=sae_tp_group)
                    _debug_prefix_tp("vllm_dp tp follower broadcast done")
                    yield _filter_buffer_acts(
                        (helper_acts, helper_tokens),
                        self.exclude_special_tokens,
                    )

    def _iterate_filtered_activations_single(
        self,
    ) -> Generator[torch.Tensor, None, None]:
        """Original single-producer filtered activations iterator."""
        while True:
            try:
                yield self.get_filtered_llm_batch(raise_on_epoch_end=True)
            except StopIteration:
                warnings.warn(
                    "All samples in the training dataset have been exhausted, beginning new epoch."
                )
                try:
                    yield self.get_filtered_llm_batch()
                except StopIteration:
                    raise ValueError(
                        "Unable to fill buffer after starting new epoch. Dataset may be too small."
                    )

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return an auto-refilling stream of filtered and mixed activations.
        """
        return mixing_buffer(
            buffer_size=self.n_batches_in_buffer * self.training_context_size,
            batch_size=self.train_batch_size_tokens,
            activations_loader=self._iterate_filtered_activations(),
            mix_fraction=self.activations_mixing_fraction,
        )

    def next_batch(self) -> torch.Tensor:
        """Get next batch, updating buffer if needed."""
        return self.__next__()

    # ActivationsStore should be an iterator
    def __next__(self) -> torch.Tensor:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return next(self._dataloader)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"n_dataset_processed": torch.tensor(self.n_dataset_processed)}

    def save(self, file_path: str):
        """save the state dict to a file in safetensors format"""
        save_file(self.state_dict(), file_path)

    def save_to_checkpoint(self, checkpoint_path: str | Path):
        """Save the state dict to a checkpoint path"""
        self.save(str(Path(checkpoint_path) / ACTIVATIONS_STORE_STATE_FILENAME))

    def load_from_checkpoint(self, checkpoint_path: str | Path):
        """Load the state dict from a checkpoint path"""
        self.load(str(Path(checkpoint_path) / ACTIVATIONS_STORE_STATE_FILENAME))

    def load(self, file_path: str):
        """Load the state dict from a file in safetensors format"""

        state_dict = load_file(file_path)

        if "n_dataset_processed" in state_dict:
            target_n_dataset_processed = state_dict["n_dataset_processed"].item()

            # Only fast-forward if needed

            if target_n_dataset_processed > self.n_dataset_processed:
                logger.info(
                    "Fast-forwarding through dataset samples to match checkpoint position"
                )
                samples_to_skip = target_n_dataset_processed - self.n_dataset_processed

                pbar = tqdm(
                    total=samples_to_skip,
                    desc="Fast-forwarding through dataset",
                    leave=False,
                )
                while target_n_dataset_processed > self.n_dataset_processed:
                    start = self.n_dataset_processed
                    try:
                        # Just consume and ignore the values to fast-forward
                        next(self.iterable_sequences)
                    except StopIteration:
                        logger.warning(
                            "Dataset exhausted during fast-forward. Resetting dataset."
                        )
                        self.iterable_sequences = self._iterate_tokenized_sequences()
                    pbar.update(self.n_dataset_processed - start)
                pbar.close()


def validate_pretokenized_dataset_tokenizer(
    dataset_path: str, model_tokenizer: PreTrainedTokenizerBase
) -> None:
    """
    Helper to validate that the tokenizer used to pretokenize the dataset matches the model tokenizer.
    """
    if Path(dataset_path).exists():
        return
    try:
        tokenization_cfg_path = hf_hub_download(
            dataset_path, "sae_lens.json", repo_type="dataset"
        )
    except HfHubHTTPError:
        return
    if tokenization_cfg_path is None:
        return
    with open(tokenization_cfg_path) as f:
        tokenization_cfg = json.load(f)
    tokenizer_name = tokenization_cfg["tokenizer_name"]
    try:
        ds_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # if we can't download the specified tokenizer to verify, just continue
    except HTTPError:
        return
    if ds_tokenizer.get_vocab() != model_tokenizer.get_vocab():
        raise ValueError(
            f"Dataset tokenizer {tokenizer_name} does not match model tokenizer {model_tokenizer}."
        )


def _get_model_device(model: HookedRootModule) -> torch.device:
    if hasattr(model, "W_E"):
        return model.W_E.device  # type: ignore
    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        return model.cfg.device  # type: ignore
    if isinstance(getattr(model, "device", None), torch.device):
        return model.device  # type: ignore
    return next(model.parameters()).device  # type: ignore


def _filter_buffer_acts(
    buffer: tuple[torch.Tensor, torch.Tensor | None],
    exclude_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """
    Filter out activations for tokens that are in exclude_tokens.
    """

    activations, tokens = buffer
    if tokens is None or exclude_tokens is None:
        return activations

    if exclude_tokens.device != tokens.device:
        exclude_tokens = exclude_tokens.to(tokens.device)
    mask = torch.isin(tokens, exclude_tokens)
    return activations[~mask]


def _debug_prefix_tp(msg: str) -> None:
    if os.environ.get("SAELENS_DEBUG_PREFIX_TP") != "1":
        return
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    line = f"[prefix-debug rank{rank}] {msg}\n"
    with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
        f.write(line)
    print(line, end="", flush=True)
