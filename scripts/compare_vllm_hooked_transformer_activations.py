#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set vLLM env vars before importing sae_lens.vllm_model.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ACTIVATION_CAPTURE_MODE", "1")

from sae_lens.vllm_model import HookedVLLMModel

DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_HOOKED_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYER = 21
DEFAULT_HOOKS = (
    "hook_resid_pre",
    "hook_attn_out",
    "hook_mlp_out",
    "hook_resid_post",
)
DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Large language models can be inspected through internal activations.",
    "Sparse autoencoders are useful for mechanistic interpretability research.",
    "请比较 vLLM 和 HookedTransformer 在同一层上的激活是否一致。",
    "A short prompt is enough for a parity check if the tokenization is identical.",
    "Transformers process context token by token during the prefill stage.",
    "Machine learning systems should be tested with clear numerical metrics.",
    "Consistency checks help catch shape, scaling, and hook placement bugs.",
]


@dataclass
class MetricAccumulator:
    cosine_sum: float = 0.0
    vector_count: int = 0
    close_count: int = 0
    element_count: int = 0
    sq_error_sum: float = 0.0
    sq_ref_sum: float = 0.0
    sq_pred_sum: float = 0.0
    max_abs_diff: float = 0.0

    def update(
        self,
        predicted: torch.Tensor,
        reference: torch.Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        predicted = predicted.float()
        reference = reference.float()
        pred_flat = predicted.reshape(-1, predicted.shape[-1])
        ref_flat = reference.reshape(-1, reference.shape[-1])

        cosine = F.cosine_similarity(pred_flat, ref_flat, dim=-1)
        self.cosine_sum += cosine.sum().item()
        self.vector_count += cosine.numel()

        close = torch.isclose(predicted, reference, atol=atol, rtol=rtol)
        self.close_count += close.sum().item()
        self.element_count += close.numel()

        diff = predicted - reference
        self.sq_error_sum += diff.square().sum().item()
        self.sq_ref_sum += reference.square().sum().item()
        self.sq_pred_sum += predicted.square().sum().item()
        self.max_abs_diff = max(self.max_abs_diff, diff.abs().max().item())

    def summary(self) -> dict[str, float]:
        mean_cosine = self.cosine_sum / max(self.vector_count, 1)
        rel_l2_error = math.sqrt(self.sq_error_sum / max(self.sq_ref_sum, 1e-12))
        ref_rms = math.sqrt(self.sq_ref_sum / max(self.element_count, 1))
        pred_rms = math.sqrt(self.sq_pred_sum / max(self.element_count, 1))
        rel_rms_error = abs(pred_rms - ref_rms) / max(ref_rms, 1e-12)
        return {
            "consistency_pct": mean_cosine * 100.0,
            "mean_cosine_pct": mean_cosine * 100.0,
            "close_element_pct": self.close_count / max(self.element_count, 1) * 100.0,
            "relative_l2_error_pct": rel_l2_error * 100.0,
            "relative_rms_error_pct": rel_rms_error * 100.0,
            "max_abs_diff": self.max_abs_diff,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare layer activations from HookedVLLMModel against "
            "TransformerLens HookedTransformer."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hooked-model-name", default=DEFAULT_HOOKED_MODEL_NAME)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--text-key", default=None)
    parser.add_argument("--token-key", default=None)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--hooked-n-devices", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def resolve_dataset_path(dataset_path: str | None) -> str | None:
    if dataset_path is not None:
        return dataset_path

    for candidate in ("/datasets", "/data/datasets"):
        if Path(candidate).exists():
            return candidate
    return None


def load_texts(
    dataset_path: str | None,
    *,
    text_key: str | None,
    limit: int,
) -> list[str]:
    dataset_path = resolve_dataset_path(dataset_path)
    if dataset_path is None:
        return DEFAULT_TEXTS[:limit]

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    texts: list[str] = []
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        texts.extend(_load_texts_from_file(file, text_key=text_key, limit=limit - len(texts)))
        if len(texts) >= limit:
            break

    texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError(f"No usable texts found under {path}")
    return texts[:limit]


def is_hf_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "state.json").exists()


def load_token_batches_from_hf_dataset(
    dataset_path: str,
    *,
    token_key: str | None,
    limit: int,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
) -> list[torch.Tensor]:
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    column_names = list(dataset.column_names)
    effective_token_key = token_key
    if effective_token_key is None:
        for candidate in ("input_ids", "tokens", "token_ids"):
            if candidate in column_names:
                effective_token_key = candidate
                break

    if effective_token_key is None:
        raise ValueError(
            f"Could not find a token column in {dataset_path}. Available columns: {column_names}"
        )

    batches: list[torch.Tensor] = []
    current_batch: list[torch.Tensor] = []
    num_loaded = 0
    for row in dataset:
        token_ids = row[effective_token_key]
        if isinstance(token_ids, torch.Tensor):
            tokens = token_ids.to(dtype=torch.long).flatten().cpu()
        else:
            tokens = torch.tensor(token_ids, dtype=torch.long).flatten().cpu()

        if tokens.numel() == 0:
            continue

        current_batch.append(tokens[:max_length])
        num_loaded += 1

        if len(current_batch) == batch_size:
            batches.append(_pad_token_batch(current_batch, pad_token_id))
            current_batch = []

        if num_loaded >= limit:
            break

    if current_batch:
        batches.append(_pad_token_batch(current_batch, pad_token_id))

    if not batches:
        raise ValueError(f"No usable token sequences found under {dataset_path}")
    return batches


def _load_texts_from_file(
    path: Path,
    *,
    text_key: str | None,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    suffixes = {suffix.lower() for suffix in path.suffixes}
    if ".jsonl" in suffixes:
        texts: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(texts) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    texts.append(line)
                    continue
                text = _extract_text(record, text_key=text_key)
                if text:
                    texts.append(text)
        return texts

    if ".json" in suffixes:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        records = _normalize_json_records(data)
        texts = []
        for record in records:
            if len(texts) >= limit:
                break
            text = _extract_text(record, text_key=text_key)
            if text:
                texts.append(text)
        return texts

    if path.suffix.lower() in {".txt", ".md"}:
        texts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(texts) >= limit:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    return []


def _normalize_json_records(data: Any) -> Iterable[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return [data]


def _extract_text(record: Any, *, text_key: str | None) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None

    if text_key is not None:
        value = record.get(text_key)
        return value if isinstance(value, str) and value.strip() else None

    for key in ("text", "content", "prompt", "instruction", "input", "question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    messages = record.get("messages") or record.get("conversation")
    if isinstance(messages, list):
        chunks = []
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(content.strip())
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())
        if chunks:
            return "\n".join(chunks)

    return None


def _pad_token_batch(token_seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_len = max(int(seq.numel()) for seq in token_seqs)
    batch = torch.full((len(token_seqs), max_len), pad_token_id, dtype=torch.long)
    for idx, seq in enumerate(token_seqs):
        batch[idx, : seq.numel()] = seq
    return batch


def build_token_batches(
    texts: list[str],
    tokenizer: Any,
    *,
    batch_size: int,
    max_length: int,
) -> list[torch.Tensor]:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    batches: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        batches.append(tokens.cpu())
    return batches


def capture_vllm_activations(
    token_batches: list[torch.Tensor],
    cache_dir: Path,
    *,
    model_path: str,
    tokenizer: Any,
    hook_names: list[str],
    tensor_parallel_size: int,
    max_length: int,
) -> None:
    # HookedVLLMModel internally calls generate(..., max_tokens=1), so vLLM
    # needs room for the prompt plus one decode token.
    effective_max_model_len = max(max_length, max(int(batch.shape[1]) for batch in token_batches)) + 1
    model = HookedVLLMModel(
        model_path,
        tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=effective_max_model_len,
    )
    try:
        for batch_idx, batch_tokens in enumerate(token_batches):
            _, activations = model.run_with_cache(batch_tokens, names_filter=hook_names)
            saved = {name: tensor.float().cpu() for name, tensor in activations.items()}
            torch.save(saved, cache_dir / f"batch_{batch_idx:04d}.pt")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_with_hooked_transformer(
    token_batches: list[torch.Tensor],
    cache_dir: Path,
    *,
    local_model_path: str,
    hooked_model_name: str,
    tokenizer: Any,
    model_path: str,
    hook_names: list[str],
    dtype: torch.dtype,
    device: str,
    n_devices: int,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, float]]:
    hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=dtype,
        local_files_only=True,
    )
    from_pretrained_kwargs: dict[str, Any] = {"local_files_only": True}
    if n_devices > 1:
        from_pretrained_kwargs["n_devices"] = n_devices

    model = HookedTransformer.from_pretrained_no_processing(
        hooked_model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        **from_pretrained_kwargs,
    )
    model.eval()

    metrics = {hook_name: MetricAccumulator() for hook_name in hook_names}
    try:
        for batch_idx, batch_tokens in enumerate(token_batches):
            reference_tokens = batch_tokens.to(device)
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    reference_tokens,
                    names_filter=hook_names,
                    return_cache_object=False,
                )
            predicted = torch.load(cache_dir / f"batch_{batch_idx:04d}.pt", map_location="cpu")
            for hook_name in hook_names:
                metrics[hook_name].update(
                    predicted[hook_name].cpu(),
                    cache[hook_name].detach().float().cpu(),
                    atol=atol,
                    rtol=rtol,
                )
    finally:
        del model
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {hook_name: metric.summary() for hook_name, metric in metrics.items()}


def torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def print_report(
    *,
    model_path: str,
    dataset_path: str | None,
    num_samples: int,
    token_batches: list[torch.Tensor],
    results: dict[str, dict[str, float]],
) -> None:
    seq_lengths = [int(batch.shape[1]) for batch in token_batches]
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path or 'built-in sample texts'}")
    print(f"Samples: {num_samples}")
    print(f"Batches: {len(token_batches)}")
    print(f"Seq lengths: {seq_lengths}")
    print()
    print("activation\tconsistency_pct\tclose_element_pct\trelative_l2_error_pct\trelative_rms_error_pct\tmax_abs_diff")
    for hook_name, summary in results.items():
        short_name = hook_name.split(".")[-1]
        print(
            f"{short_name}\t"
            f"{summary['consistency_pct']:.4f}\t"
            f"{summary['close_element_pct']:.4f}\t"
            f"{summary['relative_l2_error_pct']:.4f}\t"
            f"{summary['relative_rms_error_pct']:.4f}\t"
            f"{summary['max_abs_diff']:.6f}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    dataset_path = resolve_dataset_path(args.dataset_path)
    if dataset_path is not None and is_hf_dataset_dir(Path(dataset_path)):
        token_batches = load_token_batches_from_hf_dataset(
            dataset_path,
            token_key=args.token_key,
            limit=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pad_token_id=pad_token_id,
        )
        num_samples = sum(int(batch.shape[0]) for batch in token_batches)
    else:
        texts = load_texts(
            dataset_path,
            text_key=args.text_key,
            limit=args.num_samples,
        )
        token_batches = build_token_batches(
            texts,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        num_samples = len(texts)
    hook_names = [f"blocks.{args.layer}.{hook_name}" for hook_name in DEFAULT_HOOKS]

    with tempfile.TemporaryDirectory(prefix="vllm_hook_compare_") as temp_dir:
        cache_dir = Path(temp_dir)
        capture_vllm_activations(
            token_batches,
            cache_dir,
            model_path=args.model_path,
            tokenizer=tokenizer,
            hook_names=hook_names,
            tensor_parallel_size=args.tensor_parallel_size,
            max_length=args.max_length,
        )
        results = compare_with_hooked_transformer(
            token_batches,
            cache_dir,
            local_model_path=args.model_path,
            hooked_model_name=args.hooked_model_name,
            tokenizer=tokenizer,
            model_path=args.model_path,
            hook_names=hook_names,
            dtype=torch_dtype(args.dtype),
            device=args.device,
            n_devices=args.hooked_n_devices,
            atol=args.atol,
            rtol=args.rtol,
        )

    print_report(
        model_path=args.model_path,
        dataset_path=dataset_path,
        num_samples=num_samples,
        token_batches=token_batches,
        results=results,
    )

    if args.save_json:
        output = {
            "model_path": args.model_path,
            "dataset_path": dataset_path,
            "layer": args.layer,
            "hooks": hook_names,
            "num_samples": num_samples,
            "batch_sizes": [int(batch.shape[0]) for batch in token_batches],
            "seq_lengths": [int(batch.shape[1]) for batch in token_batches],
            "results": results,
        }
        Path(args.save_json).write_text(
            json.dumps(output, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
