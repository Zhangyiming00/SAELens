"""Resolve activation storage independently from model and SAE precision."""

from __future__ import annotations


def resolve_activation_dtype(
    source_dtype: str,
    sae_dtype: str,
    activation_dtype: str | None = None,
    conversion: str = "auto",
    *,
    autocast: bool = False,
) -> str:
    """Explicit storage wins; otherwise place the necessary conversion.

    ``vllm`` converts at production to the SAE compute dtype, ``sae`` retains
    the source dtype until consumption, and ``auto`` uses the smaller of the
    two. Auto intentionally permits FP32 -> BF16 when BF16 compute is chosen.
    """
    supported = ("float32", "bfloat16")
    if source_dtype not in supported or sae_dtype not in supported:
        raise ValueError("source and SAE precision must be float32 or bfloat16")
    if conversion not in ("auto", "vllm", "sae"):
        raise ValueError("activation conversion must be auto, vllm or sae")
    if activation_dtype not in (None, "none", *supported):
        raise ValueError("activation dtype must be none, float32 or bfloat16")
    if activation_dtype not in (None, "none"):
        return activation_dtype
    compute_dtype = "bfloat16" if autocast else sae_dtype
    if conversion == "vllm":
        return compute_dtype
    if conversion == "sae":
        return source_dtype
    return "bfloat16" if "bfloat16" in (source_dtype, compute_dtype) else "float32"
