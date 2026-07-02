"""TP-aware safetensors slice loader for SAE checkpoints.

Each TP rank reads only its own shard of the W_enc/W_dec/b_enc tensors,
keeping the b_dec (and any other replicated tensor) full. The shard
dimensions come from the SAE's ``_tp_param_shard_dims`` (None ⇒ replicated).

Used both by ``multi_sae_trainer`` (multi-SAE checkpoint loading) and
``TrainingSAE.load_weights_from_checkpoint`` (single-SAE TP resume).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open

from sae_lens.util import str_to_dtype

_SAFETENSORS_DTYPE_MAP = {
    "F32": "float32",
    "BF16": "bfloat16",
    "F16": "float16",
    "F64": "float64",
    "I32": "int32",
    "I64": "int64",
}


def tp_param_shard_dims(base_sae: Any) -> dict[str, int | None]:
    """Return the per-parameter TP shard dimensions for ``base_sae``.

    If the SAE doesn't expose ``_tp_param_shard_dims``, returns an empty dict
    (i.e. all tensors are treated as replicated).
    """
    if hasattr(base_sae, "_tp_param_shard_dims"):
        return base_sae._tp_param_shard_dims()
    return {}


def load_tp_sharded_state_dict(
    filepath: Path | str,
    base_sae: Any,
    tp_group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    """Load only this TP rank's checkpoint tensor slices from a safetensors file.

    Returns a state_dict containing only this rank's shard. Callers must NOT
    call ``process_state_dict_for_loading`` afterwards: the tensors returned
    here are already at shard shape.
    """
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    shard_dims = tp_param_shard_dims(base_sae)

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(filepath), framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 — safetensors `safe_open` lacks __iter__
            sl = f.get_slice(k)
            shape = list(sl.get_shape())
            dtype_str = str(sl.get_dtype())
            dtype = str_to_dtype(_SAFETENSORS_DTYPE_MAP.get(dtype_str, dtype_str.lower()))
            shard_dim = shard_dims.get(k)

            if shard_dim is None:
                state_dict[k] = f.get_tensor(k)
            else:
                full_size = shape[shard_dim]
                assert full_size % tp_size == 0, (
                    f"Checkpoint tensor '{k}' size {full_size} on dim "
                    f"{shard_dim} not divisible by tp_size={tp_size}"
                )
                shard_size = full_size // tp_size
                slices: list[slice] = [slice(None)] * len(shape)
                slices[shard_dim] = slice(
                    tp_rank * shard_size, (tp_rank + 1) * shard_size
                )
                state_dict[k] = sl[tuple(slices)].to(dtype=dtype)

    return state_dict
