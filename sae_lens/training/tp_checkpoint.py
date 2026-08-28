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


def get_current_sae_tp_cpu_group() -> dist.ProcessGroup | None:
    """Return the current rank's SAE-TP Gloo checkpoint/export group.

    ``distributed_v2`` is preferred when it is the initialized topology; otherwise
    fall back to the legacy prefix-overlap topology.
    """
    try:
        import sae_lens.distributed_v2 as v2_mod

        if getattr(v2_mod, "_initialized", False) and v2_mod.is_consumer():
            return v2_mod.get_sae_tp_cpu_group()
    except (ImportError, AttributeError):
        pass

    from sae_lens.distributed import get_sae_tp_cpu_group

    return get_sae_tp_cpu_group()


def gather_tp_state_dict_to_root_cpu(
    local_state_dict: dict[str, Any],
    base_sae: Any,
    tp_cpu_group: dist.ProcessGroup | None,
) -> dict[str, Any] | None:
    """Gather CPU TP parameter shards to TP rank 0 using Gloo.

    This is the checkpoint/export counterpart of the NCCL TP training group.
    All members of ``tp_cpu_group`` must call this function in identical key order.
    Sharded tensors are gathered only to TP rank 0; replicated tensors are simply
    retained from TP rank 0. Non-root ranks return ``None``.
    """
    if tp_cpu_group is None:
        for name, value in local_state_dict.items():
            if torch.is_tensor(value) and value.device.type != "cpu":
                raise RuntimeError(
                    f"CPU TP checkpoint gather expected CPU tensor for {name}, "
                    f"got {value.device}"
                )
        return dict(local_state_dict)

    tp_rank = dist.get_rank(tp_cpu_group)
    tp_size = dist.get_world_size(tp_cpu_group)
    shard_dims = tp_param_shard_dims(base_sae)
    result: dict[str, Any] | None = {} if tp_rank == 0 else None

    if tp_size == 1:
        for name, value in local_state_dict.items():
            if torch.is_tensor(value) and value.device.type != "cpu":
                raise RuntimeError(
                    f"CPU TP checkpoint gather expected CPU tensor for {name}, "
                    f"got {value.device}"
                )
        return dict(local_state_dict)

    root_global_rank = dist.get_global_rank(tp_cpu_group, 0)
    for name, value in local_state_dict.items():
        if not torch.is_tensor(value):
            if tp_rank == 0:
                assert result is not None
                result[name] = value
            continue
        if value.device.type != "cpu":
            raise RuntimeError(
                f"CPU TP checkpoint gather expected CPU tensor for {name}, "
                f"got {value.device}"
            )

        shard_dim = shard_dims.get(name)
        if shard_dim is None:
            if tp_rank == 0:
                assert result is not None
                result[name] = value.detach().contiguous()
            continue

        local = value.detach().contiguous()
        gather_list = (
            [torch.empty_like(local) for _ in range(tp_size)]
            if tp_rank == 0
            else None
        )
        dist.gather(
            local,
            gather_list=gather_list,
            dst=root_global_rank,
            group=tp_cpu_group,
        )
        if tp_rank == 0:
            assert result is not None and gather_list is not None
            result[name] = torch.cat(gather_list, dim=shard_dim)

    return result


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
