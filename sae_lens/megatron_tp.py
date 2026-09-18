"""Megatron dependency, collectives, and SAELens-compatible initialization.

Training parameters are owned by MegatronTopKSAE linear modules. No custom
linear or collective autograd implementation lives here.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sae_lens.profiling import cuda_nvtx_range
from sae_lens.util import str_to_dtype

if TYPE_CHECKING:
    from sae_lens.saes.topk_sae import TopKTrainingSAEConfig


@lru_cache(maxsize=1)
def require_megatron_core() -> ModuleType:
    """Load the optional TP backend or report how to install it."""
    try:
        return import_module("megatron.core.tensor_parallel")
    except ImportError as exc:
        raise ImportError(
            "SAE TopK training requires Megatron Core (including TP1). "
            "Install it with `pip install 'sae-lens[megatron]'` or "
            "`pip install 'megatron-core==0.16.1'`."
        ) from exc


def megatron_tp_allgather(
    local: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Gather latent shards; backward selects this rank's gradient shard."""
    if group.size() == 1:
        return local
    tp = require_megatron_core()
    # Megatron allocates the receive buffer on the current CUDA device.
    with torch.cuda.device(local.device):
        gathered = tp.gather_from_tensor_model_parallel_region(
            local.reshape(-1, local.shape[-1]), group=group
        )
    return gathered.reshape(*local.shape[:-1], local.shape[-1] * group.size())


@lru_cache(maxsize=None)
def _wavefront_stream(device: torch.device) -> torch.cuda.Stream:
    # All hooks on a device submit to one stream, in the same host order on
    # every TP rank. No process group or tensor is kept in this cache.
    return torch.cuda.Stream(device=device)


@dataclass
class MegatronTPPending:
    """Native TP result whose consumer-stream dependency is deferred."""

    result: torch.Tensor
    ready: torch.cuda.Event

    def wait(self) -> torch.Tensor:
        stream = torch.cuda.current_stream(self.result.device)
        stream.wait_event(self.ready)
        self.result.record_stream(stream)
        return self.result


def megatron_tp_launch(
    local: torch.Tensor, group: dist.ProcessGroup, *, gather: bool
) -> MegatronTPPending:
    """Enqueue a native mapping on the TP stream without stalling compute.

    Megatron's mapping retains its own autograd (gather backward is a split;
    reduce backward is identity). Its synchronous c10d call waits on the
    *calling CUDA stream*, so the event includes actual collective completion.
    No custom collective backward or NCCL Work handling is needed here.
    """
    if not local.is_cuda or group.size() <= 1:
        raise ValueError("Megatron TP wavefront requires CUDA and TP > 1")
    current = torch.cuda.current_stream(local.device)
    stream = _wavefront_stream(local.device)
    with torch.cuda.device(local.device), torch.cuda.stream(stream):
        stream.wait_stream(current)
        local.record_stream(stream)
        with cuda_nvtx_range("sae_tp_wavefront:all_gather" if gather else "sae_tp_wavefront:all_reduce"):
            result = (
                megatron_tp_allgather(local, group)
                if gather
                else require_megatron_core().reduce_from_tensor_model_parallel_region(
                    local, group=group
                )
            )
        ready = torch.cuda.Event()
        ready.record(stream)
    return MegatronTPPending(result, ready)


_RNG_CHUNK_NUMEL = 4 * 1024 * 1024


def _advance_cpu_rng(numel: int, dtype: torch.dtype, bound: float) -> None:
    """Advance the CPU MT19937 by ``numel`` uniform_(-bound, bound) draws.

    Uses a fixed-size scratch buffer that is reused across iterations so peak
    temporary memory is bounded by ``_RNG_CHUNK_NUMEL`` elements regardless of
    ``numel``. The native-reference initialization test checks the shard values.
    """
    if numel <= 0:
        return
    scratch = torch.empty(min(numel, _RNG_CHUNK_NUMEL), dtype=dtype, device="cpu")
    remaining = numel
    while remaining > 0:
        n = min(remaining, scratch.numel())
        scratch.narrow(0, 0, n).uniform_(-bound, bound)
        remaining -= n


def _shard_init_topk_cpu(
    cfg: "TopKTrainingSAEConfig",
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bit-equivalent CPU shard initialization for ``TopKTrainingSAE``.

    Reproduces the row-major slice of the legacy full-init path:

        full = torch.empty(d_sae, d_in, dtype, "cpu"); full.uniform_(-b, b)
        shard = full[r*S:(r+1)*S].contiguous()

    followed by ``TrainingSAE``'s per-row decoder norm (if ``decoder_init_norm``
    is set) and ``_init_weights_topk``'s ``b_enc=0``. The pre/post skip calls
    keep the CPU MT19937 state aligned with a single full-tensor ``uniform_``
    of length ``d_sae * d_in`` so external code in the same ``temporary_seed``
    scope sees the same RNG endpoint regardless of TP size or rank.

    Returns ``(W_dec_shard, W_enc_shard, b_enc_shard, b_dec_full)`` on CPU.
    """
    assert cfg.d_sae % tp_size == 0, (
        f"d_sae={cfg.d_sae} must be divisible by tp_size={tp_size}"
    )
    shard_size = cfg.d_sae // tp_size
    dtype = str_to_dtype(cfg.dtype)
    # kaiming_uniform_ defaults: a=0, mode='fan_in', nonlinearity='leaky_relu'
    # → gain = sqrt(2), bound = gain * sqrt(3 / fan_in) = sqrt(6 / d_in)
    bound = math.sqrt(6.0 / cfg.d_in)

    pre_numel = tp_rank * shard_size * cfg.d_in
    post_numel = (tp_size - tp_rank - 1) * shard_size * cfg.d_in

    _advance_cpu_rng(pre_numel, dtype, bound)
    w_dec_shard = torch.empty(shard_size, cfg.d_in, dtype=dtype, device="cpu")
    w_dec_shard.uniform_(-bound, bound)
    _advance_cpu_rng(post_numel, dtype, bound)

    if cfg.decoder_init_norm is not None:
        with torch.no_grad():
            w_dec_shard /= w_dec_shard.norm(dim=-1, keepdim=True)
            w_dec_shard *= cfg.decoder_init_norm

    w_enc_shard = w_dec_shard.T.contiguous()
    b_enc_shard = torch.zeros(shard_size, dtype=dtype, device="cpu")
    b_dec_full = torch.zeros(cfg.d_in, dtype=dtype, device="cpu")
    return w_dec_shard, w_enc_shard, b_enc_shard, b_dec_full
