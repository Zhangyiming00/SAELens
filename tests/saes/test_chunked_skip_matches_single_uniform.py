"""Gate test for the TP shard-init plan.

The shard-init algorithm advances the CPU MT19937 by N draws using a
fixed-size scratch buffer (chunked uniform_) instead of allocating a single
N-element tensor. The plan only works if a chunked sequence of uniform_
calls produces the same RNG-stream values as one big uniform_ call of the
same total length.

If this test ever fails on the project's pinned PyTorch version, the
shard-init implementation must fall back to a single uniform_ call (paying
the CPU memory cost) rather than ship broken bit-equivalence.
"""

from __future__ import annotations

import math

import pytest
import torch

from sae_lens.util import temporary_seed

_RNG_CHUNK_NUMEL = 4 * 1024 * 1024


def _chunked_uniform(total_numel: int, dtype: torch.dtype, bound: float) -> torch.Tensor:
    if total_numel == 0:
        return torch.empty(0, dtype=dtype, device="cpu")
    scratch = torch.empty(min(total_numel, _RNG_CHUNK_NUMEL), dtype=dtype, device="cpu")
    out: list[torch.Tensor] = []
    remaining = total_numel
    while remaining > 0:
        n = min(remaining, scratch.numel())
        scratch.narrow(0, 0, n).uniform_(-bound, bound)
        out.append(scratch.narrow(0, 0, n).clone())
        remaining -= n
    return torch.cat(out)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "total_numel",
    [
        1023,
        _RNG_CHUNK_NUMEL - 1,
        _RNG_CHUNK_NUMEL,
        _RNG_CHUNK_NUMEL + 1,
        7 * 1024 * 1024 + 137,
    ],
)
def test_chunked_uniform_matches_single_uniform(dtype: torch.dtype, total_numel: int):
    bound = math.sqrt(6.0 / 4096)
    seed = 12345

    with temporary_seed(seed):
        single = torch.empty(total_numel, dtype=dtype, device="cpu")
        single.uniform_(-bound, bound)

    with temporary_seed(seed):
        chunked = _chunked_uniform(total_numel, dtype, bound)

    torch.testing.assert_close(single, chunked, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_skip_then_sample_matches_full(dtype: torch.dtype):
    """Full uniform_ == (skip pre) + (sample shard) + (skip post) per row-major slicing."""
    d_sae, d_in = 256, 128
    tp_size = 4
    bound = math.sqrt(6.0 / d_in)
    seed = 999

    with temporary_seed(seed):
        full = torch.empty(d_sae, d_in, dtype=dtype, device="cpu")
        full.uniform_(-bound, bound)

    shard_size = d_sae // tp_size
    for tp_rank in range(tp_size):
        pre = tp_rank * shard_size * d_in
        post = (tp_size - tp_rank - 1) * shard_size * d_in

        with temporary_seed(seed):
            _chunked_uniform(pre, dtype, bound)  # advance pre
            shard = torch.empty(shard_size, d_in, dtype=dtype, device="cpu")
            shard.uniform_(-bound, bound)
            _chunked_uniform(post, dtype, bound)  # advance post
            tail = torch.rand(8, dtype=torch.float32)

        with temporary_seed(seed):
            tmp = torch.empty(d_sae, d_in, dtype=dtype, device="cpu")
            tmp.uniform_(-bound, bound)
            full_tail = torch.rand(8, dtype=torch.float32)

        expected = full[tp_rank * shard_size : (tp_rank + 1) * shard_size, :].contiguous()
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)
        # RNG endpoint must equal full-init endpoint so external code in the
        # same temporary_seed scope is unaffected.
        torch.testing.assert_close(tail, full_tail, rtol=0, atol=0)
