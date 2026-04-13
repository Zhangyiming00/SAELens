from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress

import torch
import torch.distributed as dist


def _is_nccl_group(group: dist.ProcessGroup | None) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return str(dist.get_backend(group)).lower() == "nccl"
    except Exception:
        return False


def _can_emit_nvtx(group: dist.ProcessGroup | None) -> bool:
    return torch.cuda.is_available() and _is_nccl_group(group)


@contextmanager
def nccl_nvtx_range(
    message: str,
    group: dist.ProcessGroup | None = None,
) -> Iterator[None]:
    """Emit an NVTX range only when the target process group uses NCCL.

    Profiling labels must not affect CPU/Gloo runs or tests with mocked process
    groups, so every failure to detect or emit NVTX is treated as a no-op.
    """
    if not _can_emit_nvtx(group):
        yield
        return

    pushed = False
    try:
        torch.cuda.nvtx.range_push(message)
        pushed = True
    except Exception:
        pushed = False

    try:
        yield
    finally:
        if pushed:
            with suppress(Exception):
                torch.cuda.nvtx.range_pop()
