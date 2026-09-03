import fcntl
import hashlib
import os
import threading
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sae_lens.profiling import nccl_nvtx_range
from sae_lens.saes.sae import TrainingSAE
from sae_lens.saes.topk_sae import TopKTrainingSAE


class DDPOptimizerOverlapState:
    """Track real asynchronous work from one DDP reducer per SAE hook.

    Each reducer launches a bucket as soon as that bucket is ready, preserving
    DDP's backward/communication overlap and tensor coalescing.  The comm hook
    returns the in-place bucket immediately so DDP does not add a wait for all
    communication to the backward stream.  Optimizer streams separately wait
    for the real Work objects after all backward computation has been enqueued.
    Per-hook reducers prevent a final mixed bucket from coupling every SAE.
    """

    def __init__(
        self,
        hook_names: list[str],
        ddp_by_hook: dict[str, DDP],
        group: dist.ProcessGroup,
    ) -> None:
        self.hook_names = list(hook_names)
        self.group = group
        self.world_size = dist.get_world_size(group)
        if set(ddp_by_hook) != set(self.hook_names):
            raise ValueError(
                "DDP optimizer overlap requires the exact per-hook DDP set"
            )
        self.ddp_by_hook = dict(ddp_by_hook)
        self._hook_by_param_id: dict[int, str] = {}
        for hook_name in self.hook_names:
            ddp = self.ddp_by_hook[hook_name]
            if not isinstance(ddp, DDP):
                raise TypeError(
                    f"DDP optimizer overlap hook {hook_name!r} has wrapper "
                    f"{type(ddp).__name__}"
                )
            if not ddp.gradient_as_bucket_view:
                raise ValueError(
                    "DDP optimizer overlap requires gradient_as_bucket_view=True"
                )
            params = [
                param for param in ddp.module.parameters() if param.requires_grad
            ]
            if not params:
                raise ValueError(
                    f"DDP optimizer overlap hook {hook_name!r} has no parameters"
                )
            for param in params:
                param_id = id(param)
                if param_id in self._hook_by_param_id:
                    raise ValueError(
                        "DDP optimizer overlap requires disjoint parameters per hook"
                    )
                self._hook_by_param_id[param_id] = hook_name
        self._lock = threading.Lock()
        self._step_active = False
        self._work_by_hook: dict[str, list[dist.Work]] = {}
        self._bucket_buffers_by_hook: dict[str, list[torch.Tensor]] = {}
        self._launch_index_by_hook: dict[str, int] = {}
        self._launch_index = 0
        for hook_name in self.hook_names:
            self.ddp_by_hook[hook_name].register_comm_hook(
                hook_name,
                self._comm_hook,
            )

    def begin_step(self) -> None:
        with self._lock:
            if self._step_active:
                raise RuntimeError(
                    "previous DDP optimizer-overlap step is still active"
                )
            self._step_active = True
            self._work_by_hook = {
                hook_name: [] for hook_name in self.hook_names
            }
            self._bucket_buffers_by_hook = {
                hook_name: [] for hook_name in self.hook_names
            }
            self._launch_index_by_hook.clear()
            self._launch_index = 0

    @torch.no_grad()
    def _comm_hook(self, hook_name, bucket):
        """Launch one ready bucket and return without waiting for its Work."""
        with self._lock:
            if not self._step_active:
                raise RuntimeError(
                    "DDP optimizer-overlap bucket hook ran outside an active step"
                )
            if hook_name not in self.ddp_by_hook:
                raise RuntimeError(
                    f"DDP optimizer overlap received unknown hook {hook_name!r}"
                )
            buffer = bucket.buffer()
            with nccl_nvtx_range(
                f"nccl:multi_sae:{hook_name}:dp_bucket_launch_v4",
                self.group,
            ):
                if self.world_size > 1:
                    buffer.div_(self.world_size)
                    work = dist.all_reduce(
                        buffer,
                        group=self.group,
                        async_op=True,
                    )
                    self._work_by_hook[hook_name].append(work)
                    self._bucket_buffers_by_hook[hook_name].append(buffer)
            if bucket.is_last():
                if hook_name in self._launch_index_by_hook:
                    raise RuntimeError(
                        f"DDP optimizer overlap saw two final buckets for {hook_name!r}"
                    )
                self._launch_index_by_hook[hook_name] = self._launch_index
                self._launch_index += 1

        future = torch.futures.Future()
        future.set_result(buffer)
        return future

    def end_backward(self) -> None:
        """Validate that every per-hook reducer launched its final bucket."""

        with self._lock:
            if not self._step_active:
                raise RuntimeError("DDP optimizer-overlap step is not active")
            missing = [
                hook_name
                for hook_name in self.hook_names
                if hook_name not in self._launch_index_by_hook
            ]
            if missing:
                raise RuntimeError(
                    "combined backward did not launch every per-hook DDP reducer: "
                    f"{missing}"
                )

    def completion_order(self, fallback_order: list[str]) -> list[str]:
        order_index = {name: idx for idx, name in enumerate(fallback_order)}
        with self._lock:
            return sorted(
                self.hook_names,
                key=lambda name: (
                    self._launch_index_by_hook.get(name, 1 << 60),
                    order_index.get(name, len(order_index)),
                ),
            )

    @torch.no_grad()
    def wait_for_hook(self, hook_name: str) -> None:
        """Make the current stream wait for this hook's real bucket reductions."""

        with self._lock:
            works = list(self._work_by_hook.get(hook_name, ()))
        for work in works:
            # For ProcessGroupNCCL this inserts a stream dependency without a
            # host wait.  Gloo waits synchronously, which is appropriate on CPU.
            work.wait()

    def finish_step(self) -> None:
        """Release Work references after optimizer completion."""

        with self._lock:
            self._work_by_hook.clear()
            self._bucket_buffers_by_hook.clear()
            self._launch_index_by_hook.clear()
            self._step_active = False

    def close(self) -> None:
        """Validate shutdown; DDP owns comm-hook lifetime with its reducers."""
        with self._lock:
            if self._step_active:
                raise RuntimeError("cannot close an active DDP optimizer-overlap step")


class TPPostSharedMemory:
    """Single-node CPU-SHM reducer for optimizer-critical TP-post traffic.

    This is intentionally separate from the activation streaming shared-memory
    transport.  It carries only the small replicated-gradient synchronization
    and scalar global-norm reduction needed after DP reduction.
    """

    _HEADER_I64 = 4

    def __init__(
        self,
        *,
        tp_group: dist.ProcessGroup,
        max_numel: int,
        output_path: str | None,
    ) -> None:
        self.group = tp_group
        self.rank = dist.get_rank(tp_group)
        self.size = dist.get_world_size(tp_group)
        self.max_numel = max(1, int(max_numel))
        if hasattr(dist, "get_process_group_ranks"):
            ranks = list(dist.get_process_group_ranks(tp_group))
        elif hasattr(dist, "get_global_rank"):
            ranks = [dist.get_global_rank(tp_group, i) for i in range(self.size)]
        else:
            # Very old torch fallback; all supported project environments expose
            # one of the two APIs above.
            ranks = list(range(self.size))
        local_world = int(
            os.environ.get("LOCAL_WORLD_SIZE", str(dist.get_world_size()))
        )
        my_node = dist.get_rank() // max(local_world, 1)
        if any((int(rank) // max(local_world, 1)) != my_node for rank in ranks):
            raise ValueError(
                "cpu_shm TP-post currently requires a single-node TP group"
            )

        # Give every launch/restart a fresh SHM name.  Broadcasting one scalar at
        # initialization is safe and avoids a non-root process ever attaching to
        # a stale segment from a crashed run with the same MASTER_PORT/output.
        nonce_t = torch.zeros(
            (),
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if self.rank == 0:
            nonce_t.fill_(time.time_ns() & ((1 << 63) - 1))
        dist.broadcast(nonce_t, src=int(ranks[0]), group=tp_group)
        launch_nonce = int(nonce_t.item())

        run_key = os.environ.get("TORCHELASTIC_RUN_ID") or (
            os.environ.get("MASTER_ADDR", "local")
            + ":"
            + os.environ.get("MASTER_PORT", "0")
        )
        restart_count = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
        digest = hashlib.sha1(
            (
                f"{run_key}|{restart_count}|{launch_nonce}|{output_path}|"
                f"{','.join(map(str, ranks))}"
            ).encode()
        ).hexdigest()[:16]
        self.name = f"saelens_optpost_v2_{digest}"
        self.lock_path = Path("/tmp") / f"{self.name}.lock"
        self.lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)

        header_i64 = self._HEADER_I64 + self.size + 2
        floats = self.size * self.max_numel + self.max_numel
        doubles = self.size + 1
        nbytes = header_i64 * 8 + floats * 4 + doubles * 8
        if self.rank == 0:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=self.name,
                    create=True,
                    size=nbytes,
                )
                self.shm.buf[:] = b"\x00" * nbytes
            except FileExistsError:
                stale = shared_memory.SharedMemory(name=self.name, create=False)
                stale.close()
                stale.unlink()
                self.shm = shared_memory.SharedMemory(
                    name=self.name,
                    create=True,
                    size=nbytes,
                )
                self.shm.buf[:] = b"\x00" * nbytes
        else:
            deadline = time.monotonic() + 30.0
            while True:
                try:
                    self.shm = shared_memory.SharedMemory(
                        name=self.name,
                        create=False,
                    )
                    # Python <=3.12 registers create=False attachments with the
                    # local resource_tracker.  Only TP rank0 owns/unlinks this
                    # segment; unregister other ranks so an early process exit
                    # cannot remove SHM out from under its peers.
                    try:
                        resource_tracker.unregister(self.shm._name, "shared_memory")
                    except (AttributeError, KeyError):
                        pass
                    break
                except FileNotFoundError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"timed out attaching optimizer TP-post shm {self.name}"
                        )
                    time.sleep(0.001)

        offset = 0
        self.header = np.ndarray(
            (header_i64,), dtype=np.int64, buffer=self.shm.buf, offset=offset
        )
        offset += header_i64 * 8
        self.slots = np.ndarray(
            (self.size, self.max_numel),
            dtype=np.float32,
            buffer=self.shm.buf,
            offset=offset,
        )
        offset += self.size * self.max_numel * 4
        self.result = np.ndarray(
            (self.max_numel,), dtype=np.float32, buffer=self.shm.buf, offset=offset
        )
        offset += self.max_numel * 4
        self.scalar_slots = np.ndarray(
            (self.size,), dtype=np.float64, buffer=self.shm.buf, offset=offset
        )
        offset += self.size * 8
        self.scalar_result = np.ndarray(
            (1,), dtype=np.float64, buffer=self.shm.buf, offset=offset
        )
        self._seq = 0
        self._ready_base = self._HEADER_I64
        self._vector_done_idx = self._HEADER_I64 + self.size
        self._scalar_done_idx = self._vector_done_idx + 1

    def _locked(self):
        class _Guard:
            def __init__(self, owner: TPPostSharedMemory) -> None:
                self.owner = owner

            def __enter__(self) -> None:
                fcntl.flock(self.owner.lock_fd, fcntl.LOCK_EX)

            def __exit__(self, exc_type, exc, tb) -> None:
                fcntl.flock(self.owner.lock_fd, fcntl.LOCK_UN)

        return _Guard(self)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def reduce_vector_sum(self, values: np.ndarray) -> np.ndarray:
        n = int(values.size)
        if n > self.max_numel:
            raise ValueError(
                f"TP-post vector has {n} elements > max_numel={self.max_numel}"
            )
        seq = self._next_seq()
        with self._locked():
            self.slots[self.rank, :n] = values.reshape(-1).astype(
                np.float32, copy=False
            )
            self.header[self._ready_base + self.rank] = seq
        while True:
            with self._locked():
                ready = np.all(
                    self.header[self._ready_base : self._ready_base + self.size] >= seq
                )
                if (
                    self.rank == 0
                    and ready
                    and self.header[self._vector_done_idx] < seq
                ):
                    self.result[:n] = self.slots[:, :n].sum(
                        axis=0, dtype=np.float32
                    )
                    self.header[self._vector_done_idx] = seq
                if self.header[self._vector_done_idx] >= seq:
                    return self.result[:n].copy()
            time.sleep(0)

    def reduce_scalar_sum(self, value: float) -> float:
        seq = self._next_seq()
        with self._locked():
            self.scalar_slots[self.rank] = float(value)
            self.header[self._ready_base + self.rank] = seq
        while True:
            with self._locked():
                ready = np.all(
                    self.header[self._ready_base : self._ready_base + self.size] >= seq
                )
                if (
                    self.rank == 0
                    and ready
                    and self.header[self._scalar_done_idx] < seq
                ):
                    self.scalar_result[0] = self.scalar_slots.sum(dtype=np.float64)
                    self.header[self._scalar_done_idx] = seq
                if self.header[self._scalar_done_idx] >= seq:
                    return float(self.scalar_result[0])
            time.sleep(0)

    def close(self) -> None:
        self.shm.close()
        if self.rank == 0:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
        try:
            os.close(self.lock_fd)
        except OSError:
            pass
        if self.rank == 0:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass


def tp_post_cpu_shm_prepare_clip(
    sae: TopKTrainingSAE,
    reducer: TPPostSharedMemory,
    *,
    max_norm: float = 1.0,
) -> float:
    """Synchronize replicated TP grads and compute the TP-global clip factor."""

    shard_dims = sae._tp_param_shard_dims()
    with torch.no_grad():
        for name, param in sae.named_parameters():
            if param.grad is None or shard_dims.get(name) is not None:
                continue
            host = param.grad.detach().float().cpu().numpy().reshape(-1)
            reduced = reducer.reduce_vector_sum(host)
            param.grad.copy_(
                torch.from_numpy(reduced)
                .to(param.grad.device, dtype=param.grad.dtype)
                .view_as(param.grad)
            )

        local_sq = 0.0
        tp_rank = dist.get_rank(sae._tp_group) if sae._tp_group is not None else 0
        for name, param in sae.named_parameters():
            if param.grad is None:
                continue
            if shard_dims.get(name) is None and tp_rank != 0:
                continue
            local_sq += float(param.grad.detach().float().pow(2).sum().item())
        total_norm = reducer.reduce_scalar_sum(local_sq) ** 0.5
        return min(max_norm / (total_norm + 1e-6), 1.0)


def apply_clip_coef_(sae: TrainingSAE[Any], coef: float) -> None:
    with torch.no_grad():
        for param in sae.parameters():
            if param.grad is not None:
                param.grad.mul_(coef)
