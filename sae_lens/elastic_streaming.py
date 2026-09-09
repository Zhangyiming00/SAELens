"""Control plane and fixed rank layout for hot SAE-DP streaming switches.

The elastic ranks start as vLLM replicas and can be reassigned to SAE. Each
role keeps its configured topology for the lifetime of the run: elastic ranks
are partitioned into vLLM-TP groups while producing and into SAE PP x TP
replicas while training. Process groups for both SAE-DP sizes are created once
at startup so permanent vLLM ranks never participate in a cutover collective.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch.distributed as dist


ElasticPhase = Literal["active", "switching", "finished", "failed"]


@dataclass(frozen=True)
class ElasticStreamingLayout:
    world_size: int
    vllm_tp_size: int
    sae_tp_size: int
    sae_pp_size: int
    permanent_vllm_dp: int
    permanent_sae_dp: int
    elastic_rank_count: int

    def __post_init__(self) -> None:
        if self.vllm_tp_size < 1:
            raise ValueError("elastic streaming vllm_tp_size must be >= 1")
        if self.sae_tp_size < 1:
            raise ValueError("elastic streaming sae_tp_size must be >= 1")
        if self.sae_pp_size < 1:
            raise ValueError("elastic streaming sae_pp_size must be >= 1")
        if self.permanent_vllm_dp < 1:
            raise ValueError("elastic streaming requires permanent_vllm_dp >= 1")
        if self.permanent_sae_dp < 1:
            raise ValueError("elastic streaming requires permanent_sae_dp >= 1")
        if self.elastic_rank_count < 1:
            raise ValueError("elastic streaming requires at least one elastic rank")
        if self.elastic_rank_count % self.vllm_tp_size != 0:
            raise ValueError(
                "elastic rank count must be divisible by vllm_tp_size"
            )
        if self.elastic_rank_count % self.sae_replica_size != 0:
            raise ValueError(
                "elastic rank count must be divisible by sae_tp_size * sae_pp_size"
            )
        expected = (
            self.permanent_vllm_dp * self.vllm_tp_size
            + self.elastic_rank_count
            + self.permanent_sae_dp * self.sae_replica_size
        )
        if self.world_size != expected:
            raise ValueError(
                f"WORLD_SIZE={self.world_size} does not match elastic layout size "
                f"{expected}"
            )

    @classmethod
    def from_world_size(
        cls,
        *,
        world_size: int,
        vllm_tp_size: int,
        sae_tp_size: int,
        sae_pp_size: int,
        permanent_vllm_dp: int,
        permanent_sae_dp: int,
    ) -> "ElasticStreamingLayout":
        if vllm_tp_size < 1 or sae_tp_size < 1 or sae_pp_size < 1:
            raise ValueError("vLLM TP, SAE TP, and SAE PP sizes must all be >= 1")
        fixed_ranks = (
            permanent_vllm_dp * vllm_tp_size
            + permanent_sae_dp * sae_tp_size * sae_pp_size
        )
        elastic_ranks = world_size - fixed_ranks
        if elastic_ranks <= 0:
            raise ValueError(
                "WORLD_SIZE must leave one or more elastic ranks after permanent "
                "vLLM and SAE ranks are assigned"
            )
        sae_replica_size = sae_tp_size * sae_pp_size
        if (
            elastic_ranks % vllm_tp_size != 0
            or elastic_ranks % sae_replica_size != 0
        ):
            raise ValueError(
                "elastic rank count must be divisible by both vllm_tp_size and "
                "sae_tp_size * sae_pp_size; "
                f"got elastic_ranks={elastic_ranks}, vllm_tp_size={vllm_tp_size}, "
                f"sae_tp_size={sae_tp_size}, sae_pp_size={sae_pp_size}"
            )
        return cls(
            world_size=world_size,
            vllm_tp_size=vllm_tp_size,
            sae_tp_size=sae_tp_size,
            sae_pp_size=sae_pp_size,
            permanent_vllm_dp=permanent_vllm_dp,
            permanent_sae_dp=permanent_sae_dp,
            elastic_rank_count=elastic_ranks,
        )

    @property
    def sae_replica_size(self) -> int:
        return self.sae_tp_size * self.sae_pp_size

    @property
    def elastic_vllm_dp(self) -> int:
        return self.elastic_rank_count // self.vllm_tp_size

    @property
    def elastic_sae_dp(self) -> int:
        return self.elastic_rank_count // self.sae_replica_size

    @property
    def min_sae_dp(self) -> int:
        return self.permanent_sae_dp

    @property
    def max_sae_dp(self) -> int:
        return self.permanent_sae_dp + self.elastic_sae_dp

    @property
    def max_vllm_dp(self) -> int:
        return self.permanent_vllm_dp + self.elastic_vllm_dp

    @property
    def min_vllm_dp(self) -> int:
        return self.permanent_vllm_dp

    @property
    def permanent_vllm_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.permanent_vllm_dp * self.vllm_tp_size))

    @property
    def elastic_ranks(self) -> tuple[int, ...]:
        start = len(self.permanent_vllm_ranks)
        return tuple(range(start, start + self.elastic_rank_count))

    @property
    def permanent_sae_ranks(self) -> tuple[int, ...]:
        start = len(self.permanent_vllm_ranks) + len(self.elastic_ranks)
        return tuple(range(start, self.world_size))

    @property
    def transition_ranks(self) -> tuple[int, ...]:
        return self.permanent_sae_ranks + self.elastic_ranks

    def vllm_replica_ranks(self, replica_idx: int) -> tuple[int, ...]:
        if not 0 <= replica_idx < self.max_vllm_dp:
            raise ValueError(f"vLLM replica {replica_idx} is out of range")
        start = replica_idx * self.vllm_tp_size
        return tuple(range(start, start + self.vllm_tp_size))

    def sae_replica_ranks(self, replica_idx: int) -> tuple[int, ...]:
        if not 0 <= replica_idx < self.max_sae_dp:
            raise ValueError(f"SAE replica {replica_idx} is out of range")
        source = (
            self.permanent_sae_ranks
            if replica_idx < self.permanent_sae_dp
            else self.elastic_ranks
        )
        local_idx = (
            replica_idx
            if replica_idx < self.permanent_sae_dp
            else replica_idx - self.permanent_sae_dp
        )
        start = local_idx * self.sae_replica_size
        return source[start : start + self.sae_replica_size]

    def sae_stage_ranks(
        self, replica_idx: int, pp_rank: int
    ) -> tuple[int, ...]:
        if not 0 <= pp_rank < self.sae_pp_size:
            raise ValueError(f"SAE PP stage {pp_rank} is out of range")
        replica_ranks = self.sae_replica_ranks(replica_idx)
        start = pp_rank * self.sae_tp_size
        return replica_ranks[start : start + self.sae_tp_size]

    def active_sae_ranks(self, sae_dp: int) -> tuple[int, ...]:
        self.validate_sae_dp(sae_dp)
        return tuple(
            rank
            for replica_idx in range(sae_dp)
            for rank in self.sae_replica_ranks(replica_idx)
        )

    def validate_sae_dp(self, sae_dp: int) -> None:
        if sae_dp not in (self.min_sae_dp, self.max_sae_dp):
            raise ValueError(
                "the first elastic streaming implementation switches all elastic "
                f"groups together; sae_dp must be {self.min_sae_dp} or "
                f"{self.max_sae_dp}, got {sae_dp}"
            )

    def role(self, rank: int, sae_dp: int) -> Literal["vllm", "sae"]:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank={rank} is out of range")
        self.validate_sae_dp(sae_dp)
        if rank in self.permanent_vllm_ranks:
            return "vllm"
        if rank in self.permanent_sae_ranks:
            return "sae"
        return "sae" if sae_dp == self.max_sae_dp else "vllm"


@dataclass(frozen=True)
class ElasticControlState:
    version: int
    phase: ElasticPhase
    epoch: int
    active_sae_dp: int
    target_sae_dp: int
    permanent_vllm_dp: int
    permanent_sae_dp: int
    elastic_rank_count: int
    vllm_tp_size: int
    sae_tp_size: int
    sae_pp_size: int
    ready_for_cutover: bool
    updated_at: float
    error: str | None = None
    buffer_name: str = ""
    buffer_num_chunks: int = 0
    buffer_chunk_size_tokens: int = 0


class ElasticStreamingController:
    """Atomic file-backed hot-switch control state.

    This file is coordination metadata, not a training checkpoint.  Model,
    optimizer, provider and buffer state remain in the live worker processes.
    """

    def __init__(self, path: Path | str, layout: ElasticStreamingLayout) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.layout = layout

    def initialize(self, *, overwrite: bool = False) -> ElasticControlState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists() and not overwrite:
                state = self._read_unlocked()
                self._validate_layout(state)
                return state
            state = ElasticControlState(
                version=3,
                phase="active",
                epoch=0,
                active_sae_dp=self.layout.min_sae_dp,
                target_sae_dp=self.layout.min_sae_dp,
                permanent_vllm_dp=self.layout.permanent_vllm_dp,
                permanent_sae_dp=self.layout.permanent_sae_dp,
                elastic_rank_count=self.layout.elastic_rank_count,
                vllm_tp_size=self.layout.vllm_tp_size,
                sae_tp_size=self.layout.sae_tp_size,
                sae_pp_size=self.layout.sae_pp_size,
                ready_for_cutover=False,
                updated_at=time.time(),
            )
            self._write_unlocked(state)
            return state

    def configure_buffer_monitoring(
        self,
        *,
        buffer_name: str,
        num_chunks: int,
        chunk_size_tokens: int,
    ) -> ElasticControlState:
        """Publish immutable SHM metadata for an out-of-process monitor.

        This is called once during startup. Runtime throughput monitoring reads
        the existing buffer memmaps and never writes from a training rank.
        """
        if not buffer_name:
            raise ValueError("elastic monitoring requires a non-empty buffer name")
        if num_chunks < 1 or chunk_size_tokens < 1:
            raise ValueError("elastic monitoring buffer dimensions must be positive")
        with self._locked():
            current = self._read_unlocked()
            self._validate_layout(current)
            configured = ElasticControlState(
                **{
                    **asdict(current),
                    "version": max(3, current.version),
                    "buffer_name": buffer_name,
                    "buffer_num_chunks": num_chunks,
                    "buffer_chunk_size_tokens": chunk_size_tokens,
                    "updated_at": time.time(),
                }
            )
            self._write_unlocked(configured)
            return configured

    def read(self) -> ElasticControlState:
        with self._locked(shared=True):
            state = self._read_unlocked()
            self._validate_layout(state)
            return state

    def request(self, target_sae_dp: int) -> ElasticControlState:
        self.layout.validate_sae_dp(target_sae_dp)
        with self._locked():
            current = self._read_unlocked()
            self._validate_layout(current)
            if current.phase != "active":
                raise RuntimeError(
                    f"cannot request a switch while control phase is {current.phase!r}"
                )
            if target_sae_dp == current.active_sae_dp:
                raise ValueError(f"sae_dp={target_sae_dp} is already active")
            requested = ElasticControlState(
                **{
                    **asdict(current),
                    "phase": "switching",
                    "epoch": current.epoch + 1,
                    "target_sae_dp": target_sae_dp,
                    # Shrinking SAE can stop directly at the next optimizer
                    # boundary. Expansion waits until elastic ranks have torn
                    # down vLLM and constructed their TP-sharded SAE modules.
                    "ready_for_cutover": target_sae_dp == self.layout.min_sae_dp,
                    "updated_at": time.time(),
                    "error": None,
                }
            )
            self._write_unlocked(requested)
            return requested

    def mark_ready(self, *, epoch: int) -> ElasticControlState:
        with self._locked():
            current = self._read_unlocked()
            if current.phase != "switching" or current.epoch != epoch:
                raise RuntimeError(
                    f"stale elastic ready epoch={epoch}; current state is "
                    f"{current.phase}@{current.epoch}"
                )
            ready = ElasticControlState(
                **{
                    **asdict(current),
                    "ready_for_cutover": True,
                    "updated_at": time.time(),
                }
            )
            self._write_unlocked(ready)
            return ready

    def commit(self, *, epoch: int, active_sae_dp: int) -> ElasticControlState:
        self.layout.validate_sae_dp(active_sae_dp)
        with self._locked():
            current = self._read_unlocked()
            if current.phase != "switching" or current.epoch != epoch:
                raise RuntimeError(
                    f"stale elastic commit epoch={epoch}; current state is "
                    f"{current.phase}@{current.epoch}"
                )
            if current.target_sae_dp != active_sae_dp:
                raise RuntimeError(
                    f"commit sae_dp={active_sae_dp} does not match requested "
                    f"target {current.target_sae_dp}"
                )
            committed = ElasticControlState(
                **{
                    **asdict(current),
                    "phase": "active",
                    "active_sae_dp": active_sae_dp,
                    "ready_for_cutover": False,
                    "updated_at": time.time(),
                }
            )
            self._write_unlocked(committed)
            return committed

    def finish(self, *, epoch: int) -> ElasticControlState:
        with self._locked():
            current = self._read_unlocked()
            # A request that acquired the lock first must complete. Returning
            # it lets workers perform that cutover even if training consumed
            # its final batch immediately before observing the request.
            if current.phase == "switching":
                return current
            if current.phase == "finished" and current.epoch == epoch:
                return current
            if current.epoch != epoch:
                raise RuntimeError("cannot finish a stale elastic epoch")
            if current.phase != "active":
                raise RuntimeError(
                    f"cannot finish while control phase is {current.phase!r}"
                )
            finished = ElasticControlState(
                **{
                    **asdict(current),
                    "phase": "finished",
                    "target_sae_dp": current.active_sae_dp,
                    "ready_for_cutover": False,
                    "updated_at": time.time(),
                }
            )
            self._write_unlocked(finished)
            return finished

    def fail(self, *, epoch: int, error: str) -> ElasticControlState:
        with self._locked():
            current = self._read_unlocked()
            if current.epoch != epoch:
                return current
            failed = ElasticControlState(
                **{
                    **asdict(current),
                    "phase": "failed",
                    "updated_at": time.time(),
                    "error": error,
                }
            )
            self._write_unlocked(failed)
            return failed

    def requested_target(
        self,
        *,
        local_epoch: int,
        require_ready: bool = False,
    ) -> tuple[int, int] | None:
        state = self.read()
        if (
            state.phase == "switching"
            and state.epoch > local_epoch
            and (state.ready_for_cutover or not require_ready)
        ):
            return state.epoch, state.target_sae_dp
        return None

    def _validate_layout(self, state: ElasticControlState) -> None:
        expected = (
            self.layout.permanent_vllm_dp,
            self.layout.permanent_sae_dp,
            self.layout.elastic_rank_count,
            self.layout.vllm_tp_size,
            self.layout.sae_tp_size,
            self.layout.sae_pp_size,
        )
        actual = (
            state.permanent_vllm_dp,
            state.permanent_sae_dp,
            state.elastic_rank_count,
            state.vllm_tp_size,
            state.sae_tp_size,
            state.sae_pp_size,
        )
        if actual != expected:
            raise ValueError(
                f"elastic control layout {actual} does not match this run {expected}"
            )
        self.layout.validate_sae_dp(state.active_sae_dp)
        self.layout.validate_sae_dp(state.target_sae_dp)

    def _read_unlocked(self) -> ElasticControlState:
        return ElasticControlState(**json.loads(self.path.read_text()))

    def _write_unlocked(self, state: ElasticControlState) -> None:
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    @contextmanager
    def _locked(self, *, shared: bool = False) -> Generator[None, None, None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_file:
            lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_file.fileno(), lock_mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class _SAEGroups:
    tp_groups: tuple[Any, ...]
    tp_cpu_groups: tuple[Any, ...]
    dp_groups: tuple[Any, ...]
    pp_root_groups: tuple[Any | None, ...]


class ElasticSAEContext:
    """Subset of distributed_streaming used by SAE construction/providers."""

    def __init__(
        self,
        runtime: "ElasticDistributedRuntime",
        sae_dp: int,
    ) -> None:
        runtime.layout.validate_sae_dp(sae_dp)
        self.runtime = runtime
        self.layout = runtime.layout
        self.sae_dp = sae_dp
        self.rank = dist.get_rank()
        self._replica_idx = -1
        self._pp_rank = -1
        self._tp_rank = -1
        for replica_idx in range(sae_dp):
            for pp_rank in range(self.layout.sae_pp_size):
                ranks = self.layout.sae_stage_ranks(replica_idx, pp_rank)
                if self.rank in ranks:
                    self._replica_idx = replica_idx
                    self._pp_rank = pp_rank
                    self._tp_rank = ranks.index(self.rank)
                    break
            if self._replica_idx >= 0:
                break

    def is_consumer(self) -> bool:
        return self._replica_idx >= 0

    def get_sae_dp_size(self) -> int:
        return self.sae_dp

    def get_sae_dp_idx(self) -> int:
        if not self.is_consumer():
            return -1
        # torch.distributed sorts the global ranks passed to new_group().  The
        # layout intentionally lists permanent SAE replicas before elastic
        # replicas, so the replica index is not necessarily the process-group
        # rank after expansion.
        return dist.get_rank(self.get_sae_dp_group())

    def get_sae_tp_size(self) -> int:
        return self.layout.sae_tp_size

    def get_sae_tp_rank(self) -> int:
        return self._tp_rank

    def is_sae_tp_root(self) -> bool:
        return self._tp_rank == 0

    def get_sae_tp_group(self) -> Any | None:
        if not self.is_consumer():
            return None
        endpoint_idx = self._replica_idx * self.layout.sae_pp_size + self._pp_rank
        return self.runtime.groups[self.sae_dp].tp_groups[endpoint_idx]

    def get_sae_tp_cpu_group(self) -> Any | None:
        if not self.is_consumer():
            return None
        endpoint_idx = self._replica_idx * self.layout.sae_pp_size + self._pp_rank
        return self.runtime.groups[self.sae_dp].tp_cpu_groups[endpoint_idx]

    def get_sae_dp_group(self) -> Any | None:
        if not self.is_consumer():
            return None
        group_idx = self._pp_rank * self.layout.sae_tp_size + self._tp_rank
        return self.runtime.groups[self.sae_dp].dp_groups[group_idx]

    def get_consumer_tp_root(self) -> int:
        if not self.is_consumer():
            return -1
        return self.layout.sae_stage_ranks(self._replica_idx, self._pp_rank)[0]

    def get_sae_pp_rank(self) -> int:
        return self._pp_rank

    def get_sae_pp_size(self) -> int:
        return self.layout.sae_pp_size

    def get_sae_pp_root_group(self) -> Any | None:
        if not self.is_consumer() or self._tp_rank != 0:
            return None
        return self.runtime.groups[self.sae_dp].pp_root_groups[self._replica_idx]

    def get_sae_pp_root_global_rank(self) -> int:
        if not self.is_consumer():
            return -1
        return self.layout.sae_stage_ranks(self._replica_idx, 0)[0]


class ElasticDistributedRuntime:
    """Pre-created SAE min/max groups plus a cutover-only Gloo group."""

    def __init__(self, layout: ElasticStreamingLayout) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if dist.get_world_size() != layout.world_size:
            raise ValueError("distributed world size does not match elastic layout")
        self.layout = layout
        self.groups: dict[int, _SAEGroups] = {}
        for sae_dp in (layout.min_sae_dp, layout.max_sae_dp):
            tp_groups = []
            tp_cpu_groups = []
            for replica_idx in range(sae_dp):
                for pp_rank in range(layout.sae_pp_size):
                    ranks = list(layout.sae_stage_ranks(replica_idx, pp_rank))
                    tp_groups.append(dist.new_group(ranks, backend="nccl"))
                    tp_cpu_groups.append(dist.new_group(ranks, backend="gloo"))
            dp_groups = []
            for pp_rank in range(layout.sae_pp_size):
                for tp_rank in range(layout.sae_tp_size):
                    ranks = [
                        layout.sae_stage_ranks(replica_idx, pp_rank)[tp_rank]
                        for replica_idx in range(sae_dp)
                    ]
                    dp_groups.append(dist.new_group(ranks, backend="nccl"))
            pp_root_groups: list[Any | None] = []
            for replica_idx in range(sae_dp):
                if layout.sae_pp_size == 1:
                    pp_root_groups.append(None)
                    continue
                ranks = [
                    layout.sae_stage_ranks(replica_idx, pp_rank)[0]
                    for pp_rank in range(layout.sae_pp_size)
                ]
                pp_root_groups.append(dist.new_group(ranks, backend="nccl"))
            self.groups[sae_dp] = _SAEGroups(
                tp_groups=tuple(tp_groups),
                tp_cpu_groups=tuple(tp_cpu_groups),
                dp_groups=tuple(dp_groups),
                pp_root_groups=tuple(pp_root_groups),
            )
        self.transition_group = dist.new_group(
            list(layout.transition_ranks), backend="gloo"
        )
        self.elastic_group = dist.new_group(list(layout.elastic_ranks), backend="gloo")

    def context(self, sae_dp: int) -> ElasticSAEContext:
        return ElasticSAEContext(self, sae_dp)

    def is_permanent_vllm(self) -> bool:
        return dist.get_rank() in self.layout.permanent_vllm_ranks

    def is_elastic(self) -> bool:
        return dist.get_rank() in self.layout.elastic_ranks

    def is_permanent_sae(self) -> bool:
        return dist.get_rank() in self.layout.permanent_sae_ranks

    def is_elastic_vllm_tp_root(self) -> bool:
        rank = dist.get_rank()
        if rank not in self.layout.elastic_ranks:
            return False
        return (
            self.layout.elastic_ranks.index(rank) % self.layout.vllm_tp_size == 0
        )

    def elastic_barrier(self) -> None:
        if self.is_elastic():
            dist.barrier(group=self.elastic_group)

    def transition_barrier(self) -> None:
        if dist.get_rank() in self.layout.transition_ranks:
            dist.barrier(group=self.transition_group)
