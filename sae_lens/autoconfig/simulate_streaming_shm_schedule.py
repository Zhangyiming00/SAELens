#!/usr/bin/env python3
"""Discrete-event simulator for SAELens SHM streaming scheduling.

This simulator intentionally separates *intrinsic* operation timing from
streaming scheduling:

  vLLM intrinsic activation generation time
      -> imported from simulate_vllm_time.py
  SAE intrinsic training-step time
      -> imported from simulate_sae_step_time_step_v6.py
  D2H / H2D transfer time
      -> piecewise-linear interpolation from streaming_transfer_profile.csv
  SHM producer/consumer scheduling
      -> modeled here as a discrete-event state machine

The simulator writes three trace files with the same logical role/schema as the
real streaming run:

  shm_log_vllm.jsonl
  shm_log_sae.jsonl
  buffer_monitor.jsonl

It does NOT model dataset fetch, SHM memcpy/write, CPU concat/reinterleave,
shuffle/mixing kernel time, Python/lock/backoff overhead, CUDA contention, or
startup/model-load time.  Shuffle/mixing parameters DO affect scheduling and
pool occupancy, but their execution time is zero by design.

The SHM scheduling model follows SAELens streaming v1 semantics:

Producer:
  FREE -> WRITING -> READY
  A producer must claim a FREE slot before beginning chunk production.

Consumer:
  READY -> CONSUMING -> FREE
  acquire_up_to(prefetch_chunks) returns immediately if >=1 READY chunk exists;
  it does not wait for a full prefetch batch.

Local mixing:
  When mix_chunks > 0, data is accumulated until
      max(train_batch_size_tokens, mix_chunks * chunk_size_tokens)
  logical tokens are available.  Then a zero-time shuffle/split exposes full
  training batches while retaining approximately mix_fraction of capacity.
  At upstream EOF, remaining local data is drained.

Release policy:
  repo        : release SHM slots at acquire/read-copy time, before modeled H2D
                (matches current StreamingActivationProvider ordering).
  after_fetch : keep slots CONSUMING until modeled H2D completes; useful as a
                conservative simplified planner model.

Important vLLM detail:
  get_streaming_activations() preserves residual activation tokens across chunk
  boundaries.  Each producer therefore maintains residual_tokens; the number
  of underlying activation-generation calls can vary by chunk.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_VLLM_SIM = HERE / "simulate_vllm_time.py"
DEFAULT_SAE_SIM = HERE / "simulate_sae_step_time_step_v6.py"
DEFAULT_TRANSFER_PROFILE = (HERE / "profile_results" / "streaming_transfer_profile.csv") if (HERE / "profile_results" / "streaming_transfer_profile.csv").exists() else (HERE / "streaming_transfer_profile.csv")

DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
}


class SimError(RuntimeError):
    pass


def canonical_dtype(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "fp32": "float32",
        "float": "float32",
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
    }
    key = aliases.get(key, key)
    if key not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported dtype {name!r}")
    return key


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    v = str(text).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {text!r}")


def load_module(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise SimError(f"module does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SimError(f"cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# Transfer oracle
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferEstimate:
    ms: float
    source: str
    lo_bytes: int
    hi_bytes: int


class TransferTimeOracle:
    """1-D exact / piecewise-linear transfer timing over byte size."""

    def __init__(self, path: Path, *, time_column: str = "median_ms") -> None:
        if not path.exists():
            raise SimError(f"transfer profile does not exist: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.curves: dict[str, list[tuple[int, float]]] = {}
        for direction in ("h2d", "d2h"):
            vals: dict[int, list[float]] = {}
            for r in rows:
                if str(r.get("direction", "")).lower() != direction:
                    continue
                try:
                    b = int(float(r["buffer_bytes"]))
                    t = float(r[time_column])
                except (KeyError, TypeError, ValueError):
                    continue
                vals.setdefault(b, []).append(t)
            if not vals:
                raise SimError(f"no {direction} rows in {path}")
            self.curves[direction] = sorted(
                (b, float(statistics.median(ts))) for b, ts in vals.items()
            )

    def estimate(self, direction: str, nbytes: int) -> TransferEstimate:
        direction = direction.lower()
        if direction not in self.curves:
            raise ValueError(direction)
        x = max(0, int(nbytes))
        if x == 0:
            return TransferEstimate(0.0, "zero", 0, 0)
        curve = self.curves[direction]
        for b, t in curve:
            if x == b:
                return TransferEstimate(t, "exact", b, b)
        # Linear extrapolation below/above the sampled range uses the closest
        # segment.  This keeps the oracle simple and deterministic.
        if x < curve[0][0]:
            lo, hi = curve[0], curve[1]
            source = "linear_extrapolated_low"
        elif x > curve[-1][0]:
            lo, hi = curve[-2], curve[-1]
            source = "linear_extrapolated_high"
        else:
            lo = curve[0]
            hi = curve[-1]
            source = "linear"
            for a, b in zip(curve, curve[1:]):
                if a[0] < x < b[0]:
                    lo, hi = a, b
                    break
        (x0, y0), (x1, y1) = lo, hi
        if x1 == x0:
            y = y0
        else:
            y = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
        return TransferEstimate(max(0.0, float(y)), source, x0, x1)

    def h2d_ms(self, nbytes: int) -> TransferEstimate:
        return self.estimate("h2d", nbytes)

    def d2h_ms(self, nbytes: int) -> TransferEstimate:
        return self.estimate("d2h", nbytes)


# -----------------------------------------------------------------------------
# Existing simulator adapters
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmTiming:
    ms: float
    n_calls: int
    tokens_per_generation_call: int
    simulator_calls_per_generation_call: float
    simulator_method: str
    source_stop_layers: tuple[int, ...]


class VllmTimeOracle:
    def __init__(
        self,
        simulator_path: Path,
        profile_path: Path | None,
        *,
        tp: int,
        stop_at_layer: int,
        preferred_total_tokens: int | None,
        constant_ms_per_generation_call: float | None = None,
    ) -> None:
        self.tp = int(tp)
        self.stop_at_layer = int(stop_at_layer)
        self.preferred_total_tokens = preferred_total_tokens
        self.constant = constant_ms_per_generation_call
        self.mod: ModuleType | None = None
        self.rows: Any = None
        self._cache: dict[tuple[int, int], VllmTiming] = {}
        if self.constant is None:
            self.mod = load_module(simulator_path, "_streaming_vllm_time_oracle")
            if profile_path is None:
                profile_path = Path(self.mod.DEFAULT_PROFILE_JSON)
            self.rows = self.mod.load_profile_auto(profile_path)

    def estimate_generation_calls(
        self,
        *,
        n_calls: int,
        tokens_per_generation_call: int,
    ) -> VllmTiming:
        if n_calls < 0:
            raise ValueError("n_calls must be >=0")
        key = (int(n_calls), int(tokens_per_generation_call))
        if key in self._cache:
            return self._cache[key]
        if n_calls == 0:
            out = VllmTiming(0.0, 0, tokens_per_generation_call, 0.0, "residual_only", ())
            self._cache[key] = out
            return out
        if self.constant is not None:
            out = VllmTiming(
                ms=n_calls * self.constant,
                n_calls=n_calls,
                tokens_per_generation_call=tokens_per_generation_call,
                simulator_calls_per_generation_call=1.0,
                simulator_method="constant_override",
                source_stop_layers=(self.stop_at_layer,),
            )
            self._cache[key] = out
            return out
        assert self.mod is not None
        # Each producer is an independent DP worker.  Use vllm_dp=1 here; global
        # vLLM DP is represented explicitly by multiple concurrent producer actors.
        one = self.mod.estimate_step_vllm_ms(
            self.rows,
            tp=self.tp,
            vllm_dp=1,
            batch_tokens=int(tokens_per_generation_call),
            stop_at_layer=self.stop_at_layer,
            preferred_total_tokens=self.preferred_total_tokens,
        )
        out = VllmTiming(
            ms=float(one.profiled_vllm_ms) * n_calls,
            n_calls=n_calls,
            tokens_per_generation_call=tokens_per_generation_call,
            simulator_calls_per_generation_call=float(one.calls_per_step),
            simulator_method=str(one.method),
            source_stop_layers=tuple(int(x) for x in one.source_stop_layers),
        )
        self._cache[key] = out
        return out


@dataclass(frozen=True)
class SaeHookCfg:
    name: str
    d_in: int
    d_sae: int
    k: int


@dataclass(frozen=True)
class SaeTiming:
    ms: float
    batch_tokens: int
    mode: str
    warnings: tuple[str, ...] = ()


class SaeTimeOracle:
    def __init__(
        self,
        simulator_path: Path,
        *,
        hooks: Sequence[SaeHookCfg],
        tp: int,
        dp_size: int,
        sae_dp_mode: str,
        dtype: str,
        activation_type: str,
        activation_output_layout: str,
        stats_sync_mode: str,
        normalize_activations: str,
        fsdp_sharding_strategy: str,
        fsdp_forward_prefetch: bool,
        fsdp_backward_prefetch: str,
        ddp_bucket_cap_mb: float,
        backward_hook_order: str,
        compute_csv: Path | None,
        activation_csv: Path | None,
        nccl_csv: Path | None,
        time_column: str,
        compute_device: str | None,
        activation_device: str | None,
        nccl_backend: str,
        nccl_algo: str | None,
        nccl_proto: str | None,
        nccl_p2p_level: str | None,
        constant_step_ms: float | None = None,
    ) -> None:
        self.hooks_cfg = list(hooks)
        self.tp = int(tp)
        self.dp = int(dp_size)
        self.sae_dp_mode = sae_dp_mode
        self.dtype = canonical_dtype(dtype)
        self.constant = constant_step_ms
        self.mod: ModuleType | None = None
        self.profiles: Any = None
        self.simulator: Any = None
        self._cache: dict[int, SaeTiming] = {}

        if self.constant is not None:
            self.mode = self._resolve_mode()
            return

        self.mod = load_module(simulator_path, "_streaming_sae_time_oracle")
        compute_csv = compute_csv or Path(self.mod.DEFAULT_COMPUTE_CSV)
        activation_csv = activation_csv or Path(self.mod.DEFAULT_ACTIVATION_CSV)
        nccl_csv = nccl_csv or Path(self.mod.DEFAULT_NCCL_CSV)
        opts = self.mod.InterpolationOptions()
        self.profiles = self.mod.Profiles.load(
            compute_csv,
            activation_csv,
            nccl_csv,
            time_column=time_column,
            options=opts,
            compute_device=compute_device,
            activation_device=activation_device,
            nccl_backend=nccl_backend,
            nccl_algo=nccl_algo,
            nccl_proto=nccl_proto,
            nccl_p2p_level=nccl_p2p_level,
        )
        self.mode = self._resolve_mode()
        self.simulator = self.mod.build_simulator_from_values(
            self.profiles,
            parallel_mode=self.mode,
            tp=self.tp,
            dp_size=self.dp,
            dtype=self.dtype,
            activation_type=activation_type,
            activation_output_layout=activation_output_layout,
            optimizer_impl="fused",
            stats_sync_mode=stats_sync_mode,
            normalize_activations=normalize_activations,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            fsdp_forward_prefetch=fsdp_forward_prefetch,
            fsdp_backward_prefetch=fsdp_backward_prefetch,
            ddp_bucket_cap_mb=ddp_bucket_cap_mb,
            backward_hook_order=backward_hook_order,
        )

    def _resolve_mode(self) -> str:
        if self.tp > 1:
            if self.dp > 1:
                raise SimError(
                    "uploaded SAE v6 timing model treats TP and DDP/FSDP as separate modes; "
                    "tp>1 with dp>1 is unsupported by that intrinsic oracle"
                )
            return "tp"
        if self.dp > 1:
            if self.sae_dp_mode not in {"ddp", "fsdp"}:
                raise SimError("dp_size>1 requires sae_dp_mode=ddp|fsdp for SAE timing")
            return self.sae_dp_mode
        return "single"

    def estimate(self, batch_tokens: int) -> SaeTiming:
        b = int(batch_tokens)
        if b <= 0:
            raise ValueError("SAE batch must be positive")
        if b in self._cache:
            return self._cache[b]
        if self.constant is not None:
            out = SaeTiming(float(self.constant), b, self.mode, ())
            self._cache[b] = out
            return out
        assert self.mod is not None and self.simulator is not None
        hooks = [self.mod.HookSpec(h.name, h.d_in, h.d_sae, h.k) for h in self.hooks_cfg]
        result = self.simulator.simulate(hooks, batch_size=b)
        out = SaeTiming(float(result.total_ms), b, self.mode, tuple(result.warnings))
        self._cache[b] = out
        return out


# -----------------------------------------------------------------------------
# SHM / actor state
# -----------------------------------------------------------------------------


class SlotState(str, Enum):
    FREE = "FREE"
    WRITING = "WRITING"
    READY = "READY"
    CONSUMING = "CONSUMING"


@dataclass
class ChunkSlot:
    index: int
    state: SlotState = SlotState.FREE
    seq_no: int | None = None
    producer_id: int | None = None
    valid_tokens: int = 0  # per-hook logical tokens
    nbytes: int = 0

    def clear(self) -> None:
        self.state = SlotState.FREE
        self.seq_no = None
        self.producer_id = None
        self.valid_tokens = 0
        self.nbytes = 0


@dataclass
class SharedBufferState:
    num_chunks: int
    target_chunks: int
    slots: list[ChunkSlot] = field(init=False)
    next_claim_seq: int = 0
    done_producers: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.slots = [ChunkSlot(i) for i in range(self.num_chunks)]

    def counts(self) -> dict[str, int]:
        return {
            "free": sum(s.state == SlotState.FREE for s in self.slots),
            "writing": sum(s.state == SlotState.WRITING for s in self.slots),
            "ready": sum(s.state == SlotState.READY for s in self.slots),
            "consuming": sum(s.state == SlotState.CONSUMING for s in self.slots),
        }

    def ready_indices(self) -> list[int]:
        return [s.index for s in self.slots if s.state == SlotState.READY]

    def free_indices(self) -> list[int]:
        return [s.index for s in self.slots if s.state == SlotState.FREE]

    def allocate(self, producer_id: int) -> tuple[ChunkSlot, int] | None:
        if self.next_claim_seq >= self.target_chunks:
            return None
        free = self.free_indices()
        if not free:
            return None
        # Current SharedActivationBuffer chooses the first FREE physical slot.
        idx = free[0]
        seq = self.next_claim_seq
        self.next_claim_seq += 1
        slot = self.slots[idx]
        slot.state = SlotState.WRITING
        slot.seq_no = seq
        slot.producer_id = producer_id
        return slot, seq

    def all_producers_done(self, n_producers: int) -> bool:
        return len(self.done_producers) >= n_producers


class ProducerPhase(str, Enum):
    IDLE = "IDLE"
    PRODUCING = "PRODUCING"
    WAIT_FREE = "WAIT_FREE"
    DONE = "DONE"


@dataclass
class ProducerState:
    producer_id: int
    phase: ProducerPhase = ProducerPhase.IDLE
    active_slot: int | None = None
    residual_tokens: int = 0
    chunks_written: int = 0
    intrinsic_busy_ms: float = 0.0
    d2h_busy_ms: float = 0.0
    wait_free_start_ms: float | None = None
    wait_free_ms: float = 0.0


class ConsumerPhase(str, Enum):
    NEED_BATCH = "NEED_BATCH"
    WAIT_READY = "WAIT_READY"
    FETCHING = "FETCHING"
    TRAINING = "TRAINING"
    DONE = "DONE"


@dataclass
class ConsumerState:
    phase: ConsumerPhase = ConsumerPhase.NEED_BATCH
    serving_tokens: int = 0
    mixing_tokens: int = 0
    consumed_tokens: int = 0
    train_steps: int = 0
    refill_step: int = 0
    active_fetch_slots: list[int] = field(default_factory=list)
    active_fetch_tokens: int = 0
    active_fetch_bytes: int = 0
    wait_ready_start_ms: float | None = None
    wait_ready_ms: float = 0.0
    train_busy_ms: float = 0.0
    h2d_busy_ms: float = 0.0
    upstream_eof: bool = False


class EventKind(str, Enum):
    VLLM_CHUNK_DONE = "VLLM_CHUNK_DONE"
    SAE_FETCH_DONE = "SAE_FETCH_DONE"
    SAE_STEP_DONE = "SAE_STEP_DONE"


EVENT_PRIORITY = {
    EventKind.VLLM_CHUNK_DONE: 0,
    EventKind.SAE_FETCH_DONE: 1,
    EventKind.SAE_STEP_DONE: 2,
}


@dataclass(order=True)
class Event:
    time_ms: float
    priority: int
    serial: int
    kind: EventKind = field(compare=False)
    actor_id: int = field(compare=False, default=0)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class SimConfig:
    # Data / hooks
    training_tokens: int
    train_batch_size_tokens: int
    context_size: int
    training_context_size: int
    store_batch_size_prompts: int
    hooks: list[SaeHookCfg]
    dtype: str

    # vLLM / SAE parallelism
    vllm_tp_size: int
    vllm_dp_size: int
    sae_tp_size: int
    sae_dp_size: int
    sae_dp_mode: str

    # Streaming
    streaming_chunk_size_tokens: int = 4096
    streaming_num_chunks: int = 32
    streaming_prefetch_chunks: int = 2
    streaming_mix_chunks: int = 8
    streaming_mix_fraction: float = 0.5
    streaming_shuffle: bool = True
    streaming_random_chunks: bool = True
    seed: int = 42
    release_policy: str = "after_fetch"

    @property
    def num_hooks(self) -> int:
        return len(self.hooks)

    @property
    def d_in(self) -> int:
        vals = {h.d_in for h in self.hooks}
        if len(vals) != 1:
            raise SimError(
                "SHM v1 stores a single d_model width; this simulator requires equal d_in across hooks"
            )
        return next(iter(vals))

    @property
    def itemsize(self) -> int:
        return DTYPE_BYTES[canonical_dtype(self.dtype)]

    @property
    def target_chunks(self) -> int:
        return math.ceil(self.training_tokens / self.streaming_chunk_size_tokens)

    @property
    def generation_tokens_per_call(self) -> int:
        return self.store_batch_size_prompts * self.training_context_size

    @property
    def mix_capacity_tokens(self) -> int:
        if self.streaming_mix_chunks <= 0:
            return 0
        return max(
            self.train_batch_size_tokens,
            self.streaming_mix_chunks * self.streaming_chunk_size_tokens,
        )

    def activation_bytes(self, logical_tokens_per_hook: int) -> int:
        return (
            int(logical_tokens_per_hook)
            * self.num_hooks
            * self.d_in
            * self.itemsize
        )

    def validate(self) -> None:
        if self.training_tokens <= 0:
            raise SimError("training_tokens must be positive")
        if self.train_batch_size_tokens <= 0:
            raise SimError("train_batch_size_tokens must be positive")
        if self.streaming_chunk_size_tokens <= 0:
            raise SimError("streaming_chunk_size_tokens must be positive")
        if self.streaming_num_chunks <= self.streaming_prefetch_chunks:
            raise SimError("streaming_num_chunks must be > streaming_prefetch_chunks")
        if self.streaming_prefetch_chunks < 1:
            raise SimError("streaming_prefetch_chunks must be >=1")
        if self.streaming_mix_chunks < 0:
            raise SimError("streaming_mix_chunks must be >=0")
        if not 0.0 <= self.streaming_mix_fraction <= 1.0:
            raise SimError("streaming_mix_fraction must be in [0,1]")
        if self.vllm_dp_size < 1 or self.vllm_tp_size < 1:
            raise SimError("vllm tp/dp must be >=1")
        if self.sae_tp_size < 1 or self.sae_dp_size < 1:
            raise SimError("sae tp/dp must be >=1")
        if self.release_policy not in {"repo", "after_fetch"}:
            raise SimError("release_policy must be repo|after_fetch")
        if self.generation_tokens_per_call <= 0:
            raise SimError("store_batch_size_prompts * training_context_size must be >0")


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------


class StreamingShmSimulator:
    EPS = 1e-9

    def __init__(
        self,
        cfg: SimConfig,
        *,
        vllm_oracle: VllmTimeOracle,
        sae_oracle: SaeTimeOracle,
        transfer_oracle: TransferTimeOracle,
        output_dir: Path,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.vllm = vllm_oracle
        self.sae = sae_oracle
        self.transfer = transfer_oracle
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer = SharedBufferState(cfg.streaming_num_chunks, cfg.target_chunks)
        self.producers = [ProducerState(i) for i in range(cfg.vllm_dp_size)]
        self.consumer = ConsumerState()
        self.rng = random.Random(cfg.seed)
        self.events: list[Event] = []
        self.serial = 0
        self.now_ms = 0.0
        self.vllm_log: list[dict[str, Any]] = []
        self.sae_log: list[dict[str, Any]] = []
        self.monitor_log: list[dict[str, Any]] = []
        self.timeline: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        if cfg.sae_dp_size > 1:
            self.warnings.append(
                "Current Git streaming v1 documents sae_dp_size>1 as unsupported; "
                "the simulator can time the SAE step but models one logical consumer."
            )

    # ----- logging ---------------------------------------------------------

    @property
    def now_s(self) -> float:
        return self.now_ms / 1000.0

    def _log_vllm(self, rec: dict[str, Any]) -> None:
        rec = dict(rec)
        rec["elapsed_s"] = self.now_s
        self.vllm_log.append(rec)

    def _log_sae(self, rec: dict[str, Any]) -> None:
        rec = dict(rec)
        rec["elapsed_s"] = self.now_s
        self.sae_log.append(rec)

    def _log_monitor(self) -> None:
        c = self.consumer
        rec = {
            "refill_step": c.refill_step,
            "elapsed_s": self.now_s,
            "ready_count": len(self.buffer.ready_indices()),
            "ready_indices": self.buffer.ready_indices(),
            "counts": self.buffer.counts(),
        }
        self.monitor_log.append(rec)

    def _timeline(self, event: str, **kwargs: Any) -> None:
        self.timeline.append({"t_ms": self.now_ms, "event": event, **kwargs})

    # ----- event queue -----------------------------------------------------

    def _schedule(
        self,
        kind: EventKind,
        delay_ms: float,
        *,
        actor_id: int = 0,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.serial += 1
        heapq.heappush(
            self.events,
            Event(
                self.now_ms + max(0.0, float(delay_ms)),
                EVENT_PRIORITY[kind],
                self.serial,
                kind,
                actor_id,
                dict(payload or {}),
            ),
        )

    # ----- producer --------------------------------------------------------

    def _chunk_valid_tokens(self, seq_no: int) -> int:
        start = seq_no * self.cfg.streaming_chunk_size_tokens
        return max(
            0,
            min(self.cfg.streaming_chunk_size_tokens, self.cfg.training_tokens - start),
        )

    def _producer_chunk_call_count(self, p: ProducerState, valid_tokens: int) -> tuple[int, int, int]:
        """Return (n_calls, residual_before, residual_after), updating no state."""
        residual_before = p.residual_tokens
        need = max(0, valid_tokens - residual_before)
        per_call = self.cfg.generation_tokens_per_call
        n_calls = math.ceil(need / per_call) if need > 0 else 0
        residual_after = residual_before + n_calls * per_call - valid_tokens
        return n_calls, residual_before, residual_after

    def _try_start_producer(self, p: ProducerState) -> bool:
        if p.phase == ProducerPhase.DONE or p.phase == ProducerPhase.PRODUCING:
            return False

        # Quota is global.  Once no more seq numbers can be claimed, this producer
        # signals done; other producers may still have an in-flight WRITING chunk.
        if self.buffer.next_claim_seq >= self.buffer.target_chunks:
            if p.phase == ProducerPhase.WAIT_FREE and p.wait_free_start_ms is not None:
                p.wait_free_ms += self.now_ms - p.wait_free_start_ms
                p.wait_free_start_ms = None
            p.phase = ProducerPhase.DONE
            self.buffer.done_producers.add(p.producer_id)
            self._log_vllm({
                "event": "quota_exhausted",
                "producer_id": p.producer_id,
                "total_chunks": p.chunks_written,
            })
            self._log_vllm({
                "event": "producer_done",
                "producer_id": p.producer_id,
                "total_chunks": p.chunks_written,
            })
            self._timeline("producer_done", producer_id=p.producer_id)
            return True

        alloc = self.buffer.allocate(p.producer_id)
        if alloc is None:
            if p.phase != ProducerPhase.WAIT_FREE:
                p.phase = ProducerPhase.WAIT_FREE
                p.wait_free_start_ms = self.now_ms
                self._timeline("producer_wait_free", producer_id=p.producer_id)
            return False

        if p.phase == ProducerPhase.WAIT_FREE and p.wait_free_start_ms is not None:
            p.wait_free_ms += self.now_ms - p.wait_free_start_ms
            p.wait_free_start_ms = None
        slot, seq_no = alloc
        valid_tokens = self._chunk_valid_tokens(seq_no)
        if valid_tokens <= 0:
            raise SimError(f"claimed seq {seq_no} beyond training token quota")
        slot.valid_tokens = valid_tokens
        slot.nbytes = self.cfg.activation_bytes(valid_tokens)

        n_calls, residual_before, residual_after = self._producer_chunk_call_count(p, valid_tokens)
        vtim = self.vllm.estimate_generation_calls(
            n_calls=n_calls,
            tokens_per_generation_call=self.cfg.generation_tokens_per_call,
        )
        dtim = self.transfer.d2h_ms(slot.nbytes)
        duration = vtim.ms + dtim.ms

        p.residual_tokens = residual_after
        p.phase = ProducerPhase.PRODUCING
        p.active_slot = slot.index
        p.intrinsic_busy_ms += vtim.ms
        p.d2h_busy_ms += dtim.ms

        self._log_vllm({
            "event": "chunk_allocated",
            "chunk_idx": slot.index,
            "seq_no": seq_no,
            "producer_id": p.producer_id,
        })
        self._timeline(
            "chunk_allocated",
            producer_id=p.producer_id,
            chunk_idx=slot.index,
            seq_no=seq_no,
            valid_tokens=valid_tokens,
        )
        self._schedule(
            EventKind.VLLM_CHUNK_DONE,
            duration,
            actor_id=p.producer_id,
            payload={
                "slot_idx": slot.index,
                "seq_no": seq_no,
                "valid_tokens": valid_tokens,
                "activation_bytes": slot.nbytes,
                "n_vllm_steps": n_calls,
                "vllm_compute_ms": vtim.ms,
                "d2h_ms": dtim.ms,
                "d2h_source": dtim.source,
                "residual_before": residual_before,
                "residual_after": residual_after,
                "vllm_method": vtim.simulator_method,
            },
        )
        return True

    def _complete_vllm_chunk(self, ev: Event) -> None:
        p = self.producers[ev.actor_id]
        idx = int(ev.payload["slot_idx"])
        slot = self.buffer.slots[idx]
        if slot.state != SlotState.WRITING or slot.producer_id != p.producer_id:
            raise SimError(f"invalid VLLM completion for slot {idx}: {slot}")
        slot.state = SlotState.READY
        p.phase = ProducerPhase.IDLE
        p.active_slot = None
        p.chunks_written += 1
        self._log_vllm({
            "event": "chunk_written",
            "chunk_idx": idx,
            "seq_no": slot.seq_no,
            "producer_id": p.producer_id,
            "step": p.chunks_written,
            "valid_tokens": slot.valid_tokens,
            "total_tokens": int(slot.seq_no or 0) * self.cfg.streaming_chunk_size_tokens + slot.valid_tokens,
            # Current Git times get_streaming_activations() as "inference"; that
            # routine includes activation .cpu(), so the modeled D2H belongs here.
            # SHM memcpy/write itself is intentionally zero-time in this simulator.
            "inference_time_s": (ev.payload["vllm_compute_ms"] + ev.payload["d2h_ms"]) / 1000.0,
            "write_time_s": 0.0,
            "buffer_state": self.buffer.counts(),
            "activation_bytes": ev.payload["activation_bytes"],
            "n_vllm_steps": ev.payload["n_vllm_steps"],
            "d2h_time_s": ev.payload["d2h_ms"] / 1000.0,
            "d2h_source": ev.payload["d2h_source"],
            "residual_tokens_before": ev.payload["residual_before"],
            "residual_tokens_after": ev.payload["residual_after"],
            "vllm_time_method": ev.payload["vllm_method"],
        })
        self._timeline(
            "chunk_ready",
            producer_id=p.producer_id,
            chunk_idx=idx,
            seq_no=slot.seq_no,
        )

    # ----- consumer / provider --------------------------------------------

    def _select_ready(self) -> list[int]:
        ready = self.buffer.ready_indices()
        if not ready:
            return []
        if self.cfg.streaming_random_chunks:
            self.rng.shuffle(ready)
        return ready[: self.cfg.streaming_prefetch_chunks]

    def _release_fetch_slots(self, indices: Sequence[int]) -> None:
        for idx in indices:
            slot = self.buffer.slots[idx]
            if slot.state != SlotState.CONSUMING:
                raise SimError(f"cannot release non-CONSUMING slot {idx}: {slot.state}")
            slot.clear()
        self._timeline("chunks_released", chunk_indices=list(indices))

    def _mix_new_tokens(self, new_tokens: int) -> None:
        c = self.consumer
        B = self.cfg.train_batch_size_tokens
        if self.cfg.streaming_mix_chunks <= 0:
            # No cross-chunk mixing window: append directly to serving pool.
            c.serving_tokens += new_tokens
            return

        combined = c.serving_tokens + c.mixing_tokens + new_tokens
        capacity = self.cfg.mix_capacity_tokens
        if combined < capacity and not c.upstream_eof:
            c.serving_tokens = 0
            c.mixing_tokens = combined
            return

        if c.upstream_eof:
            # Drain: all remaining local data becomes servable.
            c.serving_tokens = combined
            c.mixing_tokens = 0
            return

        # Shuffle itself is zero-time.  Only counts matter for scheduling.
        keep_for_mixing = int(capacity * self.cfg.streaming_mix_fraction)
        num_to_serve = max(0, combined - keep_for_mixing)
        num_serving_batches = max(1, num_to_serve // B)
        serving_cutoff = min(combined, num_serving_batches * B)
        c.serving_tokens = serving_cutoff
        c.mixing_tokens = combined - serving_cutoff

    def _make_all_local_servable(self) -> None:
        c = self.consumer
        c.upstream_eof = True
        total = c.serving_tokens + c.mixing_tokens
        c.serving_tokens = total
        c.mixing_tokens = 0

    def _try_acquire(self, *, resume_wait: bool = False) -> bool:
        c = self.consumer
        # A blocked acquire_up_to() is one refill call.  Waking on READY must resume
        # that same acquire rather than emitting another refill_start/monitor record.
        if not resume_wait:
            c.refill_step += 1
            self._log_sae({
                "event": "refill_start",
                "pool_tokens_remaining": c.serving_tokens,
                "consumed_tokens": c.consumed_tokens,
                "prefetch_chunks": self.cfg.streaming_prefetch_chunks,
                "mixing_tokens": c.mixing_tokens,
            })
            self._log_monitor()

        selected = self._select_ready()
        if not selected:
            if self.buffer.all_producers_done(self.cfg.vllm_dp_size):
                wait_ms = 0.0
                if c.wait_ready_start_ms is not None:
                    wait_ms = self.now_ms - c.wait_ready_start_ms
                    c.wait_ready_ms += wait_ms
                    c.wait_ready_start_ms = None
                self._make_all_local_servable()
                self._log_sae({
                    "event": "refill_exhausted",
                    "pool_tokens_after": c.serving_tokens,
                    "mixing_tokens_after": c.mixing_tokens,
                    "wait_time_s": wait_ms / 1000.0,
                })
                return True
            # Real acquire_up_to sleeps/backoffs; EDS simply waits for next producer event.
            c.phase = ConsumerPhase.WAIT_READY
            if c.wait_ready_start_ms is None:
                c.wait_ready_start_ms = self.now_ms
            self._timeline("consumer_wait_ready")
            return False

        wait_ms = 0.0
        if c.wait_ready_start_ms is not None:
            wait_ms = self.now_ms - c.wait_ready_start_ms
            c.wait_ready_ms += wait_ms
            c.wait_ready_start_ms = None

        tokens = 0
        nbytes = 0
        seq_nos: list[int] = []
        for idx in selected:
            slot = self.buffer.slots[idx]
            if slot.state != SlotState.READY:
                raise SimError(f"selected slot {idx} not READY")
            slot.state = SlotState.CONSUMING
            tokens += slot.valid_tokens
            nbytes += slot.nbytes
            seq_nos.append(int(slot.seq_no if slot.seq_no is not None else -1))

        c.active_fetch_slots = list(selected)
        c.active_fetch_tokens = tokens
        c.active_fetch_bytes = nbytes

        # Current repo releases after SHM->private-CPU copy and before .to(device).
        # The requested conservative policy holds slots through the modeled H2D.
        if self.cfg.release_policy == "repo":
            self._release_fetch_slots(selected)

        htim = self.transfer.h2d_ms(nbytes)
        c.h2d_busy_ms += htim.ms
        c.phase = ConsumerPhase.FETCHING
        self._log_sae({
            "event": "refill_acquired",
            "chunk_indices": list(selected),
            "seq_nos": seq_nos,
            "n_chunks": len(selected),
            "wait_time_s": wait_ms / 1000.0,
            "buffer_state": self.buffer.counts(),
            "activation_bytes": nbytes,
            "h2d_time_s": htim.ms / 1000.0,
            "h2d_source": htim.source,
        })
        self._timeline(
            "refill_acquired",
            chunk_indices=list(selected),
            seq_nos=seq_nos,
            bytes=nbytes,
        )
        self._schedule(
            EventKind.SAE_FETCH_DONE,
            htim.ms,
            payload={
                "chunk_indices": list(selected),
                "new_tokens": tokens,
                "activation_bytes": nbytes,
                "wait_ms": wait_ms,
                "h2d_ms": htim.ms,
                "h2d_source": htim.source,
            },
        )
        return True

    def _complete_fetch(self, ev: Event) -> None:
        c = self.consumer
        indices = list(ev.payload["chunk_indices"])
        if self.cfg.release_policy == "after_fetch":
            self._release_fetch_slots(indices)
        new_tokens = int(ev.payload["new_tokens"])
        c.active_fetch_slots = []
        c.active_fetch_tokens = 0
        c.active_fetch_bytes = 0
        c.phase = ConsumerPhase.NEED_BATCH
        self._mix_new_tokens(new_tokens)
        self._log_sae({
            "event": "refill_complete",
            "new_tokens": new_tokens,
            "pool_tokens_after": c.serving_tokens,
            "mixing_tokens_after": c.mixing_tokens,
            "wait_time_s": ev.payload["wait_ms"] / 1000.0,
            "transfer_time_s": ev.payload["h2d_ms"] / 1000.0,
            "activation_bytes": ev.payload["activation_bytes"],
            "release_policy": self.cfg.release_policy,
        })
        self._timeline(
            "refill_complete",
            new_tokens=new_tokens,
            serving_tokens=c.serving_tokens,
            mixing_tokens=c.mixing_tokens,
        )

    def _try_start_sae_step(self) -> bool:
        c = self.consumer
        if c.phase != ConsumerPhase.NEED_BATCH:
            return False
        remaining_needed = self.cfg.training_tokens - c.consumed_tokens
        if remaining_needed <= 0:
            c.phase = ConsumerPhase.DONE
            return True
        if c.serving_tokens <= 0:
            return False

        B = min(self.cfg.train_batch_size_tokens, remaining_needed, c.serving_tokens)
        # During normal streaming, do not expose an under-full batch unless upstream
        # is exhausted/draining.  This mirrors provider behavior at the tail.
        if B < self.cfg.train_batch_size_tokens and not c.upstream_eof:
            return False

        c.serving_tokens -= B
        c.consumed_tokens += B
        c.train_steps += 1
        stim = self.sae.estimate(B)
        for warning in stim.warnings:
            if warning not in self.warnings:
                self.warnings.append(warning)
        c.train_busy_ms += stim.ms
        self._log_sae({
            "event": "consume",
            "step": c.train_steps,
            "batch_tokens": B,
            "pool_tokens_remaining": c.serving_tokens,
            "mixing_tokens_remaining": c.mixing_tokens,
            "cumulative_tokens": c.consumed_tokens,
            "sae_step_time_s": stim.ms / 1000.0,
            "sae_mode": stim.mode,
        })
        self._timeline(
            "sae_step_start",
            step=c.train_steps,
            batch_tokens=B,
            serving_tokens_after_take=c.serving_tokens,
        )
        c.phase = ConsumerPhase.TRAINING
        self._schedule(
            EventKind.SAE_STEP_DONE,
            stim.ms,
            payload={"step": c.train_steps, "batch_tokens": B, "sae_ms": stim.ms},
        )
        return True

    def _complete_sae_step(self, ev: Event) -> None:
        c = self.consumer
        c.phase = ConsumerPhase.NEED_BATCH
        self._timeline(
            "sae_step_done",
            step=ev.payload["step"],
            batch_tokens=ev.payload["batch_tokens"],
        )

    # ----- zero-time drive -------------------------------------------------

    def _drive_consumer_once(self) -> bool:
        c = self.consumer
        if c.phase in {ConsumerPhase.TRAINING, ConsumerPhase.FETCHING, ConsumerPhase.DONE}:
            return False

        # Training quota is authoritative.  Do not keep issuing zero-time EOF
        # refills after the requested token count has already been consumed.
        if c.consumed_tokens >= self.cfg.training_tokens:
            c.phase = ConsumerPhase.DONE
            return True

        # Wake a waiting consumer if data or EOF became visible.
        if c.phase == ConsumerPhase.WAIT_READY:
            if self.buffer.ready_indices() or self.buffer.all_producers_done(self.cfg.vllm_dp_size):
                c.phase = ConsumerPhase.NEED_BATCH
                return self._try_acquire(resume_wait=True)
            return False

        # Training consumes all currently servable full batches before another refill.
        if c.serving_tokens >= self.cfg.train_batch_size_tokens:
            return self._try_start_sae_step()

        # At EOF, a final partial batch may be served.
        if c.upstream_eof and c.serving_tokens > 0:
            return self._try_start_sae_step()

        # Need more data.  Note: acquire_up_to() itself accepts fewer than prefetch
        # chunks; repeated refills are what build a mix window.
        return self._try_acquire()

    def _drive_until_quiescent(self) -> None:
        # A generous guard catches accidental zero-time loops.
        for _ in range(1_000_000):
            changed = False
            for p in self.producers:
                if p.phase in {ProducerPhase.IDLE, ProducerPhase.WAIT_FREE}:
                    if self._try_start_producer(p):
                        changed = True
            if self._drive_consumer_once():
                changed = True
            if not changed:
                return
        raise SimError("zero-time drive did not quiesce")

    def _apply_event(self, ev: Event) -> None:
        if ev.kind == EventKind.VLLM_CHUNK_DONE:
            self._complete_vllm_chunk(ev)
        elif ev.kind == EventKind.SAE_FETCH_DONE:
            self._complete_fetch(ev)
        elif ev.kind == EventKind.SAE_STEP_DONE:
            self._complete_sae_step(ev)
        else:
            raise AssertionError(ev.kind)

    def run(self) -> dict[str, Any]:
        # t=0: vLLM starts its first chunk; SAE is also allowed to enter acquire and
        # wait.  Model-load/startup skew is intentionally outside this simulator.
        self._drive_until_quiescent()

        while self.consumer.phase != ConsumerPhase.DONE:
            if not self.events:
                # One final drive may turn producer-done + local tail into completion.
                self._drive_until_quiescent()
                if self.consumer.phase == ConsumerPhase.DONE:
                    break
                raise SimError(
                    "event queue became empty before training completed; "
                    f"consumer={self.consumer} counts={self.buffer.counts()}"
                )
            next_t = self.events[0].time_ms
            self.now_ms = next_t
            batch: list[Event] = []
            while self.events and abs(self.events[0].time_ms - next_t) <= self.EPS:
                batch.append(heapq.heappop(self.events))
            # Apply all completions at this timestamp before issuing zero-time actions.
            for ev in sorted(batch, key=lambda x: (x.priority, x.serial)):
                self._apply_event(ev)
            self._drive_until_quiescent()

        # If consumer finishes before all producers, no new SAE work is needed.  The
        # real standalone run normally has matching target quota, so producers should
        # already be done or have only harmless in-flight work.  We stop at training end.
        self._write_logs()
        return self._summary()

    def _write_jsonl(self, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                json.dump(dict(row), f)
                f.write("\n")

    def _write_logs(self) -> None:
        self._write_jsonl(self.out / "shm_log_vllm.jsonl", self.vllm_log)
        self._write_jsonl(self.out / "shm_log_sae.jsonl", self.sae_log)
        self._write_jsonl(self.out / "buffer_monitor.jsonl", self.monitor_log)
        self._write_jsonl(self.out / "sim_timeline.jsonl", self.timeline)
        (self.out / "sim_config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2), encoding="utf-8"
        )
        (self.out / "sim_summary.json").write_text(
            json.dumps(self._summary(), indent=2), encoding="utf-8"
        )

    def _summary(self) -> dict[str, Any]:
        p_busy = sum(p.intrinsic_busy_ms + p.d2h_busy_ms for p in self.producers)
        p_wait = sum(p.wait_free_ms for p in self.producers)
        counts = self.buffer.counts()
        return {
            "total_ms": self.now_ms,
            "total_s": self.now_s,
            "training_tokens": self.cfg.training_tokens,
            "target_chunks": self.cfg.target_chunks,
            "consumer_steps": self.consumer.train_steps,
            "consumer_tokens": self.consumer.consumed_tokens,
            "consumer_train_busy_ms": self.consumer.train_busy_ms,
            "consumer_h2d_busy_ms": self.consumer.h2d_busy_ms,
            "consumer_wait_ready_ms": self.consumer.wait_ready_ms,
            "producer_intrinsic_plus_d2h_work_ms_sum": p_busy,
            "producer_wait_free_ms_sum": p_wait,
            "producer_chunks_written": [p.chunks_written for p in self.producers],
            "producer_residual_tokens": [p.residual_tokens for p in self.producers],
            "final_buffer_counts": counts,
            "final_serving_tokens": self.consumer.serving_tokens,
            "final_mixing_tokens": self.consumer.mixing_tokens,
            "release_policy": self.cfg.release_policy,
            "streaming": {
                "num_chunks": self.cfg.streaming_num_chunks,
                "prefetch_chunks": self.cfg.streaming_prefetch_chunks,
                "mix_chunks": self.cfg.streaming_mix_chunks,
                "mix_fraction": self.cfg.streaming_mix_fraction,
                "shuffle": self.cfg.streaming_shuffle,
                "random_chunks": self.cfg.streaming_random_chunks,
            },
            "modeled_costs": ["vllm intrinsic", "D2H", "H2D", "SAE intrinsic"],
            "ignored_costs": [
                "dataset fetch",
                "SHM write/read memcpy",
                "CPU concat/reinterleave",
                "shuffle/mixing execution time",
                "locks/backoff/Python overhead",
                "CUDA contention between independent GPU sets",
                "model/SAE initialization and startup skew",
            ],
            "warnings": self.warnings,
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_hook_spec(text: str) -> SaeHookCfg:
    # name:d_in:d_sae:k ; name may contain dots but not ':'
    parts = text.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--hook-spec must be NAME:D_IN:D_SAE:K, e.g. blocks.21.hook_resid_post:4096:32768:128"
        )
    name, d, f, k = parts
    try:
        return SaeHookCfg(name, int(d), int(f), int(k))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def infer_stop_layer(hooks: Sequence[SaeHookCfg], explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    import re
    layers: list[int] = []
    for h in hooks:
        m = re.search(r"\bblocks\.(\d+)\.", h.name)
        if m:
            layers.append(int(m.group(1)))
    if not layers:
        raise SimError("cannot infer vLLM stop layer; pass --stop-at-layer")
    return max(layers) + 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discrete-event simulator for SAELens /dev/shm streaming scheduling"
    )

    # Main training config
    p.add_argument("--training-tokens", type=int, required=True)
    p.add_argument("--train-batch-size-tokens", type=int, required=True)
    p.add_argument("--context-size", type=int, required=True)
    p.add_argument(
        "--training-context-size",
        type=int,
        default=None,
        help="tokens kept per prompt after seqpos_slice; default=context-size",
    )
    p.add_argument("--store-batch-size-prompts", type=int, required=True)
    p.add_argument(
        "--hook-spec",
        action="append",
        type=parse_hook_spec,
        default=[],
        help="repeatable NAME:D_IN:D_SAE:K",
    )
    p.add_argument("--d-in", type=int, default=4096)
    p.add_argument("--d-sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--hook-name", action="append", default=[])
    p.add_argument("--dtype", default="bfloat16")

    # Parallel config
    p.add_argument("--vllm-tp-size", type=int, default=1)
    p.add_argument("--vllm-dp-size", type=int, default=1)
    p.add_argument("--sae-tp-size", type=int, default=1)
    p.add_argument("--sae-dp-size", type=int, default=1)
    p.add_argument("--sae-dp-mode", choices=["manual", "ddp", "fsdp"], default="manual")

    # Streaming config -- defaults match current LanguageModelSAERunnerConfig.
    p.add_argument("--streaming-chunk-size-tokens", type=int, default=4096)
    p.add_argument("--streaming-num-chunks", type=int, default=32)
    p.add_argument("--streaming-prefetch-chunks", type=int, default=2)
    p.add_argument("--streaming-mix-chunks", type=int, default=8)
    p.add_argument("--streaming-mix-fraction", type=float, default=0.5)
    p.add_argument("--streaming-shuffle", type=parse_bool, default=True)
    p.add_argument("--streaming-random-chunks", type=parse_bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--release-policy",
        choices=["repo", "after_fetch"],
        default="after_fetch",
        help=(
            "after_fetch (default) follows the requested planner abstraction: keep "
            "slots CONSUMING through modeled H2D; repo releases after SHM->CPU read "
            "and before H2D, matching current provider ordering"
        ),
    )

    # Existing vLLM simulator
    p.add_argument("--vllm-simulator", type=Path, default=DEFAULT_VLLM_SIM)
    p.add_argument("--vllm-profile", type=Path, default=None)
    p.add_argument("--vllm-profile-total-tokens", type=int, default=None)
    p.add_argument("--stop-at-layer", type=int, default=None)

    # Existing SAE simulator
    p.add_argument("--sae-simulator", type=Path, default=DEFAULT_SAE_SIM)
    p.add_argument("--sae-compute-profile", type=Path, default=None)
    p.add_argument("--sae-activation-profile", type=Path, default=None)
    p.add_argument("--sae-nccl-profile", type=Path, default=None)
    p.add_argument("--sae-time-column", default="median_ms")
    p.add_argument("--sae-compute-device", default=None)
    p.add_argument("--sae-activation-device", default=None)
    p.add_argument("--sae-nccl-backend", default="nccl")
    p.add_argument("--sae-nccl-algo", default=None)
    p.add_argument("--sae-nccl-proto", default=None)
    p.add_argument("--sae-nccl-p2p-level", default=None)
    p.add_argument("--activation-type", default="topk")
    p.add_argument("--activation-output-layout", default="dense")
    p.add_argument("--stats-sync-mode", default="immediate")
    p.add_argument("--normalize-activations", default="none")
    p.add_argument("--fsdp-sharding-strategy", default="shard_grad_op")
    p.add_argument("--fsdp-forward-prefetch", type=parse_bool, default=True)
    p.add_argument("--fsdp-backward-prefetch", default="backward_post")
    p.add_argument("--ddp-bucket-cap-mb", type=float, default=25.0)
    p.add_argument("--backward-hook-order", choices=["forward", "reverse"], default="reverse")

    # Transfer profile
    p.add_argument("--transfer-profile", type=Path, default=DEFAULT_TRANSFER_PROFILE)
    p.add_argument("--transfer-time-column", default="median_ms")

    # Diagnostic overrides; core path uses the existing simulators.
    p.add_argument(
        "--constant-vllm-ms-per-generation-call",
        type=float,
        default=None,
        help="debug/smoke-test override; bypass simulate_vllm_time.py",
    )
    p.add_argument(
        "--constant-sae-step-ms",
        type=float,
        default=None,
        help="debug/smoke-test override; bypass SAE step simulator",
    )

    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hook_spec:
        hooks = list(args.hook_spec)
    else:
        names = args.hook_name or ["h1"]
        hooks = [SaeHookCfg(n, args.d_in, args.d_sae, args.k) for n in names]

    training_context_size = args.training_context_size or args.context_size
    cfg = SimConfig(
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.train_batch_size_tokens,
        context_size=args.context_size,
        training_context_size=training_context_size,
        store_batch_size_prompts=args.store_batch_size_prompts,
        hooks=hooks,
        dtype=canonical_dtype(args.dtype),
        vllm_tp_size=args.vllm_tp_size,
        vllm_dp_size=args.vllm_dp_size,
        sae_tp_size=args.sae_tp_size,
        sae_dp_size=args.sae_dp_size,
        sae_dp_mode=args.sae_dp_mode,
        streaming_chunk_size_tokens=args.streaming_chunk_size_tokens,
        streaming_num_chunks=args.streaming_num_chunks,
        streaming_prefetch_chunks=args.streaming_prefetch_chunks,
        streaming_mix_chunks=args.streaming_mix_chunks,
        streaming_mix_fraction=args.streaming_mix_fraction,
        streaming_shuffle=args.streaming_shuffle,
        streaming_random_chunks=args.streaming_random_chunks,
        seed=args.seed,
        release_policy=args.release_policy,
    )
    cfg.validate()

    stop_layer = infer_stop_layer(hooks, args.stop_at_layer)
    transfer = TransferTimeOracle(args.transfer_profile, time_column=args.transfer_time_column)
    vllm = VllmTimeOracle(
        args.vllm_simulator,
        args.vllm_profile,
        tp=cfg.vllm_tp_size,
        stop_at_layer=stop_layer,
        preferred_total_tokens=args.vllm_profile_total_tokens,
        constant_ms_per_generation_call=args.constant_vllm_ms_per_generation_call,
    )
    sae = SaeTimeOracle(
        args.sae_simulator,
        hooks=hooks,
        tp=cfg.sae_tp_size,
        dp_size=cfg.sae_dp_size,
        sae_dp_mode=cfg.sae_dp_mode,
        dtype=cfg.dtype,
        activation_type=args.activation_type,
        activation_output_layout=args.activation_output_layout,
        stats_sync_mode=args.stats_sync_mode,
        normalize_activations=args.normalize_activations,
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
        fsdp_forward_prefetch=args.fsdp_forward_prefetch,
        fsdp_backward_prefetch=args.fsdp_backward_prefetch,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        backward_hook_order=args.backward_hook_order,
        compute_csv=args.sae_compute_profile,
        activation_csv=args.sae_activation_profile,
        nccl_csv=args.sae_nccl_profile,
        time_column=args.sae_time_column,
        compute_device=args.sae_compute_device,
        activation_device=args.sae_activation_device,
        nccl_backend=args.sae_nccl_backend,
        nccl_algo=args.sae_nccl_algo,
        nccl_proto=args.sae_nccl_proto,
        nccl_p2p_level=args.sae_nccl_p2p_level,
        constant_step_ms=args.constant_sae_step_ms,
    )

    sim = StreamingShmSimulator(
        cfg,
        vllm_oracle=vllm,
        sae_oracle=sae,
        transfer_oracle=transfer,
        output_dir=args.output_dir,
    )
    summary = sim.run()
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {args.output_dir / 'shm_log_vllm.jsonl'}")
    print(f"       {args.output_dir / 'shm_log_sae.jsonl'}")
    print(f"       {args.output_dir / 'buffer_monitor.jsonl'}")
    print(f"       {args.output_dir / 'sim_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
