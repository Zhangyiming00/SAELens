"""Per-phase timing probe for the ordinary (non-streaming) shard-routing path.

The probe is attached to an `ActivationsStore` as `_routing_probe`. When it is
None every instrumented section costs one `is None` check, so production runs are
unaffected. When attached, each section records both:

- `wall_s`: CPU wall time. This is the meaningful number for CPU-blocking
  sections such as `dist.barrier`, where it measures rank skew.
- `gpu_s`: the span between two CUDA events on the current stream. This is the
  meaningful number for data movement (slice copies, cat, P2P, broadcast),
  because NCCL work is enqueued on the process group's own stream and only
  becomes visible to the current stream when the work is waited on.

Both are reported rather than one derived number: for a barrier `gpu_s` is
near zero, and for an async collective `wall_s` is only the launch cost.

Times accumulate until `flush`, which resolves the CUDA events, returns the
totals for the cycle and resets. Sections must not be nested; nesting would
double-count the inner section in the outer total.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class PhaseTotals:
    """Accumulated time for one routing phase within one routing cycle."""

    calls: int = 0
    wall_s: float = 0.0
    gpu_s: float = 0.0


class RoutingPhaseProbe:
    """Accumulate per-phase wall and CUDA-event times over one routing cycle."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock if clock is not None else time.perf_counter
        self._device = torch.device(device) if device is not None else None
        self._use_events = (
            self._device is not None
            and self._device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._totals: dict[str, PhaseTotals] = {}
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @property
    def cuda_events_enabled(self) -> bool:
        return self._use_events

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start: torch.cuda.Event | None = None
        if self._use_events:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = self._clock()
        try:
            yield
        finally:
            wall_s = self._clock() - t0
            totals = self._totals.setdefault(name, PhaseTotals())
            totals.calls += 1
            totals.wall_s += wall_s
            if self._use_events:
                assert start is not None
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._pending.append((name, start, end))

    def flush(self) -> dict[str, PhaseTotals]:
        """Resolve pending CUDA events, then return and reset the cycle totals.

        Synchronizes the device when CUDA events are in use, so call this after
        the measured interval has closed.
        """
        if self._pending:
            torch.cuda.synchronize(self._device)
            for name, start, end in self._pending:
                self._totals[name].gpu_s += start.elapsed_time(end) / 1000.0
            self._pending = []
        totals, self._totals = self._totals, {}
        return totals


@contextmanager
def routing_phase(probe: RoutingPhaseProbe | None, name: str) -> Iterator[None]:
    """Time a routing section when a probe is attached, otherwise do nothing."""
    if probe is None:
        yield
        return
    with probe.phase(name):
        yield
