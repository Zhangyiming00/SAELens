"""Fixed-window step timing for end-to-end throughput profiling.

A window covers a contiguous run of steps. The device is synchronized only at
the two window boundaries, so CPU/GPU overlap and cross-step pipelining inside
the window are preserved and the measured interval is a true wall-clock cost
for those steps.

``window_time_s`` is the single interval between the two boundary syncs, so
everything happening between the first and last step of a window is included
(data fetch, per-step logging, checkpoint I/O). The drain performed by the
closing sync belongs to the window and is counted; the drain performed by the
opening sync excludes work queued before the window and is not.

Windows are contiguous: window ``i`` (0-based) covers steps
``[start_step + i * window_steps, start_step + i * window_steps + window_steps - 1]``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from sae_lens import logger


def _make_sync_fn(device: str | torch.device) -> Callable[[], None]:
    if torch.device(device).type != "cuda":
        return lambda: None
    return lambda: torch.cuda.synchronize(device)


class StepWindowProfiler:
    """Time contiguous windows of steps, syncing only at window boundaries.

    Call `on_step_start` before a step's work begins and `on_step_end` after it
    completes, every step. Both are no-ops outside a window, and the profiler
    goes permanently inert once the last window closes. `close` flushes a
    partially finished window.
    """

    def __init__(
        self,
        *,
        start_step: int,
        window_steps: int,
        window_count: int,
        output_path: Path,
        role: str,
        step_unit: str,
        rank: int = 0,
        device: str | torch.device = "cpu",
        context: dict[str, Any] | None = None,
        sync_fn: Callable[[], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if start_step < 1:
            raise ValueError("start_step must be >= 1")
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if window_count < 1:
            raise ValueError("window_count must be >= 1")

        self.start_step = start_step
        self.window_steps = window_steps
        self.window_count = window_count
        self.role = role
        self.step_unit = step_unit
        self.rank = rank
        self.context = dict(context or {})
        self._sync = sync_fn if sync_fn is not None else _make_sync_fn(device)
        self._clock = clock if clock is not None else time.perf_counter

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("")

        # Anchor the injected monotonic clock to wall time so windows recorded
        # by separate processes (vLLM producer vs SAE consumer) can be aligned.
        self._clock0 = self._clock()
        self._epoch0 = time.time()

        self._window_idx: int | None = 0
        self._open = False
        self._open_start_step = 0
        self._open_end_step = 0
        self._t0 = 0.0
        self._sync_start_s = 0.0
        self._steps_done = 0
        self._last_step = 0
        self._samples = 0
        self._components: dict[str, float] = {}

    @classmethod
    def maybe_create(
        cls,
        *,
        start_step: int,
        window_steps: int,
        window_count: int,
        output_dir: str | Path | None,
        role: str,
        step_unit: str,
        rank: int = 0,
        device: str | torch.device = "cpu",
        context: dict[str, Any] | None = None,
    ) -> StepWindowProfiler | None:
        """Build a profiler, or return None when window profiling is disabled."""
        if start_step < 1 or window_steps < 1 or window_count < 1:
            return None
        if output_dir is None:
            return None
        return cls(
            start_step=start_step,
            window_steps=window_steps,
            window_count=window_count,
            output_path=Path(output_dir)
            / f"step_window_profile_{role}_rank{rank}.jsonl",
            role=role,
            step_unit=step_unit,
            rank=rank,
            device=device,
            context=context,
        )

    @property
    def done(self) -> bool:
        return self._window_idx is None and not self._open

    def _next_window_start(self) -> int | None:
        if self._window_idx is None:
            return None
        return self.start_step + self._window_idx * self.window_steps

    def on_step_start(self, step: int) -> None:
        if self._open or self._window_idx is None:
            return
        # Skip past any window whose start step was never reached (the caller
        # broke out of the loop or step numbering jumped).
        while self._window_idx is not None:
            w_start = self.start_step + self._window_idx * self.window_steps
            if w_start >= step:
                break
            self._advance_window()
        w_start = self._next_window_start()
        if w_start is None or w_start != step:
            return

        t_sync = self._clock()
        self._sync()
        self._sync_start_s = self._clock() - t_sync
        self._t0 = self._clock()
        self._open = True
        self._open_start_step = step
        self._open_end_step = step + self.window_steps - 1
        self._steps_done = 0
        self._last_step = step
        self._samples = 0
        self._components = {}

    def on_step_end(
        self,
        step: int,
        *,
        samples: int = 0,
        components: dict[str, float] | None = None,
    ) -> None:
        if not self._open:
            return
        self._steps_done += 1
        self._last_step = step
        self._samples += samples
        for key, value in (components or {}).items():
            self._components[key] = self._components.get(key, 0.0) + float(value)
        if step >= self._open_end_step:
            self._close_window(complete=True)

    def close(self) -> None:
        """Flush a window that is still open (e.g. training ended early)."""
        if self._open:
            self._close_window(complete=False)

    def _advance_window(self) -> None:
        assert self._window_idx is not None
        self._window_idx += 1
        if self._window_idx >= self.window_count:
            self._window_idx = None

    def _close_window(self, *, complete: bool) -> None:
        t_sync = self._clock()
        self._sync()
        sync_end_s = self._clock() - t_sync
        t1 = self._clock()
        window_time_s = t1 - self._t0
        window_idx = self._window_idx
        assert window_idx is not None
        record = {
            "role": self.role,
            "step_unit": self.step_unit,
            "rank": self.rank,
            "window": window_idx + 1,
            "start_step": self._open_start_step,
            "end_step": self._last_step,
            "planned_end_step": self._open_end_step,
            "steps": self._steps_done,
            "planned_steps": self.window_steps,
            "complete": complete,
            "window_time_s": window_time_s,
            "per_step_s": window_time_s / self._steps_done
            if self._steps_done > 0
            else None,
            "samples": self._samples,
            "samples_per_s": self._samples / window_time_s
            if window_time_s > 0
            else None,
            "sync_open_s": self._sync_start_s,
            "sync_close_s": sync_end_s,
            "t_start_unix": self._epoch0 + (self._t0 - self._clock0),
            "t_end_unix": self._epoch0 + (t1 - self._clock0),
            # CPU-side per-step timers summed over the window. These are launch
            # times (no sync inside the window), so treat them as a breakdown of
            # proportions, not as absolute costs.
            "components": self._components,
            "context": self.context,
        }
        with open(self.output_path, "a") as f:
            json.dump(record, f)
            f.write("\n")
        per_step = record["per_step_s"]
        per_step_str = "n/a" if per_step is None else f"{per_step:.6f}s"
        incomplete_str = "" if complete else ", INCOMPLETE"
        logger.info(
            "[window-profile %s rank%d] window %d steps %d-%d (%d/%d%s) "
            "%.4fs per_%s=%s",
            self.role,
            self.rank,
            window_idx + 1,
            self._open_start_step,
            self._last_step,
            self._steps_done,
            self.window_steps,
            incomplete_str,
            window_time_s,
            self.step_unit,
            per_step_str,
        )
        self._open = False
        self._advance_window()
