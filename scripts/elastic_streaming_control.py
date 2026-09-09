#!/usr/bin/env python3
"""Inspect, manually switch, or auto-balance a live elastic streaming run."""

from __future__ import annotations

import argparse
import json
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sae_lens.elastic_streaming import (
    ElasticControlState,
    ElasticStreamingController,
    ElasticStreamingLayout,
)


@dataclass(frozen=True)
class BufferSample:
    timestamp: float
    next_claim_seq: int
    occupied_chunks: int
    counts: dict[str, int]
    target_chunks: int = 0

    @property
    def fill_ratio(self) -> float:
        total = sum(self.counts.values())
        return self.occupied_chunks / total if total else 0.0

    @property
    def production_complete(self) -> bool:
        return self.target_chunks > 0 and self.next_claim_seq >= self.target_chunks


@dataclass(frozen=True)
class ThroughputRates:
    window_seconds: float
    fill_ratio: float
    vllm_tokens_per_s: float
    sae_tokens_per_s: float
    net_tokens_per_s: float


class SharedBufferMonitor:
    """Read existing SHM coordination data without taking the buffer lock."""

    def __init__(self, state: ElasticControlState, base_dir: Path) -> None:
        if not state.buffer_name or state.buffer_num_chunks < 1:
            raise RuntimeError(
                "control state has no streaming buffer metadata; restart the run "
                "with the current elastic streaming implementation"
            )
        self.chunk_size_tokens = state.buffer_chunk_size_tokens
        if self.chunk_size_tokens < 1:
            raise RuntimeError("control state has an invalid buffer chunk size")
        self._state = np.memmap(
            base_dir / f"{state.buffer_name}_state.bin",
            dtype=np.int8,
            mode="r",
            shape=(state.buffer_num_chunks,),
        )
        self._header = np.memmap(
            base_dir / f"{state.buffer_name}_header.bin",
            dtype=np.int32,
            mode="r",
            shape=(8,),
        )

    def sample(self, *, timestamp: float | None = None) -> BufferSample:
        # Monitoring is deliberately approximate and lock-free so it cannot
        # delay producers or consumers. Stable-sample filtering rejects the
        # rare transition snapshot seen between two memmap flushes.
        raw_counts = np.bincount(np.asarray(self._state), minlength=4)
        counts = {
            "free": int(raw_counts[0]),
            "writing": int(raw_counts[1]),
            "ready": int(raw_counts[2]),
            "consuming": int(raw_counts[3]),
        }
        return BufferSample(
            timestamp=time.monotonic() if timestamp is None else timestamp,
            next_claim_seq=int(self._header[3]),
            occupied_chunks=sum(
                counts[name] for name in ("writing", "ready", "consuming")
            ),
            counts=counts,
            target_chunks=int(self._header[2]),
        )


def calculate_rates(
    first: BufferSample,
    last: BufferSample,
    *,
    chunk_size_tokens: int,
) -> ThroughputRates | None:
    elapsed = last.timestamp - first.timestamp
    produced_chunks = last.next_claim_seq - first.next_claim_seq
    occupied_delta = last.occupied_chunks - first.occupied_chunks
    consumed_chunks = produced_chunks - occupied_delta
    if elapsed <= 0 or produced_chunks < 0 or consumed_chunks < 0:
        return None
    vllm_tps = produced_chunks * chunk_size_tokens / elapsed
    sae_tps = consumed_chunks * chunk_size_tokens / elapsed
    return ThroughputRates(
        window_seconds=elapsed,
        fill_ratio=last.fill_ratio,
        vllm_tokens_per_s=vllm_tps,
        sae_tokens_per_s=sae_tps,
        net_tokens_per_s=vllm_tps - sae_tps,
    )


class AutoSwitchPolicy:
    """Hysteresis policy driven only by buffer flow rates and watermarks."""

    def __init__(
        self,
        *,
        low_watermark: float,
        high_watermark: float,
        stable_samples: int,
        min_rate_ratio: float,
        min_rate_gap: float,
    ) -> None:
        if not 0 <= low_watermark < high_watermark <= 1:
            raise ValueError("watermarks must satisfy 0 <= low < high <= 1")
        if stable_samples < 1:
            raise ValueError("stable_samples must be >= 1")
        if min_rate_ratio < 1:
            raise ValueError("min_rate_ratio must be >= 1")
        if min_rate_gap < 0:
            raise ValueError("min_rate_gap must be >= 0")
        self.low_watermark = low_watermark
        self.high_watermark = high_watermark
        self.stable_samples = stable_samples
        self.min_rate_ratio = min_rate_ratio
        self.min_rate_gap = min_rate_gap
        self.growing_samples = 0
        self.draining_samples = 0

    def reset(self) -> None:
        self.growing_samples = 0
        self.draining_samples = 0

    def observe(
        self,
        *,
        active_sae_dp: int,
        min_sae_dp: int,
        max_sae_dp: int,
        rates: ThroughputRates,
        production_complete: bool = False,
    ) -> int | None:
        growing = (
            rates.fill_ratio >= self.high_watermark
            and rates.net_tokens_per_s >= self.min_rate_gap
            and rates.vllm_tokens_per_s
            >= rates.sae_tokens_per_s * self.min_rate_ratio
            and rates.vllm_tokens_per_s > 0
        )
        draining = (
            not production_complete
            and rates.fill_ratio <= self.low_watermark
            and -rates.net_tokens_per_s >= self.min_rate_gap
            and rates.sae_tokens_per_s
            >= rates.vllm_tokens_per_s * self.min_rate_ratio
            and rates.sae_tokens_per_s > 0
        )

        if active_sae_dp == min_sae_dp and growing:
            self.growing_samples += 1
        else:
            self.growing_samples = 0
        if active_sae_dp == max_sae_dp and draining:
            self.draining_samples += 1
        else:
            self.draining_samples = 0

        if self.growing_samples >= self.stable_samples:
            self.reset()
            return max_sae_dp
        if self.draining_samples >= self.stable_samples:
            self.reset()
            return min_sae_dp
        return None


def _controller(path: Path) -> ElasticStreamingController:
    data = json.loads(path.read_text())
    if int(data.get("version", 1)) < 2:
        raise ValueError(
            "this control file uses the old equal-TP layout; restart the run to "
            "create a current elastic control file"
        )
    vllm_tp_size = int(data["vllm_tp_size"])
    sae_tp_size = int(data["sae_tp_size"])
    sae_pp_size = int(data["sae_pp_size"])
    permanent_vllm_dp = int(data["permanent_vllm_dp"])
    permanent_sae_dp = int(data["permanent_sae_dp"])
    elastic_rank_count = int(data["elastic_rank_count"])
    layout = ElasticStreamingLayout(
        world_size=(
            permanent_vllm_dp * vllm_tp_size
            + elastic_rank_count
            + permanent_sae_dp * sae_tp_size * sae_pp_size
        ),
        vllm_tp_size=vllm_tp_size,
        sae_tp_size=sae_tp_size,
        sae_pp_size=sae_pp_size,
        permanent_vllm_dp=permanent_vllm_dp,
        permanent_sae_dp=permanent_sae_dp,
        elastic_rank_count=elastic_rank_count,
    )
    return ElasticStreamingController(path, layout)


def _append_log(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        json.dump({"timestamp": time.time(), "event": event, **fields}, handle)
        handle.write("\n")


def _run_auto(args: argparse.Namespace, controller: ElasticStreamingController) -> None:
    state = controller.read()
    monitor = SharedBufferMonitor(state, args.shm_base_dir)
    history: deque[BufferSample] = deque()
    policy = AutoSwitchPolicy(
        low_watermark=args.low_watermark,
        high_watermark=args.high_watermark,
        stable_samples=args.stable_samples,
        min_rate_ratio=args.min_rate_ratio,
        min_rate_gap=args.min_rate_gap,
    )
    log_path = args.log_path or args.control_path.with_suffix(
        args.control_path.suffix + ".auto.jsonl"
    )
    stop = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    last_epoch = state.epoch
    last_active_sae_dp = state.active_sae_dp
    last_switch_at = float("-inf")
    completed_switches = 0
    switching_epoch: int | None = None
    switching_started = 0.0
    dry_run_decisions = 0
    print(
        "auto-switch started: "
        f"sae_dp={controller.layout.min_sae_dp}<->{controller.layout.max_sae_dp}, "
        f"watermarks={args.low_watermark:.0%}/{args.high_watermark:.0%}, "
        f"window={args.rate_window_seconds:g}s, stable={args.stable_samples}"
    )

    while not stop:
        loop_started = time.monotonic()
        state = controller.read()
        if state.phase == "failed":
            raise RuntimeError(f"elastic streaming failed: {state.error}")
        if state.phase == "finished":
            print("elastic streaming run finished")
            return
        if state.phase == "switching":
            if switching_epoch != state.epoch:
                switching_epoch = state.epoch
                switching_started = loop_started
            elif loop_started - switching_started > args.switch_timeout:
                raise TimeoutError(
                    f"switch epoch {state.epoch} remained in phase=\"switching\" "
                    f"for more than {args.switch_timeout:g}s"
                )
        else:
            switching_epoch = None

        if state.epoch != last_epoch or state.active_sae_dp != last_active_sae_dp:
            history.clear()
            policy.reset()
            if state.phase == "active" and state.active_sae_dp != last_active_sae_dp:
                last_switch_at = loop_started
                completed_switches += 1
            last_epoch = state.epoch
            last_active_sae_dp = state.active_sae_dp

        sample = monitor.sample(timestamp=loop_started)
        history.append(sample)
        while (
            len(history) > 2
            and sample.timestamp - history[1].timestamp >= args.rate_window_seconds
        ):
            history.popleft()

        rates = calculate_rates(
            history[0],
            history[-1],
            chunk_size_tokens=monitor.chunk_size_tokens,
        )
        window_ready = (
            rates is not None
            and rates.window_seconds >= args.rate_window_seconds * 0.8
        )
        target = None
        cooldown_remaining = max(
            0.0, args.cooldown_seconds - (loop_started - last_switch_at)
        )
        if state.phase == "active" and window_ready and cooldown_remaining == 0:
            assert rates is not None
            target = policy.observe(
                active_sae_dp=state.active_sae_dp,
                min_sae_dp=controller.layout.min_sae_dp,
                max_sae_dp=controller.layout.max_sae_dp,
                rates=rates,
                production_complete=sample.production_complete,
            )
        elif state.phase != "active" or cooldown_remaining > 0:
            policy.reset()

        record: dict[str, object] = {
            "phase": state.phase,
            "epoch": state.epoch,
            "active_sae_dp": state.active_sae_dp,
            "target_sae_dp": state.target_sae_dp,
            "fill_ratio": sample.fill_ratio,
            "counts": sample.counts,
            "production_complete": sample.production_complete,
            "cooldown_remaining_s": cooldown_remaining,
        }
        if rates is not None:
            record.update(asdict(rates))
        _append_log(log_path, "sample", **record)

        rate_text = "warming up"
        if rates is not None:
            rate_text = (
                f"vllm={rates.vllm_tokens_per_s:,.0f} tok/s "
                f"sae={rates.sae_tokens_per_s:,.0f} tok/s "
                f"net={rates.net_tokens_per_s:+,.0f} tok/s"
            )
        print(
            f"epoch={state.epoch} phase={state.phase} sae_dp={state.active_sae_dp} "
            f"fill={sample.fill_ratio:.1%} {rate_text} "
            f"stable=+{policy.growing_samples}/-{policy.draining_samples}"
        )

        if target is not None:
            reason = (
                "high-and-growing"
                if target == controller.layout.max_sae_dp
                else "low-and-draining"
            )
            _append_log(
                log_path,
                "decision",
                decision_target_sae_dp=target,
                reason=reason,
                **record,
            )
            if args.dry_run:
                print(
                    f"dry-run: would switch sae_dp "
                    f"{state.active_sae_dp}->{target} ({reason})"
                )
                last_switch_at = loop_started
                dry_run_decisions += 1
            else:
                try:
                    requested = controller.request(target)
                except (RuntimeError, ValueError) as exc:
                    print(f"switch request skipped: {exc}")
                    _append_log(log_path, "request_skipped", error=str(exc))
                else:
                    print(
                        f"requested epoch={requested.epoch} sae_dp "
                        f"{requested.active_sae_dp}->{requested.target_sae_dp} "
                        f"({reason})"
                    )
                    _append_log(
                        log_path,
                        "switch_requested",
                        epoch=requested.epoch,
                        source_sae_dp=requested.active_sae_dp,
                        target_sae_dp=requested.target_sae_dp,
                        reason=reason,
                    )
                    history.clear()
                    policy.reset()

        counted_actions = dry_run_decisions if args.dry_run else completed_switches
        if args.max_switches > 0 and counted_actions >= args.max_switches:
            action_name = "decision(s)" if args.dry_run else "switch(es)"
            print(f"completed {counted_actions} automatic {action_name}")
            return
        sleep_for = args.poll_interval - (time.monotonic() - loop_started)
        if sleep_for > 0:
            time.sleep(sleep_for)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_path", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    switch = subparsers.add_parser("switch")
    switch.add_argument("--sae-dp", type=int, required=True)
    switch.add_argument("--wait", action="store_true")
    switch.add_argument("--timeout", type=float, default=600.0)

    auto = subparsers.add_parser("auto")
    auto.add_argument("--low-watermark", type=float, default=0.25)
    auto.add_argument("--high-watermark", type=float, default=0.75)
    auto.add_argument("--poll-interval", type=float, default=2.0)
    auto.add_argument("--rate-window-seconds", type=float, default=10.0)
    auto.add_argument("--stable-samples", type=int, default=3)
    auto.add_argument("--min-rate-ratio", type=float, default=1.05)
    auto.add_argument("--min-rate-gap", type=float, default=0.0)
    auto.add_argument("--cooldown-seconds", type=float, default=60.0)
    auto.add_argument("--switch-timeout", type=float, default=600.0)
    auto.add_argument("--max-switches", type=int, default=0)
    auto.add_argument("--dry-run", action="store_true")
    auto.add_argument("--shm-base-dir", type=Path, default=Path("/dev/shm"))
    auto.add_argument("--log-path", type=Path)
    args = parser.parse_args()

    if args.command == "auto":
        if (
            args.poll_interval <= 0
            or args.rate_window_seconds <= 0
            or args.switch_timeout <= 0
        ):
            parser.error(
                "poll interval, rate window, and switch timeout must be positive"
            )
        if args.cooldown_seconds < 0 or args.max_switches < 0:
            parser.error("cooldown and max switches must be non-negative")

    controller = _controller(args.control_path)
    if args.command == "status":
        print(json.dumps(controller.read().__dict__, indent=2, sort_keys=True))
        return
    if args.command == "auto":
        _run_auto(args, controller)
        return

    requested = controller.request(args.sae_dp)
    print(
        f"requested epoch={requested.epoch} sae_dp "
        f"{requested.active_sae_dp}->{requested.target_sae_dp}"
    )
    if not args.wait:
        return
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        state = controller.read()
        if state.epoch == requested.epoch and state.phase != "switching":
            print(json.dumps(state.__dict__, indent=2, sort_keys=True))
            if state.phase == "failed":
                raise SystemExit(1)
            return
        time.sleep(0.25)
    raise TimeoutError(f"switch epoch {requested.epoch} did not finish in time")


if __name__ == "__main__":
    main()
