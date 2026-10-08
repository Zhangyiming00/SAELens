"""Supervise online elastic TP training without repository-relative executables.

The public entry is ``ElasticTPSAETrainingRunner(config).run()``. Workers are
launched as package modules so an installed wheel works without the repository.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from sae_lens.training.elastic_tp_config import (
    OFFLINE_ENV,
    ElasticTPConfig,
    activation_dtype,
    configure_offline_environment,
    online_hooks,
    open_buffer,
    validate_local_sources,
    write_json,
)
from sae_lens.training.elastic_tp_handoff import (
    producer_quiesced,
    producer_running,
    read_status,
)

logger = logging.getLogger(__name__)
WORKER_MODULE = "sae_lens.elastic_tp_runner"


def visible_devices(pool_size: int) -> list[str]:
    """Resolve logical pool ranks within the caller's CUDA visibility mask."""
    import torch

    count = torch.cuda.device_count()
    if count < pool_size:
        raise ValueError(f"Elastic TP needs {pool_size} visible GPUs; found {count}")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [value.strip() for value in mask.split(",")]
        if mask is not None
        else [str(i) for i in range(count)]
    )
    if len(devices) < pool_size or any(not d or d == "-1" for d in devices[:pool_size]):
        raise ValueError("CUDA_VISIBLE_DEVICES does not cover the elastic TP pool")
    return devices[:pool_size]


def worker_command(
    config_path: Path, role: str, pool_size: int, producer_id: int = 1
) -> list[str]:
    command = [sys.executable]
    if role == "train":
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={pool_size}",
            "--module",
        ]
    else:
        command += ["-m"]
    return command + [
        WORKER_MODULE,
        "--config",
        str(config_path),
        "--role",
        role,
        "--producer-id",
        str(producer_id),
    ]


def worker_environment(
    devices: list[str], producer_id: int | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(OFFLINE_ENV)
    # A supervisor is never a torchrun worker; do not leak stale rendezvous state
    # into vLLM's single-device processes or the new torchrun agent.
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = (
        ",".join(devices) if producer_id is None else devices[producer_id]
    )
    for key, value in {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "NCCL_LAUNCH_ORDER_IMPLICIT": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }.items():
        env.setdefault(key, value)
    return env


def stop_process(process: subprocess.Popen, *, grace_s: float = 20) -> None:
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def raise_for_training_failure(cfg: ElasticTPConfig) -> None:
    """Observe rank-local errors before waiting for torchrun/CUDA teardown."""
    failures = []
    for rank in range(cfg.pool_size):
        path = cfg.output / f"train_rank{rank}_error.json"
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        failures.append((record, path))
    if failures:
        record, path = min(failures, key=lambda item: item[0]["timestamp"])
        raise RuntimeError(
            f"Elastic TP rank {record['rank']} failed: "
            f"{record['error_type']}: {record['error']}; see {path}"
        )


def wait_for_producers(cfg, producers, *, tp, epoch):
    """Finish startup handoffs before any initial SAE tensors are allocated."""
    started = time.monotonic()
    while True:
        if any(p.poll() is not None for p in producers):
            raise RuntimeError("A vLLM producer exited during startup; see producer logs")
        acknowledged = []
        for gpu in range(1, cfg.pool_size):
            status = read_status(cfg.output / f"producer{gpu}_status.json")
            acknowledged.append(
                producer_quiesced(status, epoch, cfg.vllm_residency)
                if gpu < tp else producer_running(status, epoch)
            )
        if all(acknowledged):
            return
        if time.monotonic() - started > cfg.startup_timeout:
            raise TimeoutError(f"vLLM startup handoff timed out at epoch {epoch}")
        time.sleep(0.02)


def prefill_for_full_tp(cfg: ElasticTPConfig, buffer, producers) -> None:
    """Fill a bounded cache and quiesce every producer before full-pool SAE init.

    No SAE worker has been started yet. Once the requested initial TP is
    running, the ordinary low-watermark controller can restore a producer.
    """
    control = cfg.output / "producer_control.json"
    target = min(cfg.chunks, max(cfg.cache_batches, cfg.gradient_accumulation_steps), cfg.steps)
    started = time.monotonic()

    def check_producers():
        if any(p.poll() is not None for p in producers):
            raise RuntimeError("A vLLM producer failed during startup prefill")
        if time.monotonic() - started > cfg.startup_timeout:
            raise TimeoutError("Elastic TP startup prefill timed out")

    write_json(control, dict(tp=1, epoch=1, stop=False, startup=True))
    while buffer.queue_counts()["ready"] < target:
        check_producers()
        time.sleep(0.02)
    write_json(control, dict(tp=cfg.pool_size, epoch=2, stop=False, startup=True))
    wait_for_producers(cfg, producers, tp=cfg.pool_size, epoch=2)
    write_json(
        cfg.output / "startup_prefill.json",
        dict(
            target_chunks=target,
            ready_chunks=buffer.queue_counts()["ready"],
            elapsed_s=time.monotonic() - started,
            epoch=2,
            vllm_residency=cfg.vllm_residency,
        ),
    )


class ElasticTPSAETrainingRunner:
    """Own the SHM buffer, releasable producers, workers and their cleanup."""

    def __init__(self, cfg: ElasticTPConfig):
        self.cfg = cfg

    def run(self) -> dict:
        cfg = self.cfg
        configure_offline_environment()
        validate_local_sources(cfg.model, cfg.dataset)
        if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "Launch --elastic-tp with plain python; it owns its torchrun worker pool"
            )
        devices = visible_devices(cfg.pool_size)
        cfg.output.mkdir(parents=True, exist_ok=True)
        config_path = cfg.output / "run_config.json"
        # Exclusive creation prevents stale status files and concurrent launches
        # from corrupting a run. No git checkout is needed for provenance.
        with config_path.open("x") as handle:
            json.dump(cfg.to_dict(), handle, indent=2, allow_nan=False)
        write_json(
            cfg.output / "startup.json",
            dict(
                pool_devices=devices,
                initial_sae_tp=cfg.initial_tp,
                minimum_sae_tp=cfg.min_tp,
                initial_sae_devices=devices[: cfg.initial_tp],
                initial_active_vllm_devices=devices[cfg.initial_tp :],
                prefill_before_training=cfg.initial_tp == cfg.pool_size,
                vllm_residency=cfg.vllm_residency,
                offline_environment=OFFLINE_ENV,
            ),
        )
        packages = {}
        for package in ("sae-lens", "torch", "vllm", "transformers", "huggingface-hub"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        write_json(
            cfg.output / "provenance.json",
            dict(
                packages=packages,
                devices=devices,
                dataset_cycles=True,
                producer_weights=(
                    "Close/release before SAE joins; cold reload after SAE releases"
                    if cfg.vllm_residency == "release"
                    else "Resident on GPU while paused; inference drains before SAE joins"
                ),
                vllm_residency=cfg.vllm_residency,
            ),
        )
        name = f"elastic_tp_{os.getpid()}_{time.time_ns()}"
        write_json(
            cfg.output / "buffer.json",
            dict(
                name=name,
                chunks=cfg.chunks,
                tokens_per_chunk=cfg.batch_size,
                activation_dtype=activation_dtype(cfg),
                **(dict(hook_names=online_hooks(cfg),
                        physical_rows_per_chunk=cfg.batch_size * len(online_hooks(cfg)))
                   if cfg.hook_names else {}),
            ),
        )
        buffer = training = None
        processes, handles = [], []
        started = time.perf_counter()
        returncode = 1
        error = None
        try:
            buffer = open_buffer(cfg, create=True, name=name)
            control_path = cfg.output / "producer_control.json"
            write_json(control_path, dict(tp=cfg.pool_size, epoch=0, stop=False, startup=True))
            for gpu in range(1, cfg.pool_size):
                handle = (cfg.output / f"producer{gpu}.log").open("w")
                handles.append(handle)
                processes.append(
                    subprocess.Popen(
                        worker_command(config_path, "producer", cfg.pool_size, gpu),
                        env=worker_environment(devices, gpu),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            producers = processes.copy()
            wait_for_producers(cfg, producers, tp=cfg.pool_size, epoch=0)
            if cfg.initial_tp == cfg.pool_size:
                prefill_for_full_tp(cfg, buffer, producers)
            else:
                write_json(control_path, dict(tp=cfg.initial_tp, epoch=1, stop=False, startup=True))
                wait_for_producers(cfg, producers, tp=cfg.initial_tp, epoch=1)
            handle = (cfg.output / "train.log").open("w")
            handles.append(handle)
            training = subprocess.Popen(
                worker_command(config_path, "train", cfg.pool_size),
                env=worker_environment(devices),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(training)
            write_json(
                cfg.output / "pids.json",
                dict(
                    supervisor=os.getpid(),
                    producers=[p.pid for p in producers],
                    trainer=training.pid,
                ),
            )
            last_status = -float("inf")
            while training.poll() is None:
                raise_for_training_failure(cfg)
                if any(p.poll() not in (None, 0) for p in producers):
                    raise RuntimeError(
                        "A vLLM producer failed during training; see producer logs"
                    )
                if time.perf_counter() - last_status > 30:
                    progress = cfg.output / "progress.json"
                    logger.info(
                        "Elastic TP: %s",
                        progress.read_text()
                        if progress.exists()
                        else "initializing SAE workers",
                    )
                    last_status = time.perf_counter()
                time.sleep(1)
            returncode = training.returncode
            raise_for_training_failure(cfg)
            if returncode:
                raise RuntimeError(
                    f"Elastic TP training exited with code {returncode}; see {cfg.output / 'train.log'}"
                )
            if any(p.poll() not in (None, 0) for p in producers):
                raise RuntimeError("A vLLM producer failed; see producer logs")
            report = json.loads((cfg.output / "training_report.json").read_text())
            if not report.get("passed"):
                raise RuntimeError("Elastic TP training did not report success")
        except BaseException as exc:
            returncode = returncode or 1
            error = str(exc)
            raise
        finally:
            try:
                control_path = cfg.output / "producer_control.json"
                control = (
                    json.loads(control_path.read_text()) if control_path.exists()
                    else dict(epoch=0, stop=False)
                )
                write_json(
                    cfg.output / "producer_control.json",
                    dict(tp=cfg.pool_size,
                         epoch=control["epoch"] + (not control["stop"]), stop=True),
                )
            finally:
                try:
                    for process in reversed(processes):
                        # A stop can arrive inside a non-interruptible cold
                        # load. Let it return and close before escalating.
                        stop_process(
                            process,
                            grace_s=(0 if error else 20)
                            if process is training else cfg.resume_timeout,
                        )
                finally:
                    if not returncode and any(p.returncode != 0 for p in processes):
                        returncode = 1
                        error = "A worker failed during shutdown; see worker logs"
                    for handle in handles:
                        handle.close()
                    if buffer is not None:
                        buffer.destroy()
                    write_json(
                        cfg.output / "result.json",
                        dict(
                            returncode=returncode,
                            error=error,
                            elapsed_s=time.perf_counter() - started,
                            process_returncodes=[p.returncode for p in processes],
                        ),
                    )
        if returncode:
            raise RuntimeError(error)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal elastic TP worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("producer", "train"), required=True)
    parser.add_argument("--producer-id", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = ElasticTPConfig(**json.loads(args.config.read_text()))
    configure_offline_environment()
    validate_local_sources(cfg.model, cfg.dataset)
    cfg.producer_id = args.producer_id
    if args.role == "producer":
        from sae_lens.training.elastic_tp_producer import producer

        producer(cfg)
    else:
        from sae_lens.training.elastic_tp_trainer import train

        train(cfg)


if __name__ == "__main__":
    main()
