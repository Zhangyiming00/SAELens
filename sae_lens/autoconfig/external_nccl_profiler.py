#!/usr/bin/env python3
"""Single-command controller for the existing external NCCL profiler.

Usage
-----
Run directly with plain Python:

    python external_nccl_profiler_2.py

or sweep multiple TP degrees in one command:

    python external_nccl_profiler_2.py --tp 2 4

The controller launches one torchrun job per TP degree, because one distributed
world has exactly one world_size.  Child workers always see exactly one TP
value.  All child rows are merged into ONE final CSV/JSON pair.

The actual collective construction, timing, row schema, and metadata are reused
from the repository's existing ``nccl_profiler_core.py`` so this file only
changes orchestration, not the NCCL measurement definition.
"""

from __future__ import annotations

# =============================================================================
# USER-EDITABLE DEFAULT PARAMETERS
# =============================================================================
# Keep [2] if the machine has only two GPUs.  Change to [2, 4] on a 4-GPU
# machine if both TP degrees should be profiled by a zero-argument invocation.
DEFAULT_TP_VALUES: list[int] = [2,4]

# None means inherit the current external_nccl_profiler.py defaults.
DEFAULT_BUFFER_BYTES: list[int] | None = None
DEFAULT_COLLECTIVES: list[str] | None = None
DEFAULT_DTYPES: list[str] | None = None

DEFAULT_WARMUP: int | None = None
DEFAULT_REPEATS: int | None = None
DEFAULT_OUTPUT_DIR: str | None = None
DEFAULT_OUTPUT_NAME: str = "nccl_comm_profile"

# Optional CUDA visibility override, e.g. "0,1,2,3".  None = inherit environment.
DEFAULT_CUDA_DEVICES: str | None = None

DEFAULT_BACKEND: str = "nccl"
DEFAULT_NCCL_ALGO: str = ""
DEFAULT_NCCL_PROTO: str = ""
DEFAULT_NCCL_P2P_LEVEL: str = ""
# =============================================================================

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    # Works when SAELens is installed / repo root is on PYTHONPATH.
    from sae_lens.autoconfig import nccl_profiler_core as core
except ImportError:
    # Works when invoked directly from ~/SAELens/sae_lens/autoconfig.
    import nccl_profiler_core as core  # type: ignore[no-redef]


_TORCHRUN_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _core_default(value: Any, core_name: str) -> Any:
    return getattr(core, core_name) if value is None else value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile NCCL collectives for one or more TP degrees from a single "
            "plain-python command. Each TP runs in its own torchrun world and "
            "all results are merged."
        )
    )

    parser.add_argument(
        "--buffer-bytes",
        dest="buffer_bytes_values",
        type=int,
        nargs="+",
        default=list(_core_default(DEFAULT_BUFFER_BYTES, "DEFAULT_BUFFER_BYTES")),
        help="Local payload bytes contributed by each rank.",
    )
    parser.add_argument(
        "--tp",
        "--group-size",
        dest="tp_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_TP_VALUES),
        help="TP/process-group degrees. Plain python may specify multiple values.",
    )
    parser.add_argument(
        "--collectives",
        nargs="+",
        choices=list(core.COLLECTIVES),
        default=list(_core_default(DEFAULT_COLLECTIVES, "DEFAULT_COLLECTIVES")),
    )
    parser.add_argument(
        "--dtype",
        "--dtypes",
        dest="dtypes",
        nargs="+",
        default=list(_core_default(DEFAULT_DTYPES, "DEFAULT_DTYPES")),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(_core_default(DEFAULT_WARMUP, "DEFAULT_WARMUP")),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=int(_core_default(DEFAULT_REPEATS, "DEFAULT_REPEATS")),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_core_default(DEFAULT_OUTPUT_DIR, "DEFAULT_OUTPUT_DIR")),
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default=DEFAULT_BACKEND)
    parser.add_argument("--nccl-algo", default=DEFAULT_NCCL_ALGO)
    parser.add_argument("--nccl-proto", default=DEFAULT_NCCL_PROTO)
    parser.add_argument("--nccl-p2p-level", default=DEFAULT_NCCL_P2P_LEVEL)
    parser.add_argument(
        "--cuda-devices",
        default=DEFAULT_CUDA_DEVICES,
        help='Optional CUDA_VISIBLE_DEVICES override, e.g. "0,1,2,3".',
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep per-TP worker result/log files after a successful run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one TP worker fails.",
    )

    # Internal child mode. Users normally never pass these.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", default=None, help=argparse.SUPPRESS)

    return parser


def _config_from_args(
    args: argparse.Namespace,
    *,
    tp_values: Sequence[int] | None = None,
) -> Any:
    return core.NcclSweepConfig(
        buffer_bytes_values=list(args.buffer_bytes_values),
        tp_values=list(args.tp_values if tp_values is None else tp_values),
        collectives=list(args.collectives),
        dtypes=list(args.dtypes),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        output_dir=str(args.output_dir),
        output_name=str(args.output_name),
        backend=str(args.backend),
        nccl_algo=str(args.nccl_algo),
        nccl_proto=str(args.nccl_proto),
        nccl_p2p_level=str(args.nccl_p2p_level),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _tail(path: Path, lines: int = 120) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(content[-lines:])


def _worker(args: argparse.Namespace) -> int:
    """One torchrun world. Exactly one TP value is allowed here."""
    import torch.distributed as dist

    result_file = Path(args.result_file).resolve() if args.result_file else None
    rank = int(os.environ.get("RANK", "0"))
    initialized_here = False

    try:
        if len(args.tp_values) != 1:
            raise RuntimeError(
                f"internal worker requires exactly one --tp value; got {args.tp_values}"
            )

        tp = int(args.tp_values[0])
        config = _config_from_args(args, tp_values=[tp]).normalize()
        config.validate_inputs()

        # Topology variables must be set before any NCCL communicator is created.
        core.apply_topo_env(config)

        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if config.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL backend requested but CUDA is unavailable")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible"
                )
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            if config.backend == "nccl":
                device = torch.device(f"cuda:{local_rank}")
                try:
                    dist.init_process_group(
                        backend=config.backend,
                        device_id=device,
                    )
                except TypeError:
                    dist.init_process_group(backend=config.backend)
            else:
                dist.init_process_group(backend=config.backend)
            initialized_here = True

        world_size = dist.get_world_size()
        if world_size != tp:
            raise RuntimeError(
                f"worker world_size={world_size} does not equal requested TP={tp}"
            )

        if rank == 0:
            print(
                f"[nccl-worker] TP={tp} world_size={world_size} "
                f"cases={len(config.collectives) * len(config.buffer_bytes_values) * len(config.dtypes)}",
                flush=True,
            )

        rows = core.run_worker(config)

        if dist.get_rank() == 0:
            if result_file is None:
                raise RuntimeError("internal worker missing --result-file")
            _atomic_json(
                result_file,
                {
                    "status": "ok",
                    "tp": tp,
                    "rows": rows,
                },
            )
            print(
                f"[nccl-worker] TP={tp} done: {len(rows)} row(s)",
                flush=True,
            )

        if config.backend == "nccl":
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        return 0

    except BaseException as exc:
        if rank == 0 and result_file is not None:
            try:
                _atomic_json(
                    result_file,
                    {
                        "status": "error",
                        "tp": (
                            int(args.tp_values[0])
                            if len(args.tp_values) == 1
                            else None
                        ),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(limit=40),
                        "rows": [],
                    },
                )
            except Exception:
                pass
        traceback.print_exc()
        return 1

    finally:
        if initialized_here and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _child_command(
    args: argparse.Namespace,
    *,
    tp: int,
    result_file: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={tp}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-file",
        str(result_file.resolve()),
        "--tp",
        str(tp),
        "--buffer-bytes",
        *[str(value) for value in args.buffer_bytes_values],
        "--collectives",
        *[str(value) for value in args.collectives],
        "--dtype",
        *[str(value) for value in args.dtypes],
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--output-dir",
        str(args.output_dir),
        "--output-name",
        str(args.output_name),
        "--backend",
        str(args.backend),
    ]

    if args.nccl_algo:
        command += ["--nccl-algo", str(args.nccl_algo)]
    if args.nccl_proto:
        command += ["--nccl-proto", str(args.nccl_proto)]
    if args.nccl_p2p_level:
        command += ["--nccl-p2p-level", str(args.nccl_p2p_level)]
    if args.cuda_devices:
        command += ["--cuda-devices", str(args.cuda_devices)]

    return command


def _controller(args: argparse.Namespace) -> int:
    parent_config = _config_from_args(args).normalize()
    parent_config.validate_inputs()

    tp_values = list(parent_config.tp_values)

    # A plain-python controller must not accidentally inherit an outer torchrun.
    inherited = [name for name in _TORCHRUN_ENV_KEYS if name in os.environ]
    if inherited:
        print(
            "[nccl-controller] warning: removing inherited distributed env keys: "
            + ", ".join(inherited),
            flush=True,
        )

    env_base = os.environ.copy()
    for key in _TORCHRUN_ENV_KEYS:
        env_base.pop(key, None)

    if args.cuda_devices:
        env_base["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)

    visible_count = None
    if parent_config.backend == "nccl":
        # This count reflects the controller's currently visible GPUs. If the
        # user changes CUDA visibility via --cuda-devices, validate by count.
        if args.cuda_devices:
            requested_visible = [
                item for item in str(args.cuda_devices).split(",") if item.strip()
            ]
            visible_count = len(requested_visible)
        else:
            visible_count = torch.cuda.device_count()

        too_large = [tp for tp in tp_values if tp > visible_count]
        if too_large:
            raise RuntimeError(
                f"requested TP value(s) {too_large} exceed visible GPU count "
                f"{visible_count}. tp_values={tp_values}"
            )

    output_dir = Path(parent_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    part_root = Path(
        tempfile.mkdtemp(
            prefix=".nccl_parts_",
            dir=str(output_dir),
        )
    )

    all_rows: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []

    try:
        print("=== NCCL multi-TP controller ===", flush=True)
        print(f"TP values     : {tp_values}", flush=True)
        print(f"backend       : {parent_config.backend}", flush=True)
        print(f"dtypes        : {parent_config.dtypes}", flush=True)
        print(f"collectives   : {parent_config.collectives}", flush=True)
        print(f"buffer points : {len(parent_config.buffer_bytes_values)}", flush=True)
        print(f"warmup/repeat : {parent_config.warmup}/{parent_config.repeats}", flush=True)
        if visible_count is not None:
            print(f"visible GPUs  : {visible_count}", flush=True)

        for tp in tp_values:
            result_file = part_root / f"tp{tp}.result.json"
            log_file = part_root / f"tp{tp}.log"
            command = _child_command(args, tp=tp, result_file=result_file)

            print(
                f"\n[RUN] TP={tp}: torchrun --nproc_per_node={tp}",
                flush=True,
            )
            print("[CMD] " + " ".join(command), flush=True)

            with log_file.open("w") as handle:
                completed = subprocess.run(
                    command,
                    env=env_base,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            payload: dict[str, Any] | None = None
            if result_file.exists():
                try:
                    payload = json.loads(result_file.read_text())
                except Exception as exc:
                    failures.append(
                        (tp, f"invalid worker result JSON: {type(exc).__name__}: {exc}")
                    )

            if (
                completed.returncode == 0
                and payload is not None
                and payload.get("status") == "ok"
            ):
                rows = list(payload.get("rows", []))
                all_rows.extend(rows)
                print(
                    f"[OK] TP={tp}: {len(rows)} row(s)",
                    flush=True,
                )
                continue

            if payload is not None and payload.get("status") == "error":
                reason = (
                    f"{payload.get('error_type', 'WorkerError')}: "
                    f"{payload.get('error_message', '')}"
                )
            else:
                reason = f"worker exit={completed.returncode}, no valid result"

            log_tail = _tail(log_file)
            failures.append((tp, reason))
            print(
                f"[FAIL] TP={tp}: {reason}\n"
                f"--- log tail: {log_file} ---\n{log_tail}",
                file=sys.stderr,
                flush=True,
            )

            if args.fail_fast:
                break

        if failures:
            summary = ", ".join(f"TP{tp}: {reason}" for tp, reason in failures)
            raise RuntimeError(
                "one or more NCCL TP jobs failed; no merged final profile written. "
                + summary
            )

        # One final profile file containing every requested TP.
        csv_path, json_path = core.write_outputs(parent_config, all_rows)

        print("\n=== NCCL profile complete ===", flush=True)
        print(f"rows : {len(all_rows)}", flush=True)
        print(f"csv  : {csv_path}", flush=True)
        print(f"json : {json_path}", flush=True)
        return 0

    finally:
        if args.keep_parts:
            print(f"[nccl-controller] kept worker files: {part_root}", flush=True)
        else:
            shutil.rmtree(part_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.worker:
            return _worker(args)
        return _controller(args)
    except KeyboardInterrupt:
        print("\n[nccl-profiler] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"[nccl-profiler] fatal: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
