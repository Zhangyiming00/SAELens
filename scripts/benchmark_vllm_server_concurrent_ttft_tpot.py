#!/usr/bin/env python3
"""
Launch a native vLLM OpenAI-compatible server, then benchmark concurrent decode
requests with TTFT / TPOT metrics.

This is a thin wrapper around the existing concurrent HTTP benchmark:
`scripts/scripts/benchmark_vllm_http_concurrent_ttft_tpot.py`

The goal is to expose a simpler entrypoint with an interface closer to
`benchmark_vllm_tp_ttft_tpot.py`:
  1. start a vLLM server for each TP setting
  2. send `batch_size` independent requests concurrently
  3. measure TTFT / TPOT / round completion

Example:
    python scripts/benchmark_vllm_server_concurrent_ttft_tpot.py \
        --model-path /data/models/Llama-3.1-8B \
        --tp 1 2 \
        --batch-sizes 4,8,12 \
        --context-sizes 1024,2048 \
        --output-len 128 \
        --num-warmup 3 \
        --num-measure 20 \
        --max-model-len 2176 \
        --gpu-memory-utilization 0.9
"""

from __future__ import annotations

import argparse
import runpy
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL_PATH = "/data/models/Llama-3.1-8B"
DEFAULT_BATCH_SIZES = [4, 8, 12]
DEFAULT_CONTEXT_SIZES = [1024, 2048]
DEFAULT_OUTPUT_LEN = 128
DEFAULT_NUM_WARMUP = 3
DEFAULT_NUM_MEASURE = 20
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_PORT_BASE = 8500
DEFAULT_SERVER_START_TIMEOUT_S = 300.0
DEFAULT_REQUEST_TIMEOUT_S = 300.0
DEFAULT_SERVED_MODEL_NAME = "benchmark-model"


def _int_list(arg: str) -> list[int]:
    return [int(x) for x in arg.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--batch-sizes",
        type=_int_list,
        default=DEFAULT_BATCH_SIZES,
        metavar="4,8,12",
    )
    parser.add_argument(
        "--context-sizes",
        type=_int_list,
        default=DEFAULT_CONTEXT_SIZES,
        metavar="1024,2048",
    )
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    parser.add_argument("--num-warmup", type=int, default=DEFAULT_NUM_WARMUP)
    parser.add_argument("--num-measure", type=int, default=DEFAULT_NUM_MEASURE)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument(
        "--server-start-timeout-s",
        type=float,
        default=DEFAULT_SERVER_START_TIMEOUT_S,
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
    )
    parser.add_argument(
        "--enforce-eager",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Whether to force eager mode on the vLLM server.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable prefix caching on the server.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_len < 1:
        raise ValueError("--output-len must be at least 1.")

    if args.max_model_len is None:
        args.max_model_len = max(args.context_sizes) + args.output_len
    if args.max_model_len < max(args.context_sizes) + args.output_len:
        raise ValueError(
            "--max-model-len must be at least max(context_sizes) + output_len. "
            f"Got {args.max_model_len}, required {max(args.context_sizes) + args.output_len}."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("results") / f"vllm_server_concurrent_ttft_tpot_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    delegate = Path(__file__).resolve().parent / "scripts" / "benchmark_vllm_http_concurrent_ttft_tpot.py"
    if not delegate.exists():
        raise FileNotFoundError(f"Delegate benchmark script not found: {delegate}")

    argv = [
        str(delegate),
        "--model-path",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--tp",
        *[str(x) for x in args.tp],
        "--batch-sizes",
        ",".join(str(x) for x in args.batch_sizes),
        "--context-sizes",
        ",".join(str(x) for x in args.context_sizes),
        "--decode-len",
        str(args.output_len),
        "--num-warmup",
        str(args.num_warmup),
        "--num-measure",
        str(args.num_measure),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--port-base",
        str(args.port_base),
        "--server-start-timeout-s",
        str(args.server_start_timeout_s),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--modes",
        "decode32",
        "--output-dir",
        str(output_dir),
    ]
    if args.enforce_eager:
        argv.extend(["--enforce-eager", "true"])
    if args.enable_prefix_caching:
        argv.append("--enable-prefix-caching")

    print(f"Model:         {args.model_path}")
    print(f"TP sizes:      {args.tp}")
    print(f"Batch sizes:   {args.batch_sizes}")
    print(f"Contexts:      {args.context_sizes}")
    print(f"output_len:    {args.output_len}")
    print(f"dtype:         {args.dtype}")
    print(f"enforce_eager: {args.enforce_eager}")
    print(f"output_dir:    {output_dir}")
    print(f"delegate:      {delegate}")

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(delegate), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
