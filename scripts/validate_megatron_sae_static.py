"""Run static TP1/2/4 native-oracle acceptance and retain per-rank error reports."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.saes.test_megatron_sae_native_reference import (  # noqa: E402, TID251
    _worker,
    test_oracle_integrity,
)
from tests.saes.test_megatron_sae_trainers import (  # noqa: E402, TID251
    _distributed_worker,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--with-trainers",
        action="store_true",
        help="Also validate TP x DDP, unequal batches, both multi-hook wrappers, and disk resume",
    )
    args = parser.parse_args()
    if torch.cuda.device_count() < 4:
        parser.error("Four CUDA GPUs are required; a skipped test is not acceptance")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    test_oracle_integrity()
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO / "tests/native_reference/generate.py"),
            "--device",
            "cuda:0",
            "--check",
        ],
        check=True,
    )
    mp.spawn(
        _worker,
        args=(f"file://{output / 'rendezvous'}", str(output)),
        nprocs=4,
        join=True,
    )
    reports = [
        json.loads((output / f"rank{rank}.json").read_text()) for rank in range(4)
    ]
    trainer_reports = None
    if args.with_trainers:
        mp.spawn(
            _distributed_worker,
            args=(f"file://{output / 'trainer_rendezvous'}", str(output), "nccl", 4),
            nprocs=4,
            join=True,
        )
        trainer_reports = [
            json.loads((output / f"trainer_rank{rank}.json").read_text())
            for rank in range(4)
        ]
    summary = {
        "status": "passed",
        "native_version": "6.37.6",
        "tp_sizes": [1, 2, 4],
        "rank_reports": reports,
        "trainer_reports": trainer_reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"TP1/2/4 passed; reports: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
