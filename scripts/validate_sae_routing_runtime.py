"""Four-GPU acceptance for explicit Megatron domains and synchronous SAE units."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_sae_routing_runtime import _routing_worker  # noqa: E402, TID251
from tests.test_sae_runtime import _runtime_worker  # noqa: E402, TID251


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 4:
        parser.error("Four CUDA GPUs are required; a skip is not acceptance")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
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
        _runtime_worker,
        args=(f"file://{output / 'domain_rdzv'}", str(output), "nccl"),
        nprocs=4,
    )
    mp.spawn(
        _routing_worker,
        args=(f"file://{output / 'routing_rdzv'}", str(output)),
        nprocs=4,
    )
    reports = {
        kind: [
            json.loads((output / f"{kind}_rank{r}.json").read_text()) for r in range(4)
        ]
        for kind in ("runtime", "routing")
    }
    summary = {
        "status": "passed",
        "megatron_version": "0.16.1",
        "group_initialization": "explicit RankGenerator/create_group/ProcessGroupCollection",
        "ddp": "pytorch_per_hook",
        "optimizer": "independent_adam_per_hook",
        "schedule": "synchronous",
        "steps": 6,
        "disk_resume_step": 3,
        "reports": reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SAE routing/runtime acceptance passed: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
