"""Compare real-runner SHM/NCCL inputs and training state for equal/exact runs."""
import argparse
import json
from pathlib import Path

import torch


def tensor_errors(a, b):
    if isinstance(a, torch.Tensor):
        difference = (a.to(torch.float64) - b.to(torch.float64)).abs()
        return bool(torch.equal(a, b)), float(difference.max()) if a.numel() else 0.0
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        pairs = [tensor_errors(a[k], b[k]) for k in a]
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        pairs = [tensor_errors(x, y) for x, y in zip(a, b, strict=True)]
    else:
        assert a == b
        return True, 0.0
    return all(x[0] for x in pairs), max((x[1] for x in pairs), default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("equal", "exact"), default=["equal", "exact"])
    args = parser.parse_args()
    reports = []
    for mode in args.modes:
        left = args.directory / f"online_{mode}_shm" / "h2"
        right = args.directory / f"online_{mode}_nccl" / "h2"
        for phase in ("continuous", "resume"):
            for rank in range(4):
                a = json.loads((left / f"{phase}_rank{rank}.json").read_text())
                b = json.loads((right / f"{phase}_rank{rank}.json").read_text())
                # Hashes cover actual mixed input tensors at the training boundary.
                for key in ("steps", "microbatches", "vllm_generations", "producer_only",
                            "filtered_batch_rows", "runtime_closed", "owned_process_groups_released"):
                    assert a[key] == b[key], (mode, phase, rank, key)
                row = dict(mode=mode, phase=phase, rank=rank, inputs_bitwise_equal=True,
                           producer_only=a["producer_only"])
                if not a["producer_only"]:
                    sa = torch.load(left / f"{phase}_rank{rank}.pt", weights_only=True)
                    sb = torch.load(right / f"{phase}_rank{rank}.pt", weights_only=True)
                    # Permit only the existing FP32 DP-reduction tolerance.
                    torch.testing.assert_close(sa, sb, atol=1e-7, rtol=1e-6)
                    for section in ("models", "optimizer"):
                        equal, error = tensor_errors(sa[section], sb[section])
                        row[f"{section}_bitwise_equal"] = equal
                        row[f"{section}_max_abs_error"] = error
                    tensor_errors({k: v for k, v in sa.items() if k not in ("models", "optimizer")},
                                  {k: v for k, v in sb.items() if k not in ("models", "optimizer")})
                reports.append(row)
    result = dict(status="passed", atol=1e-7, rtol=1e-6, rows=reports)
    path = args.directory / "cross_transport_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
