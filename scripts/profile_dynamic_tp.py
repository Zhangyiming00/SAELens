"""Measure live TP migration with real SAE/Adam state on a fixed GPU pool.

Run with torchrun. Sampled canonical state is checked before/after each switch;
the small validate_dynamic_tp.py test compares every parameter and moment.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.dynamic_tp import DynamicTPSession, TPGroupPair
from sae_lens.training.dynamic_tp_diagnostics import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--widths", type=int, nargs="+", default=[32768, 32771])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-rows", type=int, default=256)
    parser.add_argument("--report", default="dynamic_tp_profile.json")
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    groups = TPGroupPair(tuple(range(size)), (0,), device=device)
    configs = {
        str(i): TopKTrainingSAEConfig(
            d_in=args.d_in,
            d_sae=width,
            k=min(64, width),
            auxk=32,
            dtype="float32",
            device=str(device),
            normalize_activations="none",
            topk_backend="sharded_dense",
        )
        for i, width in enumerate(args.widths)
    }
    records = []
    try:
        session = DynamicTPSession(configs, groups, adam_kwargs={"fused": True})
        sequence = [tuple(range(n)) for n in range(2, size + 1)]
        sequence += [tuple(range(n)) for n in range(size - 1, 0, -1)]
        for step, ranks in enumerate(sequence):
            # Prepare before the last old-topology training step.
            groups.synchronize()
            dist.barrier(group=groups.control)
            started = time.perf_counter()
            session.prepare(ranks)
            prepare_s = time.perf_counter() - started
            batches = {}
            if groups.rank in groups.active_ranks:
                generator = torch.Generator(device=device).manual_seed(19 + step)
                batches = {
                    h: torch.randn(
                        args.batch_size, args.d_in, device=device, generator=generator
                    )
                    for h in configs
                }
            session.train_step(batches)
            session.state.activation_caches.clear()
            if groups.rank in groups.active_ranks:
                cache = (
                    torch.arange(args.cache_rows, device=device, dtype=torch.float32)
                    .unsqueeze(1)
                    .expand(-1, args.d_in)
                    .contiguous()
                )
                session.stage_inputs({h: cache.clone() for h in configs})
            before = snapshot(session)
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
            local = session.switch()
            local.update(
                rank=groups.rank,
                prepare_s=prepare_s,
                allocated_before=allocated_before,
                allocated_after=torch.cuda.memory_allocated(device),
                peak_allocated=torch.cuda.max_memory_allocated(device),
            )
            after = snapshot(session)
            for h in before:
                for a, b in zip(before[h], after[h]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                if groups.rank in groups.active_ranks:
                    expected = (
                        torch.arange(
                            args.cache_rows, device=device, dtype=torch.float32
                        )
                        .unsqueeze(1)
                        .expand(-1, args.d_in)
                    )
                    torch.testing.assert_close(
                        session.state.activation_caches[h], expected, rtol=0, atol=0
                    )
            peers = groups.agree(local)
            record = dict(
                old_ranks=local["old_ranks"],
                new_ranks=local["new_ranks"],
                pause_s=max(p["pause_s"] for p in peers),
                prepare_s=max(p["prepare_s"] for p in peers),
                transferred_bytes=local["transferred_bytes"],
                peak_allocated=max(p["peak_allocated"] for p in peers),
                peers=peers,
            )
            records.append(record)
            if groups.rank == 0:
                print(
                    json.dumps({k: v for k, v in record.items() if k != "peers"}),
                    flush=True,
                )
        if groups.rank == 0:
            report = dict(
                passed=True,
                d_in=args.d_in,
                widths=args.widths,
                batch_size=args.batch_size,
                cache_rows=args.cache_rows,
                gpu=torch.cuda.get_device_name(device),
                torch=torch.__version__,
                switches=records,
                validation="sampled canonical state, exact cache rows",
            )
            Path(args.report).write_text(json.dumps(report, indent=2))
    finally:
        groups.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
