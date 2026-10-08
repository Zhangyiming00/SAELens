"""Online multi-hook adapter; the single-hook producer/loader remain unchanged.

One queue item is one logical token batch for EVERY hook. Its physical payload
is hook-major [H * B, D]. Shuffle token rows jointly, never the flattened hook
rows; token budgets and watermarks count logical batches, not activation rows.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.distributed as dist

from sae_lens.training.dynamic_tp_input import DynamicTPInputLoader
from sae_lens.training.elastic_tp_config import (
    activation_dtype,
    emit,
    online_hooks,
    write_json,
)


def capture_multi_hook_chunk(model, tokens, args, sequence):
    from sae_lens.training.elastic_tp_producer import token_batch

    hooks = online_hooks(args)
    rows = args.batch_size // args.context
    batch = token_batch(tokens, sequence, rows)
    result = None
    for offset in range(0, rows, args.prompts):
        prompts = batch[offset : offset + args.prompts]
        _, cache = model.run_with_cache(prompts, hooks)
        count = len(prompts) * args.context
        for index, hook in enumerate(hooks):
            value = cache[hook]
            if value.shape[-1] != args.d_in or value.numel() != count * args.d_in:
                raise RuntimeError(
                    f"Invalid activation shape for {hook}: {tuple(value.shape)}; "
                    f"all elastic hooks must have d_in={args.d_in} and {count} token rows"
                )
            if result is None:
                result = torch.empty(
                    len(hooks) * args.batch_size,
                    args.d_in,
                    device=value.device,
                    dtype=getattr(torch, activation_dtype(args)),
                )
            start = index * args.batch_size + offset * args.context
            # Own each slice before the next capture can reuse its workspace.
            result[start : start + count].copy_(value.reshape(count, args.d_in))
    if args.validate_activations and not torch.isfinite(result).all():
        raise RuntimeError("Non-finite real multi-hook activation chunk")
    return result


class MultiHookTPInputLoader(DynamicTPInputLoader):
    """Use existing DMA ownership with one common token permutation per refill."""

    def __init__(self, buffer, *, hooks, batch_size, **kwargs):
        self.hooks, self.batch_size = tuple(hooks), batch_size
        if (
            len(self.hooks) < 2
            or buffer._chunk_size_tokens != len(self.hooks) * batch_size
        ):
            raise ValueError("Multi-hook SHM shape does not match hooks * batch_size")
        super().__init__(buffer, **kwargs)

    def load_raw(self, count, *, random_chunks=True):
        if self.closed:
            raise RuntimeError("input loader is closed")
        if not 1 <= count <= self.capacity_chunks:
            raise ValueError("refill exceeds input capacity")
        if self.copy_pending:
            self.copy_done.synchronize()
            self.copy_pending = False
        slots, _ = self.buffer.acquire_up_to(count, random=random_chunks)
        sequences, rows = [], 0
        try:
            if len(slots) != count:
                raise RuntimeError("refill count no longer matches READY slots")
            for slot in slots:
                valid = int(self.buffer._meta[slot, 0])
                if valid != self.buffer._chunk_size_tokens:
                    raise ValueError(
                        "multi-hook refill requires complete aligned chunks"
                    )
                sequences.append(int(self.buffer._meta[slot, 2]))
                self.buffer.copy_chunk_into(slot, self.host[rows : rows + valid])
                rows += valid
        finally:
            for slot in slots:
                self.buffer.release_chunk(slot)
        self.upload[:rows].copy_(self.host[:rows], non_blocking=True)
        if self.copy_done is not None:
            self.copy_done.record(torch.cuda.current_stream(self.device))
            self.copy_pending = True
        order = torch.randperm(count * self.batch_size, generator=self.generator)
        indices = order.to(self.device)
        # Index directly into the packed buffer, avoiding an extra H-sized
        # transpose/contiguous allocation. Each output independently owns data.
        packed = (
            indices // self.batch_size
        ) * self.buffer._chunk_size_tokens + indices % self.batch_size
        data = {
            h: self.upload.index_select(0, packed + i * self.batch_size)
            for i, h in enumerate(self.hooks)
        }
        return data, sequences, order

    def load(self, *args, **kwargs):
        raise NotImplementedError("Use load_raw and per-hook session input_scales")


def refill_multi_hook(session, args, loader, count, seen, scales, log, step):
    groups, hooks = session.groups, online_hooks(args)
    estimate = scales is None
    if groups.rank == 0:
        data, sequences, order = loader.load_raw(count)
        if estimate:
            scales = {}
            for h, value in data.items():
                norms = torch.cat(
                    [b.float().norm(dim=-1) for b in value.split(args.batch_size)]
                )
                mean = float(norms.mean())
                if not math.isfinite(mean) or mean <= 0:
                    raise ValueError(
                        f"Cannot estimate activation scale for {h}: mean norm={mean}"
                    )
                scales[h] = args.d_in**0.5 / mean
        assert len(set(sequences)) == count and not seen.intersection(sequences)
        seen.update(sequences)
        emit(
            log,
            "loaded",
            step=step,
            sequences=sequences,
            tokens=count * args.batch_size,
            activation_scale_by_hook=scales,
            input_dtype=activation_dtype(args),
            cache_dtype=str(data[hooks[0]].dtype),
            cache_scaled=False,
            hook_names=hooks,
            shuffle_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        )
    elif groups.rank in groups.active_ranks:
        data = {
            h: torch.empty(
                count * args.batch_size,
                args.d_in,
                device=groups.device,
                dtype=getattr(torch, activation_dtype(args)),
            )
            for h in hooks
        }
    if estimate:
        packet = [scales if groups.rank == 0 else None]
        dist.broadcast_object_list(packet, src=0, group=groups.control)
        scales = packet[0]
        session.input_scales = session.validate_input_scales(scales)
    if groups.rank in groups.active_ranks:
        for h in hooks:
            dist.broadcast(data[h], src=0, group=groups.active_group)
        session.stage_inputs(data)
    if args.audit_inputs:
        checks = groups.agree(multi_cache_digest(session, full=True))
        assert all(checks[r] == checks[0] for r in groups.active_ranks)
        if groups.rank == 0:
            emit(
                log,
                "input_verified",
                step=step,
                ranks=list(groups.active_ranks),
                hooks=checks[0],
            )
    return scales


def multi_cache_digest(session, *, full=False):
    from sae_lens.training.elastic_tp_trainer import cache_digest

    return {h: cache_digest(session, h, full=full) for h in session.configs}


def hook_step_metrics(session, losses, dead):
    metrics = {
        h: dict(
            loss=float(losses[h]),
            dead_before=dead[h],
            components={
                k: float(v) for k, v in session.last_loss_components[h].items()
            },
        )
        for h in session.configs
    }
    if not all(
        math.isfinite(m["loss"])
        and all(math.isfinite(v) for v in m["components"].values())
        for m in metrics.values()
    ):
        raise RuntimeError("Non-finite multi-hook SAE loss")
    return metrics


def save_multi_hook_models(session, output):
    """All active ranks participate in each TP gather, with distinct hook paths."""
    hooks = list(session.configs)
    paths = {
        h: f"{i:03d}_" + "".join(c if c.isalnum() else "_" for c in h)[:100]
        for i, h in enumerate(hooks)
    }
    for h, model in session.state.models.items():
        model.save_model(output / paths[h])
    if session.groups.rank == 0:
        write_json(
            output / "multi_sae_manifest.json",
            dict(
                format="multi_independent_sae_v1",
                hook_names=hooks,
                hook_to_dir=paths,
                shared_hyperparams=True,
                sae_dp_mode="ddp",
                seed_mode="offset",
                activation_scale_by_hook=session.input_scales,
            ),
        )
