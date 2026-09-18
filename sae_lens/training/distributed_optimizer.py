"""FP32 native Megatron shards with device clipping and explicit ownership.

The installed DistributedOptimizer owns buffer partitioning, prepare, Adam and
parameter all-gather. This adapter exposes only valid parameter intersections;
padding and unreduced regions never enter normalization, AMP or clipping.
"""

import torch
import torch.distributed as dist
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from sae_lens.training.gpu_clip_optimizer import clip_grads_on_device


def is_distributed_optimizer(optimizer):
    return bool(getattr(optimizer, "_sae_distributed_optimizer", False))


class GPUClipDistributedOptimizer(DistributedOptimizer):
    _sae_distributed_optimizer = True

    def bind_model(self, model, runtime):
        self.sae_model = model
        self.sae_runtime = runtime
        self.sae_device = next(model.parameters()).device
        names = {p: n for n, p in model.named_parameters()}
        self.sae_shards = {}
        for models, shards in zip(
            self.model_fp32_groups, self.shard_fp32_groups, strict=True
        ):
            for full, shard in zip(models, shards, strict=True):
                if full not in names or names[full] in self.sae_shards:
                    raise ValueError(
                        "Distributed optimizer has duplicate or cross-hook ownership"
                    )
                r = self._get_model_param_range_map(full)["param"]
                if not 0 <= r.start < r.end <= full.numel() or shard.numel() != r.size:
                    raise ValueError("Invalid native parameter shard range")
                expected = full.view(-1)[r.start : r.end]
                if expected.data_ptr() != shard.data_ptr():
                    raise ValueError(
                        "Native FP32 optimizer shard must alias its model parameter"
                    )
                self.sae_shards[names[full]] = (full, shard, r.start, r.end)
        owned = [id(p) for g in self.param_groups for p in g["params"]]
        if len(owned) != len(set(owned)) or set(owned) != {
            id(s) for _, s, _, _ in self.sae_shards.values()
        }:
            raise ValueError(
                "Optimizer parameter groups differ from native shard ownership"
            )
        # Setup-only CPU metadata audit: every element is owned exactly once
        # within the hook's explicit DP group, including empty intersections.
        records = [None] * runtime.require_local().dp_group.size()
        dist.all_gather_object(
            records,
            {n: (a, b) for n, (_, _, a, b) in self.sae_shards.items()},
            group=runtime.require_local().dp_group,
        )
        for full, name in names.items():
            cursor = 0
            for a, b in sorted(record[name] for record in records if name in record):
                if a != cursor:
                    raise ValueError(f"Missing or duplicate shard ownership for {name}")
                cursor = b
            if cursor != full.numel():
                raise ValueError(f"Incomplete shard ownership for {name}")

    def expose_valid_grads(self):
        # FP32 .float() in native prepare_grads is also a view. Exposing that
        # view before external GradScaler unscale keeps native step intact and
        # calls native prepare only once. No second scaler lives in Megatron.
        for full, shard, start, end in self.sae_shards.values():
            shard.grad = full.main_grad.view(-1)[start:end]

    def shard_signature(self):
        return {
            "ranges": {
                n: (a, b, tuple(p.shape)) for n, (p, _, a, b) in self.sae_shards.items()
            },
            "buffers": [
                [b.grad_data.numel() for b in buf.buckets] for buf in self.buffers
            ],
        }

    def local_checkpoint(self):
        # Native state_dict omits moments. The full inner Adam state is the
        # native optimizer's *local* shard state, not full-model indexing.
        return dict(
            version=1,
            signature=self.shard_signature(),
            adam=self.optimizer.state_dict(),
        )

    def load_local_checkpoint(self, state):
        if state["version"] != 1 or state["signature"] != self.shard_signature():
            raise ValueError(
                "Distributed optimizer checkpoint requires identical topology and bucket layout"
            )
        live_groups = self.optimizer.param_groups
        self.optimizer.load_state_dict(self._adapt_local_adam_state(state["adam"]))
        for live, loaded in zip(live_groups, self.optimizer.param_groups, strict=True):
            live.clear()
            live.update(loaded)
        self.optimizer.param_groups = live_groups

    def _adapt_local_adam_state(self, state):
        # Torch stores a tensor step per parameter; TE/Apex store one integer
        # per group. Conversion happens only at the checkpoint boundary.
        # Keep moments untouched and reject incompatible per-parameter clocks.
        group_step = hasattr(self.optimizer, "adam_w_mode")
        states = {key: dict(value) for key, value in state["state"].items()}
        groups = []
        for live, saved in zip(
            self.optimizer.param_groups, state["param_groups"], strict=True
        ):
            group = {**{k: v for k, v in live.items() if k != "params"}, **saved}
            if group_step:
                steps = {
                    int(states[p]["step"])
                    for p in saved["params"]
                    if "step" in states.get(p, {})
                }
                if len(steps) > 1:
                    raise ValueError(
                        "FusedAdam requires a common step within each SAE group"
                    )
                if steps:
                    group["step"] = steps.pop()
                for p in saved["params"]:
                    if p in states:
                        states[p].pop("step", None)
            elif "step" in saved:
                for p in saved["params"]:
                    if p in states:
                        states[p]["step"] = torch.tensor(float(saved["step"]))
                group.pop("step", None)
            groups.append(group)
        return {**state, "state": states, "param_groups": groups}

    def prepare_full_parameter_state(self, states):
        """Restore FusedAdam's shared clock, including ranks with no local shard."""
        if not hasattr(self.optimizer, "adam_w_mode"):
            return
        steps = {int(s["step"]) for s in states.values() if "step" in s}
        if len(steps) > 1:
            raise ValueError("FusedAdam requires a common step within each SAE group")
        self.optimizer.param_groups[0]["step"] = steps.pop() if steps else 0

    def load_full_parameter_state(self, parameter, state):
        """Migrate existing named FP32 Adam moments after existing TP conversion."""
        for full, shard, start, end in self.sae_shards.values():
            if full is parameter:
                self.optimizer.state[shard] = {
                    k: (
                        v.reshape(-1)[start:end].clone()
                        if torch.is_tensor(v) and v.ndim
                        else v.detach().clone().to(shard.device)
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in state.items()
                    if k != "step" or not hasattr(self.optimizer, "adam_w_mode")
                }
                return

    @torch.no_grad()
    def clip_grad_norm(self, clip_grad):
        return clip_grads_on_device(
            self.get_main_grads_for_grad_norm(),
            [p.grad for p in self.get_parameters() if p.grad is not None],
            device=self.sae_device,
            group=self.get_grad_stats_parallel_group(), max_norm=clip_grad,
        )
