"""In-memory, step-boundary TP switching for one SAE replica (possibly H hooks).

The worker pool is fixed, TP membership is not. A prepared communicator and the
active communicator coexist; WORLD, the process, and input provider stay alive.
This is deliberately separate from the existing DP/ZeRO role-switch protocol.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from sae_lens.tp_layout import FeatureLayout, plan_adjacent
from sae_lens.training.activation_scaler import prepare_sae_input


class TPGroupPair:
    """All WORLD ranks call prepare/commit in identical order, including idle ones.

    The pool controls eligibility; WORLD also contains passive coordinators.
    No background NCCL creation thread races steady-state collectives.
    prepare() may be called several completed steps before switch().
    """

    def __init__(self, pool, active, *, device, backend="nccl", timeout_s=180):
        self.pool = tuple(pool)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        self.backend = backend
        self.timeout = timedelta(seconds=timeout_s)
        if tuple(sorted(set(self.pool))) != self.pool or not self.pool:
            raise ValueError("Pool ranks must be sorted, unique and nonempty")
        self.control_ranks = tuple(range(dist.get_world_size()))
        if not set(self.pool) <= set(self.control_ranks):
            raise ValueError("Pool must be a subset of WORLD")
        self.control = dist.new_group(
            list(self.control_ranks), backend="gloo", timeout=self.timeout
        )
        self.transfer = dist.new_group(
            list(self.pool),
            backend=backend,
            timeout=self.timeout,
            # Eager NCCL binding lets P2P reuse this pool communicator instead
            # of bootstrapping a lazy pair communicator during the pause.
            device_id=self.device if backend == "nccl" else None,
        )
        self.rank = dist.get_rank()
        self.active_ranks = ()
        self.active_group = None
        self.prepared_ranks = None
        self.prepared_group = None
        self.spare_ranks = None
        self.spare_group = None
        self.epoch = 0
        self.failed = False
        if self.rank in self.pool:
            # Eagerly initialize the transfer communicator, including idle ranks.
            token = torch.zeros(1, device=self.device)
            dist.all_reduce(token, group=self.transfer)
            self.synchronize()
        self.prepare(tuple(active), initial=True)
        self.commit()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def agree(self, value):
        records = [None] * len(self.control_ranks)
        dist.all_gather_object(records, value, group=self.control)
        return records

    def check(self, error):
        errors = self.agree(error)
        if any(x is not None for x in errors):
            raise RuntimeError(f"Dynamic TP preflight failed: {errors}")

    def prepare(self, ranks, *, initial=False):
        if self.failed:
            raise RuntimeError("A failed transfer invalidated this session")
        ranks = tuple(ranks)
        records = self.agree(ranks)
        if any(r != ranks for r in records):
            raise ValueError("Pool members requested different target memberships")
        if (
            not ranks
            or tuple(sorted(set(ranks))) != ranks
            or not set(ranks) <= set(self.pool)
        ):
            raise ValueError(
                "Target ranks must be sorted members of the live worker pool"
            )
        if not initial and (
            abs(len(ranks) - len(self.active_ranks)) != 1
            or not (
                set(ranks) < set(self.active_ranks)
                or set(self.active_ranks) < set(ranks)
            )
        ):
            raise ValueError("A switch must add or remove exactly one member")
        if self.prepared_ranks is not None:
            if self.prepared_ranks == ranks:
                return
            raise RuntimeError("A different topology is already prepared")
        self.synchronize()
        dist.barrier(group=self.control)
        if self.spare_ranks == ranks:
            self.prepared_group = self.spare_group
        else:
            if self.spare_ranks is not None and self.rank in self.spare_ranks:
                dist.destroy_process_group(self.spare_group)
            dist.barrier(group=self.control)
            self.prepared_group = dist.new_group(
                list(ranks),
                backend=self.backend,
                timeout=self.timeout,
                # Use the globally ordered group counter. In PyTorch 2.10 the
                # local-sync name hashes len(pg_names), which differs on members
                # and nonmembers of earlier groups and deadlocks a later join.
                use_local_synchronization=False,
                device_id=self.device if self.backend == "nccl" else None,
            )
        self.spare_group, self.spare_ranks = None, None
        self.prepared_ranks = ranks
        if self.rank in ranks:
            token = torch.zeros(1, device=self.device)
            dist.all_reduce(token, group=self.prepared_group)
        self.synchronize()
        dist.barrier(group=self.control)

    def commit(self):
        if self.prepared_ranks is None:
            raise RuntimeError("No prepared TP topology")
        self.synchronize()
        dist.barrier(group=self.control)
        old_group, old_ranks = self.active_group, self.active_ranks
        self.active_group, self.active_ranks = self.prepared_group, self.prepared_ranks
        self.prepared_group, self.prepared_ranks = None, None
        self.epoch += 1
        # Keep exactly two training communicator slots. Destruction/rebuilding
        # belongs to prepare(), before the pause; a reverse switch reuses it.
        self.spare_group = old_group
        self.spare_ranks = old_ranks or None
        dist.barrier(group=self.control)

    def close(self):
        self.synchronize()
        dist.barrier(group=self.control)
        if self.prepared_ranks and self.rank in self.prepared_ranks:
            dist.destroy_process_group(self.prepared_group)
        if self.spare_ranks and self.rank in self.spare_ranks:
            dist.destroy_process_group(self.spare_group)
        if self.rank in self.active_ranks:
            dist.destroy_process_group(self.active_group)
        if self.rank in self.pool:
            dist.destroy_process_group(self.transfer)
        dist.destroy_process_group(self.control)


@dataclass
class TPTrainState:
    models: dict[str, Any] = field(default_factory=dict)
    optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    # Counters use canonical feature-ID order, not rank concatenation.
    # Input caches have their own multi-donor migration protocol below.
    replicated: dict[str, torch.Tensor] = field(default_factory=dict)
    # Identical, unconsumed [tokens, d_in] inputs on each active TP rank.
    # On growth each old owner supplies a disjoint row interval to the joiner.
    activation_caches: dict[str, torch.Tensor] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    in_step: bool = False
    retired: bool = False


def _schema(state):
    hooks = {}
    for hook, model in state.models.items():
        opt = state.optimizers[hook]
        if type(opt) is not torch.optim.Adam or len(opt.param_groups) != 1:
            raise ValueError(
                "Dynamic TP requires one ordinary torch Adam group per hook; DP/ZeRO is not supported"
            )
        if getattr(model, "parallel_context", None) is not None:
            raise ValueError(
                "Do not migrate a model owned by a fixed SAERuntime/DDP reducer"
            )
        params = dict(model.named_parameters())
        if set(params) != set(model._tp_param_shard_dims()):
            raise ValueError("Unregistered parameter axes")
        if {id(p) for p in opt.param_groups[0]["params"]} != {
            id(p) for p in params.values()
        }:
            raise ValueError("Optimizer does not own precisely this hook's parameters")
        if any(p.grad is not None or hasattr(p, "main_grad") for p in params.values()):
            raise ValueError(
                "Switch after optimizer.step() and zero_grad(set_to_none=True), without DDP main_grad"
            )
        options = {
            k: copy.deepcopy(v) for k, v in opt.param_groups[0].items() if k != "params"
        }
        if any(torch.is_tensor(v) for v in options.values()):
            raise ValueError(
                "Tensor-valued optimizer options must be normalized before switching"
            )
        tensors = {}
        for name, p in params.items():
            saved = {}
            for key, value in opt.state.get(p, {}).items():
                if key == "step":
                    saved[key] = (
                        "scalar",
                        float(value),
                        str(value.dtype) if torch.is_tensor(value) else None,
                        value.device.type if torch.is_tensor(value) else None,
                    )
                elif torch.is_tensor(value) and value.shape == p.shape:
                    saved[key] = ("moment", str(value.dtype))
                else:
                    raise ValueError(f"Unsupported Adam state {name}.{key}")
            axis = model._tp_param_shard_dims()[name]
            logical_shape = list(p.shape)
            if axis is not None:
                logical_shape[axis] = model.cfg.d_sae
            tensors[name] = (tuple(logical_shape), str(p.dtype), axis, saved)
        hooks[hook] = (options, tensors)
    replicas = {}
    for key, value in state.replicated.items():
        if not value.is_contiguous() or value.requires_grad:
            raise ValueError(
                "Replicated state must contain detached contiguous tensors"
            )
        replicas[key] = (tuple(value.shape), str(value.dtype))
    caches = {}
    for key, value in state.activation_caches.items():
        if value.ndim != 2 or not value.is_contiguous() or value.requires_grad:
            raise ValueError(
                "Activation caches must be detached contiguous [tokens, d_in] tensors"
            )
        caches[key] = (tuple(value.shape), str(value.dtype))
    return hooks, replicas, caches, copy.deepcopy(state.progress)


def _dtype(name):
    return getattr(torch, name.removeprefix("torch."))


@torch.no_grad()
def migrate_tensor(source, target, *, plan, axis, group, rank, chunk_bytes=16 << 20):
    """Copy retained data locally and move only ownership-changing slices.

    All pool ranks call in one deterministic tensor/move/chunk order. Pairwise
    send/recv never form a dependency cycle. Sender and receiver allocate only
    one contiguous chunk. Output storage is allocated before entering here.
    """
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    if source is not None and target is not None:
        if axis is None:
            target.copy_(source)
        else:
            keep = min(source.shape[axis], target.shape[axis])
            target.narrow(axis, 0, keep).copy_(source.narrow(axis, 0, keep))
    if axis is None:
        # Replicated b_dec/counters/input buffers need only reach a joining rank.
        newcomers = set(plan.new.ranks) - set(plan.old.ranks)
        transfers = [(plan.old.ranks[0], r, 0, 0, None) for r in sorted(newcomers)]
    else:
        transfers = [
            (m.source, m.target, m.start, m.destination, m.length) for m in plan.moves
        ]
    for sender, receiver, start, destination, length in transfers:
        if rank not in (sender, receiver):
            continue
        tensor = source if rank == sender else target
        if tensor is None:
            raise RuntimeError("Missing migration endpoint tensor")
        if axis is None:
            flat = tensor.view(-1)
            step = max(1, chunk_bytes // tensor.element_size())
            pieces = (
                (flat.narrow(0, i, min(step, flat.numel() - i)),)
                for i in range(0, flat.numel(), step)
            )
        else:
            per_feature = tensor.numel() // tensor.shape[axis]
            step = max(1, chunk_bytes // (per_feature * tensor.element_size()))
            base = start if rank == sender else destination
            pieces = (
                (tensor.narrow(axis, base + i, min(step, length - i)),)
                for i in range(0, length, step)
            )
        for (view,) in pieces:
            if rank == sender:
                dist.send(view.contiguous(), dst=receiver, group=group)
            else:
                scratch = torch.empty(view.shape, device=view.device, dtype=view.dtype)
                dist.recv(scratch, src=sender, group=group)
                view.copy_(scratch)


@torch.no_grad()
def migrate_activation_cache(source, target, *, plan, group, rank, chunk_bytes):
    """Assemble a replicated input on a joiner from all old owners in row order.

    Survivors keep the same storage. A departing rank sends no activations:
    every survivor already has every unconsumed row. Short caches may assign
    zero rows to some donors. This is not a token-sharded reservoir protocol.
    """
    newcomers = set(plan.new.ranks) - set(plan.old.ranks)
    if not newcomers:
        return
    receiver = next(iter(newcomers))
    if rank not in plan.old.ranks and rank != receiver:
        return
    tensor = target if rank == receiver else source
    rows, width = tensor.shape
    step = max(1, chunk_bytes // max(1, width * tensor.element_size()))
    q, remainder = divmod(rows, len(plan.old.ranks))
    start = 0
    for i, sender in enumerate(plan.old.ranks):
        count = q + (i < remainder)
        if rank in (sender, receiver):
            for offset in range(start, start + count, step):
                view = tensor.narrow(0, offset, min(step, start + count - offset))
                if rank == sender:
                    dist.send(view, dst=receiver, group=group)
                else:
                    dist.recv(view, src=sender, group=group)
        start += count


def switch_tp(
    state: TPTrainState,
    groups: TPGroupPair,
    layouts,
    *,
    build_model,
    chunk_bytes=16 << 20,
):
    """Transactional staged migration. Return (new state, layouts, metrics).

    build_model(hook, group, layout) MUST create an uninitialized model without
    collectives/DDP hooks. Allocation/schema errors are voted on before any
    tensor transfer; old state remains usable. A transport failure is fatal
    (no claim of failure recovery or rollback after a broken communicator).
    """
    started = time.perf_counter()
    rank = groups.rank
    old = rank in groups.active_ranks
    new = groups.prepared_ranks is not None and rank in groups.prepared_ranks
    error, metadata = None, None
    try:
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        if state.retired or state.in_step:
            raise RuntimeError("An active autograd/update window cannot migrate")
        if groups.prepared_ranks is None:
            raise RuntimeError("Call prepare(target_ranks) before switching")
        if not layouts:
            raise ValueError("At least one hook layout is required")
        # Drain BEFORE reading Adam step/options: an optimizer stream may still
        # be updating these scalar tensors after the Python step call returns.
        groups.synchronize()
        if old:
            metadata = _schema(state)
            if any(
                t.device != groups.device
                for t in (*state.replicated.values(), *state.activation_caches.values())
            ):
                raise ValueError("Migrated tensors must be on the session device")
        plans = {
            h: plan_adjacent(layout, groups.prepared_ranks)
            for h, layout in layouts.items()
        }
        if any(layout.ranks != groups.active_ranks for layout in layouts.values()):
            raise ValueError("Layout membership differs from the active communicator")
        if old and set(state.models) != set(layouts):
            raise ValueError("Model hooks and ownership layouts differ")
        for h in state.models:
            if state.models[h].feature_shard.layout != layouts[h]:
                raise ValueError(
                    "Model feature ownership differs from switch descriptor"
                )
    except Exception as exc:
        error = repr(exc)
    groups.check(error)
    signature = hashlib.sha256(
        repr(
            (chunk_bytes, [(h, p.old, p.new, p.moves) for h, p in plans.items()])
        ).encode()
    ).hexdigest()
    signatures = groups.agree(signature)
    groups.check(
        None
        if len(set(signatures)) == 1
        else "Feature maps, hook order or chunk size disagree"
    )
    # Metadata describes completed GPU updates, never an enqueued Adam step.
    records = groups.agree(metadata)
    canonical = records[groups.control_ranks.index(groups.active_ranks[0])]
    groups.check(
        None
        if all(r is None or r == canonical for r in records)
        else "Adam schemas/steps, counters or optimizer options disagree"
    )
    hooks, replicas, caches, progress = canonical
    preflight_s = time.perf_counter() - started
    allocation_started = time.perf_counter()
    target = TPTrainState(progress=copy.deepcopy(progress))
    error = None
    try:
        if new:
            for h, (options, tensors) in hooks.items():
                model = build_model(h, groups.prepared_group, plans[h].new)
                if model.feature_shard.layout != plans[h].new:
                    raise ValueError(
                        "Builder did not install the planned feature ownership"
                    )
                # Constructor options set optimizer-wide fused/AMP flags too;
                # changing only param_groups leaves a different Adam runtime.
                adam_options = inspect.signature(torch.optim.Adam).parameters
                opt = torch.optim.Adam(
                    model.parameters(),
                    **{
                        k: copy.deepcopy(v)
                        for k, v in options.items()
                        if k in adam_options
                    },
                )
                opt.param_groups[0].update(copy.deepcopy(options))
                target.models[h], target.optimizers[h] = model, opt
                if set(dict(model.named_parameters())) != set(tensors):
                    raise ValueError("Builder changed the parameter schema")
                for name, (logical_shape, dtype, axis, saved) in tensors.items():
                    p = model.get_parameter(name)
                    expected_shape = list(logical_shape)
                    if axis is not None:
                        expected_shape[axis] = model.feature_shard.width
                    if (
                        tuple(p.shape) != tuple(expected_shape)
                        or str(p.dtype) != dtype
                        or p.device != groups.device
                    ):
                        raise ValueError(
                            f"Builder changed shape/dtype/device for {h}.{name}"
                        )
                    if saved:
                        opt.state[p] = {}
                    for key, descriptor in saved.items():
                        if descriptor[0] == "scalar":
                            _, value, dtype, device = descriptor
                            opt.state[p][key] = (
                                value
                                if dtype is None
                                else torch.tensor(
                                    value,
                                    dtype=_dtype(dtype),
                                    device=p.device if device == "cuda" else "cpu",
                                )
                            )
                        else:
                            opt.state[p][key] = torch.empty_like(
                                p, dtype=_dtype(descriptor[1])
                            )
            target.replicated = {
                key: (
                    state.replicated[key]
                    if old
                    else torch.empty(shape, dtype=_dtype(dtype), device=groups.device)
                )
                for key, (shape, dtype) in replicas.items()
            }
            target.activation_caches = {
                key: state.activation_caches[key]
                if old
                else torch.empty(shape, dtype=_dtype(dtype), device=groups.device)
                for key, (shape, dtype) in caches.items()
            }
    except Exception as exc:
        error = repr(exc)
    # Allocation (including Adam) completes everywhere before sender launches.
    groups.check(error)
    groups.synchronize()
    allocation_s = time.perf_counter() - allocation_started
    dist.barrier(group=groups.control)
    transfer_started = time.perf_counter()
    try:
        for h, (_, tensors) in hooks.items():
            for name, (_, _, axis, saved) in tensors.items():
                src = state.models[h].get_parameter(name) if old else None
                dst = target.models[h].get_parameter(name) if new else None
                migrate_tensor(
                    src,
                    dst,
                    plan=plans[h],
                    axis=axis,
                    group=groups.transfer,
                    rank=rank,
                    chunk_bytes=chunk_bytes,
                )
                for key, descriptor in saved.items():
                    if descriptor[0] != "moment":
                        continue
                    migrate_tensor(
                        state.optimizers[h].state[src][key] if old else None,
                        target.optimizers[h].state[dst][key] if new else None,
                        plan=plans[h],
                        axis=axis,
                        group=groups.transfer,
                        rank=rank,
                        chunk_bytes=chunk_bytes,
                    )
        representative = next(iter(plans.values()))
        for key in replicas:
            migrate_tensor(
                state.replicated.get(key),
                target.replicated.get(key),
                plan=representative,
                axis=None,
                group=groups.transfer,
                rank=rank,
                chunk_bytes=chunk_bytes,
            )
        for key in caches:
            migrate_activation_cache(
                state.activation_caches.get(key),
                target.activation_caches.get(key),
                plan=representative,
                group=groups.transfer,
                rank=rank,
                chunk_bytes=chunk_bytes,
            )
        groups.synchronize()
        dist.barrier(group=groups.control)
        transfer_s = time.perf_counter() - transfer_started
        commit_started = time.perf_counter()
        groups.commit()
    except BaseException:
        groups.failed = True
        raise
    # Break references only AFTER all destinations have acknowledged completion.
    state.models.clear()
    state.optimizers.clear()
    state.replicated.clear()
    state.activation_caches.clear()
    state.retired = True
    commit_s = time.perf_counter() - commit_started
    transferred_bytes = 0
    growing = len(representative.new.ranks) > len(representative.old.ranks)
    for h, (_, tensors) in hooks.items():
        for shape, dtype, axis, saved in tensors.values():
            elements = (
                math.prod(shape) * int(growing)
                if axis is None
                else math.prod(shape) // plans[h].old.total * plans[h].moved_features
            )
            transferred_bytes += elements * _dtype(dtype).itemsize
            transferred_bytes += sum(
                elements * _dtype(d[1]).itemsize
                for d in saved.values()
                if d[0] == "moment"
            )
    if growing:
        transferred_bytes += sum(
            math.prod(shape) * _dtype(dtype).itemsize
            for shape, dtype in (*replicas.values(), *caches.values())
        )
    metrics = dict(
        epoch=groups.epoch,
        pause_s=time.perf_counter() - started,
        moved_features={h: p.moved_features for h, p in plans.items()},
        old_ranks=list(representative.old.ranks),
        new_ranks=list(representative.new.ranks),
        state_storage="device",
        checkpoint_io=False,
        transferred_bytes=transferred_bytes,
        phases_s=dict(
            preflight=preflight_s,
            allocation=allocation_s,
            transfer=transfer_s,
            commit=commit_s,
        ),
    )
    return target, {h: p.new for h, p in plans.items()}, metrics


class DynamicTPSession:
    """Native MegatronTopKSAE + FP32 Adam, one DP replica, multiple hooks.

    Public boundary: submit replicated [tokens,d_in] batches to train_step;
    call prepare/switch on ALL WORLD workers. An external input provider stays
    outside this object and can continue producing during parameter migration.
    No pending backward or partial gradient accumulation is accepted at switch.
    """

    def __init__(
        self,
        configs,
        groups,
        *,
        lr=3e-4,
        seed=42,
        dead_feature_window=1000,
        max_grad_norm=1.0,
        adam_kwargs=None,
        tp_overlap="off",
        tp_overlap_max_live_hooks=2,
        input_scale=1.0,
    ):
        from sae_lens.megatron_tp import require_megatron_core
        from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE

        self.configs, self.groups = copy.deepcopy(configs), groups
        if not configs:
            raise ValueError("At least one hook config is required")
        # Warm Python/native dependencies on idle workers at startup too. Their
        # first actual model/Adam must not import Megatron/Dynamo in switch().
        require_megatron_core()
        torch.optim.Adam(
            [torch.nn.Parameter(torch.empty(0, device=groups.device))],
            lr=lr,
            **(adam_kwargs or {}),
        )
        self.dead_feature_window = dead_feature_window
        self.max_grad_norm = max_grad_norm
        self.input_scale = float(input_scale)
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("input_scale must be finite and positive")
        if tp_overlap not in ("off", "eager", "lazy", "bounded"):
            raise ValueError("tp_overlap must be off/eager/lazy/bounded")
        if type(tp_overlap_max_live_hooks) is not int or tp_overlap_max_live_hooks < 1:
            raise ValueError("tp_overlap_max_live_hooks must be a positive integer")
        self.tp_overlap = tp_overlap
        self.tp_overlap_max_live_hooks = tp_overlap_max_live_hooks
        self.last_forward_schedule = "off"
        settings = (tp_overlap, tp_overlap_max_live_hooks, self.input_scale)
        groups.check(
            None
            if all(s == settings for s in groups.agree(settings))
            else "TP overlap/input scale settings disagree"
        )
        self.layouts = {
            h: FeatureLayout.balanced(c.d_sae, groups.active_ranks)
            for h, c in configs.items()
        }
        self.state = TPTrainState(progress=dict(steps=0, tokens=0))
        for i, (h, cfg) in enumerate(self.configs.items()):
            if cfg.dtype != "float32" or cfg.normalize_activations != "none":
                raise ValueError(
                    "Dynamic session expects FP32 SAE with normalize_activations=none; use input_scale for raw inputs"
                )
            if cfg.topk_tie_policy != "stable_id" or cfg.topk_backend == "legacy":
                raise ValueError(
                    "Dynamic session requires stable_id and a sharded TopK backend"
                )
            cfg.device = str(groups.device)
            if groups.rank in groups.active_ranks:
                devices = [groups.device] if groups.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + i)
                    model = MegatronTopKSAE(cfg, tp_group=groups.active_group)
                kwargs = dict(lr=lr, **(adam_kwargs or {}))
                self.state.models[h] = model
                self.state.optimizers[h] = torch.optim.Adam(
                    model.parameters(), **kwargs
                )
                self.state.replicated[h + "/since_fired"] = torch.zeros(
                    cfg.d_sae, device=groups.device, dtype=torch.long
                )
                self.state.replicated[h + "/firing_counts"] = torch.zeros(
                    cfg.d_sae, device=groups.device
                )

    def prepare(self, ranks):
        ranks = tuple(ranks)
        error = None
        try:
            if self.state.in_step or self.state.retired or self.groups.failed:
                raise RuntimeError("Prepare requires a usable optimizer boundary")
            for layout in self.layouts.values():
                plan_adjacent(layout, ranks)
        except Exception as exc:
            error = repr(exc)
        self.groups.check(error)
        self.groups.prepare(ranks)

    def stage_inputs(self, batches):
        """Attach identical pending input rows on each active TP rank."""
        if self.state.in_step or self.state.retired or self.groups.failed:
            raise RuntimeError(
                "Inputs can only be staged at a usable optimizer boundary"
            )
        if self.groups.rank not in self.groups.active_ranks:
            return
        if any(t.shape[0] for t in self.state.activation_caches.values()):
            raise RuntimeError("Consume the pending cache before staging more inputs")
        if set(batches) != set(self.configs):
            raise ValueError("A pending input tensor for every hook is required")
        sizes = set()
        for h, tensor in batches.items():
            if (
                tensor.ndim != 2
                or tensor.shape[1] != self.configs[h].d_in
                or tensor.device != self.groups.device
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or tensor.dtype not in (torch.float32, torch.bfloat16)
            ):
                raise ValueError(
                    "Stage contiguous, detached FP32/BF16 [tokens, d_in] inputs on the session device"
                )
            sizes.add(tensor.shape[0])
        if len(sizes) != 1 or min(sizes) < 1:
            raise ValueError(
                "All hooks require the same positive number of cached rows"
            )
        self.state.activation_caches = dict(batches)

    def train_cached_step(self, batch_size):
        """Consume exactly one batch; a switch never changes this row cursor."""
        if self.groups.rank not in self.groups.active_ranks:
            return {}
        cache = self.state.activation_caches
        if (
            batch_size < 1
            or set(cache) != set(self.configs)
            or any(t.shape[0] < batch_size for t in cache.values())
        ):
            raise ValueError("The pending cache must contain a full positive batch")
        result = self.train_step({h: t[:batch_size] for h, t in cache.items()})
        self.state.activation_caches = {h: t[batch_size:] for h, t in cache.items()}
        return result

    def switch(self):
        from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE

        def build(hook, group, layout):
            return MegatronTopKSAE(
                self.configs[hook],
                tp_group=group,
                initialize=False,
                feature_layout=layout,
            )

        self.state, self.layouts, metrics = switch_tp(
            self.state, self.groups, self.layouts, build_model=build
        )
        return metrics

    def save_checkpoint(self, path, *, provider_state=None):
        """Save a complete boundary; caller supplies its external provider cursor/RNG."""
        from sae_lens.training.dynamic_tp_checkpoint import save_checkpoint

        return save_checkpoint(self, path, provider_state=provider_state)

    def load_checkpoint(self, path):
        """Restore into current TP membership and return the external provider state."""
        from sae_lens.training.dynamic_tp_checkpoint import load_checkpoint

        return load_checkpoint(self, path)

    def train_step(self, batches):
        from sae_lens.saes.sae import TrainStepInput

        if self.state.in_step or self.state.retired or self.groups.failed:
            raise RuntimeError("Session is not at a usable optimizer boundary")
        if self.groups.rank not in self.groups.active_ranks:
            return {}
        if set(batches) != set(self.state.models):
            raise ValueError("A batch for each hook is required")
        sizes = {x.shape[0] for x in batches.values()}
        if len(sizes) != 1 or min(sizes) < 1:
            raise ValueError("All hooks require the same positive token count")
        self.state.in_step = True
        inputs = {}
        for h, model in self.state.models.items():
            self.state.optimizers[h].zero_grad(set_to_none=True)
            mask = self.state.replicated[h + "/since_fired"] > self.dead_feature_window
            inputs[h] = TrainStepInput(
                sae_in=prepare_sae_input(batches[h], model.dtype, self.input_scale),
                coefficients={},
                dead_neuron_mask=mask,
                n_training_steps=self.state.progress["steps"],
                is_logging_step=False,
            )
        result = {}
        models = self.state.models
        enabled = self.tp_overlap != "off" and all(
            m.tp_wavefront_supported() for m in models.values()
        )
        self.last_forward_schedule = self.tp_overlap if enabled else "off"

        def update(h, output):
            model = models[h]
            with torch.no_grad():
                if self.max_grad_norm is not None:
                    model.clip_grad_norm_(self.max_grad_norm)
                self.state.optimizers[h].step()
                self.state.optimizers[h].zero_grad(set_to_none=True)
                counts = output.feature_firing_counts
                self.state.replicated[h + "/firing_counts"].add_(counts)
                since = self.state.replicated[h + "/since_fired"]
                since.add_(1)
                since.masked_fill_(counts > 0, 0)
                result[h] = output.loss.detach()

        if enabled and self.tp_overlap != "eager":
            from sae_lens.training.multi_hook_sae import PendingWavefrontOutputs

            pending = PendingWavefrontOutputs(
                list(models),
                models,
                inputs,
                max_live_hooks=0
                if self.tp_overlap == "lazy"
                else self.tp_overlap_max_live_hooks,
            )
            for h in models:
                output = pending.pop(h)
                output.loss.backward()
                update(h, output)
                del output
            assert not pending
        else:
            if enabled:
                from sae_lens.training.multi_hook_sae import forward_tp_wavefront

                outputs = forward_tp_wavefront(list(models), models, inputs)
            else:
                outputs = {h: model(inputs[h]) for h, model in models.items()}
            sum(o.loss for o in outputs.values()).backward()
            for h, output in outputs.items():
                update(h, output)
            del outputs
        self.state.progress["steps"] += 1
        self.state.progress["tokens"] += sizes.pop()
        self.state.in_step = False
        return result
