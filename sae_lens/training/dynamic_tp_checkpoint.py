"""Complete dynamic-session checkpoints, independent of saved TP membership.

Files are written only on an explicit save, never during a TP switch. Shared
storage must be visible to every WORLD worker. External provider state belongs
to the caller; pass its cursor, permutation/RNG and scaling as provider_state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from sae_lens.tp_layout import FeatureLayout


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _config(config):
    return {k: v for k, v in config.to_dict().items() if k != "device"}


def _boundary(session, path):
    groups = session.groups
    error = None
    if session.state.in_step or session.state.retired or groups.failed:
        error = "Checkpoint requires a usable completed optimizer boundary"
    groups.check(error)
    paths = groups.agree(str(Path(path).resolve()))
    groups.check(None if len(set(paths)) == 1 else "Checkpoint paths disagree")
    groups.synchronize()
    return Path(paths[0])


def save_checkpoint(session, path, *, provider_state=None):
    from sae_lens.training.dynamic_tp import _schema

    path = _boundary(session, path)
    groups, state = session.groups, session.state
    stage = path.with_name(path.name + ".incomplete")
    schema, error = None, None
    try:
        if groups.rank in groups.active_ranks:
            schema = _schema(state)
    except Exception as exc:
        error = repr(exc)
    groups.check(error)
    schemas = groups.agree(schema)
    canonical = schemas[groups.active_ranks[0]]
    groups.check(
        None
        if all(s is None or s == canonical for s in schemas)
        else "Checkpoint schemas disagree"
    )
    error = None
    if groups.rank == 0:
        try:
            if path.exists() or stage.exists():
                raise FileExistsError("Checkpoint destination already exists")
            stage.mkdir(parents=True)
        except Exception as exc:
            error = repr(exc)
    groups.check(error)
    written, error = [], None
    try:
        if groups.rank in groups.active_ranks:
            hooks = {}
            for h, model in state.models.items():
                optimizer = state.optimizers[h]
                hooks[h] = dict(
                    parameters={name: _cpu(p) for name, p in model.named_parameters()},
                    adam={
                        name: _cpu(optimizer.state.get(p, {}))
                        for name, p in model.named_parameters()
                    },
                    options=_cpu(
                        {
                            k: v
                            for k, v in optimizer.param_groups[0].items()
                            if k != "params"
                        }
                    ),
                )
            filename = f"rank{groups.rank}.pt"
            torch.save(hooks, stage / filename)
            written.append(filename)
            del hooks
        if groups.rank == groups.active_ranks[0]:
            torch.save(
                _cpu(
                    dict(
                        replicated=state.replicated,
                        activation_caches=state.activation_caches,
                    )
                ),
                stage / "replicated.pt",
            )
            written.append("replicated.pt")
        numpy_state = np.random.get_state()
        rng = dict(
            torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state(groups.device)
            if groups.device.type == "cuda"
            else None,
            python=random.getstate(),
            numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        )
        filename = f"rng{groups.rank}.pt"
        torch.save(rng, stage / filename)
        written.append(filename)
        if groups.rank == 0:
            torch.save(_cpu(provider_state), stage / "provider.pt")
            written.append("provider.pt")
        written = [
            dict(name=n, bytes=(stage / n).stat().st_size, sha256=_digest(stage / n))
            for n in written
        ]
    except Exception as exc:
        error = repr(exc)
    groups.check(error)
    records = groups.agree(written)
    error = None
    if groups.rank == 0:
        try:
            manifest = dict(
                format_version=1,
                world_size=len(groups.control_ranks),
                configs={h: _config(c) for h, c in session.configs.items()},
                layouts={
                    h: dict(ranks=layout.ranks, ids=layout.ids)
                    for h, layout in session.layouts.items()
                },
                progress=canonical[3],
                dead_feature_window=session.dead_feature_window,
                max_grad_norm=session.max_grad_norm,
                tp_overlap=session.tp_overlap,
                tp_overlap_max_live_hooks=session.tp_overlap_max_live_hooks,
                input_scale=session.input_scale,
                input_scales=session.input_scales,
                gradient_accumulation_steps=session.gradient_accumulation_steps,
                files=[record for worker in records for record in worker],
            )
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            stage.rename(path)
        except Exception as exc:
            error = repr(exc)
    groups.check(error)
    return dict(
        path=str(path),
        bytes=sum(item["bytes"] for worker in records for item in worker),
    )


@torch.no_grad()
def load_checkpoint(session, path):
    from sae_lens.saes.megatron_topk_sae import MegatronTopKSAE
    from sae_lens.training.dynamic_tp import TPTrainState

    path = _boundary(session, path)
    groups = session.groups
    manifest, error = None, None
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        if manifest["world_size"] != len(groups.control_ranks):
            raise ValueError(
                "Resume requires the same fixed WORLD pool, with any TP membership"
            )
        if manifest["configs"] != {h: _config(c) for h, c in session.configs.items()}:
            raise ValueError("Checkpoint SAE configuration differs")
        accumulation = manifest.get("gradient_accumulation_steps", 1)
        input_scales = session.validate_input_scales(manifest.get("input_scales"))
        if type(accumulation) is not int or accumulation < 1:
            raise ValueError("Invalid checkpoint gradient_accumulation_steps")
        if groups.prepared_ranks is not None:
            raise ValueError("Resume before preparing the next topology")
        if groups.rank == 0:
            for record in manifest["files"]:
                file = path / record["name"]
                if (
                    file.stat().st_size != record["bytes"]
                    or _digest(file) != record["sha256"]
                ):
                    raise ValueError(f"Checkpoint checksum failed: {file.name}")
    except Exception as exc:
        error = repr(exc)
    groups.check(error)
    target = TPTrainState(progress=copy.deepcopy(manifest["progress"]))
    layouts, provider, rng, error = {}, None, None, None
    try:
        # mmap avoids materializing every saved tensor when current TP differs.
        shards = {}
        for h, spec in manifest["layouts"].items():
            saved = FeatureLayout(
                tuple(spec["ranks"]), tuple(tuple(ids) for ids in spec["ids"])
            )
            layout = (
                saved
                if saved.ranks == groups.active_ranks
                else FeatureLayout.balanced(saved.total, groups.active_ranks)
            )
            layouts[h] = layout
            if groups.rank not in groups.active_ranks:
                continue
            model = MegatronTopKSAE(
                session.configs[h],
                tp_group=groups.active_group,
                initialize=False,
                feature_layout=layout,
            )
            for rank in saved.ranks:
                if rank not in shards:
                    shards[rank] = torch.load(
                        path / f"rank{rank}.pt",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
            first = shards[saved.ranks[0]][h]
            options = first["options"]
            import inspect

            accepted = inspect.signature(torch.optim.Adam).parameters
            optimizer = torch.optim.Adam(
                model.parameters(),
                **{k: v for k, v in options.items() if k in accepted},
            )
            optimizer.param_groups[0].update(copy.deepcopy(options))
            ids = layout.local(groups.rank).ids(torch.device("cpu")).tolist()
            destinations = {feature: index for index, feature in enumerate(ids)}
            intersections = []
            for rank, old_ids in zip(saved.ranks, saved.ids):
                pairs = [
                    (i, destinations[feature])
                    for i, feature in enumerate(old_ids)
                    if feature in destinations
                ]
                if pairs:
                    src, dst = zip(*pairs)
                    intersections.append(
                        (
                            rank,
                            torch.tensor(src),
                            torch.tensor(dst, device=groups.device),
                        )
                    )

            def restore(name, key, axis):
                def tensor(rank):
                    hook = shards[rank][h]
                    return (
                        hook["parameters"][name]
                        if key is None
                        else hook["adam"][name][key]
                    )

                original = tensor(saved.ranks[0])
                if axis is None:
                    return original.to(groups.device)
                shape = list(original.shape)
                shape[axis] = len(ids)
                result = torch.empty(shape, dtype=original.dtype, device=groups.device)
                for rank, src, dst in intersections:
                    result.index_copy_(
                        axis,
                        dst,
                        tensor(rank).index_select(axis, src).to(groups.device),
                    )
                return result

            for name, parameter in model.named_parameters():
                axis = model._tp_param_shard_dims()[name]
                parameter.copy_(restore(name, None, axis))
                state = {}
                for key, original in first["adam"][name].items():
                    if key == "step":
                        step_device = (
                            groups.device
                            if options.get("fused") or options.get("capturable")
                            else "cpu"
                        )
                        state[key] = (
                            original.to(step_device)
                            if torch.is_tensor(original)
                            else original
                        )
                    else:
                        state[key] = restore(name, key, axis)
                if state:
                    optimizer.state[parameter] = state
            target.models[h], target.optimizers[h] = model, optimizer
        if groups.rank in groups.active_ranks:
            replicas = torch.load(
                path / "replicated.pt", map_location="cpu", weights_only=True, mmap=True
            )
            target.replicated = {
                k: v.to(groups.device).clone()
                for k, v in replicas["replicated"].items()
            }
            target.activation_caches = {
                k: v.to(groups.device).clone()
                for k, v in replicas["activation_caches"].items()
            }
        provider = torch.load(
            path / "provider.pt", map_location="cpu", weights_only=True
        )
        rng = torch.load(
            path / f"rng{groups.rank}.pt", map_location="cpu", weights_only=True
        )
    except Exception as exc:
        error = repr(exc)
    groups.check(error)
    groups.synchronize()
    old = session.state
    session.state, session.layouts = target, layouts
    old.models.clear()
    old.optimizers.clear()
    old.replicated.clear()
    old.activation_caches.clear()
    old.retired = True
    for key in (
        "dead_feature_window",
        "max_grad_norm",
        "tp_overlap",
        "tp_overlap_max_live_hooks",
    ):
        setattr(session, key, manifest[key])
    session.input_scale = float(manifest.get("input_scale", 1.0))
    session.input_scales = input_scales
    session.gradient_accumulation_steps = manifest.get("gradient_accumulation_steps", 1)
    session._window = None
    session._microbatch_active = False
    torch.set_rng_state(rng["torch_cpu"])
    if groups.device.type == "cuda" and rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"], groups.device)
    random.setstate(rng["python"])
    ns = rng["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    groups.check(None)
    return provider
