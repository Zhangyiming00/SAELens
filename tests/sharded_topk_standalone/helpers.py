from __future__ import annotations

import ast
import math

# Spawned distributed workers do not execute pytest's conftest.
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

if "sae_lens" not in sys.modules:
    _pkg = types.ModuleType("sae_lens")
    _pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "sae_lens")]
    sys.modules["sae_lens"] = _pkg

from sae_lens.auxk_compact import compact_aux_decode, prepare_auxk_dense
from sae_lens.sharded_sparse import scale_sparse_features, sparse_decode
from sae_lens.sharded_topk import (
    launch_sharded_topk,
    sharded_auxk,
    sharded_firing_counts,
    sharded_topk,
)

ROOT = Path(__file__).resolve().parents[2]


def dense_reference(scores, k, eligible=None, relu=True):
    ranked = scores.detach().clone()
    if eligible is not None:
        ranked.masked_fill_(~eligible, -torch.inf)
    indices = ranked.argsort(dim=-1, descending=True, stable=True)[..., :k]
    values = scores.gather(-1, indices)
    if relu:
        values = values.relu()
    return torch.zeros_like(scores).scatter(-1, indices, values)


class NoGlobalLatent(TorchDispatchMode):
    """Catch transient AND saved full token-by-feature tensors in torch ops.

    This does not inspect private CUDA allocator requests inside native kernels.
    The GPU acceptance script also records CUDA allocator peaks.
    """

    def __init__(self, rows, global_features):
        self.rows, self.width = rows, global_features
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for item in tree_leaves(out):
            if isinstance(item, torch.Tensor):
                if (
                    item.ndim >= 2
                    and item.shape[-1] == self.width
                    and math.prod(item.shape[:-1]) == self.rows
                ):
                    raise AssertionError(
                        f"Forbidden global latent {tuple(item.shape)} in {func}"
                    )
                self.seen.append((str(func), tuple(item.shape)))
        return out


class CopyTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        if ctx.group is not None and dist.get_world_size(ctx.group) > 1:
            dist.all_reduce(grad, group=ctx.group)
        return grad, None


class ReduceTP(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, x, group):
        y = x.clone()
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(_ctx, grad):
        return grad, None


@dataclass
class Input:
    sae_in: torch.Tensor
    dead_neuron_mask: torch.Tensor | None = None
    coefficients: dict | None = None
    n_training_steps: int = 0
    is_logging_step: bool = False


@dataclass
class Output:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor
    losses: dict
    feature_firing_counts: torch.Tensor | None = None


class Base(nn.Module):
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def forward(self, x):
        return self.training_forward_pass(x)


class Column(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


class Row(nn.Linear):
    def __init__(self, width, d, group):
        super().__init__(width, d, bias=False)
        self.group = group
        self.input_is_parallel = True
        self.sequence_parallel = self.explicit_expert_comm = False
        self.gradient_accumulation_fusion = False
        self.config = SimpleNamespace(_cpu_offloading_context=None)

    def forward(self, x):
        return ReduceTP.apply(F.linear(x, self.weight), self.group), None

    def _forward_impl(self, **kw):
        return F.linear(kw["input"], kw["weight"], kw["bias"])


def load_harness_class():
    """Execute the actual edited model methods with F.linear/Gloo dependencies.

    Does NOT instantiate Megatron Core. Tests integration of selector, losses,
    norm/bias edges, stats, and wavefront split, not GPU bucket/stream machinery.
    """
    env = dict(
        torch=torch,
        dist=dist,
        nn=nn,
        dataclass=dataclass,
        contextmanager=contextmanager,
        TrainingSAE=Base,
        TopKTrainingSAEConfig=SimpleNamespace,
        TrainStepInput=Input,
        TrainStepOutput=Output,
        launch_sharded_topk=launch_sharded_topk,
        sharded_topk=sharded_topk,
        sharded_auxk=sharded_auxk,
        prepare_auxk_dense=prepare_auxk_dense,
        compact_aux_decode=compact_aux_decode,
        sharded_firing_counts=sharded_firing_counts,
        sparse_decode=sparse_decode,
        scale_sparse_features=scale_sparse_features,
        _wavefront_stream=lambda _device: None,
        require_megatron_core=lambda: SimpleNamespace(
            copy_to_tensor_model_parallel_region=lambda x, group: CopyTP.apply(
                x, group
            ),
            reduce_from_tensor_model_parallel_region=lambda x, group: ReduceTP.apply(
                x, group
            ),
        ),
    )
    env["megatron_tp_launch"] = (
        lambda x, group, gather: SimpleNamespace(wait=lambda: ReduceTP.apply(x, group))
        if not gather
        else (_ for _ in ()).throw(AssertionError("FULL LATENT GATHER"))
    )
    env["megatron_tp_allgather"] = lambda *_a: (_ for _ in ()).throw(
        AssertionError("FULL LATENT GATHER")
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations")], level=0
    )
    # Reuse actual base loss/output code as well, rather than duplicating it.
    base_tree = ast.parse((ROOT / "sae_lens/saes/sae.py").read_text())
    training = next(
        c
        for c in base_tree.body
        if isinstance(c, ast.ClassDef) and c.name == "TrainingSAE"
    )
    methods = [
        n
        for n in training.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_build_train_step_output", "training_forward_pass")
    ]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *methods], type_ignores=[])
            ),
            "<base methods>",
            "exec",
        ),
        env,
    )
    for method in methods:
        setattr(Base, method.name, env[method.name])
    topk_tree = ast.parse((ROOT / "sae_lens/saes/topk_sae.py").read_text())
    state = next(
        c
        for c in topk_tree.body
        if isinstance(c, ast.ClassDef) and c.name == "TopKTPWavefrontState"
    )
    model_tree = ast.parse((ROOT / "sae_lens/saes/megatron_topk_sae.py").read_text())
    classes = [c for c in model_tree.body if isinstance(c, ast.ClassDef)]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, state, *classes], type_ignores=[])
            ),
            "<real MegatronTopKSAE methods, stub dependencies>",
            "exec",
        ),
        env,
    )
    return env["MegatronTopKSAE"]


def make_harness(global_weights, group, backend, rescale=True, protocol="auto"):
    cls = load_harness_class()
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    enc, benc, dec, bdec = global_weights
    size = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    width, d = enc.shape[0] // size, enc.shape[1]
    model.tp_size, model.tp_rank, model._tp_group = size, rank, group
    model._decoupled_execution = False
    model.cfg = SimpleNamespace(
        d_sae=enc.shape[0],
        d_in=d,
        k=2,
        auxk=None,
        topk_backend=backend,
        topk_candidate_protocol=protocol,
        sparse_decoder_backend="torch",
        topk_key_backend="torch",
        rescale_acts_by_decoder_norm=rescale,
        apply_b_dec_to_input=True,
        aux_loss_coefficient=0.2,
        normalize_activations="none",
    )
    model.encoder, model.decoder = Column(d, width), Row(width, d, group)
    with torch.no_grad():
        model.encoder.weight.copy_(enc[rank * width : (rank + 1) * width])
        model.encoder.bias.copy_(benc[rank * width : (rank + 1) * width])
        model.decoder.weight.copy_(dec[:, rank * width : (rank + 1) * width])
    model.b_dec = nn.Parameter(bdec.clone())
    model.hook_sae_input = model.hook_sae_acts_pre = model.hook_sae_acts_post = (
        model.hook_sae_recons
    ) = nn.Identity()
    model.reshape_fn_in = model.run_time_activation_norm_fn_in = (
        model.run_time_activation_norm_fn_out
    ) = lambda x: x
    model.reshape_fn_out = lambda x, _d_head: x
    model.d_head, model.dtype = None, torch.float32
    model.mse_loss_fn = lambda a, b: (a - b).square()
    # CPU mathematical wavefront only: launch/wait has no CUDA scheduling.
    model.tp_wavefront_supported = lambda: True
    return model


def reference_forward(weights, x, k, mask, rescale=True):
    enc, benc, dec, bdec = weights
    pre = F.linear(x - bdec, enc, benc)
    norm = dec.norm(dim=0) if rescale else torch.ones(dec.shape[1])
    pre = pre * norm
    acts = dense_reference(pre, k)
    out = F.linear(acts / norm, dec) + bdec
    zero = out.sum() * 0.0
    loss = (out - x).square().sum(-1).mean() if x.shape[0] else zero
    if mask is not None and mask.any() and x.shape[0]:
        ka = min(dec.shape[0] // 2, int(mask.sum()))
        aux = dense_reference(pre, ka, eligible=mask, relu=False)
        recons = F.linear(aux / norm, dec)
        loss = (
            loss
            + 0.2
            * min(int(mask.sum()) / (dec.shape[0] // 2), 1.0)
            * (recons - (x - out).detach()).square().sum(-1).mean()
        )
    return out, acts, loss
