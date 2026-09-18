"""TopK SAE with Megatron-owned parameters and SAE-specific TP reductions.

The semantic oracle is SAELens 6.37.6 (see tests/native_reference). Checkpoint
conversion is the only place that uses SAELens' transposed weight layout.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from sae_lens.constants import SAE_CFG_FILENAME, SAE_WEIGHTS_FILENAME
from sae_lens.megatron_tp import (
    _shard_init_topk_cpu,
    megatron_tp_allgather,
    megatron_tp_launch,
    require_megatron_core,
)
from sae_lens.sae_runtime import SAERuntime
from sae_lens.saes.sae import TrainingSAE, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import (
    SparseHookPoint,
    TopK,
    TopKTPWavefrontState,
    TopKTrainingSAEConfig,
    _fold_norm_topk,
    calculate_topk_aux_acts,
)


class MegatronTopKSAE(TrainingSAE[TopKTrainingSAEConfig]):
    """Static SAE TP, including TP1, using Column/RowParallelLinear.

    Explicit groups avoid owning Megatron's global model-parallel state. TP1
    uses a singleton group. Parameters/Adam states use the Megatron layout.
    """

    # native checkpoint name -> (module parameter name, native shard dimension)
    checkpoint_layout = {
        "W_enc": ("encoder.weight", 1),
        "W_dec": ("decoder.weight", 0),
        "b_enc": ("encoder.bias", 0),
        "b_dec": ("b_dec", None),
    }

    def __init__(
        self,
        cfg: TopKTrainingSAEConfig,
        use_error_term: bool = False,
        *,
        tp_group: dist.ProcessGroup | None = None,
        runtime: SAERuntime | None = None,
    ):
        require_megatron_core()
        if runtime is not None:
            runtime_group = runtime.require_local().tp_group
            if tp_group is not None and tp_group is not runtime_group:
                raise ValueError("Explicit TP group differs from the SAE runtime")
            tp_group = runtime_group
        if tp_group is None:
            raise ValueError(
                "MegatronTopKSAE requires an explicit TP group, including a singleton group for TP1"
            )
        self._init_tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        if cfg.d_sae % self.tp_size:
            raise ValueError("d_sae must be divisible by SAE TP size")
        if cfg.d_in < 2 or not 0 < cfg.k <= cfg.d_sae:
            raise ValueError("TopK requires d_in >= 2 and 0 < k <= d_sae")
        super().__init__(copy.deepcopy(cfg), use_error_term)
        self._tp_group = tp_group
        self.parallel_context = runtime
        self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
        self.setup()

    @classmethod
    def from_config_sharded(
        cls,
        cfg: TopKTrainingSAEConfig,
        tp_group: dist.ProcessGroup,
        use_error_term: bool = False,
    ) -> MegatronTopKSAE:
        return cls(cfg, use_error_term, tp_group=tp_group)

    def initialize_weights(self) -> None:
        from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
            set_tensor_model_parallel_attributes,
        )

        config = ModelParallelConfig(
            tensor_model_parallel_size=self.tp_size,
            params_dtype=self.dtype,
            use_cpu_initialization=True,
            perform_initialization=False,
            # Enable only after native DDP has allocated main_grad. Standalone
            # SAEs and the DP1 direct-gradient fast path have no such buffer.
            gradient_accumulation_fusion=False,
            sequence_parallel=False,
        )
        self.encoder = ColumnParallelLinear(
            self.cfg.d_in,
            self.cfg.d_sae,
            config=config,
            init_method=nn.init.zeros_,
            bias=True,
            gather_output=False,
            # Detached activation training only consumes the token-summed
            # encoder input gradient in b_dec. Reduce that vector at the bias
            # edge below instead of all-reducing [tokens, d_in] in this linear.
            disable_grad_reduce=True,
            tp_group=self._init_tp_group,
        )
        self.decoder = RowParallelLinear(
            self.cfg.d_sae,
            self.cfg.d_in,
            config=config,
            init_method=nn.init.zeros_,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=self._init_tp_group,
        )
        # Megatron sets these in its initializers, which we replace to reproduce
        # the native SAE initialization (including tied initial encoder values).
        set_tensor_model_parallel_attributes(self.encoder.weight, True, 0, 1)
        set_tensor_model_parallel_attributes(self.decoder.weight, True, 1, 1)
        w_dec, w_enc, b_enc, b_dec = _shard_init_topk_cpu(
            self.cfg, self.tp_size, self.tp_rank
        )
        with torch.no_grad():
            self.encoder.weight.copy_(w_enc.T)
            self.encoder.bias.copy_(b_enc)
            self.decoder.weight.copy_(w_dec.T)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.b_dec = nn.Parameter(b_dec.to(self.device))
        self.b_dec.tensor_model_parallel = False
        self.b_dec.allreduce = True

    def configure_gradient_accumulation_fusion(self, enabled: bool) -> bool:
        """Enable native wgrad accumulation after DDP owns the gradient buffers."""
        from megatron.core.tensor_parallel import layers

        weights = (self.encoder.weight, self.decoder.weight)
        available = layers._grad_accum_fusion_available
        buffered = all(p.is_cuda and hasattr(p, "main_grad") for p in weights)
        if enabled and buffered and not available:
            warnings.warn(
                "Megatron gradient accumulation fusion requested, but "
                "fused_weight_gradient_mlp_cuda is unavailable; using unfused wgrad.",
                RuntimeWarning,
                stacklevel=2,
            )
        active = bool(enabled and buffered and available)
        self.gradient_accumulation_fusion = active
        for linear in (self.encoder, self.decoder):
            linear.gradient_accumulation_fusion = active
            linear.config.gradient_accumulation_fusion = active
        # The decoder also receives ordinary autograd gradients through its
        # norms. Megatron's native escape hatch returns zero dummy wgrads and
        # adds the residual autograd gradient; otherwise DDP silently drops it.
        self.decoder.weight.zero_out_wgrad = (
            active and self.cfg.rescale_acts_by_decoder_norm
        )
        return active

    def get_activation_fn(self) -> TopK:
        # Megatron's linear modules consume dense activations. The sparse flag
        # controls the returned representation; decoding densifies at the edge.
        return TopK(self.cfg.k, self.cfg.use_sparse_activations)

    def get_coefficients(self) -> dict[str, float]:
        return {}

    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        sae_in = self.reshape_fn_in(sae_in.to(self.dtype))
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        bias = self.b_dec * self.cfg.apply_b_dec_to_input
        if self.tp_size > 1:
            tp = require_megatron_core()
            # Identity in forward, SUM across TP in backward. The subtraction
            # first reduces the token dimensions, so only d_in values move.
            # The replicated decode-path bias gradient is already complete
            # and bypasses this edge. AccumulateGrad therefore sees the full
            # bias gradient before DDP copies/reduce-scatters its buffer.
            if self.cfg.apply_b_dec_to_input:
                bias = tp.copy_to_tensor_model_parallel_region(
                    bias, group=self._tp_group
                )
            # Preserve differentiable-input/hook semantics when an upstream
            # consumer actually needs the full dgrad. Cached/online detached
            # activations do not create this larger backward collective.
            if sae_in.requires_grad:
                sae_in = tp.copy_to_tensor_model_parallel_region(
                    sae_in, group=self._tp_group
                )
        return sae_in - bias

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        shape = sae_in.shape[:-1]
        local, _ = self.encoder(sae_in.reshape(-1, self.cfg.d_in))
        local = self.hook_sae_acts_pre(local.reshape(*shape, -1))
        if self.cfg.rescale_acts_by_decoder_norm:
            local = local * self.decoder.weight.norm(dim=0)
        hidden_pre = megatron_tp_allgather(local, self._tp_group)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre)), hidden_pre

    def _decode_features(self, feature_acts: torch.Tensor) -> torch.Tensor:
        if feature_acts.is_sparse:
            feature_acts = feature_acts.to_dense()
        width = self.cfg.d_sae // self.tp_size
        local = feature_acts.narrow(-1, self.tp_rank * width, width)
        if self.cfg.rescale_acts_by_decoder_norm:
            local = local * (1 / self.decoder.weight.norm(dim=0))
        result, _ = self.decoder(local.reshape(-1, width))
        return result.reshape(*feature_acts.shape[:-1], self.cfg.d_in)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        # The encode bias edge sums its partial gradient; this decode path is
        # already replicated. No 1/TP scaling or post-DDP bias reduction.
        out = self.hook_sae_recons(self._decode_features(feature_acts) + self.b_dec)
        out = self.run_time_activation_norm_fn_out(out)
        return self.reshape_fn_out(out, self.d_head)

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = step_input.dead_neuron_mask
        if mask is None or (num_dead := int(mask.sum())) == 0:
            loss = sae_out.new_tensor(0.0)
        else:
            if self.cfg.normalize_activations in (
                "constant_norm_rescale",
                "layer_norm",
            ):
                raise ValueError(
                    "TopK auxiliary loss does not support activation normalization"
                )
            k_aux = self.cfg.d_in // 2
            scale = min(num_dead / k_aux, 1.0)
            aux = calculate_topk_aux_acts(min(k_aux, num_dead), hidden_pre, mask)
            recons = self.reshape_fn_out(self._decode_features(aux), self.d_head)
            residual = (step_input.sae_in - sae_out).detach()
            loss = (
                self.cfg.aux_loss_coefficient
                * scale
                * (recons - residual).pow(2).sum(dim=-1).mean()
            )
        return {"auxiliary_reconstruction_loss": loss}

    def sync_tensor_parallel_gradients(self) -> None:
        """Megatron's bias-edge reduction completes TP gradients in backward."""

    def tp_wavefront_supported(self) -> bool:
        return self.tp_size > 1 and self.encoder.weight.is_cuda

    def tp_wavefront_encode_launch(
        self, step_input: TrainStepInput
    ) -> TopKTPWavefrontState:
        if not self.tp_wavefront_supported():
            raise RuntimeError("TP wavefront requires a CUDA SAE with TP > 1")
        sae_in = self.process_sae_in(step_input.sae_in)
        local, _ = self.encoder(sae_in.reshape(-1, self.cfg.d_in))
        local = self.hook_sae_acts_pre(local.reshape(*sae_in.shape[:-1], -1))
        if self.cfg.rescale_acts_by_decoder_norm:
            local = local * self.decoder.weight.norm(dim=0)
        return TopKTPWavefrontState(
            step_input=step_input, hidden_pre_local=local,
            gather=megatron_tp_launch(local, self._tp_group, gather=True),
        )

    def tp_wavefront_decode_launch(self, state: TopKTPWavefrontState) -> None:
        hidden_pre = state.gather.wait()
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        dense = feature_acts.to_dense() if feature_acts.is_sparse else feature_acts
        width = self.cfg.d_sae // self.tp_size
        local = dense.narrow(-1, self.tp_rank * width, width)
        if self.cfg.rescale_acts_by_decoder_norm:
            local = local * (1 / self.decoder.weight.norm(dim=0))
        # Exactly the local computation of native RowParallelLinear.forward.
        # Split only its trailing native TP reduction so the next hook can run
        # before the consumer waits. Parameters and linear autograd stay native.
        decoder = self.decoder
        if (
            not decoder.input_is_parallel or decoder.sequence_parallel
            or decoder.explicit_expert_comm or decoder.bias is not None
            or decoder.config._cpu_offloading_context is not None
        ):
            raise RuntimeError("Unsupported RowParallelLinear wavefront configuration")
        recons = decoder._forward_impl(
            input=local.reshape(-1, width), weight=decoder.weight, bias=None,
            gradient_accumulation_fusion=decoder.gradient_accumulation_fusion,
            allreduce_dgrad=False, sequence_parallel=False,
            tp_group=None, grad_output_buffer=None,
        ).reshape(*dense.shape[:-1], self.cfg.d_in)
        state.hidden_pre = hidden_pre
        state.feature_acts = feature_acts
        state.decode_bias = self.b_dec
        state.reduce = megatron_tp_launch(recons, self._tp_group, gather=False)

    def tp_wavefront_finish(self, state: TopKTPWavefrontState) -> TrainStepOutput:
        if state.reduce is None or state.hidden_pre is None or state.feature_acts is None:
            raise RuntimeError("Incomplete Megatron TP wavefront state")
        out = self.hook_sae_recons(state.reduce.wait() + self.b_dec)
        out = self.run_time_activation_norm_fn_out(out)
        out = self.reshape_fn_out(out, self.d_head)
        # Auxiliary reconstruction keeps the ordinary native decoder path,
        # including the encode bias-edge reduction and input-gradient fallback.
        return self._build_train_step_output(
            state.step_input, state.feature_acts, state.hidden_pre, out
        )

    def _tp_param_shard_dims(self) -> dict[str, int | None]:
        return {
            "encoder.weight": 0,
            "decoder.weight": 1,
            "encoder.bias": 0,
            "b_dec": None,
        }

    def _gather_tp_tensor(
        self, tensor: torch.Tensor, shard_dim: int | None
    ) -> torch.Tensor:
        if self.tp_size == 1 or shard_dim is None:
            return tensor.detach().clone()
        parts = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(parts, tensor.detach().contiguous(), group=self._tp_group)
        return torch.cat(parts, dim=shard_dim)

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: float, dp_group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        """L2 clipping for this SAE alone, counting replicated b_dec once."""
        if dp_group is not None:
            raise NotImplementedError(
                "FSDP gradient clipping is not part of static Megatron TP"
            )
        sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for name, param in self.named_parameters():
            if param.grad is not None and (name != "b_dec" or self.tp_rank == 0):
                sq += param.grad.detach().float().square().sum()
        if self.tp_size > 1:
            dist.all_reduce(sq, group=self._tp_group)
        norm = sq.sqrt()
        coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for param in self.parameters():
            if param.grad is not None:
                param.grad.mul_(coef)
        return norm

    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "TopK norm folding requires decoder norm rescaling"
            )
        norms = self.decoder.weight.norm(dim=0).clamp(min=1e-8)
        self.decoder.weight.div_(norms.unsqueeze(0))
        self.encoder.weight.mul_(norms.unsqueeze(1))
        self.encoder.bias.mul_(norms)

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
        self.encoder.weight.mul_(scaling_factor)
        self.decoder.weight.div_(scaling_factor)
        self.b_dec.div_(scaling_factor)
        self.cfg.normalize_activations = "none"

    def log_histograms(self) -> dict[str, Any]:
        return {
            "weights/W_dec_norms": self.decoder.weight.detach()
            .float()
            .norm(dim=0)
            .cpu()
            .numpy()
        }

    def _native_to_local(
        self, state: dict[str, torch.Tensor], *, already_sharded: bool = False
    ) -> dict[str, torch.Tensor]:
        converted = {}
        for native, (internal, axis) in self.checkpoint_layout.items():
            value = state[native]
            expected = {
                "W_enc": (self.cfg.d_in, self.cfg.d_sae),
                "W_dec": (self.cfg.d_sae, self.cfg.d_in),
                "b_enc": (self.cfg.d_sae,),
                "b_dec": (self.cfg.d_in,),
            }[native]
            expected = list(expected)
            if already_sharded and axis is not None:
                expected[axis] //= self.tp_size
            if tuple(value.shape) != tuple(expected):
                raise ValueError(
                    f"{native}: expected shape {tuple(expected)}, got {tuple(value.shape)}"
                )
            if axis is not None and not already_sharded:
                width = self.cfg.d_sae // self.tp_size
                value = value.narrow(axis, self.tp_rank * width, width)
            converted[internal] = (
                (value.T if value.ndim == 2 else value).contiguous().clone()
            )
        extra = set(state) - self.checkpoint_layout.keys()
        if extra:
            raise ValueError(f"Unexpected SAELens checkpoint keys: {sorted(extra)}")
        return converted

    def import_saelens_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Explicitly load the same full SAELens weights on all TP ranks."""
        self.load_state_dict(self._native_to_local(state))

    def _full_megatron_to_native(
        self, state: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        return {
            native: (
                state[internal].T if state[internal].ndim == 2 else state[internal]
            )
            .detach()
            .contiguous()
            .clone()
            for native, (internal, _) in self.checkpoint_layout.items()
        }

    def export_saelens_state_dict(self) -> dict[str, torch.Tensor]:
        state = dict(self.state_dict())
        self.process_state_dict_for_saving(state)
        return state

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        full = {
            name: self._gather_tp_tensor(state_dict[name], axis)
            for name, axis in self._tp_param_shard_dims().items()
        }
        state_dict.clear()
        state_dict.update(self._full_megatron_to_native(full))

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        converted = self._native_to_local(state_dict)
        state_dict.clear()
        state_dict.update(converted)

    def postprocess_full_state_dict_for_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        if "encoder.weight" in state_dict:
            converted = self._full_megatron_to_native(state_dict)
            state_dict.clear()
            state_dict.update(converted)
        if self.cfg.rescale_acts_by_decoder_norm:
            _fold_norm_topk(
                state_dict["W_enc"], state_dict["b_enc"], state_dict["W_dec"]
            )

    def load_weights_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        self.load_state_dict(
            self.load_saelens_checkpoint_shard(
                Path(checkpoint_path) / SAE_WEIGHTS_FILENAME
            )
        )

    def load_saelens_checkpoint_shard(
        self, path: str | Path
    ) -> dict[str, torch.Tensor]:
        state = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            if set(f.keys()) != set(self.checkpoint_layout):
                raise ValueError("Expected a TopK SAELens checkpoint")
            for native, (_, axis) in self.checkpoint_layout.items():
                sl = f.get_slice(native)
                if axis is None:
                    state[native] = f.get_tensor(native)
                else:
                    if sl.get_shape()[axis] != self.cfg.d_sae:
                        raise ValueError(f"Incorrect feature width for {native}")
                    width = self.cfg.d_sae // self.tp_size
                    indices = [slice(None)] * len(sl.get_shape())
                    indices[axis] = slice(
                        self.tp_rank * width, (self.tp_rank + 1) * width
                    )
                    state[native] = sl[tuple(indices)]
        return self._native_to_local(state, already_sharded=True)

    def _save(self, path: str | Path, inference: bool) -> tuple[Path, Path]:
        path = Path(path)
        state = self.export_saelens_state_dict()
        cfg = self.cfg.to_dict()
        if inference:
            self.postprocess_full_state_dict_for_inference(state)
            cfg = self.cfg.get_inference_sae_cfg_dict()
        if self.tp_rank == 0:
            path.mkdir(parents=True, exist_ok=True)
            save_file(state, path / SAE_WEIGHTS_FILENAME)
            (path / SAE_CFG_FILENAME).write_text(json.dumps(cfg))
        if self.tp_size > 1:
            dist.barrier(group=self._tp_group)
        return path / SAE_WEIGHTS_FILENAME, path / SAE_CFG_FILENAME

    def save_model(self, path: str | Path) -> tuple[Path, Path]:
        return self._save(path, False)

    def save_inference_model(self, path: str | Path) -> tuple[Path, Path]:
        return self._save(path, True)

    def process_named_optimizer_state_for_saving(
        self, optimizer_state: dict[str, dict[str, Any]]
    ) -> None:
        converted = {}
        axes = self._tp_param_shard_dims()
        for native, (internal, _) in self.checkpoint_layout.items():
            if internal not in optimizer_state:
                continue
            converted[native] = {}
            for key, value in optimizer_state[internal].items():
                if torch.is_tensor(value) and value.ndim > 0:
                    value = self._gather_tp_tensor(value, axes[internal])
                    value = (value.T if value.ndim == 2 else value).contiguous()
                converted[native][key] = value
        optimizer_state.clear()
        optimizer_state.update(converted)

    def process_named_optimizer_state_for_loading(
        self,
        optimizer_state: dict[str, dict[str, Any]],
        *,
        already_sharded: bool = False,
    ) -> None:
        extra = set(optimizer_state) - self.checkpoint_layout.keys()
        if extra:
            raise ValueError(f"Unexpected SAELens optimizer keys: {sorted(extra)}")
        converted = {}
        for native, (internal, axis) in self.checkpoint_layout.items():
            if native not in optimizer_state:
                continue
            converted[internal] = {}
            for key, value in optimizer_state[native].items():
                if torch.is_tensor(value) and value.ndim > 0:
                    if axis is not None and not already_sharded:
                        width = self.cfg.d_sae // self.tp_size
                        value = value.narrow(axis, self.tp_rank * width, width)
                    value = (value.T if value.ndim == 2 else value).contiguous().clone()
                    expected = self.get_parameter(internal).shape
                    if value.shape != expected:
                        raise ValueError(
                            f"{native}.{key}: expected local shape {tuple(expected)}, "
                            f"got {tuple(value.shape)}"
                        )
                converted[internal][key] = value
        optimizer_state.clear()
        optimizer_state.update(converted)
