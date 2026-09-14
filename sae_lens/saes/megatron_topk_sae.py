"""TopK SAE with Megatron-owned parameters and standard TP backward semantics.

The semantic oracle is SAELens 6.37.6 (see tests/native_reference). Checkpoint
conversion is the only place that uses SAELens' transposed weight layout.
"""

from __future__ import annotations

import copy
import json
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
    require_megatron_core,
)
from sae_lens.saes.sae import TrainingSAE, TrainStepInput
from sae_lens.saes.topk_sae import (
    SparseHookPoint,
    TopK,
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
    ):
        require_megatron_core()
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
            disable_grad_reduce=False,
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

    def get_activation_fn(self) -> TopK:
        # Megatron's linear modules consume dense activations. The sparse flag
        # controls the returned representation; decoding densifies at the edge.
        return TopK(self.cfg.k, self.cfg.use_sparse_activations)

    def get_coefficients(self) -> dict[str, float]:
        return {}

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
        # ColumnParallelLinear already sums d(input) across TP. Both b_dec
        # contributions are complete on every rank: no 1/TP and no post reduce.
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
        """No compensation: Megatron modules complete TP gradients in backward."""

    def tp_wavefront_supported(self) -> bool:
        # The scheduling protocol is migrated after static TP acceptance.
        return False

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
