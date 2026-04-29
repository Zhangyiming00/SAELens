"""Inference-only TopKSAE variant, similar in spirit to StandardSAE but using a TopK-based activation."""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
from transformer_lens.hook_points import HookPoint
from typing_extensions import override
from safetensors.torch import save_file

from sae_lens.distributed import tp_allgather, tp_allreduce
from sae_lens.constants import SAE_CFG_FILENAME, SAE_WEIGHTS_FILENAME
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
)


class SparseHookPoint(HookPoint):
    """
    A HookPoint that takes in a sparse tensor.
    Overrides TransformerLens's HookPoint.
    """

    def __init__(self, d_sae: int):
        super().__init__()
        self.d_sae = d_sae

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        using_hooks = (
            self._forward_hooks is not None and len(self._forward_hooks) > 0
        ) or (self._backward_hooks is not None and len(self._backward_hooks) > 0)
        if using_hooks and x.is_sparse:
            return x.to_dense()
        return x  # if no hooks are being used, use passthrough


class _ScaleGradient(torch.autograd.Function):
    """Identity in forward, scales the incoming gradient in backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return grad * ctx.scale, None


def _scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _ScaleGradient.apply(x, scale)  # type: ignore[return-value]


class TopK(nn.Module):
    """
    A simple TopK activation that zeroes out all but the top K elements along the last dimension,
    and applies ReLU to the top K elements.
    """

    use_sparse_activations: bool

    def __init__(
        self,
        k: int,
        use_sparse_activations: bool = False,
    ):
        super().__init__()
        self.k = k
        self.use_sparse_activations = use_sparse_activations

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        1) Select top K elements along the last dimension.
        2) Apply ReLU.
        3) Zero out all other entries.
        """
        topk_values, topk_indices = torch.topk(x, k=self.k, dim=-1, sorted=False)
        values = topk_values.relu()
        if self.use_sparse_activations:
            # Produce a COO sparse tensor (use sparse matrix multiply in decode)
            original_shape = x.shape

            # Create indices for all dimensions
            # For each element in topk_indices, we need to map it back to the original tensor coordinates
            batch_dims = original_shape[:-1]  # All dimensions except the last one
            num_batch_elements = torch.prod(torch.tensor(batch_dims)).item()

            # Create batch indices - each batch element repeated k times
            batch_indices_flat = torch.arange(
                num_batch_elements, device=x.device
            ).repeat_interleave(self.k)

            # Convert flat batch indices back to multi-dimensional indices
            if len(batch_dims) == 1:
                # 2D case: [batch, features]
                sparse_indices = torch.stack(
                    [
                        batch_indices_flat,
                        topk_indices.flatten(),
                    ]
                )
            else:
                # 3D+ case: need to unravel the batch indices
                batch_indices_multi = []
                remaining = batch_indices_flat
                for dim_size in reversed(batch_dims):
                    batch_indices_multi.append(remaining % dim_size)
                    remaining = remaining // dim_size
                batch_indices_multi.reverse()

                sparse_indices = torch.stack(
                    [
                        *batch_indices_multi,
                        topk_indices.flatten(),
                    ]
                )

            return torch.sparse_coo_tensor(
                sparse_indices, values.flatten(), original_shape
            )
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result


@dataclass
class TopKSAEConfig(SAEConfig):
    """
    Configuration class for TopKSAE inference.

    Args:
        k (int): Number of top features to keep active during inference. Only the top k
            features with the highest pre-activations will be non-zero. Defaults to 100.
        rescale_acts_by_decoder_norm (bool): Whether to treat the decoder as if it was
            already normalized. This affects the topk selection by rescaling pre-activations
            by decoder norms. Requires that the SAE was trained this way. Defaults to False.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig. Defaults to True.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
            Defaults to "none".
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
            Defaults to "none".
        metadata (SAEMetadata): Metadata about the SAE (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    k: int = 100
    rescale_acts_by_decoder_norm: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


def _sparse_matmul_nd(
    sparse_tensor: torch.Tensor, dense_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Multiply a sparse tensor of shape [..., d_sae] with a dense matrix of shape [d_sae, d_out]
    to get a result of shape [..., d_out].

    This function handles sparse tensors with arbitrary batch dimensions by flattening
    the batch dimensions, performing 2D sparse matrix multiplication, and reshaping back.
    """
    original_shape = sparse_tensor.shape
    batch_dims = original_shape[:-1]
    d_sae = original_shape[-1]
    d_out = dense_matrix.shape[-1]

    if sparse_tensor.ndim == 2:
        # Simple 2D case - use torch.sparse.mm directly
        # sparse.mm errors with bfloat16 :(
        with torch.autocast(device_type=sparse_tensor.device.type, enabled=False):
            return torch.sparse.mm(sparse_tensor, dense_matrix)

    # For 3D+ case, reshape to 2D, multiply, then reshape back
    batch_size = int(torch.prod(torch.tensor(batch_dims)).item())

    # Ensure tensor is coalesced for efficient access to indices/values
    if not sparse_tensor.is_coalesced():
        sparse_tensor = sparse_tensor.coalesce()

    # Get indices and values
    indices = sparse_tensor.indices()  # [ndim, nnz]
    values = sparse_tensor.values()  # [nnz]

    # Convert multi-dimensional batch indices to flat indices
    flat_batch_indices = torch.zeros_like(indices[0])
    multiplier = 1
    for i in reversed(range(len(batch_dims))):
        flat_batch_indices += indices[i] * multiplier
        multiplier *= batch_dims[i]

    # Create 2D sparse tensor indices [batch_flat, feature]
    sparse_2d_indices = torch.stack([flat_batch_indices, indices[-1]])

    # Create 2D sparse tensor
    sparse_2d = torch.sparse_coo_tensor(
        sparse_2d_indices, values, (batch_size, d_sae)
    ).coalesce()

    # sparse.mm errors with bfloat16 :(
    with torch.autocast(device_type=sparse_tensor.device.type, enabled=False):
        # Do the matrix multiplication
        result_2d = torch.sparse.mm(sparse_2d, dense_matrix)  # [batch_size, d_out]

    # Reshape back to original batch dimensions
    result_shape = tuple(batch_dims) + (d_out,)
    return result_2d.view(result_shape)


def act_times_W_dec(
    feature_acts: torch.Tensor,
    W_dec: torch.Tensor,
    rescale_acts_by_decoder_norm: bool,
) -> torch.Tensor:
    """
    Multiply feature activations by decoder weights, with optional decoder norm rescaling.

    Handles both dense and sparse tensors. Does NOT add b_dec — callers add it if needed.
    """
    if rescale_acts_by_decoder_norm:
        feature_acts = feature_acts * (1 / W_dec.norm(dim=-1))
    if feature_acts.is_sparse:
        return _sparse_matmul_nd(feature_acts, W_dec)
    return feature_acts @ W_dec


class TopKSAE(SAE[TopKSAEConfig]):
    """
    An inference-only sparse autoencoder using a "topk" activation function.
    It uses linear encoder and decoder layers, applying the TopK activation
    to the hidden pre-activation in its encode step.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: TopKSAEConfig, use_error_term: bool = False):
        """
        Args:
            cfg: SAEConfig defining model size and behavior.
            use_error_term: Whether to apply the error-term approach in the forward pass.
        """
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_topk(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts input x into feature activations.
        Uses topk activation under the hood.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
        # The BaseSAE already sets self.activation_fn to TopK(...) if config requests topk.
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(
        self,
        feature_acts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs the input from topk feature activations.
        Applies optional finetuning scaling, hooking to recons, out normalization,
        and optional head reshaping.
        """
        sae_out_pre = (
            act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            + self.b_dec
        )
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k, use_sparse_activations=False)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Folding W_dec_norm is not safe for TopKSAEs when rescale_acts_by_decoder_norm is False, as this may change the topk activations"
            )
        _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)


@dataclass
class TopKTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a TopKTrainingSAE.

    Args:
        k (int): Number of top features to keep active. Only the top k features
            with the highest pre-activations will be non-zero. Defaults to 100.
        use_sparse_activations (bool): Whether to use sparse tensor representations
            for activations during training. This can reduce memory usage and improve
            performance when k is small relative to d_sae, but is only worthwhile if
            using float32 and not using autocast. Defaults to False.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that encourages
            dead neurons to learn useful features. This loss helps prevent neuron death
            in TopK SAEs by having dead neurons reconstruct the residual error from
            live neurons. Defaults to 1.0.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already normalized.
            This is a good idea since decoder norm can randomly drift during training, and this
            affects what the topk activations will be. Defaults to True.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            0.1 corresponds to the "heuristic" initialization from Anthropic's April update.
            Use None to disable. Inherited from TrainingSAEConfig. Defaults to 0.1.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig. Defaults to True.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
            Defaults to "none".
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
            Defaults to "none".
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    k: int = 100
    use_sparse_activations: bool = False
    aux_loss_coefficient: float = 1.0
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class TopKTrainingSAE(TrainingSAE[TopKTrainingSAEConfig]):
    """
    TopK variant with training functionality. Calculates a topk-related auxiliary loss, etc.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: TopKTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._tp_group: dist.ProcessGroup | None = None
        self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
        self.setup()

    def _tp_param_shard_dims(self) -> dict[str, int | None]:
        return {
            "W_enc": 1,
            "W_dec": 0,
            "b_enc": 0,
            "b_dec": None,
        }

    def _gather_tp_tensor(
        self,
        tensor: torch.Tensor,
        shard_dim: int | None,
    ) -> torch.Tensor:
        if self._tp_group is None or shard_dim is None:
            return tensor.detach().clone()

        tp_size = dist.get_world_size(self._tp_group)
        parts = [torch.zeros_like(tensor) for _ in range(tp_size)]
        dist.all_gather(parts, tensor.detach().contiguous(), group=self._tp_group)
        return torch.cat(parts, dim=shard_dim)

    def _slice_tp_tensor(
        self,
        tensor: torch.Tensor,
        shard_dim: int | None,
    ) -> torch.Tensor:
        if self._tp_group is None or shard_dim is None:
            return tensor

        tp_size = dist.get_world_size(self._tp_group)
        tp_rank = dist.get_rank(self._tp_group)
        full_size = tensor.shape[shard_dim]
        assert full_size % tp_size == 0, (
            f"Tensor shape {tuple(tensor.shape)} along dim {shard_dim} must be divisible "
            f"by tp_size={tp_size}"
        )
        shard_size = full_size // tp_size
        return tensor.narrow(shard_dim, tp_rank * shard_size, shard_size).contiguous()

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_topk(self)

    def shard_weights(self, tp_group: dist.ProcessGroup) -> None:
        """Shard SAE latents across TP ranks along the feature dimension.

        Each rank owns d_sae // tp_size latents.  W_enc, W_dec, and b_enc are
        sliced in-place; self._tp_group is set so that encode/decode use
        allgather / allreduce to coordinate across ranks.
        """
        tp_rank = dist.get_rank(tp_group)
        tp_size = dist.get_world_size(tp_group)
        assert self.cfg.d_sae % tp_size == 0, (
            f"d_sae={self.cfg.d_sae} must be divisible by tp_size={tp_size}"
        )
        shard_size = self.cfg.d_sae // tp_size
        start = tp_rank * shard_size
        end = start + shard_size
        self._tp_group = tp_group
        self.W_enc = nn.Parameter(self.W_enc.data[:, start:end].contiguous())
        self.W_dec = nn.Parameter(self.W_dec.data[start:end, :].contiguous())
        self.b_enc = nn.Parameter(self.b_enc.data[start:end].contiguous())

    @override
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        if self._tp_group is None:
            return
        for name, shard_dim in self._tp_param_shard_dims().items():
            if name in state_dict:
                state_dict[name] = self._gather_tp_tensor(state_dict[name], shard_dim)

    @override
    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        if self._tp_group is None:
            return
        for name, shard_dim in self._tp_param_shard_dims().items():
            if name not in state_dict or shard_dim is None:
                continue
            state_dict[name] = self._slice_tp_tensor(state_dict[name], shard_dim)

    @override
    def process_named_optimizer_state_for_saving(
        self, optimizer_state: dict[str, dict[str, Any]]
    ) -> None:
        if self._tp_group is None:
            return
        for name, shard_dim in self._tp_param_shard_dims().items():
            param_state = optimizer_state.get(name)
            if param_state is None:
                continue
            for key, value in list(param_state.items()):
                if torch.is_tensor(value) and value.ndim > 0:
                    param_state[key] = self._gather_tp_tensor(value, shard_dim)

    @override
    def process_named_optimizer_state_for_loading(
        self, optimizer_state: dict[str, dict[str, Any]]
    ) -> None:
        if self._tp_group is None:
            return
        for name, shard_dim in self._tp_param_shard_dims().items():
            param_state = optimizer_state.get(name)
            if param_state is None:
                continue
            for key, value in list(param_state.items()):
                if torch.is_tensor(value) and value.ndim > 0:
                    param_state[key] = self._slice_tp_tensor(value, shard_dim)

    @override
    def sync_tensor_parallel_gradients(self) -> None:
        if self._tp_group is None:
            return

        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            if self._tp_param_shard_dims().get(name) is not None:
                continue
            dist.all_reduce(param.grad, group=self._tp_group)

    @override
    def save_model(self, path: str | Path) -> tuple[Path, Path]:
        if self._tp_group is None:
            return super().save_model(path)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        self.process_state_dict_for_saving(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        cfg_path = path / SAE_CFG_FILENAME

        if dist.get_rank(self._tp_group) == 0:
            save_file(state_dict, model_weights_path)
            with open(cfg_path, "w") as f:
                json.dump(self.cfg.to_dict(), f)

        dist.barrier(group=self._tp_group)
        return model_weights_path, cfg_path

    @override
    def save_inference_model(self, path: str | Path) -> tuple[Path, Path]:
        if self._tp_group is None:
            return super().save_inference_model(path)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        self.process_state_dict_for_saving_inference(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        cfg_path = path / SAE_CFG_FILENAME

        if dist.get_rank(self._tp_group) == 0:
            save_file(state_dict, model_weights_path)
            with open(cfg_path, "w") as f:
                json.dump(self.cfg.get_inference_sae_cfg_dict(), f)

        dist.barrier(group=self._tp_group)
        return model_weights_path, cfg_path

    @override
    def clip_grad_norm_(
        self,
        max_norm: float,
        dp_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        if self._tp_group is None and dp_group is None:
            return super().clip_grad_norm_(max_norm)

        tp_rank = dist.get_rank(self._tp_group) if self._tp_group is not None else 0
        local_sq_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        shard_dims = self._tp_param_shard_dims()

        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            grad_sq = param.grad.detach().float().pow(2).sum()
            shard_dim = shard_dims.get(name)
            if shard_dim is None:
                # Count replicated parameters only once across TP ranks.
                if tp_rank == 0:
                    local_sq_norm += grad_sq
            else:
                local_sq_norm += grad_sq

        if self._tp_group is not None:
            dist.all_reduce(local_sq_norm, group=self._tp_group)
        if dp_group is not None:
            # In FSDP mode each DP rank holds a distinct shard; summing squared
            # norms across DP recovers the true global gradient norm.
            dist.all_reduce(local_sq_norm, group=dp_group)

        total_norm = local_sq_norm.sqrt()
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for param in self.parameters():
            if param.grad is not None:
                param.grad.mul_(clip_coef.to(device=param.grad.device, dtype=param.grad.dtype))
        return total_norm

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Similar to the base training method: calculate pre-activations, then apply TopK.

        With TP, each rank computes pre-activations for its local feature shard
        and then allgathers to reconstruct the full (d_sae,) pre-activation vector
        before applying top-k selection.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre_local = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        _debug_topk_tp("encode local pre done")

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre_local = hidden_pre_local * self.W_dec.norm(dim=-1)

        if self._tp_group is not None:
            _debug_topk_tp("encode allgather start")
            hidden_pre = tp_allgather(hidden_pre_local, self._tp_group)
            _debug_topk_tp("encode allgather done")
        else:
            hidden_pre = hidden_pre_local

        # Apply the TopK activation function (already set in self.activation_fn if config is "topk")
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        _debug_topk_tp("encode activation done")
        return feature_acts, hidden_pre

    @override
    def decode(
        self,
        feature_acts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes feature activations back into input space.

        With TP, each rank multiplies its local feature shard against its W_dec
        shard and allreduces the partial sums.
        """
        if self._tp_group is not None:
            tp_rank = dist.get_rank(self._tp_group)
            tp_size = dist.get_world_size(self._tp_group)
            shard_size = feature_acts.shape[-1] // tp_size
            local_acts = feature_acts[..., tp_rank * shard_size : (tp_rank + 1) * shard_size]
            if self.cfg.rescale_acts_by_decoder_norm:
                local_acts = local_acts * (1 / self.W_dec.norm(dim=-1))
            _debug_topk_tp("decode allreduce start")
            # b_dec also appears in process_sae_in(). Its encode-path gradient is sharded
            # across TP ranks, while its decode-path gradient is replicated on every rank.
            # Scale only the decode-path backward contribution so the later TP all-reduce on
            # b_dec.grad reconstructs the single-rank gradient exactly.
            decode_bias = _scale_gradient(self.b_dec, 1.0 / tp_size)
            sae_out_pre = tp_allreduce(local_acts @ self.W_dec, self._tp_group) + decode_bias
            _debug_topk_tp("decode allreduce done")
        else:
            sae_out_pre = (
                act_times_W_dec(
                    feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
                )
                + self.b_dec
            )
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        topk_loss = self.calculate_topk_aux_loss(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )
        return {"auxiliary_reconstruction_loss": topk_loss}

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Folding W_dec_norm is not safe for TopKSAEs when rescale_acts_by_decoder_norm is False, as this may change the topk activations"
            )
        _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k, use_sparse_activations=self.cfg.use_sparse_activations)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate TopK auxiliary loss.

        This auxiliary loss encourages dead neurons to learn useful features by having
        them reconstruct the residual error from the live neurons. It's a key part of
        preventing neuron death in TopK SAEs.
        """
        # Mostly taken from https://github.com/EleutherAI/sae/blob/main/sae/sae.py, except without variance normalization
        # NOTE: checking the number of dead neurons will force a GPU sync, so performance can likely be improved here
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)
        _debug_topk_tp(f"aux start num_dead={num_dead}")

        if self.cfg.normalize_activations in ("constant_norm_rescale", "layer_norm"):
            raise ValueError(
                "TopK auxiliary loss does not support activation normalization "
                f"(normalize_activations={self.cfg.normalize_activations!r}). "
                "The aux loss reconstruction would be in normalized space while the "
                "residual is in the original space, producing incorrect gradients."
            )

        residual = (sae_in - sae_out).detach()

        # Heuristic from Appendix B.1 in the paper
        k_aux = sae_in.shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_acts = calculate_topk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents. Per the paper (Appendix A.2), the reconstruction
        # is ê = W_dec @ z (no bias), since b_dec is already in the residual.
        if self._tp_group is not None:
            tp_rank = dist.get_rank(self._tp_group)
            tp_size = dist.get_world_size(self._tp_group)
            shard_size = auxk_acts.shape[-1] // tp_size
            local_acts = auxk_acts[..., tp_rank * shard_size : (tp_rank + 1) * shard_size]
            if self.cfg.rescale_acts_by_decoder_norm:
                local_acts = local_acts * (1 / self.W_dec.norm(dim=-1))
            _debug_topk_tp("aux allreduce start")
            recons = tp_allreduce(local_acts @ self.W_dec, self._tp_group)
            _debug_topk_tp("aux allreduce done")
        else:
            recons = act_times_W_dec(
                auxk_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
        # Apply the same reshaping as decode() so recons matches the residual's shape
        recons = self.reshape_fn_out(recons, self.d_head)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        if self.cfg.rescale_acts_by_decoder_norm:
            _fold_norm_topk(
                W_enc=state_dict["W_enc"],
                b_enc=state_dict["b_enc"],
                W_dec=state_dict["W_dec"],
            )


def calculate_topk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Helper method to calculate activations for the auxiliary loss.

    Args:
        k_aux: Number of top dead neurons to select
        hidden_pre: Pre-activation values from encoder
        dead_neuron_mask: Boolean mask indicating which neurons are dead

    Returns:
        Tensor with activations for only the top-k dead neurons, zeros elsewhere
    """

    # Don't include living latents in this loss
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
    # Top-k dead latents
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    # Set the activations to zero for all but the top k_aux dead latents
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
    # Set activations to zero for all but top k_aux dead latents
    return auxk_acts


def _init_weights_topk(
    sae: SAE[TopKSAEConfig] | TrainingSAE[TopKTrainingSAEConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )


def _fold_norm_topk(
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
) -> None:
    W_dec_norm = W_dec.norm(dim=-1).clamp(min=1e-8)
    b_enc.data = b_enc.data * W_dec_norm
    W_dec_norms = W_dec_norm.unsqueeze(1)
    W_dec.data = W_dec.data / W_dec_norms
    W_enc.data = W_enc.data * W_dec_norms.T


def _debug_topk_tp(msg: str) -> None:
    if os.environ.get("SAELENS_DEBUG_PREFIX_TP") != "1":
        return
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    line = f"[topk-debug rank{rank}] {msg}\n"
    with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
        f.write(line)
    print(line, end="", flush=True)
