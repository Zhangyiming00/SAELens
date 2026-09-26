"""TopK SAE with Megatron-owned parameters and SAE-specific TP reductions.

The semantic oracle is SAELens 6.37.6 (see tests/native_reference). Checkpoint
conversion is the only place that uses SAELens' transposed weight layout.
"""

from __future__ import annotations

import copy
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
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
    _wavefront_stream,
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
from sae_lens.auxk_compact import prepare_auxk_dense, compact_aux_decode
from sae_lens.sharded_sparse import scale_sparse_features, sparse_decode
from sae_lens.sharded_topk import (
    launch_sharded_topk,
    sharded_firing_counts,
    sharded_topk,
    sharded_auxk,
)


@dataclass
class MegatronTPWavefrontState(TopKTPWavefrontState):
    decoder_norm: torch.Tensor | None = None
    decoder_vectors: torch.Tensor | None = None


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
        from sae_lens.adaptive_sae import validate_config
        validate_config(cfg)
        if cfg.topk_backend not in ("legacy", "sharded_dense", "sharded_sparse", "sharded_ragged"):
            raise ValueError("Unknown TopK backend: " + str(cfg.topk_backend))
        if cfg.topk_key_backend not in ("torch", "triton"):
            raise ValueError("Unknown TopK comparison-key backend")
        if cfg.topk_candidate_protocol not in ("auto", "candidates", "radix"):
            raise ValueError("Unknown TopK candidate protocol")
        if cfg.auxk_selection not in ("auto", "legacy"):
            raise ValueError("Unknown AuxK selection policy")
        if cfg.auxk_decoder_backend not in ("auto", "local_dense", "compact_dense"):
            raise ValueError("Unknown AuxK dense decoder backend")
        if cfg.auxk_complement not in ("auto", "off"):
            raise ValueError("Unknown AuxK complement selection mode")
        if cfg.topk_backend != "sharded_dense" and cfg.auxk_decoder_backend == "compact_dense":
            raise ValueError("Compact AuxK decoder requires sharded_dense")
        if cfg.sparse_decoder_backend not in ("torch", "sae", "triton"):
            raise ValueError("Unknown sparse decoder backend")
        if cfg.topk_backend != "legacy" and (
            cfg.topk_key_backend == "triton" or (
                cfg.topk_backend == "sharded_sparse" and cfg.sparse_decoder_backend == "triton"
            )
        ):
            # Fail BEFORE distributed training, never silently densify a global latent.
            from sae_lens.sharded_triton import triton_sparse_decode  # noqa: F401
        if cfg.topk_backend == "sharded_ragged":
            if cfg.ragged_decoder_engine not in ("openai", "triton", "torch_reference"):
                raise ValueError("Unknown ragged decoder engine")
            for policy in (cfg.ragged_main_compute, cfg.ragged_aux_compute):
                if policy not in ("sparse", "auto", "local_dense", "compact_dense"):
                    raise ValueError("Unknown ragged compute policy")
            if cfg.ragged_wgrad_split not in (1, 2, 4, 8) or cfg.ragged_index_backend not in ("sort", "histogram"):
                raise ValueError("Invalid ragged weight-gradient configuration")
            if cfg.ragged_decoder_engine == "openai":
                if cfg.ragged_wgrad_split != 1 or cfg.ragged_index_backend != "sort":
                    raise ValueError("OpenAI kernel engine requires the original sorted-COO wgrad path (split=1, sort)")
                if cfg.ragged_openai_page_k not in (32, 64, 128, 256, 512):
                    raise ValueError("Invalid OpenAI page K")
                if not 1 <= cfg.ragged_openai_workspace_mib <= 4096:
                    raise ValueError("Invalid OpenAI workspace budget")
                if cfg.ragged_openai_forward not in ("bucketed", "coo"):
                    raise ValueError("Invalid OpenAI forward adapter")
                # Import original upstream functions before the first collective.
                from sae_lens.openai_sae_adapter import _kernels
                _kernels()
            if cfg.ragged_decoder_engine == "triton":
                # Eager import catches missing dependency before any forward
                # collective. No runtime dense fallback on import/JIT errors.
                from sae_lens.ragged_sae_triton import forward as _ragged_forward
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
        if getattr(cfg, "topk_tie_policy", "stable_id") == "torch_tp1" and self.tp_size != 1:
            raise ValueError("torch_tp1 ties are only defined for TP1; no full gather is allowed to emulate torch ties")
        if cfg.d_sae % self.tp_size:
            raise ValueError("d_sae must be divisible by SAE TP size")
        if cfg.d_in < 2 or not 0 < cfg.k <= cfg.d_sae:
            raise ValueError("TopK requires d_in >= 2 and 0 < k <= d_sae")
        super().__init__(copy.deepcopy(cfg), use_error_term)
        self._tp_group = tp_group
        self.parallel_context = runtime
        self.hook_sae_acts_post = SparseHookPoint(
            self.cfg.d_sae // self.tp_size if self.sharded_latents else self.cfg.d_sae
        )
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
            linear_active = active and not (
                linear is self.decoder and self.cfg.topk_backend in ("sharded_sparse", "sharded_ragged")
            )
            linear.gradient_accumulation_fusion = linear_active
            # Some Megatron versions share one config between both linears.
            # Never mutate the encoder's flag while disabling decoder fusion.
            linear.config = copy.copy(linear.config)
            linear.config.gradient_accumulation_fusion = linear_active
        # The decoder also receives ordinary autograd gradients through its
        # norms AND packed AuxK index-select. The native escape hatch returns
        # zero dummy wgrads and
        # adds the residual autograd gradient; otherwise DDP silently drops it.
        self.decoder.weight.zero_out_wgrad = (
            self.decoder.gradient_accumulation_fusion
            and (self.cfg.rescale_acts_by_decoder_norm or (
                self.cfg.topk_backend == "sharded_dense"
                and getattr(self.cfg, "auxk_decoder_backend", "auto") != "local_dense"
            ))
        )
        return active

    @property
    def sharded_latents(self) -> bool:
        return self.cfg.topk_backend != "legacy"

    def _check_sharded_post_hooks(self) -> None:
        # A global post-activation edit cannot be emulated on one feature shard.
        # Never silently execute such a callback on a different tensor layout.
        if self.sharded_latents and (self.tp_size > 1 or self.cfg.topk_backend == "sharded_ragged"):
            point = self.hook_sae_acts_post
            if point._forward_hooks or point._backward_hooks:
                raise NotImplementedError(
                    "Global post-latent hooks are incompatible with sharded TopK. "
                    "Consume the explicit local encode() output, or select legacy."
                )

    def _build_train_step_output(self, step_input, feature_acts, hidden_pre, sae_out):
        if self.sharded_latents and step_input.sae_in.shape[0] == 0:
            # Empty replicas must still produce connected zero parameter gradients.
            zero = sae_out.sum() * 0.0 + hidden_pre.sum() * 0.0
            result = TrainStepOutput(
                sae_in=step_input.sae_in, sae_out=sae_out,
                feature_acts=feature_acts, hidden_pre=hidden_pre,
                loss=zero, losses={"mse_loss": zero, "auxiliary_reconstruction_loss": zero},
            )
        else:
            result = super()._build_train_step_output(
                step_input, feature_acts, hidden_pre, sae_out
            )
        if self.sharded_latents:
            result.feature_firing_counts = sharded_firing_counts(feature_acts, self._tp_group)
        return result

    def get_activation_fn(self) -> TopK:
        # Megatron's linear modules consume dense activations. The sparse flag
        # controls the returned representation; decoding densifies at the edge.
        return TopK(self.cfg.k, self.cfg.use_sparse_activations)

    def get_coefficients(self) -> dict[str, float]:
        return {}

    @contextmanager
    def _share_decoder_norm(self, norm: torch.Tensor | None = None, vectors: torch.Tensor | None = None):
        """Share one differentiable norm node within this forward only.

        Its vector gradients combine before expanding to the decoder matrix.
        Restore the scope on exceptions/nested calls; never reuse across steps.
        """
        previous = getattr(self, "_decoder_norm_scope", None)
        previous_vectors = getattr(self, "_decoder_vectors_scope", None)
        self._decoder_norm_scope = [norm]
        self._decoder_vectors_scope = [vectors]
        try:
            yield
        finally:
            self._decoder_norm_scope = previous
            self._decoder_vectors_scope = previous_vectors

    def _decoder_norm(self) -> torch.Tensor:
        scope = getattr(self, "_decoder_norm_scope", None)
        if scope is None:
            return self.decoder.weight.norm(dim=0)
        if scope[0] is None:
            scope[0] = self.decoder.weight.norm(dim=0)
        return scope[0]

    def _decoder_vectors(self):
        """One feature-major differentiable LOCAL weight copy per live forward.

        No cached Parameter or optimizer state, no stale cross-step copy. Main
        and AuxK share this Tensor through normal and wavefront scopes.
        """
        scope = getattr(self, "_decoder_vectors_scope", None)
        if scope is None:
            return self.decoder.weight.T.contiguous()
        if scope[0] is None:
            scope[0] = self.decoder.weight.T.contiguous()
        return scope[0]

    def _decode_ragged_partial(self, acts, norm=None, *, auxiliary=False, known_columns=None):
        from sae_lens.ragged_sae import (RaggedLatents, decode_ragged,
                                        dense_ragged_reference, packed_diagnostics,
                                        compact_ragged_dense)
        if not isinstance(acts, RaggedLatents):
            raise TypeError("sharded_ragged decode requires explicit selected-entry metadata")
        if self.cfg.rescale_acts_by_decoder_norm:
            norm = self._decoder_norm() if norm is None else norm
            acts = acts.scale(norm.reciprocal())
        from sae_lens.adaptive_sae import active, decode_adaptive
        if active(self.cfg, auxiliary):
            result, diagnostic = decode_adaptive(acts, self._decoder_vectors(), self.cfg,
                                                 auxiliary=auxiliary, known_columns=known_columns)
            if auxiliary:
                self._last_auxk_execution = diagnostic
            else:
                self._last_main_execution = diagnostic
            return result
        engine = getattr(self.cfg, "ragged_decoder_engine", "triton")
        policy = getattr(self.cfg, "ragged_aux_compute" if auxiliary else "ragged_main_compute", "sparse")
        # 'auto' is an explicit structural policy, NOT a measured fastest-kernel
        # planner: compact only when all eligible columns are selected; sparse otherwise.
        actual = ("compact_dense" if auxiliary and acts.selection == "select_all" else "sparse") if policy == "auto" else policy
        vectors = self._decoder_vectors()
        if actual == "sparse":
            out = decode_ragged(acts, vectors, engine=engine,
                                split=getattr(self.cfg, "ragged_wgrad_split", 1),
                                index_backend=getattr(self.cfg, "ragged_index_backend", "sort"),
                                openai_page_k=getattr(self.cfg, "ragged_openai_page_k", 512),
                                openai_workspace_mib=getattr(self.cfg, "ragged_openai_workspace_mib", 64),
                                openai_forward=getattr(self.cfg, "ragged_openai_forward", "bucketed"))
        elif actual == "local_dense":
            out = dense_ragged_reference(acts, vectors)
        elif actual == "compact_dense":
            out = compact_ragged_dense(acts, vectors)
        else:
            raise ValueError("Unknown ragged computation policy")
        diagnostic = packed_diagnostics(acts, engine, actual)
        if engine == "openai" and actual == "sparse":
            from sae_lens.openai_sae_adapter import upstream_identity
            diagnostic.update(upstream_identity())
            diagnostic.update(page_k=getattr(self.cfg, "ragged_openai_page_k", 512),
                              forward_adapter=getattr(self.cfg, "ragged_openai_forward", "bucketed"),
                              workspace_mib=getattr(self.cfg, "ragged_openai_workspace_mib", 64))
        if auxiliary:
            self._last_auxk_execution = diagnostic
        else:
            self._last_main_execution = diagnostic
        return out

    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        with self._share_decoder_norm():
            return super().training_forward_pass(step_input)

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
        local = self.hook_sae_acts_pre(local.reshape(*shape, self.cfg.d_sae // self.tp_size))
        if self.cfg.rescale_acts_by_decoder_norm:
            local = local * self._decoder_norm()
        if self.sharded_latents:
            self._check_sharded_post_hooks()
            acts = sharded_topk(
                local, self.cfg.k, self._tp_group,
                sparse=self.cfg.topk_backend == "sharded_sparse",
                packed=self.cfg.topk_backend == "sharded_ragged",
                protocol=self.cfg.topk_candidate_protocol, key_backend=self.cfg.topk_key_backend,
                tie_policy=getattr(self.cfg, "topk_tie_policy", "stable_id"),
            )
            return self.hook_sae_acts_post(acts), local
        hidden_pre = megatron_tp_allgather(local, self._tp_group)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre)), hidden_pre

    def _decode_local(self, feature_acts: torch.Tensor, norm: torch.Tensor | None = None,
                      *, wavefront: bool = False) -> torch.Tensor:
        width = self.cfg.d_sae // self.tp_size
        if feature_acts.shape[-1] != width:
            raise ValueError("Sharded decode requires local [*, d_sae/TP] features")
        if self.cfg.topk_backend == "sharded_ragged":
            partial = self._decode_ragged_partial(feature_acts, norm)
            if wavefront or self.tp_size == 1:
                return partial
            return require_megatron_core().reduce_from_tensor_model_parallel_region(partial, group=self._tp_group)
        local = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            norm = self._decoder_norm() if norm is None else norm
            local = scale_sparse_features(local, norm.reciprocal())
        if self.cfg.topk_backend == "sharded_sparse":
            # TP1 hooks may supply local dense acts; convert ONLY this local shard.
            if not local.is_sparse:
                local = local.to_sparse()
            partial = sparse_decode(local, self.decoder.weight,
                                    backend=self.cfg.sparse_decoder_backend)
            if wavefront or self.tp_size == 1:
                return partial
            return require_megatron_core().reduce_from_tensor_model_parallel_region(
                partial, group=self._tp_group
            )
        if local.is_sparse:
            local = local.to_dense()  # width checked above; always local
        if not wavefront:
            result, _ = self.decoder(local.reshape(-1, width))
        else:
            decoder = self.decoder
            if (not decoder.input_is_parallel or decoder.sequence_parallel
                or decoder.explicit_expert_comm or decoder.bias is not None
                or decoder.config._cpu_offloading_context is not None):
                raise RuntimeError("Unsupported RowParallelLinear wavefront configuration")
            result = decoder._forward_impl(
                input=local.reshape(-1, width), weight=decoder.weight, bias=None,
                gradient_accumulation_fusion=decoder.gradient_accumulation_fusion,
                allreduce_dgrad=False, sequence_parallel=False,
                tp_group=None, grad_output_buffer=None,
            )
        return result.reshape(*local.shape[:-1], self.cfg.d_in)

    def _decode_features(self, feature_acts: torch.Tensor) -> torch.Tensor:
        if self.sharded_latents:
            return self._decode_local(feature_acts)
        if feature_acts.is_sparse:
            feature_acts = feature_acts.to_dense()
        width = self.cfg.d_sae // self.tp_size
        local = feature_acts.narrow(-1, self.tp_rank * width, width)
        if self.cfg.rescale_acts_by_decoder_norm:
            local = local * (1 / self._decoder_norm())
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
            self._last_auxk_execution = {"selection": "skipped", "decoder": "skipped", "num_dead": 0}
            loss = sae_out.new_tensor(0.0)
        else:
            if self.cfg.normalize_activations in (
                "constant_norm_rescale",
                "layer_norm",
            ):
                raise ValueError(
                    "TopK auxiliary loss does not support activation normalization"
                )
            configured_auxk = getattr(self.cfg, "auxk", None)
            k_aux = self.cfg.d_in // 2 if configured_auxk is None else int(configured_auxk)
            scale = min(num_dead / k_aux, 1.0)
            if self.sharded_latents:
                width = self.cfg.d_sae // self.tp_size
                if mask.shape != (self.cfg.d_sae,):
                    raise ValueError("Dead mask must be the existing global [d_sae] summary")
                local_mask = mask.narrow(0, self.tp_rank * width, width)
                if self.cfg.topk_backend == "sharded_ragged":
                    from sae_lens.ragged_sae import ragged_auxk
                    from sae_lens.adaptive_sae import active, try_direct_aux
                    enabled = active(self.cfg, True)
                    direct = None
                    if enabled:
                        norm = self._decoder_norm() if self.cfg.rescale_acts_by_decoder_norm else None
                        direct = try_direct_aux(hidden_pre, local_mask, num_dead, k_aux,
                                                self.decoder.weight, norm, self.cfg)
                    if direct is not None:
                        recons, self._last_auxk_execution = direct
                    else:
                        aux = ragged_auxk(hidden_pre, min(k_aux, num_dead), local_mask, num_dead, self._tp_group,
                                          policy=getattr(self.cfg, "auxk_selection", "auto"),
                                          protocol=self.cfg.topk_candidate_protocol, key_backend=self.cfg.topk_key_backend,
                                          complement=getattr(self.cfg, "auxk_complement", "auto"),
                                          tie_policy=getattr(self.cfg, "topk_tie_policy", "stable_id"))
                        # Eligible IDs are shared across this batch, not values.
                        columns = local_mask.nonzero(as_tuple=True)[0] if enabled else None
                        recons = self._decode_ragged_partial(aux, auxiliary=True, known_columns=columns)
                    self._last_auxk_execution.update(num_dead=num_dead, k=min(k_aux, num_dead))
                    if self.tp_size > 1:
                        recons = require_megatron_core().reduce_from_tensor_model_parallel_region(recons, group=self._tp_group)
                elif self.cfg.topk_backend == "sharded_dense":
                    aux, columns, plan = prepare_auxk_dense(
                        hidden_pre, min(k_aux, num_dead), local_mask, num_dead, self._tp_group,
                        decoder=getattr(self.cfg, "auxk_decoder_backend", "auto"),
                        complement=getattr(self.cfg, "auxk_complement", "auto"),
                        selection_policy=getattr(self.cfg, "auxk_selection", "auto"),
                        protocol=self.cfg.topk_candidate_protocol,
                        key_backend=self.cfg.topk_key_backend,
                        tie_policy=getattr(self.cfg, "topk_tie_policy", "stable_id"),
                    )
                    # Diagnostics retain scalars only; never tensors/autograd state.
                    self._last_auxk_execution = {
                        "selection": plan.selection, "decoder": plan.decoder,
                        "num_dead": num_dead, "k": plan.k, "exclude_k": plan.exclude_k,
                        "compact_width": None if columns is None else columns.numel(),
                        "comparison_key_backend": (
                            "torch" if plan.selection.startswith("complement") else
                            "none" if plan.selection == "select_all" else self.cfg.topk_key_backend
                        ),
                    }
                    if columns is not None:
                        if (self.decoder.gradient_accumulation_fusion and
                            not getattr(self.decoder.weight, "zero_out_wgrad", False)):
                            raise RuntimeError(
                                "Compact AuxK needs zero_out_wgrad for fused main + ordinary auxiliary "
                                "gradients. Configure auxk_decoder_backend before wrapping native DDP "
                                "and call configure_gradient_accumulation_fusion again after changing it."
                            )
                        norm = self._decoder_norm() if self.cfg.rescale_acts_by_decoder_norm else None
                        recons = compact_aux_decode(aux, columns, self.decoder.weight, norm)
                        if self.tp_size > 1:
                            recons = require_megatron_core().reduce_from_tensor_model_parallel_region(
                                recons, group=self._tp_group
                            )
                    else:
                        recons = self._decode_features(aux)
                else:
                    aux = sharded_auxk(
                        hidden_pre, min(k_aux, num_dead), local_mask, num_dead, self._tp_group,
                        sparse=True, policy=getattr(self.cfg, "auxk_selection", "auto"),
                        protocol=self.cfg.topk_candidate_protocol, key_backend=self.cfg.topk_key_backend,
                        tie_policy=getattr(self.cfg, "topk_tie_policy", "stable_id"),
                    )
                    recons = self._decode_features(aux)
            else:
                aux = calculate_topk_aux_acts(min(k_aux, num_dead), hidden_pre, mask)
                recons = self._decode_features(aux)
            recons = self.reshape_fn_out(recons, self.d_head)
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
    ) -> MegatronTPWavefrontState:
        if not self.tp_wavefront_supported():
            raise RuntimeError("TP wavefront requires a CUDA SAE with TP > 1")
        sae_in = self.process_sae_in(step_input.sae_in)
        local, _ = self.encoder(sae_in.reshape(-1, self.cfg.d_in))
        local = self.hook_sae_acts_pre(local.reshape(*sae_in.shape[:-1], self.cfg.d_sae // self.tp_size))
        norm = None
        if self.cfg.rescale_acts_by_decoder_norm:
            norm = self.decoder.weight.norm(dim=0)
            local = local * norm
        if self.sharded_latents:
            self._check_sharded_post_hooks()
            gather = launch_sharded_topk(
                local, self.cfg.k, self._tp_group,
                sparse=self.cfg.topk_backend == "sharded_sparse",
                packed=self.cfg.topk_backend == "sharded_ragged",
                stream=_wavefront_stream(local.device),
                protocol=self.cfg.topk_candidate_protocol, key_backend=self.cfg.topk_key_backend,
                tie_policy=getattr(self.cfg, "topk_tie_policy", "stable_id"),
            )
        else:
            gather = megatron_tp_launch(local, self._tp_group, gather=True)
        return MegatronTPWavefrontState(
            step_input=step_input, hidden_pre_local=local,
            gather=gather,
            decoder_norm=norm,
        )

    def tp_wavefront_decode_launch(self, state: MegatronTPWavefrontState) -> None:
        if self.sharded_latents:
            self._check_sharded_post_hooks()
            feature_acts = self.hook_sae_acts_post(state.gather.wait())
            with self._share_decoder_norm(state.decoder_norm, state.decoder_vectors):
                recons = self._decode_local(feature_acts, state.decoder_norm, wavefront=True)
                state.decoder_vectors = self._decoder_vectors_scope[0]
            state.hidden_pre = state.hidden_pre_local
            state.feature_acts = feature_acts
            state.decode_bias = self.b_dec
            state.reduce = megatron_tp_launch(recons, self._tp_group, gather=False)
            return
        hidden_pre = state.gather.wait()
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        dense = feature_acts.to_dense() if feature_acts.is_sparse else feature_acts
        width = self.cfg.d_sae // self.tp_size
        local = dense.narrow(-1, self.tp_rank * width, width)
        if self.cfg.rescale_acts_by_decoder_norm:
            assert state.decoder_norm is not None
            local = local * (1 / state.decoder_norm)
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

    def tp_wavefront_finish(self, state: MegatronTPWavefrontState) -> TrainStepOutput:
        if state.reduce is None or state.hidden_pre is None or state.feature_acts is None:
            raise RuntimeError("Incomplete Megatron TP wavefront state")
        out = self.hook_sae_recons(state.reduce.wait() + self.b_dec)
        out = self.run_time_activation_norm_fn_out(out)
        out = self.reshape_fn_out(out, self.d_head)
        # Auxiliary reconstruction keeps the ordinary native decoder path,
        # including the encode bias-edge reduction and input-gradient fallback.
        with self._share_decoder_norm(state.decoder_norm, state.decoder_vectors):
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
