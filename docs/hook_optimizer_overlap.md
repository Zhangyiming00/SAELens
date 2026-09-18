# Per-hook native Megatron optimizer scheduling

Fixed TopK SAE runtimes retain an independent model, optimizer, clipping norm,
LR scheduler and update count for each local hook. DP replicas use native DDP
buffers; single-replica hooks use direct parameter gradients by default.
`--sae-pp-size / -spp` places hooks in disjoint domains; it is not a pipeline
schedule. Uneven assignments and singleton TP/DP groups are supported.

The existing `--multi-sae-distributed-architecture unified_multi_hook` now also
enables [cross-hook TP forward wavefront](megatron_tp_wavefront.md) for this
runtime. `legacy_per_hook_wrapper` disables it. Native per-hook optimizer/DDP
ownership is retained in either mode; this switch is independent of optimizer
overlap below.

| Control | `--ddp-zero-optimizer` | `--multi-sae-optimizer-overlap` |
| --- | --- | --- |
| A: GPU clip + native FP32Optimizer | absent | `off` |
| B: GPU clip + native FP32Optimizer | absent | `on` |
| C: native DistributedOptimizer + shard GPU clip | present | `off` |
| D: native DistributedOptimizer + shard GPU clip | present | `on` |

Native bucket gradient/backward overlap remains enabled wherever DDP is used.
The old `ddp_zero_optimizer` name now selects **Megatron DistributedOptimizer**
in a fixed CUDA runtime with DP > 1. At DP1 the default single-replica fast path
uses native FP32Optimizer, including when distributed optimization is requested;
the fallback is logged. `--no-sae-single-replica-fast-path` restores native DDP
and the native DP1 DistributedOptimizer diagnostic path. Legacy training paths
retain their PyTorch ZeroRedundancyOptimizer meaning. DP1 gets no Adam-state
sharding saving.
Parameters and Adam states remain FP32; autocast keeps its existing BF16 compute
and shared external GradScaler semantics. Adam uses coupled weight decay and
unchanged betas/epsilon. FP32Optimizer uses torch fused Adam. DistributedOptimizer
uses the installed Megatron version's supported Adam class: Transformer Engine
FusedAdam, Apex FusedAdam, or torch Adam when neither extension is installed.
FusedAdam is explicitly configured with `adam_w_mode=False` and FP32 moments.
Logs identify the actual inner Adam implementation.

Installing TE/Apex changes native Megatron's accepted optimizer class. The
checkpoint adapter preserves moments while converting torch's per-parameter
step tensors to FusedAdam's group step. Different parameter steps within a group
are rejected. Same-topology local shard checkpoints from the torch backend and
ordinary named checkpoints can both be loaded; empty shard clocks are recovered
from the saved hook update count. This does not add cross-topology resharding.

`non_tp_only` enables cross-hook updates only for TP1.
`--no-multi-sae-param-gather-overlap` retains cross-hook Adam scheduling but
moves all parameter gathers after all local compute and Adam work. It is a
communication diagnostic. `off` remains the serial control for either backend.
Configuration flows through CLI -> runner config -> trainer config -> runtime.
Logs print actual optimizer classes, effective overlap and downgrade reasons.

`--multi-sae-param-gather-schedule` selects a fixed submission policy when
native distributed optimization and both overlap switches are enabled:

- `eager` (default) keeps the native step's immediate gather.
- `one_hook_lag` submits A's gather after B's final backward has submitted its
  gradient collectives. Remaining gathers are submitted at the window boundary.
- `after_backward` submits gathers after all local backwards and updates have
  been enqueued, without a host wait or whole optimizer-stream fence.

The delayed policies use a separate gather stream, waiting on each hook's
`update_done` event. They preserve per-hook gather order; all ranks in a
placement must agree on the policy. Ordinary optimization, overlap disabled,
or one local hook uses the original path and logs the reason. The existing
`--no-multi-sae-param-gather-overlap` diagnostic overrides the policy and joins
all Adam work before gathering. Eager gather can delay subsequent reduce-scatter
on the same DP communicator; a delayed policy can also lose useful overlap, so
select it using measured latency rather than assuming a universal improvement.

## SAE encoder input-gradient reduction

Megatron `ColumnParallelLinear` uses `disable_grad_reduce=True`. Detached SAE
activation training needs only the token-summed encode contribution to `b_dec`,
so `copy_to_tensor_model_parallel_region` is applied to that bias edge. Its
native backward reduces `[d_in]` instead of `[tokens, d_in]`. The replicated
decode bias contribution bypasses the reduction; no extra TP division is used.
The parameter receives its complete gradient before DDP buffer accumulation or
reduce-scatter, so this also works with optimizer shards and padding. A
differentiable input or input hook retains its own full input-gradient reduction.
Parameter layouts, checkpoint names and moments are unchanged by this fix.

## Single-replica fast path

`--sae-single-replica-fast-path` is enabled by default. The condition is the
hook's placement-local DP group size, not world size: TP2, TP4 and TP x SPP with
one replica also qualify. Native ColumnParallelLinear, RowParallelLinear and
explicit TP groups are retained, but Megatron DDP and its gradient/main_grad
buffers are not constructed. Gradients flow directly into `parameter.grad`;
native gradient buffer bytes are zero. Serial and per-hook overlap both work.

DP1 token counts come from input shapes without a device-to-host scalar read.

`--sae-ga1-loss-normalization` independently enables pre-backward normalization
for FP32 compute with GA=1 across all TP/DP/SPP layouts and both optimizers.
With DP1 a nonempty hook backpropagates its mean loss directly. With DP>1 it
backpropagates `mean_loss * local_valid_tokens / global_valid_tokens`; counts
are already known before backward, and native DDP uses SUM. Unequal token
counts and empty replicas therefore retain the same global mean. This avoids
the post-reduction full-gradient division for both ordinary and sharded grads.
`--no-sae-ga1-loss-normalization` restores the token-sum/divide diagnostic path.
This switch is independent of DDP buffer removal and optimizer overlap.
GA>1 retains token-sum accumulation and post-reduction normalization. Autocast
GA1 retains the same token-sum scaled backward, then folds token normalization
into the shared GradScaler's existing unscale pass. Moving normalization before
BF16 backward failed the existing continuation tolerance; this adapter leaves
backward and the window loss scale unchanged. It follows torch 2.10's private
GradScaler unscale bookkeeping, checking valid optimizer gradient views once
per hook. Cross-rank overflow agreement, per-hook skips and the one shared
scaler update at window end remain unchanged. If the scaler is disabled, the
original post-reduction division applies. Empty hooks still advance no Adam,
LR or scaler state. The same diagnostic switch disables both GA1 optimizations.

## Ordering and lifetimes

For a known final microbatch, the trainer submits hooks in the runtime's fixed
local order. Immediately after A's final backward, it records a CUDA event.
The optimizer stream waits for that event, calls native `finish_grad_sync()`
(which waits on NCCL Work on the current consumer stream), normalizes valid
gradients by the hook's global token count, unscales/checks AMP, and invokes the
native optimizer step exactly once. That native step prepares, clips the whole
SAE and runs Adam. B's forward/backward can execute concurrently.

Native DistributedOptimizer owns reduce-scatter buffer intersections and
parameter all-gather. Its normal synchronous gather is synchronous **with the
calling optimizer stream**; it can overlap B on the compute stream. The
`params_ready` event is recorded after this gather dependency. Forward pre-hooks
are not installed; the native step owns eager gathers, and the trainer owns
delayed or diagnostic gathers. No duplicate gather is dispatched.

The lifecycle distinguishes accumulating, gradient communication, gradients
ready for the consumer stream, updating, parameters enqueued, and GPU-completed
parameters. `params_complete()` queries completion; an enqueued event is not
reported as GPU completion. Parameter/buffer allocations are registered on the
optimizer and gather streams, including for failure cleanup. Window completion joins all
updates before shared scaler updates, checkpointing, buffer reuse and the next
forward. Partial failed windows cannot be checkpointed.
Direct parameter gradients are also registered on the consuming update stream
before `zero_grad()` releases them, including on exceptional exits.

Unknown-length short tails cannot prove an early last backward; they complete
at the window boundary. Sized tails can overlap. A single local hook falls back
to serial updates while retaining native bucket communication overlap.

Multi-rank overlap requires NCCL >= 2.26 and
`NCCL_LAUNCH_ORDER_IMPLICIT=1` **before communicator initialization**, negotiated
across every rank of the placement. The schedule uses deterministic host launch
order, not rank-local readiness polling. These conditions apply to this fixed
schedule; the environment variable is not a guarantee for arbitrary streams.
Failure-monitor backward handshakes remain enabled. Setup preloads clipping and
fused Adam CUDA kernels using disposable tensors/state, once per device. This
avoids a first lazy kernel load synchronizing the context behind an incomplete
collective while the failure thread needs that context to abort. The setup-only
stream synchronization does not touch real model/Adam state or training timing.

## Shards, clipping and AMP

The adapter validates native model/buffer/shard ranges, storage aliases,
no gaps, no duplicate ownership, and exact hook membership. Padding and
unreduced buffer regions do not enter normalization, AMP, clipping or Adam.
Ordinary norms use the explicit TP group (exclude DP copies); distributed norms
use the placement's TP x DP group. Native TP metadata counts replicated decoder
bias only at TP rank zero. Different hooks never share a clipping coefficient.
Norm/coefficient computation stays on the GPU without scalar host reads.
Both optimizer backends use `torch._foreach_norm` and `torch._foreach_mul_` on
valid local gradient views. Only the small vector of per-parameter norms is
reduced locally; there is no full-gradient square temporary. The shard backend
retains its native valid ranges and TP duplicate filtering.

Megatron receives no internal scaler. The external scaler unscales each hook
once, agrees overflow on its TP/DP domain (including empty shard ranks), skips
only affected hooks, and updates once at the end of a nonempty window. AMP's
existing host skip decision is distinct from the device-only clipping path.
Entirely empty hooks/windows do not advance Adam, LR, or the scaler.

## Checkpoints and migration

Distributed checkpoints include `distributed_adam_rank<R>.pt` with the actual
local Adam moments/steps, parameter-group hyperparameters, and native shard
signature. Runtime files retain topology/batch identity, local statistics and
per-hook update counts. Missing shard files and mismatched topology or bucket
layout are rejected. Ordinary named checkpoints migrate on load: the existing
TP conversion runs first, then each valid DP shard receives its exact slice of
both moments and its Adam step. No moments are reset.

This release supports same-topology distributed resume. Old DP1 distributed
checkpoints also load into the new DP1 direct-gradient FP32Optimizer: complete
shard ranges and parameter identities are validated, both moments are reshaped
exactly, and the saved Adam step is preserved. This is not a general DP>1
distributed-to-ordinary conversion or cross-topology resharding facility. Older ordinary
checkpoints lacking rank-local statistics retain their existing DP0 statistics
restore behavior; their Adam moments and steps are preserved by migration.

Static cached and colocated online paths use this backend. Fixed TopK DDP SHM
streaming now uses the same runtime and retains singleton TP groups. Streaming
still has its existing GA=1 and exact-streaming data-cursor resume restrictions.
Elastic reconfiguration and low-precision Adam states are outside this implementation.

Acceptance, commands, source fingerprints and performance data are in
[`results/hook_overlap_20260916`](../results/hook_overlap_20260916/README.md),
with the single-replica and foreach-clipping follow-up in
[`results/single_replica_fastpath_20260917`](../results/single_replica_fastpath_20260917/README.md).
