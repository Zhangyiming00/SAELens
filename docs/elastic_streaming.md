# Elastic streaming SAE-DP switching

Elastic streaming reassigns a fixed block of ranks between vLLM generation and
SAE training without restarting `torchrun` or loading a training checkpoint. The
permanent vLLM ranks never join cutover collectives and continue producing until
the shared buffer is full or its global quota is complete.

## Rank layout

Ranks are assigned in this fixed order:

1. permanent vLLM DP groups;
2. elastic ranks, initially partitioned into vLLM DP groups;
3. permanent SAE DP groups.

The two roles keep independent topology settings. A vLLM replica occupies
`vllm_tp_size` ranks, while an SAE replica occupies
`sae_tp_size * sae_pp_size` ranks. Therefore the number of elastic ranks must
be divisible by both widths. Moving the whole elastic block changes DP by:

```text
elastic_vllm_dp = elastic_rank_count / vllm_tp_size
elastic_sae_dp  = elastic_rank_count / (sae_tp_size * sae_pp_size)
```

For eight GPUs with TP size 2, permanent vLLM DP 1, and permanent SAE DP 2:

| Ranks | Fixed TP group | Initial role | Expanded role |
| --- | --- | --- | --- |
| 0-1 | permanent vLLM | vLLM | vLLM |
| 2-3 | elastic | vLLM | SAE |
| 4-5 | permanent SAE DP 0 | SAE | SAE |
| 6-7 | permanent SAE DP 1 | SAE | SAE |

This switches between `(vllm_dp=2, sae_dp=2)` and
`(vllm_dp=1, sae_dp=3)`.

An eight-GPU unequal-TP/PP example is `vllm_tp=2`, `sae_tp=1`, `sae_pp=2`,
permanent vLLM DP 1, and permanent SAE DP 1. Ranks 0-1 are permanent vLLM,
ranks 2-5 are elastic, and ranks 6-7 are the two PP stages of the permanent SAE
replica. It switches between `(vllm_dp=3, sae_dp=1)` and
`(vllm_dp=1, sae_dp=3)`.

## Launch

Add the following options to a normal `run_sae_runner_gpu.py` invocation:

```bash
torchrun --standalone --nproc-per-node=8 run_sae_runner_gpu.py \
  --elastic-streaming \
  --streaming-dp-batch-mode exact \
  --sae-dp-mode ddp \
  --vllm-tp-size 2 \
  --sae-tp-size 1 \
  --sae-pp-size 2 \
  --elastic-permanent-vllm-dp-size 1 \
  --elastic-permanent-sae-dp-size 1 \
  --hook-names model.layers.8,model.layers.16 \
  ...
```

`--elastic-streaming` (short form `-es`) enables streaming implicitly, so it
does not need to be combined with `--streaming-mode`. The elastic control path
defaults to `/tmp/sae-elastic/control.json`; override it with
`--elastic-streaming-control-path` when multiple jobs need separate control
files. Setting a control path by itself does not enable elastic streaming.

The elastic layout determines the initial `--vllm-dp-size` and
`--sae-dp-size`; values supplied for those two arguments are replaced. Training
tokens must be a multiple of the global training batch size, and that batch size
must be at least the maximum SAE DP size.

## Manual switching

Inspect the live state:

```bash
python scripts/elastic_streaming_control.py \
  /tmp/sae-elastic/control.json status
```

Move the elastic ranks to SAE and wait for the cutover to commit:

```bash
python scripts/elastic_streaming_control.py \
  /tmp/sae-elastic/control.json switch --sae-dp 3 --wait
```

Move it back to vLLM:

```bash
python scripts/elastic_streaming_control.py \
  /tmp/sae-elastic/control.json switch --sae-dp 1 --wait
```

## Automatic switching

Run the controller as a separate process; it may be started before or after the
training workers:

```bash
python -u scripts/elastic_streaming_control.py \
  /tmp/sae-elastic/control.json auto \
  --low-watermark 0.25 \
  --high-watermark 0.75 \
  --rate-window-seconds 10 \
  --stable-samples 3 \
  --cooldown-seconds 60
```

The controller waits up to 600 seconds for the control file, buffer metadata,
and SHM files to become ready. Use `--startup-timeout` to change that limit;
starting the controller before the training workers is supported.

The default policy has two symmetric decisions:

- At minimum SAE DP, buffer occupancy must be at least 75%, and either vLLM
  throughput exceeds SAE throughput by at least 5% or the two rates are tied
  under producer backpressure. After three consecutive samples it moves all
  elastic ranks to SAE.
- At maximum SAE DP, occupancy must be at most 25%, and either SAE throughput
  exceeds vLLM throughput by at least 5% or the two rates are tied while
  consumers wait for data. After three consecutive samples it moves all
  elastic ranks back to vLLM.

Rates use a 10-second sliding window by default. `--min-rate-gap` can require an
additional absolute token/s difference. `--switch-timeout` detects a cutover
that remains in `switching`, while `--cooldown-seconds` prevents oscillation
after a completed cutover. Use `--dry-run` to observe decisions without issuing
requests. Samples and decisions are appended to
`<control_path>.auto.jsonl`; `SIGINT` and `SIGTERM` stop only the controller,
not the training run.

The rates are chunk-equivalent logical tokens per hook flowing through the
buffer. The monitor derives vLLM production from the existing allocation
sequence and SAE drain from allocation minus the change in non-free chunks.
This directly measures whether the shared buffer is growing or shrinking. It
reads the SHM memmaps without taking their lock. Workers do one control-state
write at startup to publish the buffer name and dimensions; there is no
periodic worker-side file write, new collective, or extra CUDA synchronization.
The optional step-window profiler remains useful for validating compute
throughput, but automatic switching does not require or enable it.

When the producer quota is complete, the policy will not shrink SAE and restart
vLLM merely because the final buffer tail is draining.

The expansion request first stops only the elastic vLLM groups at a published
chunk boundary. It drains its asynchronous writer, destroys the vLLM engine,
and releases its model weights, KV cache, runner buffers, and workspaces before
constructing any SAE state. A CUDA-memory guard aborts the transition if vLLM
teardown retained a material allocation; this prevents an elastic rank from
temporarily holding both complete roles. Permanent SAE ranks keep training
during that preparation and stop only at the next optimizer boundary. The final
cutover broadcasts model and live optimizer/trainer state in memory.

On contraction, a replacement producer session is registered before the new
topology is committed. This prevents consumers from observing a transient EOF.
READY chunks, the logical mixer's queued tail data and RNG state, exact global
token/step progress, and each elastic producer's dataset cursor remain live
across both directions.

## GPU utilization diagnostics

The permanent SAE source is selected by its actual rank inside each sorted
PyTorch process group. It is not assumed to be DP group rank 0: after expansion,
elastic ranks can sort before the permanent SAE ranks. Making that assumption
causes the elastic rank to publish empty source batches while the permanent rank
waits, which previously made the highest-numbered permanent SAE GPU appear idle.

For a small SAE, an instantaneous `nvidia-smi` sample can still report low GPU
utilization even when the rank is healthy because its kernels occupy only a
small fraction of each data-wait cycle. Compare logical step/token progress and
the `step_window_profile_sae_rank*.jsonl` records across replicas before treating
the utilization sample as evidence that a rank is not training.

Destroying vLLM also clears its process-local callable and RoPE module caches.
The RoPE cache owns a CUDA `cos_sin_cache` buffer shared by model instances; if
the old buffer storage is released without invalidating that cache, a cold
restart reuses the emptied buffer and fails on its first rotary-embedding
forward. Cache invalidation affects only the elastic process, never a permanent
vLLM rank.

## Multi-hook memory model and the 4-hook OOM

The OOM in a four-hook exact-DP run is a source-rank data-fetch peak, not a
four-hook SAE graph that grows forever. The exact provider keeps the logical
mixing state on one source rank (the rank selected from the SAE-DP group). That
rank therefore owns the GPU tensors below; other SAE-DP ranks receive an exact
slice and do not own the logical mixer.

The analytic model is implemented in
`sae_lens/autoconfig/phase_memory_model.py`. A source-rank estimate can be
generated without starting a distributed job:

```python
from sae_lens.autoconfig.phase_memory_model import (
    SAEPhaseMemoryConfig,
    estimate_phase_memory,
)

estimate = estimate_phase_memory(
    SAEPhaseMemoryConfig(
        d_in=4096,
        d_sae=65536,
        num_hooks=4,
        train_batch_size_tokens=2048,
        dtype="fp32",
        streaming_enabled=True,
        streaming_mix_chunks=8,
        streaming_chunk_size_tokens=8192,
        streaming_prefetch_chunks=2,
        streaming_source_rank=True,
    )
)
print(estimate.peak_mb, estimate.components["streaming"])
```

The calibrated per-rank predictor exposes the same source-rank terms from the
command line. Its streaming flags are off by default, so existing v4 reports
are unchanged:

```bash
python scripts/predict_sae_memory.py \
  --d-in 4096 --d-sae 65536 --batch 2048 --dtype fp32 \
  --hooks 4 --tp 1 --dp-size 1 --dp-mode ddp \
  --streaming-enabled --streaming-source-rank \
  --streaming-mix-chunks 8 --streaming-chunk-size-tokens 8192 \
  --streaming-prefetch-chunks 2
```

For `H` hooks, element size `D` bytes, logical stream count `S`, and mixer
capacity `C` tokens per stream, the important terms are:

```text
C = max(ceil(global_batch / S), explicit_buffer, mix_chunks * chunk_tokens)
mixer storage         = S * C * H * d_in * D
prefetch/reinterleave = prefetch_chunks * chunk_tokens * H * d_in * D
refill concat peak    = H * (C + ceil(global_batch / S)) * d_in * D
shuffle copy          = mixer storage  (when mix_fraction > 0)
```

In the diagnostic run, `H=4`, `d_in=4096`, `D=4`, `S=1`, and
`8 * 8192 = 65536` tokens. The mixer backing tensor alone is therefore 4 GiB;
two prefetched chunks add 1 GiB. During refill, `_cat_batches` creates a new
tensor before the old one is released. With 36,864 rows, one hook's temporary
tensor is `36864 * 4096 * 4 = 576 MiB`, which is the allocation named in the
CUDA exception. The shuffle path can create another full-buffer copy. These
are avoidable buffer peaks, and are independent of whether SAE forward is
sequential or combined.

The `forward_bytes`, `backward_bytes`, and `optimizer_bytes` fields include
the source rank's resident streaming buffers. `peak_bytes` additionally takes
the data-fetch refill/shuffle peak. `reserved but unallocated` memory from the
CUDA caching allocator is intentionally not counted as a live tensor; it is a
fragmentation/high-water diagnostic. The model also captures the actual
multi-hook SAE terms: parameters and Adam states scale with `num_hooks`, while
TP divides only the `d_sae` dimensions and DDP bucket storage is added only
when `gradient_as_bucket_view=False`.

The runtime phase profiler follows exact-provider `_source` links, logical
mixer generator locals, and the reinterleaved provider `_pool`, so
`data_provider_buffers_mb` now exposes these root-owned tensors instead of
silently placing them in `unattributed_allocated_mb`.

For the reported failure, the practical mitigations are:

- set `--streaming-mix-chunks 0` when temporal mixing is not required; this
  removes the 65,536-token logical mixer and leaves only a one-batch queue;
- keep `PYTORCH_ALLOC_CONF=expandable_segments:True` for long runs, so the
  allocator is less likely to fail a contiguous refill allocation because of
  reserved-block fragmentation;
- reduce `--streaming-chunk-size-tokens`, `--streaming-prefetch-chunks`, or the
  global training batch if the source rank still has insufficient headroom;
- treat rank-local source ownership as intentional. Moving or eliminating the
  logical mixer requires changing exact-DP semantics, not merely changing the
  root rank label.

## Current constraints

- vLLM TP and SAE TP are fixed independently for one run; SAE PP is also fixed.
- The elastic rank count must be divisible by both `vllm_tp_size` and
  `sae_tp_size * sae_pp_size`.
- The SHM transport currently requires all ranks to run on one node.
- SAE DP mode must be DDP, and transport must be SHM.
- Exact streaming batch mode is required.
- All elastic ranks switch together; partial elastic-rank selection is not
  implemented yet.
- A run starts at minimum SAE DP / maximum vLLM DP.
- Checkpoint resume and the topology supervisor are separate paths and cannot be
  combined with elastic streaming.
