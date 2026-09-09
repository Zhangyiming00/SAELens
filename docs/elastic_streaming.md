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
  --streaming-mode \
  --streaming-dp-batch-mode exact \
  --sae-dp-mode ddp \
  --vllm-tp-size 2 \
  --sae-tp-size 1 \
  --sae-pp-size 2 \
  --elastic-streaming-control-path /tmp/sae-elastic/control.json \
  --elastic-permanent-vllm-dp-size 1 \
  --elastic-permanent-sae-dp-size 1 \
  --hook-names model.layers.8,model.layers.16 \
  ...
```

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

Run the controller as a separate process after the workers have created the
control file:

```bash
python -u scripts/elastic_streaming_control.py \
  /tmp/sae-elastic/control.json auto \
  --low-watermark 0.25 \
  --high-watermark 0.75 \
  --rate-window-seconds 10 \
  --stable-samples 3 \
  --cooldown-seconds 60
```

The default policy has two symmetric decisions:

- At minimum SAE DP, buffer occupancy must be at least 75%, and vLLM
  throughput must exceed SAE throughput by at least 5% for three consecutive
  samples. It then moves all elastic ranks to SAE.
- At maximum SAE DP, occupancy must be at most 25%, and SAE throughput must
  exceed vLLM throughput by at least 5% for three consecutive samples. It then
  moves all elastic ranks back to vLLM.

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
