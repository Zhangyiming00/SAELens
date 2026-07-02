#!/usr/bin/env bash
# Re-run the optimizer sweep under TP2 (sae_tp_size=2, 2 ranks via torchrun).
# Matrix: {foreach, forloop} x {1 hook, 2 hooks} x {d_sae 16384, 32768} = 8 runs.
# Uses cached activations (no vLLM). Records the memory_viz pickle at step 15.
set -euo pipefail

cd /home/zhangyiming/SAELens

CACHED=/home/zhangyiming/datasets/cached_activation.new_20260509_140126
OUT_ROOT=results/memory_model/sae_phase_v8/optim_sweep_tp2
HOOK1=blocks.21.hook_resid_post
HOOK2=blocks.21.hook_resid_post,blocks.31.hook_resid_post

mkdir -p "$OUT_ROOT"

run_one() {
  local impl=$1 nhooks=$2 dsae=$3
  local out="$OUT_ROOT/h${nhooks}_${impl}_d${dsae}"
  local hooks_arg
  if [[ "$nhooks" == "1" ]]; then hooks_arg="$HOOK1"; else hooks_arg="$HOOK2"; fi
  echo "=================================================================="
  echo "[RUN] impl=$impl nhooks=$nhooks d_sae=$dsae -> $out"
  echo "=================================================================="
  rm -rf "$out"
  SAE_ADAM_IMPL="$impl" torchrun --nproc_per_node=2 \
    scripts/run_sae_runner_gpu.py \
    --use-cached-activations \
    --cached-activations-path "$CACHED" \
    --hook-names "$hooks_arg" \
    --d-sae "$dsae" \
    --k 128 \
    --tp-size 2 \
    --dtype float32 \
    --training-tokens 61440 \
    --train-batch-size-tokens 2048 \
    --context-size 2048 \
    --store-batch-size-prompts 16 \
    --record-memory-timeline-step 15 \
    --output-path "$out"
}

for dsae in 16384 32768; do
  for impl in foreach forloop; do
    for nhooks in 1 2; do
      run_one "$impl" "$nhooks" "$dsae"
    done
  done
done

echo "ALL TP2 OPTIM SWEEP RUNS COMPLETE"
