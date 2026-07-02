#!/usr/bin/env bash
# v9 SAE phase-memory sweep: one-axis-at-a-time (OAT) around a baseline, for each
# of 4 parallelism modes {single, tp2, ddp2, fsdp2}. Every run dumps a
# memory_timeline pickle (step 15) + memory_phase_history jsonl.
#
# Axes: d_in (via hook type), d_sae, H (num hooks same type), B (batch tokens),
#       optimizer impl {fused,foreach,forloop}.
# Baseline: d_in=resid(4096), d_sae=16384, H=1, B=2048, opt=fused.
# Opt axis is swept at H=2 (impl only changes peak for multi-SAE).
set -uo pipefail

cd /home/zhangyiming/SAELens

RUNNER=scripts/run_sae_runner_gpu.py
OUT_ROOT=results/memory_model/sae_phase_v9
LOG="$OUT_ROOT/sweep.log"
mkdir -p "$OUT_ROOT"
: > "$LOG"

# ---- cached-activation sources per hook type ----
CACHE_attn_v=/home/zhangyiming/datasets/cached_activation.attn_v_d1024
CACHE_resid=/home/zhangyiming/datasets/cached_activation.new_20260509_140126
CACHE_mlp=/home/zhangyiming/datasets/cached_activation.mlp_pre_d14336

# layer -> hook name templates (%s = layer)
tmpl_attn_v() { echo "blocks.$1.attn.hook_v"; }
tmpl_resid()  { echo "blocks.$1.hook_resid_post"; }
tmpl_mlp()    { echo "blocks.$1.mlp.hook_pre"; }

# H-layer selection order
LAYERS=(21 31 11 26)

# build a comma-joined hook list of the first H layers for a given type
build_hooks() {
  local type=$1 H=$2 out="" i
  for ((i=0; i<H; i++)); do
    local L=${LAYERS[$i]}
    local h
    case "$type" in
      attn_v) h=$(tmpl_attn_v "$L") ;;
      resid)  h=$(tmpl_resid  "$L") ;;
      mlp)    h=$(tmpl_mlp    "$L") ;;
    esac
    out+="${out:+,}$h"
  done
  echo "$out"
}

cache_for() {
  case "$1" in
    attn_v) echo "$CACHE_attn_v" ;;
    resid)  echo "$CACHE_resid"  ;;
    mlp)    echo "$CACHE_mlp"     ;;
  esac
}

# ---- run one config -------------------------------------------------------
# args: mode type d_sae H B opt slug
run_one() {
  local mode=$1 type=$2 dsae=$3 H=$4 B=$5 opt=$6 slug=$7
  local out="$OUT_ROOT/$mode/$slug"
  local cache; cache=$(cache_for "$type")
  local hooks; hooks=$(build_hooks "$type" "$H")
  local first_hook; first_hook=$(build_hooks "$type" 1)

  # Pick a record step that always exists: total steps = training_tokens / B.
  # Record near the end but before the last step so it always fires (B=4096 gives
  # only 15 steps 0..14, so a fixed step 15 would never run).
  local total_steps=$(( 61440 / B ))
  local rec_step=$(( total_steps - 3 ))
  (( rec_step < 5 )) && rec_step=5

  local -a launch=(python3)
  local -a par=()
  case "$mode" in
    single) launch=(python3);                         par=() ;;
    tp2)    launch=(torchrun --nproc_per_node=2);      par=(--sae-tp-size 2) ;;
    ddp2)   launch=(torchrun --nproc_per_node=2);      par=(--sae-dp-size 2 --sae-dp-mode ddp) ;;
    fsdp2)  launch=(torchrun --nproc_per_node=2);      par=(--sae-dp-size 2 --sae-dp-mode fsdp) ;;
  esac

  echo "==================================================================" | tee -a "$LOG"
  echo "[RUN] mode=$mode type=$type d_sae=$dsae H=$H B=$B opt=$opt -> $out" | tee -a "$LOG"
  echo "      hooks=$hooks" | tee -a "$LOG"
  echo "==================================================================" | tee -a "$LOG"
  rm -rf "$out"; mkdir -p "$out"

  SAE_ADAM_IMPL="$opt" "${launch[@]}" "$RUNNER" \
    --use-cached-activations \
    --cached-activations-path "$cache" \
    --hook-name "$first_hook" \
    --hook-names "$hooks" \
    --d-sae "$dsae" \
    --k 128 \
    --dtype float32 \
    --training-tokens 61440 \
    --train-batch-size-tokens "$B" \
    --context-size 2048 \
    --store-batch-size-prompts 16 \
    --record-memory-timeline-step "$rec_step" \
    "${par[@]}" \
    --output-path "$out" >>"$LOG" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL rc=$rc] $out (see $LOG)" | tee -a "$LOG"
  else
    echo "[OK] $out" | tee -a "$LOG"
  fi
  return 0
}

# ---- OAT config list (shared across modes) --------------------------------
# fields: type dsae H B opt slug
CONFIGS=(
  "resid 16384 1 2048 fused base_resid_d16384_H1_B2048_fused"
  "attn_v 16384 1 2048 fused din_attn_v_d1024"
  "mlp 16384 1 2048 fused din_mlp_d14336"
  "resid 8192 1 2048 fused dsae_8192"
  "resid 32768 1 2048 fused dsae_32768"
  "resid 16384 2 2048 fused H_2"
  "resid 16384 4 2048 fused H_4"
  "resid 16384 1 1024 fused B_1024"
  "resid 16384 1 4096 fused B_4096"
  "resid 16384 2 2048 foreach opt_foreach_H2"
  "resid 16384 2 2048 forloop opt_forloop_H2"
)

MODES=(${MODES_OVERRIDE:-single tp2 ddp2 fsdp2})

for mode in "${MODES[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    read -r type dsae H B opt slug <<<"$cfg"
    run_one "$mode" "$type" "$dsae" "$H" "$B" "$opt" "$slug"
  done
done

echo "ALL v9 RUNS COMPLETE" | tee -a "$LOG"
