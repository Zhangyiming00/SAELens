"""
SAELens GPU memory model for Llama-3.1-8B + TopK SAE on A40.

Derived from empirical measurements in results/results_autocfg_e2e_2 and
results/memory_model_verify (batch-size sweep B=512..8192).
Mean absolute error 0.70%, max 1.67% across all 21 measured rank/config pairs
(batch-size sweep B=512..8192 adds 5 more points, all <0.2% for B>=2048).

Usage:
    python3 scripts/memory_model.py
    python3 scripts/memory_model.py --d-sae 65536 --n-hooks 4 --vllm-tp 2 --sae-tp 2

    # Use run_sae_runner_gpu.py-style args (auto-derives n_batches_in_buffer):
    python3 scripts/memory_model.py \\
        --store-batch-size-prompts 4 --context-size 2048 \\
        --train-batch-size-tokens 2048 --act-store-device cpu
"""

from __future__ import annotations

import argparse
import math


def derive_n_batches_in_buffer(
    train_batch_size_tokens: int,
    context_size: int,
    store_batch_size_prompts: int | None = None,
) -> int:
    """Mirror the logic in run_sae_runner_gpu.py:
    min_n_batches = ceil(train_batch / context_size), then max(2, min_n_batches).
    store_batch_size_prompts is unused in the formula but kept for documentation.
    """
    min_n = math.ceil(train_batch_size_tokens / context_size)
    return max(2, min_n)


def estimate(
    *,
    d_in: int = 4096,
    d_sae: int = 32768,
    dtype_bytes: int = 4,           # float32=4, bfloat16=2
    train_batch_tokens: int = 2048,
    n_batches_in_buffer: int = 2,
    context_size: int = 2048,
    act_store_device: str = "cuda",  # "cuda" or "cpu"
    vllm_tp: int = 1,
    vllm_dp: int = 1,
    sae_tp: int = 1,
    sae_dp_mode: str = "manual",    # "manual", "ddp", "fsdp"
    sae_dp_size: int = 1,
    n_hooks: int = 1,
    is_vllm_rank: bool = True,
    is_sae_rank: bool = True,
    sae_tp_rank: int = 0,           # position within SAE TP group (0 = root)
    gpu_total_mb: int = 45488,
    gpu_memory_utilization: float = 0.5,
) -> dict[str, float]:
    """
    Returns a breakdown dict with keys:
        vllm, sae, adam, grad, fwd, buf, overhead, dp_comm, fsdp_bwd_temp,
        batch_overhead, extra_hooks, total
    All values in MB.

    buf is 0 when act_store_device="cpu" (buffer lives in host RAM, not GPU).

    Empirical corrections (derived from results/results_autocfg_e2e_2 and
    results/memory_model_verify batch-size sweep B=512..8192):
    - vllm_tp=2: uses measured 10323 MB instead of 18484/2 = 9242 MB (+1081 MB)
    - vllm_dp=2 (no SAE DP): extra +577 MB NCCL P2P group buffers per vLLM rank
    - sae_dp_mode='ddp': +798 MB rank0, +865 MB rank1 (allreduce bucket buffers)
    - sae_dp_mode='fsdp' vLLM rank: +798 MB (allgather/reduce-scatter buffers)
    - sae_dp_mode='fsdp' SAE-only rank: overhead=370 MB (not 2000 MB); dp_comm=463 MB;
      fsdp_bwd_temp = 2 * sae_unsharded (unsharded grad + temp copy during backward)
    - batch_overhead (vLLM ranks only): 0.1997*B_eff*log2(B_eff) - 1.9010*B_eff - 501.7 MB
      where B_eff = max(train_batch, context_size); captures allocator fragmentation that
      grows super-linearly with batch size; SAE-only ranks = 0 (no vLLM generate overhead)
    """

    # --- vLLM ---
    # Empirical: tp=1 -> 18484 MB, tp=2 -> 10323 MB (not linear due to KV cache).
    _vllm_by_tp = {1: 18484.0, 2: 10323.0}
    if is_vllm_rank:
        vllm_mb = _vllm_by_tp.get(vllm_tp, 18484.0 / vllm_tp)
    else:
        vllm_mb = 0.0

    # --- SAE parameters ---
    # W_enc (d_in, d_sae) + W_dec (d_sae, d_in) + b_enc (d_sae) + b_dec (d_in)
    # With sae_tp: W_enc, W_dec, b_enc are sharded along d_sae dim.
    sae_full_bytes = (d_in * d_sae + d_sae * d_in + d_sae + d_in) * dtype_bytes
    sae_unsharded_mb = sae_full_bytes / 1024**2 / sae_tp  # per-TP-rank unsharded size
    sae_mb = sae_unsharded_mb if is_sae_rank else 0.0

    # --- Adam optimizer states (exp_avg + exp_avg_sq) ---
    # FSDP shards optimizer state across dp_size; DDP/manual replicates.
    adam_divisor = sae_tp * (sae_dp_size if sae_dp_mode == "fsdp" else 1)
    adam_mb = sae_full_bytes * 2 / 1024**2 / adam_divisor if is_sae_rank else 0.0

    # --- Gradients ---
    # FSDP shards gradients; DDP/manual keeps full grads on each rank.
    grad_divisor = sae_tp * (sae_dp_size if sae_dp_mode == "fsdp" else 1)
    grad_mb = sae_full_bytes / 1024**2 / grad_divisor if is_sae_rank else 0.0

    # --- Forward intermediates (kept for backward) ---
    # hidden_pre: (B, d_sae/sae_tp) local shard before allgather
    # feature_acts: (B, d_sae) full after allgather + topk
    # sae_in + sae_out: (B, d_in) each
    fwd_mb = 0.0
    if is_sae_rank:
        B = train_batch_tokens
        hidden_pre_local_mb = B * (d_sae // sae_tp) * dtype_bytes / 1024**2
        feature_acts_mb = B * d_sae * dtype_bytes / 1024**2
        sae_in_out_mb = B * d_in * dtype_bytes * 2 / 1024**2
        fwd_mb = hidden_pre_local_mb + feature_acts_mb + sae_in_out_mb

    # --- Activation buffer ---
    # mixing_buffer keeps mix_fraction=0.5 of buffer_size tokens.
    # buffer_size = n_batches_in_buffer * context_size
    # When act_store_device="cpu", the buffer lives in host RAM -> 0 GPU MB.
    if act_store_device == "cuda" and is_sae_rank:
        buf_tokens = int(n_batches_in_buffer * context_size * 0.5)
        buf_mb = buf_tokens * d_in * dtype_bytes / 1024**2 * n_hooks
    else:
        buf_mb = 0.0

    # --- Base overhead ---
    # vLLM rank: CUDA allocator + misc (~63 MB).
    # SAE-only DDP/manual rank: CUDA context + NCCL buffers + misc (~2000 MB).
    # SAE-only FSDP rank: much smaller because no vLLM; empirically ~370 MB
    #   (178 MB base + 192 MB dead-feature resampling peak).
    if is_vllm_rank:
        overhead_mb = 63.0
    elif sae_dp_mode == "fsdp":
        overhead_mb = 370.0
    else:
        overhead_mb = 2000.0

    # --- DP communication buffers (empirical corrections) ---
    # vllm_dp=2 without SAE DP: extra NCCL P2P group buffers per vLLM rank.
    dp_comm_mb = 0.0
    if is_vllm_rank and vllm_dp == 2 and sae_dp_size == 1:
        dp_comm_mb += 577.0

    # DDP allreduce bucket buffers: rank0=798 MB, rank1=865 MB.
    if is_sae_rank and sae_dp_mode == "ddp":
        dp_comm_mb += 798.0 if sae_tp_rank == 0 else 865.0

    # FSDP allgather/reduce-scatter buffers: +798 MB on all ranks.
    if is_sae_rank and sae_dp_mode == "fsdp":
        dp_comm_mb += 798.0 if is_vllm_rank else (798.0 - 335.0)

    # --- FSDP backward temporary tensors (SAE-only rank only) ---
    # During backward, FSDP must hold the unsharded gradient (for reduce-scatter)
    # plus a temporary copy, totalling 2 * sae_unsharded MB.
    # This is the dominant peak on SAE-only FSDP ranks (bwd > opt).
    fsdp_bwd_temp_mb = 0.0
    if is_sae_rank and not is_vllm_rank and sae_dp_mode == "fsdp":
        fsdp_bwd_temp_mb = 2.0 * sae_unsharded_mb

    # --- Batch-size-dependent allocator overhead (vLLM ranks only) ---
    # On vLLM ranks, peak memory grows super-linearly with batch size due to
    # vLLM generate temporary tensors + CUDA allocator fragmentation.
    # Empirically fitted from B=512..8192 sweep (results/memory_model_verify):
    #   batch_overhead(B_eff) = 0.1997*B_eff*log2(B_eff) - 1.9010*B_eff - 501.7 MB
    # B_eff = max(train_batch, context_size): vLLM always generates >= context_size tokens
    # per call regardless of per-replica train batch size.
    # SAE-only ranks = 0: no vLLM generate, backward temp tensors already in fsdp_bwd_temp.
    batch_overhead_mb = 0.0
    if is_sae_rank and is_vllm_rank:
        B_eff = max(train_batch_tokens, context_size)
        batch_overhead_mb = 0.1997 * B_eff * math.log2(B_eff) - 1.9010 * B_eff - 501.7

    # --- Extra per additional hook (multi-hook combined backward) ---
    # Empirically: ~1470 MB per hook beyond the first, TP-independent.
    extra_hooks_mb = max(0, n_hooks - 1) * 1470.0

    total = (
        vllm_mb
        + n_hooks * (sae_mb + adam_mb + grad_mb + fwd_mb)
        + buf_mb
        + overhead_mb
        + dp_comm_mb
        + fsdp_bwd_temp_mb
        + batch_overhead_mb
        + extra_hooks_mb
    )

    return {
        "vllm": vllm_mb,
        "sae": sae_mb * n_hooks,
        "adam": adam_mb * n_hooks,
        "grad": grad_mb * n_hooks,
        "fwd": fwd_mb * n_hooks,
        "buf": buf_mb,
        "overhead": overhead_mb,
        "dp_comm": dp_comm_mb,
        "fsdp_bwd_temp": fsdp_bwd_temp_mb,
        "batch_overhead": batch_overhead_mb,
        "extra_hooks": extra_hooks_mb,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SAELens GPU memory estimator")
    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--train-batch-size-tokens", type=int, default=2048,
                        help="Per-replica train batch size in tokens (same as run_sae_runner_gpu.py)")
    parser.add_argument("--store-batch-size-prompts", type=int, default=4,
                        help="Prompts per vLLM call (documents config, not used in formula)")
    parser.add_argument("--n-batches-in-buffer", type=int, default=None,
                        help="Override buffer depth. If None, auto-derived: "
                             "max(2, ceil(train_batch/context_size))")
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--act-store-device", default="cuda", choices=["cuda", "cpu"],
                        help="Where the activation buffer lives. 'cpu' -> 0 GPU MB for buffer.")
    parser.add_argument("--vllm-tp", type=int, default=1)
    parser.add_argument("--vllm-dp", type=int, default=1)
    parser.add_argument("--sae-tp", type=int, default=1)
    parser.add_argument("--sae-dp-mode", choices=["manual", "ddp", "fsdp"], default="manual")
    parser.add_argument("--sae-dp-size", type=int, default=1)
    parser.add_argument("--n-hooks", type=int, default=1)
    parser.add_argument("--gpu-total-mb", type=int, default=45488)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    dtype_bytes = 4 if args.dtype == "float32" else 2

    # Adjust train_batch_tokens for sae_dp (global batch is split across replicas)
    train_batch = args.train_batch_size_tokens // args.sae_dp_size

    # Auto-derive n_batches_in_buffer if not specified (mirrors run_sae_runner_gpu.py)
    if args.n_batches_in_buffer is None:
        n_batches = derive_n_batches_in_buffer(train_batch, args.context_size)
        derived = True
    else:
        n_batches = args.n_batches_in_buffer
        derived = False

    buf_tokens_steady = int(n_batches * args.context_size * 0.5)
    buf_mb_per_hook = buf_tokens_steady * args.d_in * dtype_bytes / 1024**2

    print(f"\nConfig: d_in={args.d_in} d_sae={args.d_sae} dtype={args.dtype}")
    print(f"        vllm_tp={args.vllm_tp} vllm_dp={args.vllm_dp} sae_tp={args.sae_tp} "
          f"sae_dp={args.sae_dp_size} ({args.sae_dp_mode})")
    print(f"        n_hooks={args.n_hooks} train_batch={train_batch} (global={args.train_batch_size_tokens})")
    print(f"        store_batch={args.store_batch_size_prompts} context={args.context_size} "
          f"act_store={args.act_store_device}")
    n_batches_label = f"{n_batches} (auto)" if derived else str(n_batches)
    print(f"        n_batches_in_buffer={n_batches_label}  "
          f"buf_steady={buf_tokens_steady} tokens = {buf_mb_per_hook:.0f} MB/hook "
          f"({'GPU' if args.act_store_device == 'cuda' else 'CPU — 0 GPU MB'})")
    print()

    # Determine rank roles using overlapping topology:
    #   producer ranks: [0, P*vllm_tp)
    #   consumer ranks: [0, Q*sae_tp)
    P, vtp = args.vllm_dp, args.vllm_tp
    Q, stp = args.sae_dp_size, args.sae_tp
    world = max(P * vtp, Q * stp)
    producer_ranks = set(range(P * vtp))
    consumer_ranks = set(range(Q * stp))

    ranks_to_show = []
    seen_roles: set[tuple] = set()
    for r in range(world):
        is_p = r in producer_ranks
        is_c = r in consumer_ranks
        role_key = (is_p, is_c)
        if role_key not in seen_roles:
            seen_roles.add(role_key)
            role_label = ("vLLM" if is_p else "") + ("+SAE" if is_c else " only")
            ranks_to_show.append((f"rank{r} ({role_label})", is_p, is_c, r))

    for rank_name, is_vllm, is_sae, rank_id in ranks_to_show:
        r = estimate(
            d_in=args.d_in,
            d_sae=args.d_sae,
            dtype_bytes=dtype_bytes,
            train_batch_tokens=train_batch,
            n_batches_in_buffer=n_batches,
            context_size=args.context_size,
            act_store_device=args.act_store_device,
            vllm_tp=args.vllm_tp,
            vllm_dp=args.vllm_dp,
            sae_tp=args.sae_tp,
            sae_dp_mode=args.sae_dp_mode,
            sae_dp_size=args.sae_dp_size,
            n_hooks=args.n_hooks,
            is_vllm_rank=is_vllm,
            is_sae_rank=is_sae,
            sae_tp_rank=rank_id,
            gpu_total_mb=args.gpu_total_mb,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        print(f"  {rank_name}:")
        print(f"    vLLM:       {r['vllm']:>7.0f} MB")
        print(f"    SAE params: {r['sae']:>7.0f} MB")
        print(f"    Adam:       {r['adam']:>7.0f} MB")
        print(f"    Gradients:  {r['grad']:>7.0f} MB")
        print(f"    Fwd graph:  {r['fwd']:>7.0f} MB")
        buf_label = f"{r['buf']:>7.0f} MB" if args.act_store_device == "cuda" else "      0 MB (CPU)"
        print(f"    Act buffer: {buf_label}")
        print(f"    Overhead:   {r['overhead']:>7.0f} MB")
        if r["dp_comm"] != 0:
            print(f"    DP comm:    {r['dp_comm']:>7.0f} MB")
        if r["fsdp_bwd_temp"] > 0:
            print(f"    FSDP bwd:   {r['fsdp_bwd_temp']:>7.0f} MB")
        if r["batch_overhead"] != 0:
            print(f"    Batch ovhd: {r['batch_overhead']:>7.0f} MB")
        if r["extra_hooks"] > 0:
            print(f"    Multi-hook: {r['extra_hooks']:>7.0f} MB")
        print(f"    ─────────────────────")
        print(f"    TOTAL:      {r['total']:>7.0f} MB  ({r['total']/args.gpu_total_mb*100:.1f}% of {args.gpu_total_mb} MB)")
        headroom = args.gpu_total_mb - r["total"]
        print(f"    Headroom:   {headroom:>7.0f} MB")
        print()


if __name__ == "__main__":
    main()
