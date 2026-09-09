"""Exact global-token batching helpers for SAE data parallelism."""

from __future__ import annotations


def balanced_token_counts(
    tokens: int, replicas: int, step: int = 0
) -> tuple[int, ...]:
    """Partition tokens exactly and rotate remainder ownership by step."""
    if tokens < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens}")
    if replicas < 1:
        raise ValueError(f"replicas must be >= 1, got {replicas}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    quotient, remainder = divmod(tokens, replicas)
    counts = [quotient] * replicas
    for offset in range(remainder):
        counts[(replicas - remainder + step + offset) % replicas] += 1
    return tuple(counts)


def local_token_budget(
    global_tokens: int,
    global_batch_size: int,
    replicas: int,
    replica_idx: int,
) -> int:
    """Return one replica's exact share across full steps and the final tail."""
    if global_batch_size < 1:
        raise ValueError(
            f"global_batch_size must be >= 1, got {global_batch_size}"
        )
    if not 0 <= replica_idx < replicas:
        raise ValueError(
            f"replica_idx={replica_idx} out of range for replicas={replicas}"
        )
    full_steps, tail = divmod(global_tokens, global_batch_size)
    full_counts = balanced_token_counts(global_batch_size, replicas)
    tail_counts = balanced_token_counts(tail, replicas)
    return full_steps * full_counts[replica_idx] + tail_counts[replica_idx]
