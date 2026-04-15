from collections.abc import Iterator

import torch

ActivationBatch = torch.Tensor | dict[str, torch.Tensor]


def _batch_len(batch: ActivationBatch) -> int:
    if isinstance(batch, dict):
        first = next(iter(batch.values()))
        return first.shape[0]
    return batch.shape[0]


def _cat_batches(left: ActivationBatch, right: ActivationBatch) -> ActivationBatch:
    if isinstance(left, dict):
        assert isinstance(right, dict)
        if left.keys() != right.keys():
            raise ValueError("Activation dict keys must match across buffer refills")
        return {
            key: torch.cat([left[key], right[key]], dim=0)
            for key in left.keys()
        }
    assert isinstance(right, torch.Tensor)
    return torch.cat([left, right], dim=0)


def _index_batch(batch: ActivationBatch, index: torch.Tensor | slice) -> ActivationBatch:
    if isinstance(batch, dict):
        return {key: value[index] for key, value in batch.items()}
    return batch[index]


@torch.no_grad()
def mixing_buffer(
    buffer_size: int,
    batch_size: int,
    activations_loader: Iterator[ActivationBatch],
    mix_fraction: float = 0.5,
    generator: torch.Generator | None = None,
) -> Iterator[ActivationBatch]:
    """
    A generator that maintains a mix of old and new activations for better training.
    It keeps a portion of activations and mixes them with new ones to create batches.

    Args:
        buffer_size: Total size of the buffer
        batch_size: Size of batches to return
        activations_loader: Iterator providing new activations
        mix_fraction: Fraction of buffer to keep for mixing (default 0.5).
                      Higher values mean more temporal mixing but slower throughput.
                      If 0, no shuffling occurs (passthrough mode).

    Yields:
        Batches of activations of shape (batch_size, *activation_dims)
    """

    if buffer_size < batch_size:
        raise ValueError("Buffer size must be greater than or equal to batch size")
    if not 0 <= mix_fraction <= 1:
        raise ValueError("mix_fraction must be in [0, 1]")

    storage_buffer: ActivationBatch | None = None

    for new_activations in activations_loader:
        storage_buffer = (
            new_activations
            if storage_buffer is None
            else _cat_batches(storage_buffer, new_activations)
        )

        if _batch_len(storage_buffer) >= buffer_size:
            if mix_fraction > 0:
                perm = torch.randperm(_batch_len(storage_buffer), generator=generator)
                storage_buffer = _index_batch(storage_buffer, perm)

            # Keep a fixed amount for mixing, serve the rest
            keep_for_mixing = int(buffer_size * mix_fraction)
            num_to_serve = _batch_len(storage_buffer) - keep_for_mixing
            num_serving_batches = max(1, num_to_serve // batch_size)
            serving_cutoff = num_serving_batches * batch_size
            serving_buffer = _index_batch(storage_buffer, slice(0, serving_cutoff))
            storage_buffer = _index_batch(storage_buffer, slice(serving_cutoff, None))

            # Yield batches from the serving_buffer
            for batch_idx in range(num_serving_batches):
                yield _index_batch(
                    serving_buffer,
                    slice(batch_idx * batch_size, (batch_idx + 1) * batch_size),
                )

    # If there are any remaining activations, yield them
    if storage_buffer is not None:
        remaining_batches = _batch_len(storage_buffer) // batch_size
        for i in range(remaining_batches):
            yield _index_batch(storage_buffer, slice(i * batch_size, (i + 1) * batch_size))
