"""Uneven filtered rows must not split a shared routing/training loop."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.mixing_buffer import mixing_buffer


def _worker(rank, rendezvous, dict_batch):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous,
        timeout=timedelta(seconds=15),
    )
    try:
        store = object.__new__(ActivationsStore)
        store.device = "cpu"
        cursor = 0

        def phase(value):
            tag = torch.tensor([value])
            tags = [torch.empty_like(tag) for _ in range(2)]
            dist.all_gather(tags, tag)
            assert all(t.item() == value for t in tags), "routing/training diverged"

        def source():
            nonlocal cursor
            while True:
                phase(0)  # all replicas must request the next routing round
                offset = cursor * 64
                cursor += 1
                rows = torch.arange(offset, offset + 64 - rank).unsqueeze(1)
                yield {"h0": rows, "h1": rows + 10000} if dict_batch else rows

        state = {}
        generator = torch.Generator().manual_seed(29)

        def loader():
            return mixing_buffer(
                64, 16, source(), state=state, generator=generator,
                synchronize_batch_count=store._synchronized_serving_batches,
            )

        runtime = SimpleNamespace(training_group=dist.group.WORLD)
        with patch("sae_lens.distributed_v2.get_sae_runtime", return_value=runtime):
            stream = loader()
            outputs = []
            for step in range(10):
                batch = next(stream)
                phase(1)  # all replicas must enter the next training step
                if step == 0:
                    assert cursor == 2  # 64 vs 63 must both wait for round two
                if dict_batch:
                    assert torch.equal(batch["h1"], batch["h0"] + 10000)
                outputs.append(batch)
                if step == 2:
                    saved_state = deepcopy(state)
                    saved_cursor = cursor
                    saved_rng = generator.get_state()
            assert cursor > saved_cursor  # include routing after resume
            state, cursor = saved_state, saved_cursor
            generator.set_state(saved_rng)
            restored = loader()
            for expected in outputs[3:]:
                actual = next(restored)
                phase(1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dict_batch", [False, True])
def test_equal_batches_with_64_vs_63_rows_refill_and_resume_together(tmp_path, dict_batch):
    mp.spawn(
        _worker, args=(f"file://{tmp_path / 'world'}", dict_batch), nprocs=2,
    )
