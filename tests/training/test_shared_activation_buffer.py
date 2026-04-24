import torch

from sae_lens.training.shared_activation_buffer import ChunkState, SharedActivationBuffer


def _make_buffer(tmp_path, num_chunks=8, chunk_size=16, d_model=4, num_producers=2, target_chunks=100):
    return SharedActivationBuffer(
        name="test_buf",
        num_chunks=num_chunks,
        chunk_size_tokens=chunk_size,
        d_model=d_model,
        num_producers=num_producers,
        target_chunks=target_chunks,
        create=True,
        base_dir=str(tmp_path),
    )


def test_reset_for_restart_clears_writing_chunks(tmp_path):
    buf = _make_buffer(tmp_path)
    idx0, _ = buf.allocate_write_chunk()
    idx1, _ = buf.allocate_write_chunk()
    assert int(buf._state[idx0]) == ChunkState.WRITING
    assert int(buf._state[idx1]) == ChunkState.WRITING

    buf.reset_for_restart(new_num_producers=4)

    assert int(buf._state[idx0]) == ChunkState.FREE
    assert int(buf._state[idx1]) == ChunkState.FREE
    buf.close()


def test_reset_for_restart_updates_header(tmp_path):
    buf = _make_buffer(tmp_path, num_producers=2)
    buf.signal_done()
    assert int(buf._header[1]) == 1  # done_count

    buf.reset_for_restart(new_num_producers=4)

    assert int(buf._header[0]) == 4  # num_producers updated
    assert int(buf._header[1]) == 0  # done_count zeroed
    buf.close()


def test_reset_for_restart_preserves_ready_chunks(tmp_path):
    buf = _make_buffer(tmp_path, num_producers=2)
    idx, _ = buf.allocate_write_chunk()
    buf.write_chunk(idx, torch.zeros(16, 4), valid_tokens=16)
    buf.mark_ready(idx)
    assert int(buf._state[idx]) == ChunkState.READY

    buf.reset_for_restart(new_num_producers=3)

    assert int(buf._state[idx]) == ChunkState.READY
    buf.close()


def test_reset_for_restart_preserves_next_claim_seq(tmp_path):
    buf = _make_buffer(tmp_path, num_producers=1, target_chunks=100)
    for _ in range(3):
        idx, _ = buf.allocate_write_chunk()
        buf.abort_write_chunk(idx)
    seq_before = int(buf._header[3])
    assert seq_before == 3

    buf.reset_for_restart(new_num_producers=2)

    assert int(buf._header[3]) == seq_before
    buf.close()


def test_reset_for_restart_does_not_touch_consuming(tmp_path):
    buf = _make_buffer(tmp_path, num_producers=1)
    idx, _ = buf.allocate_write_chunk()
    buf.write_chunk(idx, torch.zeros(16, 4), valid_tokens=16)
    buf.mark_ready(idx)
    consumed, _ = buf.acquire_up_to(1)
    assert int(buf._state[consumed[0]]) == ChunkState.CONSUMING

    buf.reset_for_restart(new_num_producers=2)

    # reset_for_restart must not touch CONSUMING slots
    assert int(buf._state[consumed[0]]) == ChunkState.CONSUMING
    buf.close()


def test_quiesce_signal_stops_consumer(tmp_path):
    """Consumer's fit() exits after the current step when quiesce_request appears."""
    from datasets import Dataset

    from sae_lens.saes.standard_sae import StandardTrainingSAE
    from sae_lens.training.activations_store import ActivationsStore
    from sae_lens.training.sae_trainer import SAETrainer
    from tests.helpers import TINYSTORIES_MODEL, build_runner_cfg, load_model_cached

    quiesce_request = tmp_path / "quiesce_request"
    quiesce_ack = tmp_path / "quiesce_ack_consumer"

    cfg = build_runner_cfg(
        d_in=64,
        d_sae=128,
        checkpoint_path=str(tmp_path),
        n_checkpoints=0,
        save_final_checkpoint=False,
    )
    model = load_model_cached(TINYSTORIES_MODEL)
    store = ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )
    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    trainer = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=store,
    )

    # Touch quiesce_request after the first step completes.
    original_checkpoint = trainer._checkpoint_if_needed
    step_count = [0]

    def _patched_checkpoint():
        step_count[0] += 1
        if step_count[0] == 1:
            quiesce_request.touch()
        original_checkpoint()

    trainer._checkpoint_if_needed = _patched_checkpoint

    result = trainer.fit(
        quiesce_request_path=quiesce_request,
        quiesce_ack_path=quiesce_ack,
    )

    # Stopped after a small number of steps, not the full training run.
    assert trainer.n_training_steps < 50
    # Ack file was created by the metric-writer rank (rank 0 in non-distributed).
    assert quiesce_ack.exists()
    # Checkpoint was saved.
    from sae_lens.constants import TRAINER_STATE_FILENAME
    ckpt_dirs = [p for p in tmp_path.glob("quiesce_*") if p.is_dir()]
    assert len(ckpt_dirs) >= 1
    assert (ckpt_dirs[0] / TRAINER_STATE_FILENAME).exists()
