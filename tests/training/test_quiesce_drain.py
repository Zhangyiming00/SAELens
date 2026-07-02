from types import SimpleNamespace

import torch

from sae_lens.training.sae_trainer import SAETrainer


class _FakeSAE:
    device = torch.device("cpu")
    cfg = SimpleNamespace(normalize_activations=None)

    def to(self, device):
        return self


class _FakeScaler:
    scaling_factor = None

    def __call__(self, batch):
        return batch


class _DrainProvider:
    def __init__(self, num_batches: int) -> None:
        self.remaining = num_batches
        self.drain_requested = False

    def request_drain_local_pool(self) -> None:
        self.drain_requested = True

    def __next__(self) -> torch.Tensor:
        if not self.drain_requested or self.remaining <= 0:
            raise StopIteration
        self.remaining -= 1
        return torch.zeros(1, 2)


def _minimal_trainer(provider: _DrainProvider, total_training_samples: int) -> SAETrainer:
    trainer = SAETrainer.__new__(SAETrainer)
    trainer.sae = _FakeSAE()
    trainer._base_sae = trainer.sae
    trainer.cfg = SimpleNamespace(
        device=torch.device("cpu"),
        total_training_samples=total_training_samples,
        logger=SimpleNamespace(log_to_wandb=False),
        save_final_checkpoint=False,
    )
    trainer.activation_scaler = _FakeScaler()
    trainer.data_provider = provider
    trainer.n_training_samples = 0
    trainer.n_training_steps = 0
    trainer.saved_checkpoints = []

    trainer._is_metric_writer_rank = lambda: True
    trainer._maybe_synchronize_timing = lambda: None
    trainer._consume_data_provider_timing = lambda: {
        "vllm_step_time_s": 0.0,
        "transfer_time_s": 0.0,
    }
    trainer._train_step = lambda sae, sae_in: (SimpleNamespace(), 0.0, {})
    trainer._record_mse_if_needed = lambda step_output: None
    trainer._record_timing_if_needed = lambda **kwargs: None
    trainer._record_memory_if_needed = lambda memory_stats: None
    trainer._checkpoint_if_needed = lambda: None
    trainer._update_pbar = lambda step_output, pbar: None
    trainer._log_train_step = lambda step_output: None
    trainer._run_and_log_evals = lambda: None
    trainer.save_checkpoint = lambda checkpoint_name: trainer.saved_checkpoints.append(
        checkpoint_name
    )
    return trainer


def test_quiesce_drain_continues_past_total_samples_before_ack(tmp_path):
    request = tmp_path / "stop_acquire"
    drain_ack = tmp_path / "drain_ack"
    finished_ack = tmp_path / "finished_ack"
    request.touch()

    provider = _DrainProvider(num_batches=3)
    trainer = _minimal_trainer(provider, total_training_samples=2)

    trainer.fit(
        quiesce_request_path=request,
        quiesce_drain_ack_path=drain_ack,
        quiesce_finished_ack_path=finished_ack,
    )

    assert provider.remaining == 0
    assert trainer.n_training_steps == 3
    assert drain_ack.exists()
    assert finished_ack.exists()
    assert trainer.saved_checkpoints == ["quiesce_3"]
