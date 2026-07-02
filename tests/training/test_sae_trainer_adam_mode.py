from __future__ import annotations

from sae_lens.training.sae_trainer import _adam_optimizer_kwargs_from_env


def test_adam_optimizer_kwargs_from_env_enables_fused(monkeypatch) -> None:
    monkeypatch.setenv("SAE_FUSED_ADAM", "1")

    assert _adam_optimizer_kwargs_from_env() == {"fused": True}


def test_adam_optimizer_kwargs_from_env_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("SAE_FUSED_ADAM", raising=False)

    assert _adam_optimizer_kwargs_from_env() == {}
