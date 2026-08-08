from __future__ import annotations

from pathlib import Path

from jhtvs_ft0806.ml.training import load_training_settings


def test_training_settings_are_loaded_from_the_frozen_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_training_settings(root / "spec")
    assert settings.head_warmup_epochs == 50
    assert settings.head_lr == 3e-4
    assert settings.lora_rank == 4
    assert settings.lora_alpha == 1.0
    assert settings.lora_lr == 1e-4
    assert settings.patience == 30
    assert settings.seeds == (17, 29, 43, 71, 101)
