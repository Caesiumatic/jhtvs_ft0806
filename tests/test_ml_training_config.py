from __future__ import annotations

import json
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


def test_lop_ml_environment_lock_records_exact_runtime_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "config" / "ml_environment_lock.json").read_text())
    assert lock["python_version"] == "3.11.15"
    assert lock["packages"]["mace-torch"]["version"] == "0.3.16"
    assert lock["packages"]["graph_longrange"]["version"] == "0.4.0"
    assert lock["packages"]["torch"]["version"] == "2.6.0"
    assert all(
        len(package["module_sha256"]) == 64
        for package in lock["packages"].values()
    )
    assert lock["smoke_test"]["passed"] == 72
