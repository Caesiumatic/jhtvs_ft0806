from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from jhtvs_ft0806.explicit_redox.manifest import (
    EXPECTED_VALIDATION_COUNTS,
    STATE_MATRIX,
    canonical_smiles,
    formal_charge,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflows" / "mace_polar_5solv_redox"


def _rows() -> list[dict[str, str]]:
    with (WORKFLOW / "validation_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_validation_manifest_count_and_unique_canonical_keys() -> None:
    rows = _rows()
    validate_manifest(rows)
    assert len(rows) == 21
    assert Counter(row["class"] for row in rows) == Counter(EXPECTED_VALIDATION_COUNTS)
    assert len({row["canonical_key"] for row in rows}) == 21


def test_canonical_smiles_and_charge_closure() -> None:
    for row in _rows():
        assert canonical_smiles(row["canonical_smiles"]) == row["canonical_smiles"]
        assert canonical_smiles(row["solvent_canonical_smiles"]) == row["solvent_canonical_smiles"]
        assert formal_charge(row["canonical_smiles"]) == (-1 if row["class"] == "anion" else 0)
        assert formal_charge(row["solvent_canonical_smiles"]) == 0


def test_expected_state_charge_spin_matrix() -> None:
    for row in _rows():
        for field, expected in STATE_MATRIX[row["class"]].items():
            assert int(row[field]) == expected


def test_deterministic_five_shell_seeds() -> None:
    for row in _rows():
        seeds = row["shell_seed_ids"].split(";")
        assert len(seeds) == len(set(seeds)) == 5
        assert all(seed.isdecimal() for seed in seeds)


def test_observation_mapping_is_complete_and_reuses_anion_systems() -> None:
    with (WORKFLOW / "validation_observations.csv").open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
    system_ids = {row["system_id"] for row in _rows()}
    assert observations
    assert {row["system_id"] for row in observations} == system_ids
    anion_observations = [row for row in observations if row["class"] == "anion"]
    assert len(anion_observations) > len({row["system_id"] for row in anion_observations}) == 5
