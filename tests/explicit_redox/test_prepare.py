from __future__ import annotations

import csv
from pathlib import Path

from jhtvs_ft0806.explicit_redox.prepare import build_structure_candidates, collect_entities


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflows" / "mace_polar_5solv_redox"


def test_entity_collection_deduplicates_solvents_and_targets() -> None:
    entities = collect_entities(
        [WORKFLOW / "validation_manifest.csv", WORKFLOW / "calibration_manifest.csv"]
    )
    smiles = [row["canonical_smiles"] for row in entities]
    assert len(smiles) == len(set(smiles))
    acetonitrile = next(row for row in entities if row["canonical_smiles"] == "CC#N")
    assert acetonitrile["roles"] == "solvent;target"


def test_small_structure_build_is_idempotent(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = [
        "system_id",
        "canonical_smiles",
        "solvent_canonical_smiles",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"system_id": "one", "canonical_smiles": "C", "solvent_canonical_smiles": "O"})
    first = build_structure_candidates(manifests=[manifest], output_root=tmp_path / "raw", conformer_count=3)
    second = build_structure_candidates(manifests=[manifest], output_root=tmp_path / "raw", conformer_count=3)
    assert first == second
    assert all(row["requested_conformers"] == 3 for row in first)
