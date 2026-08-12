from __future__ import annotations

import csv
from pathlib import Path

from jhtvs_ft0806.explicit_redox.calibration_manifest import build_calibration


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflows" / "mace_polar_5solv_redox"
FULLSPACE_CALIBRATION = ROOT.parent / "jhtvs_fullspace" / "calib" / "calib_data.csv"


def test_calibration_manifest_is_objective_unique_and_disjoint() -> None:
    included, audit = build_calibration(
        calibration_registry=FULLSPACE_CALIBRATION,
        validation_manifest=WORKFLOW / "validation_manifest.csv",
        monomer_registry=ROOT / "spec" / "source_fullspace_monomers.csv",
    )
    assert included
    assert len({row["canonical_key"] for row in included}) == len(included)
    with (WORKFLOW / "validation_manifest.csv").open(encoding="utf-8", newline="") as handle:
        validation_keys = {row["canonical_key"] for row in csv.DictReader(handle)}
    assert validation_keys.isdisjoint({row["canonical_key"] for row in included})
    assert all(row["decision"] in {"include", "exclude"} for row in audit)
    assert any(row["reason"] == "experimental_medium_mismatch" for row in audit)
    assert any(row["reason"] == "validation_key_overlap" for row in audit)
