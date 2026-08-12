from __future__ import annotations

import csv
import json
from pathlib import Path

from jhtvs_ft0806.explicit_redox.results import fit_reference


def _write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_reference_fit_is_intercept_only_and_validation_disjoint(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow"
    raw = tmp_path / "raw"
    results = tmp_path / "results"
    calibration = [
        {"system_id": "c1", "canonical_key": "monomer|a|s", "experimental_value_V_vs_AgAgCl": 1.0},
        {"system_id": "c2", "canonical_key": "monomer|b|s", "experimental_value_V_vs_AgAgCl": 2.0},
    ]
    _write(
        workflow / "calibration_manifest.csv",
        list(calibration[0]),
        calibration,
    )
    _write(
        workflow / "validation_manifest.csv",
        ["system_id", "canonical_key"],
        [{"system_id": "v1", "canonical_key": "monomer|v|s"}],
    )
    _write(
        raw / "calibration_system_raw_predictions.csv",
        ["system_id", "delta_F_ox_eV", "raw_voltage_V"],
        [
            {"system_id": "c1", "delta_F_ox_eV": 5.0, "raw_voltage_V": 5.0},
            {"system_id": "c2", "delta_F_ox_eV": 6.0, "raw_voltage_V": 6.0},
        ],
    )
    payload, rows = fit_reference(workflow_dir=workflow, raw_root=raw, results_dir=results)
    assert payload["slope"] == 1.0
    assert payload["C_model_V"] == -4.0
    assert [row["Eox_vs_AgAgCl_V"] for row in rows] == [1.0, 2.0]
    frozen = json.loads((results / "reference_alignment.json").read_text())
    assert frozen["validation_excluded"] is True
