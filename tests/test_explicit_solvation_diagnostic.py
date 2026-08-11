from __future__ import annotations

import importlib.util
from pathlib import Path

from jhtvs_ft0806.geometry.xyz import XYZAtom
from jhtvs_ft0806.provenance import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "explicit_solvation_sp"
    / "run_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("explicit_solvation_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_source_snapshots_are_exact_20260707_xyz_bytes() -> None:
    expected = {
        "thiophene.xyz": "2daf887bec19a863dd13e18c5e90f7062b66f2e3c36b598459fbee3adab83e32",
        "dmso.xyz": "f6715492ab438f5d4915191dfa1b095da5b21ed9c0db28477353bb3566a52d09",
        "acetonitrile.xyz": "1a610306653c88bdc334da52c4b2a9b62c35d5323d7b1e8027ce6f0532ea7859",
        "dichloromethane.xyz": "b94f39547c23588dd8d2d732b53b0a9465345b417c597eea8bbb9fe3461620d9",
    }
    root = SCRIPT.parent / "source_geometries"

    assert {name: sha256_file(root / name) for name in expected} == expected


def test_packmol_inputs_pin_counts_boxes_and_seeds() -> None:
    rendered = DIAGNOSTIC.render_packmol_inputs()
    by_name = {path.name: path.read_text(encoding="utf-8") for path in rendered}

    assert len(by_name) == 4
    assert "number 5\n" in by_name["thiophene_acetonitrile_R5.inp"]
    assert "seed 814105\n" in by_name["thiophene_acetonitrile_R5.inp"]
    assert "inside box -5.650 -5.650 -5.650 5.650 5.650 5.650" in by_name[
        "thiophene_acetonitrile_R5.inp"
    ]
    assert "number 50\n" in by_name["dmso_dichloromethane_R50.inp"]
    assert "seed 814250\n" in by_name["dmso_dichloromethane_R50.inp"]
    assert "inside box -10.250 -10.250 -10.250 10.250 10.250 10.250" in by_name[
        "dmso_dichloromethane_R50.inp"
    ]


def test_minimum_distance_ignores_intramolecular_contacts() -> None:
    atoms = (
        XYZAtom("H", 0.0, 0.0, 0.0),
        XYZAtom("H", 0.7, 0.0, 0.0),
        XYZAtom("H", 3.0, 0.0, 0.0),
        XYZAtom("H", 3.7, 0.0, 0.0),
    )

    observed = DIAGNOSTIC._minimum_intermolecular_distance(  # noqa: SLF001
        atoms, (range(0, 2), range(2, 4))
    )

    assert observed == 2.3


def test_orca_runner_accepts_only_the_isolated_diagnostic_path_extension() -> None:
    runner = (REPOSITORY_ROOT / "hpc" / "run_orca.sh").read_text(encoding="utf-8")

    assert (
        "diagnostic_gas_sp:diagnostics/explicit_solvation_sp/orca/jobs/*/*.inp"
        in runner
    )
    assert "diagnostics/explicit_solvation_sp/orca/jobs/*/*.out" in runner
