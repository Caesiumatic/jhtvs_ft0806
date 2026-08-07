from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

from jhtvs_ft0806.cli import COMMANDS, build_parser, main
from jhtvs_ft0806.geometry.topology import (
    build_repeat_chain,
    canonical_smiles,
    molecule_from_smiles,
    molecular_formula,
)
from jhtvs_ft0806.spec_validation import validate_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"


def test_supplied_spec_passes_all_acceptance_invariants() -> None:
    report = validate_spec(SPEC_DIR)

    assert report.ok, report.to_json()
    assert report.checks["manifest_entries"] == 33
    assert report.checks["calibration_split_counts"] == {
        "test": 14,
        "train": 60,
        "val": 14,
    }
    assert report.checks["complete_sp_reaction_medium_cells"] == 403
    assert report.checks["complete_optfreq_reaction_medium_cells"] == 50
    assert report.checks["expected_fullspace_predictions"] == 5300
    assert report.checks["planned_core_hours"] == {
        "sp": "1601.30",
        "optfreq": "2289.20",
        "total": "3890.50",
    }
    assert report.checks["sigma_exact_six_copy_covers"] == 100
    assert report.checks["sigma_exact_hexamer_reconstructions"] == 100
    assert report.checks["sigma_exact_neutral_dimers"] == 100
    assert report.checks["sigma_link_counts"] == {"C-C": 91, "C-N": 9}


def test_all_frozen_sigma_indices_reconstruct_exact_hexamer_and_dimer_graphs() -> None:
    with (SPEC_DIR / "sigma_coupling_topology.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    link_counts: Counter[str] = Counter()
    for row in rows:
        site_a = int(row["site_a_atom_index_0based"])
        site_b = int(row["site_b_atom_index_0based"])
        monomer_smiles = row["source_monomer_smiles"]
        reconstructed_hexamer = build_repeat_chain(
            monomer_smiles, site_a, site_b, copies=6
        )
        reconstructed_dimer = build_repeat_chain(
            monomer_smiles, site_a, site_b, copies=2
        )

        assert canonical_smiles(reconstructed_hexamer) == canonical_smiles(
            molecule_from_smiles(row["source_hexamer_smiles"])
        ), row["parent_id"]
        assert canonical_smiles(reconstructed_dimer) == canonical_smiles(
            molecule_from_smiles(row["neutral_dimer_smiles"])
        ), row["parent_id"]
        assert molecular_formula(reconstructed_dimer) == row["neutral_dimer_formula"]
        monomer = molecule_from_smiles(monomer_smiles)
        assert all(
            monomer.GetAtomWithIdx(index).GetTotalNumHs(includeNeighbors=True) >= 1
            for index in (site_a, site_b)
        ), row["parent_id"]
        link_counts[row["link_atom_pair"]] += 1

    assert len(rows) == 100
    assert link_counts == Counter({"C-C": 91, "C-N": 9})
    assert {
        row["parent_id"] for row in rows if row["link_atom_pair"] == "C-N"
    } == {f"M{index:03d}" for index in range(60, 69)}


def test_manifest_detects_byte_level_drift(tmp_path: Path) -> None:
    copied_spec = tmp_path / "spec"
    shutil.copytree(SPEC_DIR, copied_spec)
    shutil.copy2(REPOSITORY_ROOT / "AGENTS.md", tmp_path / "AGENTS.md")
    target = copied_spec / "03_ACCEPTANCE.md"
    target.write_bytes(target.read_bytes() + b"\n")

    report = validate_spec(copied_spec)

    assert not report.ok
    assert {issue.code for issue in report.issues} >= {
        "manifest_bytes",
        "manifest_sha256",
    }


def test_every_required_cli_command_is_registered() -> None:
    help_text = build_parser().format_help()
    for command in COMMANDS:
        assert command in help_text


def test_validate_spec_cli_returns_success(capsys) -> None:
    result = main(["--log-level", "ERROR", "validate-spec", "--spec-dir", str(SPEC_DIR)])
    captured = capsys.readouterr()

    assert result == 0
    assert '"status": "PASS"' in captured.out
