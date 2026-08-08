from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jhtvs_ft0806.ml.dataset import (
    DatasetError,
    assert_no_parent_leakage,
    assert_parent_weights,
    build_reaction_dataset,
    solvent_vectors,
)
from jhtvs_ft0806.provenance import sha256_file


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _feature_rows(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, state_id in enumerate(("R0", "R1", "MPLUS", "DPLUS2", "V0", "V1")):
        path = tmp_path / f"{state_id}.npz"
        np.savez(path, feature_vector=np.asarray([index, index + 1, index + 2.0]))
        rows.append(
            {
                "state_id": state_id,
                "solvent_id": "S001",
                "geometry_hash": str(index) * 64,
                "feature_cache_key": chr(97 + index) * 64,
                "feature_path": str(path),
                "feature_sha256": sha256_file(path),
            }
        )
    return rows


def test_dataset_uses_train_only_normalization_and_frozen_masks(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    feature_index = tmp_path / "features.csv"
    sp_path = tmp_path / "sp.csv"
    final_path = tmp_path / "final.csv"
    _write(feature_index, _feature_rows(tmp_path))
    _write(
        sp_path,
        [
            {
                "reaction_id": "RXN_RED",
                "reaction_class": "redox",
                "role": "monomer",
                "parent_id": "P_RED",
                "solvent_id": "S001",
                "split": "train",
                "stoichiometry": "R0:-1;R1:+1",
                "deltaE_base_MACE_rxn_eV": "2.0",
                "sp_residual_eV": "1.5",
                "qc_status": "clean",
            },
            {
                "reaction_id": "RXN_SIG",
                "reaction_class": "sigma_dimerization",
                "role": "sigma",
                "parent_id": "P_SIG",
                "solvent_id": "S001",
                "split": "train",
                "stoichiometry": "MPLUS:-2;DPLUS2:+1",
                "deltaE_base_MACE_rxn_eV": "-1.0",
                "sp_residual_eV": "2.5",
                "qc_status": "clean",
            },
            {
                "reaction_id": "RXN_VAL",
                "reaction_class": "redox",
                "role": "anion",
                "parent_id": "P_VAL",
                "solvent_id": "S001",
                "split": "val",
                "stoichiometry": "V0:-1;V1:+1",
                "deltaE_base_MACE_rxn_eV": "99.0",
                "sp_residual_eV": "999.0",
                "qc_status": "clean",
            },
        ],
    )
    _write(
        final_path,
        [
            {
                "reaction_id": "RXN_RED",
                "solvent_id": "S001",
                "final_residual_eV": "1.0",
                "rt_correction_eV": "-0.5",
                "qc_status": "clean",
            },
            {
                "reaction_id": "RXN_SIG",
                "solvent_id": "S001",
                "final_residual_eV": "3.0",
                "rt_correction_eV": "0.5",
                "qc_status": "clean",
            },
            {
                "reaction_id": "RXN_VAL",
                "solvent_id": "S001",
                "final_residual_eV": "999.0",
                "rt_correction_eV": "999.0",
                "qc_status": "clean",
            },
        ],
    )
    bundle = build_reaction_dataset(
        repository_root=root,
        spec_dir=root / "spec",
        reaction_sp_path=sp_path,
        reaction_final_path=final_path,
        feature_index_path=feature_index,
    )
    assert len(bundle.examples) == 3
    assert bundle.target_normalization["redox_final"].mean == 1.0
    assert bundle.target_normalization["sigma_final"].mean == 3.0
    assert bundle.target_normalization["sp"].mean == 2.0
    assert bundle.target_normalization["rt"].mean == 0.0
    assert bundle.target_normalization["redox_final"].count == 1
    assert bundle.solvent_mean.tolist() == pytest.approx(
        solvent_vectors(root / "spec")[0]["S001"].tolist()
    )
    assert_no_parent_leakage(bundle.examples)
    assert_parent_weights(bundle.examples)


def test_flagged_rows_are_retained_but_masked(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    feature_index = tmp_path / "features.csv"
    _write(feature_index, _feature_rows(tmp_path))
    sp = tmp_path / "sp.csv"
    final = tmp_path / "final.csv"
    _write(
        sp,
        [
            {
                "reaction_id": "RXN_RED",
                "reaction_class": "redox",
                "role": "monomer",
                "parent_id": "P_RED",
                "solvent_id": "S001",
                "split": "train",
                "stoichiometry": "R0:-1;R1:+1",
                "deltaE_base_MACE_rxn_eV": "2.0",
                "sp_residual_eV": "1.5",
                "qc_status": "flagged",
            },
            {
                "reaction_id": "RXN_SIG",
                "reaction_class": "sigma_dimerization",
                "role": "sigma",
                "parent_id": "P_SIG",
                "solvent_id": "S001",
                "split": "train",
                "stoichiometry": "MPLUS:-2;DPLUS2:+1",
                "deltaE_base_MACE_rxn_eV": "-1.0",
                "sp_residual_eV": "2.5",
                "qc_status": "clean",
            },
            {
                "reaction_id": "RXN_RED_CLEAN",
                "reaction_class": "redox",
                "role": "anion",
                "parent_id": "P_RED_CLEAN",
                "solvent_id": "S001",
                "split": "train",
                "stoichiometry": "V0:-1;V1:+1",
                "deltaE_base_MACE_rxn_eV": "1.0",
                "sp_residual_eV": "0.5",
                "qc_status": "clean",
            },
        ],
    )
    _write(
        final,
        [
            {
                "reaction_id": "RXN_RED",
                "solvent_id": "S001",
                "final_residual_eV": "1.0",
                "rt_correction_eV": "-0.5",
                "qc_status": "flagged",
            },
            {
                "reaction_id": "RXN_SIG",
                "solvent_id": "S001",
                "final_residual_eV": "3.0",
                "rt_correction_eV": "0.5",
                "qc_status": "clean",
            },
            {
                "reaction_id": "RXN_RED_CLEAN",
                "solvent_id": "S001",
                "final_residual_eV": "0.75",
                "rt_correction_eV": "0.25",
                "qc_status": "clean",
            },
        ],
    )
    bundle = build_reaction_dataset(
        repository_root=root,
        spec_dir=root / "spec",
        reaction_sp_path=sp,
        reaction_final_path=final,
        feature_index_path=feature_index,
    )
    flagged = next(example for example in bundle.examples if example.reaction_id == "RXN_RED")
    assert flagged.qc_status_sp == "flagged"
    assert flagged.qc_status_final == "flagged"
    assert not flagged.sp_mask
    assert not flagged.rt_mask
    assert not flagged.final_mask


def test_parent_leakage_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    feature_index = tmp_path / "features.csv"
    sp_path = tmp_path / "sp.csv"
    final_path = tmp_path / "final.csv"
    _write(feature_index, _feature_rows(tmp_path))
    rows = [
        {
            "reaction_id": "RXN_RED",
            "reaction_class": "redox",
            "role": "monomer",
            "parent_id": "P_RED",
            "solvent_id": "S001",
            "split": "train",
            "stoichiometry": "R0:-1;R1:+1",
            "deltaE_base_MACE_rxn_eV": "2.0",
            "sp_residual_eV": "1.5",
            "qc_status": "clean",
        },
        {
            "reaction_id": "RXN_SIG",
            "reaction_class": "sigma_dimerization",
            "role": "sigma",
            "parent_id": "P_SIG",
            "solvent_id": "S001",
            "split": "train",
            "stoichiometry": "MPLUS:-2;DPLUS2:+1",
            "deltaE_base_MACE_rxn_eV": "-1.0",
            "sp_residual_eV": "2.5",
            "qc_status": "clean",
        },
    ]
    _write(sp_path, rows)
    _write(
        final_path,
        [
            {"reaction_id": "RXN_RED", "solvent_id": "S001", "final_residual_eV": "1", "rt_correction_eV": "0", "qc_status": "clean"},
            {"reaction_id": "RXN_SIG", "solvent_id": "S001", "final_residual_eV": "1", "rt_correction_eV": "0", "qc_status": "clean"},
        ],
    )
    bundle = build_reaction_dataset(
        repository_root=root,
        spec_dir=root / "spec",
        reaction_sp_path=sp_path,
        reaction_final_path=final_path,
        feature_index_path=feature_index,
    )
    leaked = (
        bundle.examples[0],
        replace(bundle.examples[0], split="test"),
    )
    with pytest.raises(DatasetError, match="parent split leakage"):
        assert_no_parent_leakage(leaked)
