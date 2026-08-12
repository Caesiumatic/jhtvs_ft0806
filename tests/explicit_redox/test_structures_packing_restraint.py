from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from jhtvs_ft0806.explicit_redox.packing import read_xyz, run_packmol, validate_cluster
from jhtvs_ft0806.explicit_redox.restraint import FlatBottomShell
from jhtvs_ft0806.explicit_redox.structures import (
    generate_conformers,
    select_mace_conformers,
    tfsi_family,
    xyz_text,
)


def _xyz(smiles: str, path: Path, seed: int) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(molecule, params) == 0
    path.write_text(xyz_text(molecule, 0, smiles), encoding="utf-8")


def test_deterministic_conformer_generation_and_selection() -> None:
    first, first_records = generate_conformers("CCO", seed=78123, count=8)
    second, second_records = generate_conformers("CCO", seed=78123, count=8)
    assert xyz_text(first, first_records[0][0], "same") == xyz_text(second, second_records[0][0], "same")
    fake_records = []
    from jhtvs_ft0806.explicit_redox.structures import ConformerRecord

    for index in range(4):
        fake_records.append(ConformerRecord(index, 1, str(index), "MMFF94", None, str(index)))
    selected = select_mace_conformers(fake_records, [0.4, 0.0, 0.1, 0.3])
    assert [record.conformer_id for record in selected] == [1, 2]


def test_tfsi_family_coverage_is_retained_within_energy_window() -> None:
    from jhtvs_ft0806.explicit_redox.structures import ConformerRecord

    records = [ConformerRecord(index, 1, str(index), "MMFF94", None, str(index)) for index in range(4)]
    selected = select_mace_conformers(
        records,
        [0.00, 0.02, 0.03, 0.20],
        families=["cis", "cis", "cis", "trans"],
    )
    assert {record.conformer_id for record in selected} == {0, 1, 3}
    molecule, conformers = generate_conformers(
        "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", seed=7241, count=8
    )
    assert {tfsi_family(molecule, conformer_id) for conformer_id, _, _ in conformers} <= {"cis", "trans"}


@pytest.mark.skipif(shutil.which("packmol") is None, reason="Packmol is not installed")
def test_packmol_exact_five_solvents_atom_order_and_overlap(tmp_path: Path) -> None:
    target = tmp_path / "target.xyz"
    solvent = tmp_path / "solvent.xyz"
    _xyz("c1ccsc1", target, 11)
    _xyz("CC#N", solvent, 12)
    packed = run_packmol(
        target_path=target,
        solvent_path=solvent,
        solvent_smiles="CC#N",
        seed=123456,
        output_dir=tmp_path / "packed",
    )
    assert packed.solvent_count == 5
    assert len(read_xyz(packed.geometry_path).symbols) == packed.target_atoms + 5 * packed.solvent_atoms
    assert packed.minimum_intermolecular_distance_A >= 1.99


def test_overlap_rejection(tmp_path: Path) -> None:
    target = tmp_path / "target.xyz"
    solvent = tmp_path / "solvent.xyz"
    _xyz("C", target, 21)
    _xyz("O", solvent, 22)
    target_geometry = read_xyz(target)
    solvent_geometry = read_xyz(solvent)
    symbols = target_geometry.symbols + solvent_geometry.symbols * 5
    from jhtvs_ft0806.explicit_redox.packing import XYZGeometry

    overlapping = XYZGeometry(symbols=symbols, positions=((0.0, 0.0, 0.0),) * len(symbols))
    with pytest.raises(ValueError, match="overlap"):
        validate_cluster(overlapping, target=target_geometry, solvent=solvent_geometry, tolerance_A=2.0)


def _restraint() -> tuple[FlatBottomShell, np.ndarray]:
    positions = np.zeros((7, 3), dtype=float)
    positions[2:, 0] = [1.0, 2.0, 3.0, 6.0, 8.0]
    restraint = FlatBottomShell(
        target_heavy_indices=[0, 1],
        solvent_groups=[[2], [3], [4], [5], [6]],
        masses=[12.0, 12.0, 18.0, 18.0, 18.0, 18.0, 18.0],
        R0_A=4.0,
        k_eV_A2=0.5,
    )
    return restraint, positions


def test_restraint_force_conservation_and_translation_invariance() -> None:
    restraint, positions = _restraint()
    first = restraint.evaluate(positions)
    translated = restraint.evaluate(positions + np.array([13.2, -7.1, 4.4]))
    np.testing.assert_allclose(first.forces_eV_A.sum(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(first.forces_eV_A, translated.forces_eV_A, atol=1e-14)
    assert first.energy_eV == pytest.approx(translated.energy_eV, abs=1e-14)


def test_restraint_energy_cancels_between_charge_states() -> None:
    restraint, positions = _restraint()
    lower = restraint.evaluate(positions)
    oxidized = restraint.evaluate(positions.copy())
    assert oxidized.energy_eV - lower.energy_eV == 0.0
