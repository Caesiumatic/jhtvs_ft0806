from __future__ import annotations

from pathlib import Path
import csv
import shutil

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from jhtvs_ft0806.explicit_redox.clusters import pack_systems
from jhtvs_ft0806.explicit_redox.pilot import select_pilot, write_pilot
from jhtvs_ft0806.explicit_redox.structures import xyz_text


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflows" / "mace_polar_5solv_redox"


def test_pilot_selection_covers_required_four_roles(tmp_path: Path) -> None:
    rows = select_pilot(WORKFLOW / "validation_manifest.csv")
    assert [row["pilot_role"] for row in rows] == [
        "small_monomer_acetonitrile",
        "largest_flexible_monomer",
        "solvent_self_solvation",
        "anion_propylene_carbonate",
    ]
    assert rows[0]["species_id"] == "JVM001"
    assert rows[1]["species_id"] == "JVM010"
    assert rows[2]["species_id"] == "JVS001"
    assert rows[3]["species_id"] == "JVA002"
    assert len({row["system_id"] for row in rows}) == 4
    output = tmp_path / "pilot.csv"
    write_pilot(WORKFLOW / "validation_manifest.csv", output)
    assert output.read_text(encoding="utf-8").count("\n") == 5


def _xyz(smiles: str, path: Path, seed: int) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(molecule, params) == 0
    path.write_text(xyz_text(molecule, 0, smiles), encoding="utf-8")


@pytest.mark.skipif(shutil.which("packmol") is None, reason="Packmol is not installed")
def test_cluster_packing_uses_round_robin_selected_conformers(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    target_dir = raw / "target"
    solvent_dir = raw / "solvent"
    target_dir.mkdir()
    solvent_dir.mkdir()
    _xyz("C", target_dir / "a.xyz", 1)
    _xyz("C", target_dir / "b.xyz", 2)
    _xyz("O", solvent_dir / "a.xyz", 3)
    (raw / "structure_candidates.csv").write_text(
        "entity_id,canonical_smiles\ntarget,C\nsolvent,O\n", encoding="utf-8"
    )
    (raw / "isolated_selected.csv").write_text(
        "entity_id,rank,geometry_sha256,optimized_xyz\n"
        "target,0,a,target/a.xyz\n"
        "target,1,b,target/b.xyz\n"
        "solvent,0,c,solvent/a.xyz\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    fields = [
        "system_id", "class", "canonical_smiles", "solvent_canonical_smiles",
        "shell_seed_ids", "lower_charge", "lower_spin", "oxidized_charge", "oxidized_spin",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "system_id": "test", "class": "monomer", "canonical_smiles": "C",
                "solvent_canonical_smiles": "O", "shell_seed_ids": "11;12", "lower_charge": 0,
                "lower_spin": 1, "oxidized_charge": 1, "oxidized_spin": 2,
            }
        )
    rows = pack_systems(manifest=manifest, raw_root=raw, seed_limit=2)
    assert [row["target_conformer_rank"] for row in rows] == ["0", "1"]
    assert all(row["solvent_count"] == 5 and row["status"] == "clean" for row in rows)
