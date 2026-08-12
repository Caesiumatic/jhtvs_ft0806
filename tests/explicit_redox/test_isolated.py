from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from jhtvs_ft0806.explicit_redox.isolated import (
    _tfsi_family_from_xyz,
    collect_isolated,
    prepare_isolated_tasks,
)
from jhtvs_ft0806.explicit_redox.structures import xyz_text


def test_isolated_task_table_and_energy_window_selection(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    entity = raw / "isolated" / "iso-one"
    entity.mkdir(parents=True)
    records = []
    for index in range(4):
        path = entity / f"conf-{index:03d}.xyz"
        path.write_text("1\nX\nH 0 0 0\n", encoding="utf-8")
        records.append({"conformer_id": index, "xyz_path": path.name})
    (entity / "conformers.json").write_text(json.dumps(records), encoding="utf-8")
    (raw / "structure_candidates.csv").write_text(
        "entity_id,canonical_smiles,formal_charge,roles,source_system_ids,etkdg_seed,requested_conformers,deduplicated_conformers,directory,metadata_sha256\n"
        "iso-one,C,0,target,s,1,4,4,isolated/iso-one,x\n",
        encoding="utf-8",
    )
    tasks = prepare_isolated_tasks(raw)
    assert len(tasks) == 4
    for task, energy in zip(tasks, [0.0, 0.1, 0.3, 0.2], strict=True):
        output = raw / task["output_dir"]
        output.mkdir(parents=True)
        (output / "result.json").write_text(
            json.dumps({**task, "status": "clean", "energy_eV": energy, "geometry_sha256": str(energy)}),
            encoding="utf-8",
        )
    selected = collect_isolated(raw)
    assert [row["conformer_id"] for row in selected] == [0, 1, 3]
    assert {row["conformer_family"] for row in selected} == {""}


def test_optimized_tfsi_xyz_is_classified_from_its_coordinates(tmp_path: Path) -> None:
    smiles = "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 9712
    assert AllChem.EmbedMolecule(molecule, params) == 0
    path = tmp_path / "tfsi.xyz"
    path.write_text(xyz_text(molecule, 0, "optimized"), encoding="utf-8")
    assert _tfsi_family_from_xyz(smiles, path) in {"cis", "trans"}
