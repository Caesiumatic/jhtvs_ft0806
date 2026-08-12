from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _molecular_sort(row: Mapping[str, str]) -> tuple[int, int, float, str]:
    molecule = Chem.MolFromSmiles(row["canonical_smiles"])
    if molecule is None:
        raise ValueError(f"invalid pilot SMILES: {row['canonical_smiles']}")
    return (
        molecule.GetNumHeavyAtoms(),
        Lipinski.NumRotatableBonds(molecule),
        Descriptors.MolWt(molecule),
        row["system_id"],
    )


def select_pilot(validation_manifest: Path) -> list[dict[str, str]]:
    rows = _read(validation_manifest)
    monomer_acn = [
        row
        for row in rows
        if row["class"] == "monomer" and row["solvent_canonical_smiles"] == "CC#N"
    ]
    monomers = [row for row in rows if row["class"] == "monomer"]
    solvents = [row for row in rows if row["class"] == "solvent"]
    anion_pc = [
        row
        for row in rows
        if row["class"] == "anion" and row["solvent_canonical_smiles"] == "CC1COC(=O)O1"
    ]
    selected = (
        ("small_monomer_acetonitrile", min(monomer_acn, key=_molecular_sort)),
        ("largest_flexible_monomer", max(monomers, key=_molecular_sort)),
        ("solvent_self_solvation", min(solvents, key=_molecular_sort)),
        ("anion_propylene_carbonate", min(anion_pc, key=_molecular_sort)),
    )
    result = []
    for role, row in selected:
        result.append({"pilot_role": role, **row})
    if len({row["system_id"] for row in result}) != 4:
        raise ValueError("pilot selection did not produce four distinct systems")
    return result


def write_pilot(validation_manifest: Path, output: Path) -> list[dict[str, str]]:
    rows = select_pilot(validation_manifest)
    fields = ("pilot_role",) + tuple(key for key in rows[0] if key != "pilot_role")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
