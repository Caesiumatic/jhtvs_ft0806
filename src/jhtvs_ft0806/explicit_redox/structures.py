from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

KCAL_MOL_TO_EV = 0.0433641153087705


@dataclass(frozen=True)
class ConformerRecord:
    conformer_id: int
    embed_seed: int
    geometry_sha256: str
    mmff_variant: str
    mmff_energy_eV: float | None
    xyz_path: str


def xyz_text(molecule: Chem.Mol, conformer_id: int, comment: str) -> str:
    conformer = molecule.GetConformer(conformer_id)
    lines = [str(molecule.GetNumAtoms()), comment]
    for atom in molecule.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<2} {point.x: .10f} {point.y: .10f} {point.z: .10f}")
    return "\n".join(lines) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def generate_conformers(
    smiles: str,
    *,
    seed: int,
    count: int = 50,
    rmsd_threshold_A: float = 0.35,
) -> tuple[Chem.Mol, list[tuple[int, float | None, str]]]:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    params.useRandomCoords = False
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=count, params=params))
    if not conformer_ids:
        raise RuntimeError(f"ETKDGv3 produced no conformers for {smiles}")

    variant = "not_available"
    energies: dict[int, float | None] = {conformer_id: None for conformer_id in conformer_ids}
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        for candidate in ("MMFF94s", "MMFF94"):
            properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant=candidate)
            if properties is None:
                continue
            variant = candidate
            for conformer_id in conformer_ids:
                forcefield = AllChem.MMFFGetMoleculeForceField(
                    molecule, properties, confId=conformer_id
                )
                forcefield.Minimize(maxIts=1000)
                energies[conformer_id] = forcefield.CalcEnergy() * KCAL_MOL_TO_EV
            break

    heavy = Chem.RemoveHs(molecule)
    ordered = sorted(
        conformer_ids,
        key=lambda item: (
            energies[item] is None,
            energies[item] if energies[item] is not None else 0.0,
            item,
        ),
    )
    kept: list[int] = []
    for conformer_id in ordered:
        if all(
            rdMolAlign.GetBestRMS(heavy, heavy, prbId=conformer_id, refId=other) >= rmsd_threshold_A
            for other in kept
        ):
            kept.append(conformer_id)
    return molecule, [(item, energies[item], variant) for item in kept]


def write_conformer_set(
    smiles: str,
    *,
    species_id: str,
    seed: int,
    output_dir: Path,
    count: int = 50,
    rmsd_threshold_A: float = 0.35,
) -> list[ConformerRecord]:
    molecule, conformers = generate_conformers(
        smiles, seed=seed, count=count, rmsd_threshold_A=rmsd_threshold_A
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ConformerRecord] = []
    for output_id, (conformer_id, energy, variant) in enumerate(conformers):
        text = xyz_text(
            molecule,
            conformer_id,
            f"{species_id} conformer={output_id} ETKDGv3_seed={seed} source_conf={conformer_id}",
        )
        path = output_dir / f"conf-{output_id:03d}.xyz"
        path.write_text(text, encoding="utf-8", newline="\n")
        records.append(
            ConformerRecord(
                conformer_id=output_id,
                embed_seed=seed,
                geometry_sha256=_sha256_text(text),
                mmff_variant=variant,
                mmff_energy_eV=energy,
                xyz_path=path.name,
            )
        )
    (output_dir / "conformers.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records


def select_mace_conformers(
    records: Sequence[ConformerRecord],
    energies_eV: Sequence[float],
    *,
    window_eV: float = 0.25,
    families: Sequence[str] | None = None,
) -> list[ConformerRecord]:
    if len(records) != len(energies_eV) or not records:
        raise ValueError("records and MACE energies must have equal non-zero length")
    if families is not None and len(families) != len(records):
        raise ValueError("conformer families must match records")
    ranked = sorted(zip(records, energies_eV, strict=True), key=lambda item: (item[1], item[0].conformer_id))
    minimum = ranked[0][1]
    eligible = [(record, energy) for record, energy in ranked if energy <= minimum + window_eV]
    selected = [record for record, _ in eligible[:3]]
    if families is not None:
        by_id = {record.conformer_id: family for record, family in zip(records, families, strict=True)}
        eligible_families = {by_id[record.conformer_id] for record, _ in eligible}
        selected_families = {by_id[record.conformer_id] for record in selected}
        missing = sorted(eligible_families - selected_families)
        for family in missing:
            replacement = next(record for record, _ in eligible if by_id[record.conformer_id] == family)
            replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if sum(by_id[item.conformer_id] == by_id[selected[index].conformer_id] for item in selected) > 1
                ),
                None,
            )
            if replace_index is not None:
                selected[replace_index] = replacement
    return selected


def tfsi_family(molecule: Chem.Mol, conformer_id: int) -> str:
    """Classify a TFSI conformer from the CF3-S...S-CF3 pseudo-dihedral."""

    nitrogen = next(
        (
            atom
            for atom in molecule.GetAtoms()
            if atom.GetSymbol() == "N"
            and len([neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "S"]) == 2
        ),
        None,
    )
    if nitrogen is None:
        raise ValueError("molecule is not TFSI")
    sulfurs = sorted(
        (neighbor for neighbor in nitrogen.GetNeighbors() if neighbor.GetSymbol() == "S"),
        key=lambda atom: atom.GetIdx(),
    )
    carbons = []
    for sulfur in sulfurs:
        carbon = next((neighbor for neighbor in sulfur.GetNeighbors() if neighbor.GetSymbol() == "C"), None)
        if carbon is None:
            raise ValueError("TFSI sulfur lacks CF3 carbon")
        carbons.append(carbon)
    angle = abs(
        rdMolTransforms.GetDihedralDeg(
            molecule.GetConformer(conformer_id),
            carbons[0].GetIdx(),
            sulfurs[0].GetIdx(),
            sulfurs[1].GetIdx(),
            carbons[1].GetIdx(),
        )
    )
    return "cis" if angle < 90.0 else "trans"
