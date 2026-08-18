#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from common import (
    ANIONS,
    CATIONS_BY_ANION,
    SEED,
    SOLVENTS,
    TOPOLOGIES,
    Atom,
    cation_solvent_keys,
    composition_keys,
    repo_root,
    species_table,
    write_csv,
    write_xyz,
)

MIN_HEAVY_DISTANCE_ANG = 2.0
INITIAL_ANCHOR_DISTANCE_ANG = 3.5
CLASH_INCREMENT_ANG = 0.25
MAX_ANCHOR_DISTANCE_ANG = 10.0


@dataclass
class Component:
    species_id: str
    atoms: list[Atom]
    anchor: int
    formal_charge: int
    geometry_method: str


def _anchor_index(mol: Chem.Mol, species_id: str) -> int:
    if species_id in {"EMIM", "BMIM", "HMIM"}:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C" or not atom.GetIsAromatic():
                continue
            ring_n = sum(n.GetSymbol() == "N" and n.GetIsAromatic() for n in atom.GetNeighbors())
            if ring_n == 2:
                return atom.GetIdx()
        raise ValueError(f"imidazolium C2 anchor not found for {species_id}")
    target = {
        "BMPYRR": ("N", None),
        "NTF2": ("N", -1),
        "OTF": ("S", None),
        "PF6": ("P", None),
        "PC": ("O", "carbonyl"),
        "EG": ("O", "first"),
        "THF": ("O", "first"),
    }[species_id]
    candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == target[0]]
    if species_id == "NTF2":
        candidates = [a for a in candidates if a.GetFormalCharge() == -1]
    if species_id == "PC":
        candidates = [
            a
            for a in candidates
            if any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
        ]
    if not candidates:
        raise ValueError(f"anchor not found for {species_id}")
    return min(a.GetIdx() for a in candidates)


def _pf6_component(row: dict[str, str]) -> Component:
    mol = Chem.MolFromSmiles(row["smiles"])
    anchor = _anchor_index(mol, "PF6")
    bond = 1.58
    positions = [(bond, 0, 0), (-bond, 0, 0), (0, bond, 0), (0, -bond, 0), (0, 0, bond), (0, 0, -bond)]
    atoms: list[Atom] = []
    p_added = False
    f_index = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "P":
            atoms.append(Atom("P", 0.0, 0.0, 0.0))
            p_added = True
        else:
            xyz = positions[f_index]
            atoms.append(Atom("F", *xyz))
            f_index += 1
    if not p_added or f_index != 6:
        raise ValueError("PF6 ideal-octahedral construction failed")
    return Component("PF6", atoms, anchor, -1, "ideal_octahedral_PF6")


def build_component(species_id: str) -> Component:
    row = species_table()[species_id]
    if species_id == "PF6":
        return _pf6_component(row)
    mol = Chem.AddHs(Chem.MolFromSmiles(row["smiles"]))
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.useRandomCoords = False
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit embedding failed for {species_id}")
    method = "ETKDGv3"
    if AllChem.MMFFHasAllMoleculeParams(mol):
        result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=1000)
        if result < 0:
            raise RuntimeError(f"MMFF94 failed for {species_id}")
        method = "ETKDGv3+MMFF94"
    conformer = mol.GetConformer()
    atoms = [
        Atom(atom.GetSymbol(), *(float(v) for v in conformer.GetAtomPosition(atom.GetIdx())))
        for atom in mol.GetAtoms()
    ]
    return Component(
        species_id,
        atoms,
        _anchor_index(mol, species_id),
        int(row["formal_charge"]),
        method,
    )


def _translated(component: Component, anchor_x: float) -> list[Atom]:
    anchor = component.atoms[component.anchor]
    centered = [np.array([a.x - anchor.x, a.y - anchor.y, a.z - anchor.z]) for a in component.atoms]
    ranges = np.ptp(np.stack(centered), axis=0)
    # Put the narrowest deterministic component axis on the assembly x axis.
    # The three cyclic permutations are proper rotations (not reflections).
    narrowest = int(np.argmin(ranges))
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    permutation = permutations[narrowest]
    oriented = [coords[list(permutation)] for coords in centered]
    return [Atom(a.element, coords[0] + anchor_x, coords[1], coords[2]) for a, coords in zip(component.atoms, oriented, strict=True)]


def _minimum_heavy_distance(left: list[Atom], right: list[Atom]) -> float:
    return min(
        np.linalg.norm(np.array([a.x - b.x, a.y - b.y, a.z - b.z]))
        for a in left
        for b in right
        if a.element != "H" and b.element != "H"
    )


def assemble(ordered: list[Component]) -> tuple[list[Atom], dict]:
    left_distance = INITIAL_ANCHOR_DISTANCE_ANG
    right_distance = INITIAL_ANCHOR_DISTANCE_ANG
    while True:
        placed = [
            _translated(ordered[0], -left_distance),
            _translated(ordered[1], 0.0),
            _translated(ordered[2], right_distance),
        ]
        d01 = _minimum_heavy_distance(placed[0], placed[1])
        d12 = _minimum_heavy_distance(placed[1], placed[2])
        d02 = _minimum_heavy_distance(placed[0], placed[2])
        if d01 >= MIN_HEAVY_DISTANCE_ANG and d12 >= MIN_HEAVY_DISTANCE_ANG and d02 >= MIN_HEAVY_DISTANCE_ANG:
            break
        if d01 < MIN_HEAVY_DISTANCE_ANG:
            left_distance += CLASH_INCREMENT_ANG
        if d12 < MIN_HEAVY_DISTANCE_ANG:
            right_distance += CLASH_INCREMENT_ANG
        if d02 < MIN_HEAVY_DISTANCE_ANG:
            left_distance += CLASH_INCREMENT_ANG / 2
            right_distance += CLASH_INCREMENT_ANG / 2
        if max(left_distance, right_distance) > MAX_ANCHOR_DISTANCE_ANG:
            raise RuntimeError("deterministic clash removal exceeded maximum anchor distance")

    atoms: list[Atom] = []
    fragments: dict[str, dict] = {}
    anchor_indices: dict[str, int] = {}
    for component, component_atoms in zip(ordered, placed, strict=True):
        start = len(atoms)
        atoms.extend(component_atoms)
        stop = len(atoms)
        fragment = {"cation": "C", "anion": "A", "solvent": "S"}[species_table()[component.species_id]["kind"]]
        fragments[fragment] = {
            "species_id": component.species_id,
            "atom_indices_zero_based": list(range(start, stop)),
            "geometry_method": component.geometry_method,
        }
        anchor_indices[fragment] = start + component.anchor
    metadata = {
        "fragments": fragments,
        "anchor_indices_zero_based": anchor_indices,
        "initial_anchor_distances_ang": {"left_middle": left_distance, "middle_right": right_distance},
        "minimum_heavy_distance_ang": min(d01, d12, d02),
        "formal_charge": sum(c.formal_charge for c in ordered),
        "rdkit_seed": SEED,
    }
    return atoms, metadata


def assemble_pair(anion: Component, solvent: Component) -> tuple[list[Atom], dict]:
    dummy = Component("_dummy", [Atom("He", 0.0, 0.0, 0.0)], 0, 0, "dummy")
    distance = INITIAL_ANCHOR_DISTANCE_ANG
    while True:
        placed_a = _translated(anion, -distance / 2)
        placed_s = _translated(solvent, distance / 2)
        if _minimum_heavy_distance(placed_a, placed_s) >= MIN_HEAVY_DISTANCE_ANG:
            break
        distance += CLASH_INCREMENT_ANG
    atoms = placed_a + placed_s
    return atoms, {
        "fragments": {
            "A": {"species_id": anion.species_id, "atom_indices_zero_based": list(range(len(placed_a))), "geometry_method": anion.geometry_method},
            "S": {"species_id": solvent.species_id, "atom_indices_zero_based": list(range(len(placed_a), len(atoms))), "geometry_method": solvent.geometry_method},
        },
        "anchor_indices_zero_based": {"A": anion.anchor, "S": len(placed_a) + solvent.anchor},
        "initial_anchor_distance_ang": distance,
        "formal_charge": -1,
        "rdkit_seed": SEED,
    }


def _write_structure(path: Path, atoms: list[Atom], metadata: dict) -> None:
    metadata["structure_id"] = path.stem
    write_xyz(path, atoms, json.dumps({"structure_id": path.stem}, separators=(",", ":")))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def generate(output_dir: Path) -> list[dict]:
    components = {species_id: build_component(species_id) for species_id in species_table()}
    rows: list[dict] = []
    for cation, anion, solvent in composition_keys():
        mapping = {"C": components[cation], "A": components[anion], "S": components[solvent]}
        for topology in TOPOLOGIES:
            atoms, metadata = assemble([mapping[label] for label in topology])
            metadata.update({"kind": "triad", "cation": cation, "anion": anion, "solvent": solvent, "topology": topology})
            path = output_dir / "initial" / f"triad__{cation}__{anion}__{solvent}__{topology}.xyz"
            _write_structure(path, atoms, metadata)
            rows.append({"structure_id": path.stem, "kind": "triad", "cation": cation, "anion": anion, "solvent": solvent, "topology": topology, "formal_charge": 0, "xyz_path": _manifest_path(path)})
    for anion in ANIONS:
        for solvent in SOLVENTS:
            atoms, metadata = assemble_pair(components[anion], components[solvent])
            metadata.update({"kind": "as_pair", "anion": anion, "solvent": solvent})
            path = output_dir / "initial" / f"as__{anion}__{solvent}.xyz"
            _write_structure(path, atoms, metadata)
            rows.append({"structure_id": path.stem, "kind": "as_pair", "cation": "", "anion": anion, "solvent": solvent, "topology": "AS", "formal_charge": -1, "xyz_path": _manifest_path(path)})
    for solvent in SOLVENTS:
        component = components[solvent]
        atoms = _translated(component, 0.0)
        metadata = {"kind": "solvent", "solvent": solvent, "fragments": {"S": {"species_id": solvent, "atom_indices_zero_based": list(range(len(atoms))), "geometry_method": component.geometry_method}}, "anchor_indices_zero_based": {"S": component.anchor}, "formal_charge": 0, "rdkit_seed": SEED}
        path = output_dir / "initial" / f"solvent__{solvent}.xyz"
        _write_structure(path, atoms, metadata)
        rows.append({"structure_id": path.stem, "kind": "solvent", "cation": "", "anion": "", "solvent": solvent, "topology": "", "formal_charge": 0, "xyz_path": _manifest_path(path)})
    for anion in ANIONS:
        for solvent in SOLVENTS:
            component = components[anion]
            atoms = _translated(component, 0.0)
            metadata = {"kind": "anion", "anion": anion, "solvent": solvent, "fragments": {"A": {"species_id": anion, "atom_indices_zero_based": list(range(len(atoms))), "geometry_method": component.geometry_method}}, "anchor_indices_zero_based": {"A": component.anchor}, "formal_charge": -1, "rdkit_seed": SEED}
            path = output_dir / "initial" / f"anion__{anion}__{solvent}.xyz"
            _write_structure(path, atoms, metadata)
            rows.append({"structure_id": path.stem, "kind": "anion", "cation": "", "anion": anion, "solvent": solvent, "topology": "", "formal_charge": -1, "xyz_path": _manifest_path(path)})
    for cation, solvent in cation_solvent_keys():
        component = components[cation]
        atoms = _translated(component, 0.0)
        metadata = {
            "kind": "cation",
            "cation": cation,
            "solvent": solvent,
            "fragments": {
                "C": {
                    "species_id": cation,
                    "atom_indices_zero_based": list(range(len(atoms))),
                    "geometry_method": component.geometry_method,
                }
            },
            "anchor_indices_zero_based": {"C": component.anchor},
            "formal_charge": 1,
            "rdkit_seed": SEED,
        }
        path = output_dir / "initial" / f"cation__{cation}__{solvent}.xyz"
        _write_structure(path, atoms, metadata)
        rows.append({"structure_id": path.stem, "kind": "cation", "cation": cation, "anion": "", "solvent": solvent, "topology": "", "formal_charge": 1, "xyz_path": _manifest_path(path)})
    write_csv(output_dir / "structure_manifest.csv", rows, ["structure_id", "kind", "cation", "anion", "solvent", "topology", "formal_charge", "xyz_path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "data" / "chauhan_cation_eox")
    args = parser.parse_args()
    rows = generate(args.output_dir.resolve())
    print(f"wrote {len(rows)} deterministic initial structures to {args.output_dir}")


if __name__ == "__main__":
    main()
