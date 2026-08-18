#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from .common import ANIONS, SEED, SOLVENTS, TOPOLOGIES, Atom, chemistry_keys, repo_root, species_table, write_csv, write_xyz
except ImportError:
    from common import ANIONS, SEED, SOLVENTS, TOPOLOGIES, Atom, chemistry_keys, repo_root, species_table, write_csv, write_xyz

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
    if species_id == "Li":
        candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == "Li"]
    elif species_id in {"TDI", "TFSI"}:
        candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() == -1]
    elif species_id == "BF4":
        candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == "B"]
    elif species_id == "PF6":
        candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    elif species_id in {"DMSO", "PC"}:
        candidates = [
            a for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
        ]
    elif species_id == "DME":
        candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == "O"]
    elif species_id == "ACN":
        candidates = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    else:
        raise ValueError(f"unknown anchor rule: {species_id}")
    if not candidates:
        raise ValueError(f"anchor not found for {species_id}")
    return min(atom.GetIdx() for atom in candidates)


def _ideal_component(species_id: str, central: str, ligand_count: int, bond: float) -> Component:
    row = species_table()[species_id]
    mol = Chem.MolFromSmiles(row["smiles"])
    anchor = _anchor_index(mol, species_id)
    if ligand_count == 4:
        scale = bond / np.sqrt(3.0)
        positions = [(scale, scale, scale), (scale, -scale, -scale), (-scale, scale, -scale), (-scale, -scale, scale)]
    else:
        positions = [(bond, 0, 0), (-bond, 0, 0), (0, bond, 0), (0, -bond, 0), (0, 0, bond), (0, 0, -bond)]
    atoms, ligand_index = [], 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == central:
            atoms.append(Atom(central, 0.0, 0.0, 0.0))
        else:
            atoms.append(Atom("F", *positions[ligand_index]))
            ligand_index += 1
    if ligand_index != ligand_count:
        raise ValueError(f"ideal construction failed for {species_id}")
    return Component(species_id, atoms, anchor, -1, f"ideal_{'tetrahedral' if ligand_count == 4 else 'octahedral'}_{species_id}")


def build_component(species_id: str) -> Component:
    row = species_table()[species_id]
    if species_id == "BF4":
        return _ideal_component("BF4", "B", 4, 1.40)
    if species_id == "PF6":
        return _ideal_component("PF6", "P", 6, 1.58)
    if species_id == "Li":
        return Component("Li", [Atom("Li", 0.0, 0.0, 0.0)], 0, 1, "single_atom")
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
        Atom(atom.GetSymbol(), *(float(value) for value in conformer.GetAtomPosition(atom.GetIdx())))
        for atom in mol.GetAtoms()
    ]
    return Component(species_id, atoms, _anchor_index(mol, species_id), int(row["formal_charge"]), method)


def _translated(component: Component, anchor_x: float) -> list[Atom]:
    anchor = component.atoms[component.anchor]
    centered = [np.array([a.x - anchor.x, a.y - anchor.y, a.z - anchor.z]) for a in component.atoms]
    ranges = np.ptp(np.stack(centered), axis=0)
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    permutation = permutations[int(np.argmin(ranges))]
    oriented = [coords[list(permutation)] for coords in centered]
    return [Atom(atom.element, coords[0] + anchor_x, coords[1], coords[2]) for atom, coords in zip(component.atoms, oriented, strict=True)]


def _minimum_heavy_distance(left: list[Atom], right: list[Atom]) -> float:
    return min(
        np.linalg.norm(np.array([a.x - b.x, a.y - b.y, a.z - b.z]))
        for a in left for b in right if a.element != "H" and b.element != "H"
    )


def _fragment_label(component: Component) -> str:
    return {"cation": "C", "anion": "A", "solvent": "S"}[species_table()[component.species_id]["kind"]]


def assemble(ordered: list[Component]) -> tuple[list[Atom], dict]:
    left_distance = right_distance = INITIAL_ANCHOR_DISTANCE_ANG
    while True:
        placed = [_translated(ordered[0], -left_distance), _translated(ordered[1], 0.0), _translated(ordered[2], right_distance)]
        d01, d12, d02 = (_minimum_heavy_distance(placed[0], placed[1]), _minimum_heavy_distance(placed[1], placed[2]), _minimum_heavy_distance(placed[0], placed[2]))
        if min(d01, d12, d02) >= MIN_HEAVY_DISTANCE_ANG:
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
    atoms, fragments, anchors = [], {}, {}
    for component, component_atoms in zip(ordered, placed, strict=True):
        start = len(atoms)
        atoms.extend(component_atoms)
        label = _fragment_label(component)
        fragments[label] = {"species_id": component.species_id, "atom_indices_zero_based": list(range(start, len(atoms))), "geometry_method": component.geometry_method}
        anchors[label] = start + component.anchor
    return atoms, {
        "fragments": fragments,
        "anchor_indices_zero_based": anchors,
        "initial_anchor_distances_ang": {"left_middle": left_distance, "middle_right": right_distance},
        "minimum_heavy_distance_ang": min(d01, d12, d02),
        "formal_charge": sum(component.formal_charge for component in ordered),
        "rdkit_seed": SEED,
    }


def assemble_pair(anion: Component, solvent: Component) -> tuple[list[Atom], dict]:
    distance = INITIAL_ANCHOR_DISTANCE_ANG
    while True:
        placed_a, placed_s = _translated(anion, -distance / 2), _translated(solvent, distance / 2)
        minimum = _minimum_heavy_distance(placed_a, placed_s)
        if minimum >= MIN_HEAVY_DISTANCE_ANG:
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
        "minimum_heavy_distance_ang": minimum,
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
    rows = []
    for anion, solvent in chemistry_keys():
        atoms, metadata = assemble_pair(components[anion], components[solvent])
        metadata.update({"kind": "as_pair", "anion": anion, "solvent": solvent, "topology": "AS"})
        path = output_dir / "initial" / f"as__{anion}__{solvent}.xyz"
        _write_structure(path, atoms, metadata)
        rows.append({"structure_id": path.stem, "kind": "as_pair", "cation": "", "anion": anion, "solvent": solvent, "topology": "AS", "formal_charge": -1, "xyz_path": _manifest_path(path)})
        mapping = {"C": components["Li"], "A": components[anion], "S": components[solvent]}
        for topology in TOPOLOGIES:
            atoms, metadata = assemble([mapping[label] for label in topology])
            metadata.update({"kind": "triad", "cation": "Li", "anion": anion, "solvent": solvent, "topology": topology})
            path = output_dir / "initial" / f"triad__Li__{anion}__{solvent}__{topology}.xyz"
            _write_structure(path, atoms, metadata)
            rows.append({"structure_id": path.stem, "kind": "triad", "cation": "Li", "anion": anion, "solvent": solvent, "topology": topology, "formal_charge": 0, "xyz_path": _manifest_path(path)})
    write_csv(output_dir / "structure_manifest.csv", rows, ["structure_id", "kind", "cation", "anion", "solvent", "topology", "formal_charge", "xyz_path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "data" / "fadel_benchmark")
    args = parser.parse_args()
    rows = generate(args.output_dir.resolve())
    print(f"wrote {len(rows)} deterministic Fadel benchmark structures")


if __name__ == "__main__":
    main()
