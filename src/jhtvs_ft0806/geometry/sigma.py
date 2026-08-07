"""Geometry-level construction of the frozen sigma-complex dications."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

from jhtvs_ft0806.geometry.topology import (
    build_repeat_chain,
    canonical_smiles,
    molecule_from_smiles,
)

EMBED_SEED = 20260707
N_CONFORMERS = 100
RESTORED_CH_BOND_ANGSTROM = 1.09


class SigmaConstructionError(ValueError):
    """Raised when construction or mandatory sigma QC fails."""


def _hill_formula(symbols: Sequence[str], charge: int) -> str:
    counts = Counter(symbols)
    order = (
        ["C", "H"] + sorted(element for element in counts if element not in {"C", "H"})
        if "C" in counts
        else sorted(counts)
    )
    body = "".join(
        element + (str(counts[element]) if counts[element] > 1 else "")
        for element in order
    )
    if charge == 0:
        return body
    magnitude = "" if abs(charge) == 1 else str(abs(charge))
    return f"{body}{'+' if charge > 0 else '-'}{magnitude}"


@dataclass(frozen=True, slots=True)
class SigmaTopology:
    parent_id: str
    sigma_state_id: str
    monomer_smiles: str
    neutral_dimer_smiles: str
    site_a_atom_index_0based: int
    site_b_atom_index_0based: int
    junction_copy1_atom_index_0based: int
    junction_copy2_atom_index_0based: int
    expected_formula: str
    charge: int
    multiplicity: int
    link_atom_pair: str
    topology_sha256: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> SigmaTopology:
        return cls(
            parent_id=row["parent_id"],
            sigma_state_id=row["sigma_state_id"],
            monomer_smiles=row["source_monomer_smiles"],
            neutral_dimer_smiles=row["neutral_dimer_smiles"],
            site_a_atom_index_0based=int(row["site_a_atom_index_0based"]),
            site_b_atom_index_0based=int(row["site_b_atom_index_0based"]),
            junction_copy1_atom_index_0based=int(
                row["junction_copy1_atom_index_0based"]
            ),
            junction_copy2_atom_index_0based=int(
                row["junction_copy2_atom_index_0based"]
            ),
            expected_formula=row["sigma_formula_expected"],
            charge=int(row["sigma_product_charge"]),
            multiplicity=int(row["sigma_product_multiplicity"]),
            link_atom_pair=row["link_atom_pair"],
            topology_sha256=row["topology_sha256"],
        )


@dataclass(frozen=True, slots=True)
class SigmaComplex:
    topology: SigmaTopology
    neutral_dimer: Mol
    symbols: tuple[str, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    restored_hydrogen_indices: tuple[int, int]
    embed_seed: int
    conformers_attempted: int
    force_field: str
    embedding_note: str

    @property
    def charge(self) -> int:
        return self.topology.charge

    @property
    def multiplicity(self) -> int:
        return self.topology.multiplicity

    @property
    def formula(self) -> str:
        return _hill_formula(self.symbols, self.charge)

    @property
    def junction_atom_indices(self) -> tuple[int, int]:
        return (
            self.topology.junction_copy1_atom_index_0based,
            self.topology.junction_copy2_atom_index_0based,
        )

    def xyz_text(self) -> str:
        comment = (
            f"{self.topology.sigma_state_id} q={self.charge} mult={self.multiplicity} "
            f"junctions={self.junction_atom_indices[0]},{self.junction_atom_indices[1]} "
            f"topology_sha256={self.topology.topology_sha256}"
        )
        lines = [str(len(self.symbols)), comment]
        lines.extend(
            f"{symbol:<2s} {x:18.10f} {y:18.10f} {z:18.10f}"
            for symbol, (x, y, z) in zip(
                self.symbols, self.coordinates_angstrom, strict=True
            )
        )
        return "\n".join(lines) + "\n"


def load_sigma_topologies(path: Path) -> list[SigmaTopology]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [SigmaTopology.from_row(row) for row in csv.DictReader(handle)]


def _embed_neutral_dimer(
    neutral_dimer: Mol,
    *,
    seed: int,
    n_conformers: int,
) -> tuple[Mol, int, str, str]:
    if n_conformers < 1:
        raise SigmaConstructionError("n_conformers must be positive")
    molecule = Chem.AddHs(Chem.Mol(neutral_dimer))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                molecule, numConfs=n_conformers, params=params
            )
        )
        if not conformer_ids:
            raise SigmaConstructionError("ETKDG embedding returned no conformers")
        optimized = AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=1)
        converged = [
            (energy, conformer_id)
            for conformer_id, (not_converged, energy) in zip(
                conformer_ids, optimized, strict=True
            )
            if not_converged == 0
        ]
        candidates = converged or [
            (energy, conformer_id)
            for conformer_id, (_not_converged, energy) in zip(
                conformer_ids, optimized, strict=True
            )
        ]
        best_conformer = min(candidates)[1]
        note = f"lowest of {len(conformer_ids)} fixed-seed ETKDGv3/MMFF94 conformers"
        return molecule, best_conformer, "MMFF94", note

    if AllChem.EmbedMolecule(molecule, params) != 0:
        raise SigmaConstructionError("ETKDG embedding failed for MMFF-untypable dimer")
    return (
        molecule,
        molecule.GetConformer().GetId(),
        "none(ETKDG)",
        "MMFF-untypable: fixed-seed ETKDGv3 embedding without force-field preoptimization",
    )


def _vector(
    origin: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(target[index] - origin[index] for index in range(3))  # type: ignore[return-value]


def _cross(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _unit(vector: Sequence[float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 1.0e-8:
        return None
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _local_plane_normal(
    molecule: Mol,
    coordinates: Sequence[Sequence[float]],
    junction_index: int,
) -> tuple[float, float, float]:
    center = coordinates[junction_index]
    neighbors = [atom.GetIdx() for atom in molecule.GetAtomWithIdx(junction_index).GetNeighbors()]
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for left_offset, left_index in enumerate(neighbors):
        for right_index in neighbors[left_offset + 1 :]:
            normal = _cross(
                _vector(center, coordinates[left_index]),
                _vector(center, coordinates[right_index]),
            )
            candidates.append((_dot(normal, normal), normal))
    if not candidates:
        raise SigmaConstructionError(
            f"junction atom {junction_index} has fewer than two graph neighbors"
        )
    normal = _unit(max(candidates, key=lambda item: item[0])[1])
    if normal is None:
        raise SigmaConstructionError(
            f"junction atom {junction_index} has a degenerate local neighbor plane"
        )
    return normal


def build_sigma_complex(
    topology: SigmaTopology,
    *,
    seed: int = EMBED_SEED,
    n_conformers: int = N_CONFORMERS,
) -> SigmaComplex:
    """Build [HM-MH]2+ from the indexed neutral n=2 graph and restore two anti H atoms."""

    neutral_dimer = build_repeat_chain(
        topology.monomer_smiles,
        topology.site_a_atom_index_0based,
        topology.site_b_atom_index_0based,
        copies=2,
    )
    expected_dimer = molecule_from_smiles(topology.neutral_dimer_smiles)
    if canonical_smiles(neutral_dimer) != canonical_smiles(expected_dimer):
        raise SigmaConstructionError(
            f"{topology.parent_id}: indexed neutral dimer differs from frozen topology"
        )

    embedded, conformer_id, force_field, note = _embed_neutral_dimer(
        neutral_dimer, seed=seed, n_conformers=n_conformers
    )
    conformer = embedded.GetConformer(conformer_id)
    symbols = [atom.GetSymbol() for atom in embedded.GetAtoms()]
    coordinates = [
        tuple(conformer.GetAtomPosition(index)) for index in range(embedded.GetNumAtoms())
    ]
    normals: list[tuple[float, float, float]] = []
    restored_hydrogen_indices: list[int] = []
    for junction_index in (
        topology.junction_copy1_atom_index_0based,
        topology.junction_copy2_atom_index_0based,
    ):
        normal = _local_plane_normal(neutral_dimer, coordinates, junction_index)
        if normals and _dot(normals[0], normal) > 0:
            normal = tuple(-component for component in normal)
        normals.append(normal)
        center = coordinates[junction_index]
        restored_hydrogen_indices.append(len(symbols))
        symbols.append("H")
        coordinates.append(
            tuple(
                center[axis] + RESTORED_CH_BOND_ANGSTROM * normal[axis]
                for axis in range(3)
            )
        )

    result = SigmaComplex(
        topology=topology,
        neutral_dimer=neutral_dimer,
        symbols=tuple(symbols),
        coordinates_angstrom=tuple(coordinates),
        restored_hydrogen_indices=(
            restored_hydrogen_indices[0],
            restored_hydrogen_indices[1],
        ),
        embed_seed=seed,
        conformers_attempted=n_conformers,
        force_field=force_field,
        embedding_note=note,
    )
    validate_sigma_complex(result)
    return result


def validate_sigma_complex(complex_geometry: SigmaComplex) -> None:
    """Enforce construction-time graph, composition, charge and junction invariants."""

    topology = complex_geometry.topology
    monomer = molecule_from_smiles(topology.monomer_smiles)
    atoms_per_copy = monomer.GetNumAtoms()
    expected_junctions = (
        topology.site_b_atom_index_0based,
        atoms_per_copy + topology.site_a_atom_index_0based,
    )
    errors: list[str] = []
    if complex_geometry.junction_atom_indices != expected_junctions:
        errors.append(
            f"junction indices {complex_geometry.junction_atom_indices} != {expected_junctions}"
        )
    inter_copy_bonds = [
        bond
        for bond in complex_geometry.neutral_dimer.GetBonds()
        if (bond.GetBeginAtomIdx() < atoms_per_copy)
        != (bond.GetEndAtomIdx() < atoms_per_copy)
    ]
    inter_copy_pairs = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in inter_copy_bonds
    }
    if len(inter_copy_bonds) != 1 or inter_copy_pairs != {frozenset(expected_junctions)}:
        errors.append("neutral dimer does not contain exactly the frozen inter-copy bond")
    if len(Chem.GetMolFrags(complex_geometry.neutral_dimer)) != 1:
        errors.append("neutral dimer is not one connected component")
    if canonical_smiles(complex_geometry.neutral_dimer) != canonical_smiles(
        molecule_from_smiles(topology.neutral_dimer_smiles)
    ):
        errors.append("neutral dimer heavy-atom graph differs from frozen topology")

    monomer_heavy = Counter(atom.GetSymbol() for atom in monomer.GetAtoms())
    expected_heavy = Counter(
        {element: 2 * count for element, count in monomer_heavy.items()}
    )
    actual_heavy = Counter(symbol for symbol in complex_geometry.symbols if symbol != "H")
    if actual_heavy != expected_heavy:
        errors.append(f"heavy composition {actual_heavy} != 2M {expected_heavy}")
    monomer_all_h = Chem.AddHs(Chem.Mol(monomer))
    monomer_total = Counter(atom.GetSymbol() for atom in monomer_all_h.GetAtoms())
    expected_total = Counter(
        {element: 2 * count for element, count in monomer_total.items()}
    )
    actual_total = Counter(complex_geometry.symbols)
    if actual_total != expected_total:
        errors.append(f"restored total composition {actual_total} != 2M {expected_total}")
    if complex_geometry.charge != 2 or complex_geometry.multiplicity != 1:
        errors.append(
            f"charge/multiplicity {(complex_geometry.charge, complex_geometry.multiplicity)} != (2, 1)"
        )

    if complex_geometry.formula != topology.expected_formula:
        errors.append(
            f"formula {complex_geometry.formula} != frozen {topology.expected_formula}"
        )

    restored_displacements: list[tuple[float, float, float]] = []
    for junction, hydrogen in zip(
        complex_geometry.junction_atom_indices,
        complex_geometry.restored_hydrogen_indices,
        strict=True,
    ):
        if complex_geometry.symbols[hydrogen] != "H":
            errors.append(f"restored atom {hydrogen} is not hydrogen")
            continue
        displacement = _vector(
            complex_geometry.coordinates_angstrom[junction],
            complex_geometry.coordinates_angstrom[hydrogen],
        )
        restored_displacements.append(displacement)
        distance = math.sqrt(_dot(displacement, displacement))
        if not math.isclose(
            distance, RESTORED_CH_BOND_ANGSTROM, rel_tol=0.0, abs_tol=1.0e-8
        ):
            errors.append(f"restored H {hydrogen} has junction distance {distance:.8f} Å")
    if (
        len(restored_displacements) == 2
        and _dot(restored_displacements[0], restored_displacements[1]) > 1.0e-12
    ):
        errors.append("restored junction hydrogens are not on opposite local faces")
    if errors:
        raise SigmaConstructionError(
            f"{topology.parent_id} sigma construction QC failed: " + "; ".join(errors)
        )
