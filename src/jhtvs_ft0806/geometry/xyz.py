"""XYZ parsing and reference-graph connectivity checks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


COVALENT_RADII_ANGSTROM = {
    "H": 0.31,
    "Li": 1.28,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Na": 1.66,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
}
BOND_TOLERANCE = 1.3
DEFAULT_COVALENT_RADIUS_ANGSTROM = 0.75


@dataclass(frozen=True, slots=True)
class XYZAtom:
    symbol: str
    x: float
    y: float
    z: float

    @property
    def coordinates(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class ConnectivityResult:
    ok: bool
    atom_count: int
    bonds_broken: int
    bonds_formed: int


def read_xyz(path: Path) -> tuple[XYZAtom, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ is missing its atom-count or comment line: {path}")
    atom_count = int(lines[0].split()[0])
    atoms: list[XYZAtom] = []
    for line in lines[2 : 2 + atom_count]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed XYZ atom line in {path}: {line!r}")
        atoms.append(
            XYZAtom(fields[0], float(fields[1]), float(fields[2]), float(fields[3]))
        )
    if len(atoms) != atom_count:
        raise ValueError(f"truncated XYZ: expected {atom_count} atoms in {path}")
    if any(line.strip() for line in lines[2 + atom_count :]):
        raise ValueError(f"unexpected non-empty content after XYZ atoms: {path}")
    return tuple(atoms)


def inferred_bonds(atoms: Sequence[XYZAtom]) -> set[frozenset[int]]:
    result: set[frozenset[int]] = set()
    for first, atom_a in enumerate(atoms):
        radius_a = COVALENT_RADII_ANGSTROM.get(
            atom_a.symbol, DEFAULT_COVALENT_RADIUS_ANGSTROM
        )
        for second in range(first + 1, len(atoms)):
            atom_b = atoms[second]
            cutoff = BOND_TOLERANCE * (
                radius_a
                + COVALENT_RADII_ANGSTROM.get(
                    atom_b.symbol, DEFAULT_COVALENT_RADIUS_ANGSTROM
                )
            )
            if math.dist(atom_a.coordinates, atom_b.coordinates) < cutoff:
                result.add(frozenset((first, second)))
    return result


def normalize_reference_bonds(
    reference_bonds: Iterable[tuple[int, int]], atom_count: int
) -> set[frozenset[int]]:
    normalized: set[frozenset[int]] = set()
    for first, second in reference_bonds:
        if first == second or not (0 <= first < atom_count and 0 <= second < atom_count):
            raise ValueError(f"invalid reference bond ({first}, {second})")
        normalized.add(frozenset((first, second)))
    return normalized


def check_connectivity(
    input_xyz: Path,
    optimized_xyz: Path,
    *,
    reference_bonds: Iterable[tuple[int, int]],
) -> ConnectivityResult:
    """Compare optimized distances with an authored graph and input close contacts."""

    input_atoms = read_xyz(input_xyz)
    optimized_atoms = read_xyz(optimized_xyz)
    if [atom.symbol for atom in input_atoms] != [atom.symbol for atom in optimized_atoms]:
        raise ValueError("atom order or composition changed during preoptimization")
    expected = normalize_reference_bonds(reference_bonds, len(input_atoms))
    input_inferred = inferred_bonds(input_atoms)
    optimized_inferred = inferred_bonds(optimized_atoms)
    broken = expected - optimized_inferred
    formed = optimized_inferred - expected - input_inferred
    return ConnectivityResult(
        ok=not broken and not formed,
        atom_count=len(input_atoms),
        bonds_broken=len(broken),
        bonds_formed=len(formed),
    )
