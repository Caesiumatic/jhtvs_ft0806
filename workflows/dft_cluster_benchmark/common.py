from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

METHOD = "M06-HF"
BASIS = "aug-cc-pVTZ"
SOFTWARE = "ORCA"
METHOD_ID = "DFT_M06-HF_aug-cc-pVTZ_vertical_DSCF_v1"
INTEGRAL_APPROXIMATION = "RIJCOSX_AutoAux"
POPULATION_SCHEME = "Mulliken"
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_KCAL_MOL = 627.5094740631
TOPOLOGIES = ("CAS", "CSA", "ACS")
BIAS_ALPHA_ANGSTROM_INV = 0.01


@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xyz(path: Path) -> tuple[list[Atom], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        element, x, y, z, *_ = line.split()
        atoms.append(Atom(element, float(x), float(y), float(z)))
    if len(atoms) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    return atoms, lines[1] if len(lines) > 1 else ""


def load_metadata(xyz_path: Path) -> dict:
    return json.loads(xyz_path.with_suffix(".json").read_text(encoding="utf-8"))


def atom_distance(left: Atom, right: Atom) -> float:
    return math.dist((left.x, left.y, left.z), (right.x, right.y, right.z))


def harmonic_force_kcal_mol_angstrom2(force_eh_bohr2: float) -> float:
    return force_eh_bohr2 * HARTREE_TO_KCAL_MOL / (BOHR_TO_ANGSTROM**2)


def orca_bias_depth_kcal_mol(force_eh_bohr2: float, alpha_angstrom_inv: float = BIAS_ALPHA_ANGSTROM_INV) -> float:
    """Match the local curvature k=2*E*alpha^2 of ORCA's Morse-like bias."""
    return harmonic_force_kcal_mol_angstrom2(force_eh_bohr2) / (2 * alpha_angstrom_inv**2)
