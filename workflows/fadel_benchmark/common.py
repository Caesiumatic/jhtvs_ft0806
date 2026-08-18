from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

HARTREE_TO_EV = 27.211386245988
SEED = 20260818
ANIONS = ("TDI", "TFSI", "BF4", "PF6")
SOLVENTS = ("DMSO", "DME", "PC", "ACN")
TOPOLOGIES = ("CAS", "CSA", "ACS")


@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float


def workflow_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return workflow_dir().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def species_table() -> dict[str, dict[str, str]]:
    return {row["id"]: row for row in read_csv(workflow_dir() / "species.csv")}


def chemistry_keys() -> list[tuple[str, str]]:
    return [(anion, solvent) for solvent in SOLVENTS for anion in ANIONS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xyz(path: Path) -> tuple[list[Atom], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + count]:
        element, x, y, z, *_ = line.split()
        atoms.append(Atom(element, float(x), float(y), float(z)))
    if len(atoms) != count:
        raise ValueError(f"XYZ atom count mismatch: {path}")
    return atoms, lines[1] if len(lines) > 1 else ""


def write_xyz(path: Path, atoms: Sequence[Atom], comment: str) -> None:
    lines = [str(len(atoms)), comment]
    lines.extend(f"{a.element:<2} {a.x: .10f} {a.y: .10f} {a.z: .10f}" for a in atoms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_metadata(xyz_path: Path) -> dict:
    return json.loads(xyz_path.with_suffix(".json").read_text(encoding="utf-8"))


def distance(a: Atom, b: Atom) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def ip_ev(energy_oxidized_eh: float, energy_reduced_eh: float) -> float:
    return (energy_oxidized_eh - energy_reduced_eh) * HARTREE_TO_EV
