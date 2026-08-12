from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class XYZGeometry:
    symbols: tuple[str, ...]
    positions: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class PackedCluster:
    geometry_path: Path
    geometry_sha256: str
    input_path: Path
    input_sha256: str
    log_path: Path
    target_atoms: int
    solvent_atoms: int
    solvent_count: int
    minimum_intermolecular_distance_A: float
    containment_radius_A: float
    packmol_executable: str
    packmol_executable_sha256: str
    packmol_version: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_xyz(path: Path) -> XYZGeometry:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    rows = [line.split() for line in lines[2 : 2 + count]]
    if len(rows) != count:
        raise ValueError(f"truncated XYZ: {path}")
    return XYZGeometry(
        symbols=tuple(row[0] for row in rows),
        positions=tuple(tuple(float(value) for value in row[1:4]) for row in rows),
    )


def molecular_volume_A3(smiles: str, seed: int) -> float:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(molecule, params) < 0:
        raise RuntimeError(f"cannot embed solvent for volume: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
    return float(AllChem.ComputeMolVolume(molecule))


def target_radius_A(geometry: XYZGeometry) -> float:
    heavy = [
        position
        for symbol, position in zip(geometry.symbols, geometry.positions, strict=True)
        if symbol != "H"
    ]
    if not heavy:
        raise ValueError("target has no heavy atoms")
    centroid = tuple(sum(point[axis] for point in heavy) / len(heavy) for axis in range(3))
    return max(math.dist(point, centroid) for point in geometry.positions)


def containment_radius_A(target: XYZGeometry, solvent_volume_A3: float) -> float:
    shell_volume_radius = (3.0 * (5.0 * solvent_volume_A3) / (4.0 * math.pi * 0.32)) ** (1.0 / 3.0)
    return target_radius_A(target) + max(5.5, shell_volume_radius)


def render_packmol_input(
    *, target_path: Path, solvent_path: Path, output_path: Path, tolerance_A: float, radius_A: float, seed: int
) -> str:
    return (
        f"tolerance {tolerance_A:.3f}\n"
        "filetype xyz\n"
        f"output {output_path.resolve()}\n"
        f"seed {seed}\n\n"
        f"structure {target_path.resolve()}\n"
        "  number 1\n"
        "  fixed 0.000 0.000 0.000 0.000 0.000 0.000\n"
        "end structure\n\n"
        f"structure {solvent_path.resolve()}\n"
        "  number 5\n"
        f"  inside sphere 0.000 0.000 0.000 {radius_A:.3f}\n"
        "end structure\n"
    )


def molecule_ranges(target_atoms: int, solvent_atoms: int, solvent_count: int = 5) -> tuple[range, ...]:
    groups: list[range] = [range(target_atoms)]
    offset = target_atoms
    for _ in range(solvent_count):
        groups.append(range(offset, offset + solvent_atoms))
        offset += solvent_atoms
    return tuple(groups)


def minimum_intermolecular_distance(geometry: XYZGeometry, groups: Sequence[range]) -> float:
    result = math.inf
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            result = min(
                result,
                *(math.dist(geometry.positions[i], geometry.positions[j]) for i in first for j in second),
            )
    return result


def validate_cluster(
    geometry: XYZGeometry,
    *,
    target: XYZGeometry,
    solvent: XYZGeometry,
    tolerance_A: float,
) -> float:
    expected_symbols = target.symbols + solvent.symbols * 5
    if geometry.symbols != expected_symbols:
        raise ValueError("packed cluster molecule count or atom order mismatch")
    groups = molecule_ranges(len(target.symbols), len(solvent.symbols))
    distance = minimum_intermolecular_distance(geometry, groups)
    if distance < tolerance_A - 0.01:
        raise ValueError(f"packed cluster overlap: {distance:.6f} A")
    return distance


def run_packmol(
    *,
    target_path: Path,
    solvent_path: Path,
    solvent_smiles: str,
    seed: int,
    output_dir: Path,
    tolerance_A: float = 2.0,
    executable: str = "packmol",
) -> PackedCluster:
    binary = shutil.which(executable)
    if binary is None:
        raise FileNotFoundError(f"Packmol executable not found: {executable}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = read_xyz(target_path)
    solvent = read_xyz(solvent_path)
    radius = containment_radius_A(target, molecular_volume_A3(solvent_smiles, seed))
    geometry_path = output_dir / "cluster.xyz"
    input_path = output_dir / "packmol.inp"
    log_path = output_dir / "packmol.log"
    text = render_packmol_input(
        target_path=target_path,
        solvent_path=solvent_path,
        output_path=geometry_path,
        tolerance_A=tolerance_A,
        radius_A=radius,
        seed=seed,
    )
    input_path.write_text(text, encoding="utf-8", newline="\n")
    with input_path.open("rb") as input_handle:
        completed = subprocess.run(
            [binary], stdin=input_handle, capture_output=True, check=False, cwd=output_dir
        )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    log_path.write_text(stdout + stderr, encoding="utf-8", newline="\n")
    if completed.returncode != 0 or "Success!" not in stdout or not geometry_path.is_file():
        raise RuntimeError(f"Packmol failed; inspect {log_path}")
    version_match = re.search(r"\bVersion\s+([^\s]+)", stdout)
    if version_match is None:
        raise RuntimeError("Packmol output did not report a version")
    distance = validate_cluster(
        read_xyz(geometry_path), target=target, solvent=solvent, tolerance_A=tolerance_A
    )
    return PackedCluster(
        geometry_path=geometry_path,
        geometry_sha256=sha256_file(geometry_path),
        input_path=input_path,
        input_sha256=sha256_file(input_path),
        log_path=log_path,
        target_atoms=len(target.symbols),
        solvent_atoms=len(solvent.symbols),
        solvent_count=5,
        minimum_intermolecular_distance_A=distance,
        containment_radius_A=radius,
        packmol_executable=str(Path(binary).resolve()),
        packmol_executable_sha256=sha256_file(Path(binary).resolve()),
        packmol_version=version_match.group(1),
    )
