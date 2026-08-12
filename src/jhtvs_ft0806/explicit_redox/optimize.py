from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .calculator import apply_state_metadata
from .restraint import FlatBottomShell


def atoms_geometry_sha256(atoms: Any) -> str:
    digest = hashlib.sha256()
    digest.update(";".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def combined_energy_forces(atoms: Any, calculator: Any, restraint: FlatBottomShell) -> tuple[float, np.ndarray]:
    model_energy = float(calculator.get_potential_energy(atoms))
    model_forces = np.asarray(calculator.get_forces(atoms), dtype=np.float64)
    restrained = restraint.evaluate(np.asarray(atoms.positions, dtype=np.float64))
    return model_energy + restrained.energy_eV, model_forces + restrained.forces_eV_A


class RestrainedCalculator:
    implemented_properties = ["energy", "forces"]

    def __init__(self, model: Any, restraint: FlatBottomShell) -> None:
        self.model = model
        self.restraint = restraint
        self.results: dict[str, Any] = {}

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        del force_consistent
        energy, forces = combined_energy_forces(atoms, self.model, self.restraint)
        self.results = {"energy": energy, "forces": forces}
        return energy

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        energy, forces = combined_energy_forces(atoms, self.model, self.restraint)
        self.results = {"energy": energy, "forces": forces}
        return forces


def optimize_state(
    *,
    atoms: Any,
    model_calculator: Any,
    restraint: FlatBottomShell,
    charge: int,
    spin: int,
    output_dir: Path,
    fmax_eV_A: float = 0.02,
    max_steps: int = 10_000,
) -> dict[str, Any]:
    try:
        from ase.io import read, write
        from ase.optimize import FIRE
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for optimization") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "optimization.json"
    optimized_path = output_dir / "optimized.xyz"
    restart_path = output_dir / "fire.restart.json"
    if receipt_path.is_file() and optimized_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        completed = read(optimized_path)
        if receipt["geometry_sha256"] == atoms_geometry_sha256(completed):
            return receipt
        raise RuntimeError("completed optimization geometry hash mismatch")

    apply_state_metadata(atoms, charge=charge, spin=spin)
    atoms.calc = RestrainedCalculator(model_calculator, restraint)
    initial_hash = atoms_geometry_sha256(atoms)
    optimizer = FIRE(
        atoms,
        restart=str(restart_path),
        logfile=str(output_dir / "fire.log"),
        trajectory=str(output_dir / "optimization.traj"),
    )
    converged = bool(optimizer.run(fmax=fmax_eV_A, steps=max_steps))
    write(optimized_path, atoms)
    max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    receipt = {
        "status": "clean" if converged else "incomplete",
        "charge": charge,
        "spin": spin,
        "initial_geometry_sha256": initial_hash,
        "geometry_sha256": atoms_geometry_sha256(atoms),
        "converged": converged,
        "steps": int(optimizer.nsteps),
        "fmax_eV_A": fmax_eV_A,
        "observed_max_force_eV_A": max_force,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
