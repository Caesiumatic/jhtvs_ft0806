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


def restore_fire_geometry(atoms: Any, *, restart_path: Path, trajectory_path: Path) -> bool:
    if not restart_path.is_file():
        return False
    if not trajectory_path.is_file():
        raise RuntimeError("FIRE restart exists without an optimization trajectory")
    try:
        from ase.io import read
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for optimization restart") from exc
    resumed = read(trajectory_path, index=-1)
    if atoms.get_chemical_symbols() != resumed.get_chemical_symbols():
        raise RuntimeError("FIRE restart atom order or composition changed")
    atoms.positions[:] = resumed.positions
    return True


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
    trajectory_path = output_dir / "optimization.traj"
    start_path = output_dir / "optimization_start.json"
    if receipt_path.is_file() and optimized_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        completed = read(optimized_path)
        if receipt["geometry_sha256"] == atoms_geometry_sha256(completed):
            return receipt
        raise RuntimeError("completed optimization geometry hash mismatch")

    input_hash = atoms_geometry_sha256(atoms)
    if start_path.is_file():
        start = json.loads(start_path.read_text(encoding="utf-8"))
        if (
            start["initial_geometry_sha256"] != input_hash
            or int(start["charge"]) != charge
            or int(start["spin"]) != spin
        ):
            raise RuntimeError("optimization restart input or electronic state drift")
    else:
        start = {
            "initial_geometry_sha256": input_hash,
            "charge": charge,
            "spin": spin,
        }
        start_path.write_text(
            json.dumps(start, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    resumed = restore_fire_geometry(
        atoms, restart_path=restart_path, trajectory_path=trajectory_path
    )
    apply_state_metadata(atoms, charge=charge, spin=spin)
    atoms.calc = RestrainedCalculator(model_calculator, restraint)
    optimizer = FIRE(
        atoms,
        restart=str(restart_path),
        logfile=str(output_dir / "fire.log"),
        trajectory=str(trajectory_path),
    )
    converged = bool(optimizer.run(fmax=fmax_eV_A, steps=max_steps))
    write(optimized_path, atoms)
    max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    receipt = {
        "status": "clean" if converged else "incomplete",
        "charge": charge,
        "spin": spin,
        "initial_geometry_sha256": start["initial_geometry_sha256"],
        "resumed_from_fire_restart": resumed,
        "geometry_sha256": atoms_geometry_sha256(atoms),
        "converged": converged,
        "steps": int(optimizer.nsteps),
        "fmax_eV_A": fmax_eV_A,
        "observed_max_force_eV_A": max_force,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
