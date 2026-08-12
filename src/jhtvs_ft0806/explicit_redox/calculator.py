from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from jhtvs_ft0806.ml.features import EXPECTED_CHECKPOINT_SHA256, PolarMACEBackend


def ensure_torch_compiler_compat(torch_module: Any) -> bool:
    """Expose the eager-mode compilation probe expected by MACE 0.3.16."""

    compiler = getattr(torch_module, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return False
    is_compiling = torch_module._dynamo.is_compiling
    if compiler is None:
        torch_module.compiler = SimpleNamespace(is_compiling=is_compiling)
    else:
        compiler.is_compiling = is_compiling
    return True


def apply_state_metadata(atoms: Any, *, charge: int, spin: int) -> None:
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    atoms.info["external_field"] = np.zeros(3, dtype=np.float64)
    atoms.pbc = False


@dataclass(frozen=True)
class CalculatorProvenance:
    checkpoint_sha256: str
    mace_version: str
    graph_electrostatics_version: str
    torch_version: str
    cuda_version: str
    device: str
    default_dtype: str
    model_parameter_sha256_before: str


def model_parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PolarMACEStateCalculator:
    """ASE-compatible force calculator using the repository's verified raw backend."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, *, checkpoint: str, charge: int, spin: int, device: str = "cpu") -> None:
        try:
            import torch
            from ase.calculators.calculator import Calculator
        except ImportError as exc:  # pragma: no cover - exercised on the execution host
            raise RuntimeError("ASE, PyTorch and MACE are required") from exc
        ensure_torch_compiler_compat(torch)
        self._ase_base = Calculator()
        self.results: dict[str, Any] = {}
        self.atoms = None
        self.parameters = {}
        self.charge = int(charge)
        self.spin = int(spin)
        self.device = device
        self.backend = PolarMACEBackend(checkpoint=checkpoint, device=device)
        parameter_hash = model_parameter_sha256(self.backend.model)
        provenance = self.backend.provenance
        if provenance.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("MACE-POLAR checkpoint hash mismatch")
        self.provenance = CalculatorProvenance(
            checkpoint_sha256=provenance.checkpoint_sha256,
            mace_version=provenance.mace_version,
            graph_electrostatics_version=provenance.graph_electrostatics_version,
            torch_version=torch.__version__,
            cuda_version=str(torch.version.cuda or "none"),
            device=device,
            default_dtype=provenance.default_dtype,
            model_parameter_sha256_before=parameter_hash,
        )
        self.raw_diagnostics: dict[str, np.ndarray] = {}
        self._last_geometry_key: str | None = None

    def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
        del force_consistent
        self.calculate(atoms=atoms)
        return float(self.results["energy"])

    def get_forces(self, atoms: Any = None) -> np.ndarray:
        self.calculate(atoms=atoms)
        return np.asarray(self.results["forces"], dtype=np.float64)

    def calculate(self, atoms: Any = None, properties: Any = None, system_changes: Any = None) -> None:
        del properties, system_changes
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("atoms are required")
        geometry_key = hashlib.sha256(
            np.asarray(atoms.positions, dtype="<f8").tobytes()
            + f"|{self.charge}|{self.spin}".encode()
        ).hexdigest()
        if geometry_key == getattr(self, "_last_geometry_key", None) and self.results:
            return
        self.atoms = atoms.copy()
        apply_state_metadata(atoms, charge=self.charge, spin=self.spin)
        batch = self.backend.build_graph_from_atoms(
            atoms=atoms, formal_charge=self.charge, multiplicity=self.spin
        )
        outputs = self.backend.model(
            batch.to_dict(), training=False, compute_force=True, compute_stress=False
        )
        energy = outputs["energy"].detach().cpu().numpy().reshape(-1)
        forces = outputs["forces"].detach().cpu().numpy()
        if energy.size != 1 or forces.shape != (len(atoms), 3):
            raise RuntimeError("unexpected PolarMACE energy/force shape")
        if not np.isfinite(energy[0]) or not np.all(np.isfinite(forces)):
            raise RuntimeError("non-finite PolarMACE energy or force")
        self.results = {"energy": float(energy[0]), "forces": forces.astype(np.float64)}
        self._last_geometry_key = geometry_key
        self.raw_diagnostics = {
            key: value.detach().cpu().numpy()
            for key, value in outputs.items()
            if key in {"density_coefficients", "spin_density", "spin_charge_density"}
            and value is not None
        }

    def assert_model_unchanged(self) -> None:
        if model_parameter_sha256(self.backend.model) != self.provenance.model_parameter_sha256_before:
            raise RuntimeError("MACE model parameters changed")

    def provenance_dict(self) -> Mapping[str, str]:
        return asdict(self.provenance)

    def __deepcopy__(self, memo: dict[int, object]) -> "PolarMACEStateCalculator":
        del memo
        return copy.copy(self)
