from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .restraint import FlatBottomShell


@dataclass(frozen=True)
class GapBatch:
    coordinate_sha256: tuple[str, ...]
    lower_energy_eV: NDArray[np.float64]
    oxidized_energy_eV: NDArray[np.float64]
    delta_E_eV: NDArray[np.float64]
    restraint_energy_eV: NDArray[np.float64]


def coordinate_sha256(atoms: Any) -> str:
    digest = hashlib.sha256()
    digest.update(";".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def _state_energies(
    backend: Any,
    atoms_batch: Sequence[Any],
    *,
    charge: int,
    spin: int,
) -> NDArray[np.float64]:
    graphs = [
        backend.build_graph_from_atoms(
            atoms=atoms.copy(), formal_charge=charge, multiplicity=spin
        )
        for atoms in atoms_batch
    ]
    batch = backend.batch_graphs(graphs)
    with backend._torch.no_grad():  # pylint: disable=protected-access
        outputs = backend.model(
            batch.to_dict(), training=False, compute_force=False, compute_stress=False
        )
    energies = outputs["energy"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    if energies.size != len(atoms_batch) or not np.all(np.isfinite(energies)):
        raise RuntimeError("invalid batched state energies")
    return energies


def evaluate_gap_batch(
    *,
    backend: Any,
    atoms_batch: Sequence[Any],
    lower_charge: int,
    lower_spin: int,
    oxidized_charge: int,
    oxidized_spin: int,
    restraint: FlatBottomShell,
) -> GapBatch:
    if not atoms_batch:
        raise ValueError("cannot evaluate an empty frame batch")
    hashes = tuple(coordinate_sha256(atoms) for atoms in atoms_batch)
    lower = _state_energies(backend, atoms_batch, charge=lower_charge, spin=lower_spin)
    oxidized = _state_energies(
        backend, atoms_batch, charge=oxidized_charge, spin=oxidized_spin
    )
    restraint_energies = np.asarray(
        [restraint.evaluate(np.asarray(atoms.positions)).energy_eV for atoms in atoms_batch],
        dtype=np.float64,
    )
    lower_total = lower + restraint_energies
    oxidized_total = oxidized + restraint_energies
    delta = oxidized_total - lower_total
    if not np.allclose(delta, oxidized - lower, rtol=0.0, atol=1e-12):
        raise RuntimeError("flat-bottom restraint did not cancel from vertical gap")
    return GapBatch(hashes, lower, oxidized, delta, restraint_energies)


def write_gap_chunk(path: Path, batch: GapBatch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinate_sha256=np.asarray(batch.coordinate_sha256),
        lower_energy_eV=batch.lower_energy_eV,
        oxidized_energy_eV=batch.oxidized_energy_eV,
        delta_E_eV=batch.delta_E_eV,
        restraint_energy_eV=batch.restraint_energy_eV,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gap_chunks(paths: Sequence[Path]) -> NDArray[np.float64]:
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["delta_E_eV"], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid gap chunk: {path}")
        arrays.append(values)
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)
