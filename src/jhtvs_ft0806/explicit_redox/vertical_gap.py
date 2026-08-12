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
    lower_diagnostics: dict[str, NDArray[np.float64]]
    oxidized_diagnostics: dict[str, NDArray[np.float64]]


def coordinate_sha256(atoms: Any) -> str:
    digest = hashlib.sha256()
    digest.update(";".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def _state_outputs(
    backend: Any,
    atoms_batch: Sequence[Any],
    *,
    charge: int,
    spin: int,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
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
    atom_count = len(atoms_batch[0].get_chemical_symbols())
    diagnostics: dict[str, NDArray[np.float64]] = {}
    for key in ("density_coefficients", "spin_density", "spin_charge_density"):
        if outputs.get(key) is None:
            raise RuntimeError(f"batched PolarMACE output missing {key}")
        values = outputs[key].detach().cpu().numpy().astype(np.float64)
        if values.shape[0] != len(atoms_batch) * atom_count or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid batched {key}")
        diagnostics[key] = values.reshape(len(atoms_batch), atom_count, *values.shape[1:])
    return energies, diagnostics


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
    atom_counts = {len(atoms.get_chemical_symbols()) for atoms in atoms_batch}
    if len(atom_counts) != 1:
        raise ValueError("all frames in a gap batch must have the same atom count")
    lower, lower_diagnostics = _state_outputs(
        backend, atoms_batch, charge=lower_charge, spin=lower_spin
    )
    oxidized, oxidized_diagnostics = _state_outputs(
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
    return GapBatch(
        hashes,
        lower,
        oxidized,
        delta,
        restraint_energies,
        lower_diagnostics,
        oxidized_diagnostics,
    )


def evaluate_gap_batch_chunked(
    *,
    backend: Any,
    atoms_batch: Sequence[Any],
    lower_charge: int,
    lower_spin: int,
    oxidized_charge: int,
    oxidized_spin: int,
    restraint: FlatBottomShell,
    batch_size: int = 5,
) -> GapBatch:
    if batch_size < 1:
        raise ValueError("gap batch size must be positive")
    parts = [
        evaluate_gap_batch(
            backend=backend,
            atoms_batch=atoms_batch[start : start + batch_size],
            lower_charge=lower_charge,
            lower_spin=lower_spin,
            oxidized_charge=oxidized_charge,
            oxidized_spin=oxidized_spin,
            restraint=restraint,
        )
        for start in range(0, len(atoms_batch), batch_size)
    ]
    if not parts:
        raise ValueError("cannot evaluate an empty frame batch")
    diagnostic_keys = set(parts[0].lower_diagnostics)
    if any(
        set(part.lower_diagnostics) != diagnostic_keys
        or set(part.oxidized_diagnostics) != diagnostic_keys
        for part in parts
    ):
        raise RuntimeError("gap diagnostic keys changed between microbatches")
    return GapBatch(
        coordinate_sha256=tuple(value for part in parts for value in part.coordinate_sha256),
        lower_energy_eV=np.concatenate([part.lower_energy_eV for part in parts]),
        oxidized_energy_eV=np.concatenate([part.oxidized_energy_eV for part in parts]),
        delta_E_eV=np.concatenate([part.delta_E_eV for part in parts]),
        restraint_energy_eV=np.concatenate([part.restraint_energy_eV for part in parts]),
        lower_diagnostics={
            key: np.concatenate([part.lower_diagnostics[key] for part in parts])
            for key in sorted(diagnostic_keys)
        },
        oxidized_diagnostics={
            key: np.concatenate([part.oxidized_diagnostics[key] for part in parts])
            for key in sorted(diagnostic_keys)
        },
    )


def write_gap_chunk(path: Path, batch: GapBatch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, NDArray[Any]] = {
        "coordinate_sha256": np.asarray(batch.coordinate_sha256),
        "lower_energy_eV": batch.lower_energy_eV,
        "oxidized_energy_eV": batch.oxidized_energy_eV,
        "delta_E_eV": batch.delta_E_eV,
        "restraint_energy_eV": batch.restraint_energy_eV,
    }
    for state, diagnostics in (
        ("lower", batch.lower_diagnostics),
        ("oxidized", batch.oxidized_diagnostics),
    ):
        for name, values in diagnostics.items():
            payload[f"{state}_{name}"] = values
    np.savez_compressed(path, **payload)
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
