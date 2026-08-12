from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .calculator import apply_state_metadata
from .optimize import RestrainedCalculator, atoms_geometry_sha256
from .restraint import FlatBottomShell


@dataclass(frozen=True)
class MDChunk:
    index: int
    phase: str
    start_step: int
    steps: int
    random_seed: int


def chunk_plan(
    logical_id: str,
    *,
    timestep_fs: float = 0.5,
    equilibration_ps: float = 50.0,
    production_ps: float = 150.0,
    checkpoint_ps: float = 1.0,
) -> list[MDChunk]:
    steps_per_chunk = round(checkpoint_ps * 1000.0 / timestep_fs)
    equilibration_steps = round(equilibration_ps * 1000.0 / timestep_fs)
    production_steps = round(production_ps * 1000.0 / timestep_fs)
    if equilibration_steps % steps_per_chunk or production_steps % steps_per_chunk:
        raise ValueError("phase lengths must be integer checkpoint chunks")
    chunks: list[MDChunk] = []
    total_chunks = (equilibration_steps + production_steps) // steps_per_chunk
    for index in range(total_chunks):
        start = index * steps_per_chunk
        phase = "equilibration" if start < equilibration_steps else "production"
        seed = 1_000_000 + int.from_bytes(
            hashlib.sha256(f"{logical_id}|md-chunk|{index}".encode()).digest()[:4], "big"
        ) % 8_000_000
        chunks.append(MDChunk(index, phase, start, steps_per_chunk, seed))
    return chunks


def completed_chunk_indices(directory: Path) -> set[int]:
    completed: set[int] = set()
    for path in directory.glob("chunk-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            completed.add(int(payload["chunk_index"]))
    return completed


def pending_chunks(plan: Sequence[MDChunk], directory: Path) -> list[MDChunk]:
    completed = completed_chunk_indices(directory)
    return [chunk for chunk in plan if chunk.index not in completed]


def run_md(
    *,
    logical_id: str,
    atoms: Any,
    model_calculator: Any,
    restraint: FlatBottomShell,
    charge: int,
    spin: int,
    velocity_seed: int,
    output_dir: Path,
    temperature_K: float = 300.0,
    timestep_fs: float = 0.5,
    friction_fs_inverse: float = 0.01,
    equilibration_ps: float = 50.0,
    production_ps: float = 150.0,
    checkpoint_ps: float = 1.0,
    sample_interval_fs: float = 20.0,
) -> dict[str, Any]:
    try:
        from ase import units
        from ase.io import read, write
        from ase.io.trajectory import Trajectory
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for molecular dynamics") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = chunk_plan(
        logical_id,
        timestep_fs=timestep_fs,
        equilibration_ps=equilibration_ps,
        production_ps=production_ps,
        checkpoint_ps=checkpoint_ps,
    )
    done = completed_chunk_indices(output_dir)
    if done:
        latest = max(done)
        if done != set(range(latest + 1)):
            raise RuntimeError("MD chunk ledger has a gap")
        atoms = read(output_dir / f"chunk-{latest:04d}.traj", index=-1)
    else:
        rng = np.random.default_rng(velocity_seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
        Stationary(atoms)
    apply_state_metadata(atoms, charge=charge, spin=spin)
    atoms.calc = RestrainedCalculator(model_calculator, restraint)
    sample_every = round(sample_interval_fs / timestep_fs)
    production_samples = 0
    for chunk in pending_chunks(plan, output_dir):
        chunk_restraint_active_steps = 0
        chunk_maximum_excursion = 0.0
        rng = np.random.default_rng(chunk.random_seed)
        dynamics = Langevin(
            atoms,
            timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=friction_fs_inverse / units.fs,
            fixcm=True,
            rng=rng,
        )
        trajectory_path = output_dir / f"chunk-{chunk.index:04d}.traj"
        trajectory = Trajectory(trajectory_path, "w", atoms)

        def sample() -> None:
            nonlocal chunk_restraint_active_steps, chunk_maximum_excursion, production_samples
            if dynamics.nsteps == 0:
                return
            result = restraint.evaluate(np.asarray(atoms.positions, dtype=np.float64))
            chunk_restraint_active_steps += int(result.active_count > 0)
            chunk_maximum_excursion = max(
                chunk_maximum_excursion, result.maximum_excursion_A
            )
            if chunk.phase == "production":
                trajectory.write(atoms)
                production_samples += 1

        dynamics.attach(sample, interval=sample_every)
        dynamics.run(chunk.steps)
        if chunk.phase == "equilibration":
            trajectory.write(atoms)
        trajectory.close()
        write(output_dir / "latest.extxyz", atoms)
        chunk_receipt = {
            "status": "complete",
            "logical_id": logical_id,
            "chunk_index": chunk.index,
            "phase": chunk.phase,
            "steps": chunk.steps,
            "random_seed": chunk.random_seed,
            "end_geometry_sha256": atoms_geometry_sha256(atoms),
            "production_samples": 0 if chunk.phase == "equilibration" else chunk.steps // sample_every,
            "restraint_activation_samples": chunk_restraint_active_steps,
            "maximum_excursion_A": chunk_maximum_excursion,
        }
        (output_dir / f"chunk-{chunk.index:04d}.json").write_text(
            json.dumps(chunk_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    expected_samples = round(production_ps * 1000.0 / sample_interval_fs)
    chunk_receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("chunk-*.json"))
    ]
    receipt = {
        "status": "complete" if not pending_chunks(plan, output_dir) else "incomplete",
        "logical_id": logical_id,
        "charge": charge,
        "spin": spin,
        "chunks": len(plan),
        "completed_chunks": len(completed_chunk_indices(output_dir)),
        "expected_production_samples": expected_samples,
        "completed_production_samples": sum(int(row["production_samples"]) for row in chunk_receipts),
        "new_production_samples": production_samples,
        "restraint_activation_samples": sum(
            int(row["restraint_activation_samples"]) for row in chunk_receipts
        ),
        "maximum_excursion_A": max(
            (float(row["maximum_excursion_A"]) for row in chunk_receipts), default=0.0
        ),
    }
    (output_dir / "md.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
