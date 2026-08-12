from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .calculator import PolarMACEStateCalculator
from .optimize import atoms_geometry_sha256


def probe_device(
    *, geometry: Path, charge: int, spin: int, device: str, output: Path
) -> dict[str, object]:
    try:
        from ase.io import read
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for the device probe") from exc
    atoms = read(geometry)
    started = time.monotonic()
    calculator = PolarMACEStateCalculator(
        checkpoint="polar-1-l", charge=charge, spin=spin, device=device
    )
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    energy = calculator.get_potential_energy(atoms)
    forces = calculator.get_forces(atoms)
    evaluation_seconds = time.monotonic() - started
    calculator.assert_model_unchanged()
    payload: dict[str, object] = {
        "status": "PASS",
        "device": device,
        "geometry": geometry.as_posix(),
        "geometry_sha256": atoms_geometry_sha256(atoms),
        "charge": charge,
        "spin": spin,
        "energy_eV": energy,
        "maximum_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "forces_sha256": hashlib.sha256(np.asarray(forces, dtype="<f8").tobytes()).hexdigest(),
        "forces_eV_A": np.asarray(forces).tolist(),
        "model_load_seconds": load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "calculator": calculator.provenance_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def compare_probes(cpu: Path, gpu: Path, *, energy_atol_eV: float = 1e-8, force_atol_eV_A: float = 1e-7) -> dict[str, object]:
    cpu_payload = json.loads(cpu.read_text(encoding="utf-8"))
    gpu_payload = json.loads(gpu.read_text(encoding="utf-8"))
    if cpu_payload["geometry_sha256"] != gpu_payload["geometry_sha256"]:
        raise RuntimeError("device probes used different coordinates")
    if cpu_payload["calculator"]["checkpoint_sha256"] != gpu_payload["calculator"]["checkpoint_sha256"]:
        raise RuntimeError("device probes used different checkpoints")
    energy_difference = float(gpu_payload["energy_eV"]) - float(cpu_payload["energy_eV"])
    force_difference = float(
        np.max(
            np.abs(
                np.asarray(gpu_payload["forces_eV_A"], dtype=np.float64)
                - np.asarray(cpu_payload["forces_eV_A"], dtype=np.float64)
            )
        )
    )
    return {
        "status": "PASS"
        if abs(energy_difference) <= energy_atol_eV and force_difference <= force_atol_eV_A
        else "FAIL",
        "energy_difference_eV": energy_difference,
        "maximum_force_component_difference_eV_A": force_difference,
        "energy_atol_eV": energy_atol_eV,
        "force_atol_eV_A": force_atol_eV_A,
        "cpu_evaluation_seconds": cpu_payload["evaluation_seconds"],
        "gpu_evaluation_seconds": gpu_payload["evaluation_seconds"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--geometry", type=Path, required=True)
    run.add_argument("--charge", type=int, required=True)
    run.add_argument("--spin", type=int, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--cpu", type=Path, required=True)
    compare.add_argument("--gpu", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        payload = probe_device(
            geometry=args.geometry,
            charge=args.charge,
            spin=args.spin,
            device=args.device,
            output=args.output,
        )
    else:
        payload = compare_probes(args.cpu, args.gpu)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
