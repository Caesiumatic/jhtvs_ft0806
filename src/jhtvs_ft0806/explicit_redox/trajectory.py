from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calculator import PolarMACEStateCalculator
from .dynamics import run_md
from .optimize import optimize_state
from .packing import read_xyz, target_radius_A
from .restraint import FlatBottomShell


TASK_FIELDS = (
    "task_index",
    "logical_trajectory_id",
    "system_id",
    "seed_index",
    "state",
    "charge",
    "spin",
    "cluster_geometry_path",
    "cluster_geometry_sha256",
    "target_atoms",
    "solvent_atoms",
    "R0_A",
    "velocity_seed",
    "mode",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _velocity_seed(logical_id: str) -> int:
    return 1_000_000 + int.from_bytes(
        hashlib.sha256(f"mace-polar-5solv-v1|velocity|{logical_id}".encode()).digest()[:4], "big"
    ) % 8_000_000


def prepare_trajectory_tasks(
    *, cluster_manifest: Path, raw_root: Path, mode: str
) -> list[dict[str, object]]:
    if mode not in {"pilot", "production"}:
        raise ValueError("trajectory mode must be pilot or production")
    rows: list[dict[str, object]] = []
    for cluster in _read(cluster_manifest):
        if cluster["status"] != "clean":
            continue
        geometry_path = raw_root / cluster["geometry_path"]
        geometry = read_xyz(geometry_path)
        target_atoms = int(cluster["target_atoms"])
        radius = target_radius_A(
            type(geometry)(geometry.symbols[:target_atoms], geometry.positions[:target_atoms])
        )
        for state in ("lower", "oxidized"):
            logical_id = f"{cluster['system_id']}__seed-{cluster['seed_index']}__{state}"
            rows.append(
                {
                    "task_index": len(rows) + 1,
                    "logical_trajectory_id": logical_id,
                    "system_id": cluster["system_id"],
                    "seed_index": cluster["seed_index"],
                    "state": state,
                    "charge": cluster[f"{state}_charge"],
                    "spin": cluster[f"{state}_spin"],
                    "cluster_geometry_path": cluster["geometry_path"],
                    "cluster_geometry_sha256": cluster["geometry_sha256"],
                    "target_atoms": target_atoms,
                    "solvent_atoms": cluster["solvent_atoms"],
                    "R0_A": format(radius + 4.5, ".10f"),
                    "velocity_seed": _velocity_seed(logical_id),
                    "mode": mode,
                }
            )
    path = raw_root / f"{mode}_trajectory_tasks.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _task(raw_root: Path, mode: str, task_index: int) -> dict[str, str]:
    matches = [
        row
        for row in _read_tsv(raw_root / f"{mode}_trajectory_tasks.tsv")
        if int(row["task_index"]) == task_index
    ]
    if len(matches) != 1:
        raise ValueError(f"trajectory task index not unique: {task_index}")
    return matches[0]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def restraint_from_task(atoms: Any, task: dict[str, str]) -> FlatBottomShell:
    target_atoms = int(task["target_atoms"])
    solvent_atoms = int(task["solvent_atoms"])
    target_heavy = [
        index
        for index, symbol in enumerate(atoms.get_chemical_symbols()[:target_atoms])
        if symbol != "H"
    ]
    solvent_groups = [
        list(range(target_atoms + index * solvent_atoms, target_atoms + (index + 1) * solvent_atoms))
        for index in range(5)
    ]
    if target_atoms + 5 * solvent_atoms != len(atoms):
        raise RuntimeError("trajectory task atom grouping does not close")
    return FlatBottomShell(
        target_heavy_indices=target_heavy,
        solvent_groups=solvent_groups,
        masses=atoms.get_masses(),
        R0_A=float(task["R0_A"]),
        k_eV_A2=0.5,
    )


def run_trajectory_task(
    *,
    raw_root: Path,
    mode: str,
    task_index: int,
    checkpoint: str = "polar-1-l",
    device: str = "cpu",
) -> dict[str, Any]:
    try:
        from ase.io import read
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for trajectory execution") from exc
    task = _task(raw_root, mode, task_index)
    input_path = raw_root / task["cluster_geometry_path"]
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != task["cluster_geometry_sha256"]:
        raise RuntimeError("cluster geometry hash drift")
    atoms = read(input_path)
    restraint = restraint_from_task(atoms, task)
    calculator = PolarMACEStateCalculator(
        checkpoint=checkpoint,
        charge=int(task["charge"]),
        spin=int(task["spin"]),
        device=device,
    )
    output_dir = raw_root / "trajectories" / task["logical_trajectory_id"]
    started = time.monotonic()
    optimization = optimize_state(
        atoms=atoms,
        model_calculator=calculator,
        restraint=restraint,
        charge=int(task["charge"]),
        spin=int(task["spin"]),
        output_dir=output_dir / "optimization",
    )
    if not optimization["converged"]:
        raise RuntimeError("state optimization did not converge")
    optimized_atoms = read(output_dir / "optimization" / "optimized.xyz")
    dynamics_kwargs = (
        {"equilibration_ps": 1.0, "production_ps": 2.0, "checkpoint_ps": 1.0}
        if mode == "pilot"
        else {}
    )
    md = run_md(
        logical_id=task["logical_trajectory_id"],
        atoms=optimized_atoms,
        model_calculator=calculator,
        restraint=restraint,
        charge=int(task["charge"]),
        spin=int(task["spin"]),
        velocity_seed=int(task["velocity_seed"]),
        output_dir=output_dir / "md",
        **dynamics_kwargs,
    )
    calculator.assert_model_unchanged()
    receipt = {
        **task,
        "status": "complete" if md["status"] == "complete" else "incomplete",
        "optimization": optimization,
        "md": md,
        "calculator": calculator.provenance_dict(),
        "wallclock_seconds": time.monotonic() - started,
    }
    (output_dir / "trajectory.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--cluster-manifest", type=Path, required=True)
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--mode", choices=("pilot", "production"), required=True)
    run = sub.add_parser("run-task")
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--mode", choices=("pilot", "production"), required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--checkpoint", default="polar-1-l")
    run.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        rows = prepare_trajectory_tasks(
            cluster_manifest=args.cluster_manifest, raw_root=args.raw_root, mode=args.mode
        )
        print(json.dumps({"status": "PASS", "tasks": len(rows)}, sort_keys=True))
    else:
        print(
            json.dumps(
                run_trajectory_task(
                    raw_root=args.raw_root,
                    mode=args.mode,
                    task_index=args.task_index,
                    checkpoint=args.checkpoint,
                    device=args.device,
                ),
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
