from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from .calculator import PolarMACEStateCalculator, apply_state_metadata
from .optimize import atoms_geometry_sha256
from .packing import read_xyz
from .structures import ConformerRecord, select_mace_conformers, tfsi_family


TASK_FIELDS = ("task_index", "entity_id", "conformer_id", "formal_charge", "spin", "input_xyz", "output_dir")


def prepare_isolated_tasks(raw_root: Path) -> list[dict[str, object]]:
    with (raw_root / "structure_candidates.csv").open(encoding="utf-8", newline="") as handle:
        entities = list(csv.DictReader(handle))
    tasks: list[dict[str, object]] = []
    for entity in entities:
        directory = raw_root / entity["directory"]
        records = json.loads((directory / "conformers.json").read_text(encoding="utf-8"))
        for record in records:
            tasks.append(
                {
                    "task_index": len(tasks) + 1,
                    "entity_id": entity["entity_id"],
                    "conformer_id": record["conformer_id"],
                    "formal_charge": entity["formal_charge"],
                    "spin": 1,
                    "input_xyz": (directory / record["xyz_path"]).relative_to(raw_root).as_posix(),
                    "output_dir": f"isolated_optimized/{entity['entity_id']}/conf-{int(record['conformer_id']):03d}",
                }
            )
    path = raw_root / "isolated_tasks.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)
    return tasks


def _task(raw_root: Path, task_index: int) -> dict[str, str]:
    with (raw_root / "isolated_tasks.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if int(row["task_index"]) == task_index]
    if len(matches) != 1:
        raise ValueError(f"isolated task index not unique: {task_index}")
    return matches[0]


def run_isolated_task(
    *,
    raw_root: Path,
    task_index: int,
    checkpoint: str = "polar-1-l",
    device: str = "cpu",
    fmax_eV_A: float = 0.02,
    max_steps: int = 5000,
) -> dict[str, Any]:
    try:
        from ase.io import read, write
        from ase.optimize import FIRE
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for isolated optimization") from exc
    task = _task(raw_root, task_index)
    output_dir = raw_root / task["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "result.json"
    optimized_path = output_dir / "optimized.xyz"
    if receipt_path.is_file() and optimized_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "clean":
            return receipt
    atoms = read(raw_root / task["input_xyz"])
    charge, spin = int(task["formal_charge"]), int(task["spin"])
    apply_state_metadata(atoms, charge=charge, spin=spin)
    calculator = PolarMACEStateCalculator(
        checkpoint=checkpoint, charge=charge, spin=spin, device=device
    )
    atoms.calc = calculator
    optimizer = FIRE(
        atoms,
        restart=str(output_dir / "fire.restart.json"),
        logfile=str(output_dir / "fire.log"),
        trajectory=str(output_dir / "optimization.traj"),
    )
    converged = bool(optimizer.run(fmax=fmax_eV_A, steps=max_steps))
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    calculator.assert_model_unchanged()
    write(optimized_path, atoms)
    receipt = {
        **task,
        "status": "clean" if converged else "incomplete",
        "converged": converged,
        "steps": int(optimizer.nsteps),
        "energy_eV": energy,
        "maximum_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "geometry_sha256": atoms_geometry_sha256(atoms),
        "optimized_xyz_sha256": hashlib.sha256(optimized_path.read_bytes()).hexdigest(),
        "calculator": calculator.provenance_dict(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _tfsi_family_from_xyz(smiles: str, path: Path) -> str | None:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    nitrogen = next(
        (
            atom
            for atom in molecule.GetAtoms()
            if atom.GetSymbol() == "N"
            and sum(neighbor.GetSymbol() == "S" for neighbor in atom.GetNeighbors()) == 2
        ),
        None,
    )
    if nitrogen is None:
        return None
    geometry = read_xyz(path)
    symbols = tuple(atom.GetSymbol() for atom in molecule.GetAtoms())
    if geometry.symbols != symbols:
        raise RuntimeError("optimized TFSI atom ordering drift")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, (x, y, z) in enumerate(geometry.positions):
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.RemoveAllConformers()
    conformer_id = molecule.AddConformer(conformer, assignId=True)
    return tfsi_family(molecule, conformer_id)


def collect_isolated(raw_root: Path, *, window_eV: float = 0.25) -> list[dict[str, object]]:
    tasks = prepare_isolated_tasks(raw_root)
    with (raw_root / "structure_candidates.csv").open(encoding="utf-8", newline="") as handle:
        entity_metadata = {row["entity_id"]: row for row in csv.DictReader(handle)}
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        receipt = raw_root / str(task["output_dir"]) / "result.json"
        if not receipt.is_file():
            raise RuntimeError(f"missing isolated result: task {task['task_index']}")
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload["status"] != "clean":
            raise RuntimeError(f"unclean isolated result: task {task['task_index']}")
        by_entity.setdefault(str(task["entity_id"]), []).append(payload)
    selected: list[dict[str, object]] = []
    for entity_id, rows in sorted(by_entity.items()):
        ranked = sorted(rows, key=lambda row: (float(row["energy_eV"]), int(row["conformer_id"])))
        minimum = float(ranked[0]["energy_eV"])
        records = [
            ConformerRecord(
                conformer_id=int(row["conformer_id"]),
                embed_seed=0,
                geometry_sha256=str(row["geometry_sha256"]),
                mmff_variant="not_used_for_selection",
                mmff_energy_eV=None,
                xyz_path=f"{row['output_dir']}/optimized.xyz",
            )
            for row in rows
        ]
        families = [
            _tfsi_family_from_xyz(
                entity_metadata[entity_id]["canonical_smiles"],
                raw_root / str(row["output_dir"]) / "optimized.xyz",
            )
            for row in rows
        ]
        family_selection = None if all(family is None for family in families) else [str(family) for family in families]
        chosen = select_mace_conformers(
            records,
            [float(row["energy_eV"]) for row in rows],
            window_eV=window_eV,
            families=family_selection,
        )
        by_conformer = {int(row["conformer_id"]): row for row in rows}
        family_by_conformer = {
            record.conformer_id: family for record, family in zip(records, families, strict=True)
        }
        for rank, record in enumerate(chosen):
            row = by_conformer[record.conformer_id]
            selected.append(
                {
                    "entity_id": entity_id,
                    "rank": rank,
                    "conformer_id": row["conformer_id"],
                    "energy_eV": row["energy_eV"],
                    "relative_energy_eV": float(row["energy_eV"]) - minimum,
                    "conformer_family": family_by_conformer[record.conformer_id] or "",
                    "geometry_sha256": row["geometry_sha256"],
                    "optimized_xyz": f"{row['output_dir']}/optimized.xyz",
                }
            )
    fields = ("entity_id", "rank", "conformer_id", "energy_eV", "relative_energy_eV", "conformer_family", "geometry_sha256", "optimized_xyz")
    with (raw_root / "isolated_selected.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--raw-root", type=Path, required=True)
    run = sub.add_parser("run-task")
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--checkpoint", default="polar-1-l")
    run.add_argument("--device", default="cpu")
    collect = sub.add_parser("collect")
    collect.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps({"tasks": len(prepare_isolated_tasks(args.raw_root))}))
    elif args.command == "run-task":
        print(json.dumps(run_isolated_task(raw_root=args.raw_root, task_index=args.task_index, checkpoint=args.checkpoint, device=args.device), sort_keys=True))
    else:
        print(json.dumps({"selected": len(collect_isolated(args.raw_root))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
