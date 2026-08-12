from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calculator import PolarMACEStateCalculator
from .marcus import assemble_seed, assemble_system
from .trajectory import _read_tsv, _task, restraint_from_task
from .vertical_gap import evaluate_gap_batch, read_gap_chunks, write_gap_chunk


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_specs(raw_root: Path, mode: str, task: dict[str, str]) -> dict[str, tuple[int, int]]:
    peers = [
        row
        for row in _read_tsv(raw_root / f"{mode}_trajectory_tasks.tsv")
        if row["system_id"] == task["system_id"] and row["seed_index"] == task["seed_index"]
    ]
    specs = {row["state"]: (int(row["charge"]), int(row["spin"])) for row in peers}
    if set(specs) != {"lower", "oxidized"} or len(peers) != 2:
        raise RuntimeError("trajectory state pair is incomplete or duplicated")
    return specs


def evaluate_trajectory_gaps(
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
        raise RuntimeError("ASE is required for trajectory gap evaluation") from exc
    task = _task(raw_root, mode, task_index)
    trajectory_dir = raw_root / "trajectories" / task["logical_trajectory_id"]
    trajectory_receipt = trajectory_dir / "trajectory.json"
    if not trajectory_receipt.is_file():
        raise RuntimeError("trajectory receipt is missing")
    trajectory_payload = json.loads(trajectory_receipt.read_text(encoding="utf-8"))
    if trajectory_payload["status"] != "complete":
        raise RuntimeError("trajectory is not complete")
    gap_dir = raw_root / "gaps" / task["logical_trajectory_id"]
    receipt_path = gap_dir / "gaps.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "complete" and all(
            _sha256(raw_root / row["path"]) == row["sha256"] for row in receipt["gap_chunks"]
        ):
            return receipt
    gap_dir.mkdir(parents=True, exist_ok=True)
    specs = _state_specs(raw_root, mode, task)
    lower_charge, lower_spin = specs["lower"]
    oxidized_charge, oxidized_spin = specs["oxidized"]
    calculator = PolarMACEStateCalculator(
        checkpoint=checkpoint,
        charge=lower_charge,
        spin=lower_spin,
        device=device,
    )
    started = time.monotonic()
    gap_chunks: list[dict[str, Any]] = []
    for chunk_receipt_path in sorted((trajectory_dir / "md").glob("chunk-*.json")):
        chunk = json.loads(chunk_receipt_path.read_text(encoding="utf-8"))
        if chunk["phase"] != "production":
            continue
        if chunk["status"] != "complete":
            raise RuntimeError(f"incomplete production chunk: {chunk_receipt_path.name}")
        trajectory_path = chunk_receipt_path.with_suffix(".traj")
        frames = read(trajectory_path, index=":")
        if len(frames) != int(chunk["production_samples"]):
            raise RuntimeError("trajectory frame count differs from chunk receipt")
        restraint = restraint_from_task(frames[0], task)
        batch = evaluate_gap_batch(
            backend=calculator.backend,
            atoms_batch=frames,
            lower_charge=lower_charge,
            lower_spin=lower_spin,
            oxidized_charge=oxidized_charge,
            oxidized_spin=oxidized_spin,
            restraint=restraint,
        )
        output_path = gap_dir / f"gap-{int(chunk['chunk_index']):04d}.npz"
        digest = write_gap_chunk(output_path, batch)
        gap_chunks.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "samples": len(frames),
                "path": output_path.relative_to(raw_root).as_posix(),
                "sha256": digest,
            }
        )
    expected = int(trajectory_payload["md"]["expected_production_samples"])
    observed = sum(int(row["samples"]) for row in gap_chunks)
    if observed != expected:
        raise RuntimeError(f"gap sample count {observed} differs from expected {expected}")
    calculator.assert_model_unchanged()
    receipt = {
        **task,
        "status": "complete",
        "lower_charge": lower_charge,
        "lower_spin": lower_spin,
        "oxidized_charge": oxidized_charge,
        "oxidized_spin": oxidized_spin,
        "sample_count": observed,
        "gap_chunks": gap_chunks,
        "calculator": calculator.provenance_dict(),
        "wallclock_seconds": time.monotonic() - started,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _gaps(raw_root: Path, receipt: dict[str, Any]) -> np.ndarray:
    paths = [raw_root / row["path"] for row in receipt["gap_chunks"]]
    for row, path in zip(receipt["gap_chunks"], paths, strict=True):
        if _sha256(path) != row["sha256"]:
            raise RuntimeError("gap chunk hash drift")
    return read_gap_chunks(paths)


def collect_gap_summaries(*, raw_root: Path, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = _read_tsv(raw_root / f"{mode}_trajectory_tasks.tsv")
    by_seed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for task in tasks:
        receipt_path = raw_root / "gaps" / task["logical_trajectory_id"] / "gaps.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"missing gap receipt: {task['logical_trajectory_id']}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] != "complete":
            raise RuntimeError(f"incomplete gap receipt: {task['logical_trajectory_id']}")
        by_seed.setdefault((task["system_id"], task["seed_index"]), {})[task["state"]] = receipt
    seed_rows: list[dict[str, Any]] = []
    seed_objects: dict[str, list[Any]] = {}
    for (system_id, seed_index), states in sorted(by_seed.items()):
        if set(states) != {"lower", "oxidized"}:
            raise RuntimeError("gap state pair is incomplete")
        lower = _gaps(raw_root, states["lower"])
        oxidized = _gaps(raw_root, states["oxidized"])
        base: dict[str, Any] = {
            "system_id": system_id,
            "seed_index": int(seed_index),
            "lower_samples": int(lower.size),
            "oxidized_samples": int(oxidized.size),
            "mu_lower_eV": float(lower.mean()),
            "mu_oxidized_eV": float(oxidized.mean()),
            "lower_sd_eV": float(lower.std(ddof=1)) if lower.size > 1 else 0.0,
            "oxidized_sd_eV": float(oxidized.std(ddof=1)) if oxidized.size > 1 else 0.0,
        }
        if mode == "production":
            seed = assemble_seed(lower, oxidized)
            base.update(asdict(seed))
            seed_objects.setdefault(system_id, []).append(seed)
        seed_rows.append(base)
    seed_path = raw_root / f"{mode}_seed_gap_summary.csv"
    with seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(seed_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(seed_rows)
    system_rows: list[dict[str, Any]] = []
    if mode == "production":
        for system_id, seeds in sorted(seed_objects.items()):
            system_rows.append({"system_id": system_id, **asdict(assemble_system(seeds))})
        system_path = raw_root / "production_system_raw_predictions.csv"
        with system_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(system_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(system_rows)
    return seed_rows, system_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate-gaps")
    evaluate.add_argument("--raw-root", type=Path, required=True)
    evaluate.add_argument("--mode", choices=("pilot", "production"), required=True)
    evaluate.add_argument("--task-index", type=int, required=True)
    evaluate.add_argument("--checkpoint", default="polar-1-l")
    evaluate.add_argument("--device", default="cpu")
    collect = sub.add_parser("assemble-marcus")
    collect.add_argument("--raw-root", type=Path, required=True)
    collect.add_argument("--mode", choices=("pilot", "production"), required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate-gaps":
        payload = evaluate_trajectory_gaps(
            raw_root=args.raw_root,
            mode=args.mode,
            task_index=args.task_index,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        seeds, systems = collect_gap_summaries(raw_root=args.raw_root, mode=args.mode)
        payload = {"status": "PASS", "seed_rows": len(seeds), "system_rows": len(systems)}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
