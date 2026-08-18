#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from common import load_metadata, read_csv, read_xyz, repo_root, sha256_file

VERSION_RE = re.compile(r"(?:xtb version|version)\s+([0-9]+(?:\.[0-9]+){1,2})", re.I)
ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", re.I)


def parse_energy(text: str) -> float:
    matches = ENERGY_RE.findall(text)
    if not matches:
        raise ValueError("TOTAL ENERGY in Eh not found")
    return float(matches[-1])


def parse_charges(path: Path, atom_count: int) -> list[float]:
    values = []
    for token in path.read_text(encoding="utf-8").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if len(values) != atom_count:
        raise ValueError(f"expected {atom_count} atomic charges, found {len(values)}")
    return values


def xtb_version(executable: str) -> tuple[str, str]:
    resolved = shutil.which(executable)
    if not resolved:
        raise RuntimeError(f"xTB executable not found: {executable}")
    completed = subprocess.run([resolved, "--version"], text=True, capture_output=True, check=True)
    text = completed.stdout + completed.stderr
    match = VERSION_RE.search(text)
    if not match:
        raise RuntimeError(f"could not parse xTB version from: {text.strip()}")
    version = match.group(1)
    parts = tuple(int(part) for part in version.split("."))
    if parts < (6, 6):
        raise RuntimeError(f"xTB >= 6.6 required for --cosmo; found {version}")
    return resolved, version


def restraint_text(metadata: dict, topology: str, force: float) -> str:
    anchors = metadata["anchor_indices_zero_based"]
    left, middle, right = topology
    return (
        "$constrain\n"
        f"  force constant={force:.8f}\n"
        f"  distance: {anchors[left] + 1}, {anchors[middle] + 1}, auto\n"
        f"  distance: {anchors[middle] + 1}, {anchors[right] + 1}, auto\n"
        "$end\n"
    )


def _state_complete(state_dir: Path, atom_count: int, optimized: bool) -> bool:
    try:
        parse_energy((state_dir / "xtb.out").read_text(encoding="utf-8"))
        parse_charges(state_dir / "charges", atom_count)
        if optimized:
            atoms, _ = read_xyz(state_dir / "xtbopt.xyz")
            if len(atoms) != atom_count:
                return False
    except (OSError, ValueError):
        return False
    return True


def _run(command: list[str], cwd: Path) -> None:
    with (cwd / "xtb.out").open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"xTB failed with exit code {completed.returncode}: {' '.join(command)}")


def run_task(row: dict[str, str], executable: str, run_root: Path, force: bool = False) -> dict:
    root = repo_root()
    input_xyz = root / row["input_xyz"]
    metadata = load_metadata(input_xyz)
    atoms, _ = read_xyz(input_xyz)
    atom_count = len(atoms)
    xtb_path, version = xtb_version(executable)
    task_dir = run_root / row["task_id"]
    optimization_dir = task_dir / "reduced_opt"
    reduced_sp_dir = task_dir / "reduced_sp"
    oxidized_sp_dir = task_dir / "oxidized_sp"
    for state_dir in (optimization_dir, reduced_sp_dir, oxidized_sp_dir):
        state_dir.mkdir(parents=True, exist_ok=True)

    optimization_input = optimization_dir / "in.xyz"
    optimization_command = [xtb_path, "in.xyz", "--gfn", "2", "--chrg", str(row["charge_reduced"]), "--uhf", str(row["uhf_reduced"]), "--opt", "normal", "--cosmo", str(row["epsilon"])]
    if row["restraint"] != "none":
        control = optimization_dir / "xcontrol.inp"
        control.write_text(restraint_text(metadata, row["topology"], float(row["restraint_force_constant_eh_bohr2"])), encoding="utf-8")
        optimization_command.extend(["--input", control.name])
    optimized_xyz = optimization_dir / "xtbopt.xyz"
    reduced_sp_input = reduced_sp_dir / "in.xyz"
    oxidized_sp_input = oxidized_sp_dir / "in.xyz"
    reduced_sp_command = [xtb_path, "in.xyz", "--gfn", "2", "--chrg", str(row["charge_reduced"]), "--uhf", str(row["uhf_reduced"]), "--cosmo", str(row["epsilon"])]
    oxidized_sp_command = [xtb_path, "in.xyz", "--gfn", "2", "--chrg", str(row["charge_oxidized"]), "--uhf", str(row["uhf_oxidized"]), "--cosmo", str(row["epsilon"])]
    optimization_executed = False
    reduced_sp_executed = False
    oxidized_sp_executed = False
    provenance = {
        "task": row,
        "status": "running",
        "input_geometry_sha256": sha256_file(input_xyz),
        "xtb_executable": xtb_path,
        "xtb_version": version,
        "optimization_command": optimization_command,
        "reduced_sp_command": reduced_sp_command,
        "oxidized_sp_command": oxidized_sp_command,
        "restraint": {
            "form": row["restraint"],
            "force_constant_eh_bohr2": row["restraint_force_constant_eh_bohr2"],
            "reference_distances": "auto from initial geometry" if row["restraint"] != "none" else "not applicable",
            "applies_to": "reduced_opt_only" if row["restraint"] != "none" else "not applicable",
        },
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    try:
        if force or not _state_complete(optimization_dir, atom_count, optimized=True):
            shutil.copy2(input_xyz, optimization_input)
            optimization_executed = True
            _run(optimization_command, optimization_dir)
            if not _state_complete(optimization_dir, atom_count, optimized=True):
                raise RuntimeError("reduced optimization ended without parseable energy, charges, and xtbopt.xyz")

        if force or not _state_complete(reduced_sp_dir, atom_count, optimized=False) or not reduced_sp_input.exists() or sha256_file(reduced_sp_input) != sha256_file(optimized_xyz):
            shutil.copy2(optimized_xyz, reduced_sp_input)
            reduced_sp_executed = True
            _run(reduced_sp_command, reduced_sp_dir)
            if not _state_complete(reduced_sp_dir, atom_count, optimized=False):
                raise RuntimeError("reduced single point ended without parseable energy and charges")

        if force or not _state_complete(oxidized_sp_dir, atom_count, optimized=False) or not oxidized_sp_input.exists() or sha256_file(oxidized_sp_input) != sha256_file(optimized_xyz):
            shutil.copy2(optimized_xyz, oxidized_sp_input)
            oxidized_sp_executed = True
            _run(oxidized_sp_command, oxidized_sp_dir)
            if not _state_complete(oxidized_sp_dir, atom_count, optimized=False):
                raise RuntimeError("oxidized single point ended without parseable energy and charges")
        optimized_hash = sha256_file(optimized_xyz)
        reduced_sp_hash = sha256_file(reduced_sp_input)
        oxidized_sp_hash = sha256_file(oxidized_sp_input)
        provenance.update({
            "status": "complete",
            "optimized_geometry_sha256": optimized_hash,
            "reduced_sp_input_geometry_sha256": reduced_sp_hash,
            "oxidized_sp_input_geometry_sha256": oxidized_sp_hash,
            "same_geometry_reduced_sp": optimized_hash == reduced_sp_hash,
            "same_geometry_oxidized_sp": optimized_hash == oxidized_sp_hash,
        })
    except Exception as exc:
        provenance.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        provenance["optimization_executed_this_invocation"] = optimization_executed
        provenance["reduced_sp_executed_this_invocation"] = reduced_sp_executed
        provenance["oxidized_sp_executed_this_invocation"] = oxidized_sp_executed
        (task_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    root = repo_root()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "chauhan_cation_eox" / "calculation_manifest.csv")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--task-id")
    parser.add_argument("--xtb", default="xtb")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "chauhan_cation_eox")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_csv(args.manifest.resolve())
    if args.task_index is not None:
        matches = [row for row in rows if int(row["task_index"]) == args.task_index]
    else:
        matches = [row for row in rows if row["task_id"] == args.task_id]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one manifest task, found {len(matches)}")
    result = run_task(matches[0], args.xtb, args.run_root.resolve(), args.force)
    print(json.dumps({"task_id": matches[0]["task_id"], "status": result["status"]}))


if __name__ == "__main__":
    main()
