#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    from .common import load_metadata, read_csv, read_xyz, repo_root, sha256_file
except ImportError:
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
    return resolved, match.group(1)


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
        if optimized and len(read_xyz(state_dir / "xtbopt.xyz")[0]) != atom_count:
            return False
    except (OSError, ValueError):
        return False
    return True


def _run(command: list[str], cwd: Path, output_name: str = "xtb.out") -> None:
    with (cwd / output_name).open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"xTB failed with exit code {completed.returncode}: {' '.join(command)}")


def run_task(row: dict[str, str], executable: str, run_root: Path, force: bool = False) -> dict:
    if row["environment"] != "vacuum":
        raise ValueError(f"primary Fadel task is not vacuum: {row['task_id']}")
    input_xyz = repo_root() / row["input_xyz"]
    metadata = load_metadata(input_xyz)
    atom_count = len(read_xyz(input_xyz)[0])
    xtb_path, version = xtb_version(executable)
    task_dir = run_root / row["task_id"]
    optimization_dir, reduced_sp_dir, oxidized_sp_dir = (task_dir / state for state in ("reduced_opt", "reduced_sp", "oxidized_sp"))
    for state_dir in (optimization_dir, reduced_sp_dir, oxidized_sp_dir):
        state_dir.mkdir(parents=True, exist_ok=True)

    optimization_input = optimization_dir / "in.xyz"
    optimized_xyz = optimization_dir / "xtbopt.xyz"
    reduced_sp_input = reduced_sp_dir / "in.xyz"
    oxidized_sp_input = oxidized_sp_dir / "in.xyz"
    base_reduced = [xtb_path, "in.xyz", "--gfn", "2", "--chrg", str(row["charge_reduced"]), "--uhf", str(row["uhf_reduced"]), "--iterations", "500"]
    optimization_command = [*base_reduced, "--opt", "normal"]
    if row["restraint"] != "none":
        control = optimization_dir / "xcontrol.inp"
        control.write_text(restraint_text(metadata, row["topology"], float(row["restraint_force_constant_eh_bohr2"])), encoding="utf-8")
        optimization_command.extend(["--input", control.name])
    reduced_sp_command = base_reduced
    oxidized_sp_command = [xtb_path, "in.xyz", "--gfn", "2", "--chrg", str(row["charge_oxidized"]), "--uhf", str(row["uhf_oxidized"]), "--iterations", "500"]
    if any("--cosmo" in command or "--alpb" in command or "--gbsa" in command for command in (optimization_command, reduced_sp_command, oxidized_sp_command)):
        raise AssertionError("solvation flag leaked into the vacuum Fadel protocol")

    executed = {"optimization": False, "reduced_sp": False, "oxidized_sp": False}
    recovery = {"used": False, "warmstart_command": [], "restart_optimization_command": [], "loose_scc_optimization_command": []}
    provenance = {
        "task": row, "status": "running", "input_geometry_sha256": sha256_file(input_xyz),
        "xtb_executable": xtb_path, "xtb_version": version, "environment": "vacuum",
        "optimization_command": optimization_command, "reduced_sp_command": reduced_sp_command,
        "oxidized_sp_command": oxidized_sp_command,
        "restraint": {
            "form": row["restraint"], "force_constant_eh_bohr2": row["restraint_force_constant_eh_bohr2"],
            "reference_distances": "auto from initial geometry" if row["restraint"] != "none" else "not applicable",
            "applies_to": "reduced_opt_only" if row["restraint"] != "none" else "not applicable",
        },
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "1"),
    }
    try:
        if force or not _state_complete(optimization_dir, atom_count, optimized=True):
            shutil.copy2(input_xyz, optimization_input)
            executed["optimization"] = True
            try:
                _run(optimization_command, optimization_dir)
            except RuntimeError:
                shutil.move(optimization_dir / "xtb.out", optimization_dir / "xtb_initial_failed.out")
                warmstart_command = [*base_reduced, "--etemp", "1000"]
                restart_optimization_command = [*optimization_command, "--restart"]
                recovery.update({
                    "used": True,
                    "warmstart_command": warmstart_command,
                    "restart_optimization_command": restart_optimization_command,
                })
                _run(warmstart_command, optimization_dir, "xtb_warmstart.out")
                try:
                    _run(restart_optimization_command, optimization_dir)
                except RuntimeError:
                    shutil.move(optimization_dir / "xtb.out", optimization_dir / "xtb_restart_failed.out")
                    loose_scc_command = [*optimization_command, "--acc", "2"]
                    recovery["loose_scc_optimization_command"] = loose_scc_command
                    _run(loose_scc_command, optimization_dir)
            if not _state_complete(optimization_dir, atom_count, optimized=True):
                raise RuntimeError("reduced optimization incomplete")
        if force or not _state_complete(reduced_sp_dir, atom_count, optimized=False) or not reduced_sp_input.exists() or sha256_file(reduced_sp_input) != sha256_file(optimized_xyz):
            shutil.copy2(optimized_xyz, reduced_sp_input)
            executed["reduced_sp"] = True
            _run(reduced_sp_command, reduced_sp_dir)
            if not _state_complete(reduced_sp_dir, atom_count, optimized=False):
                raise RuntimeError("reduced single point incomplete")
        if force or not _state_complete(oxidized_sp_dir, atom_count, optimized=False) or not oxidized_sp_input.exists() or sha256_file(oxidized_sp_input) != sha256_file(optimized_xyz):
            shutil.copy2(optimized_xyz, oxidized_sp_input)
            executed["oxidized_sp"] = True
            _run(oxidized_sp_command, oxidized_sp_dir)
            if not _state_complete(oxidized_sp_dir, atom_count, optimized=False):
                raise RuntimeError("oxidized single point incomplete")
        optimized_hash = sha256_file(optimized_xyz)
        reduced_hash = sha256_file(reduced_sp_input)
        oxidized_hash = sha256_file(oxidized_sp_input)
        provenance.update({
            "status": "complete", "optimized_geometry_sha256": optimized_hash,
            "reduced_sp_input_geometry_sha256": reduced_hash, "oxidized_sp_input_geometry_sha256": oxidized_hash,
            "same_geometry_reduced_sp": optimized_hash == reduced_hash,
            "same_geometry_oxidized_sp": optimized_hash == oxidized_hash,
        })
    except Exception as exc:
        provenance.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        provenance["optimization_recovery"] = recovery
        provenance.update({f"{name}_executed_this_invocation": value for name, value in executed.items()})
        (task_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "fadel_benchmark" / "calculation_manifest.csv")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--task-id")
    parser.add_argument("--xtb", default="xtb")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "fadel_benchmark")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_csv(args.manifest.resolve())
    matches = [row for row in rows if int(row["task_index"]) == args.task_index] if args.task_index is not None else [row for row in rows if row["task_id"] == args.task_id]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one manifest task, found {len(matches)}")
    result = run_task(matches[0], args.xtb, args.run_root.resolve(), args.force)
    print(json.dumps({"task_id": matches[0]["task_id"], "status": result["status"]}))


if __name__ == "__main__":
    main()
