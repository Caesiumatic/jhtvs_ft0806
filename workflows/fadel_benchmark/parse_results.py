#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .common import Atom, HARTREE_TO_EV, load_metadata, read_csv, read_xyz, repo_root, sha256_file, write_csv
    from .run_calculation import parse_charges, parse_energy
except ImportError:
    from common import Atom, HARTREE_TO_EV, load_metadata, read_csv, read_xyz, repo_root, sha256_file, write_csv
    from run_calculation import parse_charges, parse_energy

FIELDS = [
    "task_id", "kind", "cation", "anion", "solvent", "initial_topology", "final_inferred_topology",
    "topology_preserved", "environment", "charge_reduced", "uhf_reduced", "charge_oxidized", "uhf_oxidized",
    "energy_reduced_opt_eh", "energy_reduced_sp_eh", "energy_oxidized_sp_eh", "ip_vertical_ev",
    "q_C_reduced", "q_A_reduced", "q_S_reduced", "q_C_oxidized", "q_A_oxidized", "q_S_oxidized",
    "dq_C", "dq_A", "dq_S", "oxidized_fragment", "optimized_geometry_sha256", "same_geometry_pass",
    "status", "note",
]


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _terminated(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "normal termination of xtb" in text or "finished run on" in text


def _fragment_sums(charges: list[float], metadata: dict) -> dict[str, float]:
    return {
        label: sum(charges[index] for index in fragment["atom_indices_zero_based"])
        for label, fragment in metadata["fragments"].items()
    }


def _centroid(atoms: list[Atom], indices: list[int]) -> tuple[float, float, float]:
    selected = [atoms[index] for index in indices]
    return tuple(sum(getattr(atom, axis) for atom in selected) / len(selected) for axis in ("x", "y", "z"))


def _inferred_topology(atoms: list[Atom], metadata: dict) -> str:
    if set(metadata["fragments"]) != {"C", "A", "S"}:
        return "AS"
    centroids = {label: _centroid(atoms, fragment["atom_indices_zero_based"]) for label, fragment in metadata["fragments"].items()}
    sums = {label: sum(math.dist(centroids[label], centroids[other]) for other in centroids if other != label) for label in centroids}
    return {"A": "CAS", "S": "CSA", "C": "ACS"}[min(sums, key=sums.get)]


def _validate_protocol(row: dict[str, str], provenance: dict, metadata: dict, task_dir: Path) -> None:
    if row["environment"] != "vacuum" or provenance.get("environment") != "vacuum":
        raise ValueError("non-vacuum Fadel task")
    if row["kind"] == "as_pair":
        expected_states = (("-1", "0"), ("-1", "0"), ("0", "1"))
        if row["restraint"] != "none" or set(metadata["fragments"]) != {"A", "S"}:
            raise ValueError("invalid A-S fragments or restraint")
    elif row["kind"] == "triad":
        expected_states = (("0", "0"), ("0", "0"), ("1", "1"))
        if row["cation"] != "Li" or row["restraint"] != "two_adjacent_anchor_distances" or float(row["restraint_force_constant_eh_bohr2"]) != 0.005:
            raise ValueError("invalid Li-A-S triad protocol")
        if set(metadata["fragments"]) != {"C", "A", "S"} or metadata["formal_charge"] != 0:
            raise ValueError("invalid Li-A-S fragments or formal charge")
    else:
        raise ValueError("unexpected task kind")
    commands = (provenance["optimization_command"], provenance["reduced_sp_command"], provenance["oxidized_sp_command"])
    for command, expected in zip(commands, expected_states):
        if any(flag in command for flag in ("--cosmo", "--alpb", "--gbsa")):
            raise ValueError("solvation flag found in primary vacuum command")
        if _flag(command, "--gfn") != "2" or (_flag(command, "--chrg"), _flag(command, "--uhf")) != expected:
            raise ValueError("incorrect GFN2 charge/UHF command")
    if "--opt" not in commands[0] or any("--opt" in command for command in commands[1:]):
        raise ValueError("incorrect optimization/SP boundary")
    if row["kind"] == "as_pair" and any("--input" in command for command in commands):
        raise ValueError("A-S unexpectedly restrained")
    if row["kind"] == "triad" and ("--input" not in commands[0] or any("--input" in command for command in commands[1:])):
        raise ValueError("triad restraint is not optimization-only")
    if provenance.get("status") != "complete" or not all(_terminated(task_dir / state / "xtb.out") for state in ("reduced_opt", "reduced_sp", "oxidized_sp")):
        raise ValueError("xTB task did not terminate successfully")


def parse_row(row: dict[str, str], run_root: Path) -> dict:
    base = {
        "task_id": row["task_id"], "kind": row["kind"], "cation": row["cation"], "anion": row["anion"],
        "solvent": row["solvent"], "initial_topology": row["topology"], "environment": row["environment"],
        "charge_reduced": row["charge_reduced"], "uhf_reduced": row["uhf_reduced"],
        "charge_oxidized": row["charge_oxidized"], "uhf_oxidized": row["uhf_oxidized"],
    }
    try:
        task_dir = run_root / row["task_id"]
        input_xyz = repo_root() / row["input_xyz"]
        metadata = load_metadata(input_xyz)
        provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
        _validate_protocol(row, provenance, metadata, task_dir)
        optimized_xyz = task_dir / "reduced_opt" / "xtbopt.xyz"
        sp_inputs = (task_dir / "reduced_sp" / "in.xyz", task_dir / "oxidized_sp" / "in.xyz")
        hashes = [sha256_file(path) for path in (optimized_xyz, *sp_inputs)]
        if len(set(hashes)) != 1 or not provenance.get("same_geometry_reduced_sp") or not provenance.get("same_geometry_oxidized_sp"):
            raise ValueError("reduced/oxidized SP geometry hashes differ")
        atoms, _ = read_xyz(optimized_xyz)
        energy_opt = parse_energy((task_dir / "reduced_opt" / "xtb.out").read_text(encoding="utf-8"))
        energy_reduced = parse_energy((task_dir / "reduced_sp" / "xtb.out").read_text(encoding="utf-8"))
        energy_oxidized = parse_energy((task_dir / "oxidized_sp" / "xtb.out").read_text(encoding="utf-8"))
        q0 = _fragment_sums(parse_charges(task_dir / "reduced_sp" / "charges", len(atoms)), metadata)
        q1 = _fragment_sums(parse_charges(task_dir / "oxidized_sp" / "charges", len(atoms)), metadata)
        fragments = ("A", "S") if row["kind"] == "as_pair" else ("C", "A", "S")
        dq = {fragment: q1[fragment] - q0[fragment] for fragment in fragments}
        ip = (energy_oxidized - energy_reduced) * HARTREE_TO_EV
        inferred = _inferred_topology(atoms, metadata)
        base.update({
            "final_inferred_topology": inferred, "topology_preserved": inferred == row["topology"],
            "energy_reduced_opt_eh": energy_opt, "energy_reduced_sp_eh": energy_reduced,
            "energy_oxidized_sp_eh": energy_oxidized, "ip_vertical_ev": ip,
            **{f"q_{fragment}_reduced": q0.get(fragment, "") for fragment in ("C", "A", "S")},
            **{f"q_{fragment}_oxidized": q1.get(fragment, "") for fragment in ("C", "A", "S")},
            **{f"dq_{fragment}": dq.get(fragment, "") for fragment in ("C", "A", "S")},
            "oxidized_fragment": max(dq, key=dq.get), "optimized_geometry_sha256": hashes[0],
            "same_geometry_pass": True, "status": "complete", "note": "",
        })
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": str(exc)})
    return base


def parse_all(manifest: Path, run_root: Path, output: Path) -> list[dict]:
    tasks = read_csv(manifest)
    if len(tasks) != 64:
        raise ValueError("Fadel parser requires exactly 64 tasks")
    rows = [parse_row(row, run_root) for row in tasks]
    write_csv(output, rows, FIELDS)
    return rows


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "fadel_benchmark" / "calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "fadel_benchmark")
    parser.add_argument("--output", type=Path, default=root / "data" / "fadel_benchmark" / "fadel_task_results.csv")
    args = parser.parse_args()
    rows = parse_all(args.manifest.resolve(), args.run_root.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} Fadel task rows; complete={sum(row['status'] == 'complete' for row in rows)}")


if __name__ == "__main__":
    main()
