#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import Atom, distance, ip_ev, load_metadata, read_csv, read_xyz, repo_root, write_csv
from run_calculation import parse_charges, parse_energy

TRIAD_FIELDS = [
    "cation", "anion", "solvent", "topology", "epsilon", "charge_neutral", "uhf_neutral",
    "charge_oxidized", "uhf_oxidized", "energy_neutral_eh", "energy_oxidized_eh",
    "ip_vertical_ev", "q_C_neutral", "q_A_neutral", "q_S_neutral", "q_C_oxidized",
    "q_A_oxidized", "q_S_oxidized", "dq_C", "dq_A", "dq_S", "oxidized_fragment",
    "topology_preserved", "inferred_topology", "dmin_CA_ang", "dmin_CS_ang", "dmin_AS_ang",
    "anchor_CA_ang", "anchor_CS_ang", "anchor_AS_ang", "status", "note",
]
AS_FIELDS = [
    "anion", "solvent", "epsilon", "energy_as_reduced_eh", "energy_as_oxidized_eh",
    "ip_as_direct_ev", "ip_solvent_ev", "ip_anion_ev", "ip_fadel_2p8_ev", "status", "note",
]


def _centroid(atoms: list[Atom], indices: list[int]) -> tuple[float, float, float]:
    selected = [atoms[i] for i in indices]
    return tuple(sum(getattr(atom, axis) for atom in selected) / len(selected) for axis in ("x", "y", "z"))


def _dmin(atoms: list[Atom], left: list[int], right: list[int]) -> float:
    return min(distance(atoms[i], atoms[j]) for i in left for j in right if atoms[i].element != "H" and atoms[j].element != "H")


def geometry_metrics(atoms: list[Atom], metadata: dict) -> dict:
    indices = {label: data["atom_indices_zero_based"] for label, data in metadata["fragments"].items()}
    centroids = {label: _centroid(atoms, ids) for label, ids in indices.items()}
    sums = {
        label: sum(math.dist(centroids[label], centroids[other]) for other in centroids if other != label)
        for label in centroids
    }
    middle = min(sums, key=sums.get)
    inferred = {"A": "CAS", "S": "CSA", "C": "ACS"}[middle]
    anchors = metadata["anchor_indices_zero_based"]
    return {
        "inferred_topology": inferred,
        "dmin_CA_ang": _dmin(atoms, indices["C"], indices["A"]),
        "dmin_CS_ang": _dmin(atoms, indices["C"], indices["S"]),
        "dmin_AS_ang": _dmin(atoms, indices["A"], indices["S"]),
        "anchor_CA_ang": distance(atoms[anchors["C"]], atoms[anchors["A"]]),
        "anchor_CS_ang": distance(atoms[anchors["C"]], atoms[anchors["S"]]),
        "anchor_AS_ang": distance(atoms[anchors["A"]], atoms[anchors["S"]]),
    }


def _state_data(task_dir: Path, state: str, atom_count: int) -> tuple[float, list[float]]:
    state_dir = task_dir / state
    return (
        parse_energy((state_dir / "xtb.out").read_text(encoding="utf-8")),
        parse_charges(state_dir / "charges", atom_count),
    )


def _fragment_sums(charges: list[float], metadata: dict) -> dict[str, float]:
    return {
        label: sum(charges[index] for index in fragment["atom_indices_zero_based"])
        for label, fragment in metadata["fragments"].items()
    }


def _incomplete_note(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "calculation artifacts unavailable"
    return str(exc)


def parse_triad(row: dict[str, str], run_root: Path) -> dict:
    root = repo_root()
    input_xyz = root / row["input_xyz"]
    metadata = load_metadata(input_xyz)
    base = {
        "cation": row["cation"], "anion": row["anion"], "solvent": row["solvent"],
        "topology": row["topology"], "epsilon": row["epsilon"],
        "charge_neutral": row["charge_reduced"], "uhf_neutral": row["uhf_reduced"],
        "charge_oxidized": row["charge_oxidized"], "uhf_oxidized": row["uhf_oxidized"],
    }
    try:
        provenance = json.loads((run_root / row["task_id"] / "provenance.json").read_text(encoding="utf-8"))
        if not provenance.get("same_geometry_vertical_sp"):
            raise ValueError("oxidized SP geometry hash differs from optimized neutral geometry")
        optimized_xyz = run_root / row["task_id"] / "reduced_opt" / "xtbopt.xyz"
        atoms, _ = read_xyz(optimized_xyz)
        e0, q0_atoms = _state_data(run_root / row["task_id"], "reduced_opt", len(atoms))
        e1, q1_atoms = _state_data(run_root / row["task_id"], "oxidized_sp", len(atoms))
        q0 = _fragment_sums(q0_atoms, metadata)
        q1 = _fragment_sums(q1_atoms, metadata)
        dq = {label: q1[label] - q0[label] for label in ("C", "A", "S")}
        geom = geometry_metrics(atoms, metadata)
        base.update({
            "energy_neutral_eh": e0, "energy_oxidized_eh": e1, "ip_vertical_ev": ip_ev(e1, e0),
            "q_C_neutral": q0["C"], "q_A_neutral": q0["A"], "q_S_neutral": q0["S"],
            "q_C_oxidized": q1["C"], "q_A_oxidized": q1["A"], "q_S_oxidized": q1["S"],
            "dq_C": dq["C"], "dq_A": dq["A"], "dq_S": dq["S"],
            "oxidized_fragment": max(dq, key=dq.get), **geom,
            "topology_preserved": geom["inferred_topology"] == row["topology"],
            "status": "complete", "note": "",
        })
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": _incomplete_note(exc)})
    return base


def _ip_for_task(row: dict[str, str], run_root: Path) -> float:
    task_dir = run_root / row["task_id"]
    atoms, _ = read_xyz(task_dir / "reduced_opt" / "xtbopt.xyz")
    e0, _ = _state_data(task_dir, "reduced_opt", len(atoms))
    e1, _ = _state_data(task_dir, "oxidized_sp", len(atoms))
    provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
    if not provenance.get("same_geometry_vertical_sp"):
        raise ValueError(f"vertical geometry mismatch for {row['task_id']}")
    return ip_ev(e1, e0)


def parse_as_reference(row: dict[str, str], tasks: list[dict[str, str]], run_root: Path) -> dict:
    base = {"anion": row["anion"], "solvent": row["solvent"], "epsilon": row["epsilon"]}
    try:
        solvent_row = next(r for r in tasks if r["kind"] == "solvent" and r["solvent"] == row["solvent"])
        anion_row = next(r for r in tasks if r["kind"] == "anion" and r["anion"] == row["anion"] and r["solvent"] == row["solvent"])
        task_dir = run_root / row["task_id"]
        atoms, _ = read_xyz(task_dir / "reduced_opt" / "xtbopt.xyz")
        e0, _ = _state_data(task_dir, "reduced_opt", len(atoms))
        e1, _ = _state_data(task_dir, "oxidized_sp", len(atoms))
        direct = _ip_for_task(row, run_root)
        solvent_ip = _ip_for_task(solvent_row, run_root)
        anion_ip = _ip_for_task(anion_row, run_root)
        base.update({
            "energy_as_reduced_eh": e0, "energy_as_oxidized_eh": e1,
            "ip_as_direct_ev": direct, "ip_solvent_ev": solvent_ip, "ip_anion_ev": anion_ip,
            "ip_fadel_2p8_ev": min(anion_ip, solvent_ip - 2.8), "status": "complete", "note": "",
        })
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": _incomplete_note(exc)})
    return base


def parse_all(manifest: Path, run_root: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    tasks = read_csv(manifest)
    triads = [parse_triad(row, run_root) for row in tasks if row["kind"] == "triad"]
    references = [parse_as_reference(row, tasks, run_root) for row in tasks if row["kind"] == "as_pair"]
    write_csv(output_dir / "triad_results.csv", triads, TRIAD_FIELDS)
    write_csv(output_dir / "as_reference_results.csv", references, AS_FIELDS)
    return triads, references


def main() -> None:
    parser = argparse.ArgumentParser()
    root = repo_root()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "chauhan_cation_eox" / "calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "chauhan_cation_eox")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "chauhan_cation_eox")
    args = parser.parse_args()
    triads, references = parse_all(args.manifest.resolve(), args.run_root.resolve(), args.output_dir.resolve())
    print(f"wrote {len(triads)} triad rows and {len(references)} A-S reference rows")


if __name__ == "__main__":
    main()
