#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import Atom, distance, ip_ev, load_metadata, read_csv, read_xyz, repo_root, sha256_file, write_csv
from run_calculation import parse_charges, parse_energy

TRIAD_FIELDS = [
    "cation", "anion", "solvent", "topology", "epsilon", "charge_neutral", "uhf_neutral",
    "charge_oxidized", "uhf_oxidized", "energy_neutral_opt_eh", "energy_neutral_sp_eh",
    "energy_oxidized_sp_eh", "ip_vertical_ev", "ip_cation_ev", "delta_ip_vs_isolated_cation_ev",
    "q_C_neutral", "q_A_neutral", "q_S_neutral", "q_C_oxidized", "q_A_oxidized",
    "q_S_oxidized", "dq_C", "dq_A", "dq_S", "oxidized_fragment", "topology_preserved",
    "inferred_topology", "dmin_CA_ang", "dmin_CS_ang", "dmin_AS_ang", "anchor_CA_ang",
    "anchor_CS_ang", "anchor_AS_ang", "status", "note",
]
AS_FIELDS = [
    "anion", "solvent", "epsilon", "energy_as_reduced_opt_eh", "energy_as_reduced_sp_eh",
    "energy_as_oxidized_sp_eh", "ip_as_direct_ev", "ip_solvent_ev", "ip_anion_ev",
    "ip_fadel_2p8_ev", "status", "note",
]
CATION_FIELDS = [
    "cation", "solvent", "epsilon", "energy_reduced_opt_eh", "energy_reduced_sp_eh",
    "energy_oxidized_sp_eh", "ip_cation_ev", "status", "note",
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


def _validated_task_data(row: dict[str, str], run_root: Path) -> dict:
    task_dir = run_root / row["task_id"]
    provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("status") != "complete":
        raise ValueError(f"task is not complete: {row['task_id']}")
    optimized_xyz = task_dir / "reduced_opt" / "xtbopt.xyz"
    reduced_sp_input = task_dir / "reduced_sp" / "in.xyz"
    oxidized_sp_input = task_dir / "oxidized_sp" / "in.xyz"
    hashes = {
        "optimized": sha256_file(optimized_xyz),
        "reduced_sp": sha256_file(reduced_sp_input),
        "oxidized_sp": sha256_file(oxidized_sp_input),
    }
    if not provenance.get("same_geometry_reduced_sp") or hashes["optimized"] != hashes["reduced_sp"]:
        raise ValueError(f"reduced SP geometry hash differs from optimized geometry: {row['task_id']}")
    if not provenance.get("same_geometry_oxidized_sp") or hashes["optimized"] != hashes["oxidized_sp"]:
        raise ValueError(f"oxidized SP geometry hash differs from optimized geometry: {row['task_id']}")
    for key, actual in (
        ("optimized_geometry_sha256", hashes["optimized"]),
        ("reduced_sp_input_geometry_sha256", hashes["reduced_sp"]),
        ("oxidized_sp_input_geometry_sha256", hashes["oxidized_sp"]),
    ):
        if provenance.get(key) != actual:
            raise ValueError(f"provenance geometry hash mismatch for {key}: {row['task_id']}")
    atoms, _ = read_xyz(optimized_xyz)
    energy_opt = parse_energy((task_dir / "reduced_opt" / "xtb.out").read_text(encoding="utf-8"))
    energy_reduced_sp, charges_reduced = _state_data(task_dir, "reduced_sp", len(atoms))
    energy_oxidized_sp, charges_oxidized = _state_data(task_dir, "oxidized_sp", len(atoms))
    return {
        "atoms": atoms,
        "energy_opt_eh": energy_opt,
        "energy_reduced_sp_eh": energy_reduced_sp,
        "energy_oxidized_sp_eh": energy_oxidized_sp,
        "charges_reduced": charges_reduced,
        "charges_oxidized": charges_oxidized,
        "ip_ev": ip_ev(energy_oxidized_sp, energy_reduced_sp),
    }


def _ip_for_task(row: dict[str, str], run_root: Path) -> float:
    return _validated_task_data(row, run_root)["ip_ev"]


def parse_cation_reference(row: dict[str, str], run_root: Path) -> dict:
    base = {"cation": row["cation"], "solvent": row["solvent"], "epsilon": row["epsilon"]}
    try:
        data = _validated_task_data(row, run_root)
        base.update({
            "energy_reduced_opt_eh": data["energy_opt_eh"],
            "energy_reduced_sp_eh": data["energy_reduced_sp_eh"],
            "energy_oxidized_sp_eh": data["energy_oxidized_sp_eh"],
            "ip_cation_ev": data["ip_ev"],
            "status": "complete",
            "note": "",
        })
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": _incomplete_note(exc)})
    return base


def parse_triad(row: dict[str, str], tasks: list[dict[str, str]], run_root: Path) -> dict:
    input_xyz = repo_root() / row["input_xyz"]
    metadata = load_metadata(input_xyz)
    base = {
        "cation": row["cation"], "anion": row["anion"], "solvent": row["solvent"],
        "topology": row["topology"], "epsilon": row["epsilon"],
        "charge_neutral": row["charge_reduced"], "uhf_neutral": row["uhf_reduced"],
        "charge_oxidized": row["charge_oxidized"], "uhf_oxidized": row["uhf_oxidized"],
    }
    try:
        data = _validated_task_data(row, run_root)
        cation_row = next(
            task for task in tasks
            if task["kind"] == "cation" and task["cation"] == row["cation"] and task["solvent"] == row["solvent"]
        )
        cation_ip = _ip_for_task(cation_row, run_root)
        q0 = _fragment_sums(data["charges_reduced"], metadata)
        q1 = _fragment_sums(data["charges_oxidized"], metadata)
        dq = {label: q1[label] - q0[label] for label in ("C", "A", "S")}
        geom = geometry_metrics(data["atoms"], metadata)
        base.update({
            "energy_neutral_opt_eh": data["energy_opt_eh"],
            "energy_neutral_sp_eh": data["energy_reduced_sp_eh"],
            "energy_oxidized_sp_eh": data["energy_oxidized_sp_eh"],
            "ip_vertical_ev": data["ip_ev"],
            "ip_cation_ev": cation_ip,
            "delta_ip_vs_isolated_cation_ev": data["ip_ev"] - cation_ip,
            "q_C_neutral": q0["C"], "q_A_neutral": q0["A"], "q_S_neutral": q0["S"],
            "q_C_oxidized": q1["C"], "q_A_oxidized": q1["A"], "q_S_oxidized": q1["S"],
            "dq_C": dq["C"], "dq_A": dq["A"], "dq_S": dq["S"],
            "oxidized_fragment": max(dq, key=dq.get), **geom,
            "topology_preserved": geom["inferred_topology"] == row["topology"],
            "status": "complete", "note": "",
        })
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": _incomplete_note(exc)})
    return base


def parse_as_reference(row: dict[str, str], tasks: list[dict[str, str]], run_root: Path) -> dict:
    base = {"anion": row["anion"], "solvent": row["solvent"], "epsilon": row["epsilon"]}
    try:
        solvent_row = next(r for r in tasks if r["kind"] == "solvent" and r["solvent"] == row["solvent"])
        anion_row = next(r for r in tasks if r["kind"] == "anion" and r["anion"] == row["anion"] and r["solvent"] == row["solvent"])
        data = _validated_task_data(row, run_root)
        solvent_ip = _ip_for_task(solvent_row, run_root)
        anion_ip = _ip_for_task(anion_row, run_root)
        base.update({
            "energy_as_reduced_opt_eh": data["energy_opt_eh"],
            "energy_as_reduced_sp_eh": data["energy_reduced_sp_eh"],
            "energy_as_oxidized_sp_eh": data["energy_oxidized_sp_eh"],
            "ip_as_direct_ev": data["ip_ev"], "ip_solvent_ev": solvent_ip, "ip_anion_ev": anion_ip,
            "ip_fadel_2p8_ev": min(anion_ip, solvent_ip - 2.8), "status": "complete", "note": "",
        })
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": _incomplete_note(exc)})
    return base


def parse_all(manifest: Path, run_root: Path, output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    tasks = read_csv(manifest)
    cations = [parse_cation_reference(row, run_root) for row in tasks if row["kind"] == "cation"]
    triads = [parse_triad(row, tasks, run_root) for row in tasks if row["kind"] == "triad"]
    references = [parse_as_reference(row, tasks, run_root) for row in tasks if row["kind"] == "as_pair"]
    write_csv(output_dir / "triad_results.csv", triads, TRIAD_FIELDS)
    write_csv(output_dir / "as_reference_results.csv", references, AS_FIELDS)
    write_csv(output_dir / "cation_reference_results.csv", cations, CATION_FIELDS)
    return triads, references, cations


def main() -> None:
    parser = argparse.ArgumentParser()
    root = repo_root()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "chauhan_cation_eox" / "calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "chauhan_cation_eox")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "chauhan_cation_eox")
    args = parser.parse_args()
    triads, references, cations = parse_all(args.manifest.resolve(), args.run_root.resolve(), args.output_dir.resolve())
    print(f"wrote {len(triads)} triad rows, {len(references)} A-S reference rows, and {len(cations)} cation reference rows")


if __name__ == "__main__":
    main()
