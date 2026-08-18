#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_metadata, read_csv, repo_root, sha256_file, write_csv
from parse_results import _fragment_sums, _incomplete_note, _validated_task_data, geometry_metrics

FIELDS = [
    "task_id", "cation", "anion", "solvent", "initial_topology", "final_inferred_topology",
    "topology_changed", "epsilon", "energy_neutral_opt_eh", "energy_neutral_sp_eh",
    "energy_oxidized_sp_eh", "ip_vertical_ev", "q_C_neutral", "q_A_neutral", "q_S_neutral",
    "q_C_oxidized", "q_A_oxidized", "q_S_oxidized", "dq_C", "dq_A", "dq_S",
    "oxidized_fragment", "dmin_CA_ang", "dmin_CS_ang", "dmin_AS_ang", "anchor_CA_ang",
    "anchor_CS_ang", "anchor_AS_ang", "source_initial_xyz_sha256", "optimized_geometry_sha256",
    "same_geometry_pass", "status", "note",
]


def parse_row(row: dict[str, str], run_root: Path) -> dict:
    root = repo_root()
    input_xyz = root / row["input_xyz"]
    base = {
        "task_id": row["task_id"], "cation": row["cation"], "anion": row["anion"],
        "solvent": row["solvent"], "initial_topology": row["topology"], "epsilon": row["epsilon"],
        "source_initial_xyz_sha256": row["source_initial_xyz_sha256"],
    }
    try:
        if sha256_file(input_xyz) != row["source_initial_xyz_sha256"]:
            raise ValueError(f"initial XYZ hash differs from constrained source: {row['task_id']}")
        task_dir = run_root / row["task_id"]
        provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
        if row["restraint"] != "none" or provenance["restraint"]["form"] != "none":
            raise ValueError(f"unconstrained task records a restraint: {row['task_id']}")
        if "--input" in provenance["optimization_command"] or (task_dir / "reduced_opt" / "xcontrol.inp").exists():
            raise ValueError(f"unconstrained optimization used xcontrol: {row['task_id']}")
        data = _validated_task_data(row, run_root)
        metadata = load_metadata(input_xyz)
        q0 = _fragment_sums(data["charges_reduced"], metadata)
        q1 = _fragment_sums(data["charges_oxidized"], metadata)
        dq = {label: q1[label] - q0[label] for label in ("C", "A", "S")}
        geom = geometry_metrics(data["atoms"], metadata)
        final_topology = geom.pop("inferred_topology")
        base.update({
            "final_inferred_topology": final_topology,
            "topology_changed": final_topology != row["topology"],
            "energy_neutral_opt_eh": data["energy_opt_eh"],
            "energy_neutral_sp_eh": data["energy_reduced_sp_eh"],
            "energy_oxidized_sp_eh": data["energy_oxidized_sp_eh"],
            "ip_vertical_ev": data["ip_ev"],
            "q_C_neutral": q0["C"], "q_A_neutral": q0["A"], "q_S_neutral": q0["S"],
            "q_C_oxidized": q1["C"], "q_A_oxidized": q1["A"], "q_S_oxidized": q1["S"],
            "dq_C": dq["C"], "dq_A": dq["A"], "dq_S": dq["S"],
            "oxidized_fragment": max(dq, key=dq.get), **geom,
            "optimized_geometry_sha256": provenance["optimized_geometry_sha256"],
            "same_geometry_pass": provenance["same_geometry_reduced_sp"] and provenance["same_geometry_oxidized_sp"],
            "status": "complete", "note": "",
        })
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        base.update({"status": "not_run_or_incomplete", "note": _incomplete_note(exc)})
    return base


def parse_all(manifest: Path, run_root: Path, output: Path) -> list[dict]:
    rows = read_csv(manifest)
    if len(rows) != 72 or any(row["kind"] != "triad" for row in rows):
        raise ValueError("unconstrained parser requires exactly 72 triads")
    parsed = [parse_row(row, run_root) for row in rows]
    write_csv(output, parsed, FIELDS)
    return parsed


def main() -> None:
    root = repo_root()
    data = root / "data" / "chauhan_cation_eox" / "unconstrained"
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=data / "calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "chauhan_cation_eox_unconstrained")
    parser.add_argument("--output", type=Path, default=data / "unconstrained_triad_results.csv")
    args = parser.parse_args()
    rows = parse_all(args.manifest.resolve(), args.run_root.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} unconstrained triad rows")


if __name__ == "__main__":
    main()
