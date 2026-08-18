#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import read_csv, repo_root, write_csv
except ImportError:
    from common import read_csv, repo_root, write_csv

RESTRAINT_FORCE = 0.005
FIELDS = [
    "task_index", "task_id", "kind", "cation", "anion", "solvent", "topology",
    "environment", "input_xyz", "charge_reduced", "uhf_reduced", "charge_oxidized",
    "uhf_oxidized", "optimize_reduced", "restraint", "restraint_force_constant_eh_bohr2",
]


def generate(structure_manifest: Path, output: Path) -> list[dict]:
    structures = read_csv(structure_manifest)
    rows = []
    for index, structure in enumerate(structures, start=1):
        if structure["kind"] == "as_pair":
            reduced, oxidized = (-1, 0), (0, 1)
            restraint, force = "none", ""
        elif structure["kind"] == "triad":
            reduced, oxidized = (0, 0), (1, 1)
            restraint, force = "two_adjacent_anchor_distances", RESTRAINT_FORCE
        else:
            raise ValueError(f"unsupported Fadel task kind: {structure['kind']}")
        rows.append({
            "task_index": index, "task_id": structure["structure_id"], "kind": structure["kind"],
            "cation": structure["cation"], "anion": structure["anion"], "solvent": structure["solvent"],
            "topology": structure["topology"], "environment": "vacuum", "input_xyz": structure["xyz_path"],
            "charge_reduced": reduced[0], "uhf_reduced": reduced[1], "charge_oxidized": oxidized[0],
            "uhf_oxidized": oxidized[1], "optimize_reduced": 1, "restraint": restraint,
            "restraint_force_constant_eh_bohr2": force,
        })
    if len([row for row in rows if row["kind"] == "as_pair"]) != 16 or len([row for row in rows if row["kind"] == "triad"]) != 48:
        raise AssertionError("Fadel manifest must contain exactly 16 A-S and 48 Li-A-S tasks")
    write_csv(output, rows, FIELDS)
    return rows


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-manifest", type=Path, default=root / "data" / "fadel_benchmark" / "structure_manifest.csv")
    parser.add_argument("--output", type=Path, default=root / "data" / "fadel_benchmark" / "calculation_manifest.csv")
    args = parser.parse_args()
    rows = generate(args.structure_manifest.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} Fadel vacuum tasks ({3 * len(rows)} xTB invocations)")


if __name__ == "__main__":
    main()
