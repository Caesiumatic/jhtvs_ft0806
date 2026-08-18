#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import repo_root, species_table, write_csv

FIELDNAMES = [
    "task_index", "task_id", "kind", "cation", "anion", "solvent", "topology",
    "epsilon", "input_xyz", "charge_reduced", "uhf_reduced", "charge_oxidized",
    "uhf_oxidized", "optimize_reduced", "restraint", "restraint_force_constant_eh_bohr2",
]
RESTRAINT_FORCE = 0.005


def generate(structure_manifest: Path, output: Path) -> list[dict]:
    import csv

    with structure_manifest.open(newline="", encoding="utf-8") as handle:
        structures = list(csv.DictReader(handle))
    species = species_table()
    rows: list[dict] = []
    for index, structure in enumerate(structures, start=1):
        kind = structure["kind"]
        if kind == "triad":
            reduced = (0, 0)
            oxidized = (1, 1)
            restraint = "two_adjacent_anchor_distances"
            force = RESTRAINT_FORCE
        elif kind in {"as_pair", "anion"}:
            reduced = (-1, 0)
            oxidized = (0, 1)
            restraint = "none"
            force = ""
        elif kind == "solvent":
            reduced = (0, 0)
            oxidized = (1, 1)
            restraint = "none"
            force = ""
        elif kind == "cation":
            reduced = (1, 0)
            oxidized = (2, 1)
            restraint = "none"
            force = ""
        else:
            raise ValueError(f"unknown structure kind: {kind}")
        task_id = structure["structure_id"]
        rows.append({
            "task_index": index,
            "task_id": task_id,
            "kind": kind,
            "cation": structure["cation"],
            "anion": structure["anion"],
            "solvent": structure["solvent"],
            "topology": structure["topology"],
            "epsilon": species[structure["solvent"]]["epsilon"],
            "input_xyz": structure["xyz_path"],
            "charge_reduced": reduced[0],
            "uhf_reduced": reduced[1],
            "charge_oxidized": oxidized[0],
            "uhf_oxidized": oxidized[1],
            "optimize_reduced": 1,
            "restraint": restraint,
            "restraint_force_constant_eh_bohr2": force,
        })
    write_csv(output, rows, FIELDNAMES)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    root = repo_root()
    parser.add_argument("--structure-manifest", type=Path, default=root / "data" / "chauhan_cation_eox" / "structure_manifest.csv")
    parser.add_argument("--output", type=Path, default=root / "data" / "chauhan_cation_eox" / "calculation_manifest.csv")
    args = parser.parse_args()
    rows = generate(args.structure_manifest.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} optimized geometries ({3 * len(rows)} xTB invocations) to {args.output}")


if __name__ == "__main__":
    main()
