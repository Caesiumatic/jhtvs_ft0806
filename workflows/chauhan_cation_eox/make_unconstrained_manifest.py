#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv, repo_root, sha256_file, write_csv
from make_manifest import FIELDNAMES

EXTRA_FIELDS = ["source_restrained_task_id", "source_initial_xyz_sha256"]


def generate(source: Path, output: Path) -> list[dict]:
    root = repo_root()
    triads = [row for row in read_csv(source) if row["kind"] == "triad"]
    if len(triads) != 72:
        raise ValueError(f"expected 72 source triads, found {len(triads)}")
    rows = []
    for index, source_row in enumerate(triads, 1):
        row = dict(source_row)
        row.update({
            "task_index": index,
            "restraint": "none",
            "restraint_force_constant_eh_bohr2": "",
            "source_restrained_task_id": source_row["task_id"],
            "source_initial_xyz_sha256": sha256_file(root / source_row["input_xyz"]),
        })
        rows.append(row)
    if any(row["kind"] != "triad" or row["restraint"] != "none" for row in rows):
        raise AssertionError("unconstrained manifest contains a non-triad or restraint")
    write_csv(output, rows, FIELDNAMES + EXTRA_FIELDS)
    return rows


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root / "data" / "chauhan_cation_eox" / "calculation_manifest.csv")
    parser.add_argument("--output", type=Path, default=root / "data" / "chauhan_cation_eox" / "unconstrained" / "calculation_manifest.csv")
    args = parser.parse_args()
    rows = generate(args.source.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} unconstrained triads ({3 * len(rows)} xTB invocations)")


if __name__ == "__main__":
    main()
