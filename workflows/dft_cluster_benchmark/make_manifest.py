#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from .common import BASIS, INTEGRAL_APPROXIMATION, METHOD, METHOD_ID, POPULATION_SCHEME, SOFTWARE, read_csv, repo_root, write_csv
except ImportError:
    from common import BASIS, INTEGRAL_APPROXIMATION, METHOD, METHOD_ID, POPULATION_SCHEME, SOFTWARE, read_csv, repo_root, write_csv

FIELDS = [
    "task_index", "task_id", "benchmark", "source_task_id", "kind", "cation", "anion", "solvent", "topology",
    "environment", "solvation_model", "epsilon", "input_xyz", "charge_reduced", "multiplicity_reduced",
    "charge_oxidized", "multiplicity_oxidized", "restraint", "restraint_force_constant_eh_bohr2",
    "method", "basis", "software", "integral_approximation", "population_scheme", "method_id",
]


def _base(row: dict[str, str], benchmark: str, environment: str, solvation_model: str, epsilon: str) -> dict:
    return {
        "task_id": f"{benchmark}__{row['task_id']}",
        "benchmark": benchmark,
        "source_task_id": row["task_id"],
        "kind": row["kind"],
        "cation": row["cation"],
        "anion": row["anion"],
        "solvent": row["solvent"],
        "topology": row["topology"],
        "environment": environment,
        "solvation_model": solvation_model,
        "epsilon": epsilon,
        "input_xyz": row["input_xyz"],
        "charge_reduced": row["charge_reduced"],
        "multiplicity_reduced": int(row["uhf_reduced"]) + 1,
        "charge_oxidized": row["charge_oxidized"],
        "multiplicity_oxidized": int(row["uhf_oxidized"]) + 1,
        "restraint": row["restraint"],
        "restraint_force_constant_eh_bohr2": row["restraint_force_constant_eh_bohr2"],
        "method": METHOD,
        "basis": BASIS,
        "software": SOFTWARE,
        "integral_approximation": INTEGRAL_APPROXIMATION,
        "population_scheme": POPULATION_SCHEME,
        "method_id": METHOD_ID,
    }


def generate(chauhan_manifest: Path, fadel_manifest: Path, output: Path) -> list[dict]:
    rows = []
    for source in read_csv(chauhan_manifest):
        if source["kind"] not in {"as_pair", "triad"}:
            continue
        rows.append(_base(source, "chauhan", "implicit", "ORCA_CPCM_epsilon_vdw_gaussian", source["epsilon"]))
    for source in read_csv(fadel_manifest):
        rows.append(_base(source, "fadel", "vacuum", "none", ""))
    for index, row in enumerate(rows, 1):
        row["task_index"] = index

    counts = Counter((row["benchmark"], row["kind"]) for row in rows)
    expected = {("chauhan", "as_pair"): 9, ("chauhan", "triad"): 72, ("fadel", "as_pair"): 16, ("fadel", "triad"): 48}
    if counts != expected or len(rows) != 145:
        raise ValueError(f"unexpected DFT task counts: {counts}")
    if any(row["epsilon"] not in {"65.0", "37.0", "7.6"} for row in rows if row["benchmark"] == "chauhan"):
        raise ValueError("unexpected Chauhan dielectric")
    if any(row["epsilon"] or row["solvation_model"] != "none" for row in rows if row["benchmark"] == "fadel"):
        raise ValueError("Fadel DFT tasks must be vacuum")
    write_csv(output, rows, FIELDS)
    return rows


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--chauhan-manifest", type=Path, default=root / "data/chauhan_cation_eox/calculation_manifest.csv")
    parser.add_argument("--fadel-manifest", type=Path, default=root / "data/fadel_benchmark/calculation_manifest.csv")
    parser.add_argument("--output", type=Path, default=root / "data/dft_cluster_benchmark/calculation_manifest.csv")
    args = parser.parse_args()
    rows = generate(args.chauhan_manifest.resolve(), args.fadel_manifest.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} DFT cases and {3 * len(rows)} electronic-structure invocations")


if __name__ == "__main__":
    main()
