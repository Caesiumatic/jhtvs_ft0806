#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .common import HARTREE_TO_EV, load_metadata, read_csv, repo_root, sha256_file, write_csv
except ImportError:
    from common import HARTREE_TO_EV, load_metadata, read_csv, repo_root, sha256_file, write_csv

ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
CHARGE_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s*:\s*(-?\d+(?:\.\d+)?)")
S2_RE = re.compile(r"Expectation value of\s*<S\*\*2>\s*:\s*(-?\d+(?:\.\d+)?)", re.I)
VERSION_RE = re.compile(r"Program Version\s+([0-9]+(?:\.[0-9]+){1,2})", re.I)
FIELDS = [
    "task_index", "task_id", "benchmark", "kind", "cation", "anion", "solvent", "topology", "environment",
    "solvation_model", "epsilon", "method", "basis", "software", "software_version", "functional_implementation",
    "libxc_exchange", "libxc_correlation", "integral_approximation", "population_scheme", "charge_reduced",
    "multiplicity_reduced", "charge_oxidized", "multiplicity_oxidized",
    "energy_reduced_sp_eh", "energy_oxidized_sp_eh", "ip_vertical_ev", "q_C_reduced", "q_A_reduced", "q_S_reduced",
    "q_C_oxidized", "q_A_oxidized", "q_S_oxidized", "dq_C", "dq_A", "dq_S", "oxidized_fragment",
    "s2_reduced", "s2_oxidized", "scf_reduced_converged", "scf_oxidized_converged", "optimization_converged",
    "protocol_decks_verified",
    "optimized_geometry_sha256", "reduced_sp_input_geometry_sha256", "oxidized_sp_input_geometry_sha256",
    "same_sp_geometry", "status", "note",
]


def verify_protocol_decks(row: dict[str, str], task_dir: Path) -> None:
    states = ("reduced_opt", "reduced_sp", "oxidized_sp")
    decks = {state: (task_dir / state / "orca.inp").read_text(encoding="utf-8") for state in states}
    for state, text in decks.items():
        for required in (
            "! aug-cc-pVTZ ",
            f"Exchange {row['libxc_exchange']}",
            f"Correlation {row['libxc_correlation']}",
        ):
            if required not in text:
                raise ValueError(f"{state} input is missing required protocol token: {required.strip()}")
        reduced = state != "oxidized_sp"
        charge = row["charge_reduced"] if reduced else row["charge_oxidized"]
        multiplicity = row["multiplicity_reduced"] if reduced else row["multiplicity_oxidized"]
        if f"* xyzfile {charge} {multiplicity} in.xyz" not in text:
            raise ValueError(f"{state} input charge/multiplicity mismatch")
        if row["benchmark"] == "chauhan":
            if "%cpcm" not in text or f"epsilon {float(row['epsilon']):.8f}" not in text:
                raise ValueError(f"{state} input solvent mismatch")
        elif "%cpcm" in text:
            raise ValueError(f"{state} Fadel input is not vacuum")
    if " Opt" not in decks["reduced_opt"] or " Opt" in decks["reduced_sp"] or " Opt" in decks["oxidized_sp"]:
        raise ValueError("optimization scope mismatch")
    has_opt_bias = "BIAS" in decks["reduced_opt"]
    if has_opt_bias != (row["kind"] == "triad") or "BIAS" in decks["reduced_sp"] or "BIAS" in decks["oxidized_sp"]:
        raise ValueError("topology-bias scope mismatch")


def parse_energy(text: str) -> float:
    values = ENERGY_RE.findall(text)
    if not values:
        raise ValueError("FINAL SINGLE POINT ENERGY not found")
    return float(values[-1])


def parse_mulliken_charges(text: str, atom_count: int) -> list[float]:
    starts = [match.start() for match in re.finditer(r"MULLIKEN ATOMIC CHARGES(?: AND SPIN POPULATIONS)?", text)]
    if not starts:
        raise ValueError("Mulliken atomic charge section not found")
    values: dict[int, float] = {}
    for line in text[starts[-1] :].splitlines()[1:]:
        match = CHARGE_RE.match(line)
        if match:
            values[int(match.group(1))] = float(match.group(3))
            if len(values) == atom_count:
                break
        elif values and (not line.strip() or "Sum of atomic charges" in line):
            break
    if sorted(values) != list(range(atom_count)):
        raise ValueError(f"expected {atom_count} Mulliken charges, found {len(values)}")
    return [values[index] for index in range(atom_count)]


def parse_s2(text: str, multiplicity: int) -> float:
    values = S2_RE.findall(text)
    if multiplicity == 1 and not values:
        return 0.0
    if not values:
        raise ValueError("<S^2> not found for open-shell state")
    return float(values[-1])


def _fragment_sums(charges: list[float], metadata: dict) -> dict[str, float]:
    return {
        label: sum(charges[index] for index in payload["atom_indices_zero_based"])
        for label, payload in metadata["fragments"].items()
    }


def parse_task(row: dict[str, str], run_root: Path) -> dict:
    task_dir = run_root / row["task_id"]
    out = {field: row.get(field, "") for field in FIELDS}
    try:
        provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
        if provenance["status"] != "complete" or not provenance["same_sp_geometry"]:
            raise ValueError("run provenance is not complete and same-geometry")
        verify_protocol_decks(row, task_dir)
        opt_text = (task_dir / "reduced_opt/orca.out").read_text(encoding="utf-8", errors="replace")
        reduced_text = (task_dir / "reduced_sp/orca.out").read_text(encoding="utf-8", errors="replace")
        oxidized_text = (task_dir / "oxidized_sp/orca.out").read_text(encoding="utf-8", errors="replace")
        metadata = load_metadata(repo_root() / row["input_xyz"])
        atom_count = int(provenance["atom_count"])
        reduced_energy, oxidized_energy = parse_energy(reduced_text), parse_energy(oxidized_text)
        reduced_fragments = _fragment_sums(parse_mulliken_charges(reduced_text, atom_count), metadata)
        oxidized_fragments = _fragment_sums(parse_mulliken_charges(oxidized_text, atom_count), metadata)
        labels = ("A", "S") if row["kind"] == "as_pair" else ("C", "A", "S")
        deltas = {label: oxidized_fragments[label] - reduced_fragments[label] for label in labels}
        oxidized_fragment = max(labels, key=deltas.__getitem__)
        version_match = VERSION_RE.search(reduced_text)
        out.update({
            "software_version": version_match.group(1) if version_match else provenance.get("software_version", "unparsed"),
            "energy_reduced_sp_eh": reduced_energy,
            "energy_oxidized_sp_eh": oxidized_energy,
            "ip_vertical_ev": (oxidized_energy - reduced_energy) * HARTREE_TO_EV,
            **{f"q_{label}_reduced": reduced_fragments.get(label, "") for label in ("C", "A", "S")},
            **{f"q_{label}_oxidized": oxidized_fragments.get(label, "") for label in ("C", "A", "S")},
            **{f"dq_{label}": deltas.get(label, "") for label in ("C", "A", "S")},
            "oxidized_fragment": oxidized_fragment,
            "s2_reduced": parse_s2(reduced_text, int(row["multiplicity_reduced"])),
            "s2_oxidized": parse_s2(oxidized_text, int(row["multiplicity_oxidized"])),
            "scf_reduced_converged": "SCF CONVERGED AFTER" in reduced_text and "SCF NOT CONVERGED" not in reduced_text,
            "scf_oxidized_converged": "SCF CONVERGED AFTER" in oxidized_text and "SCF NOT CONVERGED" not in oxidized_text,
            "optimization_converged": "THE OPTIMIZATION HAS CONVERGED" in opt_text,
            "protocol_decks_verified": True,
            "optimized_geometry_sha256": sha256_file(task_dir / "reduced_opt/orca.xyz"),
            "reduced_sp_input_geometry_sha256": sha256_file(task_dir / "reduced_sp/in.xyz"),
            "oxidized_sp_input_geometry_sha256": sha256_file(task_dir / "oxidized_sp/in.xyz"),
            "same_sp_geometry": provenance["same_sp_geometry"],
            "status": "complete",
            "note": "",
        })
        if not all((out["scf_reduced_converged"], out["scf_oxidized_converged"], out["optimization_converged"])):
            raise ValueError("required ORCA convergence marker missing")
    except Exception as exc:
        out.update({"status": "failed", "note": str(exc)})
    return out


def parse_all(manifest: Path, run_root: Path, output: Path) -> list[dict]:
    rows = [parse_task(row, run_root) for row in read_csv(manifest)]
    write_csv(output, rows, FIELDS)
    return rows


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data/dft_cluster_benchmark/calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs/dft_cluster_benchmark")
    parser.add_argument("--output", type=Path, default=root / "data/dft_cluster_benchmark/task_results.csv")
    args = parser.parse_args()
    rows = parse_all(args.manifest.resolve(), args.run_root.resolve(), args.output.resolve())
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"parsed {complete}/{len(rows)} complete DFT cases")


if __name__ == "__main__":
    main()
