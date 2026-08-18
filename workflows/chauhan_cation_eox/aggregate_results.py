#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import benchmark_rows, read_csv, repo_root, write_csv

FIELDS = [
    "cation", "anion", "solvent", "eox_exp_v", "eox_sd_v", "ip_CAS_ev", "ip_CSA_ev",
    "ip_ACS_ev", "ip_min_ev", "ip_mean_ev", "ip_span_ev", "topology_of_min_ip",
    "oxidized_fragment_CAS", "oxidized_fragment_CSA", "oxidized_fragment_ACS",
    "ip_as_direct_ev", "ip_fadel_2p8_ev", "ip_cation_ev", "delta_ip_CAS_vs_AS",
    "delta_ip_CSA_vs_AS", "delta_ip_ACS_vs_AS", "delta_ip_min_vs_AS", "status", "note",
]


def _number(value: str) -> float:
    if value in {"", None}:
        raise ValueError("missing numeric result")
    return float(value)


def aggregate(triad_path: Path, reference_path: Path, benchmark_path: Path, output: Path) -> list[dict]:
    triads = read_csv(triad_path)
    references = read_csv(reference_path)
    benchmark = read_csv(benchmark_path)
    rows = []
    for exp in benchmark:
        key = (exp["cation"], exp["anion"], exp["solvent"])
        selected = {
            row["topology"]: row
            for row in triads
            if (row["cation"], row["anion"], row["solvent"]) == key
        }
        reference = next((row for row in references if (row["anion"], row["solvent"]) == key[1:]), None)
        out = {field: exp.get(field, "") for field in ("cation", "anion", "solvent", "eox_exp_v", "eox_sd_v")}
        try:
            ips = {topology: _number(selected[topology]["ip_vertical_ev"]) for topology in ("CAS", "CSA", "ACS")}
            direct = _number(reference["ip_as_direct_ev"] if reference else "")
            fadel = _number(reference["ip_fadel_2p8_ev"] if reference else "")
            cation_ips = {_number(selected[topology]["ip_cation_ev"]) for topology in ("CAS", "CSA", "ACS")}
            if len(cation_ips) != 1:
                raise ValueError("inconsistent isolated-cation IP across triad topologies")
            minimum_topology = min(ips, key=ips.get)
            out.update({
                "ip_CAS_ev": ips["CAS"], "ip_CSA_ev": ips["CSA"], "ip_ACS_ev": ips["ACS"],
                "ip_min_ev": ips[minimum_topology], "ip_mean_ev": sum(ips.values()) / 3,
                "ip_span_ev": max(ips.values()) - min(ips.values()), "topology_of_min_ip": minimum_topology,
                "oxidized_fragment_CAS": selected["CAS"]["oxidized_fragment"],
                "oxidized_fragment_CSA": selected["CSA"]["oxidized_fragment"],
                "oxidized_fragment_ACS": selected["ACS"]["oxidized_fragment"],
                "ip_as_direct_ev": direct, "ip_fadel_2p8_ev": fadel, "ip_cation_ev": cation_ips.pop(),
                "delta_ip_CAS_vs_AS": ips["CAS"] - direct,
                "delta_ip_CSA_vs_AS": ips["CSA"] - direct,
                "delta_ip_ACS_vs_AS": ips["ACS"] - direct,
                "delta_ip_min_vs_AS": ips[minimum_topology] - direct,
                "status": "complete", "note": "",
            })
        except (KeyError, ValueError) as exc:
            out.update({"status": "not_run_or_incomplete", "note": str(exc)})
        rows.append(out)
    write_csv(output, rows, FIELDS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    root = repo_root()
    data = root / "data" / "chauhan_cation_eox"
    parser.add_argument("--triads", type=Path, default=data / "triad_results.csv")
    parser.add_argument("--references", type=Path, default=data / "as_reference_results.csv")
    parser.add_argument("--benchmark", type=Path, default=root / "workflows" / "chauhan_cation_eox" / "benchmark_chauhan.csv")
    parser.add_argument("--output", type=Path, default=data / "composition_summary.csv")
    args = parser.parse_args()
    rows = aggregate(args.triads.resolve(), args.references.resolve(), args.benchmark.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} Chauhan composition rows")


if __name__ == "__main__":
    main()
