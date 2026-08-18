#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_cation_effects import pearson, spearman
from common import HARTREE_TO_EV, load_metadata, read_csv, read_xyz, repo_root, sha256_file, species_table, write_csv
from parse_results import _fragment_sums, _validated_task_data

AGAGCL_SHIFT_V = 4.477
PAIR_FIELDS = [
    "cation", "anion", "solvent", "eox_exp_v", "eox_sd_v", "ip_as_direct_ev",
    "eox_as_calc_vs_agagcl_v", "error_calc_minus_exp_v", "abs_error_v",
    "oxidized_fragment_as", "dq_A", "dq_S", "same_geometry_pass", "status", "note",
]
UNIQUE_FIELDS = [
    "anion", "solvent", "epsilon", "energy_reduced_opt_eh", "energy_reduced_sp_eh",
    "energy_oxidized_sp_eh", "ip_as_direct_ev", "eox_as_calc_vs_agagcl_v", "dq_A", "dq_S",
    "oxidized_fragment_as", "n_chauhan_cations", "mean_eox_exp_v", "min_eox_exp_v",
    "max_eox_exp_v", "mean_eox_sd_v", "same_geometry_pass", "status", "note",
]
METRIC_FIELDS = ["descriptor", "n", "MAE_v", "RMSE_v", "mean_signed_error_v", "Pearson_r", "Spearman_rho"]


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _terminated(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "normal termination of xtb" in text or "finished run on" in text


def _validate_protocol(row: dict[str, str], provenance: dict, metadata: dict, task_dir: Path) -> None:
    expected_epsilon = species_table()[row["solvent"]]["epsilon"]
    states = tuple(str(row[field]) for field in ("charge_reduced", "uhf_reduced", "charge_oxidized", "uhf_oxidized"))
    if row["kind"] != "as_pair" or states != ("-1", "0", "0", "1"):
        raise ValueError(f"incorrect A-S state definition: {row['task_id']}")
    if float(row["epsilon"]) != float(expected_epsilon):
        raise ValueError(f"incorrect ddCOSMO epsilon: {row['task_id']}")
    if row["restraint"] != "none" or set(metadata["fragments"]) != {"A", "S"}:
        raise ValueError(f"A-S task contains a restraint or non-A/S fragment: {row['task_id']}")
    input_atoms, _ = read_xyz(repo_root() / row["input_xyz"])
    covered = sorted(index for fragment in metadata["fragments"].values() for index in fragment["atom_indices_zero_based"])
    if covered != list(range(len(input_atoms))):
        raise ValueError(f"A-S fragment metadata does not cover exactly the input atoms: {row['task_id']}")

    commands = [provenance["optimization_command"], provenance["reduced_sp_command"], provenance["oxidized_sp_command"]]
    expected_states = (("-1", "0"), ("-1", "0"), ("0", "1"))
    for command, state in zip(commands, expected_states):
        if _flag(command, "--gfn") != "2" or _flag(command, "--cosmo") != str(row["epsilon"]):
            raise ValueError(f"incorrect GFN2/ddCOSMO command: {row['task_id']}")
        if (_flag(command, "--chrg"), _flag(command, "--uhf")) != state:
            raise ValueError(f"incorrect command charge/UHF: {row['task_id']}")
        if "--input" in command:
            raise ValueError(f"unexpected restraint input: {row['task_id']}")
    if "--opt" not in commands[0] or any("--opt" in command for command in commands[1:]):
        raise ValueError(f"incorrect optimization/SP boundary: {row['task_id']}")
    if provenance.get("status") != "complete" or not all(_terminated(task_dir / state / "xtb.out") for state in ("reduced_opt", "reduced_sp", "oxidized_sp")):
        raise ValueError(f"xTB did not terminate successfully: {row['task_id']}")


def load_pair_record(row: dict[str, str], run_root: Path) -> dict:
    task_dir = run_root / row["task_id"]
    provenance = json.loads((task_dir / "provenance.json").read_text(encoding="utf-8"))
    metadata = load_metadata(repo_root() / row["input_xyz"])
    _validate_protocol(row, provenance, metadata, task_dir)
    data = _validated_task_data(row, run_root)
    q_reduced = _fragment_sums(data["charges_reduced"], metadata)
    q_oxidized = _fragment_sums(data["charges_oxidized"], metadata)
    dq = {fragment: q_oxidized[fragment] - q_reduced[fragment] for fragment in ("A", "S")}
    ip_from_sp = (data["energy_oxidized_sp_eh"] - data["energy_reduced_sp_eh"]) * HARTREE_TO_EV
    if not math.isclose(data["ip_ev"], ip_from_sp, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"IP is not the reduced-SP to oxidized-SP energy difference: {row['task_id']}")
    geometry_hashes = {
        sha256_file(task_dir / "reduced_opt" / "xtbopt.xyz"),
        sha256_file(task_dir / "reduced_sp" / "in.xyz"),
        sha256_file(task_dir / "oxidized_sp" / "in.xyz"),
    }
    if len(geometry_hashes) != 1:
        raise ValueError(f"A-S SP geometry hashes differ: {row['task_id']}")
    return {
        "anion": row["anion"], "solvent": row["solvent"], "epsilon": row["epsilon"],
        "energy_reduced_opt_eh": data["energy_opt_eh"],
        "energy_reduced_sp_eh": data["energy_reduced_sp_eh"],
        "energy_oxidized_sp_eh": data["energy_oxidized_sp_eh"],
        "ip_as_direct_ev": ip_from_sp,
        "eox_as_calc_vs_agagcl_v": ip_from_sp - AGAGCL_SHIFT_V,
        "dq_A": dq["A"], "dq_S": dq["S"],
        "oxidized_fragment_as": max(dq, key=dq.get),
        "same_geometry_pass": True, "status": "complete", "note": "",
    }


def build_tables(pair_records: list[dict], benchmark: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    if len(pair_records) != 9 or len({(row["anion"], row["solvent"]) for row in pair_records}) != 9:
        raise ValueError("pair-only analysis requires exactly 9 unique A-S calculations")
    if len(benchmark) != 24:
        raise ValueError("pair-only analysis requires exactly 24 Chauhan compositions")
    pair_by_key = {(row["anion"], row["solvent"]): row for row in pair_records}
    experiments: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    comparison = []
    for exp in benchmark:
        key = (exp["anion"], exp["solvent"])
        pair = pair_by_key.get(key)
        if pair is None:
            raise ValueError(f"no A-S calculation for {key}")
        experiments[key].append(exp)
        error = float(pair["eox_as_calc_vs_agagcl_v"]) - float(exp["eox_exp_v"])
        comparison.append({
            "cation": exp["cation"], "anion": exp["anion"], "solvent": exp["solvent"],
            "eox_exp_v": exp["eox_exp_v"], "eox_sd_v": exp["eox_sd_v"],
            "ip_as_direct_ev": pair["ip_as_direct_ev"],
            "eox_as_calc_vs_agagcl_v": pair["eox_as_calc_vs_agagcl_v"],
            "error_calc_minus_exp_v": error, "abs_error_v": abs(error),
            "oxidized_fragment_as": pair["oxidized_fragment_as"], "dq_A": pair["dq_A"], "dq_S": pair["dq_S"],
            "same_geometry_pass": pair["same_geometry_pass"], "status": pair["status"], "note": pair["note"],
        })
    unique = []
    for pair in pair_records:
        group = experiments[(pair["anion"], pair["solvent"])]
        eox = [float(row["eox_exp_v"]) for row in group]
        sds = [float(row["eox_sd_v"]) for row in group]
        unique.append({
            **pair, "n_chauhan_cations": len(group), "mean_eox_exp_v": statistics.fmean(eox),
            "min_eox_exp_v": min(eox), "max_eox_exp_v": max(eox), "mean_eox_sd_v": statistics.fmean(sds),
        })
    for key in pair_by_key:
        values = {(row["ip_as_direct_ev"], row["eox_as_calc_vs_agagcl_v"]) for row in comparison if (row["anion"], row["solvent"]) == key}
        if len(values) != 1:
            raise AssertionError(f"calculated A-S value differs across cations: {key}")
    return comparison, unique


def metric_row(descriptor: str, observed: list[float], predicted: list[float]) -> dict:
    errors = [calc - exp for exp, calc in zip(observed, predicted)]
    return {
        "descriptor": descriptor, "n": len(errors),
        "MAE_v": statistics.fmean(abs(value) for value in errors),
        "RMSE_v": math.sqrt(statistics.fmean(value * value for value in errors)),
        "mean_signed_error_v": statistics.fmean(errors),
        "Pearson_r": pearson(observed, predicted), "Spearman_rho": spearman(observed, predicted),
    }


def build_metrics(comparison: list[dict], unique: list[dict], composition_summary: list[dict[str, str]]) -> tuple[dict, dict, list[dict]]:
    observed = [float(row["eox_exp_v"]) for row in comparison]
    composition_metrics = metric_row("AS_direct", observed, [float(row["eox_as_calc_vs_agagcl_v"]) for row in comparison])
    unique_metrics = metric_row(
        "AS_direct_unique", [float(row["mean_eox_exp_v"]) for row in unique],
        [float(row["eox_as_calc_vs_agagcl_v"]) for row in unique],
    )
    summary_by_key = {(row["cation"], row["anion"], row["solvent"]): row for row in composition_summary}
    if len(summary_by_key) != 24 or any(row["status"] != "complete" for row in composition_summary):
        raise ValueError("triad comparison requires 24 complete composition-summary rows")
    descriptor_fields = (
        ("AS_direct", None), ("CAS", "ip_CAS_ev"), ("CSA", "ip_CSA_ev"),
        ("ACS", "ip_ACS_ev"), ("triad_min", "ip_min_ev"), ("triad_mean", "ip_mean_ev"),
    )
    metrics = []
    for descriptor, field in descriptor_fields:
        if field is None:
            predicted = [float(row["eox_as_calc_vs_agagcl_v"]) for row in comparison]
        else:
            predicted = [float(summary_by_key[(row["cation"], row["anion"], row["solvent"])][field]) - AGAGCL_SHIFT_V for row in comparison]
        metrics.append(metric_row(descriptor, observed, predicted))
    return composition_metrics, unique_metrics, metrics


def _value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def build_report(comparison: list[dict], unique: list[dict], composition_metrics: dict, unique_metrics: dict, absolute_metrics: list[dict]) -> str:
    calculated = [float(row["eox_as_calc_vs_agagcl_v"]) for row in comparison]
    experimental = [float(row["eox_exp_v"]) for row in comparison]
    localization = Counter(row["oxidized_fragment_as"] for row in unique)
    pair_mae = float(composition_metrics["MAE_v"])
    deltas = {row["descriptor"]: float(row["MAE_v"]) - pair_mae for row in absolute_metrics if row["descriptor"] != "AS_direct"}
    lines = [
        "# Cation-free solvent-anion pair Eox test", "",
        "All nine existing A-S calculations passed the frozen protocol and were reused; no A-S, triad, or cation calculation was rerun.", "",
        f"Calculated pair-only Eox range: {min(calculated):.6f} to {max(calculated):.6f} V vs Ag/AgCl. Experimental range: {min(experimental):.6f} to {max(experimental):.6f} V.",
        f"Oxidation localization across the 9 unique pairs: A={localization['A']}, S={localization['S']}.", "",
        "## Absolute 24-row comparison", "",
        "| Anion | Solvent | Cation | Experimental Eox vs Ag/AgCl / V | Pair IP / eV | Pair calculated Eox vs Ag/AgCl / V | Error / V | Oxidized fragment |",
        "| ----- | ------- | ------ | ------------------------------: | -----------: | ---------------------------------: | --------: | ----------------- |",
    ]
    for row in comparison:
        lines.append(f"| {row['anion']} | {row['solvent']} | {row['cation']} | {float(row['eox_exp_v']):.6f} | {float(row['ip_as_direct_ev']):.6f} | {float(row['eox_as_calc_vs_agagcl_v']):.6f} | {float(row['error_calc_minus_exp_v']):.6f} | {row['oxidized_fragment_as']} |")
    lines.extend(["", "## Pair-only metrics", "", "| weighting | n | MAE / V | RMSE / V | mean signed error / V | Pearson r | Spearman rho |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label, metric in (("24-row composition-weighted", composition_metrics), ("9-row unique-AS", unique_metrics)):
        lines.append(f"| {label} | {metric['n']} | {_value(metric['MAE_v'])} | {_value(metric['RMSE_v'])} | {_value(metric['mean_signed_error_v'])} | {_value(metric['Pearson_r'])} | {_value(metric['Spearman_rho'])} |")
    lines.extend(["", "## Pair-only versus existing triad descriptors", "", "| descriptor | n | MAE / V | RMSE / V | mean signed error / V | Pearson r | Spearman rho |", "|---|---:|---:|---:|---:|---:|---:|"])
    for metric in absolute_metrics:
        lines.append(f"| {metric['descriptor']} | {metric['n']} | {_value(metric['MAE_v'])} | {_value(metric['RMSE_v'])} | {_value(metric['mean_signed_error_v'])} | {_value(metric['Pearson_r'])} | {_value(metric['Spearman_rho'])} |")
    comparison_text = "; ".join(f"{name}: {'triad better' if delta < 0 else 'pair better'} by {abs(delta):.6f} V MAE" for name, delta in deltas.items())
    lines.extend([
        "", f"At the fixed 4.477 V reference conversion, pair-only MAE is {pair_mae:.6f} V. Relative absolute agreement: {comparison_text}.",
        "No offset was fitted, no Chauhan calibration was performed, and no data were centered.", "",
    ])
    return "\n".join(lines)


def analyze(manifest_path: Path, run_root: Path, benchmark_path: Path, composition_path: Path, output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    tasks = [row for row in read_csv(manifest_path) if row["kind"] == "as_pair"]
    pair_records = [load_pair_record(row, run_root) for row in tasks]
    comparison, unique = build_tables(pair_records, read_csv(benchmark_path))
    composition_metrics, unique_metrics, absolute_metrics = build_metrics(comparison, unique, read_csv(composition_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pair_only_vs_experiment.csv", comparison, PAIR_FIELDS)
    write_csv(output_dir / "pair_only_unique_AS.csv", unique, UNIQUE_FIELDS)
    write_csv(output_dir / "pair_vs_triad_absolute_metrics.csv", absolute_metrics, METRIC_FIELDS)
    (output_dir / "pair_only_eox_report.md").write_text(build_report(comparison, unique, composition_metrics, unique_metrics, absolute_metrics), encoding="utf-8")
    return comparison, unique, absolute_metrics


def main() -> None:
    root = repo_root()
    data = root / "data" / "chauhan_cation_eox"
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=data / "calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "chauhan_cation_eox")
    parser.add_argument("--benchmark", type=Path, default=root / "workflows" / "chauhan_cation_eox" / "benchmark_chauhan.csv")
    parser.add_argument("--composition-summary", type=Path, default=data / "composition_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=data)
    args = parser.parse_args()
    comparison, unique, metrics = analyze(args.manifest.resolve(), args.run_root.resolve(), args.benchmark.resolve(), args.composition_summary.resolve(), args.output_dir.resolve())
    print(f"wrote {len(comparison)} composition rows, {len(unique)} unique A-S rows, and {len(metrics)} absolute metric rows")


if __name__ == "__main__":
    main()
