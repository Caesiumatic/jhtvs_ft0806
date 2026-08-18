#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_pair_only import AGAGCL_SHIFT_V, metric_row
from common import Atom, read_csv, read_xyz, repo_root, write_csv

RMSD_EQUIVALENCE_THRESHOLD_ANG = 0.25
HARTREE_TO_KCALMOL = 627.5094740631
TOPOLOGIES = ("CAS", "CSA", "ACS")
COMPOSITION_FIELDS = [
    "cation", "anion", "solvent", "eox_exp_v", "eox_sd_v",
    "ip_free_from_CAS_ev", "ip_free_from_CSA_ev", "ip_free_from_ACS_ev",
    "eox_free_from_CAS_v", "eox_free_from_CSA_v", "eox_free_from_ACS_v",
    "ip_free_min_ev", "eox_free_min_v", "ip_free_lowest_energy_geometry_ev",
    "eox_free_lowest_energy_geometry_v", "lowest_energy_initial_topology",
    "minimum_ip_initial_topology", "final_topology_from_CAS", "final_topology_from_CSA",
    "final_topology_from_ACS", "final_geometry_class", "rmsd_CAS_CSA_ang",
    "rmsd_CAS_ACS_ang", "rmsd_CSA_ACS_ang", "relE_CAS_kcalmol", "relE_CSA_kcalmol",
    "relE_ACS_kcalmol", "status", "note",
]
GEOMETRY_FIELDS = [
    "cation", "anion", "solvent", "rmsd_CAS_CSA_ang", "rmsd_CAS_ACS_ang",
    "rmsd_CSA_ACS_ang", "rmsd_threshold_ang", "final_geometry_class",
    "relE_CAS_kcalmol", "relE_CSA_kcalmol", "relE_ACS_kcalmol",
]
ABSOLUTE_FIELDS = [
    "cation", "anion", "solvent", "eox_exp_v", "eox_sd_v", "eox_as_direct_v",
    "eox_restrained_triad_min_v", "eox_unconstrained_free_min_v",
    "eox_unconstrained_lowest_energy_geometry_v",
]
METRIC_FIELDS = ["descriptor", "n", "MAE_v", "RMSE_v", "mean_signed_error_v", "Pearson_r", "Spearman_rho"]
OFFSET_FIELDS = [
    "descriptor", "n", "slope_fixed", "offset_v", "MAE_after_offset_v",
    "RMSE_after_offset_v", "R2_after_offset", "Pearson_r", "Spearman_rho",
]
RESTRAINT_COMPARISON_FIELDS = [
    "cation", "anion", "solvent", "initial_topology", "ip_restrained_ev",
    "ip_unconstrained_ev", "delta_ip_free_minus_restrained_ev", "restrained_final_topology",
    "unconstrained_final_topology", "neutral_energy_restrained_eh", "neutral_energy_unconstrained_eh",
]


def heavy_atom_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError("RMSD coordinate arrays must have identical N x 3 shape")
    p = left - left.mean(axis=0)
    q = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    aligned = p @ (u @ vt)
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))


def classify_geometry(rmsds: tuple[float, float, float], threshold: float = RMSD_EQUIVALENCE_THRESHOLD_ANG) -> str:
    if max(rmsds) <= threshold:
        return "all_same"
    if min(rmsds) <= threshold:
        return "two_same_one_distinct"
    return "three_distinct"


def select_topologies(group: dict[str, dict[str, str]]) -> tuple[str, str]:
    minimum_ip = min(TOPOLOGIES, key=lambda topology: float(group[topology]["ip_vertical_ev"]))
    lowest_energy = min(TOPOLOGIES, key=lambda topology: float(group[topology]["energy_neutral_sp_eh"]))
    return minimum_ip, lowest_energy


def _canonical_heavy_coordinates(row: dict[str, str], manifest_row: dict[str, str], run_root: Path) -> tuple[list[str], np.ndarray]:
    atoms, _ = read_xyz(run_root / row["task_id"] / "reduced_opt" / "xtbopt.xyz")
    import json

    metadata = json.loads((repo_root() / manifest_row["input_xyz"]).with_suffix(".json").read_text(encoding="utf-8"))
    elements, coordinates = [], []
    for fragment in ("C", "A", "S"):
        for index in metadata["fragments"][fragment]["atom_indices_zero_based"]:
            atom: Atom = atoms[index]
            if atom.element == "H":
                continue
            elements.append(atom.element)
            coordinates.append((atom.x, atom.y, atom.z))
    return elements, np.asarray(coordinates, dtype=float)


def geometry_summary(group: dict[str, dict[str, str]], manifest_by_task: dict[str, dict[str, str]], run_root: Path) -> dict:
    coordinates = {}
    elements = {}
    for topology in TOPOLOGIES:
        elements[topology], coordinates[topology] = _canonical_heavy_coordinates(group[topology], manifest_by_task[group[topology]["task_id"]], run_root)
    if len({tuple(elements[topology]) for topology in TOPOLOGIES}) != 1:
        raise ValueError("canonical heavy-atom identities differ across starting topologies")
    rmsds = (
        heavy_atom_rmsd(coordinates["CAS"], coordinates["CSA"]),
        heavy_atom_rmsd(coordinates["CAS"], coordinates["ACS"]),
        heavy_atom_rmsd(coordinates["CSA"], coordinates["ACS"]),
    )
    energies = {topology: float(group[topology]["energy_neutral_sp_eh"]) for topology in TOPOLOGIES}
    lowest = min(energies.values())
    first = group["CAS"]
    return {
        "cation": first["cation"], "anion": first["anion"], "solvent": first["solvent"],
        "rmsd_CAS_CSA_ang": rmsds[0], "rmsd_CAS_ACS_ang": rmsds[1], "rmsd_CSA_ACS_ang": rmsds[2],
        "rmsd_threshold_ang": RMSD_EQUIVALENCE_THRESHOLD_ANG,
        "final_geometry_class": classify_geometry(rmsds),
        **{f"relE_{topology}_kcalmol": (energies[topology] - lowest) * HARTREE_TO_KCALMOL for topology in TOPOLOGIES},
    }


def build_composition_summary(parsed: list[dict[str, str]], manifest: list[dict[str, str]], benchmark: list[dict[str, str]], run_root: Path) -> tuple[list[dict], list[dict]]:
    if len(parsed) != 72 or any(row["status"] != "complete" for row in parsed):
        raise ValueError("composition analysis requires 72 complete unconstrained triads")
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in parsed:
        grouped[(row["cation"], row["anion"], row["solvent"])][row["initial_topology"]] = row
    manifest_by_task = {row["task_id"]: row for row in manifest}
    experimental = {(row["cation"], row["anion"], row["solvent"]): row for row in benchmark}
    if len(grouped) != 24 or set(grouped) != set(experimental):
        raise ValueError("unconstrained and experimental composition keys differ")
    compositions, geometry_rows = [], []
    for key, exp in experimental.items():
        group = grouped[key]
        if set(group) != set(TOPOLOGIES):
            raise ValueError(f"composition does not contain CAS/CSA/ACS: {key}")
        geom = geometry_summary(group, manifest_by_task, run_root)
        geometry_rows.append(geom)
        minimum_ip, lowest_energy = select_topologies(group)
        row = {
            "cation": key[0], "anion": key[1], "solvent": key[2],
            "eox_exp_v": exp["eox_exp_v"], "eox_sd_v": exp["eox_sd_v"],
            "ip_free_min_ev": float(group[minimum_ip]["ip_vertical_ev"]),
            "eox_free_min_v": float(group[minimum_ip]["ip_vertical_ev"]) - AGAGCL_SHIFT_V,
            "ip_free_lowest_energy_geometry_ev": float(group[lowest_energy]["ip_vertical_ev"]),
            "eox_free_lowest_energy_geometry_v": float(group[lowest_energy]["ip_vertical_ev"]) - AGAGCL_SHIFT_V,
            "lowest_energy_initial_topology": lowest_energy,
            "minimum_ip_initial_topology": minimum_ip,
            "final_geometry_class": geom["final_geometry_class"], "status": "complete", "note": "",
        }
        for topology in TOPOLOGIES:
            ip = float(group[topology]["ip_vertical_ev"])
            row[f"ip_free_from_{topology}_ev"] = ip
            row[f"eox_free_from_{topology}_v"] = ip - AGAGCL_SHIFT_V
            row[f"final_topology_from_{topology}"] = group[topology]["final_inferred_topology"]
            row[f"relE_{topology}_kcalmol"] = geom[f"relE_{topology}_kcalmol"]
        for field in ("rmsd_CAS_CSA_ang", "rmsd_CAS_ACS_ang", "rmsd_CSA_ACS_ang"):
            row[field] = geom[field]
        compositions.append(row)
    return compositions, geometry_rows


def build_absolute_comparison(compositions: list[dict], restrained: list[dict[str, str]], pair_only: list[dict[str, str]]) -> list[dict]:
    restrained_by_key = {(row["cation"], row["anion"], row["solvent"]): row for row in restrained}
    pair_by_key = {(row["cation"], row["anion"], row["solvent"]): row for row in pair_only}
    rows = []
    for row in compositions:
        key = (row["cation"], row["anion"], row["solvent"])
        rows.append({
            "cation": key[0], "anion": key[1], "solvent": key[2], "eox_exp_v": row["eox_exp_v"],
            "eox_sd_v": row["eox_sd_v"], "eox_as_direct_v": pair_by_key[key]["eox_as_calc_vs_agagcl_v"],
            "eox_restrained_triad_min_v": float(restrained_by_key[key]["ip_min_ev"]) - AGAGCL_SHIFT_V,
            "eox_unconstrained_free_min_v": row["eox_free_min_v"],
            "eox_unconstrained_lowest_energy_geometry_v": row["eox_free_lowest_energy_geometry_v"],
        })
    return rows


def build_raw_metrics(rows: list[dict]) -> list[dict]:
    fields = (
        ("AS_direct", "eox_as_direct_v"),
        ("restrained_triad_min", "eox_restrained_triad_min_v"),
        ("unconstrained_free_min", "eox_unconstrained_free_min_v"),
        ("unconstrained_lowest_energy_geometry", "eox_unconstrained_lowest_energy_geometry_v"),
    )
    observed = [float(row["eox_exp_v"]) for row in rows]
    return [metric_row(name, observed, [float(row[field]) for row in rows]) for name, field in fields]


def offset_metric_row(descriptor: str, observed: list[float], calculated: list[float]) -> tuple[dict, list[float]]:
    offset = statistics.fmean(exp - calc for exp, calc in zip(observed, calculated))
    fitted = [value + offset for value in calculated]
    errors = [pred - exp for exp, pred in zip(observed, fitted)]
    ss_res = sum(value * value for value in errors)
    mean_observed = statistics.fmean(observed)
    ss_tot = sum((value - mean_observed) ** 2 for value in observed)
    raw = metric_row(descriptor, observed, fitted)
    return ({
        "descriptor": descriptor, "n": len(observed), "slope_fixed": 1.0, "offset_v": offset,
        "MAE_after_offset_v": raw["MAE_v"], "RMSE_after_offset_v": raw["RMSE_v"],
        "R2_after_offset": 1 - ss_res / ss_tot, "Pearson_r": raw["Pearson_r"], "Spearman_rho": raw["Spearman_rho"],
    }, fitted)


def build_offset_metrics(rows: list[dict]) -> tuple[list[dict], dict[str, list[float]]]:
    fields = (
        ("AS_direct", "eox_as_direct_v"),
        ("restrained_triad_min", "eox_restrained_triad_min_v"),
        ("unconstrained_free_min", "eox_unconstrained_free_min_v"),
        ("unconstrained_lowest_energy_geometry", "eox_unconstrained_lowest_energy_geometry_v"),
    )
    observed = [float(row["eox_exp_v"]) for row in rows]
    metrics, fitted = [], {}
    for descriptor, field in fields:
        metric, predictions = offset_metric_row(descriptor, observed, [float(row[field]) for row in rows])
        metrics.append(metric)
        fitted[descriptor] = predictions
    return metrics, fitted


def build_restrained_comparison(parsed: list[dict[str, str]], restrained: list[dict[str, str]]) -> list[dict]:
    old = {(row["cation"], row["anion"], row["solvent"], row["topology"]): row for row in restrained}
    rows = []
    for row in parsed:
        key = (row["cation"], row["anion"], row["solvent"], row["initial_topology"])
        source = old[key]
        rows.append({
            "cation": key[0], "anion": key[1], "solvent": key[2], "initial_topology": key[3],
            "ip_restrained_ev": source["ip_vertical_ev"], "ip_unconstrained_ev": row["ip_vertical_ev"],
            "delta_ip_free_minus_restrained_ev": float(row["ip_vertical_ev"]) - float(source["ip_vertical_ev"]),
            "restrained_final_topology": source["inferred_topology"],
            "unconstrained_final_topology": row["final_inferred_topology"],
            "neutral_energy_restrained_eh": source["energy_neutral_sp_eh"],
            "neutral_energy_unconstrained_eh": row["energy_neutral_sp_eh"],
        })
    return rows


def make_scatter(path: Path, rows: list[dict], fitted: list[float], metric: dict, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jhtvs_ft0806_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"NTF2": "#1f77b4", "OTF": "#d95f02", "PF6": "#2ca02c"}
    markers = {"PC": "o", "EG": "s", "THF": "^"}
    observed = [float(row["eox_exp_v"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    for row, x, y in zip(rows, observed, fitted):
        ax.scatter(x, y, color=colors[row["anion"]], marker=markers[row["solvent"]], s=55, edgecolor="black", linewidth=0.45)
    low = min(observed + fitted) - 0.12
    high = max(observed + fitted) + 0.12
    ax.plot([low, high], [low, high], color="#444444", linestyle="--", linewidth=1.1, label="1:1")
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Experimental Eox (V vs Ag/AgCl)", ylabel="Offset-fitted calculated Eox (V vs Ag/AgCl)", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.text(0.035, 0.965, f"R² = {metric['R2_after_offset']:.3f}\nMAE = {metric['MAE_after_offset_v']:.3f} V\noffset = {metric['offset_v']:.3f} V", transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#aaaaaa"})
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black", label=anion) for anion, color in colors.items()]
    handles += [Line2D([0], [0], marker=marker, color="#555555", linestyle="none", label=solvent) for solvent, marker in markers.items()]
    ax.legend(handles=handles, ncol=2, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _metric_table(lines: list[str], metrics: list[dict], offset: bool = False) -> None:
    if offset:
        lines.extend(["| descriptor | offset / V | MAE / V | RMSE / V | R² | Pearson | Spearman |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in metrics:
            lines.append(f"| {row['descriptor']} | {row['offset_v']:.6f} | {row['MAE_after_offset_v']:.6f} | {row['RMSE_after_offset_v']:.6f} | {row['R2_after_offset']:.6f} | {row['Pearson_r']:.6f} | {row['Spearman_rho']:.6f} |")
    else:
        lines.extend(["| descriptor | MAE / V | RMSE / V | signed error / V | Pearson | Spearman |", "|---|---:|---:|---:|---:|---:|"])
        for row in metrics:
            lines.append(f"| {row['descriptor']} | {row['MAE_v']:.6f} | {row['RMSE_v']:.6f} | {row['mean_signed_error_v']:.6f} | {row['Pearson_r']:.6f} | {row['Spearman_rho']:.6f} |")


def build_report(parsed: list[dict[str, str]], compositions: list[dict], geometry_rows: list[dict], restrained_comparison: list[dict], raw_metrics: list[dict], offset_metrics: list[dict], job_ids: str) -> str:
    preservation = {topology: sum(row["initial_topology"] == topology and row["topology_changed"] == "False" for row in parsed) for topology in TOPOLOGIES}
    classes = Counter(row["final_geometry_class"] for row in geometry_rows)
    localization = Counter(row["oxidized_fragment"] for row in parsed)
    rmsd_medians = {field: statistics.median(float(row[field]) for row in geometry_rows) for field in ("rmsd_CAS_CSA_ang", "rmsd_CAS_ACS_ang", "rmsd_CSA_ACS_ang")}
    delta_ip = [float(row["delta_ip_free_minus_restrained_ev"]) for row in restrained_comparison]
    same_choice = sum(row["lowest_energy_initial_topology"] == row["minimum_ip_initial_topology"] for row in compositions)
    lines = [
        "# Fully unconstrained Chauhan triad report", "",
        f"- SGE job IDs: `{job_ids}`; complete unconstrained triads: {sum(row['status'] == 'complete' for row in parsed)}/72.",
        f"- Initial-topology preservation: CAS {preservation['CAS']}/24, CSA {preservation['CSA']}/24, ACS {preservation['ACS']}/24.",
        f"- Final-minimum classes at heavy-atom RMSD ≤ {RMSD_EQUIVALENCE_THRESHOLD_ANG:.2f} Å: all_same={classes['all_same']}, two_same_one_distinct={classes['two_same_one_distinct']}, three_distinct={classes['three_distinct']}.",
        f"- Median RMSD (Å): CAS–CSA={rmsd_medians['rmsd_CAS_CSA_ang']:.4f}, CAS–ACS={rmsd_medians['rmsd_CAS_ACS_ang']:.4f}, CSA–ACS={rmsd_medians['rmsd_CSA_ACS_ang']:.4f}.",
        f"- Free-minus-restrained IP change (eV): mean={statistics.fmean(delta_ip):.6f}, median={statistics.median(delta_ip):.6f}, median absolute={statistics.median(abs(value) for value in delta_ip):.6f}, range={min(delta_ip):.6f} to {max(delta_ip):.6f}.",
        f"- Oxidation localization: C={localization['C']}, A={localization['A']}, S={localization['S']} (restrained: C=41, A=0, S=31).", "",
        "## Raw absolute metrics", "",
    ]
    _metric_table(lines, raw_metrics)
    lines.extend(["", "## Global offset-only metrics (slope fixed at 1)", ""])
    _metric_table(lines, offset_metrics, offset=True)
    raw = {row["descriptor"]: row for row in raw_metrics}
    fitted = {row["descriptor"]: row for row in offset_metrics}
    lines.extend([
        "", "## Direct answers", "",
        f"Unconstrained free-min versus restrained triad min: raw MAE change={raw['unconstrained_free_min']['MAE_v'] - raw['restrained_triad_min']['MAE_v']:+.6f} V; offset-fitted MAE change={fitted['unconstrained_free_min']['MAE_after_offset_v'] - fitted['restrained_triad_min']['MAE_after_offset_v']:+.6f} V; R² change={fitted['unconstrained_free_min']['R2_after_offset'] - fitted['restrained_triad_min']['R2_after_offset']:+.6f}; Pearson change={raw['unconstrained_free_min']['Pearson_r'] - raw['restrained_triad_min']['Pearson_r']:+.6f}; Spearman change={raw['unconstrained_free_min']['Spearman_rho'] - raw['restrained_triad_min']['Spearman_rho']:+.6f}.",
        f"Lowest-neutral-energy and minimum-IP choices are the same for {same_choice}/24 compositions. Their raw MAE difference (lowest-energy minus free-min) is {raw['unconstrained_lowest_energy_geometry']['MAE_v'] - raw['unconstrained_free_min']['MAE_v']:+.6f} V; offset-fitted difference is {fitted['unconstrained_lowest_energy_geometry']['MAE_after_offset_v'] - fitted['unconstrained_free_min']['MAE_after_offset_v']:+.6f} V.",
        f"Unconstrained free-min versus A-S pair-only: raw MAE change={raw['unconstrained_free_min']['MAE_v'] - raw['AS_direct']['MAE_v']:+.6f} V; offset-fitted MAE change={fitted['unconstrained_free_min']['MAE_after_offset_v'] - fitted['AS_direct']['MAE_after_offset_v']:+.6f} V.",
        "Topology changes and charge-localization maxima are reported as numerical outcomes only; no mechanistic assignment is inferred.", "",
    ])
    return "\n".join(lines)


def analyze(parsed_path: Path, manifest_path: Path, run_root: Path, benchmark_path: Path, restrained_summary_path: Path, restrained_triads_path: Path, pair_only_path: Path, output_dir: Path, job_ids: str) -> tuple[list[dict], list[dict], list[dict]]:
    parsed = read_csv(parsed_path)
    manifest = read_csv(manifest_path)
    compositions, geometry_rows = build_composition_summary(parsed, manifest, read_csv(benchmark_path), run_root)
    absolute = build_absolute_comparison(compositions, read_csv(restrained_summary_path), read_csv(pair_only_path))
    raw_metrics = build_raw_metrics(absolute)
    offset_metrics, fitted = build_offset_metrics(absolute)
    restrained_comparison = build_restrained_comparison(parsed, read_csv(restrained_triads_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "unconstrained_composition_summary.csv", compositions, COMPOSITION_FIELDS)
    write_csv(output_dir / "geometry_convergence_summary.csv", geometry_rows, GEOMETRY_FIELDS)
    write_csv(output_dir / "absolute_eox_comparison.csv", absolute, ABSOLUTE_FIELDS)
    write_csv(output_dir / "raw_absolute_metrics.csv", raw_metrics, METRIC_FIELDS)
    write_csv(output_dir / "offset_only_metrics.csv", offset_metrics, OFFSET_FIELDS)
    write_csv(output_dir / "restrained_vs_unconstrained.csv", restrained_comparison, RESTRAINT_COMPARISON_FIELDS)
    by_descriptor = {row["descriptor"]: row for row in offset_metrics}
    make_scatter(output_dir / "free_min_offset_scatter.png", absolute, fitted["unconstrained_free_min"], by_descriptor["unconstrained_free_min"], "Unconstrained free-min Eox")
    make_scatter(output_dir / "lowest_energy_offset_scatter.png", absolute, fitted["unconstrained_lowest_energy_geometry"], by_descriptor["unconstrained_lowest_energy_geometry"], "Unconstrained lowest-neutral-energy Eox")
    (output_dir / "unconstrained_triad_report.md").write_text(build_report(parsed, compositions, geometry_rows, restrained_comparison, raw_metrics, offset_metrics, job_ids), encoding="utf-8")
    return compositions, raw_metrics, offset_metrics


def main() -> None:
    root = repo_root()
    data = root / "data" / "chauhan_cation_eox"
    unconstrained = data / "unconstrained"
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", type=Path, default=unconstrained / "unconstrained_triad_results.csv")
    parser.add_argument("--manifest", type=Path, default=unconstrained / "calculation_manifest.csv")
    parser.add_argument("--run-root", type=Path, default=root / "runs" / "chauhan_cation_eox_unconstrained")
    parser.add_argument("--benchmark", type=Path, default=root / "workflows" / "chauhan_cation_eox" / "benchmark_chauhan.csv")
    parser.add_argument("--restrained-summary", type=Path, default=data / "composition_summary.csv")
    parser.add_argument("--restrained-triads", type=Path, default=data / "triad_results.csv")
    parser.add_argument("--pair-only", type=Path, default=data / "pair_only_vs_experiment.csv")
    parser.add_argument("--output-dir", type=Path, default=unconstrained)
    parser.add_argument("--job-ids", default="not recorded")
    args = parser.parse_args()
    compositions, raw_metrics, offset_metrics = analyze(args.parsed.resolve(), args.manifest.resolve(), args.run_root.resolve(), args.benchmark.resolve(), args.restrained_summary.resolve(), args.restrained_triads.resolve(), args.pair_only.resolve(), args.output_dir.resolve(), args.job_ids)
    print(f"wrote {len(compositions)} composition rows, {len(raw_metrics)} raw metrics, and {len(offset_metrics)} offset metrics")


if __name__ == "__main__":
    main()
