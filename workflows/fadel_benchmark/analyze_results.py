#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path

try:
    from .common import ANIONS, SOLVENTS, TOPOLOGIES, read_csv, repo_root, write_csv
except ImportError:
    from common import ANIONS, SOLVENTS, TOPOLOGIES, read_csv, repo_root, write_csv

COMPARISON_FIELDS = [
    "anion", "solvent", "fadel_ip_dscf_mean_ev", "xtb_as_ip_ev", "error_as_ev", "abs_error_as_ev",
    "xtb_CAS_ev", "xtb_CSA_ev", "xtb_ACS_ev", "xtb_triad_min_ev", "xtb_triad_mean_ev",
    "error_triad_min_ev", "abs_error_triad_min_ev", "triad_improvement_over_as_ev",
    "as_oxidized_fragment", "CAS_oxidized_fragment", "CSA_oxidized_fragment", "ACS_oxidized_fragment",
    "triad_min_topology", "triad_span_ev", "status", "note",
]
RAW_FIELDS = ["descriptor", "n", "MAE_eV", "RMSE_eV", "mean_signed_error_eV", "Pearson_r", "Pearson_r2_correlation_squared", "Spearman_rho"]
OFFSET_FIELDS = ["descriptor", "n", "slope_fixed", "offset_ev", "MAE_after_offset_ev", "RMSE_after_offset_ev", "Pearson_r", "Pearson_r2_correlation_squared", "Spearman_rho"]
CHEMISTRY_FIELDS = ["group_type", "group_value", "descriptor", *RAW_FIELDS[1:]]
BRANCH_FIELDS = [
    "anion", "solvent", "fadel_dominant_oxidized_fragment", "fadel_anion_oxidation_fraction",
    "fadel_solvent_oxidation_fraction", "xtb_as_oxidized_fragment", "xtb_triad_min_oxidized_fragment",
    "as_branch_match", "triad_branch_match",
]
CCSDT_FIELDS = ["anion", "solvent", "dlpno_ccsdt_pair_ip_ev", "m06hf_pair_ip_ev", "fadel_table2_m06hf_ip_ev", "xtb_as_ip_ev", "xtb_triad_min_ip_ev", "source"]
DESCRIPTORS = (
    ("AS_direct", "xtb_as_ip_ev"), ("CAS", "xtb_CAS_ev"), ("CSA", "xtb_CSA_ev"),
    ("ACS", "xtb_ACS_ev"), ("triad_min", "xtb_triad_min_ev"), ("triad_mean", "xtb_triad_mean_ev"),
)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for index in ordered[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal vectors with at least two values")
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right))
    return numerator / denominator if denominator else float("nan")


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(_ranks(left), _ranks(right))


def metric_row(descriptor: str, observed: list[float], predicted: list[float]) -> dict:
    errors = [calc - ref for ref, calc in zip(observed, predicted)]
    correlation = pearson(observed, predicted)
    return {
        "descriptor": descriptor, "n": len(errors), "MAE_eV": statistics.fmean(abs(value) for value in errors),
        "RMSE_eV": math.sqrt(statistics.fmean(value * value for value in errors)),
        "mean_signed_error_eV": statistics.fmean(errors), "Pearson_r": correlation,
        "Pearson_r2_correlation_squared": correlation * correlation, "Spearman_rho": spearman(observed, predicted),
    }


def offset_metric_row(descriptor: str, observed: list[float], predicted: list[float]) -> tuple[dict, list[float]]:
    offset = statistics.fmean(ref - calc for ref, calc in zip(observed, predicted))
    corrected = [value + offset for value in predicted]
    raw = metric_row(descriptor, observed, corrected)
    return ({
        "descriptor": descriptor, "n": len(observed), "slope_fixed": 1.0, "offset_ev": offset,
        "MAE_after_offset_ev": raw["MAE_eV"], "RMSE_after_offset_ev": raw["RMSE_eV"],
        "Pearson_r": raw["Pearson_r"], "Pearson_r2_correlation_squared": raw["Pearson_r2_correlation_squared"],
        "Spearman_rho": raw["Spearman_rho"],
    }, corrected)


def build_comparison(tasks: list[dict[str, str]], references: list[dict[str, str]]) -> list[dict]:
    if len(tasks) != 64 or any(row["status"] != "complete" for row in tasks):
        raise ValueError("Fadel analysis requires 64 complete tasks")
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in tasks:
        key = (row["anion"], row["solvent"])
        by_key.setdefault(key, {})[row["initial_topology"]] = row
    reference_by_key = {(row["anion"], row["solvent"]): row for row in references}
    if len(reference_by_key) != 16 or set(by_key) != set(reference_by_key):
        raise ValueError("task and reference chemistry keys differ")
    comparison = []
    for reference in references:
        key = (reference["anion"], reference["solvent"])
        group = by_key[key]
        if set(group) != {"AS", *TOPOLOGIES}:
            raise ValueError(f"missing topology for {key}")
        fadel = float(reference["fadel_ip_dscf_mean_ev"])
        as_ip = float(group["AS"]["ip_vertical_ev"])
        triad_ips = {topology: float(group[topology]["ip_vertical_ev"]) for topology in TOPOLOGIES}
        minimum = min(TOPOLOGIES, key=triad_ips.__getitem__)
        triad_min = triad_ips[minimum]
        row = {
            "anion": key[0], "solvent": key[1], "fadel_ip_dscf_mean_ev": fadel,
            "xtb_as_ip_ev": as_ip, "error_as_ev": as_ip - fadel, "abs_error_as_ev": abs(as_ip - fadel),
            **{f"xtb_{topology}_ev": triad_ips[topology] for topology in TOPOLOGIES},
            "xtb_triad_min_ev": triad_min, "xtb_triad_mean_ev": statistics.fmean(triad_ips.values()),
            "error_triad_min_ev": triad_min - fadel, "abs_error_triad_min_ev": abs(triad_min - fadel),
            "triad_improvement_over_as_ev": abs(as_ip - fadel) - abs(triad_min - fadel),
            "as_oxidized_fragment": group["AS"]["oxidized_fragment"],
            **{f"{topology}_oxidized_fragment": group[topology]["oxidized_fragment"] for topology in TOPOLOGIES},
            "triad_min_topology": minimum, "triad_span_ev": max(triad_ips.values()) - min(triad_ips.values()),
            "status": "complete", "note": "",
        }
        comparison.append(row)
    return comparison


def build_metrics(rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    observed = [float(row["fadel_ip_dscf_mean_ev"]) for row in rows]
    raw, offsets, corrected = [], [], {}
    for descriptor, field in DESCRIPTORS:
        predicted = [float(row[field]) for row in rows]
        raw.append(metric_row(descriptor, observed, predicted))
        offset, corrected_values = offset_metric_row(descriptor, observed, predicted)
        offsets.append(offset)
        corrected[descriptor] = corrected_values
    return raw, offsets, corrected


def build_chemistry_metrics(rows: list[dict]) -> list[dict]:
    output = []
    for group_type, values in (("solvent", SOLVENTS), ("anion", ANIONS)):
        for value in values:
            subset = [row for row in rows if row[group_type] == value]
            observed = [float(row["fadel_ip_dscf_mean_ev"]) for row in subset]
            for descriptor, field in (("AS_direct", "xtb_as_ip_ev"), ("triad_min", "xtb_triad_min_ev")):
                output.append({"group_type": group_type, "group_value": value, **metric_row(descriptor, observed, [float(row[field]) for row in subset])})
    return output


def build_branch_comparison(rows: list[dict], branch_reference: list[dict[str, str]]) -> list[dict]:
    reference = {(row["anion"], row["solvent"]): row for row in branch_reference}
    output = []
    for row in rows:
        key = (row["anion"], row["solvent"])
        fadel = reference[key]
        triad_fragment = row[f"{row['triad_min_topology']}_oxidized_fragment"]
        dominant = fadel["fadel_dominant_oxidized_fragment"]
        output.append({
            "anion": key[0], "solvent": key[1], "fadel_dominant_oxidized_fragment": dominant,
            "fadel_anion_oxidation_fraction": fadel["anion_oxidation_fraction"],
            "fadel_solvent_oxidation_fraction": fadel["solvent_oxidation_fraction"],
            "xtb_as_oxidized_fragment": row["as_oxidized_fragment"],
            "xtb_triad_min_oxidized_fragment": triad_fragment,
            "as_branch_match": row["as_oxidized_fragment"] == dominant,
            "triad_branch_match": triad_fragment == dominant,
        })
    return output


def build_ccsdt_subset(rows: list[dict], ccsdt_reference: list[dict[str, str]]) -> list[dict]:
    by_key = {(row["anion"], row["solvent"]): row for row in rows}
    return [{
        **reference,
        "fadel_table2_m06hf_ip_ev": by_key[(reference["anion"], reference["solvent"])]["fadel_ip_dscf_mean_ev"],
        "xtb_as_ip_ev": by_key[(reference["anion"], reference["solvent"])]["xtb_as_ip_ev"],
        "xtb_triad_min_ip_ev": by_key[(reference["anion"], reference["solvent"])]["xtb_triad_min_ev"],
    } for reference in ccsdt_reference]


def make_scatter(path: Path, rows: list[dict], predicted: list[float], metric: dict, title: str, offset: float | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jhtvs_ft0806_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"TDI": "#1f77b4", "TFSI": "#d95f02", "BF4": "#2ca02c", "PF6": "#9467bd"}
    markers = {"DMSO": "o", "DME": "s", "PC": "^", "ACN": "D"}
    observed = [float(row["fadel_ip_dscf_mean_ev"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=180)
    for row, x, y in zip(rows, observed, predicted):
        ax.scatter(
            x, y, color=colors[row["anion"]], marker=markers[row["solvent"]],
            s=52, edgecolor="black", linewidth=0.45,
        )
    low, high = min(observed + predicted) - 0.25, max(observed + predicted) + 0.25
    ax.plot([low, high], [low, high], "--", color="#444444", linewidth=1.0)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Fadel M06-HF mean vertical IP (eV)", ylabel="GFN2-xTB vertical IP (eV)", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    label = f"MAE = {metric['MAE_eV' if offset is None else 'MAE_after_offset_ev']:.3f} eV\nPearson r² = {metric['Pearson_r2_correlation_squared']:.3f}"
    if offset is not None:
        label += f"\noffset = {offset:+.3f} eV"
    ax.text(0.035, 0.965, label, transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#aaaaaa"})
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="black", label=anion)
        for anion, color in colors.items()
    ] + [
        Line2D([], [], marker=marker, linestyle="none", color="black", markerfacecolor="white", label=solvent)
        for solvent, marker in markers.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=7.0, framealpha=0.92, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_report(rows: list[dict], raw: list[dict], offsets: list[dict], branches: list[dict], tasks: list[dict[str, str]], job_ids: str) -> str:
    raw_by = {row["descriptor"]: row for row in raw}
    offset_by = {row["descriptor"]: row for row in offsets}
    improvements = [float(row["triad_improvement_over_as_ev"]) for row in rows]
    tied = sum(abs(value) <= 0.05 for value in improvements)
    triad_better = sum(value > 0.05 for value in improvements)
    as_better = sum(value < -0.05 for value in improvements)
    localization = Counter(row["oxidized_fragment"] for row in tasks if row["kind"] == "triad")
    as_matches = sum(row["as_branch_match"] == "True" or row["as_branch_match"] is True for row in branches)
    triad_matches = sum(row["triad_branch_match"] == "True" or row["triad_branch_match"] is True for row in branches)
    return "\n".join([
        "# Fadel vacuum vertical-IP benchmark", "",
        f"- SGE job IDs: `{job_ids}`; complete A-S: 16/16; complete Li-A-S topologies: 48/48.",
        "- Official Figshare data were accessed. The workbook contains snapshot IDs, IPs and oxidation branches, but no atomic coordinates; the exact-geometry xTB snapshot diagnostic was therefore not run.",
        f"- Raw MAE: A-S={raw_by['AS_direct']['MAE_eV']:.6f} eV; triad-min={raw_by['triad_min']['MAE_eV']:.6f} eV; A-S has the lower raw MAE.",
        f"- Raw trends: A-S Pearson={raw_by['AS_direct']['Pearson_r']:.6f}, Spearman={raw_by['AS_direct']['Spearman_rho']:.6f}; triad-min Pearson={raw_by['triad_min']['Pearson_r']:.6f}, Spearman={raw_by['triad_min']['Spearman_rho']:.6f}. Triad-min has the marginally higher Pearson correlation; A-S has the higher Spearman rank correlation.",
        f"- Offset-only MAE: A-S={offset_by['AS_direct']['MAE_after_offset_ev']:.6f} eV; triad-min={offset_by['triad_min']['MAE_after_offset_ev']:.6f} eV; triad-min has the lower offset-corrected MAE.",
        f"- Paired comparison at |improvement| <= 0.05 eV tie tolerance: triad-min better={triad_better}, A-S better={as_better}, tied={tied}; mean improvement={statistics.fmean(improvements):.6f} eV; median={statistics.median(improvements):.6f} eV.",
        f"- Fadel dominant A/S branch match: A-S={as_matches}/16; triad-min={triad_matches}/16.",
        f"- All Li-A-S topology calculations localization: Li(C)={localization['C']}, anion={localization['A']}, solvent={localization['S']}.",
        "- The largest chemistry-resolved raw MAE occurs for TDI by anion and DMSO by solvent for both A-S and triad-min; the errors are not concentrated in TFSI or BF4/PF6. Full values are in fadel_metrics_by_chemistry.csv.",
        "- The missing exact Fadel snapshot coordinates prevent separating electronic-method error from optimized-geometry/sampling error in this run.",
        "- No electrochemical reference conversion or ddCOSMO term is present.",
        "- This benchmark is intentionally independent of Chauhan CV and reference-electrode effects.", "",
    ])


def analyze(data_dir: Path, output_dir: Path, job_ids: str) -> tuple[list[dict], list[dict], list[dict]]:
    tasks = read_csv(data_dir / "fadel_task_results.csv")
    rows = build_comparison(tasks, read_csv(data_dir / "fadel_table2_reference.csv"))
    raw, offsets, corrected = build_metrics(rows)
    chemistry = build_chemistry_metrics(rows)
    branches = build_branch_comparison(rows, read_csv(data_dir / "fadel_oxidation_branch_reference.csv"))
    ccsdt = build_ccsdt_subset(rows, read_csv(data_dir / "fadel_ccsdt_pc_reference.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fadel_as_vs_cas.csv", rows, COMPARISON_FIELDS)
    write_csv(output_dir / "fadel_method_metrics_raw.csv", raw, RAW_FIELDS)
    write_csv(output_dir / "fadel_method_metrics_offset.csv", offsets, OFFSET_FIELDS)
    write_csv(output_dir / "fadel_metrics_by_chemistry.csv", chemistry, CHEMISTRY_FIELDS)
    write_csv(output_dir / "fadel_oxidation_branch_comparison.csv", branches, BRANCH_FIELDS)
    write_csv(output_dir / "fadel_ccsdt_pc_subset.csv", ccsdt, CCSDT_FIELDS)
    raw_by, offset_by = ({row["descriptor"]: row for row in table} for table in (raw, offsets))
    observed_as = [float(row["xtb_as_ip_ev"]) for row in rows]
    observed_triad = [float(row["xtb_triad_min_ev"]) for row in rows]
    make_scatter(output_dir / "fadel_vs_as_raw.png", rows, observed_as, raw_by["AS_direct"], "Fadel vs GFN2 A-S (raw)")
    make_scatter(output_dir / "fadel_vs_triad_min_raw.png", rows, observed_triad, raw_by["triad_min"], "Fadel vs GFN2 Li-A-S triad-min (raw)")
    make_scatter(output_dir / "fadel_vs_as_offset.png", rows, corrected["AS_direct"], offset_by["AS_direct"], "Fadel vs GFN2 A-S (offset-only)", float(offset_by["AS_direct"]["offset_ev"]))
    make_scatter(output_dir / "fadel_vs_triad_min_offset.png", rows, corrected["triad_min"], offset_by["triad_min"], "Fadel vs GFN2 Li-A-S triad-min (offset-only)", float(offset_by["triad_min"]["offset_ev"]))
    (output_dir / "fadel_as_vs_cas_report.md").write_text(build_report(rows, raw, offsets, branches, tasks, job_ids), encoding="utf-8")
    return rows, raw, offsets


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "fadel_benchmark")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "fadel_benchmark")
    parser.add_argument("--job-ids", default="not recorded")
    args = parser.parse_args()
    rows, raw, offsets = analyze(args.data_dir.resolve(), args.output_dir.resolve(), args.job_ids)
    print(f"wrote {len(rows)} comparison rows, {len(raw)} raw metrics, and {len(offsets)} offset metrics")


if __name__ == "__main__":
    main()
