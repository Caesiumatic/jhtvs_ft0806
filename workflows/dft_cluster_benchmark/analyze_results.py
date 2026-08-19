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
    from .common import TOPOLOGIES, read_csv, repo_root, write_csv
except ImportError:
    from common import TOPOLOGIES, read_csv, repo_root, write_csv

DESCRIPTORS = (
    ("A-S", "AS"), ("CAS", "CAS"), ("CSA", "CSA"), ("ACS", "ACS"),
    ("(min)", "min"), ("mean", "mean"),
)
RAW_FIELDS = ["descriptor", "n", "MAE", "RMSE", "mean_signed_error", "Pearson_r", "Pearson_r2_correlation_squared", "Spearman_rho"]
OFFSET_FIELDS = ["descriptor", "n", "slope_fixed", "offset", "MAE_after_offset_fit", "RMSE_after_offset_fit", "Pearson_r", "Pearson_r2_correlation_squared", "Spearman_rho"]
CHAUHAN_SUMMARY_FIELDS = [
    "cation", "anion", "solvent", "eox_exp_v", "eox_sd_v",
    "dft_ip_AS_ev", "dft_ip_CAS_ev", "dft_ip_CSA_ev", "dft_ip_ACS_ev", "dft_ip_min_ev", "dft_ip_mean_ev",
    "dft_eox_AS_v", "dft_eox_CAS_v", "dft_eox_CSA_v", "dft_eox_ACS_v", "dft_eox_min_v", "dft_eox_mean_v",
    "topology_of_min", "triad_span_ev", "oxidized_fragment_AS", "oxidized_fragment_CAS",
    "oxidized_fragment_CSA", "oxidized_fragment_ACS", "status", "note",
]
FADEL_SUMMARY_FIELDS = [
    "anion", "solvent", "fadel_ip_ev", "dft_ip_AS_ev", "dft_ip_CAS_ev", "dft_ip_CSA_ev", "dft_ip_ACS_ev",
    "dft_ip_min_ev", "dft_ip_mean_ev", "error_AS_ev", "error_CAS_ev", "error_CSA_ev", "error_ACS_ev",
    "error_min_ev", "error_mean_ev", "topology_of_min", "triad_span_ev", "oxidized_fragment_AS",
    "oxidized_fragment_CAS", "oxidized_fragment_CSA", "oxidized_fragment_ACS", "status", "note",
]
CHAUHAN_XTB_DFT_FIELDS = ["cation", "anion", "solvent", "descriptor", "xtb_ip_ev", "dft_ip_ev", "delta_dft_minus_xtb_ev", "xtb_eox_v", "dft_eox_v"]
FADEL_XTB_DFT_FIELDS = ["anion", "solvent", "descriptor", "xtb_ip_ev", "dft_ip_ev", "delta_dft_minus_xtb_ev", "fadel_ip_ev"]
BRANCH_FIELDS = [
    "anion", "solvent", "fadel_dominant_oxidized_fragment", "fadel_anion_oxidation_fraction",
    "fadel_solvent_oxidation_fraction", "dft_AS_oxidized_fragment", "dft_CAS_oxidized_fragment",
    "dft_CSA_oxidized_fragment", "dft_ACS_oxidized_fragment", "AS_branch_match", "CAS_branch_match",
    "CSA_branch_match", "ACS_branch_match",
]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    dl, dr = [value - mean_left for value in left], [value - mean_right for value in right]
    denominator = math.sqrt(sum(value * value for value in dl) * sum(value * value for value in dr))
    return sum(a * b for a, b in zip(dl, dr)) / denominator if denominator else float("nan")


def metric_row(descriptor: str, observed: list[float], predicted: list[float]) -> dict:
    errors = [prediction - reference for reference, prediction in zip(observed, predicted)]
    correlation = pearson(observed, predicted)
    return {
        "descriptor": descriptor, "n": len(errors),
        "MAE": statistics.fmean(abs(value) for value in errors),
        "RMSE": math.sqrt(statistics.fmean(value * value for value in errors)),
        "mean_signed_error": statistics.fmean(errors),
        "Pearson_r": correlation, "Pearson_r2_correlation_squared": correlation * correlation,
        "Spearman_rho": pearson(_ranks(observed), _ranks(predicted)),
    }


def offset_row(descriptor: str, observed: list[float], predicted: list[float]) -> tuple[dict, list[float]]:
    offset = statistics.fmean(reference - value for reference, value in zip(observed, predicted))
    corrected = [value + offset for value in predicted]
    raw = metric_row(descriptor, observed, corrected)
    return ({
        "descriptor": descriptor, "n": len(observed), "slope_fixed": 1.0, "offset": offset,
        "MAE_after_offset_fit": raw["MAE"], "RMSE_after_offset_fit": raw["RMSE"],
        "Pearson_r": raw["Pearson_r"], "Pearson_r2_correlation_squared": raw["Pearson_r2_correlation_squared"],
        "Spearman_rho": raw["Spearman_rho"],
    }, corrected)


def metric_tables(rows: list[dict], reference_field: str, prediction_prefix: str) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    observed = [float(row[reference_field]) for row in rows]
    raw_rows, offset_rows, corrected = [], [], {}
    for descriptor, key in DESCRIPTORS:
        predicted = [float(row[f"{prediction_prefix}{key}"]) for row in rows]
        raw_rows.append(metric_row(descriptor, observed, predicted))
        offset, values = offset_row(descriptor, observed, predicted)
        offset_rows.append(offset)
        corrected[key] = values
    return raw_rows, offset_rows, corrected


def _task_maps(tasks: list[dict[str, str]], benchmark: str) -> tuple[dict, dict]:
    subset = [row for row in tasks if row["benchmark"] == benchmark]
    expected = 81 if benchmark == "chauhan" else 64
    if len(subset) != expected or any(row["status"] != "complete" for row in subset):
        raise ValueError(f"{benchmark} analysis requires {expected} complete DFT cases")
    pairs = {(row["anion"], row["solvent"]): row for row in subset if row["kind"] == "as_pair"}
    triads = {(row["cation"], row["anion"], row["solvent"], row["topology"]): row for row in subset if row["kind"] == "triad"}
    return pairs, triads


def build_chauhan_summary(tasks: list[dict[str, str]], experiment: list[dict[str, str]]) -> list[dict]:
    pairs, triads = _task_maps(tasks, "chauhan")
    output = []
    for reference in experiment:
        cation, anion, solvent = reference["cation"], reference["anion"], reference["solvent"]
        pair = pairs[(anion, solvent)]
        selected = {topology: triads[(cation, anion, solvent, topology)] for topology in TOPOLOGIES}
        ips = {"AS": float(pair["ip_vertical_ev"]), **{topology: float(selected[topology]["ip_vertical_ev"]) for topology in TOPOLOGIES}}
        minimum = min(TOPOLOGIES, key=ips.__getitem__)
        ips.update({"min": ips[minimum], "mean": statistics.fmean(ips[topology] for topology in TOPOLOGIES)})
        row = {field: reference.get(field, "") for field in ("cation", "anion", "solvent", "eox_exp_v", "eox_sd_v")}
        row.update({f"dft_ip_{key}_ev": ips[key] for key in ("AS", *TOPOLOGIES, "min", "mean")})
        row.update({f"dft_eox_{key}_v": ips[key] - 4.477 for key in ("AS", *TOPOLOGIES, "min", "mean")})
        row.update({
            "topology_of_min": minimum,
            "triad_span_ev": max(ips[topology] for topology in TOPOLOGIES) - min(ips[topology] for topology in TOPOLOGIES),
            "oxidized_fragment_AS": pair["oxidized_fragment"],
            **{f"oxidized_fragment_{topology}": selected[topology]["oxidized_fragment"] for topology in TOPOLOGIES},
            "status": "complete", "note": "",
        })
        output.append(row)
    if len(output) != 24:
        raise AssertionError("unexpected Chauhan composition count")
    return output


def build_fadel_summary(tasks: list[dict[str, str]], references: list[dict[str, str]]) -> list[dict]:
    pairs, triads = _task_maps(tasks, "fadel")
    output = []
    for reference in references:
        anion, solvent = reference["anion"], reference["solvent"]
        pair = pairs[(anion, solvent)]
        selected = {topology: triads[("Li", anion, solvent, topology)] for topology in TOPOLOGIES}
        ips = {"AS": float(pair["ip_vertical_ev"]), **{topology: float(selected[topology]["ip_vertical_ev"]) for topology in TOPOLOGIES}}
        minimum = min(TOPOLOGIES, key=ips.__getitem__)
        ips.update({"min": ips[minimum], "mean": statistics.fmean(ips[topology] for topology in TOPOLOGIES)})
        fadel = float(reference["fadel_ip_dscf_mean_ev"])
        row = {"anion": anion, "solvent": solvent, "fadel_ip_ev": fadel}
        row.update({f"dft_ip_{key}_ev": ips[key] for key in ("AS", *TOPOLOGIES, "min", "mean")})
        row.update({f"error_{key}_ev": ips[key] - fadel for key in ("AS", *TOPOLOGIES, "min", "mean")})
        row.update({
            "topology_of_min": minimum,
            "triad_span_ev": max(ips[topology] for topology in TOPOLOGIES) - min(ips[topology] for topology in TOPOLOGIES),
            "oxidized_fragment_AS": pair["oxidized_fragment"],
            **{f"oxidized_fragment_{topology}": selected[topology]["oxidized_fragment"] for topology in TOPOLOGIES},
            "status": "complete", "note": "",
        })
        output.append(row)
    if len(output) != 16:
        raise AssertionError("unexpected Fadel composition count")
    return output


def build_chauhan_xtb_vs_dft(rows: list[dict], xtb_rows: list[dict[str, str]]) -> list[dict]:
    xtb = {(row["cation"], row["anion"], row["solvent"]): row for row in xtb_rows}
    fields = {"AS": "ip_as_direct_ev", "CAS": "ip_CAS_ev", "CSA": "ip_CSA_ev", "ACS": "ip_ACS_ev", "min": "ip_min_ev", "mean": "ip_mean_ev"}
    output = []
    for row in rows:
        key = (row["cation"], row["anion"], row["solvent"])
        for descriptor, field in fields.items():
            xtb_ip, dft_ip = float(xtb[key][field]), float(row[f"dft_ip_{descriptor}_ev"])
            output.append({
                "cation": key[0], "anion": key[1], "solvent": key[2], "descriptor": "A-S" if descriptor == "AS" else ("(min)" if descriptor == "min" else descriptor),
                "xtb_ip_ev": xtb_ip, "dft_ip_ev": dft_ip, "delta_dft_minus_xtb_ev": dft_ip - xtb_ip,
                "xtb_eox_v": xtb_ip - 4.477, "dft_eox_v": dft_ip - 4.477,
            })
    return output


def build_fadel_xtb_vs_dft(rows: list[dict], xtb_rows: list[dict[str, str]]) -> list[dict]:
    xtb = {(row["anion"], row["solvent"]): row for row in xtb_rows}
    fields = {"AS": "xtb_as_ip_ev", "CAS": "xtb_CAS_ev", "CSA": "xtb_CSA_ev", "ACS": "xtb_ACS_ev", "min": "xtb_triad_min_ev", "mean": "xtb_triad_mean_ev"}
    output = []
    for row in rows:
        key = (row["anion"], row["solvent"])
        for descriptor, field in fields.items():
            xtb_ip, dft_ip = float(xtb[key][field]), float(row[f"dft_ip_{descriptor}_ev"])
            output.append({
                "anion": key[0], "solvent": key[1], "descriptor": "A-S" if descriptor == "AS" else ("(min)" if descriptor == "min" else descriptor),
                "xtb_ip_ev": xtb_ip, "dft_ip_ev": dft_ip, "delta_dft_minus_xtb_ev": dft_ip - xtb_ip,
                "fadel_ip_ev": row["fadel_ip_ev"],
            })
    return output


def build_branch_comparison(rows: list[dict], references: list[dict[str, str]]) -> list[dict]:
    reference = {(row["anion"], row["solvent"]): row for row in references}
    output = []
    for row in rows:
        fadel = reference[(row["anion"], row["solvent"])]
        dominant = fadel["fadel_dominant_oxidized_fragment"]
        item = {
            "anion": row["anion"], "solvent": row["solvent"], "fadel_dominant_oxidized_fragment": dominant,
            "fadel_anion_oxidation_fraction": fadel["anion_oxidation_fraction"],
            "fadel_solvent_oxidation_fraction": fadel["solvent_oxidation_fraction"],
        }
        for key in ("AS", *TOPOLOGIES):
            item[f"dft_{key}_oxidized_fragment"] = row[f"oxidized_fragment_{key}"]
            item[f"{key}_branch_match"] = row[f"oxidized_fragment_{key}"] == dominant
        output.append(item)
    return output


def make_plot(path: Path, rows: list[dict], reference_field: str, predicted: list[float], metric: dict, title: str, xlabel: str, ylabel: str, offset: float | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jhtvs_ft0806_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    anions = list(dict.fromkeys(row["anion"] for row in rows))
    solvents = list(dict.fromkeys(row["solvent"] for row in rows))
    palette = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    marker_list = ["o", "s", "^", "D"]
    colors, markers = dict(zip(anions, palette)), dict(zip(solvents, marker_list))
    observed = [float(row[reference_field]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=180)
    for row, x, y in zip(rows, observed, predicted):
        ax.scatter(x, y, color=colors[row["anion"]], marker=markers[row["solvent"]], s=52, edgecolor="black", linewidth=0.45)
    low, high = min(observed + predicted) - 0.2, max(observed + predicted) + 0.2
    ax.plot([low, high], [low, high], "--", color="#444444", linewidth=1.0)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    mae_field = "MAE" if offset is None else "MAE_after_offset_fit"
    label = f"MAE = {metric[mae_field]:.3f}\nPearson r² = {metric['Pearson_r2_correlation_squared']:.3f}"
    if offset is not None:
        label += f"\noffset-fit = {offset:+.3f}"
    ax.text(0.035, 0.965, label, transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#aaaaaa"})
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="black", label=anion) for anion, color in colors.items()]
    handles += [Line2D([], [], marker=marker, linestyle="none", color="black", markerfacecolor="white", label=solvent) for solvent, marker in markers.items()]
    ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=7.0, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _metric_map(rows: list[dict]) -> dict[str, dict]:
    return {row["descriptor"]: row for row in rows}


def _task_localization(tasks: list[dict[str, str]], benchmark: str) -> Counter:
    return Counter(row["oxidized_fragment"] for row in tasks if row["benchmark"] == benchmark)


def _report(benchmark: str, raw: list[dict], offsets: list[dict], xtb_raw: list[dict], xtb_offsets: list[dict], tasks: list[dict[str, str]], job_ids: str, branches: list[dict] | None = None) -> str:
    raw_by, offset_by, xtb_raw_by, xtb_offset_by = map(_metric_map, (raw, offsets, xtb_raw, xtb_offsets))
    localization = _task_localization(tasks, benchmark)
    units = "V" if benchmark == "chauhan" else "eV"
    lines = [
        f"# {benchmark.title()} M06-HF/aug-cc-pVTZ cluster benchmark", "",
        f"- SGE job IDs: `{job_ids}`.",
        f"- DFT cases complete: {sum(row['benchmark'] == benchmark and row['status'] == 'complete' for row in tasks)}/{81 if benchmark == 'chauhan' else 64}.",
        f"- Oxidation localization across all cases: C/Li={localization['C']}, A={localization['A']}, S={localization['S']}.",
        f"- A-S offset-fit MAE: xTB={xtb_offset_by['A-S']['MAE_after_offset_fit']:.6f} {units}, DFT={offset_by['A-S']['MAE_after_offset_fit']:.6f} {units}.",
        f"- ACS offset-fit MAE: xTB={xtb_offset_by['ACS']['MAE_after_offset_fit']:.6f} {units}, DFT={offset_by['ACS']['MAE_after_offset_fit']:.6f} {units}.",
        f"- A-S raw Pearson/Spearman: xTB={xtb_raw_by['A-S']['Pearson_r']:.6f}/{xtb_raw_by['A-S']['Spearman_rho']:.6f}; DFT={raw_by['A-S']['Pearson_r']:.6f}/{raw_by['A-S']['Spearman_rho']:.6f}.",
        f"- ACS raw Pearson/Spearman: xTB={xtb_raw_by['ACS']['Pearson_r']:.6f}/{xtb_raw_by['ACS']['Spearman_rho']:.6f}; DFT={raw_by['ACS']['Pearson_r']:.6f}/{raw_by['ACS']['Spearman_rho']:.6f}.",
    ]
    if benchmark == "fadel":
        lines.append("- ACS/Fadel survival is reported from the xTB and DFT Pearson/Spearman values above without introducing an additional fitted threshold.")
    if branches is not None:
        for key in ("AS", *TOPOLOGIES):
            matches = sum(row[f"{key}_branch_match"] is True or row[f"{key}_branch_match"] == "True" for row in branches)
            lines.append(f"- Fadel branch agreement {key}: {matches}/16.")
    lines.extend(["", "## Offset-fit metrics", "", "| Descriptor | Offset | MAE | RMSE | r | r² | Spearman rho |", "|---|---:|---:|---:|---:|---:|---:|"])
    for descriptor, _ in DESCRIPTORS:
        row = offset_by[descriptor]
        lines.append(f"| {descriptor} | {row['offset']:.6f} | {row['MAE_after_offset_fit']:.6f} | {row['RMSE_after_offset_fit']:.6f} | {row['Pearson_r']:.6f} | {row['Pearson_r2_correlation_squared']:.6f} | {row['Spearman_rho']:.6f} |")
    lines.append("")
    return "\n".join(lines)


def analyze(root: Path, task_results: Path, job_ids: str) -> None:
    tasks = read_csv(task_results)
    chauhan_dir, fadel_dir = root / "data/chauhan_cation_eox/dft", root / "data/fadel_benchmark/dft"
    chauhan_dir.mkdir(parents=True, exist_ok=True)
    fadel_dir.mkdir(parents=True, exist_ok=True)

    chauhan = build_chauhan_summary(tasks, read_csv(root / "workflows/chauhan_cation_eox/benchmark_chauhan.csv"))
    fadel = build_fadel_summary(tasks, read_csv(root / "data/fadel_benchmark/fadel_table2_reference.csv"))
    write_csv(chauhan_dir / "composition_summary.csv", chauhan, CHAUHAN_SUMMARY_FIELDS)
    write_csv(fadel_dir / "fadel_as_vs_cas.csv", fadel, FADEL_SUMMARY_FIELDS)

    chauhan_raw, chauhan_offsets, chauhan_corrected = metric_tables(chauhan, "eox_exp_v", "dft_eox_")
    fadel_raw, fadel_offsets, fadel_corrected = metric_tables(fadel, "fadel_ip_ev", "dft_ip_")
    write_csv(chauhan_dir / "method_metrics_raw.csv", chauhan_raw, RAW_FIELDS)
    write_csv(chauhan_dir / "method_metrics_offset_fit.csv", chauhan_offsets, OFFSET_FIELDS)
    write_csv(fadel_dir / "method_metrics_raw.csv", fadel_raw, RAW_FIELDS)
    write_csv(fadel_dir / "method_metrics_offset_fit.csv", fadel_offsets, OFFSET_FIELDS)

    chauhan_xtb_source = read_csv(root / "data/chauhan_cation_eox/composition_summary.csv")
    fadel_xtb_source = read_csv(root / "data/fadel_benchmark/fadel_as_vs_cas.csv")
    chauhan_paired = build_chauhan_xtb_vs_dft(chauhan, chauhan_xtb_source)
    fadel_paired = build_fadel_xtb_vs_dft(fadel, fadel_xtb_source)
    write_csv(chauhan_dir / "xtb_vs_dft.csv", chauhan_paired, CHAUHAN_XTB_DFT_FIELDS)
    write_csv(fadel_dir / "xtb_vs_dft.csv", fadel_paired, FADEL_XTB_DFT_FIELDS)

    branches = build_branch_comparison(fadel, read_csv(root / "data/fadel_benchmark/fadel_oxidation_branch_reference.csv"))
    write_csv(fadel_dir / "oxidation_branch_comparison.csv", branches, BRANCH_FIELDS)

    xtb_chauhan_rows = []
    for source in chauhan_xtb_source:
        xtb_chauhan_rows.append({
            "eox_exp_v": source["eox_exp_v"],
            "xtb_eox_AS": float(source["ip_as_direct_ev"]) - 4.477,
            **{f"xtb_eox_{key}": float(source[field]) - 4.477 for key, field in (("CAS", "ip_CAS_ev"), ("CSA", "ip_CSA_ev"), ("ACS", "ip_ACS_ev"), ("min", "ip_min_ev"), ("mean", "ip_mean_ev"))},
        })
    xtb_fadel_rows = []
    for source in fadel_xtb_source:
        xtb_fadel_rows.append({
            "fadel_ip_ev": source["fadel_ip_dscf_mean_ev"], "xtb_ip_AS": source["xtb_as_ip_ev"],
            "xtb_ip_CAS": source["xtb_CAS_ev"], "xtb_ip_CSA": source["xtb_CSA_ev"], "xtb_ip_ACS": source["xtb_ACS_ev"],
            "xtb_ip_min": source["xtb_triad_min_ev"], "xtb_ip_mean": source["xtb_triad_mean_ev"],
        })
    xtb_chauhan_raw, xtb_chauhan_offsets, _ = metric_tables(xtb_chauhan_rows, "eox_exp_v", "xtb_eox_")
    xtb_fadel_raw, xtb_fadel_offsets, _ = metric_tables(xtb_fadel_rows, "fadel_ip_ev", "xtb_ip_")

    offset_by_chauhan = _metric_map(chauhan_offsets)
    raw_by_fadel, offset_by_fadel = _metric_map(fadel_raw), _metric_map(fadel_offsets)
    for descriptor, key in DESCRIPTORS[:5]:
        make_plot(
            chauhan_dir / f"eox_vs_dft_{key}_offset_fit.png", chauhan, "eox_exp_v", chauhan_corrected[key], offset_by_chauhan[descriptor],
            f"Chauhan Eox vs DFT {descriptor} (offset-fit)", "Experimental Eox (V vs Ag/AgCl)", "DFT Eox (V vs Ag/AgCl, offset-fit)", float(offset_by_chauhan[descriptor]["offset"]),
        )
        raw_values = [float(row[f"dft_ip_{key}_ev"]) for row in fadel]
        make_plot(
            fadel_dir / f"fadel_vs_dft_{key}_raw.png", fadel, "fadel_ip_ev", raw_values, raw_by_fadel[descriptor],
            f"Fadel vs DFT {descriptor} (raw)", "Fadel M06-HF mean vertical IP (eV)", "DFT vertical IP (eV)",
        )
        make_plot(
            fadel_dir / f"fadel_vs_dft_{key}_offset_fit.png", fadel, "fadel_ip_ev", fadel_corrected[key], offset_by_fadel[descriptor],
            f"Fadel vs DFT {descriptor} (offset-fit)", "Fadel M06-HF mean vertical IP (eV)", "DFT vertical IP (eV, offset-fit)", float(offset_by_fadel[descriptor]["offset"]),
        )

    (chauhan_dir / "report.md").write_text(_report("chauhan", chauhan_raw, chauhan_offsets, xtb_chauhan_raw, xtb_chauhan_offsets, tasks, job_ids), encoding="utf-8")
    (fadel_dir / "report.md").write_text(_report("fadel", fadel_raw, fadel_offsets, xtb_fadel_raw, xtb_fadel_offsets, tasks, job_ids, branches), encoding="utf-8")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-results", type=Path, default=root / "data/dft_cluster_benchmark/task_results.csv")
    parser.add_argument("--job-ids", default="not recorded")
    args = parser.parse_args()
    analyze(root, args.task_results.resolve(), args.job_ids)
    print("wrote Chauhan and Fadel DFT benchmark analyses")


if __name__ == "__main__":
    main()
