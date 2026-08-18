#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import read_csv, repo_root, write_csv

CATION_ORDER = ("EMIM", "BMIM", "HMIM", "BMPYRR")
DESCRIPTORS = (
    ("fadel_as_zero", None),
    ("isolated_cation", "ip_cation_ev"),
    ("CAS", "ip_CAS_ev"),
    ("CSA", "ip_CSA_ev"),
    ("ACS", "ip_ACS_ev"),
    ("triad_min", "ip_min_ev"),
    ("triad_mean", "ip_mean_ev"),
)


def _f(value: str | float) -> float:
    return float(value)


def _fmt(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.12g}"


def _rank(values: list[float]) -> list[float]:
    comparable = [round(value, 12) for value in values]
    order = sorted(range(len(values)), key=comparable.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and comparable[order[end]] == comparable[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    dl = [value - mean_left for value in left]
    dr = [value - mean_right for value in right]
    denominator = math.sqrt(sum(value * value for value in dl) * sum(value * value for value in dr))
    return None if denominator == 0 else sum(a * b for a, b in zip(dl, dr)) / denominator


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(_rank(left), _rank(right))


def metric_values(observed: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    errors = [prediction - actual for actual, prediction in zip(observed, predicted)]
    signs = lambda value: 0 if value == 0 else (1 if value > 0 else -1)
    return {
        "n": len(errors),
        "mae": statistics.fmean(abs(value) for value in errors),
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "pearson_r": pearson(observed, predicted),
        "spearman_rho": spearman(observed, predicted),
        "sign_agreement": statistics.fmean(signs(actual) == signs(prediction) for actual, prediction in zip(observed, predicted)),
    }


def build_sensitivity(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["anion"], row["solvent"])].append(row)
    output = []
    for (anion, solvent), group in grouped.items():
        output.append({
            "anion": anion,
            "solvent": solvent,
            "n_cations": len(group),
            "experimental_eox_spread_v": max(_f(row["eox_exp_v"]) for row in group) - min(_f(row["eox_exp_v"]) for row in group),
            "ip_cation_only_spread_ev": max(_f(row["ip_cation_ev"]) for row in group) - min(_f(row["ip_cation_ev"]) for row in group),
            "ip_CAS_spread_ev": max(_f(row["ip_CAS_ev"]) for row in group) - min(_f(row["ip_CAS_ev"]) for row in group),
            "ip_CSA_spread_ev": max(_f(row["ip_CSA_ev"]) for row in group) - min(_f(row["ip_CSA_ev"]) for row in group),
            "ip_ACS_spread_ev": max(_f(row["ip_ACS_ev"]) for row in group) - min(_f(row["ip_ACS_ev"]) for row in group),
            "ip_min_spread_ev": max(_f(row["ip_min_ev"]) for row in group) - min(_f(row["ip_min_ev"]) for row in group),
            "ip_mean_spread_ev": max(_f(row["ip_mean_ev"]) for row in group) - min(_f(row["ip_mean_ev"]) for row in group),
            "mean_triad_topology_span_ev": statistics.fmean(_f(row["ip_span_ev"]) for row in group),
        })
    return output


def build_pairwise(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["anion"], row["solvent"])].append(row)
    output = []
    for (anion, solvent), group in grouped.items():
        by_cation = {row["cation"]: row for row in group}
        ordered = [cation for cation in CATION_ORDER if cation in by_cation]
        for left, right in itertools.combinations(ordered, 2):
            a, b = by_cation[left], by_cation[right]
            experimental = _f(a["eox_exp_v"]) - _f(b["eox_exp_v"])
            out = {
                "anion": anion, "solvent": solvent, "cation_i": left, "cation_j": right,
                "delta_eox_exp_v": experimental,
                "delta_eox_sd_rss_v": math.hypot(_f(a["eox_sd_v"]), _f(b["eox_sd_v"])),
            }
            for label, field in DESCRIPTORS:
                predicted = 0.0 if field is None else _f(a[field]) - _f(b[field])
                out[f"delta_{label}_ev"] = predicted
                out[f"sign_match_{label}"] = (0 if experimental == 0 else (1 if experimental > 0 else -1)) == (0 if predicted == 0 else (1 if predicted > 0 else -1))
            output.append(out)
    return output


def _centered_vectors(rows: list[dict[str, str]], field: str) -> tuple[list[float], list[float]]:
    observed, predicted = [], []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["anion"], row["solvent"])].append(row)
    for group in grouped.values():
        exp_mean = statistics.fmean(_f(row["eox_exp_v"]) for row in group)
        pred_mean = statistics.fmean(_f(row[field]) for row in group)
        observed.extend(_f(row["eox_exp_v"]) - exp_mean for row in group)
        predicted.extend(_f(row[field]) - pred_mean for row in group)
    return observed, predicted


def build_metrics(rows: list[dict[str, str]], contrasts: list[dict]) -> list[dict]:
    output = []
    observed = [_f(row["delta_eox_exp_v"]) for row in contrasts]
    for label, field in DESCRIPTORS:
        predicted = [_f(row[f"delta_{label}_ev"]) for row in contrasts]
        metrics = metric_values(observed, predicted)
        if field is None:
            centered = {"n": len(rows), "pearson_r": None, "spearman_rho": None}
        else:
            centered_observed, centered_predicted = _centered_vectors(rows, field)
            centered = metric_values(centered_observed, centered_predicted)
            for group_key in {(row["anion"], row["solvent"]) for row in rows}:
                group = [row for row in rows if (row["anion"], row["solvent"]) == group_key]
                assert abs(sum(_f(row[field]) - statistics.fmean(_f(item[field]) for item in group) for row in group)) < 1e-10
        output.append({
            "descriptor": label,
            "n_pairwise": metrics["n"],
            "mae_pairwise_v": _fmt(metrics["mae"]),
            "rmse_pairwise_v": _fmt(metrics["rmse"]),
            "pearson_pairwise_r": _fmt(metrics["pearson_r"]),
            "spearman_pairwise_rho": _fmt(metrics["spearman_rho"]),
            "sign_agreement_pairwise": _fmt(metrics["sign_agreement"]),
            "n_centered_compositions": centered["n"],
            "pearson_centered_r": _fmt(centered["pearson_r"]),
            "spearman_centered_rho": _fmt(centered["spearman_rho"]),
        })
    return output


def build_localization(triads: list[dict[str, str]]) -> list[dict]:
    dimensions = (
        ("overall", ()),
        ("topology", ("topology",)),
        ("anion", ("anion",)),
        ("solvent", ("solvent",)),
        ("topology_anion_solvent", ("topology", "anion", "solvent")),
    )
    output = []
    for scope, keys in dimensions:
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in triads:
            groups[tuple(row[key] for key in keys)].append(row)
        for values, group in groups.items():
            labels = dict(zip(keys, values))
            counts = Counter(row["oxidized_fragment"] for row in group)
            for fragment in ("C", "A", "S"):
                output.append({
                    "scope": scope,
                    "topology": labels.get("topology", "all"),
                    "anion": labels.get("anion", "all"),
                    "solvent": labels.get("solvent", "all"),
                    "oxidized_fragment": fragment,
                    "count": counts[fragment],
                    "total": len(group),
                    "fraction": counts[fragment] / len(group),
                })
    return output


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def build_report(rows: list[dict[str, str]], triads: list[dict[str, str]], sensitivity: list[dict], metrics: list[dict], job_ids: str) -> str:
    spans = [_f(row["ip_span_ev"]) for row in rows]
    preserved = sum(row["topology_preserved"] == "True" for row in triads)
    top = sorted(rows, key=lambda row: _f(row["ip_span_ev"]), reverse=True)[:10]
    best = min((row for row in metrics if row["descriptor"] != "fadel_as_zero"), key=lambda row: _f(row["mae_pairwise_v"]))
    baseline = next(row for row in metrics if row["descriptor"] == "fadel_as_zero")
    localization = Counter(row["oxidized_fragment"] for row in triads)
    lines = [
        "# Chauhan cation-effect full-run report", "",
        f"- SGE production job IDs: `{job_ids}`.",
        f"- Complete evidence: {len(triads)}/72 triads, {len(rows)}/24 compositions; same-geometry validation passed during parsing for every retained task.",
        f"- Requested topology preserved: {preserved}/72 ({preserved / 72:.1%}).",
        f"- Oxidation localization overall: C={localization['C']}, A={localization['A']}, S={localization['S']}.", "",
        "## Does cation identity measurably affect oxidation?", "",
        "The within-anion/solvent experimental and calculated spreads are listed below. These are direct group contrasts; no fitted calibration or mechanistic assignment was introduced.", "",
        "| anion | solvent | n | experimental spread (V) | cation-only IP spread (eV) | CAS | CSA | ACS | min | mean | mean topology span |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append("| {anion} | {solvent} | {n_cations} | {experimental_eox_spread_v:.4f} | {ip_cation_only_spread_ev:.4f} | {ip_CAS_spread_ev:.4f} | {ip_CSA_spread_ev:.4f} | {ip_ACS_spread_ev:.4f} | {ip_min_spread_ev:.4f} | {ip_mean_spread_ev:.4f} | {mean_triad_topology_span_ev:.4f} |".format(**row))
    lines.extend([
        "", "## Descriptor comparison on 21 within-group cation contrasts", "",
        "| descriptor | MAE (V) | RMSE (V) | Pearson r | Spearman rho | sign agreement | centered Pearson r | centered Spearman rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metrics:
        lines.append(f"| {row['descriptor']} | {row['mae_pairwise_v']} | {row['rmse_pairwise_v']} | {row['pearson_pairwise_r'] or 'NA'} | {row['spearman_pairwise_rho'] or 'NA'} | {row['sign_agreement_pairwise']} | {row['pearson_centered_r'] or 'NA'} | {row['spearman_centered_rho'] or 'NA'} |")
    improvement = _f(baseline["mae_pairwise_v"]) - _f(best["mae_pairwise_v"])
    lines.extend([
        "", f"Lowest pairwise MAE among calculated descriptors: **{best['descriptor']}**, {best['mae_pairwise_v']} V; improvement over the Fadel A-S zero-contrast baseline: {improvement:.6f} V.", "",
        "## Topology diagnostics", "",
        f"Composition topology-span distribution (eV): min={min(spans):.6f}, Q1={_quantile(spans, 0.25):.6f}, median={statistics.median(spans):.6f}, mean={statistics.fmean(spans):.6f}, Q3={_quantile(spans, 0.75):.6f}, max={max(spans):.6f}.", "",
        "Ten largest topology spans:", "",
        "| rank | cation | anion | solvent | span (eV) | min topology |",
        "|---:|---|---|---|---:|---|",
    ])
    for index, row in enumerate(top, 1):
        lines.append(f"| {index} | {row['cation']} | {row['anion']} | {row['solvent']} | {_f(row['ip_span_ev']):.6f} | {row['topology_of_min_ip']} |")
    lines.append("")
    return "\n".join(lines)


def analyze(summary_path: Path, triad_path: Path, output_dir: Path, job_ids: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows = read_csv(summary_path)
    triads = read_csv(triad_path)
    if len(rows) != 24 or any(row["status"] != "complete" for row in rows):
        raise ValueError("analysis requires 24 complete composition rows")
    if len(triads) != 72 or any(row["status"] != "complete" for row in triads):
        raise ValueError("analysis requires 72 complete triad rows")
    sensitivity = build_sensitivity(rows)
    contrasts = build_pairwise(rows)
    metrics = build_metrics(rows, contrasts)
    localization = build_localization(triads)
    if len(sensitivity) != 9 or len(contrasts) != 21:
        raise AssertionError("unexpected Chauhan grouping cardinality")
    write_csv(output_dir / "cation_sensitivity_by_AS.csv", sensitivity, list(sensitivity[0]))
    write_csv(output_dir / "cation_pairwise_contrasts.csv", contrasts, list(contrasts[0]))
    write_csv(output_dir / "cation_descriptor_metrics.csv", metrics, list(metrics[0]))
    write_csv(output_dir / "oxidation_localization_summary.csv", localization, list(localization[0]))
    (output_dir / "full_run_report.md").write_text(build_report(rows, triads, sensitivity, metrics, job_ids), encoding="utf-8")
    return sensitivity, contrasts, metrics, localization


def main() -> None:
    root = repo_root()
    data = root / "data" / "chauhan_cation_eox"
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=data / "composition_summary.csv")
    parser.add_argument("--triads", type=Path, default=data / "triad_results.csv")
    parser.add_argument("--output-dir", type=Path, default=data)
    parser.add_argument("--job-ids", default="not recorded")
    args = parser.parse_args()
    sensitivity, contrasts, metrics, localization = analyze(args.summary, args.triads, args.output_dir, args.job_ids)
    print(f"wrote {len(sensitivity)} sensitivity rows, {len(contrasts)} contrasts, {len(metrics)} metric rows, and {len(localization)} localization rows")


if __name__ == "__main__":
    main()
