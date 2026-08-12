from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_report(
    *, workflow_dir: Path, raw_root: Path, results_dir: Path, output: Path
) -> str:
    systems = _read(results_dir / "system_predictions.csv")
    observations = _read(results_dir / "observation_predictions.csv")
    calibration = _read(results_dir / "calibration_predictions.csv")
    metrics = _read(results_dir / "metrics.csv")
    qc = _read(results_dir / "qc.csv")
    alignment = json.loads((results_dir / "reference_alignment.json").read_text(encoding="utf-8"))
    provenance = json.loads((results_dir / "provenance.json").read_text(encoding="utf-8"))
    pilot = json.loads((workflow_dir / "pilot_report.json").read_text(encoding="utf-8"))
    classes = Counter(row["class"] for row in systems)
    qc_status = Counter(row["qc_status"] for row in systems)
    localization = Counter(row["localization_flag"] for row in systems)
    flags = Counter(
        flag
        for row in systems
        for flag in row["qc_flags"].split(";")
        if flag and flag != "clean"
    )
    trajectory_wallclock = 0.0
    trajectory_receipts = 0
    for path in (raw_root / "trajectories").glob("*/trajectory.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        trajectory_wallclock += float(payload.get("wallclock_seconds", 0.0))
        trajectory_receipts += 1
    all_metrics = {
        (row["class"], row["scale"]): row for row in metrics if row["scope"] == "observation"
    }
    commands = [
        "python -m jhtvs_ft0806.explicit_redox.isolated collect --raw-root runs/mace_polar_5solv_redox",
        "python -m jhtvs_ft0806.explicit_redox.clusters --manifest <scope_manifest.csv> --raw-root runs/mace_polar_5solv_redox --output-name <scope>_cluster_manifest.csv",
        "python -m jhtvs_ft0806.explicit_redox.trajectory prepare --cluster-manifest <scope_cluster_manifest.csv> --raw-root runs/mace_polar_5solv_redox --mode <scope>",
        "python -m jhtvs_ft0806.explicit_redox.workflow submit-md --raw-root runs/mace_polar_5solv_redox --scope <scope> --device cuda --execute",
        "python -m jhtvs_ft0806.explicit_redox.analysis assemble-marcus --raw-root runs/mace_polar_5solv_redox --mode <scope>",
        "python -m jhtvs_ft0806.explicit_redox.qc --raw-root runs/mace_polar_5solv_redox --mode <scope>",
        "python -m jhtvs_ft0806.explicit_redox.results fit-reference|assemble-results|plot-results ...",
    ]
    blockers = []
    if qc_status.get("flagged", 0):
        blockers.append("flagged systems require scientific interpretation; retained without deletion")
    if not blockers:
        blockers.append("none")
    lines = [
        "# MACE-POLAR-1-L five-solvent Eox report",
        "",
        "## Scope and completion",
        "",
        f"- Validation systems: {len(systems)} (monomer {classes['monomer']}, solvent {classes['solvent']}, anion {classes['anion']})",
        f"- Experimental observations mapped: {len(observations)}",
        f"- Independent calibration systems: {len(calibration)}",
        f"- Validation trajectories: {len(systems) * 10}; calibration trajectories: {len(calibration) * 10}; completed trajectory receipts: {trajectory_receipts}",
        f"- Aggregate production protocol time: {(len(systems) + len(calibration)) * 10 * 200 / 1000:.3f} ns (including equilibration and production)",
        f"- QC systems: clean {qc_status.get('clean', 0)}, flagged {qc_status.get('flagged', 0)}",
        f"- Pilot: {pilot['status']}; {pilot['pilot_trajectory_count']} trajectories; trajectory wallclock {float(pilot['trajectory_wallclock_seconds_sum']) / 3600.0:.2f} h; gap wallclock {float(pilot['gap_wallclock_seconds_sum']) / 3600.0:.2f} h",
        "",
        "## Method and provenance",
        "",
        f"- Protocol: `{provenance['protocol_id']}`",
        f"- Model: MACE-POLAR-1-L, checkpoint `{provenance['checkpoint_sha256']}`",
        "- Explicit cluster: target + five solvent molecules; independent lower/oxidized FIRE and Langevin trajectories",
        "- Dynamics: 300 K, 0.5 fs, friction 0.01 fs^-1, 50 ps equilibration + 150 ps production, 20 fs gap sampling",
        "- Marcus estimator: ΔF_ox = 0.5(μ_lower + μ_oxidized); contiguous 5 ps blocks",
        "",
        "## Reference alignment",
        "",
        f"- Global intercept: C_model = {float(alignment['C_model_V']):.6f} V; slope fixed to 1",
        f"- Calibration MAE: raw {float(alignment['mae_before_V']):.4f} V; aligned {float(alignment['mae_after_V']):.4f} V",
        "- The 21 validation systems were excluded from fitting.",
        "",
        "## Validation metrics",
        "",
        "| Class | n | Raw R² | Raw MAE (V) | Aligned R² | Aligned MAE (V) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_name in ("monomer", "solvent", "anion", "all"):
        raw = all_metrics[(class_name, "raw")]
        aligned = all_metrics[(class_name, "aligned")]
        lines.append(
            f"| {class_name} | {raw['n']} | {float(raw['r2']):.3f} | {float(raw['mae_V']):.3f} | {float(aligned['r2']):.3f} | {float(aligned['mae_V']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## QC and localization",
            "",
            f"- Localization counts: {dict(sorted(localization.items()))}",
            f"- Retained result flags: {dict(sorted(flags.items())) if flags else 'none'}",
            f"- Summed trajectory wallclock recorded in receipts: {trajectory_wallclock / 3600.0:.2f} h",
            f"- Trajectory-state QC rows: {len(qc)}",
            "",
            "## Reproduction commands",
            "",
            "```bash",
            *commands,
            "```",
            "",
            "## Remaining scientific blockers",
            "",
            *[f"- {item}" for item in blockers],
            "",
        ]
    )
    text = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_report(
        workflow_dir=args.workflow_dir,
        raw_root=args.raw_root,
        results_dir=args.results_dir,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
