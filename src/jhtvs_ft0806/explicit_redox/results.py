from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .alignment import assert_disjoint_keys, fit_global_intercept


CHECKPOINT_SHA256 = "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b"
PROTOCOL_ID = "mace-polar-5solv-redox-v1"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fit_reference(
    *, workflow_dir: Path, raw_root: Path, results_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration = _read(workflow_dir / "calibration_manifest.csv")
    validation = _read(workflow_dir / "validation_manifest.csv")
    assert_disjoint_keys(
        [row["canonical_key"] for row in calibration],
        [row["canonical_key"] for row in validation],
    )
    raw = {
        row["system_id"]: row for row in _read(raw_root / "calibration_system_raw_predictions.csv")
    }
    if set(raw) != {row["system_id"] for row in calibration}:
        raise RuntimeError("calibration raw prediction scope differs from frozen manifest")
    experimental = [float(row["experimental_value_V_vs_AgAgCl"]) for row in calibration]
    raw_values = [float(raw[row["system_id"]]["raw_voltage_V"]) for row in calibration]
    fit = fit_global_intercept(experimental, raw_values)
    rows: list[dict[str, Any]] = []
    for manifest, experiment, raw_value, residual in zip(
        calibration, experimental, raw_values, fit.residuals_V, strict=True
    ):
        rows.append(
            {
                **manifest,
                "delta_F_ox_eV": raw[manifest["system_id"]]["delta_F_ox_eV"],
                "raw_voltage_V": raw_value,
                "C_model_V": fit.C_model_V,
                "Eox_vs_AgAgCl_V": raw_value + fit.C_model_V,
                "residual_V": residual,
            }
        )
    payload = {
        "status": "FROZEN",
        "alignment_type": "single_global_intercept",
        "slope": 1.0,
        "reference": "Ag/AgCl",
        "calibration_system_count": len(rows),
        "calibration_system_ids": [row["system_id"] for row in rows],
        "C_model_V": fit.C_model_V,
        "mae_before_V": fit.mae_before_V,
        "mae_after_V": fit.mae_after_V,
        "validation_excluded": True,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    _write(results_dir / "calibration_predictions.csv", rows)
    (results_dir / "reference_alignment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload, rows


def _aggregate_qc(rows: Sequence[dict[str, str]]) -> tuple[str, str, str]:
    flags = sorted(
        {
            flag
            for row in rows
            for flag in row["flags"].split(";")
            if flag and flag != "clean"
        }
    )
    localizations = {row["localization_flag"] for row in rows}
    localization = next(iter(localizations)) if len(localizations) == 1 else "localization_ambiguous"
    return ("clean" if not flags else "flagged", ";".join(flags) if flags else "clean", localization)


def _metrics(experimental: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = predicted - experimental
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((experimental - experimental.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator else float("nan")
    return r2, mae


def _execution_provenance(raw_root: Path, scope: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("trajectory", "gap"):
        path = raw_root / "submissions" / f"{scope}_{stage}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != "SUBMITTED":
            raise RuntimeError(f"{scope} {stage} execution ledger is not final")
        result[stage] = {
            "repository_commit": payload["repository_commit"],
            "task_table_sha256": payload["task_table_sha256"],
            "scheduler_job_ids": payload["scheduler_job_ids"],
            "device": payload["device"],
        }
    return result


def assemble_validation_results(
    *, workflow_dir: Path, raw_root: Path, results_dir: Path
) -> dict[str, Any]:
    manifest = _read(workflow_dir / "validation_manifest.csv")
    if len(manifest) != 21:
        raise RuntimeError("validation result assembly requires exactly 21 frozen systems")
    raw_system = {
        row["system_id"]: row for row in _read(raw_root / "validation_system_raw_predictions.csv")
    }
    if set(raw_system) != {row["system_id"] for row in manifest}:
        raise RuntimeError("validation raw prediction scope differs from frozen manifest")
    seed_rows = _read(raw_root / "validation_seed_gap_summary.csv")
    qc_rows = _read(raw_root / "validation_trajectory_qc.csv")
    clusters = _read(raw_root / "validation_cluster_manifest.csv")
    alignment = json.loads((results_dir / "reference_alignment.json").read_text(encoding="utf-8"))
    if alignment["status"] != "FROZEN" or alignment["slope"] != 1.0:
        raise RuntimeError("reference alignment is not a frozen intercept-only fit")
    constant = float(alignment["C_model_V"])
    systems: list[dict[str, Any]] = []
    seeds_output: list[dict[str, Any]] = []
    qc_output: list[dict[str, Any]] = []
    for system in manifest:
        system_id = system["system_id"]
        raw = raw_system[system_id]
        system_seeds = sorted(
            (row for row in seed_rows if row["system_id"] == system_id),
            key=lambda row: int(row["seed_index"]),
        )
        system_qc = [row for row in qc_rows if row["system_id"] == system_id]
        system_clusters = [row for row in clusters if row["system_id"] == system_id]
        if len(system_seeds) != 5 or len(system_qc) != 10 or len(system_clusters) != 5:
            raise RuntimeError(f"incomplete five-seed result scope: {system_id}")
        qc_status, flags, localization = _aggregate_qc(system_qc)
        lambda_value = float(raw["lambda_eV"])
        if lambda_value < 0.0:
            flags = ";".join(sorted(set(flags.split(";")) | {"negative_lambda"}))
            qc_status = "flagged"
        atom_count = int(system_clusters[0]["target_atoms"]) + 5 * int(
            system_clusters[0]["solvent_atoms"]
        )
        systems.append(
            {
                "system_id": system_id,
                "class": system["class"],
                "species_id": system["species_id"],
                "species_name": system["species_name"],
                "canonical_smiles": system["canonical_smiles"],
                "solvent_id": system["solvent_id"],
                "solvent_name": system["solvent_name"],
                "solvent_canonical_smiles": system["solvent_canonical_smiles"],
                "lower_charge": system["lower_charge"],
                "lower_spin": system["lower_spin"],
                "oxidized_charge": system["oxidized_charge"],
                "oxidized_spin": system["oxidized_spin"],
                "atom_count": atom_count,
                "shell_seed_ids": system["shell_seed_ids"],
                "mu_lower_eV": float(np.mean([float(row["mu_lower_eV"]) for row in system_seeds])),
                "mu_oxidized_eV": float(
                    np.mean([float(row["mu_oxidized_eV"]) for row in system_seeds])
                ),
                "delta_F_ox_eV": raw["delta_F_ox_eV"],
                "lambda_eV": raw["lambda_eV"],
                "raw_voltage_V": raw["raw_voltage_V"],
                "C_model_V": constant,
                "Eox_vs_AgAgCl_V": float(raw["raw_voltage_V"]) + constant,
                "shell_seed_sd_eV": raw["shell_seed_sd_eV"],
                "shell_seed_sem_eV": raw["shell_seed_sem_eV"],
                "within_seed_block_se_eV": raw["within_seed_block_se_eV"],
                "qc_status": qc_status,
                "qc_flags": flags,
                "localization_flag": localization,
                "model": "MACE-POLAR-1-L",
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "protocol_id": PROTOCOL_ID,
            }
        )
        for seed in system_seeds:
            seed_qc = [row for row in system_qc if row["seed_index"] == seed["seed_index"]]
            seed_status, seed_flags, seed_localization = _aggregate_qc(seed_qc)
            seeds_output.append(
                {
                    **seed,
                    "qc_status": seed_status,
                    "qc_flags": seed_flags,
                    "localization_flag": seed_localization,
                }
            )
        qc_output.extend(system_qc)
    observations = _read(workflow_dir / "validation_observations.csv")
    by_system = {row["system_id"]: row for row in systems}
    observation_rows = [
        {
            **observation,
            "raw_voltage_V": by_system[observation["system_id"]]["raw_voltage_V"],
            "C_model_V": constant,
            "Eox_vs_AgAgCl_V": by_system[observation["system_id"]]["Eox_vs_AgAgCl_V"],
            "qc_status": by_system[observation["system_id"]]["qc_status"],
            "localization_flag": by_system[observation["system_id"]]["localization_flag"],
        }
        for observation in observations
    ]
    metric_rows: list[dict[str, Any]] = []
    for class_name in ("all", "monomer", "solvent", "anion"):
        selected = observation_rows if class_name == "all" else [
            row for row in observation_rows if row["class"] == class_name
        ]
        experimental = np.asarray(
            [float(row["experimental_value_V_vs_AgAgCl"]) for row in selected]
        )
        for scale, field in (("raw", "raw_voltage_V"), ("aligned", "Eox_vs_AgAgCl_V")):
            predicted = np.asarray([float(row[field]) for row in selected])
            r2, mae = _metrics(experimental, predicted)
            metric_rows.append(
                {"scope": "observation", "class": class_name, "scale": scale, "n": len(selected), "r2": r2, "mae_V": mae}
            )
    _write(results_dir / "system_predictions.csv", systems)
    _write(results_dir / "observation_predictions.csv", observation_rows)
    _write(results_dir / "seed_predictions.csv", seeds_output)
    _write(results_dir / "qc.csv", qc_output)
    _write(results_dir / "metrics.csv", metric_rows)
    provenance = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_system_count": len(systems),
        "validation_observation_count": len(observation_rows),
        "calibration_system_count": alignment["calibration_system_count"],
        "reference_alignment_sha256": hashlib.sha256(
            (results_dir / "reference_alignment.json").read_bytes()
        ).hexdigest(),
        "execution": {
            "calibration": _execution_provenance(raw_root, "calibration"),
            "validation": _execution_provenance(raw_root, "validation"),
        },
    }
    (results_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def plot_results(*, results_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read(results_dir / "observation_predictions.csv")
    metric_rows = _read(results_dir / "metrics.csv")
    paths = []
    for class_name in ("monomer", "solvent", "anion"):
        selected = [row for row in rows if row["class"] == class_name]
        x = np.asarray([float(row["experimental_value_V_vs_AgAgCl"]) for row in selected])
        raw = np.asarray([float(row["raw_voltage_V"]) for row in selected])
        aligned = np.asarray([float(row["Eox_vs_AgAgCl_V"]) for row in selected])
        lower = float(min(x.min(), raw.min(), aligned.min()) - 0.15)
        upper = float(max(x.max(), raw.max(), aligned.max()) + 0.15)
        figure, axis = plt.subplots(figsize=(5.4, 5.0))
        axis.scatter(x, raw, marker="x", color="0.55", label="raw ΔF/e")
        axis.scatter(x, aligned, color="#1769aa", label="aligned vs Ag/AgCl")
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="y = x")
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, aligned, 1)
            axis.plot([lower, upper], slope * np.asarray([lower, upper]) + intercept, color="#d95f02", linewidth=1, label="descriptive fit")
        raw_metric = next(
            row for row in metric_rows if row["class"] == class_name and row["scale"] == "raw"
        )
        aligned_metric = next(
            row for row in metric_rows if row["class"] == class_name and row["scale"] == "aligned"
        )
        axis.text(
            0.03,
            0.97,
            f"n={len(x)}\nraw: R²={float(raw_metric['r2']):.3f}, MAE={float(raw_metric['mae_V']):.3f} V\naligned: R²={float(aligned_metric['r2']):.3f}, MAE={float(aligned_metric['mae_V']):.3f} V",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Experiment Eox (V vs Ag/AgCl)", ylabel="MACE-POLAR-1 5-solvent Eox (V)", title=f"{class_name.capitalize()} validation")
        axis.legend(loc="lower right", fontsize=8)
        figure.tight_layout()
        path = results_dir / f"{class_name}_validation.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit-reference", "assemble-results"):
        command = sub.add_parser(name)
        command.add_argument("--workflow-dir", type=Path, required=True)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--results-dir", type=Path, required=True)
    plot = sub.add_parser("plot-results")
    plot.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fit-reference":
        payload, _ = fit_reference(
            workflow_dir=args.workflow_dir, raw_root=args.raw_root, results_dir=args.results_dir
        )
    elif args.command == "assemble-results":
        payload = assemble_validation_results(
            workflow_dir=args.workflow_dir, raw_root=args.raw_root, results_dir=args.results_dir
        )
    else:
        paths = plot_results(results_dir=args.results_dir)
        payload = {"status": "PASS", "plots": [path.as_posix() for path in paths]}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
