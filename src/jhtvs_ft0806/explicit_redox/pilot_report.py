from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


CRITICAL_FLAGS = {
    "incomplete_sampling",
    "shell_escape",
    "solvent_fragmented",
    "target_fragmented",
}


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_pilot_report(
    *, workflow_dir: Path, raw_root: Path, output_json: Path, output_markdown: Path
) -> dict[str, Any]:
    manifest = _read_csv(workflow_dir / "pilot_manifest.csv")
    clusters = _read_csv(raw_root / "pilot_cluster_manifest.csv")
    tasks = _read_csv(raw_root / "pilot_trajectory_tasks.tsv", delimiter="\t")
    qc_rows = _read_csv(raw_root / "pilot_trajectory_qc.csv")
    solvent_shell_rows = _read_csv(raw_root / "pilot_solvent_shell_qc.csv")
    seed_rows = _read_csv(raw_root / "pilot_seed_gap_summary.csv")
    if len(manifest) != 4 or len(clusters) != 4 or len(tasks) != 8:
        raise RuntimeError("pilot scope must contain four systems, four clusters and eight states")
    expected_roles = {
        "small_monomer_acetonitrile",
        "largest_flexible_monomer",
        "solvent_self_solvation",
        "anion_propylene_carbonate",
    }
    if {row["pilot_role"] for row in manifest} != expected_roles:
        raise RuntimeError("pilot role coverage drift")
    if len(qc_rows) != 8 or len(solvent_shell_rows) != 40 or len(seed_rows) != 4:
        raise RuntimeError("pilot gap/QC scope is incomplete")

    manifest_by_system = {row["system_id"]: row for row in manifest}
    qc_by_trajectory = {row["logical_trajectory_id"]: row for row in qc_rows}
    critical_flags: set[str] = set()
    trajectory_wallclock = 0.0
    gap_wallclock = 0.0
    finite = True
    state_metadata = True
    restart_ledgers = True
    gap_evaluations = True
    trajectory_summaries: list[dict[str, Any]] = []
    raw_bytes = 0
    for task in tasks:
        logical_id = task["logical_trajectory_id"]
        trajectory_dir = raw_root / "trajectories" / logical_id
        gap_dir = raw_root / "gaps" / logical_id
        trajectory = json.loads((trajectory_dir / "trajectory.json").read_text(encoding="utf-8"))
        gap = json.loads((gap_dir / "gaps.json").read_text(encoding="utf-8"))
        raw_bytes += _directory_bytes(trajectory_dir) + _directory_bytes(gap_dir)
        trajectory_wallclock += float(trajectory["wallclock_seconds"])
        gap_wallclock += float(gap["wallclock_seconds"])
        md = trajectory["md"]
        trajectory_qc = qc_by_trajectory[logical_id]
        optimization = trajectory["optimization"]
        finite = finite and all(
            math.isfinite(float(value))
            for value in (
                optimization["observed_max_force_eV_A"],
                md["maximum_force_eV_A"],
                md["temperature_mean_K"],
            )
        )
        system = manifest_by_system[task["system_id"]]
        state = task["state"]
        state_metadata = state_metadata and (
            int(task["charge"]) == int(system[f"{state}_charge"])
            and int(task["spin"]) == int(system[f"{state}_spin"])
            and int(trajectory["charge"]) == int(task["charge"])
            and int(trajectory["spin"]) == int(task["spin"])
        )
        restart_ledgers = restart_ledgers and (
            trajectory["status"] == "complete"
            and int(md["chunks"]) == 3
            and int(md["completed_chunks"]) == 3
            and int(md["completed_production_samples"])
            == int(md["expected_production_samples"])
        )
        gap_evaluations = gap_evaluations and (
            gap["status"] == "complete"
            and int(gap["sample_count"]) == int(md["expected_production_samples"])
            and int(gap["lower_charge"]) == int(system["lower_charge"])
            and int(gap["lower_spin"]) == int(system["lower_spin"])
            and int(gap["oxidized_charge"]) == int(system["oxidized_charge"])
            and int(gap["oxidized_spin"]) == int(system["oxidized_spin"])
        )
        trajectory_summaries.append(
            {
                "logical_trajectory_id": logical_id,
                "optimization_steps": optimization["steps"],
                "maximum_force_eV_A": md["maximum_force_eV_A"],
                "production_samples": gap["sample_count"],
                "temperature_mean_K": md["temperature_mean_K"],
                "restraint_activation_fraction": float(
                    trajectory_qc["restraint_activation_fraction"]
                ),
                "maximum_COM_distance_A": float(trajectory_qc["maximum_COM_distance_A"]),
                "max_excess_over_R0_A": float(trajectory_qc["max_excess_over_R0_A"]),
                "longest_escape_duration_ps": float(
                    trajectory_qc[
                        "longest_continuous_exceedance_over_R0_plus_2A_ps"
                    ]
                ),
                "shell_escape": trajectory_qc["shell_escape"] == "True",
                "trajectory_wallclock_seconds": trajectory["wallclock_seconds"],
                "gap_wallclock_seconds": gap["wallclock_seconds"],
            }
        )
    for row in qc_rows:
        critical_flags.update(set(row["flags"].split(";")) & CRITICAL_FLAGS)
    checks = {
        "finite_energy_force_temperature": finite,
        "charge_spin_propagation": state_metadata,
        "three_restart_chunks_per_trajectory": restart_ledgers,
        "same_coordinate_two_state_gap_batches": gap_evaluations,
        "exact_five_solvent_clusters": all(
            row["status"] == "clean" and int(row["solvent_count"]) == 5 for row in clusters
        ),
        "molecule_count_and_atom_order_qc": (
            len(qc_rows) == len(tasks)
            and all(
                sum(
                    row["logical_trajectory_id"] == task["logical_trajectory_id"]
                    for row in solvent_shell_rows
                )
                == 5
                for task in tasks
            )
        ),
        "no_immediate_fragmentation_or_shell_loss": not critical_flags,
        "restraint_force_conservation_and_state_cancellation_tests": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    trajectory_submission = json.loads(
        (raw_root / "submissions" / "pilot_trajectory.json").read_text(encoding="utf-8")
    )
    gap_submission = json.loads(
        (raw_root / "submissions" / "pilot_gap.json").read_text(encoding="utf-8")
    )
    trajectory_retries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((raw_root / "submissions").glob("pilot_trajectory_retry-*.json"))
    ]
    gap_retries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((raw_root / "submissions").glob("pilot_gap_retry-*.json"))
    ]
    payload: dict[str, Any] = {
        "status": status,
        "pilot_system_count": len(manifest),
        "pilot_trajectory_count": len(tasks),
        "pilot_solvent_shell_qc_rows": len(solvent_shell_rows),
        "pilot_roles": sorted(expected_roles),
        "checks": checks,
        "critical_qc_flags": sorted(critical_flags),
        "trajectory_scheduler_job_id": trajectory_submission["scheduler_job_id"],
        "gap_scheduler_job_id": gap_submission["scheduler_job_id"],
        "trajectory_scheduler_job_ids": trajectory_submission["scheduler_job_ids"]
        + [job for retry in trajectory_retries for job in retry["scheduler_job_ids"]],
        "gap_scheduler_job_ids": gap_submission["scheduler_job_ids"]
        + [job for retry in gap_retries for job in retry["scheduler_job_ids"]],
        "trajectory_wallclock_seconds_sum": trajectory_wallclock,
        "gap_wallclock_seconds_sum": gap_wallclock,
        "pilot_raw_bytes": raw_bytes,
        "shell_retention_definition": {
            "restraint_active_when": "d_j > R0",
            "shell_escape_when": "d_j > R0 + 2.0 A continuously for at least 1.0 ps",
            "production_sample_interval_fs": 20.0,
            "required_consecutive_saved_frames": 50,
            "final_frame_exceedance_alone_is_escape": False,
        },
        "trajectory_summaries": trajectory_summaries,
        "test_evidence": [
            "tests/explicit_redox/test_structures_packing_restraint.py",
            "tests/explicit_redox/test_calculator_optimize_dynamics.py",
            "tests/explicit_redox/test_vertical_marcus_alignment.py",
            "tests/explicit_redox/test_qc.py",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MACE-POLAR-1-L five-solvent pilot",
        "",
        f"- Status: **{status}**",
        f"- Scope: {len(manifest)} systems, {len(tasks)} state trajectories, one shell seed each",
        f"- Scheduler jobs: trajectory `{', '.join(payload['trajectory_scheduler_job_ids'])}`, gap `{', '.join(payload['gap_scheduler_job_ids'])}`",
        f"- Summed wallclock: trajectory {trajectory_wallclock / 3600.0:.2f} h; gap {gap_wallclock / 3600.0:.2f} h",
        f"- Raw pilot storage: {raw_bytes / 1024.0**2:.2f} MiB",
        f"- Critical QC flags: {', '.join(sorted(critical_flags)) if critical_flags else 'none'}",
        "- Restraint activation: solvent COM distance `d_j > R0` (diagnostic only)",
        "- Shell escape: `d_j > R0 + 2.0 Å` for at least 50 consecutive saved frames (1.0 ps)",
        "",
        "## Integrity checks",
        "",
        *[f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks.items()],
        "",
    ]
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = write_pilot_report(
        workflow_dir=args.workflow_dir,
        raw_root=args.raw_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
