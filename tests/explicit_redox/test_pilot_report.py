from __future__ import annotations

import csv
import json
from pathlib import Path

from jhtvs_ft0806.explicit_redox.pilot_report import write_pilot_report


def _write_csv(path: Path, rows: list[dict[str, object]], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_pilot_report_requires_complete_four_system_two_state_scope(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow"
    raw = tmp_path / "raw"
    roles = (
        "small_monomer_acetonitrile",
        "largest_flexible_monomer",
        "solvent_self_solvation",
        "anion_propylene_carbonate",
    )
    manifest = []
    clusters = []
    tasks = []
    qc = []
    solvent_shell_qc = []
    seeds = []
    for system_index, role in enumerate(roles):
        system_id = f"system-{system_index}"
        manifest.append(
            {
                "pilot_role": role,
                "system_id": system_id,
                "lower_charge": 0,
                "lower_spin": 1,
                "oxidized_charge": 1,
                "oxidized_spin": 2,
            }
        )
        clusters.append({"system_id": system_id, "status": "clean", "solvent_count": 5})
        seeds.append({"system_id": system_id, "seed_index": 0})
        for state, charge, spin in (("lower", 0, 1), ("oxidized", 1, 2)):
            logical_id = f"{system_id}__{state}"
            tasks.append(
                {
                    "logical_trajectory_id": logical_id,
                    "system_id": system_id,
                    "state": state,
                    "charge": charge,
                    "spin": spin,
                }
            )
            qc.append(
                {
                    "logical_trajectory_id": logical_id,
                    "flags": "clean",
                    "restraint_activation_fraction": 0.0,
                    "maximum_COM_distance_A": 8.0,
                    "max_excess_over_R0_A": 0.0,
                    "longest_continuous_exceedance_over_R0_plus_2A_ps": 0.0,
                    "shell_escape": False,
                }
            )
            for solvent_index in range(5):
                solvent_shell_qc.append(
                    {
                        "logical_trajectory_id": logical_id,
                        "solvent_index": solvent_index,
                        "restraint_activation_fraction": 0.0,
                        "maximum_COM_distance_A": 8.0,
                        "max_excess_over_R0_A": 0.0,
                        "longest_continuous_exceedance_over_R0_plus_2A_ps": 0.0,
                        "shell_escape": False,
                    }
                )
            trajectory_dir = raw / "trajectories" / logical_id
            gap_dir = raw / "gaps" / logical_id
            trajectory_dir.mkdir(parents=True)
            gap_dir.mkdir(parents=True)
            (trajectory_dir / "trajectory.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "charge": charge,
                        "spin": spin,
                        "wallclock_seconds": 20.0,
                        "optimization": {"steps": 4, "observed_max_force_eV_A": 0.01},
                        "md": {
                            "chunks": 3,
                            "completed_chunks": 3,
                            "completed_production_samples": 100,
                            "expected_production_samples": 100,
                            "maximum_force_eV_A": 0.5,
                            "temperature_mean_K": 300.0,
                            "restraint_activation_samples": 0,
                            "maximum_excursion_A": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (gap_dir / "gaps.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "sample_count": 100,
                        "lower_charge": 0,
                        "lower_spin": 1,
                        "oxidized_charge": 1,
                        "oxidized_spin": 2,
                        "wallclock_seconds": 5.0,
                    }
                ),
                encoding="utf-8",
            )
    _write_csv(workflow / "pilot_manifest.csv", manifest)
    _write_csv(raw / "pilot_cluster_manifest.csv", clusters)
    _write_csv(raw / "pilot_trajectory_tasks.tsv", tasks, delimiter="\t")
    _write_csv(raw / "pilot_trajectory_qc.csv", qc)
    _write_csv(raw / "pilot_solvent_shell_qc.csv", solvent_shell_qc)
    _write_csv(raw / "pilot_seed_gap_summary.csv", seeds)
    submissions = raw / "submissions"
    submissions.mkdir()
    (submissions / "pilot_trajectory.json").write_text(
        json.dumps({"scheduler_job_id": "100", "scheduler_job_ids": ["100"]}),
        encoding="utf-8",
    )
    (submissions / "pilot_gap.json").write_text(
        json.dumps({"scheduler_job_id": "101", "scheduler_job_ids": ["101"]}),
        encoding="utf-8",
    )
    payload = write_pilot_report(
        workflow_dir=workflow,
        raw_root=raw,
        output_json=workflow / "pilot_report.json",
        output_markdown=tmp_path / "PILOT.md",
    )
    assert payload["status"] == "PASS"
    assert payload["pilot_trajectory_count"] == 8
    assert payload["trajectory_wallclock_seconds_sum"] == 160.0
    assert (workflow / "pilot_report.json").is_file()
    assert "Critical QC flags: none" in (tmp_path / "PILOT.md").read_text()
