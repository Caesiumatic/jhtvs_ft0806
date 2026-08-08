from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path
import shutil

from jhtvs_ft0806.hpc.feature_submission import prepare_feature_submission
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import write_csv_deterministic


ROOT = Path(__file__).resolve().parents[1]


def test_feature_submission_preflight_is_budgeted_and_content_addressed(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "spec", repository / "spec")
    shutil.copy2(ROOT / "AGENTS.md", repository / "AGENTS.md")
    (repository / "hpc").mkdir()
    shutil.copy2(
        ROOT / "hpc" / "run_feature_extraction.sh",
        repository / "hpc" / "run_feature_extraction.sh",
    )
    geometry = repository / "runs" / "geometry" / "state.xyz"
    geometry.parent.mkdir(parents=True)
    geometry.write_text("1\nfixture\nH 0 0 0\n", encoding="utf-8")
    rows = [
        {
            "state_id": f"STATE-{index:04d}",
            "solvent_id": f"S{index % 25 + 1:03d}",
            "status": "resolved",
            "xyz_path": "runs/geometry/state.xyz",
            "xyz_sha256": sha256_file(geometry),
        }
        for index in range(705)
    ]
    geometry_index = repository / "data" / "resolved" / "geometry_index.csv"
    write_csv_deterministic(
        geometry_index,
        ("state_id", "solvent_id", "status", "xyz_path", "xyz_sha256"),
        rows,
    )
    plan = prepare_feature_submission(
        submission_id="feature-fixture",
        spec_dir=repository / "spec",
        geometry_index_path=geometry_index,
        submissions_root=repository / "runs" / "hpc" / "submissions",
        accounting_path=repository / "data" / "resolved" / "accounting.csv",
        ledger_path=repository / "data" / "resolved" / "submission_ledger.csv",
        runner_path=repository / "hpc" / "run_feature_extraction.sh",
        planning_core_h=Decimal("8"),
    )
    repeated = prepare_feature_submission(
        submission_id="feature-fixture",
        spec_dir=repository / "spec",
        geometry_index_path=geometry_index,
        submissions_root=repository / "runs" / "hpc" / "submissions",
        accounting_path=repository / "data" / "resolved" / "accounting.csv",
        ledger_path=repository / "data" / "resolved" / "submission_ledger.csv",
        runner_path=repository / "hpc" / "run_feature_extraction.sh",
        planning_core_h=Decimal("8"),
    )
    with plan.task_table_path.open(encoding="utf-8", newline="") as handle:
        task = next(csv.DictReader(handle, delimiter="\t"))

    assert plan.job_count == plan.array_task_count == 1
    assert plan.nprocs == 1
    assert plan.planned_core_h == Decimal("8")
    assert task["job_id"] == "MACEBASE-CALIBRATION"
    assert task["input_sha256"] == sha256_file(geometry_index)
    assert plan.budget.projected_core_h == Decimal("8")
    assert repeated.submission_sha256 == plan.submission_sha256


def test_fullspace_feature_submission_uses_8100_rows_and_eight_slots(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "spec", repository / "spec")
    shutil.copy2(ROOT / "AGENTS.md", repository / "AGENTS.md")
    (repository / "hpc").mkdir()
    runner = repository / "hpc" / "run_feature_extraction.sh"
    shutil.copy2(ROOT / "hpc" / "run_feature_extraction.sh", runner)
    geometry = repository / "runs" / "geometry" / "state.xyz"
    geometry.parent.mkdir(parents=True)
    geometry.write_text("1\nfixture\nH 0 0 0\n", encoding="utf-8")
    rows = [
        {
            "state_id": f"STATE-{index:05d}",
            "solvent_id": f"S{index % 25 + 1:03d}",
            "status": "resolved",
            "xyz_path": "runs/geometry/state.xyz",
            "xyz_sha256": sha256_file(geometry),
        }
        for index in range(8100)
    ]
    rows[-1]["status"] = "failed"
    rows[-1]["xyz_path"] = ""
    rows[-1]["xyz_sha256"] = ""
    geometry_index = (
        repository / "data" / "resolved" / "fullspace_geometry_index.csv"
    )
    write_csv_deterministic(
        geometry_index,
        ("state_id", "solvent_id", "status", "xyz_path", "xyz_sha256"),
        rows,
    )

    plan = prepare_feature_submission(
        submission_id="fullspace-feature-fixture",
        spec_dir=repository / "spec",
        geometry_index_path=geometry_index,
        submissions_root=repository / "runs" / "hpc" / "submissions",
        accounting_path=repository / "data" / "resolved" / "accounting.csv",
        ledger_path=repository / "data" / "resolved" / "submission_ledger.csv",
        runner_path=runner,
        planning_core_h=Decimal("36"),
        dataset_kind="fullspace",
    )
    with plan.task_table_path.open(encoding="utf-8", newline="") as handle:
        task = next(csv.DictReader(handle, delimiter="\t"))

    assert plan.job_count == plan.array_task_count == 1
    assert plan.nprocs == 8
    assert task["job_id"] == "MACEBASE-FULLSPACE"
    assert task["nprocs"] == "8"
    assert task["input_path"] == "data/resolved/fullspace_geometry_index.csv"
    assert task["output_path"].endswith("/feature_completion.json")
    preflight = json.loads(plan.preflight_report_path.read_text(encoding="utf-8"))
    assert preflight["geometry_rows"] == 8100
    assert preflight["resolved_geometry_rows"] == 8099
    assert preflight["failed_geometry_rows"] == 1
