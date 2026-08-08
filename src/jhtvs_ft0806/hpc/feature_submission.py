"""Immutable, budget-guarded SGE preparation for frozen PolarMACE features."""

from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path

from jhtvs_ft0806.hpc.submission import (
    ACTIVE_SUBMISSION_STATUSES,
    SubmissionError,
    SubmissionPlan,
    TASK_FIELDS,
    check_budget,
)
from jhtvs_ft0806.ml.features import (
    EXPECTED_CHECKPOINT_NAME,
    EXPECTED_CHECKPOINT_SHA256,
    FEATURE_SCHEMA_REVISION,
)
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows
from jhtvs_ft0806.spec_validation import validate_spec


FEATURE_WORKFLOW_REVISION = "jhtvs-ft0806-polar1l-base-features-v1"
FEATURE_METHOD_ID = "MACE_POLAR_1_L_raw_invariant_base_v1"
FEATURE_JOB_ID = "MACEBASE-CALIBRATION"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SubmissionError(f"feature submission path is outside repository: {path}") from exc


def _write_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SubmissionError(f"immutable feature submission file changed: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _task_text(row: dict[str, str]) -> str:
    output = ["\t".join(TASK_FIELDS)]
    output.append("\t".join(row[field] for field in TASK_FIELDS))
    return "\n".join(output) + "\n"


def _active_job_ids(ledger_path: Path, repository_root: Path) -> set[str]:
    active: set[str] = set()
    if not ledger_path.is_file():
        return active
    for ledger in read_csv_rows(ledger_path):
        if ledger["status"] not in ACTIVE_SUBMISSION_STATUSES:
            continue
        task_path = Path(ledger["task_table_path"])
        if not task_path.is_absolute():
            task_path = repository_root / task_path
        if not task_path.is_file() or sha256_file(task_path) != ledger["task_table_sha256"]:
            raise SubmissionError("active submission task table is missing or hash-mismatched")
        with task_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != TASK_FIELDS:
                raise SubmissionError("active submission task-table schema mismatch")
            active.update(row["job_id"] for row in reader)
    return active


def prepare_feature_submission(
    *,
    submission_id: str,
    spec_dir: Path,
    geometry_index_path: Path,
    submissions_root: Path,
    accounting_path: Path,
    ledger_path: Path,
    runner_path: Path,
    planning_core_h: Decimal,
    queue: str = "amd16smt",
    parallel_environment: str = "orte",
    budget_scope: str = "first_round",
) -> SubmissionPlan:
    if not submission_id or any(character.isspace() for character in submission_id):
        raise SubmissionError("invalid feature submission ID")
    if planning_core_h <= 0:
        raise SubmissionError("feature planning core-hours must be positive")
    spec_dir = spec_dir.resolve()
    repository_root = spec_dir.parent
    geometry_index_path = geometry_index_path.resolve()
    runner_path = runner_path.resolve()
    validation = validate_spec(spec_dir)
    if not validation.ok:
        raise SubmissionError("scientific specification validation failed")
    if not runner_path.is_file() or runner_path.is_symlink():
        raise SubmissionError("feature runner is missing or unsafe")
    geometry_rows = read_csv_rows(geometry_index_path)
    resolved = [row for row in geometry_rows if row["status"] == "resolved"]
    if len(geometry_rows) != 705 or len(resolved) != 705:
        raise SubmissionError(
            f"calibration feature extraction requires 705 resolved geometries, "
            f"found {len(resolved)}/{len(geometry_rows)}"
        )
    keys = {(row["state_id"], row["solvent_id"]) for row in resolved}
    if len(keys) != len(resolved):
        raise SubmissionError("duplicate calibration state-medium geometry key")
    submission_dir = (submissions_root / submission_id).resolve()
    task_table_path = submission_dir / "tasks.tsv"
    preflight_path = submission_dir / "preflight.json"
    plan_path = submission_dir / "submission_plan.json"
    completion_path = submission_dir / "feature_completion.json"
    feature_index_path = repository_root / "data" / "resolved" / "base_feature_index.csv"
    baseline_path = repository_root / "data" / "resolved" / "base_state_energies.csv"
    if completion_path.exists() or feature_index_path.exists() or baseline_path.exists():
        raise SubmissionError("feature production outputs already exist; reconcile before submission")

    geometry_sha = sha256_file(geometry_index_path)
    task = {
        "array_task": "1",
        "sequence": "1",
        "job_id": FEATURE_JOB_ID,
        "job_class": "mace_base_features",
        "input_path": _relative(geometry_index_path, repository_root),
        "input_sha256": geometry_sha,
        "output_path": _relative(completion_path, repository_root),
        "nprocs": "1",
        "planning_core_h": str(planning_core_h),
        "workflow_revision": FEATURE_WORKFLOW_REVISION,
        "method_id": FEATURE_METHOD_ID,
    }
    _write_exact(task_table_path, _task_text(task))
    preflight = {
        "status": "PASS",
        "job_id": FEATURE_JOB_ID,
        "resolved_geometry_rows": len(resolved),
        "geometry_index_sha256": geometry_sha,
        "checkpoint_name": EXPECTED_CHECKPOINT_NAME,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "default_dtype": "float64",
        "feature_schema_revision": FEATURE_SCHEMA_REVISION,
        "runner_sha256": sha256_file(runner_path),
        "task_table_sha256": sha256_file(task_table_path),
        "planning_core_h": str(planning_core_h),
        "outputs_absent": True,
    }
    _write_exact(
        preflight_path,
        json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    budget = check_budget(
        spec_dir=spec_dir,
        accounting_path=accounting_path,
        ledger_path=ledger_path,
        proposed_core_h=planning_core_h,
        budget_scope=budget_scope,
    )
    plan_payload = {
        "submission_id": submission_id,
        "job_ids": [FEATURE_JOB_ID],
        "job_count": 1,
        "array_task_count": 1,
        "planned_core_h": str(planning_core_h),
        "task_table_sha256": sha256_file(task_table_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "runner_sha256": sha256_file(runner_path),
        "budget_scope": budget_scope,
        "budget": budget.to_dict(),
        "queue": queue,
        "parallel_environment": parallel_environment,
        "nprocs": 1,
        "max_concurrent": 1,
    }
    submission_sha = content_hash(plan_payload)
    _write_exact(
        plan_path,
        json.dumps(
            {**plan_payload, "submission_sha256": submission_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    active = _active_job_ids(ledger_path, repository_root)
    if FEATURE_JOB_ID in active:
        ledger_matches = [
            row
            for row in read_csv_rows(ledger_path)
            if row["submission_id"] == submission_id
            and row["submission_sha256"] == submission_sha
            and row["status"] in ACTIVE_SUBMISSION_STATUSES
        ]
        if len(ledger_matches) != 1:
            raise SubmissionError(
                "calibration base-feature extraction belongs to another active submission"
            )
    return SubmissionPlan(
        submission_id=submission_id,
        submission_dir=submission_dir,
        task_table_path=task_table_path,
        preflight_report_path=preflight_path,
        plan_path=plan_path,
        task_table_sha256=sha256_file(task_table_path),
        submission_sha256=submission_sha,
        job_count=1,
        array_task_count=1,
        planned_core_h=planning_core_h,
        budget=budget,
        queue=queue,
        parallel_environment=parallel_environment,
        nprocs=1,
        max_concurrent=1,
    )
