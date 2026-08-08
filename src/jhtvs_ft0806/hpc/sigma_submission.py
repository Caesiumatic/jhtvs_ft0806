"""Budgeted immutable preparation for the 2500 full-space sigma xTB tasks."""

from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path

from jhtvs_ft0806.geometry.resolution import SIGMA_PREOPT_METHOD_ID
from jhtvs_ft0806.hpc.submission import (
    SubmissionError,
    SubmissionPlan,
    TASK_FIELDS,
    check_budget,
)
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows


SIGMA_WORKFLOW_REVISION = "jhtvs-ft0806-fullspace-sigma-preopt-v1"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SubmissionError(f"sigma submission path is outside repository: {path}") from exc


def _write_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise SubmissionError(f"immutable sigma submission file changed: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def prepare_sigma_submission(
    *,
    submission_id: str,
    spec_dir: Path,
    run_root: Path,
    preflight_path: Path,
    submissions_root: Path,
    accounting_path: Path,
    ledger_path: Path,
    runner_path: Path,
    planning_core_h: Decimal,
    queue: str = "amd16smt",
    parallel_environment: str = "orte",
    max_concurrent: int = 20,
    budget_scope: str = "first_round",
) -> SubmissionPlan:
    if planning_core_h <= 0 or max_concurrent < 1:
        raise SubmissionError("invalid sigma planning cost or concurrency")
    spec_dir = spec_dir.resolve()
    root = spec_dir.parent
    run_root = run_root.resolve()
    runner_path = runner_path.resolve()
    native_array = run_root / "sigma_preopt_array.tsv"
    manifest_path = run_root / "sigma_preopt_manifest.csv"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS"
        or preflight.get("task_count") != 2500
        or preflight.get("exact_parameter_rows") != 2500
        or preflight.get("exact_command_rows") != 2500
        or preflight.get("source_hash_rows") != 2500
        or preflight.get("array_sha256") != sha256_file(native_array)
        or preflight.get("manifest_sha256") != sha256_file(manifest_path)
        or preflight.get("launcher_sha256") != sha256_file(runner_path)
    ):
        raise SubmissionError("full-space sigma preflight or execution hashes differ")
    rows = read_csv_rows(manifest_path)
    if len(rows) != 2500 or any(
        row["formal_charge"] != "2"
        or row["multiplicity"] != "1"
        or row["uhf"] != "0"
        or row["preopt_method_id"] != SIGMA_PREOPT_METHOD_ID
        for row in rows
    ):
        raise SubmissionError("sigma manifest parameters differ from the frozen method")
    per_task = planning_core_h / Decimal(len(rows))
    tasks: list[dict[str, str]] = []
    for array_task, row in enumerate(rows, 1):
        source = run_root / row["source_xyz"]
        output = run_root / row["output_dir"] / "task_status.tsv"
        if not source.is_file() or sha256_file(source) != row["source_xyz_sha256"]:
            raise SubmissionError(f"sigma source hash mismatch: {row['task_id']}")
        if output.exists():
            raise SubmissionError(f"sigma output already exists: {row['task_id']}")
        tasks.append(
            {
                "array_task": str(array_task),
                "sequence": "1",
                "job_id": row["task_id"],
                "job_class": "sigma_preopt",
                "input_path": _relative(source, root),
                "input_sha256": row["source_xyz_sha256"],
                "output_path": _relative(output, root),
                "nprocs": "1",
                "planning_core_h": format(per_task, ".10f"),
                "workflow_revision": SIGMA_WORKFLOW_REVISION,
                "method_id": SIGMA_PREOPT_METHOD_ID,
            }
        )
    submission_dir = (submissions_root / submission_id).resolve()
    task_path = submission_dir / "tasks.tsv"
    prepared_preflight = submission_dir / "preflight.json"
    plan_path = submission_dir / "submission_plan.json"
    task_text = "\t".join(TASK_FIELDS) + "\n" + "".join(
        "\t".join(row[field] for field in TASK_FIELDS) + "\n" for row in tasks
    )
    _write_exact(task_path, task_text)
    submission_preflight = {
        **preflight,
        "status": "PASS",
        "native_preflight_sha256": sha256_file(preflight_path),
        "common_task_table_sha256": sha256_file(task_path),
        "planning_core_h": str(planning_core_h),
        "max_concurrent": max_concurrent,
        "outputs_absent": True,
    }
    _write_exact(
        prepared_preflight,
        json.dumps(submission_preflight, sort_keys=True, indent=2) + "\n",
    )
    budget = check_budget(
        spec_dir=spec_dir,
        accounting_path=accounting_path,
        ledger_path=ledger_path,
        proposed_core_h=planning_core_h,
        budget_scope=budget_scope,
    )
    payload = {
        "submission_id": submission_id,
        "job_ids_sha256": content_hash([row["job_id"] for row in tasks]),
        "job_count": len(tasks),
        "array_task_count": len(tasks),
        "planned_core_h": str(planning_core_h),
        "task_table_sha256": sha256_file(task_path),
        "preflight_report_sha256": sha256_file(prepared_preflight),
        "runner_sha256": sha256_file(runner_path),
        "native_array_sha256": sha256_file(native_array),
        "budget_scope": budget_scope,
        "budget": budget.to_dict(),
        "queue": queue,
        "parallel_environment": parallel_environment,
        "nprocs": 1,
        "max_concurrent": max_concurrent,
    }
    submission_sha = content_hash(payload)
    _write_exact(
        plan_path,
        json.dumps({**payload, "submission_sha256": submission_sha}, sort_keys=True, indent=2)
        + "\n",
    )
    return SubmissionPlan(
        submission_id=submission_id,
        submission_dir=submission_dir,
        task_table_path=task_path,
        preflight_report_path=prepared_preflight,
        plan_path=plan_path,
        task_table_sha256=sha256_file(task_path),
        submission_sha256=submission_sha,
        job_count=len(tasks),
        array_task_count=len(tasks),
        planned_core_h=planning_core_h,
        budget=budget,
        queue=queue,
        parallel_environment=parallel_environment,
        nprocs=1,
        max_concurrent=max_concurrent,
    )
