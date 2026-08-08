"""Budget-guarded, idempotent SGE submission preparation for ORCA jobs."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Mapping, Sequence

from jhtvs_ft0806.orca.preflight import audit_decks
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


TASK_FIELDS = (
    "array_task",
    "sequence",
    "job_id",
    "job_class",
    "input_path",
    "input_sha256",
    "output_path",
    "nprocs",
    "planning_core_h",
    "workflow_revision",
    "method_id",
)
LEDGER_FIELDS = (
    "submission_id",
    "submission_sha256",
    "scheduler_job_id",
    "status",
    "submitted_at_utc",
    "job_count",
    "array_task_count",
    "planned_core_h",
    "task_table_path",
    "task_table_sha256",
    "preflight_report_path",
    "preflight_report_sha256",
    "budget_scope",
    "queue",
    "parallel_environment",
    "nprocs",
    "max_concurrent",
)
ACTIVE_SUBMISSION_STATUSES = {"submitted", "queued", "running"}
_SUBMISSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_QSUB_JOB_ID = re.compile(r"^(\d+)")


class SubmissionError(ValueError):
    """Raised when a submission cannot be made reproducibly and safely."""


@dataclass(frozen=True, slots=True)
class BudgetState:
    consumed_core_h: Decimal
    active_planned_core_h: Decimal
    proposed_core_h: Decimal
    first_round_limit_core_h: Decimal
    project_limit_core_h: Decimal

    @property
    def projected_core_h(self) -> Decimal:
        return (
            self.consumed_core_h
            + self.active_planned_core_h
            + self.proposed_core_h
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "consumed_core_h": str(self.consumed_core_h),
            "active_planned_core_h": str(self.active_planned_core_h),
            "proposed_core_h": str(self.proposed_core_h),
            "projected_core_h": str(self.projected_core_h),
            "first_round_limit_core_h": str(self.first_round_limit_core_h),
            "project_limit_core_h": str(self.project_limit_core_h),
        }


@dataclass(frozen=True, slots=True)
class SubmissionPlan:
    submission_id: str
    submission_dir: Path
    task_table_path: Path
    preflight_report_path: Path
    plan_path: Path
    task_table_sha256: str
    submission_sha256: str
    job_count: int
    array_task_count: int
    planned_core_h: Decimal
    budget: BudgetState
    queue: str
    parallel_environment: str
    nprocs: int
    max_concurrent: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PREPARED",
            "submission_id": self.submission_id,
            "submission_dir": str(self.submission_dir),
            "task_table_path": str(self.task_table_path),
            "preflight_report_path": str(self.preflight_report_path),
            "plan_path": str(self.plan_path),
            "task_table_sha256": self.task_table_sha256,
            "submission_sha256": self.submission_sha256,
            "job_count": self.job_count,
            "array_task_count": self.array_task_count,
            "planned_core_h": str(self.planned_core_h),
            "budget": self.budget.to_dict(),
            "queue": self.queue,
            "parallel_environment": self.parallel_environment,
            "nprocs": self.nprocs,
            "max_concurrent": self.max_concurrent,
        }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _portable_path(path_text: str, repository_root: Path) -> str:
    path = Path(path_text)
    absolute = path if path.is_absolute() else repository_root / path
    try:
        return absolute.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise SubmissionError(
            f"submission input is outside repository root: {absolute}"
        ) from exc


def _read_optional_csv(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.is_file() else []


def _budget_limits(spec_dir: Path) -> tuple[Decimal, Decimal]:
    rows = read_csv_rows(spec_dir / "compute_budget.csv")
    first_round = next(
        (
            Decimal(row["hard_stop_core_h"])
            for row in rows
            if row["record_type"] == "guard" and row["phase"] == "first_round"
        ),
        None,
    )
    project = next(
        (
            Decimal(row["project_cap_core_h"])
            for row in rows
            if row["record_type"] == "guard"
            and row["phase"] == "whole_project"
        ),
        None,
    )
    if first_round is None or project is None:
        raise SubmissionError("compute budget lacks first-round or whole-project guard")
    return first_round, project


def check_budget(
    *,
    spec_dir: Path,
    accounting_path: Path,
    ledger_path: Path,
    proposed_core_h: Decimal,
    budget_scope: str = "first_round",
) -> BudgetState:
    if proposed_core_h <= 0:
        raise SubmissionError("proposed planned core-hours must be positive")
    if budget_scope not in {"first_round", "whole_project"}:
        raise SubmissionError(f"unsupported budget scope: {budget_scope!r}")
    first_round_limit, project_limit = _budget_limits(spec_dir)
    consumed = sum(
        (Decimal(row["core_h"]) for row in _read_optional_csv(accounting_path)),
        Decimal("0"),
    )
    active_planned = sum(
        (
            Decimal(row["planned_core_h"])
            for row in _read_optional_csv(ledger_path)
            if row["status"] in ACTIVE_SUBMISSION_STATUSES
        ),
        Decimal("0"),
    )
    state = BudgetState(
        consumed_core_h=consumed,
        active_planned_core_h=active_planned,
        proposed_core_h=proposed_core_h,
        first_round_limit_core_h=first_round_limit,
        project_limit_core_h=project_limit,
    )
    applicable_limit = (
        first_round_limit if budget_scope == "first_round" else project_limit
    )
    if state.projected_core_h > applicable_limit:
        raise SubmissionError(
            f"budget guard blocks submission: projected {state.projected_core_h} "
            f"> {budget_scope} limit {applicable_limit} core-h"
        )
    if state.projected_core_h > project_limit:
        raise SubmissionError(
            f"project cap blocks submission: projected {state.projected_core_h} "
            f"> {project_limit} core-h"
        )
    return state


def _read_task_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TASK_FIELDS:
            raise SubmissionError(f"task table header mismatch: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise SubmissionError(f"task table row has extra fields: {path}")
    return rows


def _active_job_ids(ledger_path: Path, repository_root: Path) -> set[str]:
    active: set[str] = set()
    for row in _read_optional_csv(ledger_path):
        if row["status"] not in ACTIVE_SUBMISSION_STATUSES:
            continue
        task_table_path = Path(row["task_table_path"])
        if not task_table_path.is_absolute():
            task_table_path = repository_root / task_table_path
        if not task_table_path.is_file():
            raise SubmissionError(
                f"active submission task table is unavailable: {task_table_path}"
            )
        if sha256_file(task_table_path) != row["task_table_sha256"]:
            raise SubmissionError(
                f"active submission task table hash mismatch: {task_table_path}"
            )
        active.update(task["job_id"] for task in _read_task_rows(task_table_path))
    return active


def _bundle_rows(
    selected_rows: Sequence[Mapping[str, str]], repository_root: Path
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in selected_rows:
        group = (
            f"optfreq:{row['job_id']}"
            if row["job_class"] == "optfreq"
            else f"sp:{row['geometry_key']}"
        )
        grouped[group].append(row)

    tasks: list[dict[str, object]] = []
    for array_task, group_key in enumerate(sorted(grouped), start=1):
        rows = sorted(grouped[group_key], key=lambda row: row["job_id"])
        for sequence, row in enumerate(rows, start=1):
            input_path = _portable_path(row["input_path"], repository_root)
            tasks.append(
                {
                    "array_task": array_task,
                    "sequence": sequence,
                    "job_id": row["job_id"],
                    "job_class": row["job_class"],
                    "input_path": input_path,
                    "input_sha256": row["input_sha256"],
                    "output_path": str(Path(input_path).with_suffix(".out")),
                    "nprocs": row["nprocs"],
                    "planning_core_h": row["planning_core_h"],
                    "workflow_revision": row["workflow_revision"],
                    "method_id": row["method_id"],
                }
            )
    return tasks


def _task_table_text(rows: Sequence[Mapping[str, object]]) -> str:
    lines = ["\t".join(TASK_FIELDS)]
    for row in rows:
        values = [str(row[field]) for field in TASK_FIELDS]
        if any("\t" in value or "\n" in value for value in values):
            raise SubmissionError("task table value contains a tab or newline")
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def _submitted_plan(
    *,
    submission_id: str,
    selected_job_ids: set[str],
    submission_dir: Path,
    task_table_path: Path,
    preflight_report_path: Path,
    plan_path: Path,
    runner_path: Path,
    budget_scope: str,
    queue: str,
    parallel_environment: str,
    nprocs: int,
    max_concurrent: int,
) -> SubmissionPlan | None:
    receipt_path = submission_dir / "qsub_receipt.json"
    if not receipt_path.exists():
        return None
    if not plan_path.is_file() or not task_table_path.is_file():
        raise SubmissionError("submitted receipt exists without its immutable plan/tasks")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_submission_sha = payload.pop("submission_sha256", None)
    if content_hash(payload) != expected_submission_sha:
        raise SubmissionError("existing submitted plan hash mismatch")
    expected = {
        "submission_id": submission_id,
        "job_ids": sorted(selected_job_ids),
        "task_table_sha256": sha256_file(task_table_path),
        "runner_sha256": sha256_file(runner_path),
        "budget_scope": budget_scope,
        "queue": queue,
        "parallel_environment": parallel_environment,
        "nprocs": nprocs,
        "max_concurrent": max_concurrent,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise SubmissionError(
                f"submitted plan {field} differs from requested value"
            )
    if (
        not preflight_report_path.is_file()
        or sha256_file(preflight_report_path)
        != payload["preflight_report_sha256"]
    ):
        raise SubmissionError("existing submitted preflight report hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("submission_sha256") != expected_submission_sha:
        raise SubmissionError("qsub receipt does not match existing submitted plan")
    budget_values = payload["budget"]
    budget = BudgetState(
        consumed_core_h=Decimal(budget_values["consumed_core_h"]),
        active_planned_core_h=Decimal(budget_values["active_planned_core_h"]),
        proposed_core_h=Decimal(budget_values["proposed_core_h"]),
        first_round_limit_core_h=Decimal(
            budget_values["first_round_limit_core_h"]
        ),
        project_limit_core_h=Decimal(budget_values["project_limit_core_h"]),
    )
    return SubmissionPlan(
        submission_id=submission_id,
        submission_dir=submission_dir,
        task_table_path=task_table_path,
        preflight_report_path=preflight_report_path,
        plan_path=plan_path,
        task_table_sha256=payload["task_table_sha256"],
        submission_sha256=str(expected_submission_sha),
        job_count=int(payload["job_count"]),
        array_task_count=int(payload["array_task_count"]),
        planned_core_h=Decimal(payload["planned_core_h"]),
        budget=budget,
        queue=queue,
        parallel_environment=parallel_environment,
        nprocs=nprocs,
        max_concurrent=max_concurrent,
    )


def prepare_submission(
    *,
    submission_id: str,
    selected_job_ids: set[str],
    spec_dir: Path,
    geometry_index_path: Path,
    deck_manifest_path: Path,
    submissions_root: Path,
    accounting_path: Path,
    ledger_path: Path,
    runner_path: Path,
    budget_scope: str = "first_round",
    queue: str = "amd16smt",
    parallel_environment: str = "orte",
    nprocs: int = 8,
    max_concurrent: int = 8,
) -> SubmissionPlan:
    if not _SUBMISSION_ID.fullmatch(submission_id):
        raise SubmissionError(f"invalid submission ID: {submission_id!r}")
    if not selected_job_ids:
        raise SubmissionError("submit requires at least one explicit job ID")
    if nprocs != 8:
        raise SubmissionError("frozen ORCA manifests require exactly 8 MPI ranks")
    if max_concurrent < 1:
        raise SubmissionError("max_concurrent must be positive")

    spec_dir = spec_dir.resolve()
    repository_root = spec_dir.parent
    runner_path = runner_path.resolve()
    if not runner_path.is_file() or runner_path.is_symlink():
        raise SubmissionError(f"ORCA runner is missing or unsafe: {runner_path}")
    submission_dir = (submissions_root / submission_id).resolve()
    submission_dir.mkdir(parents=True, exist_ok=True)
    task_table_path = submission_dir / "tasks.tsv"
    preflight_report_path = submission_dir / "preflight.json"
    plan_path = submission_dir / "submission_plan.json"
    existing_submitted = _submitted_plan(
        submission_id=submission_id,
        selected_job_ids=selected_job_ids,
        submission_dir=submission_dir,
        task_table_path=task_table_path,
        preflight_report_path=preflight_report_path,
        plan_path=plan_path,
        runner_path=runner_path,
        budget_scope=budget_scope,
        queue=queue,
        parallel_environment=parallel_environment,
        nprocs=nprocs,
        max_concurrent=max_concurrent,
    )
    if existing_submitted is not None:
        return existing_submitted

    audit = audit_decks(
        spec_dir=spec_dir,
        geometry_index_path=geometry_index_path,
        deck_manifest_path=deck_manifest_path,
        selected_job_ids=selected_job_ids,
        report_path=preflight_report_path,
        require_output_absent=True,
    )
    if not audit.ok:
        raise SubmissionError(
            f"deck preflight failed: {audit.issues[:5]}"
        )

    manifest_rows = read_csv_rows(deck_manifest_path)
    manifest_by_id = {row["job_id"]: row for row in manifest_rows}
    selected_rows = [manifest_by_id[job_id] for job_id in sorted(selected_job_ids)]
    if any(row["nprocs"] != str(nprocs) for row in selected_rows):
        raise SubmissionError("selected deck nprocs differs from scheduler request")
    active_overlap = selected_job_ids & _active_job_ids(ledger_path, repository_root)
    if active_overlap:
        raise SubmissionError(
            f"jobs already belong to active submissions: {sorted(active_overlap)}"
        )

    tasks = _bundle_rows(selected_rows, repository_root)
    task_text = _task_table_text(tasks)
    if task_table_path.exists():
        if task_table_path.read_text(encoding="utf-8") != task_text:
            raise SubmissionError(
                f"submission ID {submission_id!r} already has different tasks"
            )
    else:
        _write_text_atomic(task_table_path, task_text)
    task_table_sha256 = sha256_file(task_table_path)
    planned_core_h = sum(
        (Decimal(row["planning_core_h"]) for row in selected_rows), Decimal("0")
    )
    budget = check_budget(
        spec_dir=spec_dir,
        accounting_path=accounting_path,
        ledger_path=ledger_path,
        proposed_core_h=planned_core_h,
        budget_scope=budget_scope,
    )
    plan_payload = {
        "submission_id": submission_id,
        "job_ids": sorted(selected_job_ids),
        "job_count": len(selected_rows),
        "array_task_count": max(int(row["array_task"]) for row in tasks),
        "planned_core_h": str(planned_core_h),
        "task_table_sha256": task_table_sha256,
        "preflight_report_sha256": sha256_file(preflight_report_path),
        "runner_sha256": sha256_file(runner_path),
        "budget_scope": budget_scope,
        "budget": budget.to_dict(),
        "queue": queue,
        "parallel_environment": parallel_environment,
        "nprocs": nprocs,
        "max_concurrent": max_concurrent,
    }
    submission_sha256 = content_hash(plan_payload)
    rendered_plan = json.dumps(
        {**plan_payload, "submission_sha256": submission_sha256},
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    if plan_path.exists() and plan_path.read_text(encoding="utf-8") != rendered_plan:
        raise SubmissionError(
            f"submission ID {submission_id!r} already has a different plan"
        )
    _write_text_atomic(plan_path, rendered_plan)
    return SubmissionPlan(
        submission_id=submission_id,
        submission_dir=submission_dir,
        task_table_path=task_table_path,
        preflight_report_path=preflight_report_path,
        plan_path=plan_path,
        task_table_sha256=task_table_sha256,
        submission_sha256=submission_sha256,
        job_count=len(selected_rows),
        array_task_count=max(int(row["array_task"]) for row in tasks),
        planned_core_h=planned_core_h,
        budget=budget,
        queue=queue,
        parallel_environment=parallel_environment,
        nprocs=nprocs,
        max_concurrent=max_concurrent,
    )


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _append_ledger(
    *,
    plan: SubmissionPlan,
    ledger_path: Path,
    scheduler_job_id: str,
    budget_scope: str,
    repository_root: Path,
) -> None:
    rows = _read_optional_csv(ledger_path)
    matching = [row for row in rows if row["submission_id"] == plan.submission_id]
    if matching:
        if len(matching) != 1 or matching[0]["submission_sha256"] != plan.submission_sha256:
            raise SubmissionError("submission ledger contains an ambiguous ID")
        return
    rows.append(
        {
            "submission_id": plan.submission_id,
            "submission_sha256": plan.submission_sha256,
            "scheduler_job_id": scheduler_job_id,
            "status": "submitted",
            "submitted_at_utc": datetime.now(UTC).isoformat(),
            "job_count": plan.job_count,
            "array_task_count": plan.array_task_count,
            "planned_core_h": str(plan.planned_core_h),
            "task_table_path": _relative_or_absolute(
                plan.task_table_path, repository_root
            ),
            "task_table_sha256": plan.task_table_sha256,
            "preflight_report_path": _relative_or_absolute(
                plan.preflight_report_path, repository_root
            ),
            "preflight_report_sha256": sha256_file(plan.preflight_report_path),
            "budget_scope": budget_scope,
            "queue": plan.queue,
            "parallel_environment": plan.parallel_environment,
            "nprocs": plan.nprocs,
            "max_concurrent": plan.max_concurrent,
        }
    )
    write_csv_deterministic(
        ledger_path, LEDGER_FIELDS, rows, sort_by=("submission_id",)
    )


def execute_submission(
    *,
    plan: SubmissionPlan,
    runner_path: Path,
    spec_dir: Path,
    accounting_path: Path,
    ledger_path: Path,
    budget_scope: str = "first_round",
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    repository_root = runner_path.resolve().parent.parent
    receipt_path = plan.submission_dir / "qsub_receipt.json"
    intent_path = plan.submission_dir / "qsub_intent.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.parent / ".submission.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["submission_sha256"] != plan.submission_sha256:
                raise SubmissionError("qsub receipt does not match prepared submission")
            scheduler_job_id = str(receipt["scheduler_job_id"])
            _append_ledger(
                plan=plan,
                ledger_path=ledger_path,
                scheduler_job_id=scheduler_job_id,
                budget_scope=budget_scope,
                repository_root=repository_root,
            )
            return {**plan.to_dict(), **receipt, "status": "ALREADY_SUBMITTED"}
        if intent_path.exists():
            raise SubmissionError(
                "qsub intent exists without a receipt; reconcile scheduler state "
                "before any retry"
            )

        execution_budget = check_budget(
            spec_dir=spec_dir,
            accounting_path=accounting_path,
            ledger_path=ledger_path,
            proposed_core_h=plan.planned_core_h,
            budget_scope=budget_scope,
        )
        selected_job_ids = {
            row["job_id"] for row in _read_task_rows(plan.task_table_path)
        }
        active_overlap = selected_job_ids & _active_job_ids(
            ledger_path, repository_root
        )
        if active_overlap:
            raise SubmissionError(
                f"jobs became active after preparation: {sorted(active_overlap)}"
            )

        scheduler_log_dir = plan.submission_dir / "scheduler_logs"
        scheduler_log_dir.mkdir(parents=True, exist_ok=True)
        job_name = "jft_" + re.sub(
            r"[^A-Za-z0-9_]", "_", plan.submission_id
        )[:60]
        environment = {
            "TASK_FILE": str(plan.task_table_path),
            "TASK_FILE_SHA256": plan.task_table_sha256,
            **(extra_environment or {}),
        }
        if any(
            not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
            or not value
            or "," in value
            for key, value in environment.items()
        ):
            raise SubmissionError("invalid qsub environment override")
        command = [
            "qsub",
            "-terse",
            "-cwd",
            "-j",
            "y",
            "-o",
            str(scheduler_log_dir),
            "-N",
            job_name,
            "-t",
            f"1-{plan.array_task_count}",
            "-tc",
            str(plan.max_concurrent),
            "-pe",
            plan.parallel_environment,
            str(plan.nprocs),
            "-q",
            plan.queue,
            "-v",
            ",".join(f"{key}={value}" for key, value in sorted(environment.items())),
            str(runner_path.resolve()),
        ]
        intent = {
            "submission_id": plan.submission_id,
            "submission_sha256": plan.submission_sha256,
            "execution_budget": execution_budget.to_dict(),
            "qsub_command": command,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_text_atomic(
            intent_path,
            json.dumps(intent, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        match = _QSUB_JOB_ID.match(completed.stdout.strip())
        if match is None:
            raise SubmissionError(f"cannot parse qsub job ID: {completed.stdout!r}")
        scheduler_job_id = match.group(1)
        receipt = {
            "scheduler_job_id": scheduler_job_id,
            "qsub_stdout": completed.stdout.strip(),
            "submission_sha256": plan.submission_sha256,
            "submitted_at_utc": datetime.now(UTC).isoformat(),
            "execution_budget": execution_budget.to_dict(),
        }
        _write_text_atomic(
            receipt_path,
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        _append_ledger(
            plan=plan,
            ledger_path=ledger_path,
            scheduler_job_id=scheduler_job_id,
            budget_scope=budget_scope,
            repository_root=repository_root,
        )
        return {**plan.to_dict(), **receipt, "status": "SUBMITTED"}


def selected_ids_from_file(path: Path) -> set[str]:
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise SubmissionError(f"job ID file is empty: {path}")
    return ids
