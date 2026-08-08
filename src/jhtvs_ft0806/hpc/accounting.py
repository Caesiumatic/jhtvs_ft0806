"""SGE status and qacct accounting for immutable ORCA submissions."""

from __future__ import annotations

from collections import Counter
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
from typing import Mapping

from jhtvs_ft0806.hpc.submission import LEDGER_FIELDS, TASK_FIELDS, SubmissionError
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


ACCOUNTING_FIELDS = (
    "record_id",
    "submission_id",
    "scheduler_job_id",
    "array_task",
    "slots",
    "wallclock_s",
    "core_h",
    "exit_status",
    "failed",
    "logical_jobs",
    "completed_logical_jobs",
    "runner_status",
    "qacct_path",
    "qacct_sha256",
    "collected_at_utc",
)
_SEPARATOR = re.compile(r"^=+$")
_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+(?:\.\d+)?)d)?"
    r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?$"
)
_LEADING_INTEGER_RE = re.compile(r"^(?P<value>\d+)(?:\s|$)")


class AccountingError(SubmissionError):
    """Raised when scheduler evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class AccountingSummary:
    submission_id: str
    scheduler_job_id: str
    array_tasks: int
    logical_jobs: int
    completed_logical_jobs: int
    failed_array_tasks: int
    actual_core_h: Decimal
    ledger_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS" if self.ledger_status == "complete" else "FAILED",
            "submission_id": self.submission_id,
            "scheduler_job_id": self.scheduler_job_id,
            "array_tasks": self.array_tasks,
            "logical_jobs": self.logical_jobs,
            "completed_logical_jobs": self.completed_logical_jobs,
            "failed_array_tasks": self.failed_array_tasks,
            "actual_core_h": str(self.actual_core_h),
            "ledger_status": self.ledger_status,
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


def _read_task_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TASK_FIELDS:
            raise AccountingError(f"task table header mismatch: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise AccountingError(f"task table row has extra fields: {path}")
    return rows


def _qacct_blocks(text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _SEPARATOR.fullmatch(line):
            if current:
                blocks.append(current)
                current = {}
            continue
        key, separator, value = line.partition(" ")
        if not separator:
            continue
        value = value.strip()
        if key in current:
            raise AccountingError(f"duplicate qacct field {key!r} in one record")
        current[key] = value
    if current:
        blocks.append(current)
    return blocks


def _duration_seconds(value: str) -> Decimal:
    try:
        return Decimal(value)
    except ArithmeticError:
        pass
    match = _DURATION_RE.fullmatch(value)
    if match is None or not any(match.groupdict().values()):
        raise ValueError(f"invalid duration: {value!r}")
    units = {
        "days": Decimal("86400"),
        "hours": Decimal("3600"),
        "minutes": Decimal("60"),
        "seconds": Decimal("1"),
    }
    return sum(
        (Decimal(raw) * units[name] for name, raw in match.groupdict().items() if raw),
        Decimal("0"),
    )


def _leading_integer(value: str) -> int:
    match = _LEADING_INTEGER_RE.match(value)
    if match is None:
        raise ValueError(f"invalid annotated integer: {value!r}")
    return int(match.group("value"))


def parse_qacct(text: str, *, scheduler_job_id: str) -> dict[int, dict[str, str]]:
    records: dict[int, dict[str, str]] = {}
    for block in _qacct_blocks(text):
        if block.get("jobnumber") != scheduler_job_id:
            continue
        required = {"taskid", "slots", "ru_wallclock", "failed", "exit_status"}
        missing = required - set(block)
        if missing:
            raise AccountingError(
                f"qacct record {scheduler_job_id} lacks fields {sorted(missing)}"
            )
        try:
            task_id = int(block["taskid"])
            slots = int(block["slots"])
            wallclock = _duration_seconds(block["ru_wallclock"])
            failed = _leading_integer(block["failed"])
            exit_status = _leading_integer(block["exit_status"])
        except (ValueError, ArithmeticError) as exc:
            raise AccountingError(
                f"invalid numeric qacct record for {scheduler_job_id}: {block}"
            ) from exc
        if task_id < 1 or slots < 1 or wallclock < 0:
            raise AccountingError(
                f"invalid qacct task/slots/wallclock for {scheduler_job_id}: {block}"
            )
        if task_id in records:
            raise AccountingError(
                f"duplicate qacct array task {scheduler_job_id}.{task_id}"
            )
        block["failed"] = str(failed)
        block["exit_status"] = str(exit_status)
        records[task_id] = block
    if not records:
        raise AccountingError(f"qacct contains no records for job {scheduler_job_id}")
    return records


def _resolve_repository_path(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repository_root / path


def _output_complete(task: Mapping[str, str], repository_root: Path) -> bool:
    output = _resolve_repository_path(task["output_path"], repository_root)
    if not output.is_file() or output.is_symlink():
        return False
    if task["job_class"] == "mace_base_features":
        try:
            receipt = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return (
            receipt.get("status") == "PASS"
            and receipt.get("job_id") == task["job_id"]
            and receipt.get("input_sha256") == task["input_sha256"]
            and receipt.get("workflow_revision") == task["workflow_revision"]
            and receipt.get("method_id") == task["method_id"]
            and receipt.get("missing") == 0
        )
    if task["job_class"] == "mace_training":
        try:
            receipt = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return (
            receipt.get("status") == "PASS"
            and receipt.get("member_count") == 5
            and receipt.get("job_id") == task["job_id"]
            and receipt.get("input_sha256") == task["input_sha256"]
            and receipt.get("workflow_revision") == task["workflow_revision"]
            and receipt.get("method_id") == task["method_id"]
        )
    if task["job_class"] == "sigma_preopt":
        try:
            rows = read_csv_rows(output)
        except (OSError, ValueError):
            return False
        return (
            len(rows) == 1
            and rows[0].get("task_id") == task["job_id"]
            and rows[0].get("source_xyz_sha256") == task["input_sha256"]
            and rows[0].get("charge") == "2"
            and rows[0].get("uhf") == "0"
        )
    text = output.read_text(encoding="utf-8", errors="replace")
    return (
        f"# job_id: {task['job_id']}\n" in text
        and f"# input_sha256: {task['input_sha256']}\n" in text
        and "ORCA TERMINATED NORMALLY" in text
        and "ERROR !!!" not in text
    )


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _submission_row(ledger_path: Path, submission_id: str) -> dict[str, str]:
    rows = read_csv_rows(ledger_path)
    matching = [row for row in rows if row["submission_id"] == submission_id]
    if len(matching) != 1:
        raise AccountingError(
            f"submission ledger has {len(matching)} rows for {submission_id!r}"
        )
    return matching[0]


def collect_accounting(
    *,
    submission_id: str,
    repository_root: Path,
    ledger_path: Path,
    accounting_path: Path,
    qacct_file: Path | None = None,
    allow_partial: bool = False,
) -> AccountingSummary:
    repository_root = repository_root.resolve()
    ledger_path = ledger_path.resolve()
    accounting_path = accounting_path.resolve()
    ledger_row = _submission_row(ledger_path, submission_id)
    scheduler_job_id = ledger_row["scheduler_job_id"]
    task_table_path = _resolve_repository_path(
        ledger_row["task_table_path"], repository_root
    )
    if not task_table_path.is_file() or task_table_path.is_symlink():
        raise AccountingError("submission task table is missing or unsafe")
    if sha256_file(task_table_path) != ledger_row["task_table_sha256"]:
        raise AccountingError("submission task table hash mismatch")
    tasks = _read_task_rows(task_table_path)

    if qacct_file is None:
        completed = subprocess.run(
            ["qacct", "-j", scheduler_job_id],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        qacct_text = completed.stdout
    else:
        qacct_text = qacct_file.read_text(encoding="utf-8", errors="replace")
    records = parse_qacct(qacct_text, scheduler_job_id=scheduler_job_id)
    tasks_by_array: dict[int, list[dict[str, str]]] = {}
    for task in tasks:
        tasks_by_array.setdefault(int(task["array_task"]), []).append(task)
    expected_tasks = set(tasks_by_array)
    actual_tasks = set(records)
    unexpected_tasks = actual_tasks - expected_tasks
    missing_tasks = expected_tasks - actual_tasks
    if unexpected_tasks or (missing_tasks and not allow_partial):
        raise AccountingError(
            f"qacct task coverage mismatch; missing={sorted(missing_tasks)}, "
            f"unexpected={sorted(unexpected_tasks)}"
        )
    submission_dir = task_table_path.parent
    stored_qacct_path = submission_dir / "qacct.txt"
    if stored_qacct_path.exists():
        if stored_qacct_path.read_text(encoding="utf-8", errors="replace") != qacct_text:
            raise AccountingError("stored qacct evidence differs from current evidence")
    else:
        _write_text_atomic(stored_qacct_path, qacct_text)
    qacct_sha256 = sha256_file(stored_qacct_path)

    new_rows: list[dict[str, object]] = []
    failed_array_tasks = 0
    completed_logical_jobs = 0
    actual_core_h = Decimal("0")
    collected_at = datetime.now(UTC).isoformat()
    for array_task in sorted(actual_tasks):
        record = records[array_task]
        slots = int(record["slots"])
        if slots != int(ledger_row["nprocs"]):
            raise AccountingError(
                f"qacct slots mismatch for {scheduler_job_id}.{array_task}"
            )
        wallclock = _duration_seconds(record["ru_wallclock"])
        core_h = wallclock * Decimal(slots) / Decimal("3600")
        array_jobs = tasks_by_array[array_task]
        completed_jobs = sum(
            _output_complete(task, repository_root) for task in array_jobs
        )
        completed_logical_jobs += completed_jobs
        scheduler_clean = record["failed"] == "0" and record["exit_status"] == "0"
        if scheduler_clean and completed_jobs != len(array_jobs):
            raise AccountingError(
                f"clean qacct record lacks complete logical outputs for "
                f"{scheduler_job_id}.{array_task}"
            )
        runner_status = "complete" if scheduler_clean else "failed"
        if not scheduler_clean:
            failed_array_tasks += 1
        actual_core_h += core_h
        new_rows.append(
            {
                "record_id": f"{scheduler_job_id}.{array_task}",
                "submission_id": submission_id,
                "scheduler_job_id": scheduler_job_id,
                "array_task": array_task,
                "slots": slots,
                "wallclock_s": str(wallclock),
                "core_h": format(core_h, ".8f"),
                "exit_status": record["exit_status"],
                "failed": record["failed"],
                "logical_jobs": len(array_jobs),
                "completed_logical_jobs": completed_jobs,
                "runner_status": runner_status,
                "qacct_path": _relative_or_absolute(
                    stored_qacct_path, repository_root
                ),
                "qacct_sha256": qacct_sha256,
                "collected_at_utc": collected_at,
            }
        )

    failed_array_tasks += len(missing_tasks)
    ledger_status = "complete" if failed_array_tasks == 0 else "failed"
    lock_path = ledger_path.parent / ".submission.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_rows = (
            read_csv_rows(accounting_path) if accounting_path.is_file() else []
        )
        existing_by_id = {row["record_id"]: row for row in existing_rows}
        for row in new_rows:
            existing = existing_by_id.get(str(row["record_id"]))
            if existing is not None:
                stable_fields = set(ACCOUNTING_FIELDS) - {"collected_at_utc"}
                if any(existing[field] != str(row[field]) for field in stable_fields):
                    raise AccountingError(
                        f"accounting record changed for {row['record_id']}"
                    )
            else:
                existing_rows.append({key: str(value) for key, value in row.items()})
        write_csv_deterministic(
            accounting_path,
            ACCOUNTING_FIELDS,
            existing_rows,
            sort_by=("submission_id", "array_task"),
        )

        ledger_rows = read_csv_rows(ledger_path)
        matches = [
            row for row in ledger_rows if row["submission_id"] == submission_id
        ]
        if len(matches) != 1:
            raise AccountingError("submission ledger changed during collection")
        if any(
            matches[0][field] != ledger_row[field]
            for field in (
                "submission_sha256",
                "scheduler_job_id",
                "task_table_path",
                "task_table_sha256",
                "nprocs",
            )
        ):
            raise AccountingError("submission identity changed during collection")
        matches[0]["status"] = ledger_status
        write_csv_deterministic(
            ledger_path,
            LEDGER_FIELDS,
            ledger_rows,
            sort_by=("submission_id",),
        )

    return AccountingSummary(
        submission_id=submission_id,
        scheduler_job_id=scheduler_job_id,
        array_tasks=len(records),
        logical_jobs=len(tasks),
        completed_logical_jobs=completed_logical_jobs,
        failed_array_tasks=failed_array_tasks,
        actual_core_h=actual_core_h,
        ledger_status=ledger_status,
    )


def import_sigma_preopt_accounting(
    *,
    submission_id: str,
    completion_path: Path,
    accounting_path: Path,
) -> AccountingSummary:
    """Import a validated aggregate xTB array receipt into the common CPU budget."""

    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    accounting = payload.get("accounting", {})
    scheduler_job_id = str(payload.get("job_id", ""))
    task_count = int(payload.get("task_count", 0))
    scheduler_records = int(accounting.get("scheduler_records", 0))
    failed_tasks = int(accounting.get("failed_tasks", -1))
    normal_termination_tasks = int(payload.get("normal_termination_tasks", 0))
    if (
        payload.get("status") != "complete"
        or payload.get("source_and_output_hash_check") != "PASS"
        or not scheduler_job_id.isdigit()
        or task_count <= 0
        or scheduler_records != task_count
        or failed_tasks != 0
        or normal_termination_tasks != task_count
    ):
        raise AccountingError("sigma preoptimization completion is not clean and complete")
    actual_core_h = Decimal(str(accounting["actual_core_hours"]))
    if actual_core_h <= 0:
        raise AccountingError("sigma preoptimization actual core-hours must be positive")
    qacct_sha256 = str(accounting.get("qacct_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", qacct_sha256):
        raise AccountingError("sigma preoptimization qacct SHA256 is invalid")
    record = {
        "record_id": f"{scheduler_job_id}.aggregate",
        "submission_id": submission_id,
        "scheduler_job_id": scheduler_job_id,
        "array_task": "aggregate",
        "slots": "1",
        "wallclock_s": format(actual_core_h * Decimal("3600"), ".8f"),
        "core_h": format(actual_core_h, ".8f"),
        "exit_status": "0",
        "failed": "0",
        "logical_jobs": str(task_count),
        "completed_logical_jobs": str(task_count),
        "runner_status": "complete",
        "qacct_path": str(accounting.get("qacct_path", "")),
        "qacct_sha256": qacct_sha256,
        "collected_at_utc": str(payload.get("completed_at", "")),
    }
    if not record["qacct_path"] or not record["collected_at_utc"]:
        raise AccountingError("sigma preoptimization accounting provenance is incomplete")

    lock_path = accounting_path.parent / ".submission.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = read_csv_rows(accounting_path) if accounting_path.is_file() else []
        existing = next(
            (row for row in rows if row["record_id"] == record["record_id"]), None
        )
        if existing is not None and existing != record:
            raise AccountingError(
                f"accounting record changed for {record['record_id']}"
            )
        if existing is None:
            rows.append(record)
        write_csv_deterministic(
            accounting_path,
            ACCOUNTING_FIELDS,
            rows,
            sort_by=("submission_id", "array_task"),
        )
    return AccountingSummary(
        submission_id=submission_id,
        scheduler_job_id=scheduler_job_id,
        array_tasks=scheduler_records,
        logical_jobs=task_count,
        completed_logical_jobs=task_count,
        failed_array_tasks=0,
        actual_core_h=actual_core_h,
        ledger_status="complete",
    )


def submission_status(
    *, repository_root: Path, ledger_path: Path
) -> dict[str, object]:
    repository_root = repository_root.resolve()
    if not ledger_path.is_file():
        return {"status": "PASS", "submissions": 0, "logical_job_counts": {}}
    ledger_rows = read_csv_rows(ledger_path)
    job_counts: Counter[str] = Counter()
    submissions: list[dict[str, object]] = []
    for ledger_row in ledger_rows:
        task_table_path = _resolve_repository_path(
            ledger_row["task_table_path"], repository_root
        )
        if not task_table_path.is_file():
            raise AccountingError(f"missing task table: {task_table_path}")
        if sha256_file(task_table_path) != ledger_row["task_table_sha256"]:
            raise AccountingError(f"task table hash mismatch: {task_table_path}")
        local_counts: Counter[str] = Counter()
        for task in _read_task_rows(task_table_path):
            output = _resolve_repository_path(task["output_path"], repository_root)
            state = (
                "complete"
                if _output_complete(task, repository_root)
                else "incomplete_output"
                if output.exists()
                else "not_started"
            )
            local_counts[state] += 1
            job_counts[state] += 1
        submissions.append(
            {
                "submission_id": ledger_row["submission_id"],
                "scheduler_job_id": ledger_row["scheduler_job_id"],
                "ledger_status": ledger_row["status"],
                "logical_job_counts": dict(sorted(local_counts.items())),
            }
        )
    return {
        "status": "PASS",
        "submissions": len(ledger_rows),
        "logical_job_counts": dict(sorted(job_counts.items())),
        "submission_details": submissions,
    }
