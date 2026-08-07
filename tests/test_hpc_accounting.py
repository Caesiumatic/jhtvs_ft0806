from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

from jhtvs_ft0806.hpc.accounting import (
    collect_accounting,
    parse_qacct,
    submission_status,
)
from jhtvs_ft0806.hpc.submission import LEDGER_FIELDS
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QACCT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "hpc" / "qacct_complete.txt"
)


def _accounting_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "repository"
    run_dir = root / "runs" / "orca" / "sp" / "SP0001"
    run_dir.mkdir(parents=True)
    output = run_dir / "SP0001.out"
    output.write_text(
        "# job_id: SP0001\n"
        "# input_sha256: fixture-input-sha\n"
        "ORCA TERMINATED NORMALLY\n",
        encoding="utf-8",
    )
    submission_dir = root / "runs" / "hpc" / "submissions" / "fixture"
    submission_dir.mkdir(parents=True)
    task_table = submission_dir / "tasks.tsv"
    fields = (
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
    with task_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "array_task": "1",
                "sequence": "1",
                "job_id": "SP0001",
                "job_class": "diagnostic_gas_sp",
                "input_path": "runs/orca/sp/SP0001/SP0001.inp",
                "input_sha256": "fixture-input-sha",
                "output_path": "runs/orca/sp/SP0001/SP0001.out",
                "nprocs": "8",
                "planning_core_h": "1.74",
                "workflow_revision": "jhtvs-ft0806-sp-v3",
                "method_id": "fixture-method",
            }
        )
    ledger = root / "data" / "resolved" / "submission_ledger.csv"
    write_csv_deterministic(
        ledger,
        LEDGER_FIELDS,
        [
            {
                "submission_id": "fixture",
                "submission_sha256": "fixture-submission-sha",
                "scheduler_job_id": "12345",
                "status": "submitted",
                "submitted_at_utc": "2026-08-07T00:00:00+00:00",
                "job_count": "1",
                "array_task_count": "1",
                "planned_core_h": "1.74",
                "task_table_path": "runs/hpc/submissions/fixture/tasks.tsv",
                "task_table_sha256": sha256_file(task_table),
                "preflight_report_path": "runs/hpc/submissions/fixture/preflight.json",
                "preflight_report_sha256": "fixture-preflight-sha",
                "budget_scope": "first_round",
                "queue": "amd16smt",
                "parallel_environment": "orte",
                "nprocs": "8",
                "max_concurrent": "1",
            }
        ],
    )
    return root, ledger, root / "data" / "resolved" / "accounting.csv", output


def test_parse_qacct_requires_array_task_records() -> None:
    records = parse_qacct(
        QACCT_FIXTURE.read_text(encoding="utf-8"), scheduler_job_id="12345"
    )

    assert set(records) == {1}
    assert records[1]["slots"] == "8"
    assert records[1]["ru_wallclock"] == "90.5"


def test_collect_accounting_uses_qacct_slots_wallclock_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root, ledger, accounting, _ = _accounting_fixture(tmp_path)

    first = collect_accounting(
        submission_id="fixture",
        repository_root=root,
        ledger_path=ledger,
        accounting_path=accounting,
        qacct_file=QACCT_FIXTURE,
    )
    second = collect_accounting(
        submission_id="fixture",
        repository_root=root,
        ledger_path=ledger,
        accounting_path=accounting,
        qacct_file=QACCT_FIXTURE,
    )
    row = read_csv_rows(accounting)[0]

    assert first.to_dict() == second.to_dict()
    assert first.actual_core_h == Decimal("90.5") * Decimal("8") / Decimal("3600")
    assert row["core_h"] == "0.20111111"
    assert row["runner_status"] == "complete"
    assert read_csv_rows(ledger)[0]["status"] == "complete"
    status = submission_status(repository_root=root, ledger_path=ledger)
    assert status["logical_job_counts"] == {"complete": 1}
