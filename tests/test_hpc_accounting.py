from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from jhtvs_ft0806.hpc.accounting import (
    AccountingError,
    collect_accounting,
    import_sigma_preopt_accounting,
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


def test_parse_qacct_normalizes_lop_killed_status_annotations() -> None:
    text = QACCT_FIXTURE.read_text(encoding="utf-8")
    text = text.replace("failed       0", "failed       100 : assumedly after job")
    text = text.replace("exit_status  0", "exit_status  137                  (Killed)")

    records = parse_qacct(text, scheduler_job_id="12345")

    assert records[1]["failed"] == "100"
    assert records[1]["exit_status"] == "137"


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
    assert status["missing_task_tables"] == 0
    assert status["submission_details"][0]["task_table_available"] is True


def test_submission_status_reports_cluster_local_task_tables_as_unavailable(
    tmp_path: Path,
) -> None:
    root, ledger, _, _ = _accounting_fixture(tmp_path)
    task_table = root / "runs" / "hpc" / "submissions" / "fixture" / "tasks.tsv"
    task_table.unlink()

    status = submission_status(repository_root=root, ledger_path=ledger)

    assert status["status"] == "PARTIAL"
    assert status["missing_task_tables"] == 1
    assert status["logical_job_counts"] == {"unavailable_local": 1}
    assert status["submission_details"] == [
        {
            "submission_id": "fixture",
            "scheduler_job_id": "12345",
            "ledger_status": "submitted",
            "task_table_available": False,
            "logical_job_counts": {"unavailable_local": 1},
        }
    ]


def test_collect_accounting_accepts_lop_duration_suffixes(
    tmp_path: Path,
) -> None:
    root, ledger, accounting, _ = _accounting_fixture(tmp_path)
    lop_qacct = tmp_path / "qacct_lop.txt"
    lop_qacct.write_text(
        QACCT_FIXTURE.read_text(encoding="utf-8").replace(
            "ru_wallclock 90.5", "ru_wallclock 1m30.5s"
        ),
        encoding="utf-8",
    )

    summary = collect_accounting(
        submission_id="fixture",
        repository_root=root,
        ledger_path=ledger,
        accounting_path=accounting,
        qacct_file=lop_qacct,
    )

    assert summary.actual_core_h == Decimal("90.5") * Decimal("8") / Decimal(
        "3600"
    )


def test_collect_accounting_accepts_complete_sigma_tsv_output(
    tmp_path: Path,
) -> None:
    root, ledger, accounting, output = _accounting_fixture(tmp_path)
    optimized = output.with_name("xtbopt.xyz")
    optimized.write_text("1\nfixture\nH 0 0 0\n", encoding="utf-8")
    output = output.with_name("task_status.tsv")
    output.write_text(
        "task_id\tsource_xyz_sha256\toptimized_xyz_sha256\t"
        "topology_sha256\tcharge\tuhf\tepsilon\n"
        f"sigma-preopt-D001-S001\tfixture-input-sha\t{sha256_file(optimized)}\t"
        "fixture-topology-sha\t2\t0\t35.688\n",
        encoding="utf-8",
    )
    task_table = root / "runs" / "hpc" / "submissions" / "fixture" / "tasks.tsv"
    with task_table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[0].update(
        {
            "job_id": "sigma-preopt-D001-S001",
            "job_class": "sigma_preopt",
            "output_path": str(output.relative_to(root)),
            "nprocs": "1",
        }
    )
    with task_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    ledger_rows = read_csv_rows(ledger)
    ledger_rows[0]["nprocs"] = "1"
    ledger_rows[0]["task_table_sha256"] = sha256_file(task_table)
    write_csv_deterministic(ledger, LEDGER_FIELDS, ledger_rows)
    qacct = tmp_path / "qacct_sigma.txt"
    qacct.write_text(
        QACCT_FIXTURE.read_text(encoding="utf-8").replace("slots        8", "slots        1"),
        encoding="utf-8",
    )

    summary = collect_accounting(
        submission_id="fixture",
        repository_root=root,
        ledger_path=ledger,
        accounting_path=accounting,
        qacct_file=qacct,
    )

    assert summary.logical_jobs == 1
    assert summary.completed_logical_jobs == 1
    assert summary.ledger_status == "complete"


def test_partial_qacct_requires_opt_in_and_accounts_started_tasks(
    tmp_path: Path,
) -> None:
    root, ledger, accounting, _ = _accounting_fixture(tmp_path)
    task_table = root / "runs" / "hpc" / "submissions" / "fixture" / "tasks.tsv"
    with task_table.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "2",
                "1",
                "SP0002",
                "diagnostic_gas_sp",
                "runs/orca/sp/SP0002/SP0002.inp",
                "fixture-input-sha-2",
                "runs/orca/sp/SP0002/SP0002.out",
                "8",
                "1.74",
                "jhtvs-ft0806-sp-v3",
                "fixture-method",
            )
        )
    ledger_rows = read_csv_rows(ledger)
    ledger_rows[0]["job_count"] = "2"
    ledger_rows[0]["array_task_count"] = "2"
    ledger_rows[0]["task_table_sha256"] = sha256_file(task_table)
    write_csv_deterministic(ledger, LEDGER_FIELDS, ledger_rows)

    with pytest.raises(AccountingError, match="qacct task coverage mismatch"):
        collect_accounting(
            submission_id="fixture",
            repository_root=root,
            ledger_path=ledger,
            accounting_path=accounting,
            qacct_file=QACCT_FIXTURE,
        )

    summary = collect_accounting(
        submission_id="fixture",
        repository_root=root,
        ledger_path=ledger,
        accounting_path=accounting,
        qacct_file=QACCT_FIXTURE,
        allow_partial=True,
    )

    assert summary.array_tasks == 1
    assert summary.logical_jobs == 2
    assert summary.failed_array_tasks == 1
    assert summary.ledger_status == "failed"
    assert len(read_csv_rows(accounting)) == 1
    assert read_csv_rows(ledger)[0]["status"] == "failed"


def test_import_sigma_preopt_accounting_is_complete_and_idempotent(
    tmp_path: Path,
) -> None:
    completion = tmp_path / "completion.json"
    completion.write_text(
        """{
  "status": "complete",
  "job_id": "423357",
  "task_count": 2,
  "normal_termination_tasks": 2,
  "source_and_output_hash_check": "PASS",
  "completed_at": "2026-08-07T13:24:17-05:00",
  "accounting": {
    "actual_core_hours": "1.25000000",
    "failed_tasks": 0,
    "scheduler_records": 2,
    "qacct_path": "runs/geometry/qacct_423357.txt",
    "qacct_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}\n""",
        encoding="utf-8",
    )
    accounting = tmp_path / "accounting.csv"

    first = import_sigma_preopt_accounting(
        submission_id="sigma-v1",
        completion_path=completion,
        accounting_path=accounting,
    )
    second = import_sigma_preopt_accounting(
        submission_id="sigma-v1",
        completion_path=completion,
        accounting_path=accounting,
    )
    row = read_csv_rows(accounting)[0]

    assert first.to_dict() == second.to_dict()
    assert first.actual_core_h == Decimal("1.25000000")
    assert row["record_id"] == "423357.aggregate"
    assert row["wallclock_s"] == "4500.00000000"
    assert row["completed_logical_jobs"] == "2"
