from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess

import pytest

from jhtvs_ft0806.hpc.submission import (
    SubmissionError,
    check_budget,
    execute_submission,
    prepare_submission,
)
from jhtvs_ft0806.orca.decks import build_decks
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import write_csv_deterministic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"
FIXTURE_XYZ = Path(__file__).resolve().parent / "fixtures" / "orca" / "small.xyz"


def _fake_repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repository"
    shutil.copytree(SPEC_DIR, root / "spec")
    shutil.copy2(REPOSITORY_ROOT / "AGENTS.md", root / "AGENTS.md")
    (root / "hpc").mkdir()
    shutil.copy2(REPOSITORY_ROOT / "hpc" / "run_orca.sh", root / "hpc" / "run_orca.sh")
    geometry = root / "runs" / "geometry" / "fixture.xyz"
    geometry.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_XYZ, geometry)
    index = root / "data" / "resolved" / "geometry_index.csv"
    write_csv_deterministic(
        index,
        ("geometry_key", "status", "reason", "xyz_path", "xyz_sha256"),
        [
            {
                "geometry_key": "tier1:redox:A002:S001:q0:m2",
                "status": "resolved",
                "reason": "",
                "xyz_path": "runs/geometry/fixture.xyz",
                "xyz_sha256": sha256_file(geometry),
            }
        ],
    )
    manifest = root / "data" / "resolved" / "deck_manifest.csv"
    build_decks(
        spec_dir=root / "spec",
        geometry_index_path=index,
        run_dir=root / "runs" / "orca",
        manifest_path=manifest,
        selected_job_ids={"SP0001", "SP0033"},
    )
    return root, index, manifest


def test_submission_preflight_bundles_same_geometry_without_losing_job_ids(
    tmp_path: Path,
) -> None:
    root, index, manifest = _fake_repository(tmp_path)

    plan = prepare_submission(
        submission_id="fixture-wave",
        selected_job_ids={"SP0001", "SP0033"},
        spec_dir=root / "spec",
        geometry_index_path=index,
        deck_manifest_path=manifest,
        submissions_root=root / "runs" / "hpc" / "submissions",
        accounting_path=root / "data" / "resolved" / "accounting.csv",
        ledger_path=root / "data" / "resolved" / "submission_ledger.csv",
        runner_path=root / "hpc" / "run_orca.sh",
    )

    with plan.task_table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert plan.job_count == 2
    assert plan.array_task_count == 1
    assert {row["job_id"] for row in rows} == {"SP0001", "SP0033"}
    assert {row["array_task"] for row in rows} == {"1"}
    assert plan.budget.projected_core_h == plan.planned_core_h
    assert plan.preflight_report_path.is_file()

    repeated = prepare_submission(
        submission_id="fixture-wave",
        selected_job_ids={"SP0001", "SP0033"},
        spec_dir=root / "spec",
        geometry_index_path=index,
        deck_manifest_path=manifest,
        submissions_root=root / "runs" / "hpc" / "submissions",
        accounting_path=root / "data" / "resolved" / "accounting.csv",
        ledger_path=root / "data" / "resolved" / "submission_ledger.csv",
        runner_path=root / "hpc" / "run_orca.sh",
    )
    assert repeated.submission_sha256 == plan.submission_sha256


def test_budget_guard_counts_consumed_and_active_planned_cost(tmp_path: Path) -> None:
    accounting = tmp_path / "accounting.csv"
    ledger = tmp_path / "ledger.csv"
    write_csv_deterministic(
        accounting,
        ("record_id", "core_h"),
        [{"record_id": "used", "core_h": "7000"}],
    )
    write_csv_deterministic(
        ledger,
        ("submission_id", "status", "planned_core_h"),
        [
            {
                "submission_id": "active",
                "status": "queued",
                "planned_core_h": "900",
            },
            {
                "submission_id": "done",
                "status": "complete",
                "planned_core_h": "500",
            },
        ],
    )

    with pytest.raises(SubmissionError, match="budget guard blocks"):
        check_budget(
            spec_dir=SPEC_DIR,
            accounting_path=accounting,
            ledger_path=ledger,
            proposed_core_h=Decimal("101"),
        )

    allowed = check_budget(
        spec_dir=SPEC_DIR,
        accounting_path=accounting,
        ledger_path=ledger,
        proposed_core_h=Decimal("100"),
    )
    assert allowed.projected_core_h == Decimal("8000")


def test_execute_submission_writes_receipt_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, index, manifest = _fake_repository(tmp_path)
    ledger = root / "data" / "resolved" / "submission_ledger.csv"
    keyword_args = {
        "submission_id": "idempotent-wave",
        "selected_job_ids": {"SP0001", "SP0033"},
        "spec_dir": root / "spec",
        "geometry_index_path": index,
        "deck_manifest_path": manifest,
        "submissions_root": root / "runs" / "hpc" / "submissions",
        "accounting_path": root / "data" / "resolved" / "accounting.csv",
        "ledger_path": ledger,
        "runner_path": root / "hpc" / "run_orca.sh",
    }
    plan = prepare_submission(**keyword_args)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="12345.1-1:1\n", stderr=""
        ),
    )
    first = execute_submission(
        plan=plan,
        runner_path=root / "hpc" / "run_orca.sh",
        spec_dir=root / "spec",
        accounting_path=root / "data" / "resolved" / "accounting.csv",
        ledger_path=ledger,
    )
    second = execute_submission(
        plan=plan,
        runner_path=root / "hpc" / "run_orca.sh",
        spec_dir=root / "spec",
        accounting_path=root / "data" / "resolved" / "accounting.csv",
        ledger_path=ledger,
    )
    resumed_plan = prepare_submission(**keyword_args)

    assert first["status"] == "SUBMITTED"
    assert second["status"] == "ALREADY_SUBMITTED"
    assert resumed_plan.submission_sha256 == plan.submission_sha256
    assert (plan.submission_dir / "qsub_intent.json").is_file()
    assert len(list(csv.DictReader(ledger.open(encoding="utf-8")))) == 1


def test_orca_runner_pins_lop_modules_mpi_and_deck_resource_checks() -> None:
    runner = (REPOSITORY_ROOT / "hpc" / "run_orca.sh").read_text(
        encoding="utf-8"
    )

    assert "module load openmpi/4.1.8 orca/6.1.0-418" in runner
    assert 'export OMPI_MCA_btl="^sm"' in runner
    assert "export OMPI_MCA_mpi_yield_when_idle=1" in runner
    assert "export OMPI_MCA_btl_vader_single_copy_mechanism=none" in runner
    assert 'grep -Fqx "%pal nprocs 8 end" "$INPUT"' in runner
    assert 'grep -Fqx "%maxcore 3000" "$INPUT"' in runner
    assert "ORCA TERMINATED NORMALLY" in runner
