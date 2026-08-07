from __future__ import annotations

import shutil
from pathlib import Path

from jhtvs_ft0806.cli import COMMANDS, build_parser, main
from jhtvs_ft0806.spec_validation import validate_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"


def test_supplied_spec_passes_all_acceptance_invariants() -> None:
    report = validate_spec(SPEC_DIR)

    assert report.ok, report.to_json()
    assert report.checks["manifest_entries"] == 26
    assert report.checks["calibration_split_counts"] == {
        "test": 14,
        "train": 60,
        "val": 14,
    }
    assert report.checks["complete_sp_reaction_medium_cells"] == 403
    assert report.checks["complete_optfreq_reaction_medium_cells"] == 50
    assert report.checks["expected_fullspace_predictions"] == 5300
    assert report.checks["planned_core_hours"] == {
        "sp": "1601.30",
        "optfreq": "2289.20",
        "total": "3890.50",
    }


def test_manifest_detects_byte_level_drift(tmp_path: Path) -> None:
    copied_spec = tmp_path / "spec"
    shutil.copytree(SPEC_DIR, copied_spec)
    shutil.copy2(REPOSITORY_ROOT / "AGENTS.md", tmp_path / "AGENTS.md")
    target = copied_spec / "03_ACCEPTANCE.md"
    target.write_bytes(target.read_bytes() + b"\n")

    report = validate_spec(copied_spec)

    assert not report.ok
    assert {issue.code for issue in report.issues} >= {
        "manifest_bytes",
        "manifest_sha256",
    }


def test_every_required_cli_command_is_registered() -> None:
    help_text = build_parser().format_help()
    for command in COMMANDS:
        assert command in help_text


def test_validate_spec_cli_returns_success(capsys) -> None:
    result = main(["--log-level", "ERROR", "validate-spec", "--spec-dir", str(SPEC_DIR)])
    captured = capsys.readouterr()

    assert result == 0
    assert '"status": "PASS"' in captured.out
