"""Fail-closed pre-submission audit for manifest-bound ORCA decks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from jhtvs_ft0806.orca.decks import (
    THERMOCHEMISTRY_CONVENTION_ID,
    render_optfreq_deck,
    render_sp_deck,
)
from jhtvs_ft0806.orca.smd import smd_payload_sha256
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows
from jhtvs_ft0806.spec_validation import validate_spec


@dataclass(slots=True)
class DeckAuditReport:
    checks: dict[str, object] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS" if self.ok else "FAIL",
            "checks": self.checks,
            "issues": self.issues,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
        )


def _unique_by(
    rows: list[dict[str, str]], key: str, *, table: str, report: DeckAuditReport
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key]
        if value in result:
            report.issues.append(f"{table}: duplicate {key} {value!r}")
        result[value] = row
    return result


def _resolve_path(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repository_root / path


def _write_report_atomic(path: Path, report: DeckAuditReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(report.to_json() + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _logical_jobs(spec_dir: Path) -> list[tuple[str, dict[str, str], str]]:
    jobs: list[tuple[str, dict[str, str], str]] = []
    for row in read_csv_rows(spec_dir / "sp_job_manifest.csv"):
        jobs.append((row["job_class"], row, row["geometry_key"]))
    for row in read_csv_rows(spec_dir / "optfreq_job_manifest.csv"):
        jobs.append(("optfreq", row, row["start_geometry_key"]))
    return jobs


def _audit_manifest_fields(
    manifest: Mapping[str, str],
    job: Mapping[str, str],
    *,
    job_class: str,
    geometry_key: str,
    expected_smd_sha: str,
    report: DeckAuditReport,
) -> None:
    expected = {
        "job_class": job_class,
        "state_id": job["state_id"],
        "solvent_id": job["solvent_id"],
        "geometry_key": geometry_key,
        "smd_payload_sha256": expected_smd_sha,
        "workflow_revision": job["workflow_revision"],
        "method_id": job["method_id"],
        "thermochemistry_convention_id": (
            THERMOCHEMISTRY_CONVENTION_ID
            if job_class == "optfreq"
            else "not_applicable"
        ),
        "nprocs": job["nprocs"],
        "maxcore_mb_per_rank": job["maxcore_mb_per_rank"],
        "planning_core_h": job["planning_core_h"],
    }
    for field, expected_value in expected.items():
        if manifest[field] != expected_value:
            report.issues.append(
                f"{job['job_id']}: deck manifest {field}={manifest[field]!r}; "
                f"expected {expected_value!r}"
            )


def audit_decks(
    *,
    spec_dir: Path,
    geometry_index_path: Path,
    deck_manifest_path: Path,
    selected_job_ids: set[str] | None = None,
    report_path: Path | None = None,
    require_output_absent: bool = True,
) -> DeckAuditReport:
    """Re-render selected decks and require byte-for-byte provenance agreement."""

    spec_dir = spec_dir.resolve()
    geometry_index_path = geometry_index_path.resolve()
    deck_manifest_path = deck_manifest_path.resolve()
    repository_root = spec_dir.parent
    report = DeckAuditReport()

    validation = validate_spec(spec_dir)
    if not validation.ok:
        report.issues.append("scientific specification validation failed")

    manifest_rows = read_csv_rows(deck_manifest_path)
    manifest_by_id = _unique_by(
        manifest_rows, "job_id", table="deck manifest", report=report
    )
    geometry_by_key = _unique_by(
        read_csv_rows(geometry_index_path),
        "geometry_key",
        table="geometry index",
        report=report,
    )
    solvent_by_id = _unique_by(
        read_csv_rows(spec_dir / "solvent_smd_registry.csv"),
        "solvent_id",
        table="SMD registry",
        report=report,
    )
    registry_sha256 = sha256_file(spec_dir / "solvent_smd_registry.csv")
    logical_jobs = _logical_jobs(spec_dir)
    logical_by_id = {item[1]["job_id"]: item for item in logical_jobs}

    if selected_job_ids is None:
        selected_job_ids = {
            row["job_id"] for row in manifest_rows if row["status"] == "ready"
        }
    unknown = selected_job_ids - set(logical_by_id)
    absent = selected_job_ids - set(manifest_by_id)
    if unknown:
        report.issues.append(f"unknown selected job IDs: {sorted(unknown)}")
    if absent:
        report.issues.append(f"selected jobs absent from deck manifest: {sorted(absent)}")

    class_counts: Counter[str] = Counter()
    smd_mode_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    input_hashes: list[dict[str, str]] = []
    total_planned_core_h = Decimal("0")

    for job_id in sorted(selected_job_ids & set(logical_by_id) & set(manifest_by_id)):
        job_class, job, geometry_key = logical_by_id[job_id]
        manifest = manifest_by_id[job_id]
        class_counts[job_class] += 1
        method_counts[job["method_id"]] += 1
        total_planned_core_h += Decimal(job["planning_core_h"])

        if manifest["status"] != "ready":
            report.issues.append(
                f"{job_id}: deck status {manifest['status']!r} is not submit-ready"
            )
            continue
        geometry = geometry_by_key.get(geometry_key)
        if geometry is None or geometry["status"] != "resolved":
            report.issues.append(f"{job_id}: resolved geometry is unavailable")
            continue
        geometry_path = _resolve_path(geometry["xyz_path"], repository_root)
        if not geometry_path.is_file() or geometry_path.is_symlink():
            report.issues.append(f"{job_id}: geometry is missing or unsafe")
            continue
        actual_geometry_sha = sha256_file(geometry_path)
        if actual_geometry_sha != geometry["xyz_sha256"]:
            report.issues.append(f"{job_id}: geometry index SHA-256 mismatch")
            continue
        if manifest["geometry_sha256"] != actual_geometry_sha:
            report.issues.append(f"{job_id}: deck manifest geometry SHA-256 mismatch")

        solvent = None
        expected_smd_sha = ""
        if job_class == "diagnostic_gas_sp":
            smd_mode_counts["gas"] += 1
        else:
            solvent = solvent_by_id.get(job["solvent_id"])
            if solvent is None:
                report.issues.append(f"{job_id}: solvent is absent from SMD registry")
                continue
            smd_mode_counts[solvent["orca_smd_mode"]] += 1
            expected_smd_sha = smd_payload_sha256(solvent)
        _audit_manifest_fields(
            manifest,
            job,
            job_class=job_class,
            geometry_key=geometry_key,
            expected_smd_sha=expected_smd_sha,
            report=report,
        )

        if job_class == "optfreq":
            if solvent is None:
                report.issues.append(f"{job_id}: Opt/Freq SMD routing is missing")
                continue
            expected_text = render_optfreq_deck(
                job,
                geometry,
                geometry_path,
                solvent,
                registry_sha256=registry_sha256,
            )
        else:
            expected_text = render_sp_deck(
                job,
                geometry,
                geometry_path,
                solvent,
                registry_sha256=registry_sha256,
            )

        input_path = _resolve_path(manifest["input_path"], repository_root)
        if not input_path.is_file() or input_path.is_symlink():
            report.issues.append(f"{job_id}: ORCA input is missing or unsafe")
            continue
        actual_text = input_path.read_text(encoding="utf-8")
        if actual_text != expected_text:
            report.issues.append(f"{job_id}: ORCA input differs from exact re-render")
        actual_input_sha = sha256_file(input_path)
        if actual_input_sha != manifest["input_sha256"]:
            report.issues.append(f"{job_id}: ORCA input SHA-256 mismatch")
        output_path = input_path.with_suffix(".out")
        if require_output_absent and output_path.exists():
            report.issues.append(f"{job_id}: output already exists; refusing resubmission")
        input_hashes.append({"job_id": job_id, "input_sha256": actual_input_sha})

    report.checks = {
        "selected_jobs": len(selected_job_ids),
        "audited_jobs": len(input_hashes),
        "job_class_counts": dict(sorted(class_counts.items())),
        "smd_mode_counts": dict(sorted(smd_mode_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "nprocs": sorted(
            {
                manifest_by_id[job_id]["nprocs"]
                for job_id in selected_job_ids
                if job_id in manifest_by_id
            }
        ),
        "maxcore_mb_per_rank": sorted(
            {
                manifest_by_id[job_id]["maxcore_mb_per_rank"]
                for job_id in selected_job_ids
                if job_id in manifest_by_id
            }
        ),
        "planned_core_h": str(total_planned_core_h),
        "spec_sha256": sha256_file(spec_dir / "01_SCIENTIFIC_SPEC.md"),
        "smd_registry_sha256": registry_sha256,
        "geometry_index_sha256": sha256_file(geometry_index_path),
        "deck_manifest_sha256": sha256_file(deck_manifest_path),
        "input_bundle_sha256": content_hash(input_hashes),
        "output_absence_required": require_output_absent,
    }
    if report_path is not None:
        _write_report_atomic(report_path.resolve(), report)
    return report
