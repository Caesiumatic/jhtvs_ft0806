"""Manifest-driven ORCA 6.1 deck generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

from jhtvs_ft0806.geometry.xyz import read_xyz
from jhtvs_ft0806.orca.smd import render_smd_block, smd_payload_sha256
from jhtvs_ft0806.provenance import content_hash, csv_record_sha256, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic
from jhtvs_ft0806.spec_validation import validate_spec

THERMOCHEMISTRY_CONVENTION_ID = "orca_298.15K_1atm_QuasiRRHO_project_1M_v1"

DECK_MANIFEST_FIELDS = (
    "job_id",
    "job_class",
    "state_id",
    "solvent_id",
    "input_path",
    "input_sha256",
    "geometry_key",
    "geometry_sha256",
    "smd_registry_row_sha256",
    "smd_payload_sha256",
    "exact_reuse_key",
    "workflow_revision",
    "method_id",
    "thermochemistry_convention_id",
    "nprocs",
    "maxcore_mb_per_rank",
    "planning_core_h",
    "status",
    "reason",
)


class DeckGenerationError(ValueError):
    """Raised when a logical job cannot be rendered without scientific drift."""


@dataclass(frozen=True, slots=True)
class DeckBuildSummary:
    total: int
    ready: int
    waiting_geometry: int
    existing_outputs: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": "PASS" if self.waiting_geometry == 0 else "INCOMPLETE",
            "total": self.total,
            "ready": self.ready,
            "waiting_geometry": self.waiting_geometry,
            "existing_outputs": self.existing_outputs,
        }


def _geometry_lines(path: Path) -> str:
    return "\n".join(
        f"{atom.symbol:<2s} {atom.x:18.10f} {atom.y:18.10f} {atom.z:18.10f}"
        for atom in read_xyz(path)
    )


def _metadata(
    job: Mapping[str, str],
    geometry: Mapping[str, str],
    *,
    job_class: str,
    smd_sha256: str,
    smd_registry_row_sha256: str,
    exact_reuse_key: str,
    registry_sha256: str,
) -> str:
    values = {
        "workflow_revision": job["workflow_revision"],
        "job_id": job["job_id"],
        "job_class": job_class,
        "state_id": job["state_id"],
        "solvent_id": job["solvent_id"],
        "formal_charge": job["formal_charge"],
        "multiplicity": job["multiplicity"],
        "geometry_key": geometry["geometry_key"],
        "geometry_sha256": geometry["xyz_sha256"],
        "method_id": job["method_id"],
        "smd_registry_sha256": registry_sha256,
        "smd_registry_row_sha256": smd_registry_row_sha256 or "not_applicable",
        "smd_payload_sha256": smd_sha256 or "not_applicable",
        "exact_reuse_key": exact_reuse_key,
        "thermochemistry_convention_id": (
            THERMOCHEMISTRY_CONVENTION_ID
            if job_class == "optfreq"
            else "not_applicable"
        ),
    }
    return "".join(f"# {key}: {value}\n" for key, value in values.items())


def build_exact_reuse_key(
    job: Mapping[str, str],
    geometry: Mapping[str, str],
    *,
    job_class: str,
    smd_registry_row_sha256: str,
    smd_payload_sha256: str,
) -> str:
    """Bind reuse to scientific identity, not ORCA's display-only solvent label."""

    thermochemistry = (
        THERMOCHEMISTRY_CONVENTION_ID
        if job_class == "optfreq"
        else "not_applicable"
    )
    return content_hash(
        {
            "schema": "jhtvs_ft0806_exact_reuse_v1",
            "job_class": job_class,
            "state_id": job["state_id"],
            "medium_id": job["solvent_id"],
            "geometry_sha256": geometry["xyz_sha256"],
            "smd_registry_row_sha256": (
                smd_registry_row_sha256 or "not_applicable"
            ),
            "input_payload_sha256": smd_payload_sha256 or "not_applicable",
            "method_id": job["method_id"],
            "workflow_revision": job["workflow_revision"],
            "thermochemistry_convention_id": thermochemistry,
        }
    )


def render_sp_deck(
    job: Mapping[str, str],
    geometry: Mapping[str, str],
    xyz_path: Path,
    solvent: Mapping[str, str] | None,
    *,
    registry_sha256: str,
    registry_row_sha256: str = "",
) -> str:
    is_gas = job["job_class"] == "diagnostic_gas_sp"
    if is_gas != (solvent is None):
        raise DeckGenerationError(f"{job['job_id']}: gas/SMD solvent routing mismatch")
    smd_block = "" if solvent is None else render_smd_block(solvent)
    smd_sha = "" if solvent is None else smd_payload_sha256(solvent)
    reuse_key = build_exact_reuse_key(
        job,
        geometry,
        job_class=job["job_class"],
        smd_registry_row_sha256=registry_row_sha256,
        smd_payload_sha256=smd_sha,
    )
    header = _metadata(
        job,
        geometry,
        job_class=job["job_class"],
        smd_sha256=smd_sha,
        smd_registry_row_sha256=registry_row_sha256,
        exact_reuse_key=reuse_key,
        registry_sha256=registry_sha256,
    )
    return (
        header
        + f"! {job['functional']} {job['basis']} def2/J RIJCOSX TightSCF DEFGRID3\n"
        + f"%pal nprocs {job['nprocs']} end\n"
        + f"%maxcore {job['maxcore_mb_per_rank']}\n"
        + smd_block
        + f"* xyz {job['formal_charge']} {job['multiplicity']}\n"
        + _geometry_lines(xyz_path)
        + "\n*\n"
    )


def render_optfreq_deck(
    job: Mapping[str, str],
    geometry: Mapping[str, str],
    xyz_path: Path,
    solvent: Mapping[str, str],
    *,
    registry_sha256: str,
    registry_row_sha256: str = "",
) -> str:
    smd = render_smd_block(solvent)
    smd_sha = smd_payload_sha256(solvent)
    reuse_key = build_exact_reuse_key(
        job,
        geometry,
        job_class="optfreq",
        smd_registry_row_sha256=registry_row_sha256,
        smd_payload_sha256=smd_sha,
    )
    header = _metadata(
        job,
        geometry,
        job_class="optfreq",
        smd_sha256=smd_sha,
        smd_registry_row_sha256=registry_row_sha256,
        exact_reuse_key=reuse_key,
        registry_sha256=registry_sha256,
    )
    first_step = (
        "NewStep\n"
        f"! {job['functional']} {job['optfreq_basis']} def2/J RIJCOSX "
        "TightSCF DEFGRID3 Opt Freq\n"
        + smd
        + "%freq\n"
        "  Temp 298.15\n"
        "  Pressure 1.0\n"
        "  QuasiRRHO true\n"
        "end\n"
        "%geom\n"
        "  MaxIter 200\n"
        "  Convergence tight\n"
        "end\n"
        "StepEnd\n"
    )
    second_step = (
        "NewStep\n"
        f"! {job['functional']} {job['final_sp_basis']} def2/J RIJCOSX "
        "TightSCF DEFGRID3\n"
        + smd
        + f"* xyzfile {job['formal_charge']} {job['multiplicity']} "
        f"{job['job_id']}_Compound_1.xyz\n"
        "StepEnd\n"
    )
    return (
        header
        + f"%pal nprocs {job['nprocs']} end\n"
        + f"%maxcore {job['maxcore_mb_per_rank']}\n"
        + f"* xyz {job['formal_charge']} {job['multiplicity']}\n"
        + _geometry_lines(xyz_path)
        + "\n*\n"
        + "%compound\n"
        + first_step
        + second_step
        + "End\n"
    )


def _deck_path(run_dir: Path, job_class: str, job_id: str) -> Path:
    category = "sp" if job_class in {"diagnostic_gas_sp", "smd_energy_sp"} else "optfreq"
    return run_dir / category / job_id / f"{job_id}.inp"


def _geometry_path(row: Mapping[str, str], repository_root: Path) -> Path:
    path = Path(row["xyz_path"])
    return path if path.is_absolute() else repository_root / path


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_decks(
    *,
    spec_dir: Path,
    geometry_index_path: Path,
    run_dir: Path,
    manifest_path: Path,
    selected_job_ids: set[str] | None = None,
) -> DeckBuildSummary:
    spec_dir = spec_dir.resolve()
    geometry_index_path = geometry_index_path.resolve()
    run_dir = run_dir.resolve()
    manifest_path = manifest_path.resolve()
    validation = validate_spec(spec_dir)
    if not validation.ok:
        raise DeckGenerationError(f"spec validation failed: {validation.to_json()}")
    repository_root = spec_dir.parent
    geometry_rows = read_csv_rows(geometry_index_path)
    geometry_by_key = {row["geometry_key"]: row for row in geometry_rows}
    solvent_by_id = {
        row["solvent_id"]: row
        for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
    }
    registry_sha256 = sha256_file(spec_dir / "solvent_smd_registry.csv")
    registry_row_sha256 = {
        solvent_id: csv_record_sha256(
            spec_dir / "solvent_smd_registry.csv",
            key_field="solvent_id",
            key_value=solvent_id,
        )
        for solvent_id in solvent_by_id
    }
    logical_jobs: list[tuple[str, dict[str, str], str]] = []
    for row in read_csv_rows(spec_dir / "sp_job_manifest.csv"):
        logical_jobs.append((row["job_class"], row, row["geometry_key"]))
    for row in read_csv_rows(spec_dir / "optfreq_job_manifest.csv"):
        logical_jobs.append(("optfreq", row, row["start_geometry_key"]))
    if selected_job_ids is not None:
        known_job_ids = {item[1]["job_id"] for item in logical_jobs}
        unknown_job_ids = selected_job_ids - known_job_ids
        if unknown_job_ids:
            raise DeckGenerationError(
                f"unknown selected job IDs: {sorted(unknown_job_ids)}"
            )
        logical_jobs = [
            item for item in logical_jobs if item[1]["job_id"] in selected_job_ids
        ]

    manifest: list[dict[str, object]] = []
    for job_class, job, geometry_key in logical_jobs:
        geometry = geometry_by_key.get(geometry_key)
        reason = ""
        status = "ready"
        input_path = _deck_path(run_dir, job_class, job["job_id"])
        input_sha = ""
        geometry_sha = ""
        smd_row_sha = ""
        smd_sha = ""
        reuse_key = ""
        if geometry is None or geometry["status"] != "resolved":
            status = "waiting_geometry"
            reason = (
                "geometry index row is missing"
                if geometry is None
                else geometry["reason"]
            )
        else:
            xyz_path = _geometry_path(geometry, repository_root)
            if not xyz_path.is_file():
                raise DeckGenerationError(
                    f"{job['job_id']}: resolved geometry file is absent: {xyz_path}"
                )
            geometry_sha = sha256_file(xyz_path)
            if geometry_sha != geometry["xyz_sha256"]:
                raise DeckGenerationError(
                    f"{job['job_id']}: geometry hash differs from resolved index"
                )
            solvent = None if job_class == "diagnostic_gas_sp" else solvent_by_id[job["solvent_id"]]
            if solvent is not None:
                smd_row_sha = registry_row_sha256[job["solvent_id"]]
            if job_class == "optfreq":
                if solvent is None:
                    raise DeckGenerationError(
                        f"{job['job_id']}: Opt/Freq job has no SMD registry row"
                    )
                text = render_optfreq_deck(
                    job,
                    geometry,
                    xyz_path,
                    solvent,
                    registry_sha256=registry_sha256,
                    registry_row_sha256=smd_row_sha,
                )
            else:
                text = render_sp_deck(
                    job,
                    geometry,
                    xyz_path,
                    solvent,
                    registry_sha256=registry_sha256,
                    registry_row_sha256=smd_row_sha,
                )
            if solvent is not None:
                smd_sha = smd_payload_sha256(solvent)
            reuse_key = build_exact_reuse_key(
                job,
                geometry,
                job_class=job_class,
                smd_registry_row_sha256=smd_row_sha,
                smd_payload_sha256=smd_sha,
            )
            output_path = input_path.with_suffix(".out")
            if output_path.exists():
                if input_path.is_file() and input_path.read_text(encoding="utf-8") == text:
                    status = "existing_output"
                else:
                    raise DeckGenerationError(
                        f"{job['job_id']}: existing output prevents deck replacement"
                    )
            else:
                _write_text_atomic(input_path, text)
            input_sha = sha256_file(input_path)

        manifest.append(
            {
                "job_id": job["job_id"],
                "job_class": job_class,
                "state_id": job["state_id"],
                "solvent_id": job["solvent_id"],
                "input_path": (
                    str(input_path.relative_to(repository_root))
                    if input_path.is_relative_to(repository_root)
                    else str(input_path)
                ),
                "input_sha256": input_sha,
                "geometry_key": geometry_key,
                "geometry_sha256": geometry_sha,
                "smd_registry_row_sha256": smd_row_sha,
                "smd_payload_sha256": smd_sha,
                "exact_reuse_key": reuse_key,
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
                "status": status,
                "reason": reason,
            }
        )
    write_csv_deterministic(
        manifest_path,
        DECK_MANIFEST_FIELDS,
        manifest,
        sort_by=("job_id",),
    )
    counts = Counter(str(row["status"]) for row in manifest)
    return DeckBuildSummary(
        total=len(manifest),
        ready=counts["ready"],
        waiting_geometry=counts["waiting_geometry"],
        existing_outputs=counts["existing_output"],
    )
