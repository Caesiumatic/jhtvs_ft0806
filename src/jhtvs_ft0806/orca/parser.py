"""Parse ORCA 6.1 SP and Opt/Freq outputs with fail-closed QC."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Mapping

from jhtvs_ft0806.geometry.xyz import (
    check_connectivity,
    inferred_bonds,
    read_xyz,
)
from jhtvs_ft0806.orca.smd import (
    SELF_SEEDED_CUSTOM_SPECIAL_CASES,
    render_smd_block,
    smd_payload_sha256,
)
from jhtvs_ft0806.orca.decks import build_exact_reuse_key
from jhtvs_ft0806.provenance import csv_record_sha256, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic
from jhtvs_ft0806.spec_validation import validate_spec


HARTREE_TO_EV = 27.211386245988
KCAL_PER_HARTREE = 627.5094740631
SIGNIFICANT_IMAGINARY_CM1 = -50.0
THERMOCHEMISTRY_TEMPERATURE_K = 298.15
THERMOCHEMISTRY_PRESSURE_ATM = 1.0
GAS_CONSTANT_J_MOL_K = 8.31446261815324
HARTREE_J_MOL = 2625.4996394799 * 1000.0
STANDARD_STATE_CONCENTRATION_M = 1.0
STANDARD_STATE_1M_CORRECTION_EH = (
    GAS_CONSTANT_J_MOL_K
    * THERMOCHEMISTRY_TEMPERATURE_K
    * math.log(
        GAS_CONSTANT_J_MOL_K
        * THERMOCHEMISTRY_TEMPERATURE_K
        * STANDARD_STATE_CONCENTRATION_M
        / 101.325
    )
    / HARTREE_J_MOL
)

ECHO_TOLERANCE_POLICY = "max(5e-4,5e-5*abs(expected))"
ECHO_FIELDS = {
    "epsilon": "epsilon",
    "refrac": "refrac_cpcm",
    "soln": "soln_293K",
    "soln25": "soln25_298K",
    "sola": "sola",
    "solb": "solb",
    "solg": "solg",
    "solc": "solc",
    "solh": "solh",
}
RESULT_FIELDS = (
    "job_id",
    "job_class",
    "state_id",
    "solvent_id",
    "workflow_revision",
    "method_id",
    "geometry_key",
    "geometry_sha256",
    "smd_registry_row_sha256",
    "smd_payload_sha256",
    "exact_reuse_key",
    "input_path",
    "input_sha256",
    "output_path",
    "output_sha256",
    "orca_version",
    "normal_termination",
    "orca_error",
    "final_energy_Eh",
    "cpcm_dielectric_Eh",
    "smd_cds_Eh",
    "E_freq_Eh",
    "G_minus_E_freq_Eh",
    "E_final_SP_Eh",
    "G_composite_1atm_Eh",
    "standard_state_1M_correction_Eh",
    "G_composite_1M_Eh",
    "thermochemistry_temperature_K",
    "thermochemistry_pressure_atm",
    "quasi_rrho",
    "frequency_count",
    "significant_imaginary_count",
    "most_negative_frequency_cm1",
    "optimization_converged",
    "optimized_geometry_path",
    "optimized_geometry_sha256",
    "connectivity_status",
    "bonds_broken",
    "bonds_formed",
    "echo_solvent_name",
    "echo_epsilon",
    "echo_refrac",
    "echo_soln",
    "echo_soln25",
    "echo_sola",
    "echo_solb",
    "echo_solg",
    "echo_solc",
    "echo_solh",
    "echo_max_abs_delta",
    "echo_qc",
    "qc_status",
    "qc_reasons",
    "scientific_stop_required",
)

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_SPE_RE = re.compile(rf"FINAL SINGLE POINT ENERGY\s+({_NUMBER})")
_COMPOUND_JOB_RE = re.compile(r"COMPOUND JOB\s+(\d+)")
_E_FREQ_RE = re.compile(rf"Electronic energy\s+\.\.\.\s+({_NUMBER})\s+Eh")
_G_MINUS_E_RE = re.compile(rf"G-E\(el\)\s+\.\.\.\s+({_NUMBER})\s+Eh")
_CPCM_RE = re.compile(rf"CPCM Dielectric\s+:\s+({_NUMBER})\s+Eh")
_SMD_CDS_RE = re.compile(rf"SMD CDS \(Gcds\)\s+:\s+({_NUMBER})\s+Eh")
_ORCA_VERSION_RE = re.compile(r"Program Version\s+([^\s]+)", re.IGNORECASE)
_SMD_SOLVENT_NAME_RE = re.compile(
    r"^\s*Solvent:\s+\.\.\.\s+(\S+)", re.IGNORECASE | re.MULTILINE
)
_THERMO_TEMP_RE = re.compile(r"THERMOCHEMISTRY AT\s+([0-9.]+)K", re.IGNORECASE)
_THERMO_PRESSURE_RE = re.compile(
    r"Pressure\s+\.\.\.\s+([0-9.]+)\s+atm", re.IGNORECASE
)
_QUASI_RRHO_RE = re.compile(
    r"Quasi\s+RRHO\s+\.\.\.\s+True", re.IGNORECASE
)
_FREQUENCY_RE = re.compile(
    rf"^\s*\d+:\s*({_NUMBER})\s+cm\*\*-1", re.MULTILINE
)
_ECHO_RES = {
    field: re.compile(
        rf"^\s*{label}\s+\.\.\.\s+({_NUMBER})(?:\s|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    for field, label in {
        "epsilon": "Epsilon",
        "refrac": "Refrac",
        "soln": "Soln",
        "soln25": "Soln25",
        "sola": "Sola",
        "solb": "Solb",
        "solg": "Solg",
        "solc": "Solc",
        "solh": "Solh",
    }.items()
}


class OrcaParseError(ValueError):
    """Raised when parser inputs or provenance tables are inconsistent."""


@dataclass(frozen=True, slots=True)
class ParseSummary:
    total: int
    clean: int
    flagged: int
    missing: int
    scientific_stops: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": "PASS" if self.scientific_stops == 0 else "SCIENTIFIC_STOP",
            "total": self.total,
            "clean": self.clean,
            "flagged": self.flagged,
            "missing": self.missing,
            "scientific_stops": self.scientific_stops,
        }


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def _expected_final_step_energy(text: str) -> float | None:
    starts = [
        match.start()
        for match in _COMPOUND_JOB_RE.finditer(text)
        if match.group(1) == "2"
    ]
    if not starts:
        return None
    values = [float(value) for value in _SPE_RE.findall(text[starts[-1] :])]
    return values[-1] if values else None


def _frequency_values(text: str) -> list[float]:
    start = text.rfind("VIBRATIONAL FREQUENCIES")
    if start < 0:
        return []
    section = text[start:]
    normal_modes = section.find("NORMAL MODES")
    if normal_modes >= 0:
        section = section[:normal_modes]
    return [float(value) for value in _FREQUENCY_RE.findall(section)]


def _echo_tolerance(expected: float) -> float:
    return max(5e-4, 5e-5 * abs(expected))


def _audit_echo(
    text: str, solvent: Mapping[str, str]
) -> tuple[dict[str, str], str, str, bool]:
    if solvent["echo_absolute_tolerance"] != ECHO_TOLERANCE_POLICY:
        raise OrcaParseError(
            f"{solvent['solvent_id']}: unsupported echo tolerance policy"
        )
    rendered: dict[str, str] = {}
    deltas: list[float] = []
    reasons: list[str] = []
    required = {
        field.strip()
        for field in solvent["orca_echo_fields_to_capture"].split(";")
        if field.strip() in ECHO_FIELDS
    }
    solvent_names = _SMD_SOLVENT_NAME_RE.findall(text)
    rendered["solvent_name"] = solvent_names[-1] if solvent_names else ""
    if solvent.get("special_case", "") in SELF_SEEDED_CUSTOM_SPECIAL_CASES:
        if not solvent_names:
            reasons.append("echo_solvent_name_missing")
        elif any(name.upper() != "CUSTOM" for name in solvent_names):
            reasons.append("echo_solvent_name_mismatch")
    for echo_field, registry_field in ECHO_FIELDS.items():
        observations = [
            float(value) for value in _ECHO_RES[echo_field].findall(text)
        ]
        rendered[echo_field] = str(observations[-1]) if observations else ""
        expected = float(solvent[registry_field])
        if echo_field not in required:
            continue
        if not observations:
            reasons.append(f"echo_{echo_field}_missing")
            continue
        field_deltas = [abs(value - expected) for value in observations]
        deltas.extend(field_deltas)
        if any(delta > _echo_tolerance(expected) for delta in field_deltas):
            reasons.append(f"echo_{echo_field}_mismatch")
    return (
        rendered,
        format(max(deltas), ".10g") if deltas else "",
        "pass" if not reasons else "mismatch",
        bool(reasons),
    )


def _path(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repository_root / path


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _fmt(value: float | None) -> str:
    return "" if value is None else format(value, ".15g")


def parse_job_result(
    *,
    manifest: Mapping[str, str],
    job: Mapping[str, str],
    geometry: Mapping[str, str] | None,
    solvent: Mapping[str, str] | None,
    repository_root: Path,
    registry_row_sha256: str = "",
) -> dict[str, object]:
    job_id = job["job_id"]
    job_class = manifest["job_class"]
    input_path = _path(manifest["input_path"], repository_root)
    output_path = input_path.with_suffix(".out")
    result: dict[str, object] = {field: "" for field in RESULT_FIELDS}
    result.update(
        {
            "job_id": job_id,
            "job_class": job_class,
            "state_id": job["state_id"],
            "solvent_id": job["solvent_id"],
            "workflow_revision": job["workflow_revision"],
            "method_id": job["method_id"],
            "geometry_key": manifest["geometry_key"],
            "geometry_sha256": manifest["geometry_sha256"],
            "smd_registry_row_sha256": manifest.get(
                "smd_registry_row_sha256", ""
            ),
            "smd_payload_sha256": manifest.get("smd_payload_sha256", ""),
            "exact_reuse_key": manifest.get("exact_reuse_key", ""),
            "input_path": manifest["input_path"],
            "input_sha256": manifest["input_sha256"],
            "output_path": _relative_or_absolute(output_path, repository_root),
            "normal_termination": "false",
            "orca_error": "false",
            "quasi_rrho": "not_applicable" if job_class != "optfreq" else "false",
            "optimization_converged": (
                "not_applicable" if job_class != "optfreq" else "false"
            ),
            "connectivity_status": "not_applicable",
            "echo_qc": "not_applicable" if solvent is None else "missing",
            "qc_status": "missing",
            "scientific_stop_required": "false",
        }
    )
    reasons: list[str] = []
    if manifest["status"] not in {"ready", "existing_output"}:
        reasons.append(f"deck_{manifest['status']}")
        result["qc_reasons"] = ";".join(reasons)
        return result
    if not input_path.is_file() or sha256_file(input_path) != manifest["input_sha256"]:
        reasons.append("input_missing_or_hash_mismatch")
        result["qc_reasons"] = ";".join(reasons)
        return result
    input_text = input_path.read_text(encoding="utf-8")
    input_identity_mismatch = False
    strict_input_identity = solvent is not None and (
        bool(registry_row_sha256)
        or solvent.get("special_case", "") in SELF_SEEDED_CUSTOM_SPECIAL_CASES
    )
    if solvent is not None and strict_input_identity:
        expected_payload_sha = smd_payload_sha256(solvent)
        if manifest.get("smd_payload_sha256", "") != expected_payload_sha:
            reasons.append("smd_input_payload_sha_mismatch")
            input_identity_mismatch = True
        if render_smd_block(solvent) not in input_text:
            reasons.append("smd_input_payload_mismatch")
            input_identity_mismatch = True
        if registry_row_sha256:
            if manifest.get("smd_registry_row_sha256", "") != registry_row_sha256:
                reasons.append("smd_registry_row_sha_mismatch")
                input_identity_mismatch = True
            if geometry is None:
                reasons.append("exact_reuse_geometry_missing")
                input_identity_mismatch = True
            else:
                expected_reuse_key = build_exact_reuse_key(
                    job,
                    geometry,
                    job_class=job_class,
                    smd_registry_row_sha256=registry_row_sha256,
                    smd_payload_sha256=expected_payload_sha,
                )
                if manifest.get("exact_reuse_key", "") != expected_reuse_key:
                    reasons.append("exact_reuse_key_mismatch")
                    input_identity_mismatch = True
        if input_identity_mismatch:
            result["scientific_stop_required"] = "true"
    if not output_path.is_file() or output_path.is_symlink():
        reasons.append("output_missing")
        result["qc_reasons"] = ";".join(reasons)
        return result

    text = output_path.read_text(encoding="utf-8", errors="replace")
    result["output_sha256"] = sha256_file(output_path)
    version_match = _ORCA_VERSION_RE.search(text)
    result["orca_version"] = version_match.group(1) if version_match else ""
    normal = "ORCA TERMINATED NORMALLY" in text
    orca_error = "ERROR !!!" in text
    result["normal_termination"] = str(normal).lower()
    result["orca_error"] = str(orca_error).lower()
    if not normal:
        reasons.append("normal_termination_missing")
    if orca_error:
        reasons.append("orca_error")
    output_identity_ok = all(
        marker in text
        for marker in (
            f"# job_id: {job_id}\n",
            f"# input_sha256: {manifest['input_sha256']}\n",
            f"# workflow_revision: {job['workflow_revision']}\n",
            f"# method_id: {job['method_id']}\n",
        )
    )
    if not output_identity_ok:
        reasons.append("output_identity_mismatch")

    final_energy = _last_float(_SPE_RE, text)
    cpcm = _last_float(_CPCM_RE, text)
    cds = _last_float(_SMD_CDS_RE, text)
    result["final_energy_Eh"] = _fmt(final_energy)
    result["cpcm_dielectric_Eh"] = _fmt(cpcm)
    result["smd_cds_Eh"] = _fmt(cds)

    echo_mismatch = False
    if solvent is not None:
        echo, max_delta, echo_qc, echo_mismatch = _audit_echo(text, solvent)
        for field, value in echo.items():
            result[f"echo_{field}"] = value
        result["echo_max_abs_delta"] = max_delta
        result["echo_qc"] = echo_qc
        if echo_mismatch:
            reasons.append("smd_echo_mismatch")
            result["scientific_stop_required"] = "true"
        if cpcm is None:
            reasons.append("cpcm_dielectric_missing")
        if cds is None:
            reasons.append("smd_cds_missing")
    elif cpcm is not None or cds is not None or "SMDsolvent" in text:
        reasons.append("gas_job_contains_smd_evidence")

    composite_complete = final_energy is not None
    if job_class == "optfreq":
        e_freq = _last_float(_E_FREQ_RE, text)
        correction = _last_float(_G_MINUS_E_RE, text)
        final_sp = _expected_final_step_energy(text)
        temperature = _last_float(_THERMO_TEMP_RE, text)
        pressure = _last_float(_THERMO_PRESSURE_RE, text)
        quasi_rrho = _QUASI_RRHO_RE.search(text) is not None
        frequencies = _frequency_values(text)
        significant = [value for value in frequencies if value < SIGNIFICANT_IMAGINARY_CM1]
        result.update(
            {
                "E_freq_Eh": _fmt(e_freq),
                "G_minus_E_freq_Eh": _fmt(correction),
                "E_final_SP_Eh": _fmt(final_sp),
                "thermochemistry_temperature_K": _fmt(temperature),
                "thermochemistry_pressure_atm": _fmt(pressure),
                "quasi_rrho": str(quasi_rrho).lower(),
                "frequency_count": len(frequencies),
                "significant_imaginary_count": len(significant),
                "most_negative_frequency_cm1": (
                    _fmt(min(frequencies)) if frequencies else ""
                ),
                "optimization_converged": str(
                    "THE OPTIMIZATION HAS CONVERGED" in text
                ).lower(),
            }
        )
        if final_sp is not None and correction is not None:
            composite_1atm = final_sp + correction
            composite_1m = composite_1atm + STANDARD_STATE_1M_CORRECTION_EH
            result["G_composite_1atm_Eh"] = _fmt(composite_1atm)
            result["standard_state_1M_correction_Eh"] = _fmt(
                STANDARD_STATE_1M_CORRECTION_EH
            )
            result["G_composite_1M_Eh"] = _fmt(composite_1m)
        else:
            composite_complete = False
            reasons.append("composite_gibbs_incomplete")
        if e_freq is None:
            reasons.append("frequency_electronic_energy_missing")
        if temperature is None or not math.isclose(
            temperature, THERMOCHEMISTRY_TEMPERATURE_K, abs_tol=0.01
        ):
            reasons.append("thermochemistry_temperature_mismatch")
        if pressure is None or not math.isclose(
            pressure, THERMOCHEMISTRY_PRESSURE_ATM, abs_tol=0.01
        ):
            reasons.append("thermochemistry_pressure_mismatch")
        if not quasi_rrho:
            reasons.append("quasi_rrho_missing")
        if geometry is None:
            reasons.append("geometry_index_missing")
        else:
            geometry_path = _path(geometry["xyz_path"], repository_root)
            expected_atoms = len(read_xyz(geometry_path))
            if len(frequencies) < 3 * expected_atoms:
                reasons.append("frequency_table_incomplete")
            optimized_path = input_path.parent / f"{job_id}_Compound_1.xyz"
            result["optimized_geometry_path"] = _relative_or_absolute(
                optimized_path, repository_root
            )
            if optimized_path.is_file() and not optimized_path.is_symlink():
                result["optimized_geometry_sha256"] = sha256_file(optimized_path)
                try:
                    connectivity = check_connectivity(
                        geometry_path,
                        optimized_path,
                        reference_bonds=(
                            tuple(bond)
                            for bond in inferred_bonds(read_xyz(geometry_path))
                        ),
                    )
                except ValueError:
                    result["connectivity_status"] = "composition_or_order_changed"
                    reasons.append("optimized_geometry_composition_or_order_changed")
                else:
                    result["connectivity_status"] = (
                        "pass" if connectivity.ok else "changed"
                    )
                    result["bonds_broken"] = connectivity.bonds_broken
                    result["bonds_formed"] = connectivity.bonds_formed
                    if not connectivity.ok:
                        reasons.append("optimized_geometry_connectivity_changed")
            else:
                result["connectivity_status"] = "optimized_geometry_missing"
                reasons.append("optimized_geometry_missing")
        if "THE OPTIMIZATION HAS CONVERGED" not in text:
            reasons.append("optimization_not_converged")
        if significant:
            reasons.append("significant_imaginary_frequency")
    elif final_energy is None:
        composite_complete = False
        reasons.append("final_energy_missing")

    smd_components_complete = solvent is None or (cpcm is not None and cds is not None)
    hard_missing = (
        not normal
        or orca_error
        or not output_identity_ok
        or not composite_complete
        or not smd_components_complete
    )
    result["qc_status"] = "missing" if hard_missing else "flagged" if reasons else "clean"
    result["qc_reasons"] = ";".join(dict.fromkeys(reasons))
    return result


def parse_results(
    *,
    spec_dir: Path,
    geometry_index_path: Path,
    deck_manifest_path: Path,
    output_path: Path,
) -> ParseSummary:
    spec_dir = spec_dir.resolve()
    repository_root = spec_dir.parent
    validation = validate_spec(spec_dir)
    if not validation.ok:
        raise OrcaParseError("scientific specification validation failed")
    manifest_rows = read_csv_rows(deck_manifest_path)
    geometry_by_key = {
        row["geometry_key"]: row for row in read_csv_rows(geometry_index_path)
    }
    solvent_by_id = {
        row["solvent_id"]: row
        for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
    }
    registry_row_sha256 = {
        solvent_id: csv_record_sha256(
            spec_dir / "solvent_smd_registry.csv",
            key_field="solvent_id",
            key_value=solvent_id,
        )
        for solvent_id in solvent_by_id
    }
    jobs: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(spec_dir / "sp_job_manifest.csv"):
        jobs[row["job_id"]] = row
    for row in read_csv_rows(spec_dir / "optfreq_job_manifest.csv"):
        jobs[row["job_id"]] = row

    results: list[dict[str, object]] = []
    for manifest in manifest_rows:
        job = jobs.get(manifest["job_id"])
        if job is None:
            raise OrcaParseError(
                f"deck manifest contains unknown job {manifest['job_id']}"
            )
        solvent = (
            None
            if manifest["job_class"] == "diagnostic_gas_sp"
            else solvent_by_id[job["solvent_id"]]
        )
        results.append(
            parse_job_result(
                manifest=manifest,
                job=job,
                geometry=geometry_by_key.get(manifest["geometry_key"]),
                solvent=solvent,
                repository_root=repository_root,
                registry_row_sha256=registry_row_sha256.get(
                    job["solvent_id"], ""
                ),
            )
        )
    write_csv_deterministic(
        output_path, RESULT_FIELDS, results, sort_by=("job_id",)
    )
    counts = Counter(str(row["qc_status"]) for row in results)
    return ParseSummary(
        total=len(results),
        clean=counts["clean"],
        flagged=counts["flagged"],
        missing=counts["missing"],
        scientific_stops=sum(
            row["scientific_stop_required"] == "true" for row in results
        ),
    )
