#!/usr/bin/env python3
"""Prepare, parse, and report the explicit-R5 cluster-continuum Eox benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from jhtvs_ft0806.geometry.xyz import XYZAtom, inferred_bonds, read_xyz
from jhtvs_ft0806.hpc.submission import TASK_FIELDS
from jhtvs_ft0806.orca.decks import (
    THERMOCHEMISTRY_CONVENTION_ID,
    build_exact_reuse_key,
    render_optfreq_deck,
)
from jhtvs_ft0806.orca.parser import HARTREE_TO_EV, RESULT_FIELDS, parse_job_result
from jhtvs_ft0806.orca.smd import render_smd_block, smd_payload_sha256
from jhtvs_ft0806.provenance import csv_record_sha256, sha256_bytes, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


DIAGNOSTIC_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = DIAGNOSTIC_ROOT.parents[1]
PROTOCOL_PATH = DIAGNOSTIC_ROOT / "protocol.json"
INPUT_ROOT = DIAGNOSTIC_ROOT / "input_files"
SOURCE_ROOT = DIAGNOSTIC_ROOT / "source_geometries"
CLUSTER_ROOT = DIAGNOSTIC_ROOT / "clusters"
PACKMOL_ROOT = DIAGNOSTIC_ROOT / "packmol"
ORCA_ROOT = DIAGNOSTIC_ROOT / "orca"
ORCA_MANIFEST_PATH = ORCA_ROOT / "job_manifest.csv"
ORCA_TASKS_PATH = ORCA_ROOT / "tasks.tsv"
CONTINUATION_ROOT = DIAGNOSTIC_ROOT / "continuations"
CONTINUATION_MANIFEST_PATH = ORCA_ROOT / "continuation_manifest.csv"
CONTINUATION_TASK_ROOT = ORCA_ROOT / "continuation_tasks"
WORKFLOW_REVISION = "jhtvs-ft0806-explicit-r5-eox-v1"
PACKMOL_SUCCESS = "Success!"
HARTREE_TO_EV_DECIMAL = Decimal(str(HARTREE_TO_EV))
ATOMIC_MASS = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "S": 32.06,
    "Cl": 35.45,
}

BENCHMARK_FIELDS = (
    "record_id", "property", "species", "environment", "calculation_key",
    "experimental_value", "calibrated_tier1", "tier2_dft", "unit", "source",
    "source_url", "source_file", "source_location", "protocol_or_conditions",
    "normalization", "dft_species_id", "dft_theory_level", "dft_qc_status",
    "dft_workflow_revision", "dft_source_file", "dft_source_row",
)
REGISTRY_FIELDS = (
    "phase", "calculation_key", "property", "species", "environment",
    "species_id", "medium_id", "source_class", "experimental_record_count",
    "experimental_min_V", "experimental_max_V", "calibrated_tier1_V",
    "implicit_dft_V", "reduced_charge", "reduced_multiplicity",
    "oxidized_charge", "oxidized_multiplicity", "reduced_optfreq_basis",
    "oxidized_optfreq_basis", "final_sp_basis", "target_geometry_source",
    "shell_geometry_source", "status",
)
SOURCE_FIELDS = (
    "name", "mode", "snapshot_path", "snapshot_sha256", "atom_count",
    "source_xyz_path", "source_xyz_sha256", "source_orca_input_path",
    "source_orca_input_sha256", "source_orca_output_path",
    "source_orca_output_sha256", "source_charge", "source_multiplicity",
    "source_qc",
)
CLUSTER_FIELDS = (
    "phase", "calculation_key", "target_source", "shell_source", "n_shell",
    "target_atoms", "shell_atoms", "total_atoms", "molecule_count",
    "box_side_A", "box_min_A", "box_max_A", "tolerance_A", "seed",
    "packmol_input_path", "packmol_input_sha256", "packmol_log_path",
    "packmol_log_sha256", "packmol_executable", "packmol_executable_sha256",
    "packmol_version", "geometry_path", "geometry_sha256",
    "minimum_intermolecular_distance_A", "max_shell_box_violation_A",
    "molecule_count_qc", "atom_order_qc", "overlap_qc", "containment_qc", "status",
)
ORCA_FIELDS = (
    "phase", "calculation_key", "state_role", "job_id", "job_class",
    "state_id", "solvent_id", "formal_charge", "multiplicity", "input_path",
    "input_sha256", "output_path", "geometry_key", "geometry_sha256",
    "coordinate_payload_sha256", "smd_registry_row_sha256",
    "smd_payload_sha256", "exact_reuse_key", "workflow_revision", "method_id",
    "thermochemistry_convention_id", "functional", "optfreq_basis",
    "final_sp_basis", "final_sp_hirshfeld", "nprocs", "maxcore_mb_per_rank",
    "planning_core_h", "status",
)
CONTINUATION_FIELDS = (
    "logical_job_id", "attempt", "attempt_job_id", "calculation_key",
    "state_role", "trigger", "trigger_output_path", "trigger_output_sha256",
    "trigger_geometry_path", "trigger_geometry_sha256", "source_geometry_path",
    "source_geometry_sha256", "root_initial_cluster_sha256",
    "input_path", "input_sha256", "output_path", "geometry_key",
    "coordinate_payload_sha256", "exact_reuse_key", "status",
    "task_path", "task_sha256", "scheduler_job_id", "submitted_at_utc",
)
SPIN_FIELDS = (
    "calculation_key", "job_id", "molecule_index", "fragment_role",
    "hirshfeld_spin", "spin_fraction", "target_spin_fraction",
    "dominant_spin_fragment", "total_hirshfeld_spin",
    "oxidation_identity_status", "qc_status",
)
UNIQUE_FIELDS = (
    "calculation_key", "property", "reduced_job_id", "oxidized_job_id",
    "G_reduced_1M_Eh", "G_oxidized_1M_Eh", "delta_G_ox_eV",
    "explicit_R5_DFT_Eox_vs_AgAgCl_V", "independent_recompute_V",
    "recompute_abs_delta_V", "initial_cluster_sha256",
    "optimized_geometry_sha256_reduced", "optimized_geometry_sha256_oxidized",
    "optimized_geometries_independent", "oxidation_identity_status", "qc_status",
    "qc_reasons",
)
COMPARISON_FIELDS = (
    "record_id", "property", "species", "environment", "calculation_key",
    "experimental_Eox_V", "implicit_calibrated_xTB_Eox_V", "implicit_DFT_Eox_V",
    "explicit_R5_DFT_Eox_V", "implicit_xTB_signed_error_V",
    "implicit_DFT_signed_error_V", "explicit_R5_DFT_signed_error_V",
    "implicit_xTB_absolute_error_V", "implicit_DFT_absolute_error_V",
    "explicit_R5_DFT_absolute_error_V", "explicit_minus_implicit_DFT_V",
    "explicit_improvement_over_implicit_DFT_V",
    "explicit_improvement_over_implicit_xTB_V", "calculation_qc", "source",
    "protocol_or_conditions", "normalization",
)
METRIC_FIELDS = (
    "category", "aggregation", "method", "n_experimental_records",
    "n_unique_calculation_keys", "mae_V", "rmse_V", "mean_signed_error_V",
    "median_absolute_error_V", "r_squared", "unique_keys_improved_vs_implicit_DFT",
    "unique_keys_worsened_vs_implicit_DFT", "mean_improvement_vs_implicit_DFT_V",
    "median_improvement_vs_implicit_DFT_V",
)
PARSED_FIELDS = ("calculation_key", "state_role") + RESULT_FIELDS
DISTANCE_FIELDS = (
    "calculation_key", "job_id", "state_role", "shell_index",
    "initial_target_shell_COM_distance_A", "optimized_target_shell_COM_distance_A",
    "delta_COM_distance_A", "target_internal_bonds_broken",
    "target_internal_bonds_formed", "shell_internal_bonds_broken",
    "shell_internal_bonds_formed", "intermolecular_bonds_formed",
    "intermolecular_bonds_broken", "fragment_qc_status",
)


class DiagnosticError(RuntimeError):
    """Raised when a frozen protocol or QC invariant is violated."""


def _protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != WORKFLOW_REVISION:
        raise DiagnosticError("protocol schema/revision mismatch")
    return payload


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _diag_relative(path: Path) -> str:
    return path.resolve().relative_to(DIAGNOSTIC_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_generated(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise DiagnosticError(f"immutable artifact differs: {path}")
        return
    path.write_bytes(payload)


def _read_csv_utf8_sig(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DiagnosticError(f"CSV has no header: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise DiagnosticError(f"CSV row has extra fields: {path}")
    return rows


def _system_map() -> dict[str, dict[str, Any]]:
    systems = _protocol()["systems"]
    result = {row["calculation_key"]: row for row in systems}
    if len(result) != len(systems):
        raise DiagnosticError("duplicate calculation key in protocol")
    return result


def deterministic_seed(calculation_key: str) -> int:
    seed = int(hashlib.sha256(calculation_key.encode("utf-8")).hexdigest()[:8], 16)
    seed &= 0x7FFFFFFF
    return seed or 1


def molecular_volume_box_side(
    target_volume_A3: float, n_shell: int, shell_volume_A3: float, shell_span_A: float
) -> float:
    raw = (target_volume_A3 + n_shell * shell_volume_A3) ** (1.0 / 3.0)
    raw += shell_span_A
    return math.ceil((raw - 1e-12) * 10.0) / 10.0


def snapshot_inputs(benchmark_dir: Path, detail_dir: Path) -> list[dict[str, str]]:
    protocol = _protocol()
    locations = {
        "metrics_summary.csv": benchmark_dir / "metrics_summary.csv",
        "anion_eox_dft.csv": benchmark_dir / "anion_eox_dft.csv",
        "solvent_eox_dft.csv": benchmark_dir / "solvent_eox_dft.csv",
        "anion_eox_calibrated_tier1.csv": benchmark_dir / "anion_eox_calibrated_tier1.csv",
        "solvent_eox_calibrated_tier1.csv": benchmark_dir / "solvent_eox_calibrated_tier1.csv",
        "anion_dft.csv": detail_dir / "anion_dft.csv",
        "solvent_dft.csv": detail_dir / "solvent_dft.csv",
    }
    rows: list[dict[str, str]] = []
    for name, expected in protocol["input_files"].items():
        source = locations[name]
        if not source.is_file() or source.is_symlink():
            raise DiagnosticError(f"missing or unsafe input file: {source}")
        observed = sha256_file(source)
        if observed != expected:
            raise DiagnosticError(f"input SHA mismatch for {name}: {observed}")
        snapshot = INPUT_ROOT / name
        _write_immutable_bytes(snapshot, source.read_bytes())
        rows.append(
            {
                "name": name,
                "sha256": expected,
                "snapshot_path": _repo_relative(snapshot),
                "byte_count": str(snapshot.stat().st_size),
                "source_basename": source.name,
            }
        )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "input_file_manifest.csv",
        ("name", "sha256", "snapshot_path", "byte_count", "source_basename"),
        rows,
        sort_by=("name",),
    )
    return rows


def _last_orca_cartesian_geometry(text: str) -> tuple[XYZAtom, ...]:
    marker = "CARTESIAN COORDINATES (ANGSTROEM)"
    positions = [match.start() for match in re.finditer(re.escape(marker), text)]
    for position in reversed(positions):
        atoms: list[XYZAtom] = []
        for line in text[position + len(marker) :].splitlines():
            fields = line.split()
            if len(fields) >= 4 and re.fullmatch(r"[A-Z][a-z]?", fields[0]):
                try:
                    atoms.append(
                        XYZAtom(fields[0], float(fields[1]), float(fields[2]), float(fields[3]))
                    )
                except ValueError:
                    continue
            elif atoms:
                break
        if atoms:
            return tuple(atoms)
    raise DiagnosticError("ORCA output has no Cartesian Angstrom geometry")


def _xyz_bytes(atoms: Sequence[XYZAtom], comment: str) -> bytes:
    lines = [str(len(atoms)), comment]
    lines.extend(
        f"{atom.symbol:<2s} {atom.x:18.10f} {atom.y:18.10f} {atom.z:18.10f}"
        for atom in atoms
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def audit_and_snapshot_sources(source_checkout: Path) -> list[dict[str, str]]:
    protocol = _protocol()
    rows: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    for name, spec in sorted(protocol["source_geometries"].items()):
        checked: dict[str, Any] = {"name": name, "mode": spec["mode"]}
        for kind in ("input", "output"):
            path = source_checkout / spec[f"{kind}_path"]
            expected = spec[f"{kind}_sha256"]
            if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
                raise DiagnosticError(f"source {kind} missing or hash drifted: {path}")
            checked[f"{kind}_path"] = str(path)
            checked[f"{kind}_sha256"] = expected
        output_path = source_checkout / spec["output_path"]
        output_text = output_path.read_text(encoding="utf-8", errors="replace")
        if (
            "THE OPTIMIZATION HAS CONVERGED" not in output_text
            or "ORCA TERMINATED NORMALLY" not in output_text
        ):
            raise DiagnosticError(f"source ORCA QC is not clean: {output_path}")
        if spec["mode"] == "copy_xyz":
            xyz_path = source_checkout / spec["xyz_path"]
            if (
                not xyz_path.is_file()
                or xyz_path.is_symlink()
                or sha256_file(xyz_path) != spec["xyz_sha256"]
            ):
                raise DiagnosticError(f"source XYZ missing or hash drifted: {xyz_path}")
            payload = xyz_path.read_bytes()
            source_xyz_path = spec["xyz_path"]
            source_xyz_sha = spec["xyz_sha256"]
        elif spec["mode"] == "extract_last_orca_geometry":
            atoms = _last_orca_cartesian_geometry(output_text)
            payload = _xyz_bytes(atoms, f"extracted from {spec['output_path']}")
            source_xyz_path = spec["output_path"] + "#last_cartesian_angstrom"
            source_xyz_sha = sha256_bytes(payload)
        else:
            raise DiagnosticError(f"unsupported source geometry mode: {spec['mode']}")
        snapshot = SOURCE_ROOT / f"{name}.xyz"
        _write_immutable_bytes(snapshot, payload)
        atoms = read_xyz(snapshot)
        row = {
            "name": name,
            "mode": spec["mode"],
            "snapshot_path": _repo_relative(snapshot),
            "snapshot_sha256": sha256_file(snapshot),
            "atom_count": str(len(atoms)),
            "source_xyz_path": source_xyz_path,
            "source_xyz_sha256": source_xyz_sha,
            "source_orca_input_path": spec["input_path"],
            "source_orca_input_sha256": spec["input_sha256"],
            "source_orca_output_path": spec["output_path"],
            "source_orca_output_sha256": spec["output_sha256"],
            "source_charge": str(spec["charge"]),
            "source_multiplicity": str(spec["multiplicity"]),
            "source_qc": "clean",
        }
        rows.append(row)
        checked.update(
            {
                "snapshot_path": row["snapshot_path"],
                "snapshot_sha256": row["snapshot_sha256"],
                "atom_count": len(atoms),
                "optimization_converged": True,
                "normal_termination": True,
                "status": "clean",
            }
        )
        audit.append(checked)
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "source_provenance.csv",
        SOURCE_FIELDS,
        rows,
        sort_by=("name",),
    )
    _write_json(
        DIAGNOSTIC_ROOT / "source_audit.json",
        {
            "status": "PASS",
            "workflow_revision": WORKFLOW_REVISION,
            "source_checkout": str(source_checkout.resolve()),
            "read_only_policy": True,
            "sources": audit,
        },
    )
    return rows


def _strict_merge_by_record_id(
    dft_rows: Sequence[dict[str, str]], tier1_rows: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    def keyed(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
        result = {row["record_id"]: row for row in rows}
        if len(result) != len(rows):
            raise DiagnosticError("duplicate record_id in benchmark input")
        return result

    dft, tier1 = keyed(dft_rows), keyed(tier1_rows)
    if set(dft) != set(tier1):
        raise DiagnosticError("DFT and calibrated Tier-1 record_id sets differ")
    merged: list[dict[str, str]] = []
    invariant_fields = (
        "record_id", "property", "species", "environment", "experimental_value",
        "unit", "calibrated_tier1", "tier2_dft", "source", "source_url",
        "source_file", "source_location", "protocol_or_conditions", "normalization",
    )
    for record_id in sorted(dft):
        if any(dft[record_id][field] != tier1[record_id][field] for field in invariant_fields):
            raise DiagnosticError(f"record_id merge identity mismatch: {record_id}")
        merged.append(dict(dft[record_id]))
    return merged


def _detail_row_for(system: Mapping[str, Any], rows: Sequence[dict[str, str]]) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["Species_ID"] == system["species_id"]
        and row["Workflow_Revision"] == system["dft_workflow_revision"]
    ]
    if len(matches) != 1:
        raise DiagnosticError(f"detailed DFT row is not unique: {system['calculation_key']}")
    row = matches[0]
    if row["QC_Status"] != "clean":
        raise DiagnosticError(f"implicit DFT source is not clean: {system['calculation_key']}")
    return row


def build_benchmark_records() -> list[dict[str, str]]:
    systems = list(_system_map().values())
    merged = _strict_merge_by_record_id(
        _read_csv_utf8_sig(INPUT_ROOT / "anion_eox_dft.csv"),
        _read_csv_utf8_sig(INPUT_ROOT / "anion_eox_calibrated_tier1.csv"),
    ) + _strict_merge_by_record_id(
        _read_csv_utf8_sig(INPUT_ROOT / "solvent_eox_dft.csv"),
        _read_csv_utf8_sig(INPUT_ROOT / "solvent_eox_calibrated_tier1.csv"),
    )
    detail_rows = _read_csv_utf8_sig(INPUT_ROOT / "anion_dft.csv") + _read_csv_utf8_sig(
        INPUT_ROOT / "solvent_dft.csv"
    )
    output: list[dict[str, str]] = []
    registry: list[dict[str, str]] = []
    for system in systems:
        selected = [
            row for row in merged
            if row["property"] == system["benchmark_property"]
            and row["species"] == system["species"]
            and row["environment"] == system["environment"]
        ]
        if not selected:
            raise DiagnosticError(f"no eligible benchmark rows: {system['calculation_key']}")
        detail = _detail_row_for(system, detail_rows)
        implicit_values = {Decimal(row["tier2_dft"]) for row in selected}
        tier1_values = {Decimal(row["calibrated_tier1"]) for row in selected}
        if len(implicit_values) != 1 or len(tier1_values) != 1:
            raise DiagnosticError(f"calculation-key values are inconsistent: {system['calculation_key']}")
        if abs(Decimal(detail["Value"]) - next(iter(implicit_values))) > Decimal("0.0000005"):
            raise DiagnosticError(f"detailed DFT value mismatch: {system['calculation_key']}")
        for row in selected:
            output.append(
                {
                    "record_id": row["record_id"],
                    "property": row["property"],
                    "species": row["species"],
                    "environment": row["environment"],
                    "calculation_key": system["calculation_key"],
                    "experimental_value": row["experimental_value"],
                    "calibrated_tier1": row["calibrated_tier1"],
                    "tier2_dft": row["tier2_dft"],
                    "unit": row["unit"],
                    "source": row["source"],
                    "source_url": row["source_url"],
                    "source_file": row["source_file"],
                    "source_location": row["source_location"],
                    "protocol_or_conditions": row["protocol_or_conditions"],
                    "normalization": row["normalization"],
                    "dft_species_id": detail["Species_ID"],
                    "dft_theory_level": detail["Theory_Level"],
                    "dft_qc_status": detail["QC_Status"],
                    "dft_workflow_revision": detail["Workflow_Revision"],
                    "dft_source_file": detail["Source_File"],
                    "dft_source_row": detail["Source_Row"],
                }
            )
        experiments = [Decimal(row["experimental_value"]) for row in selected]
        registry.append(
            {
                "phase": system["phase"],
                "calculation_key": system["calculation_key"],
                "property": system["property"],
                "species": system["species"],
                "environment": system["environment"],
                "species_id": system["species_id"],
                "medium_id": system["medium_id"],
                "source_class": system["source_class"],
                "experimental_record_count": str(len(selected)),
                "experimental_min_V": str(min(experiments)),
                "experimental_max_V": str(max(experiments)),
                "calibrated_tier1_V": str(next(iter(tier1_values))),
                "implicit_dft_V": str(next(iter(implicit_values))),
                "reduced_charge": str(system["states"][0][0]),
                "reduced_multiplicity": str(system["states"][0][1]),
                "oxidized_charge": str(system["states"][1][0]),
                "oxidized_multiplicity": str(system["states"][1][1]),
                "reduced_optfreq_basis": system["optfreq_basis_reduced"],
                "oxidized_optfreq_basis": system["optfreq_basis_oxidized"],
                "final_sp_basis": _protocol()["orca"]["final_sp_basis"],
                "target_geometry_source": system["target_geometry_source"],
                "shell_geometry_source": system["shell_geometry_source"],
                "status": "selected",
            }
        )
    if len(output) != 15 or len(registry) != 6:
        raise DiagnosticError(f"Phase A cardinality mismatch: {len(output)} records, {len(registry)} keys")
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "benchmark_records.csv", BENCHMARK_FIELDS, output,
        sort_by=("calculation_key", "record_id"),
    )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "calculation_registry.csv", REGISTRY_FIELDS, registry,
        sort_by=("phase", "calculation_key"),
    )
    return output


def _source_rows() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(DIAGNOSTIC_ROOT / "source_provenance.csv")
    result = {row["name"]: row for row in rows}
    if len(result) != len(rows):
        raise DiagnosticError("duplicate source geometry name")
    return result


def _source_path(name: str) -> Path:
    return REPOSITORY_ROOT / _source_rows()[name]["snapshot_path"]


def _continuation_rows() -> list[dict[str, str]]:
    if not CONTINUATION_MANIFEST_PATH.is_file():
        return []
    rows = read_csv_rows(CONTINUATION_MANIFEST_PATH)
    if len({(row["logical_job_id"], row["attempt"]) for row in rows}) != len(rows):
        raise DiagnosticError("duplicate continuation attempt")
    if any(row["attempt"] != "1" for row in rows):
        raise DiagnosticError("only one continuation is permitted per failed state")
    return rows


def _effective_manifests() -> list[dict[str, str]]:
    """Resolve each logical state to its original or sole continuation attempt."""

    manifests = read_csv_rows(ORCA_MANIFEST_PATH)
    by_job = {row["job_id"]: dict(row) for row in manifests}
    for attempt in _continuation_rows():
        logical_job_id = attempt["logical_job_id"]
        if logical_job_id not in by_job:
            raise DiagnosticError(f"continuation references unknown job: {logical_job_id}")
        effective = by_job[logical_job_id]
        effective.update(
            {
                "job_id": attempt["attempt_job_id"],
                "input_path": attempt["input_path"],
                "input_sha256": attempt["input_sha256"],
                "output_path": attempt["output_path"],
                "geometry_key": attempt["geometry_key"],
                "geometry_sha256": attempt["source_geometry_sha256"],
                "coordinate_payload_sha256": attempt["coordinate_payload_sha256"],
                "exact_reuse_key": attempt["exact_reuse_key"],
                "input_geometry_path": attempt["source_geometry_path"],
                "logical_job_id": logical_job_id,
                "attempt": attempt["attempt"],
            }
        )
    return list(by_job.values())


def _molecule_ranges(
    target_atoms: int, shell_atoms: int, n_shell: int
) -> tuple[range, ...]:
    ranges: list[range] = [range(0, target_atoms)]
    offset = target_atoms
    for _ in range(n_shell):
        ranges.append(range(offset, offset + shell_atoms))
        offset += shell_atoms
    return tuple(ranges)


def _minimum_intermolecular_distance(
    atoms: Sequence[XYZAtom], ranges: Sequence[range]
) -> float:
    minimum = math.inf
    for index, first in enumerate(ranges):
        for second in ranges[index + 1 :]:
            for atom_index in first:
                for other_index in second:
                    minimum = min(
                        minimum,
                        math.dist(
                            atoms[atom_index].coordinates,
                            atoms[other_index].coordinates,
                        ),
                    )
    return minimum


def _molecular_span(atoms: Sequence[XYZAtom]) -> float:
    return max(
        max(getattr(atom, axis) for atom in atoms)
        - min(getattr(atom, axis) for atom in atoms)
        for axis in ("x", "y", "z")
    )


def _packmol_input_path(key: str) -> Path:
    return PACKMOL_ROOT / f"{key}_R5.inp"


def _packmol_log_path(key: str) -> Path:
    return PACKMOL_ROOT / f"{key}_R5.log"


def _cluster_path(key: str) -> Path:
    return CLUSTER_ROOT / f"{key}_R5.xyz"


def render_packmol_inputs() -> list[Path]:
    protocol = _protocol()
    sources = _source_rows()
    volume = protocol["packmol"]["molecular_volumes_A3"]
    tolerance = float(protocol["packmol"]["tolerance_A"])
    paths: list[Path] = []
    for system in protocol["systems"]:
        key = system["calculation_key"]
        target_name = system["target_geometry_source"]
        shell_name = system["shell_geometry_source"]
        shell_atoms = read_xyz(REPOSITORY_ROOT / sources[shell_name]["snapshot_path"])
        expected_box = molecular_volume_box_side(
            float(volume[target_name]),
            int(protocol["packmol"]["n_shell"]),
            float(volume[shell_name]),
            _molecular_span(shell_atoms),
        )
        if not math.isclose(float(system["box_side_A"]), expected_box, abs_tol=1e-9):
            raise DiagnosticError(
                f"molecular-volume box mismatch for {key}: protocol={system['box_side_A']} expected={expected_box}"
            )
        seed = deterministic_seed(key)
        if int(system["seed"]) != seed:
            raise DiagnosticError(f"deterministic seed mismatch for {key}")
        half = expected_box / 2.0
        output = _diag_relative(_cluster_path(key))
        text = (
            f"tolerance {tolerance:.3f}\n"
            "filetype xyz\n"
            f"output {output}\n"
            f"seed {seed}\n\n"
            f"structure source_geometries/{target_name}.xyz\n"
            "  number 1\n"
            "  fixed 0.000 0.000 0.000 0.000 0.000 0.000\n"
            "end structure\n\n"
            f"structure source_geometries/{shell_name}.xyz\n"
            f"  number {protocol['packmol']['n_shell']}\n"
            f"  inside box {-half:.3f} {-half:.3f} {-half:.3f} "
            f"{half:.3f} {half:.3f} {half:.3f}\n"
            "end structure\n"
        )
        path = _packmol_input_path(key)
        _write_immutable_bytes(path, text.encode("utf-8"))
        paths.append(path)
    return paths


def _resolve_executable(command: str) -> Path:
    resolved = shutil.which(command)
    if resolved is None:
        candidate = Path(command).expanduser()
        if not candidate.is_file():
            raise DiagnosticError(f"executable not found: {command}")
        resolved = str(candidate)
    path = Path(resolved).resolve()
    if not path.is_file() or path.is_symlink():
        raise DiagnosticError(f"unsafe executable: {path}")
    return path


def _packmol_version(text: str) -> str:
    for pattern in (
        r"Packmol version\s+([^\s]+)",
        r"PACKMOL\s+-\s+Version\s+([^\s]+)",
        r"Version\s+([^\s]+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return "unreported"


def run_packmol(executable: str) -> list[dict[str, str]]:
    binary = _resolve_executable(executable)
    protocol = _protocol()
    for input_path in render_packmol_inputs():
        key = input_path.stem.removesuffix("_R5")
        cluster = _cluster_path(key)
        log = _packmol_log_path(key)
        if not cluster.exists():
            cluster.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            with input_path.open("rb") as source, log.open("wb") as output:
                completed = subprocess.run(
                    [str(binary)],
                    cwd=DIAGNOSTIC_ROOT,
                    stdin=source,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise DiagnosticError(f"Packmol failed for {key}")
        elif not (cluster.is_file() and log.is_file()):
            raise DiagnosticError(f"partial Packmol artifact set for {key}")
        if PACKMOL_SUCCESS not in log.read_text(encoding="utf-8", errors="replace"):
            raise DiagnosticError(f"Packmol success marker missing for {key}")

    sources = _source_rows()
    rows: list[dict[str, str]] = []
    tolerance = float(protocol["packmol"]["tolerance_A"])
    containment_tolerance = float(
        protocol["packmol"]["packmol_containment_numerical_tolerance_A"]
    )
    for system in protocol["systems"]:
        key = system["calculation_key"]
        target = read_xyz(REPOSITORY_ROOT / sources[system["target_geometry_source"]]["snapshot_path"])
        shell = read_xyz(REPOSITORY_ROOT / sources[system["shell_geometry_source"]]["snapshot_path"])
        cluster_path = _cluster_path(key)
        log_path = _packmol_log_path(key)
        input_path = _packmol_input_path(key)
        atoms = read_xyz(cluster_path)
        n_shell = int(protocol["packmol"]["n_shell"])
        expected_symbols = [atom.symbol for atom in target] + [
            atom.symbol for _ in range(n_shell) for atom in shell
        ]
        count_ok = len(atoms) == len(expected_symbols)
        order_ok = [atom.symbol for atom in atoms] == expected_symbols
        ranges = _molecule_ranges(len(target), len(shell), n_shell)
        minimum = _minimum_intermolecular_distance(atoms, ranges)
        overlap_ok = minimum >= tolerance - 0.01
        half = float(system["box_side_A"]) / 2.0
        shell_atom_indices = [index for atom_range in ranges[1:] for index in atom_range]
        max_violation = max(
            max(abs(value) - half, 0.0)
            for index in shell_atom_indices
            for value in atoms[index].coordinates
        )
        containment_ok = max_violation <= containment_tolerance + 1e-9
        if not (count_ok and order_ok and overlap_ok and containment_ok):
            raise DiagnosticError(
                f"cluster QC failed for {key}: count={count_ok}, order={order_ok}, "
                f"min={minimum}, max_box_violation={max_violation}"
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "phase": system["phase"],
                "calculation_key": key,
                "target_source": system["target_geometry_source"],
                "shell_source": system["shell_geometry_source"],
                "n_shell": str(n_shell),
                "target_atoms": str(len(target)),
                "shell_atoms": str(len(shell)),
                "total_atoms": str(len(atoms)),
                "molecule_count": str(1 + n_shell),
                "box_side_A": f"{float(system['box_side_A']):.3f}",
                "box_min_A": f"{-half:.3f}",
                "box_max_A": f"{half:.3f}",
                "tolerance_A": f"{tolerance:.3f}",
                "seed": str(system["seed"]),
                "packmol_input_path": _repo_relative(input_path),
                "packmol_input_sha256": sha256_file(input_path),
                "packmol_log_path": _repo_relative(log_path),
                "packmol_log_sha256": sha256_file(log_path),
                "packmol_executable": str(binary),
                "packmol_executable_sha256": sha256_file(binary),
                "packmol_version": _packmol_version(log_text),
                "geometry_path": _repo_relative(cluster_path),
                "geometry_sha256": sha256_file(cluster_path),
                "minimum_intermolecular_distance_A": f"{minimum:.10f}",
                "max_shell_box_violation_A": f"{max_violation:.10f}",
                "molecule_count_qc": "pass" if count_ok else "fail",
                "atom_order_qc": "pass" if order_ok else "fail",
                "overlap_qc": "pass" if overlap_ok else "fail",
                "containment_qc": "pass" if containment_ok else "fail",
                "status": "clean",
            }
        )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "cluster_manifest.csv", CLUSTER_FIELDS, rows,
        sort_by=("phase", "calculation_key"),
    )
    return rows


def _coordinate_payload(deck: str) -> str:
    lines = deck.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith("* xyz ")), None)
    if start is None:
        raise DiagnosticError("ORCA deck lacks inline XYZ")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "*"), None)
    if end is None:
        raise DiagnosticError("ORCA deck inline XYZ is unterminated")
    return "".join(lines[start + 1 : end])


def _header_value(deck: str, field: str) -> str:
    prefix = f"# {field}: "
    values = [line[len(prefix):] for line in deck.splitlines() if line.startswith(prefix)]
    if len(values) != 1:
        raise DiagnosticError(f"ORCA deck metadata missing/duplicate: {field}")
    return values[0]


def render_orca_decks() -> list[dict[str, str]]:
    protocol = _protocol()
    solvent_rows = {row["solvent_id"]: row for row in read_csv_rows(REPOSITORY_ROOT / "spec/solvent_smd_registry.csv")}
    registry_path = REPOSITORY_ROOT / "spec/solvent_smd_registry.csv"
    registry_sha = sha256_file(registry_path)
    clusters = {row["calculation_key"]: row for row in read_csv_rows(DIAGNOSTIC_ROOT / "cluster_manifest.csv")}
    rows: list[dict[str, str]] = []
    task_rows: list[dict[str, str]] = []
    for array_task, system in enumerate(protocol["systems"], start=1):
        key = system["calculation_key"]
        cluster = clusters.get(key)
        if cluster is None or cluster["status"] != "clean":
            raise DiagnosticError(f"clean R5 cluster missing: {key}")
        solvent = solvent_rows[system["medium_id"]]
        if solvent["production_status"] != "authoritative_for_jhtvs_ft0806_v3":
            raise DiagnosticError(f"SMD registry row is not authoritative: {system['medium_id']}")
        solvent_row_sha = csv_record_sha256(
            registry_path, key_field="solvent_id", key_value=system["medium_id"]
        )
        xyz_path = REPOSITORY_ROOT / cluster["geometry_path"]
        geometry = {
            "geometry_key": f"explicit_r5:{key}",
            "xyz_sha256": cluster["geometry_sha256"],
        }
        payload_hashes: list[str] = []
        for sequence, (role, state) in enumerate(
            zip(("reduced", "oxidized"), system["states"], strict=True), start=1
        ):
            charge, multiplicity = int(state[0]), int(state[1])
            basis = system[f"optfreq_basis_{role}"]
            job_id = f"R5EOX_A{array_task:02d}_{'RED' if role == 'reduced' else 'OX'}"
            hirshfeld = role == "oxidized"
            method_id = (
                f"T2_wB97X-D3_{basis}_OptFreq_def2-TZVPD-SP_SMD_ExplicitR5"
                f"{'_Hirshfeld' if hirshfeld else ''}_v1"
            )
            job = {
                "workflow_revision": WORKFLOW_REVISION,
                "job_id": job_id,
                "job_class": "optfreq",
                "state_id": f"{key}_{role}_q{charge}_m{multiplicity}",
                "solvent_id": system["medium_id"],
                "formal_charge": str(charge),
                "multiplicity": str(multiplicity),
                "method_id": method_id,
                "functional": protocol["orca"]["functional"],
                "optfreq_basis": basis,
                "final_sp_basis": protocol["orca"]["final_sp_basis"],
                "final_sp_hirshfeld": str(hirshfeld).lower(),
                "nprocs": str(protocol["orca"]["nprocs"]),
                "maxcore_mb_per_rank": str(protocol["orca"]["maxcore_mb_per_rank"]),
            }
            deck = render_optfreq_deck(
                job, geometry, xyz_path, solvent,
                registry_sha256=registry_sha,
                registry_row_sha256=solvent_row_sha,
            )
            payload_sha = sha256_bytes(_coordinate_payload(deck).encode("utf-8"))
            payload_hashes.append(payload_sha)
            input_path = ORCA_ROOT / "jobs" / job_id / f"{job_id}.inp"
            _write_immutable_bytes(input_path, deck.encode("utf-8"))
            row = {
                "phase": system["phase"],
                "calculation_key": key,
                "state_role": role,
                "job_id": job_id,
                "job_class": "optfreq",
                "state_id": job["state_id"],
                "solvent_id": system["medium_id"],
                "formal_charge": str(charge),
                "multiplicity": str(multiplicity),
                "input_path": _repo_relative(input_path),
                "input_sha256": sha256_file(input_path),
                "output_path": _repo_relative(input_path.with_suffix(".out")),
                "geometry_key": geometry["geometry_key"],
                "geometry_sha256": geometry["xyz_sha256"],
                "coordinate_payload_sha256": payload_sha,
                "smd_registry_row_sha256": solvent_row_sha,
                "smd_payload_sha256": smd_payload_sha256(solvent),
                "exact_reuse_key": _header_value(deck, "exact_reuse_key"),
                "workflow_revision": WORKFLOW_REVISION,
                "method_id": method_id,
                "thermochemistry_convention_id": THERMOCHEMISTRY_CONVENTION_ID,
                "functional": protocol["orca"]["functional"],
                "optfreq_basis": basis,
                "final_sp_basis": protocol["orca"]["final_sp_basis"],
                "final_sp_hirshfeld": str(hirshfeld).lower(),
                "nprocs": str(protocol["orca"]["nprocs"]),
                "maxcore_mb_per_rank": str(protocol["orca"]["maxcore_mb_per_rank"]),
                "planning_core_h": str(system["planning_core_h_per_state"]),
                "status": "ready",
            }
            rows.append(row)
            task_rows.append(
                {field: value for field, value in {
                    "array_task": str(array_task),
                    "sequence": str(sequence),
                    "job_id": job_id,
                    "job_class": "optfreq",
                    "input_path": row["input_path"],
                    "input_sha256": row["input_sha256"],
                    "output_path": row["output_path"],
                    "nprocs": row["nprocs"],
                    "planning_core_h": row["planning_core_h"],
                    "workflow_revision": WORKFLOW_REVISION,
                    "method_id": method_id,
                }.items()}
            )
        if len(set(payload_hashes)) != 1:
            raise DiagnosticError(f"redox pair initial coordinates differ: {key}")
    write_csv_deterministic(
        ORCA_MANIFEST_PATH, ORCA_FIELDS, rows,
        sort_by=("phase", "calculation_key", "state_role"),
    )
    task_text = "\t".join(TASK_FIELDS) + "\n" + "".join(
        "\t".join(row[field] for field in TASK_FIELDS) + "\n" for row in task_rows
    )
    _write_immutable_bytes(ORCA_TASKS_PATH, task_text.encode("utf-8"))
    return rows


def prepare_continuation(logical_job_id: str) -> dict[str, str]:
    """Prepare the sole allowed same-method continuation for one failed state."""

    if _continuation_rows():
        existing = [
            row for row in _continuation_rows()
            if row["logical_job_id"] == logical_job_id
        ]
        if existing:
            raise DiagnosticError(f"continuation already exists: {logical_job_id}")
    originals = {
        row["job_id"]: row for row in read_csv_rows(ORCA_MANIFEST_PATH)
    }
    original = originals.get(logical_job_id)
    if original is None:
        raise DiagnosticError(f"unknown logical job: {logical_job_id}")
    original_input = REPOSITORY_ROOT / original["input_path"]
    original_output = REPOSITORY_ROOT / original["output_path"]
    if not original_output.is_file() or original_output.is_symlink():
        raise DiagnosticError(f"failed output is missing or unsafe: {logical_job_id}")
    output_text = original_output.read_text(encoding="utf-8", errors="replace")
    if (
        "The optimization did not converge but reached the maximum" not in output_text
        or "ERROR !!!" not in output_text
        or "THE OPTIMIZATION HAS CONVERGED" in output_text
    ):
        raise DiagnosticError(f"continuation trigger is not a MaxIter failure: {logical_job_id}")
    trigger_geometry = original_input.parent / f"{logical_job_id}_Compound_1.xyz"
    if not trigger_geometry.is_file() or trigger_geometry.is_symlink():
        raise DiagnosticError(f"continuation geometry is missing or unsafe: {trigger_geometry}")
    if [atom.symbol for atom in read_xyz(trigger_geometry)] != [
        atom.symbol for atom in read_xyz(REPOSITORY_ROOT / _cluster_row(original["calculation_key"])["geometry_path"])
    ]:
        raise DiagnosticError(f"continuation geometry composition/order drifted: {logical_job_id}")

    attempt_job_id = f"{logical_job_id}_CONT1"
    attempt_input = ORCA_ROOT / "jobs" / attempt_job_id / f"{attempt_job_id}.inp"
    solvent_rows = {
        row["solvent_id"]: row
        for row in read_csv_rows(REPOSITORY_ROOT / "spec/solvent_smd_registry.csv")
    }
    registry_path = REPOSITORY_ROOT / "spec/solvent_smd_registry.csv"
    solvent = solvent_rows[original["solvent_id"]]
    trigger_geometry_sha = sha256_file(trigger_geometry)
    source_geometry = CONTINUATION_ROOT / f"{attempt_job_id}_source.xyz"
    _write_immutable_bytes(source_geometry, trigger_geometry.read_bytes())
    source_sha = sha256_file(source_geometry)
    if source_sha != trigger_geometry_sha:
        raise DiagnosticError(f"continuation geometry snapshot hash mismatch: {logical_job_id}")
    geometry = {
        "geometry_key": f"continuation1:{logical_job_id}",
        "xyz_sha256": source_sha,
    }
    job = {
        "workflow_revision": original["workflow_revision"],
        "job_id": attempt_job_id,
        "job_class": original["job_class"],
        "state_id": original["state_id"],
        "solvent_id": original["solvent_id"],
        "formal_charge": original["formal_charge"],
        "multiplicity": original["multiplicity"],
        "method_id": original["method_id"],
        "functional": original["functional"],
        "optfreq_basis": original["optfreq_basis"],
        "final_sp_basis": original["final_sp_basis"],
        "final_sp_hirshfeld": original["final_sp_hirshfeld"],
        "nprocs": original["nprocs"],
        "maxcore_mb_per_rank": original["maxcore_mb_per_rank"],
    }
    solvent_row_sha = csv_record_sha256(
        registry_path, key_field="solvent_id", key_value=original["solvent_id"]
    )
    deck = render_optfreq_deck(
        job, geometry, source_geometry, solvent,
        registry_sha256=sha256_file(registry_path),
        registry_row_sha256=solvent_row_sha,
    )
    payload_sha = sha256_bytes(_coordinate_payload(deck).encode("utf-8"))
    _write_immutable_bytes(attempt_input, deck.encode("utf-8"))
    task_path = CONTINUATION_TASK_ROOT / f"{attempt_job_id}.tsv"
    task_row = {
        "array_task": "1",
        "sequence": "1",
        "job_id": attempt_job_id,
        "job_class": original["job_class"],
        "input_path": _repo_relative(attempt_input),
        "input_sha256": sha256_file(attempt_input),
        "output_path": _repo_relative(attempt_input.with_suffix(".out")),
        "nprocs": original["nprocs"],
        "planning_core_h": original["planning_core_h"],
        "workflow_revision": original["workflow_revision"],
        "method_id": original["method_id"],
    }
    task_text = "\t".join(TASK_FIELDS) + "\n" + "\t".join(
        task_row[field] for field in TASK_FIELDS
    ) + "\n"
    _write_immutable_bytes(task_path, task_text.encode("utf-8"))
    cluster = _cluster_row(original["calculation_key"])
    row = {
        "logical_job_id": logical_job_id,
        "attempt": "1",
        "attempt_job_id": attempt_job_id,
        "calculation_key": original["calculation_key"],
        "state_role": original["state_role"],
        "trigger": "optimization_maxiter_200",
        "trigger_output_path": original["output_path"],
        "trigger_output_sha256": sha256_file(original_output),
        "trigger_geometry_path": _repo_relative(trigger_geometry),
        "trigger_geometry_sha256": trigger_geometry_sha,
        "source_geometry_path": _repo_relative(source_geometry),
        "source_geometry_sha256": source_sha,
        "root_initial_cluster_sha256": cluster["geometry_sha256"],
        "input_path": task_row["input_path"],
        "input_sha256": task_row["input_sha256"],
        "output_path": task_row["output_path"],
        "geometry_key": geometry["geometry_key"],
        "coordinate_payload_sha256": payload_sha,
        "exact_reuse_key": _header_value(deck, "exact_reuse_key"),
        "status": "prepared_not_submitted",
        "task_path": _repo_relative(task_path),
        "task_sha256": sha256_file(task_path),
        "scheduler_job_id": "",
        "submitted_at_utc": "",
    }
    rows = _continuation_rows() + [row]
    write_csv_deterministic(
        CONTINUATION_MANIFEST_PATH, CONTINUATION_FIELDS, rows,
        sort_by=("logical_job_id", "attempt"),
    )
    return row


def _cluster_row(calculation_key: str) -> dict[str, str]:
    matches = [
        row for row in read_csv_rows(DIAGNOSTIC_ROOT / "cluster_manifest.csv")
        if row["calculation_key"] == calculation_key
    ]
    if len(matches) != 1:
        raise DiagnosticError(f"cluster manifest row is not unique: {calculation_key}")
    return matches[0]


def validate_prepared() -> dict[str, Any]:
    protocol = _protocol()
    checks: dict[str, Any] = {}
    issues: list[str] = []

    input_rows = read_csv_rows(DIAGNOSTIC_ROOT / "input_file_manifest.csv")
    checks["input_file_count"] = len(input_rows)
    if len(input_rows) != 7:
        issues.append("input_file_count")
    for row in input_rows:
        path = REPOSITORY_ROOT / row["snapshot_path"]
        if not path.is_file() or sha256_file(path) != protocol["input_files"][row["name"]]:
            issues.append(f"input_sha:{row['name']}")

    benchmark = read_csv_rows(DIAGNOSTIC_ROOT / "benchmark_records.csv")
    registry = read_csv_rows(DIAGNOSTIC_ROOT / "calculation_registry.csv")
    checks["eligible_record_count"] = len(benchmark)
    checks["phase_a_unique_key_count"] = len(registry)
    if len(benchmark) != 15:
        issues.append("eligible_record_count")
    if len(registry) != 6 or {row["phase"] for row in registry} != {"A"}:
        issues.append("phase_a_selection")

    clusters = read_csv_rows(DIAGNOSTIC_ROOT / "cluster_manifest.csv")
    clusters_by_key = {row["calculation_key"]: row for row in clusters}
    checks["cluster_count"] = len(clusters)
    if len(clusters) != 6:
        issues.append("cluster_count")
    for row in clusters:
        if row["n_shell"] != "5" or row["molecule_count"] != "6":
            issues.append(f"cluster_composition:{row['calculation_key']}")
        if row["status"] != "clean":
            issues.append(f"cluster_qc:{row['calculation_key']}")
        if row["containment_qc"] != "pass":
            issues.append(f"cluster_containment:{row['calculation_key']}")

    solvent_rows = {
        row["solvent_id"]: row
        for row in read_csv_rows(REPOSITORY_ROOT / "spec/solvent_smd_registry.csv")
    }
    manifests = read_csv_rows(ORCA_MANIFEST_PATH)
    checks["orca_job_count"] = len(manifests)
    if len(manifests) != 12:
        issues.append("orca_job_count")
    by_key: dict[str, list[dict[str, str]]] = {}
    for row in manifests:
        by_key.setdefault(row["calculation_key"], []).append(row)
        input_path = REPOSITORY_ROOT / row["input_path"]
        if sha256_file(input_path) != row["input_sha256"]:
            issues.append(f"deck_sha:{row['job_id']}")
            continue
        text = input_path.read_text(encoding="utf-8")
        system = _system_map()[row["calculation_key"]]
        solvent = solvent_rows[row["solvent_id"]]
        expected_basis = system[f"optfreq_basis_{row['state_role']}"]
        if f"! wB97X-D3 {expected_basis} def2/J RIJCOSX TightSCF DEFGRID3 Opt Freq" not in text:
            issues.append(f"optfreq_method:{row['job_id']}")
        if "! wB97X-D3 def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3" not in text:
            issues.append(f"final_sp_method:{row['job_id']}")
        if render_smd_block(solvent) not in text or text.count(render_smd_block(solvent)) != 2:
            issues.append(f"smd_routing:{row['job_id']}")
        if f"{row['job_id']}_Compound_1.xyz" not in text:
            issues.append(f"optimized_sp_geometry:{row['job_id']}")
        if ("Hirshfeld" in text) != (row["state_role"] == "oxidized"):
            issues.append(f"hirshfeld_routing:{row['job_id']}")
        if row["optfreq_basis"] != expected_basis or row["final_sp_basis"] != "def2-TZVPD":
            issues.append(f"basis_manifest:{row['job_id']}")
        if row["formal_charge"] != str(system["states"][0 if row["state_role"] == "reduced" else 1][0]):
            issues.append(f"charge:{row['job_id']}")
        if row["multiplicity"] != str(system["states"][0 if row["state_role"] == "reduced" else 1][1]):
            issues.append(f"multiplicity:{row['job_id']}")
    for key, pair in by_key.items():
        if len(pair) != 2:
            issues.append(f"pair_cardinality:{key}")
            continue
        if len({row["geometry_sha256"] for row in pair}) != 1:
            issues.append(f"pair_initial_cluster_hash:{key}")
        if len({row["coordinate_payload_sha256"] for row in pair}) != 1:
            issues.append(f"pair_initial_coordinate_payload:{key}")

    task_rows: list[dict[str, str]] = []
    with ORCA_TASKS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TASK_FIELDS:
            issues.append("task_table_header")
        task_rows = list(reader)
    checks["task_row_count"] = len(task_rows)
    checks["array_task_count"] = len({row["array_task"] for row in task_rows})
    if len(task_rows) != 12 or checks["array_task_count"] != 6:
        issues.append("task_table_cardinality")

    continuations = _continuation_rows()
    checks["continuation_count"] = len(continuations)
    original_by_job = {row["job_id"]: row for row in manifests}
    for attempt in continuations:
        original = original_by_job.get(attempt["logical_job_id"])
        if original is None:
            issues.append(f"continuation_unknown_job:{attempt['logical_job_id']}")
            continue
        input_path = REPOSITORY_ROOT / attempt["input_path"]
        source_path = REPOSITORY_ROOT / attempt["source_geometry_path"]
        task_path = REPOSITORY_ROOT / attempt["task_path"]
        if not input_path.is_file() or sha256_file(input_path) != attempt["input_sha256"]:
            issues.append(f"continuation_input_sha:{attempt['attempt_job_id']}")
            continue
        if not source_path.is_file() or sha256_file(source_path) != attempt["source_geometry_sha256"]:
            issues.append(f"continuation_geometry_sha:{attempt['attempt_job_id']}")
        if not task_path.is_file() or sha256_file(task_path) != attempt["task_sha256"]:
            issues.append(f"continuation_task_sha:{attempt['attempt_job_id']}")
        root_cluster = clusters_by_key.get(original["calculation_key"])
        if (
            root_cluster is None
            or attempt["root_initial_cluster_sha256"]
            != root_cluster["geometry_sha256"]
        ):
            issues.append(f"continuation_root_lineage:{attempt['attempt_job_id']}")
        text = input_path.read_text(encoding="utf-8")
        for field in ("formal_charge", "multiplicity", "method_id", "solvent_id"):
            if _header_value(text, field) != original[field]:
                issues.append(f"continuation_{field}_drift:{attempt['attempt_job_id']}")
        if f"! {original['functional']} {original['optfreq_basis']} def2/J RIJCOSX TightSCF DEFGRID3 Opt Freq" not in text:
            issues.append(f"continuation_optfreq_method_drift:{attempt['attempt_job_id']}")
        if f"! {original['functional']} {original['final_sp_basis']} def2/J RIJCOSX TightSCF DEFGRID3" not in text:
            issues.append(f"continuation_final_sp_method_drift:{attempt['attempt_job_id']}")
        solvent = solvent_rows[original["solvent_id"]]
        if render_smd_block(solvent) not in text or text.count(render_smd_block(solvent)) != 2:
            issues.append(f"continuation_smd_drift:{attempt['attempt_job_id']}")
        if ("Hirshfeld" in text) != (original["final_sp_hirshfeld"] == "true"):
            issues.append(f"continuation_hirshfeld_drift:{attempt['attempt_job_id']}")
        if sha256_bytes(_coordinate_payload(text).encode("utf-8")) != attempt["coordinate_payload_sha256"]:
            issues.append(f"continuation_coordinate_payload:{attempt['attempt_job_id']}")

    script_text = Path(__file__).read_text(encoding="utf-8")
    imports = [line for line in script_text.splitlines() if line.startswith(("import ", "from "))]
    forbidden = [line for line in imports if re.search(r"\b(?:mace|torch)\b", line, re.IGNORECASE)]
    checks["forbidden_ml_imports"] = forbidden
    if forbidden:
        issues.append("forbidden_ml_import")

    report = {
        "status": "PASS" if not issues else "FAIL",
        "workflow_revision": WORKFLOW_REVISION,
        "checks": checks,
        "issues": issues,
    }
    _write_json(DIAGNOSTIC_ROOT / "prepared_qc.json", report)
    if issues:
        raise DiagnosticError(f"prepared QC failed: {issues}")
    return report


def _fragment_ranges_for_key(key: str) -> tuple[range, ...]:
    cluster = next(
        row for row in read_csv_rows(DIAGNOSTIC_ROOT / "cluster_manifest.csv")
        if row["calculation_key"] == key
    )
    return _molecule_ranges(
        int(cluster["target_atoms"]), int(cluster["shell_atoms"]), int(cluster["n_shell"])
    )


def _center_of_mass(atoms: Sequence[XYZAtom], indices: range) -> tuple[float, float, float]:
    masses = [ATOMIC_MASS[atoms[index].symbol] for index in indices]
    total_mass = sum(masses)
    return tuple(
        sum(
            getattr(atoms[index], axis) * mass
            for index, mass in zip(indices, masses, strict=True)
        )
        / total_mass for axis in ("x", "y", "z")
    )  # type: ignore[return-value]


def _fragment_connectivity_and_distances(
    *, key: str, job_id: str, state_role: str, initial_path: Path, optimized_path: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    initial = read_xyz(initial_path)
    optimized = read_xyz(optimized_path)
    ranges = _fragment_ranges_for_key(key)
    if [atom.symbol for atom in initial] != [atom.symbol for atom in optimized]:
        return (
            {
                "status": "flagged",
                "reasons": ["optimized_geometry_composition_or_order_changed"],
                "target_broken": 0,
                "target_formed": 0,
                "shell_broken": 0,
                "shell_formed": 0,
                "inter_formed": 0,
                "inter_broken": 0,
            },
            [],
        )
    initial_bonds = inferred_bonds(initial)
    optimized_bonds = inferred_bonds(optimized)
    membership = {
        atom_index: molecule_index
        for molecule_index, atom_range in enumerate(ranges)
        for atom_index in atom_range
    }
    broken = initial_bonds - optimized_bonds
    formed = optimized_bonds - initial_bonds

    def classify(bonds: set[frozenset[int]]) -> tuple[int, int, int]:
        target = shell = inter = 0
        for bond in bonds:
            first, second = tuple(bond)
            first_molecule, second_molecule = membership[first], membership[second]
            if first_molecule != second_molecule:
                inter += 1
            elif first_molecule == 0:
                target += 1
            else:
                shell += 1
        return target, shell, inter

    target_broken, shell_broken, inter_broken = classify(broken)
    target_formed, shell_formed, inter_formed = classify(formed)
    reasons: list[str] = []
    if target_broken or target_formed:
        reasons.append("target_covalent_connectivity_changed")
    if shell_broken or shell_formed:
        reasons.append("shell_covalent_connectivity_changed")
    if inter_broken or inter_formed:
        reasons.append("intermolecular_covalent_connectivity_changed")
    initial_target_center = _center_of_mass(initial, ranges[0])
    optimized_target_center = _center_of_mass(optimized, ranges[0])
    rows: list[dict[str, str]] = []
    for shell_index, atom_range in enumerate(ranges[1:], start=1):
        before = math.dist(initial_target_center, _center_of_mass(initial, atom_range))
        after = math.dist(optimized_target_center, _center_of_mass(optimized, atom_range))
        rows.append(
            {
                "calculation_key": key,
                "job_id": job_id,
                "state_role": state_role,
                "shell_index": str(shell_index),
                "initial_target_shell_COM_distance_A": f"{before:.10f}",
                "optimized_target_shell_COM_distance_A": f"{after:.10f}",
                "delta_COM_distance_A": f"{after - before:.10f}",
                "target_internal_bonds_broken": str(target_broken),
                "target_internal_bonds_formed": str(target_formed),
                "shell_internal_bonds_broken": str(shell_broken),
                "shell_internal_bonds_formed": str(shell_formed),
                "intermolecular_bonds_formed": str(inter_formed),
                "intermolecular_bonds_broken": str(inter_broken),
                "fragment_qc_status": "clean" if not reasons else "flagged",
            }
        )
    return (
        {
            "status": "clean" if not reasons else "flagged",
            "reasons": reasons,
            "target_broken": target_broken,
            "target_formed": target_formed,
            "shell_broken": shell_broken,
            "shell_formed": shell_formed,
            "inter_formed": inter_formed,
            "inter_broken": inter_broken,
        },
        rows,
    )


_HIRSHFELD_ATOM_RE = re.compile(
    r"^\s*(\d+)\s+[A-Z][a-z]?\s+[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def parse_hirshfeld_spins(text: str, atom_count: int) -> list[Decimal]:
    start = text.rfind("HIRSHFELD ANALYSIS")
    if start < 0:
        raise DiagnosticError("Hirshfeld analysis section missing")
    matches = _HIRSHFELD_ATOM_RE.findall(text[start:])
    spins_by_index = {int(index): Decimal(spin) for index, spin in matches}
    if set(spins_by_index) != set(range(atom_count)):
        raise DiagnosticError(
            f"Hirshfeld atom coverage mismatch: {len(spins_by_index)} != {atom_count}"
        )
    return [spins_by_index[index] for index in range(atom_count)]


def aggregate_fragment_spins(
    spins: Sequence[Decimal], ranges: Sequence[range], *, anion: bool
) -> tuple[list[Decimal], dict[str, str]]:
    fragments = [sum((spins[index] for index in atom_range), Decimal("0")) for atom_range in ranges]
    total = sum(fragments, Decimal("0"))
    if total == 0:
        raise DiagnosticError("zero total Hirshfeld spin")
    dominant_index = max(range(len(fragments)), key=lambda index: abs(fragments[index]))
    target_fraction = fragments[0] / total
    if anion:
        identity = "clean" if dominant_index == 0 else "oxidation_identity_mismatch"
    else:
        identity = "not_applicable_identical_solvent_fragments"
    return fragments, {
        "target_spin_fraction": f"{target_fraction:.12f}",
        "dominant_spin_fragment": "target" if dominant_index == 0 else f"shell_{dominant_index}",
        "total_hirshfeld_spin": f"{total:.12f}",
        "oxidation_identity_status": identity,
    }


def _pinned_conversion(source_checkout: Path):
    protocol = _protocol()["reference_conversion"]
    path = source_checkout / protocol["pinned_relative_path"]
    if not path.is_file() or path.is_symlink() or sha256_file(path) != protocol["pinned_sha256"]:
        raise DiagnosticError("pinned Ag/AgCl conversion source missing or hash drifted")
    module_name = "jhtvs_explicit_r5_pinned_parse_orca"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError("cannot load pinned Ag/AgCl conversion module")
    sys.path.insert(0, str(path.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    conversion = getattr(module, protocol["callable"], None)
    if not callable(conversion):
        raise DiagnosticError("pinned Ag/AgCl conversion callable missing")
    return conversion


def _explicit_eox_from_raw(
    reduced_G_Eh: str, oxidized_G_Eh: str, conversion
) -> tuple[Decimal, Decimal, Decimal]:
    delta_eV = (Decimal(oxidized_G_Eh) - Decimal(reduced_G_Eh)) * HARTREE_TO_EV_DECIMAL
    primary = Decimal(str(conversion(float(delta_eV))))
    independent_delta = Decimal(oxidized_G_Eh) * HARTREE_TO_EV_DECIMAL
    independent_delta -= Decimal(reduced_G_Eh) * HARTREE_TO_EV_DECIMAL
    independent = Decimal(str(conversion(float(independent_delta))))
    return delta_eV, primary, independent


def _format_decimal(value: Decimal | None, digits: int = 12) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def pair_qc_status(statuses: Sequence[str]) -> str:
    if "missing" in statuses:
        return "missing"
    if "flagged" in statuses:
        return "flagged"
    if statuses and all(status == "clean" for status in statuses):
        return "clean"
    raise DiagnosticError(f"unsupported pair QC statuses: {list(statuses)}")


def optimized_geometries_independent(reduced_sha256: str, oxidized_sha256: str) -> bool:
    return bool(reduced_sha256 and oxidized_sha256 and reduced_sha256 != oxidized_sha256)


def _metric_values(experimental: Sequence[float], predicted: Sequence[float]) -> dict[str, float | str]:
    errors = [prediction - observation for observation, prediction in zip(experimental, predicted, strict=True)]
    absolute = [abs(error) for error in errors]
    mean_exp = statistics.fmean(experimental)
    ss_total = sum((value - mean_exp) ** 2 for value in experimental)
    ss_residual = sum(error * error for error in errors)
    r_squared: float | str = ""
    if len(experimental) >= 2 and ss_total > 0:
        r_squared = 1.0 - ss_residual / ss_total
    return {
        "mae_V": statistics.fmean(absolute),
        "rmse_V": math.sqrt(statistics.fmean([error * error for error in errors])),
        "mean_signed_error_V": statistics.fmean(errors),
        "median_absolute_error_V": statistics.median(absolute),
        "r_squared": r_squared,
    }


def build_metrics(comparison: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    clean = [
        row for row in comparison
        if row["calculation_qc"] == "clean" and row["explicit_R5_DFT_Eox_V"]
    ]
    method_columns = {
        "implicit_calibrated_xTB": "implicit_calibrated_xTB_Eox_V",
        "implicit_DFT": "implicit_DFT_Eox_V",
        "explicit_R5_DFT": "explicit_R5_DFT_Eox_V",
    }
    rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {"shared_completed_clean_subset": {}}
    for category, property_name in (("solvent", "solvent_eox"), ("anion", "anion_eox")):
        category_rows = [row for row in clean if row["property"] == property_name]
        category_summary: dict[str, Any] = {
            "n_experimental_records": len(category_rows),
            "n_unique_calculation_keys": len({row["calculation_key"] for row in category_rows}),
            "record_level": {},
        }
        aggregations: list[tuple[str, list[dict[str, str]]]] = [("record", category_rows)]
        if category == "anion":
            macro: list[dict[str, str]] = []
            for key in sorted({row["calculation_key"] for row in category_rows}):
                key_rows = [row for row in category_rows if row["calculation_key"] == key]
                macro.append(
                    {
                        "calculation_key": key,
                        "experimental_Eox_V": str(statistics.fmean(float(row["experimental_Eox_V"]) for row in key_rows)),
                        **{
                            column: str(statistics.fmean(float(row[column]) for row in key_rows))
                            for column in method_columns.values()
                        },
                    }
                )
            aggregations.append(("unique_calculation_macro", macro))
            category_summary["unique_calculation_macro"] = {}
        for aggregation, metric_source in aggregations:
            if not metric_source:
                continue
            experimental = [float(row["experimental_Eox_V"]) for row in metric_source]
            improvements_by_key: dict[str, list[float]] = {}
            for row in metric_source:
                improvement = abs(float(row["implicit_DFT_Eox_V"]) - float(row["experimental_Eox_V"]))
                improvement -= abs(float(row["explicit_R5_DFT_Eox_V"]) - float(row["experimental_Eox_V"]))
                improvements_by_key.setdefault(row["calculation_key"], []).append(improvement)
            key_improvements = [statistics.fmean(values) for values in improvements_by_key.values()]
            improved = sum(value > 0 for value in key_improvements)
            worsened = sum(value < 0 for value in key_improvements)
            destination = category_summary[
                "record_level" if aggregation == "record" else "unique_calculation_macro"
            ]
            for method, column in method_columns.items():
                metrics = _metric_values(
                    experimental, [float(row[column]) for row in metric_source]
                )
                rendered = {
                    key: ("" if value == "" else f"{float(value):.12f}")
                    for key, value in metrics.items()
                }
                rendered.update(
                    {
                        "category": category,
                        "aggregation": aggregation,
                        "method": method,
                        "n_experimental_records": str(len(category_rows)),
                        "n_unique_calculation_keys": str(len(improvements_by_key)),
                        "unique_keys_improved_vs_implicit_DFT": str(improved),
                        "unique_keys_worsened_vs_implicit_DFT": str(worsened),
                        "mean_improvement_vs_implicit_DFT_V": f"{statistics.fmean(key_improvements):.12f}",
                        "median_improvement_vs_implicit_DFT_V": f"{statistics.median(key_improvements):.12f}",
                    }
                )
                rows.append(rendered)
                destination[method] = rendered
        summary["shared_completed_clean_subset"][category] = category_summary
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "metrics.csv", METRIC_FIELDS, rows,
        sort_by=("category", "aggregation", "method"),
    )
    _write_json(DIAGNOSTIC_ROOT / "metrics_summary.json", summary)
    return rows, summary


def _write_figures(comparison: Sequence[dict[str, str]]) -> None:
    clean = [
        row for row in comparison
        if row["calculation_qc"] == "clean" and row["explicit_R5_DFT_Eox_V"]
    ]
    if not clean:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = (
        ("Implicit calibrated xTB", "implicit_calibrated_xTB_Eox_V", "o"),
        ("Implicit DFT", "implicit_DFT_Eox_V", "s"),
        ("Explicit-R5 DFT", "explicit_R5_DFT_Eox_V", "^"),
    )
    for category, property_name, filename in (
        ("Solvent", "solvent_eox", "solvent_exp_vs_methods.png"),
        ("Anion", "anion_eox", "anion_exp_vs_methods.png"),
    ):
        rows = [row for row in clean if row["property"] == property_name]
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        observed = [float(row["experimental_Eox_V"]) for row in rows]
        all_values = list(observed)
        for label, column, marker in methods:
            predicted = [float(row[column]) for row in rows]
            all_values.extend(predicted)
            axis.scatter(observed, predicted, label=label, marker=marker, alpha=0.82)
        lower, upper = min(all_values) - 0.15, max(all_values) + 0.15
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1, label="Identity")
        axis.set(xlabel="Experimental Eox vs Ag/AgCl (V)", ylabel="Computed Eox vs Ag/AgCl (V)", title=f"{category} shared completed-clean subset")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(DIAGNOSTIC_ROOT / filename, dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    keys = sorted({row["calculation_key"] for row in clean})
    values = []
    for key in keys:
        rows = [row for row in clean if row["calculation_key"] == key]
        values.append(statistics.fmean(float(row["explicit_improvement_over_implicit_DFT_V"]) for row in rows))
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]
    axis.bar(range(len(keys)), values, color=colors)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(range(len(keys)), [key.replace("__", "/") for key in keys], rotation=35, ha="right")
    axis.set_ylabel("Absolute-error improvement vs implicit DFT (V)")
    figure.tight_layout()
    figure.savefig(DIAGNOSTIC_ROOT / "absolute_error_improvement.png", dpi=180)
    plt.close(figure)


def _write_report(
    unique_rows: Sequence[dict[str, str]], metrics_summary: Mapping[str, Any], qc: Mapping[str, Any]
) -> None:
    lines = [
        "# Explicit-R5 DFT Eox benchmark report",
        "",
        f"Workflow: `{WORKFLOW_REVISION}`",
        "",
        f"Completed clean unique systems: {qc['completed_clean_unique_systems']} / {qc['attempted_unique_systems']}",
        "",
        "| calculation key | explicit R5 Eox / V | QC | reasons |",
        "|---|---:|---|---|",
    ]
    for row in unique_rows:
        lines.append(
            f"| {row['calculation_key']} | {row['explicit_R5_DFT_Eox_vs_AgAgCl_V'] or '—'} | {row['qc_status']} | {row['qc_reasons'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "All accuracy metrics use the same completed-clean subset for implicit calibrated xTB, implicit DFT, and explicit-R5 DFT. Anion pairs with solvent-dominant oxidized Hirshfeld spin are retained as flagged raw values and excluded from primary metrics.",
            "",
            "See `metrics_summary.json`, `record_comparison.csv`, `fragment_spin_qc.csv`, and `qc.json` for machine-readable results.",
        ]
    )
    _write_generated(DIAGNOSTIC_ROOT / "REPORT.md", "\n".join(lines) + "\n")


def collect(*, source_checkout: Path) -> dict[str, Any]:
    validate_prepared()
    conversion = _pinned_conversion(source_checkout.resolve())
    manifests = _effective_manifests()
    clusters = {
        row["calculation_key"]: row
        for row in read_csv_rows(DIAGNOSTIC_ROOT / "cluster_manifest.csv")
    }
    solvent_rows = {
        row["solvent_id"]: row
        for row in read_csv_rows(REPOSITORY_ROOT / "spec/solvent_smd_registry.csv")
    }
    parsed_rows: list[dict[str, Any]] = []
    state_payloads: list[dict[str, Any]] = []
    distance_rows: list[dict[str, str]] = []
    spin_rows: list[dict[str, str]] = []
    specialized_by_job: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        key = manifest["calculation_key"]
        cluster = clusters[key]
        geometry = {
            "geometry_key": manifest["geometry_key"],
            "xyz_sha256": manifest["geometry_sha256"],
            "xyz_path": manifest.get("input_geometry_path", cluster["geometry_path"]),
        }
        solvent = solvent_rows[manifest["solvent_id"]]
        parsed = parse_job_result(
            manifest=manifest,
            job=manifest,
            geometry=geometry,
            solvent=solvent,
            repository_root=REPOSITORY_ROOT,
            registry_row_sha256=manifest["smd_registry_row_sha256"],
        )
        parsed_rows.append({"calculation_key": key, "state_role": manifest["state_role"], **parsed})
        reasons = [reason for reason in str(parsed["qc_reasons"]).split(";") if reason]
        status = str(parsed["qc_status"])
        fragment: dict[str, Any] = {"status": status, "reasons": []}
        optimized_path_text = str(parsed["optimized_geometry_path"])
        if optimized_path_text:
            optimized_path = REPOSITORY_ROOT / optimized_path_text
            if optimized_path.is_file():
                fragment, distances = _fragment_connectivity_and_distances(
                    key=key,
                    job_id=manifest["job_id"],
                    state_role=manifest["state_role"],
                    initial_path=REPOSITORY_ROOT / cluster["geometry_path"],
                    optimized_path=optimized_path,
                )
                distance_rows.extend(distances)
                reasons.extend(fragment["reasons"])
                if fragment["status"] == "flagged" and status == "clean":
                    status = "flagged"
        identity_status = "not_evaluated"
        if manifest["state_role"] == "oxidized":
            if parsed["normal_termination"] != "true":
                spin_rows.append(
                    {
                        "calculation_key": key, "job_id": manifest["job_id"],
                        "molecule_index": "", "fragment_role": "", "hirshfeld_spin": "",
                        "spin_fraction": "", "target_spin_fraction": "",
                        "dominant_spin_fragment": "", "total_hirshfeld_spin": "",
                        "oxidation_identity_status": "missing", "qc_status": "missing",
                    }
                )
                identity_status = "missing"
            else:
                try:
                    output_text = (REPOSITORY_ROOT / manifest["output_path"]).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    ranges = _fragment_ranges_for_key(key)
                    spins = parse_hirshfeld_spins(output_text, int(cluster["total_atoms"]))
                    fragments, identity = aggregate_fragment_spins(
                        spins, ranges, anion=_system_map()[key]["property"] == "anion_eox"
                    )
                    identity_status = identity["oxidation_identity_status"]
                    total = Decimal(identity["total_hirshfeld_spin"])
                    for index, fragment_spin in enumerate(fragments):
                        spin_rows.append(
                            {
                                "calculation_key": key,
                                "job_id": manifest["job_id"],
                                "molecule_index": str(index),
                                "fragment_role": "target" if index == 0 else f"shell_{index}",
                                "hirshfeld_spin": f"{fragment_spin:.12f}",
                                "spin_fraction": f"{fragment_spin / total:.12f}",
                                **identity,
                                "qc_status": "clean" if identity_status != "oxidation_identity_mismatch" else "flagged",
                            }
                        )
                    if identity_status == "oxidation_identity_mismatch":
                        reasons.append(identity_status)
                        if status == "clean":
                            status = "flagged"
                except (DiagnosticError, OSError, ValueError) as exc:
                    identity_status = "missing"
                    reasons.append(f"hirshfeld_spin_qc:{exc}")
                    status = "missing"
        specialized = {
            "job_id": manifest["job_id"],
            "calculation_key": key,
            "state_role": manifest["state_role"],
            "qc_status": status,
            "qc_reasons": list(dict.fromkeys(reasons)),
            "fragment_connectivity": fragment,
            "oxidation_identity_status": identity_status,
            "parsed": parsed,
        }
        specialized_by_job[manifest["job_id"]] = specialized
        state_payloads.append(specialized)
    write_csv_deterministic(
        ORCA_ROOT / "parsed_states.csv", PARSED_FIELDS, parsed_rows,
        sort_by=("calculation_key", "state_role"),
    )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "shell_distance_qc.csv", DISTANCE_FIELDS, distance_rows,
        sort_by=("calculation_key", "job_id", "shell_index"),
    )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "fragment_spin_qc.csv", SPIN_FIELDS, spin_rows,
        sort_by=("calculation_key", "job_id", "molecule_index"),
    )
    _write_json(
        ORCA_ROOT / "raw_results.json",
        {"workflow_revision": WORKFLOW_REVISION, "states": state_payloads},
    )

    unique_rows: list[dict[str, str]] = []
    for key, system in _system_map().items():
        pair_manifest = [row for row in manifests if row["calculation_key"] == key]
        states = {row["state_role"]: specialized_by_job[row["job_id"]] for row in pair_manifest}
        reasons: list[str] = []
        statuses = [states[role]["qc_status"] for role in ("reduced", "oxidized")]
        pair_status = pair_qc_status(statuses)
        for role in ("reduced", "oxidized"):
            reasons.extend(f"{role}:{reason}" for reason in states[role]["qc_reasons"])
        reduced = states["reduced"]["parsed"]
        oxidized = states["oxidized"]["parsed"]
        reduced_G, oxidized_G = str(reduced["G_composite_1M_Eh"]), str(oxidized["G_composite_1M_Eh"])
        delta_eV = explicit = independent = recompute_delta = None
        if reduced_G and oxidized_G:
            delta_eV, explicit, independent = _explicit_eox_from_raw(
                reduced_G, oxidized_G, conversion
            )
            recompute_delta = abs(explicit - independent)
            if recompute_delta > Decimal("1e-10"):
                reasons.append("independent_energy_recompute_mismatch")
                if pair_status == "clean":
                    pair_status = "flagged"
        reduced_opt = str(reduced["optimized_geometry_sha256"])
        oxidized_opt = str(oxidized["optimized_geometry_sha256"])
        independent_geometries = optimized_geometries_independent(reduced_opt, oxidized_opt)
        if reduced_opt and oxidized_opt and not independent_geometries:
            reasons.append("optimized_state_geometries_not_independent")
            if pair_status == "clean":
                pair_status = "flagged"
        identity = states["oxidized"]["oxidation_identity_status"]
        unique_rows.append(
            {
                "calculation_key": key,
                "property": system["property"],
                "reduced_job_id": next(row["job_id"] for row in pair_manifest if row["state_role"] == "reduced"),
                "oxidized_job_id": next(row["job_id"] for row in pair_manifest if row["state_role"] == "oxidized"),
                "G_reduced_1M_Eh": reduced_G,
                "G_oxidized_1M_Eh": oxidized_G,
                "delta_G_ox_eV": _format_decimal(delta_eV),
                "explicit_R5_DFT_Eox_vs_AgAgCl_V": _format_decimal(explicit),
                "independent_recompute_V": _format_decimal(independent),
                "recompute_abs_delta_V": _format_decimal(recompute_delta),
                "initial_cluster_sha256": clusters[key]["geometry_sha256"],
                "optimized_geometry_sha256_reduced": reduced_opt,
                "optimized_geometry_sha256_oxidized": oxidized_opt,
                "optimized_geometries_independent": str(independent_geometries).lower(),
                "oxidation_identity_status": identity,
                "qc_status": pair_status,
                "qc_reasons": ";".join(dict.fromkeys(reasons)),
            }
        )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "explicit_eox_unique.csv", UNIQUE_FIELDS, unique_rows,
        sort_by=("calculation_key",),
    )

    unique_by_key = {row["calculation_key"]: row for row in unique_rows}
    comparison: list[dict[str, str]] = []
    for benchmark in read_csv_rows(DIAGNOSTIC_ROOT / "benchmark_records.csv"):
        explicit_row = unique_by_key[benchmark["calculation_key"]]
        experimental = Decimal(benchmark["experimental_value"])
        tier1 = Decimal(benchmark["calibrated_tier1"])
        implicit = Decimal(benchmark["tier2_dft"])
        explicit_value = (
            Decimal(explicit_row["explicit_R5_DFT_Eox_vs_AgAgCl_V"])
            if explicit_row["explicit_R5_DFT_Eox_vs_AgAgCl_V"] else None
        )
        tier1_error = tier1 - experimental
        implicit_error = implicit - experimental
        explicit_error = explicit_value - experimental if explicit_value is not None else None
        comparison.append(
            {
                "record_id": benchmark["record_id"],
                "property": explicit_row["property"],
                "species": benchmark["species"],
                "environment": benchmark["environment"],
                "calculation_key": benchmark["calculation_key"],
                "experimental_Eox_V": str(experimental),
                "implicit_calibrated_xTB_Eox_V": str(tier1),
                "implicit_DFT_Eox_V": str(implicit),
                "explicit_R5_DFT_Eox_V": _format_decimal(explicit_value),
                "implicit_xTB_signed_error_V": _format_decimal(tier1_error),
                "implicit_DFT_signed_error_V": _format_decimal(implicit_error),
                "explicit_R5_DFT_signed_error_V": _format_decimal(explicit_error),
                "implicit_xTB_absolute_error_V": _format_decimal(abs(tier1_error)),
                "implicit_DFT_absolute_error_V": _format_decimal(abs(implicit_error)),
                "explicit_R5_DFT_absolute_error_V": _format_decimal(abs(explicit_error) if explicit_error is not None else None),
                "explicit_minus_implicit_DFT_V": _format_decimal(explicit_value - implicit if explicit_value is not None else None),
                "explicit_improvement_over_implicit_DFT_V": _format_decimal(abs(implicit_error) - abs(explicit_error) if explicit_error is not None else None),
                "explicit_improvement_over_implicit_xTB_V": _format_decimal(abs(tier1_error) - abs(explicit_error) if explicit_error is not None else None),
                "calculation_qc": explicit_row["qc_status"],
                "source": benchmark["source"],
                "protocol_or_conditions": benchmark["protocol_or_conditions"],
                "normalization": benchmark["normalization"],
            }
        )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "record_comparison.csv", COMPARISON_FIELDS, comparison,
        sort_by=("calculation_key", "record_id"),
    )
    _, metrics_summary = build_metrics(comparison)
    _write_figures(comparison)
    attempted = sum(
        any((REPOSITORY_ROOT / row["output_path"]).exists() for row in manifests if row["calculation_key"] == key)
        for key in _system_map()
    )
    completed_clean = sum(row["qc_status"] == "clean" for row in unique_rows)
    flagged = [row["calculation_key"] for row in unique_rows if row["qc_status"] == "flagged"]
    missing = [row["calculation_key"] for row in unique_rows if row["qc_status"] == "missing"]
    qc = {
        "status": "PASS" if completed_clean == 6 else "INCOMPLETE",
        "attempted_unique_systems": attempted,
        "completed_clean_unique_systems": completed_clean,
        "flagged_unique_systems": flagged,
        "missing_unique_systems": missing,
        "state_counts": {
            status: sum(payload["qc_status"] == status for payload in state_payloads)
            for status in ("clean", "flagged", "missing")
        },
        "primary_metric_policy": "completed-clean shared subset; anion requires target-dominant oxidized Hirshfeld spin",
        "workflow_revision": WORKFLOW_REVISION,
    }
    _write_json(DIAGNOSTIC_ROOT / "qc.json", qc)
    _write_report(unique_rows, metrics_summary, qc)
    status_path = DIAGNOSTIC_ROOT / "execution_status.json"
    execution = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    execution.update(
        {
            "status": "COMPLETE" if completed_clean == 6 else "INCOMPLETE",
            "attempted_unique_systems": attempted,
            "completed_unique_systems": completed_clean,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write_json(status_path, execution)
    return qc


def _write_readme() -> None:
    text = """# Explicit-R5 DFT Eox benchmark

This isolated diagnostic evaluates adiabatic oxidation potentials for one target plus five explicit solvent molecules, with the same solvent also represented by the frozen project SMD row.

Phase A contains six unique species-medium keys and twelve independent state-specific `wB97X-D3` Opt/Freq calculations. Each Compound deck evaluates its final `def2-TZVPD` SMD single point on the state-specific optimized `Compound_1.xyz`; only the oxidized final step requests Hirshfeld analysis.

The seven input CSVs and all upstream ORCA/XYZ sources are SHA-256 bound. `/Users/shichen/GitHub/20260707` is a read-only source checkout. Packmol uses a deterministic key-derived seed, 2.000 Å tolerance, one fixed central target during packing, and five shell molecules. No atoms are constrained during DFT optimization.

Commands:

```bash
PYTHONPATH=src python diagnostics/explicit_solvation_eox_r5/run_diagnostic.py prepare \\
  --benchmark-dir /path/to/jhtvs_8_validation_plots_csv \\
  --detail-dir /path/to/detail_csvs \\
  --source-checkout /path/to/20260707 \\
  --packmol packmol
PYTHONPATH=src python diagnostics/explicit_solvation_eox_r5/run_diagnostic.py validate-prepared
PYTHONPATH=src python diagnostics/explicit_solvation_eox_r5/run_diagnostic.py collect \\
  --source-checkout /path/to/20260707
```
"""
    _write_generated(DIAGNOSTIC_ROOT / "README.md", text)


def prepare(
    *, benchmark_dir: Path, detail_dir: Path, source_checkout: Path, packmol: str
) -> dict[str, Any]:
    snapshot_inputs(benchmark_dir.resolve(), detail_dir.resolve())
    audit_and_snapshot_sources(source_checkout.resolve())
    build_benchmark_records()
    run_packmol(packmol)
    render_orca_decks()
    report = validate_prepared()
    _write_readme()
    _write_json(
        DIAGNOSTIC_ROOT / "execution_status.json",
        {
            "status": "PREPARED_NOT_SUBMITTED",
            "phase": "A",
            "attempted_unique_systems": 0,
            "completed_unique_systems": 0,
            "first_submission_utc": None,
            "scheduler_job_ids": [],
            "workflow_revision": WORKFLOW_REVISION,
            "task_table_sha256": sha256_file(ORCA_TASKS_PATH),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return report


def record_submission(scheduler_job_id: str, submitted_at_utc: str | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"\d+(?:\.\d+-\d+:\d+)?", scheduler_job_id):
        raise DiagnosticError(f"invalid scheduler job ID: {scheduler_job_id}")
    path = DIAGNOSTIC_ROOT / "execution_status.json"
    if not path.is_file():
        raise DiagnosticError("execution status is missing; run prepare first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp = submitted_at_utc or datetime.now(UTC).isoformat()
    if payload.get("first_submission_utc") is None:
        payload["first_submission_utc"] = timestamp
    identifiers = list(payload.get("scheduler_job_ids", []))
    if scheduler_job_id not in identifiers:
        identifiers.append(scheduler_job_id)
    payload.update(
        {
            "status": "SUBMITTED",
            "phase": "A",
            "attempted_unique_systems": 6,
            "scheduler_job_ids": identifiers,
            "updated_at_utc": timestamp,
        }
    )
    _write_json(path, payload)
    return payload


def record_continuation_submission(
    logical_job_id: str, scheduler_job_id: str, submitted_at_utc: str | None = None
) -> dict[str, str]:
    if not re.fullmatch(r"\d+(?:\.\d+-\d+:\d+)?", scheduler_job_id):
        raise DiagnosticError(f"invalid scheduler job ID: {scheduler_job_id}")
    rows = _continuation_rows()
    matches = [row for row in rows if row["logical_job_id"] == logical_job_id]
    if len(matches) != 1:
        raise DiagnosticError(f"prepared continuation is not unique: {logical_job_id}")
    if matches[0]["scheduler_job_id"]:
        raise DiagnosticError(f"continuation submission already recorded: {logical_job_id}")
    timestamp = submitted_at_utc or datetime.now(UTC).isoformat()
    matches[0].update(
        {
            "status": "submitted",
            "scheduler_job_id": scheduler_job_id,
            "submitted_at_utc": timestamp,
        }
    )
    write_csv_deterministic(
        CONTINUATION_MANIFEST_PATH, CONTINUATION_FIELDS, rows,
        sort_by=("logical_job_id", "attempt"),
    )
    return matches[0]


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--benchmark-dir", type=Path, required=True)
    prepare_parser.add_argument("--detail-dir", type=Path, required=True)
    prepare_parser.add_argument("--source-checkout", type=Path, required=True)
    prepare_parser.add_argument("--packmol", default="packmol")
    commands.add_parser("validate-prepared")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--source-checkout", type=Path, required=True)
    submission_parser = commands.add_parser("record-submission")
    submission_parser.add_argument("--scheduler-job-id", required=True)
    submission_parser.add_argument("--submitted-at-utc")
    continuation_parser = commands.add_parser("prepare-continuation")
    continuation_parser.add_argument("--logical-job-id", required=True)
    continuation_submission_parser = commands.add_parser("record-continuation-submission")
    continuation_submission_parser.add_argument("--logical-job-id", required=True)
    continuation_submission_parser.add_argument("--scheduler-job-id", required=True)
    continuation_submission_parser.add_argument("--submitted-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    if args.command == "prepare":
        payload = prepare(
            benchmark_dir=args.benchmark_dir,
            detail_dir=args.detail_dir,
            source_checkout=args.source_checkout,
            packmol=args.packmol,
        )
    elif args.command == "validate-prepared":
        payload = validate_prepared()
    elif args.command == "collect":
        payload = collect(source_checkout=args.source_checkout)
    elif args.command == "record-submission":
        payload = record_submission(args.scheduler_job_id, args.submitted_at_utc)
    elif args.command == "prepare-continuation":
        payload = prepare_continuation(args.logical_job_id)
    elif args.command == "record-continuation-submission":
        payload = record_continuation_submission(
            args.logical_job_id, args.scheduler_job_id, args.submitted_at_utc
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
