#!/usr/bin/env python3
"""Prepare, execute, parse, and QC the explicit-solvation SPE diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

from jhtvs_ft0806.geometry.xyz import XYZAtom, read_xyz
from jhtvs_ft0806.hpc.submission import TASK_FIELDS
from jhtvs_ft0806.ml.features import PolarMACEBackend
from jhtvs_ft0806.orca.decks import render_sp_deck
from jhtvs_ft0806.orca.parser import HARTREE_TO_EV, parse_job_result
from jhtvs_ft0806.provenance import sha256_bytes, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


DIAGNOSTIC_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = DIAGNOSTIC_ROOT.parents[1]
PROTOCOL_PATH = DIAGNOSTIC_ROOT / "protocol.json"
SOURCE_PROVENANCE_PATH = DIAGNOSTIC_ROOT / "source_provenance.csv"
CLUSTER_MANIFEST_PATH = DIAGNOSTIC_ROOT / "cluster_manifest.csv"
ORCA_MANIFEST_PATH = DIAGNOSTIC_ROOT / "orca" / "job_manifest.csv"
ORCA_TASKS_PATH = DIAGNOSTIC_ROOT / "orca" / "tasks.tsv"
MACE_RESULTS_PATH = DIAGNOSTIC_ROOT / "mace" / "raw_results.json"
WORKFLOW_REVISION = "jhtvs-ft0806-explicit-solvation-sp-v1"
ORCA_METHOD_ID = "T2_wB97M-V_def2-TZVPD_explicit_cluster_gas_SPE_v1"
MACE_METHOD_ID = "MACE_POLAR_1_L_explicit_cluster_SPE_v1"
PACKMOL_SUCCESS = "Success!"
HARTREE_TO_EV_DECIMAL = Decimal(str(HARTREE_TO_EV))

CLUSTER_FIELDS = (
    "system",
    "solute",
    "solvent",
    "n_solvent",
    "solute_atoms",
    "solvent_atoms",
    "total_atoms",
    "box_side_A",
    "box_min_A",
    "box_max_A",
    "tolerance_A",
    "seed",
    "solute_source_sha256",
    "solvent_source_sha256",
    "packmol_input_path",
    "packmol_input_sha256",
    "packmol_log_path",
    "packmol_log_sha256",
    "packmol_executable",
    "packmol_executable_sha256",
    "packmol_version",
    "geometry_path",
    "geometry_sha256",
    "min_intermolecular_distance_A",
    "molecule_count_qc",
    "atom_order_qc",
    "overlap_qc",
    "status",
)

ORCA_FIELDS = (
    "job_id",
    "system",
    "solute",
    "solvent",
    "n_solvent",
    "job_class",
    "state_id",
    "solvent_id",
    "formal_charge",
    "multiplicity",
    "input_path",
    "input_sha256",
    "output_path",
    "geometry_key",
    "geometry_sha256",
    "coordinate_payload_sha256",
    "workflow_revision",
    "method_id",
    "functional",
    "basis",
    "nprocs",
    "maxcore_mb_per_rank",
    "planning_core_h",
    "status",
)

COMPARISON_FIELDS = (
    "system",
    "solute",
    "solvent",
    "n_solvent",
    "geometry_sha256",
    "method",
    "charge",
    "multiplicity",
    "energy_raw",
    "energy_unit",
    "energy_eV",
    "normal_termination",
    "deltaE_vertical_eV",
)

SUMMARY_FIELDS = (
    "system",
    "deltaE_ORCA_R5_eV",
    "deltaE_MACE_R5_eV",
    "MACE_minus_ORCA_R5_eV",
    "deltaE_MACE_R50_eV",
    "MACE_R50_minus_R5_eV",
)


class DiagnosticError(RuntimeError):
    """Raised when a fixed protocol or QC invariant is violated."""


def _protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != WORKFLOW_REVISION:
        raise DiagnosticError("protocol schema/revision mismatch")
    return payload


def _repository_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _diagnostic_relative(path: Path) -> str:
    return path.resolve().relative_to(DIAGNOSTIC_ROOT.resolve()).as_posix()


def _write_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != text:
            raise DiagnosticError(f"immutable diagnostic file differs: {path}")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_generated_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _source_rows() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(SOURCE_PROVENANCE_PATH)
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise DiagnosticError("duplicate source geometry name")
    return by_name


def _snapshot_path(row: Mapping[str, str]) -> Path:
    return REPOSITORY_ROOT / row["snapshot_path"]


def _same_atoms(first: Iterable[XYZAtom], second: Iterable[XYZAtom]) -> bool:
    return tuple(first) == tuple(second)


def audit_sources(source_root: Path | None) -> dict[str, Any]:
    """Validate snapshots and, when mounted, the read-only 20260707 provenance."""

    checks: list[dict[str, Any]] = []
    for row in _source_rows().values():
        snapshot = _snapshot_path(row)
        snapshot_atoms = read_xyz(snapshot)
        item: dict[str, Any] = {
            "name": row["name"],
            "snapshot_path": _repository_relative(snapshot),
            "snapshot_sha256": sha256_file(snapshot),
            "atom_count": len(snapshot_atoms),
            "source_checked": source_root is not None,
        }
        if source_root is not None:
            paths = {
                "xyz": source_root / row["source_path"],
                "orca_input": source_root / row["source_orca_input_path"],
                "orca_output": source_root / row["source_orca_output_path"],
            }
            expected = {
                "xyz": row["source_xyz_sha256"],
                "orca_input": row["source_orca_input_sha256"],
                "orca_output": row["source_orca_output_sha256"],
            }
            for kind, path in paths.items():
                if not path.is_file() or path.is_symlink():
                    raise DiagnosticError(f"missing or unsafe upstream {kind}: {path}")
                if sha256_file(path) != expected[kind]:
                    raise DiagnosticError(f"upstream {kind} hash drift: {path}")
            output_text = paths["orca_output"].read_text(
                encoding="utf-8", errors="replace"
            )
            if (
                "THE OPTIMIZATION HAS CONVERGED" not in output_text
                or "ORCA TERMINATED NORMALLY" not in output_text
            ):
                raise DiagnosticError(f"upstream ORCA QC failed: {paths['orca_output']}")
            if not _same_atoms(snapshot_atoms, read_xyz(paths["xyz"])):
                raise DiagnosticError(f"snapshot coordinates differ: {row['name']}")
            item.update(
                {
                    "source_xyz_sha256": expected["xyz"],
                    "source_orca_input_sha256": expected["orca_input"],
                    "source_orca_output_sha256": expected["orca_output"],
                    "source_qc": "PASS",
                }
            )
        checks.append(item)
    report = {
        "status": "PASS",
        "workflow_revision": WORKFLOW_REVISION,
        "source_root": str(source_root.resolve()) if source_root else "not_mounted",
        "sources": sorted(checks, key=lambda item: item["name"]),
    }
    _write_json(DIAGNOSTIC_ROOT / "source_audit.json", report)
    return report


def _cluster_label(n_solvent: int) -> str:
    return f"R{n_solvent}"


def _cluster_path(system: str, n_solvent: int) -> Path:
    return DIAGNOSTIC_ROOT / "clusters" / f"{system}_{_cluster_label(n_solvent)}.xyz"


def _packmol_input_path(system: str, n_solvent: int) -> Path:
    return DIAGNOSTIC_ROOT / "packmol" / f"{system}_{_cluster_label(n_solvent)}.inp"


def _packmol_log_path(system: str, n_solvent: int) -> Path:
    return DIAGNOSTIC_ROOT / "packmol" / f"{system}_{_cluster_label(n_solvent)}.log"


def _render_packmol_input(
    *,
    solute: str,
    solvent: str,
    system: str,
    n_solvent: int,
    box_side: float,
    tolerance: float,
    seed: int,
) -> str:
    half = box_side / 2.0
    output = _diagnostic_relative(_cluster_path(system, n_solvent))
    return (
        f"tolerance {tolerance:.3f}\n"
        "filetype xyz\n"
        f"output {output}\n"
        f"seed {seed}\n\n"
        f"structure source_geometries/{solute}.xyz\n"
        "  number 1\n"
        "  fixed 0.000 0.000 0.000 0.000 0.000 0.000\n"
        "end structure\n\n"
        f"structure source_geometries/{solvent}.xyz\n"
        f"  number {n_solvent}\n"
        f"  inside box {-half:.3f} {-half:.3f} {-half:.3f} "
        f"{half:.3f} {half:.3f} {half:.3f}\n"
        "end structure\n"
    )


def render_packmol_inputs() -> list[Path]:
    protocol = _protocol()
    tolerance = float(protocol["packmol"]["tolerance_angstrom"])
    paths: list[Path] = []
    for system in protocol["systems"]:
        for cluster in system["clusters"]:
            path = _packmol_input_path(system["system"], cluster["n_solvent"])
            _write_exact(
                path,
                _render_packmol_input(
                    solute=system["solute"],
                    solvent=system["solvent"],
                    system=system["system"],
                    n_solvent=int(cluster["n_solvent"]),
                    box_side=float(cluster["box_side_A"]),
                    tolerance=tolerance,
                    seed=int(cluster["seed"]),
                ),
            )
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
    if not path.is_file():
        raise DiagnosticError(f"unsafe executable: {path}")
    return path


def run_packmol(executable: str) -> None:
    binary = _resolve_executable(executable)
    for input_path in render_packmol_inputs():
        stem = input_path.stem
        system, label = stem.rsplit("_R", maxsplit=1)
        n_solvent = int(label)
        cluster_path = _cluster_path(system, n_solvent)
        log_path = _packmol_log_path(system, n_solvent)
        if cluster_path.exists() or log_path.exists():
            if cluster_path.is_file() and log_path.is_file():
                continue
            raise DiagnosticError(f"partial existing Packmol result for {stem}")
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("rb") as source, log_path.open("wb") as log:
            completed = subprocess.run(
                [str(binary)],
                cwd=DIAGNOSTIC_ROOT,
                stdin=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not cluster_path.is_file():
            raise DiagnosticError(
                f"Packmol failed for {stem}; inspect {_diagnostic_relative(log_path)}"
            )
        if PACKMOL_SUCCESS not in log_path.read_text(encoding="utf-8", errors="replace"):
            raise DiagnosticError(f"Packmol success marker missing for {stem}")
    build_cluster_manifest(binary)


def _molecule_ranges(
    *, solute_atoms: int, solvent_atoms: int, n_solvent: int
) -> tuple[range, ...]:
    ranges: list[range] = [range(0, solute_atoms)]
    offset = solute_atoms
    for _ in range(n_solvent):
        ranges.append(range(offset, offset + solvent_atoms))
        offset += solvent_atoms
    return tuple(ranges)


def _minimum_intermolecular_distance(
    atoms: tuple[XYZAtom, ...], molecule_ranges: tuple[range, ...]
) -> float:
    minimum = math.inf
    for first_index, first in enumerate(molecule_ranges):
        for second in molecule_ranges[first_index + 1 :]:
            for atom_index in first:
                for other_index in second:
                    distance = math.dist(
                        atoms[atom_index].coordinates,
                        atoms[other_index].coordinates,
                    )
                    minimum = min(minimum, distance)
    return minimum


def _packmol_version(log_text: str) -> str:
    for pattern in (
        r"Packmol version\s+([^\s]+)",
        r"PACKMOL\s+-\s+Version\s+([^\s]+)",
        r"Version\s+([^\s]+)",
    ):
        match = re.search(pattern, log_text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return "unreported"


def build_cluster_manifest(packmol_binary: Path) -> list[dict[str, object]]:
    protocol = _protocol()
    tolerance = float(protocol["packmol"]["tolerance_angstrom"])
    sources = _source_rows()
    rows: list[dict[str, object]] = []
    for system in protocol["systems"]:
        solute_path = _snapshot_path(sources[system["solute"]])
        solvent_path = _snapshot_path(sources[system["solvent"]])
        solute_atoms = read_xyz(solute_path)
        solvent_atoms = read_xyz(solvent_path)
        for cluster in system["clusters"]:
            n_solvent = int(cluster["n_solvent"])
            input_path = _packmol_input_path(system["system"], n_solvent)
            log_path = _packmol_log_path(system["system"], n_solvent)
            geometry_path = _cluster_path(system["system"], n_solvent)
            for required in (input_path, log_path, geometry_path):
                if not required.is_file() or required.is_symlink():
                    raise DiagnosticError(f"missing or unsafe Packmol artifact: {required}")
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if PACKMOL_SUCCESS not in log_text:
                raise DiagnosticError(f"Packmol did not report success: {log_path}")
            atoms = read_xyz(geometry_path)
            expected_symbols = [atom.symbol for atom in solute_atoms] + [
                atom.symbol
                for _ in range(n_solvent)
                for atom in solvent_atoms
            ]
            observed_symbols = [atom.symbol for atom in atoms]
            expected_count = len(expected_symbols)
            count_ok = len(atoms) == expected_count
            order_ok = observed_symbols == expected_symbols
            ranges = _molecule_ranges(
                solute_atoms=len(solute_atoms),
                solvent_atoms=len(solvent_atoms),
                n_solvent=n_solvent,
            )
            minimum = _minimum_intermolecular_distance(atoms, ranges)
            overlap_ok = minimum >= tolerance - 0.01
            if not (count_ok and order_ok and overlap_ok):
                raise DiagnosticError(
                    f"cluster QC failed for {system['system']} R{n_solvent}: "
                    f"count={count_ok}, order={order_ok}, min_distance={minimum:.6f}"
                )
            half = float(cluster["box_side_A"]) / 2.0
            rows.append(
                {
                    "system": system["system"],
                    "solute": system["solute"],
                    "solvent": system["solvent"],
                    "n_solvent": n_solvent,
                    "solute_atoms": len(solute_atoms),
                    "solvent_atoms": len(solvent_atoms),
                    "total_atoms": len(atoms),
                    "box_side_A": format(float(cluster["box_side_A"]), ".3f"),
                    "box_min_A": format(-half, ".3f"),
                    "box_max_A": format(half, ".3f"),
                    "tolerance_A": format(tolerance, ".3f"),
                    "seed": cluster["seed"],
                    "solute_source_sha256": sources[system["solute"]][
                        "source_xyz_sha256"
                    ],
                    "solvent_source_sha256": sources[system["solvent"]][
                        "source_xyz_sha256"
                    ],
                    "packmol_input_path": _repository_relative(input_path),
                    "packmol_input_sha256": sha256_file(input_path),
                    "packmol_log_path": _repository_relative(log_path),
                    "packmol_log_sha256": sha256_file(log_path),
                    "packmol_executable": str(packmol_binary),
                    "packmol_executable_sha256": sha256_file(packmol_binary),
                    "packmol_version": _packmol_version(log_text),
                    "geometry_path": _repository_relative(geometry_path),
                    "geometry_sha256": sha256_file(geometry_path),
                    "min_intermolecular_distance_A": format(minimum, ".10f"),
                    "molecule_count_qc": "pass" if count_ok else "fail",
                    "atom_order_qc": "pass" if order_ok else "fail",
                    "overlap_qc": "pass" if overlap_ok else "fail",
                    "status": "clean" if count_ok and order_ok and overlap_ok else "failed",
                }
            )
    write_csv_deterministic(
        CLUSTER_MANIFEST_PATH,
        CLUSTER_FIELDS,
        rows,
        sort_by=("system", "n_solvent"),
    )
    return rows


def _coordinate_payload(deck_text: str) -> str:
    lines = deck_text.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("* xyz ")),
        None,
    )
    if start is None:
        raise DiagnosticError("ORCA deck lacks xyz block")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip() == "*"),
        None,
    )
    if end is None:
        raise DiagnosticError("ORCA deck has unterminated xyz block")
    return "".join(lines[start + 1 : end])


def render_orca_decks() -> list[dict[str, object]]:
    protocol = _protocol()
    orca = protocol["orca"]
    clusters = {
        (row["system"], int(row["n_solvent"])): row
        for row in read_csv_rows(CLUSTER_MANIFEST_PATH)
    }
    rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for array_task, system in enumerate(protocol["systems"], start=1):
        cluster = clusters.get((system["system"], 5))
        if cluster is None or cluster["status"] != "clean":
            raise DiagnosticError(f"clean R5 cluster missing: {system['system']}")
        xyz_path = REPOSITORY_ROOT / cluster["geometry_path"]
        geometry = {
            "geometry_key": f"explicit_solvation:{system['system']}:R5",
            "xyz_sha256": cluster["geometry_sha256"],
        }
        payload_hashes: list[str] = []
        for sequence, state in enumerate(protocol["states"], start=1):
            charge = int(state["formal_charge"])
            multiplicity = int(state["multiplicity"])
            suffix = "Q0" if charge == 0 else "QP1"
            job_id = f"EXSOLV{array_task:02d}_{suffix}"
            job = {
                "workflow_revision": WORKFLOW_REVISION,
                "job_id": job_id,
                "job_class": "diagnostic_gas_sp",
                "state_id": f"{system['system']}_R5_{suffix}_M{multiplicity}",
                "solvent_id": f"EXPLICIT_{system['solvent'].upper()}_R5",
                "formal_charge": str(charge),
                "multiplicity": str(multiplicity),
                "method_id": ORCA_METHOD_ID,
                "functional": orca["functional"],
                "basis": orca["basis"],
                "nprocs": str(orca["nprocs"]),
                "maxcore_mb_per_rank": str(orca["maxcore_mb_per_rank"]),
            }
            deck_text = render_sp_deck(
                job,
                geometry,
                xyz_path,
                None,
                registry_sha256="not_applicable_explicit_cluster",
            )
            payload_sha = sha256_bytes(_coordinate_payload(deck_text).encode("utf-8"))
            payload_hashes.append(payload_sha)
            input_path = DIAGNOSTIC_ROOT / "orca" / "jobs" / job_id / f"{job_id}.inp"
            _write_exact(input_path, deck_text)
            row = {
                "job_id": job_id,
                "system": system["system"],
                "solute": system["solute"],
                "solvent": system["solvent"],
                "n_solvent": 5,
                "job_class": job["job_class"],
                "state_id": job["state_id"],
                "solvent_id": job["solvent_id"],
                "formal_charge": charge,
                "multiplicity": multiplicity,
                "input_path": _repository_relative(input_path),
                "input_sha256": sha256_file(input_path),
                "output_path": _repository_relative(input_path.with_suffix(".out")),
                "geometry_key": geometry["geometry_key"],
                "geometry_sha256": geometry["xyz_sha256"],
                "coordinate_payload_sha256": payload_sha,
                "workflow_revision": WORKFLOW_REVISION,
                "method_id": ORCA_METHOD_ID,
                "functional": orca["functional"],
                "basis": orca["basis"],
                "nprocs": orca["nprocs"],
                "maxcore_mb_per_rank": orca["maxcore_mb_per_rank"],
                "planning_core_h": "32",
                "status": "ready",
            }
            rows.append(row)
            task_rows.append(
                {
                    "array_task": array_task,
                    "sequence": sequence,
                    "job_id": job_id,
                    "job_class": job["job_class"],
                    "input_path": row["input_path"],
                    "input_sha256": row["input_sha256"],
                    "output_path": row["output_path"],
                    "nprocs": orca["nprocs"],
                    "planning_core_h": row["planning_core_h"],
                    "workflow_revision": WORKFLOW_REVISION,
                    "method_id": ORCA_METHOD_ID,
                }
            )
        if len(set(payload_hashes)) != 1:
            raise DiagnosticError(
                f"charge-state coordinate payload differs for {system['system']}"
            )
    write_csv_deterministic(
        ORCA_MANIFEST_PATH,
        ORCA_FIELDS,
        rows,
        sort_by=("system", "formal_charge"),
    )
    task_text = "\t".join(TASK_FIELDS) + "\n" + "".join(
        "\t".join(str(row[field]) for field in TASK_FIELDS) + "\n"
        for row in task_rows
    )
    _write_exact(ORCA_TASKS_PATH, task_text)
    return rows


def run_mace(*, checkpoint: str, device: str) -> dict[str, Any]:
    protocol = _protocol()
    if checkpoint != protocol["mace"]["checkpoint"]:
        raise DiagnosticError("checkpoint differs from the fixed protocol")
    clusters = read_csv_rows(CLUSTER_MANIFEST_PATH)
    backend = PolarMACEBackend(checkpoint=checkpoint, device=device)
    if backend.provenance.default_dtype != "float64":
        raise DiagnosticError("MACE backend is not float64")
    if (
        backend.provenance.checkpoint_sha256
        != protocol["mace"]["checkpoint_sha256"]
    ):
        raise DiagnosticError("MACE checkpoint hash differs from protocol")
    results: list[dict[str, Any]] = []
    for cluster in clusters:
        if cluster["status"] != "clean":
            raise DiagnosticError(f"unclean cluster cannot enter MACE: {cluster['system']}")
        xyz_path = REPOSITORY_ROOT / cluster["geometry_path"]
        for state in protocol["states"]:
            charge = int(state["formal_charge"])
            multiplicity = int(state["multiplicity"])
            record = backend.extract(
                xyz_path=xyz_path,
                formal_charge=charge,
                multiplicity=multiplicity,
            )
            results.append(
                {
                    "system": cluster["system"],
                    "solute": cluster["solute"],
                    "solvent": cluster["solvent"],
                    "n_solvent": int(cluster["n_solvent"]),
                    "geometry_path": cluster["geometry_path"],
                    "geometry_sha256": cluster["geometry_sha256"],
                    "formal_charge": charge,
                    "multiplicity": multiplicity,
                    "energy_raw": record.base_energy_eV,
                    "energy_unit": "eV",
                    "output_shapes": {
                        key: list(value)
                        for key, value in sorted(record.output_shapes.items())
                    },
                    "status": "clean",
                }
            )
    checkpoint_provenance = asdict(backend.provenance)
    checkpoint_provenance["checkpoint_path"] = Path(
        checkpoint_provenance["checkpoint_path"]
    ).name
    payload = {
        "status": "PASS",
        "workflow_revision": WORKFLOW_REVISION,
        "method_id": MACE_METHOD_ID,
        "checkpoint": checkpoint_provenance,
        "execution": {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "device": device,
            "scheduler_job_id": os.environ.get("JOB_ID", "interactive"),
        },
        "results": sorted(
            results,
            key=lambda row: (
                row["system"], row["n_solvent"], row["formal_charge"]
            ),
        ),
    }
    _write_json(MACE_RESULTS_PATH, payload)
    return payload


def _parse_orca_results() -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for manifest in read_csv_rows(ORCA_MANIFEST_PATH):
        job = {
            key: manifest[key]
            for key in (
                "job_id",
                "state_id",
                "solvent_id",
                "workflow_revision",
                "method_id",
                "formal_charge",
                "multiplicity",
            )
        }
        geometry = {
            "geometry_key": manifest["geometry_key"],
            "xyz_sha256": manifest["geometry_sha256"],
        }
        result = parse_job_result(
            manifest=manifest,
            job=job,
            geometry=geometry,
            solvent=None,
            repository_root=REPOSITORY_ROOT,
        )
        result.update(
            {
                "system": manifest["system"],
                "solute": manifest["solute"],
                "solvent": manifest["solvent"],
                "n_solvent": int(manifest["n_solvent"]),
                "formal_charge": int(manifest["formal_charge"]),
                "multiplicity": int(manifest["multiplicity"]),
            }
        )
        if result["qc_status"] != "clean" or not result["final_energy_Eh"]:
            raise DiagnosticError(
                f"ORCA QC failed for {manifest['job_id']}: {result['qc_reasons']}"
            )
        parsed.append(result)
    payload = {
        "status": "PASS",
        "workflow_revision": WORKFLOW_REVISION,
        "method_id": ORCA_METHOD_ID,
        "results": parsed,
    }
    _write_json(DIAGNOSTIC_ROOT / "orca" / "raw_results.json", payload)
    fields = (
        "job_id",
        "system",
        "solute",
        "solvent",
        "n_solvent",
        "formal_charge",
        "multiplicity",
        "geometry_sha256",
        "input_sha256",
        "output_path",
        "output_sha256",
        "orca_version",
        "final_energy_Eh",
        "normal_termination",
        "orca_error",
        "qc_status",
        "qc_reasons",
    )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "orca" / "parsed_results.csv",
        fields,
        ({field: row.get(field, "") for field in fields} for row in parsed),
        sort_by=("system", "formal_charge"),
    )
    return parsed


def _gap(energies: Mapping[int, Decimal]) -> Decimal:
    if set(energies) != {0, 1}:
        raise DiagnosticError(f"charge-state energy pair incomplete: {sorted(energies)}")
    return energies[1] - energies[0]


def collect_results() -> dict[str, Any]:
    orca_results = _parse_orca_results()
    mace_payload = json.loads(
        MACE_RESULTS_PATH.read_text(encoding="utf-8"), parse_float=Decimal
    )
    if mace_payload.get("status") != "PASS":
        raise DiagnosticError("MACE result status is not PASS")

    comparison: list[dict[str, Any]] = []
    orca_groups: dict[str, dict[int, Decimal]] = {}
    for row in orca_results:
        energy_eh = Decimal(row["final_energy_Eh"])
        energy_ev = energy_eh * HARTREE_TO_EV_DECIMAL
        orca_groups.setdefault(row["system"], {})[int(row["formal_charge"])] = energy_ev
        comparison.append(
            {
                "system": row["system"],
                "solute": row["solute"],
                "solvent": row["solvent"],
                "n_solvent": row["n_solvent"],
                "geometry_sha256": row["geometry_sha256"],
                "method": "ORCA_6.1_wB97M-V_def2-TZVPD",
                "charge": row["formal_charge"],
                "multiplicity": row["multiplicity"],
                "energy_raw": format(energy_eh, ".15g"),
                "energy_unit": "Hartree",
                "energy_eV": format(energy_ev, ".15g"),
                "normal_termination": row["normal_termination"],
                "deltaE_vertical_eV": "",
            }
        )
    mace_groups: dict[tuple[str, int], dict[int, Decimal]] = {}
    for row in mace_payload["results"]:
        key = (row["system"], int(row["n_solvent"]))
        energy_ev = Decimal(row["energy_raw"])
        mace_groups.setdefault(key, {})[int(row["formal_charge"])] = energy_ev
        comparison.append(
            {
                "system": row["system"],
                "solute": row["solute"],
                "solvent": row["solvent"],
                "n_solvent": row["n_solvent"],
                "geometry_sha256": row["geometry_sha256"],
                "method": "MACE-POLAR-1-L",
                "charge": row["formal_charge"],
                "multiplicity": row["multiplicity"],
                "energy_raw": format(energy_ev, ".15g"),
                "energy_unit": "eV",
                "energy_eV": format(energy_ev, ".15g"),
                "normal_termination": "true",
                "deltaE_vertical_eV": "",
            }
        )
    gaps: dict[tuple[str, str, int], Decimal] = {}
    for system, values in orca_groups.items():
        gaps[(system, "ORCA_6.1_wB97M-V_def2-TZVPD", 5)] = _gap(values)
    for (system, n_solvent), values in mace_groups.items():
        gaps[(system, "MACE-POLAR-1-L", n_solvent)] = _gap(values)
    for row in comparison:
        row["deltaE_vertical_eV"] = format(
            gaps[(row["system"], row["method"], int(row["n_solvent"]))], ".12f"
        )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "comparison.csv",
        COMPARISON_FIELDS,
        comparison,
        sort_by=("system", "n_solvent", "method", "charge"),
    )

    summary: list[dict[str, Any]] = []
    for system in sorted(orca_groups):
        orca_r5 = gaps[(system, "ORCA_6.1_wB97M-V_def2-TZVPD", 5)]
        mace_r5 = gaps[(system, "MACE-POLAR-1-L", 5)]
        mace_r50 = gaps[(system, "MACE-POLAR-1-L", 50)]
        summary.append(
            {
                "system": system,
                "deltaE_ORCA_R5_eV": format(orca_r5, ".12f"),
                "deltaE_MACE_R5_eV": format(mace_r5, ".12f"),
                "MACE_minus_ORCA_R5_eV": format(mace_r5 - orca_r5, ".12f"),
                "deltaE_MACE_R50_eV": format(mace_r50, ".12f"),
                "MACE_R50_minus_R5_eV": format(mace_r50 - mace_r5, ".12f"),
            }
        )
    write_csv_deterministic(
        DIAGNOSTIC_ROOT / "summary.csv",
        SUMMARY_FIELDS,
        summary,
        sort_by=("system",),
    )
    qc = validate_complete(summary=summary, comparison=comparison)
    _write_report(summary, qc)
    return {"status": "PASS", "summary": summary, "qc": qc}


def validate_prepared() -> dict[str, Any]:
    protocol = _protocol()
    clusters = read_csv_rows(CLUSTER_MANIFEST_PATH)
    jobs = read_csv_rows(ORCA_MANIFEST_PATH)
    checks: dict[str, Any] = {
        "cluster_count": len(clusters),
        "orca_job_count": len(jobs),
        "cluster_statuses": sorted({row["status"] for row in clusters}),
    }
    issues: list[str] = []
    if len(clusters) != 4 or any(row["status"] != "clean" for row in clusters):
        issues.append("cluster manifest is not four clean R5/R50 clusters")
    if len(jobs) != 4:
        issues.append("ORCA manifest is not four charge-state jobs")
    for system in protocol["systems"]:
        system_jobs = [row for row in jobs if row["system"] == system["system"]]
        if {row["formal_charge"] for row in system_jobs} != {"0", "1"}:
            issues.append(f"charge-state pair incomplete: {system['system']}")
            continue
        if len({row["geometry_sha256"] for row in system_jobs}) != 1:
            issues.append(f"geometry hash differs across charge states: {system['system']}")
        if len({row["coordinate_payload_sha256"] for row in system_jobs}) != 1:
            issues.append(f"coordinate bytes differ across charge states: {system['system']}")
        for row in system_jobs:
            text = (REPOSITORY_ROOT / row["input_path"]).read_text(encoding="utf-8")
            required = (
                "! wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3\n",
                "%pal nprocs 8 end\n",
                "%maxcore 3000\n",
            )
            if not all(marker in text for marker in required):
                issues.append(f"ORCA method/resource marker missing: {row['job_id']}")
            lower = text.lower()
            method_lines = [line.lower().split() for line in text.splitlines() if line.startswith("!")]
            contains_continuum = any(
                marker in lower
                for marker in ("%cpcm", "smdsolvent", "! smd(", "! cpcm(")
            )
            contains_non_spe_task = any(
                token in {"opt", "freq"}
                for tokens in method_lines
                for token in tokens
            )
            if contains_continuum or contains_non_spe_task:
                issues.append(f"ORCA deck is not gas SPE: {row['job_id']}")
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


def validate_complete(
    *, summary: list[dict[str, Any]], comparison: list[dict[str, Any]]
) -> dict[str, Any]:
    prepared = validate_prepared()
    issues: list[str] = []
    mace_payload = json.loads(MACE_RESULTS_PATH.read_text(encoding="utf-8"))
    protocol = _protocol()
    checkpoint = mace_payload.get("checkpoint", {})
    if checkpoint.get("checkpoint_sha256") != protocol["mace"]["checkpoint_sha256"]:
        issues.append("MACE checkpoint SHA-256 mismatch")
    if checkpoint.get("default_dtype") != "float64":
        issues.append("MACE dtype is not float64")
    expected_states = {
        (system["system"], size, state["formal_charge"], state["multiplicity"])
        for system in protocol["systems"]
        for size in (5, 50)
        for state in protocol["states"]
    }
    observed_states = {
        (
            row["system"],
            int(row["n_solvent"]),
            int(row["formal_charge"]),
            int(row["multiplicity"]),
        )
        for row in mace_payload.get("results", [])
    }
    if observed_states != expected_states:
        issues.append("MACE charge/multiplicity/cluster matrix mismatch")
    if len(summary) != 2 or len(comparison) != 12:
        issues.append("comparison row count mismatch")
    for row in summary:
        orca = Decimal(row["deltaE_ORCA_R5_eV"])
        mace5 = Decimal(row["deltaE_MACE_R5_eV"])
        mace50 = Decimal(row["deltaE_MACE_R50_eV"])
        if abs(Decimal(row["MACE_minus_ORCA_R5_eV"]) - (mace5 - orca)) > Decimal(
            "2e-12"
        ):
            issues.append(f"matched-model arithmetic mismatch: {row['system']}")
        if abs(Decimal(row["MACE_R50_minus_R5_eV"]) - (mace50 - mace5)) > Decimal(
            "2e-12"
        ):
            issues.append(f"size-shift arithmetic mismatch: {row['system']}")
    report = {
        "status": "PASS" if not issues else "FAIL",
        "workflow_revision": WORKFLOW_REVISION,
        "prepared_qc": prepared["status"],
        "checks": {
            "summary_rows": len(summary),
            "comparison_rows": len(comparison),
            "mace_states": len(observed_states),
            "orca_normal_terminations": sum(
                1
                for row in comparison
                if row["method"].startswith("ORCA_")
                and row["normal_termination"] == "true"
            ),
        },
        "issues": issues,
    }
    _write_json(DIAGNOSTIC_ROOT / "qc.json", report)
    if issues:
        raise DiagnosticError(f"complete QC failed: {issues}")
    return report


def _write_report(summary: list[dict[str, Any]], qc: Mapping[str, Any]) -> None:
    lines = [
        "# Explicit-solvation vertical oxidation SPE diagnostic",
        "",
        f"QC status: **{qc['status']}**",
        "",
        "| system | ORCA R5 ΔE (eV) | MACE R5 ΔE (eV) | MACE−ORCA R5 (eV) | MACE R50 ΔE (eV) | MACE R50−R5 (eV) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['system']} | {row['deltaE_ORCA_R5_eV']} | "
            f"{row['deltaE_MACE_R5_eV']} | {row['MACE_minus_ORCA_R5_eV']} | "
            f"{row['deltaE_MACE_R50_eV']} | {row['MACE_R50_minus_R5_eV']} |"
        )
    lines.extend(
        [
            "",
            "All values are fixed-coordinate electronic energy differences E(q=+1,R) − E(q=0,R).",
            "No frequency, Gibbs free energy, continuum model, reference-electrode conversion, or geometry optimization is included.",
            "",
        ]
    )
    _write_generated_text(DIAGNOSTIC_ROOT / "REPORT.md", "\n".join(lines))


def prepare(*, source_root: Path | None, packmol: str) -> dict[str, Any]:
    audit_sources(source_root)
    render_packmol_inputs()
    run_packmol(packmol)
    render_orca_decks()
    return validate_prepared()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-packmol")
    render.add_argument("--source-root", type=Path)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path)
    prepare_parser.add_argument("--packmol", default="packmol")
    mace = subparsers.add_parser("mace")
    mace.add_argument("--checkpoint", default="polar-1-l")
    mace.add_argument("--device", default="cpu")
    subparsers.add_parser("validate-prepared")
    subparsers.add_parser("collect")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "render-packmol":
        audit_sources(args.source_root)
        paths = render_packmol_inputs()
        payload: Any = {"status": "PASS", "inputs": [str(path) for path in paths]}
    elif args.command == "prepare":
        payload = prepare(source_root=args.source_root, packmol=args.packmol)
    elif args.command == "mace":
        payload = run_mace(checkpoint=args.checkpoint, device=args.device)
    elif args.command == "validate-prepared":
        payload = validate_prepared()
    elif args.command == "collect":
        payload = collect_results()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
