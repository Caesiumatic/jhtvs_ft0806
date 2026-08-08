"""Resolve manifest geometry keys to hashed same-run XYZ files."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem

from jhtvs_ft0806.geometry.sigma import (
    N_CONFORMERS,
    SigmaTopology,
    build_sigma_complex,
    load_sigma_topologies,
)
from jhtvs_ft0806.geometry.topology import build_repeat_chain, molecule_from_smiles
from jhtvs_ft0806.geometry.xyz import check_connectivity, read_xyz
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic

SIGMA_PREOPT_METHOD_ID = "GFN2-xTB_default_opt_ddCOSMO_v1"
SIGMA_PREOPT_RUN_ID = "jhtvs-ft0806-sigma-preopt-v1"

GEOMETRY_INDEX_FIELDS = (
    "geometry_key",
    "state_id",
    "solvent_id",
    "formal_charge",
    "multiplicity",
    "status",
    "xyz_path",
    "xyz_sha256",
    "source_kind",
    "source_run_id",
    "source_task_id",
    "source_xyz_sha256",
    "source_output_sha256",
    "source_qc_status",
    "connectivity_status",
    "bonds_broken",
    "bonds_formed",
    "monomer_source_sha256",
    "topology_sha256",
    "preopt_method_id",
    "preopt_epsilon",
    "reason",
)

SIGMA_PREOPT_MANIFEST_FIELDS = (
    "task_id",
    "geometry_key",
    "state_id",
    "parent_id",
    "solvent_id",
    "solvent_name",
    "source_xyz",
    "source_xyz_sha256",
    "output_dir",
    "formal_charge",
    "multiplicity",
    "uhf",
    "epsilon",
    "preopt_method_id",
    "monomer_source_sha256",
    "topology_sha256",
    "xtb_command",
)


class GeometryResolutionError(ValueError):
    """Raised when source provenance or geometry identity is inconsistent."""


@dataclass(frozen=True, slots=True)
class GeometryRequest:
    geometry_key: str
    state_id: str
    solvent_id: str
    formal_charge: int
    multiplicity: int
    geometry_source: str
    required_by_job_ids: tuple[str, ...]

    @property
    def is_sigma(self) -> bool:
        return self.geometry_key.startswith("surrogate:sigma_preopt:")


@dataclass(frozen=True, slots=True)
class GeometryResolutionSummary:
    total: int
    resolved: int
    pending: int
    failed: int
    tier1_resolved: int
    sigma_resolved: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": "PASS" if self.failed == 0 and self.pending == 0 else "INCOMPLETE",
            "total": self.total,
            "resolved": self.resolved,
            "pending": self.pending,
            "failed": self.failed,
            "tier1_resolved": self.tier1_resolved,
            "sigma_resolved": self.sigma_resolved,
        }


def geometry_requests(
    spec_dir: Path, *, include_fullspace_inference: bool = False
) -> list[GeometryRequest]:
    grouped: dict[str, dict[str, object]] = {}

    def add(
        *,
        geometry_key: str,
        state_id: str,
        solvent_id: str,
        formal_charge: int,
        multiplicity: int,
        geometry_source: str,
        required_by: str,
    ) -> None:
        identity = (
            state_id,
            solvent_id,
            formal_charge,
            multiplicity,
            geometry_source,
        )
        current = grouped.get(geometry_key)
        if current is None:
            grouped[geometry_key] = {"identity": identity, "job_ids": [required_by]}
        else:
            if current["identity"] != identity:
                raise GeometryResolutionError(
                    f"geometry key {geometry_key} has inconsistent identity"
                )
            current["job_ids"].append(required_by)  # type: ignore[union-attr]

    if include_fullspace_inference:
        states = {
            row["state_id"]: row
            for row in read_csv_rows(spec_dir / "fullspace_state_registry.csv")
        }
        solvent_ids = tuple(
            row["solvent_id"]
            for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
        )
        for reaction in read_csv_rows(spec_dir / "fullspace_reaction_registry.csv"):
            policy = reaction["solvent_policy"]
            if policy == "all project media at inference":
                assigned = solvent_ids
            elif policy.startswith("self only (") and policy.endswith(")"):
                assigned = (policy.removeprefix("self only (").removesuffix(")"),)
            else:
                raise GeometryResolutionError(
                    f"{reaction['reaction_id']}: unsupported inference solvent policy {policy}"
                )
            for state_id in reaction["stoichiometry"].split(";"):
                state_id = state_id.rsplit(":", 1)[0]
                state = states.get(state_id)
                if state is None:
                    raise GeometryResolutionError(
                        f"{reaction['reaction_id']}: unknown state {state_id}"
                    )
                for solvent_id in assigned:
                    add(
                        geometry_key=state["reference_geometry_key"].format(
                            solvent_id=solvent_id
                        ),
                        state_id=state_id,
                        solvent_id=solvent_id,
                        formal_charge=int(state["formal_charge"]),
                        multiplicity=int(state["multiplicity"]),
                        geometry_source=state["reference_geometry_protocol"],
                        required_by=reaction["reaction_id"],
                    )
    else:
        for row in read_csv_rows(spec_dir / "sp_job_manifest.csv"):
            solvent_id = (
                row["paired_solvent_id"]
                if row["job_class"] == "diagnostic_gas_sp"
                else row["solvent_id"]
            )
            add(
                geometry_key=row["geometry_key"],
                state_id=row["state_id"],
                solvent_id=solvent_id,
                formal_charge=int(row["formal_charge"]),
                multiplicity=int(row["multiplicity"]),
                geometry_source=row["geometry_source"],
                required_by=row["job_id"],
            )
    return [
        GeometryRequest(
            geometry_key=key,
            state_id=value["identity"][0],  # type: ignore[index]
            solvent_id=value["identity"][1],  # type: ignore[index]
            formal_charge=value["identity"][2],  # type: ignore[index]
            multiplicity=value["identity"][3],  # type: ignore[index]
            geometry_source=value["identity"][4],  # type: ignore[index]
            required_by_job_ids=tuple(sorted(value["job_ids"])),  # type: ignore[arg-type]
        )
        for key, value in sorted(grouped.items())
    ]


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def _copy_hashed(source: Path, destination: Path, expected_sha256: str) -> str:
    if not source.is_file():
        raise GeometryResolutionError(f"missing source XYZ: {source}")
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise GeometryResolutionError(
            f"source XYZ hash mismatch for {source}: {actual} != {expected_sha256}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = sha256_file(destination)
    if copied != expected_sha256:
        raise GeometryResolutionError(f"copied XYZ hash mismatch: {destination}")
    return copied


def _source_species_id(parent_id: str) -> str:
    prefix = {"M": "monomer", "S": "solvent", "A": "anion"}.get(parent_id[0])
    if prefix is None:
        raise GeometryResolutionError(f"unsupported Tier-1 parent ID: {parent_id}")
    return f"{prefix}-{parent_id[1:]}"


def resolve_tier1_requests(
    requests: Sequence[GeometryRequest],
    *,
    spec_dir: Path,
    tier1_run: Path,
    run_dir: Path,
    require_clean: bool = True,
) -> list[dict[str, object]]:
    states = {
        row["state_id"]: row
        for row in read_csv_rows(spec_dir / "fullspace_state_registry.csv")
    }
    solvents = {
        row["solvent_id"]: row
        for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
    }
    task_rows = read_csv_rows(tier1_run / "manifests" / "redox_tasks.csv")
    status_rows = [
        row
        for row in read_csv_rows(tier1_run / "descriptors" / "task_status.csv")
        if row["batch"] == "redox"
    ]
    run_manifest = json.loads(
        (tier1_run / "manifests" / "run_manifest.json").read_text(encoding="utf-8")
    )
    run_id = str(run_manifest["run_id"])
    tasks_by_identity: dict[tuple[str, str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in task_rows:
        tasks_by_identity[
            (
                row["species_id"],
                row["solvent_name"],
                row["charge"],
                row["multiplicity"],
            )
        ].append(row)
    status_by_task = {row["task_id"]: row for row in status_rows}
    repository_root = spec_dir.parent
    results: list[dict[str, object]] = []

    def failed_row(
        request: GeometryRequest,
        *,
        reason: str,
        task: dict[str, str] | None = None,
        status: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return {
            "geometry_key": request.geometry_key,
            "state_id": request.state_id,
            "solvent_id": request.solvent_id,
            "formal_charge": request.formal_charge,
            "multiplicity": request.multiplicity,
            "status": "failed",
            "xyz_path": "",
            "xyz_sha256": "",
            "source_kind": "tier1_same_run_redox",
            "source_run_id": run_id,
            "source_task_id": "" if task is None else task.get("task_id", ""),
            "source_xyz_sha256": "" if status is None else status.get("optimized_xyz_sha256", ""),
            "source_output_sha256": "" if status is None else status.get("output_sha256", ""),
            "source_qc_status": "missing" if status is None else status.get("qc_status", "failed"),
            "connectivity_status": "missing" if status is None else status.get("geometry_ok", "false"),
            "bonds_broken": "" if status is None else status.get("bonds_broken", ""),
            "bonds_formed": "" if status is None else status.get("bonds_formed", ""),
            "monomer_source_sha256": "" if status is None else status.get("source_sha256", ""),
            "topology_sha256": "",
            "preopt_method_id": "Tier1_GFN2-xTB_default_opt_ddCOSMO",
            "preopt_epsilon": "" if task is None else task.get("epsilon_r", ""),
            "reason": reason,
        }

    for request in requests:
        if request.is_sigma:
            continue
        state = states.get(request.state_id)
        solvent = solvents.get(request.solvent_id)
        if state is None or solvent is None:
            raise GeometryResolutionError(
                f"{request.geometry_key}: state or solvent registry row is missing"
            )
        identity = (
            _source_species_id(state["parent_id"]),
            solvent["solvent_name"],
            str(request.formal_charge),
            str(request.multiplicity),
        )
        candidates = tasks_by_identity.get(identity, [])
        if len(candidates) != 1:
            reason = (
                f"expected one same-run Tier-1 task, found {len(candidates)}"
            )
            if require_clean:
                raise GeometryResolutionError(f"{request.geometry_key}: {reason}")
            results.append(failed_row(request, reason=reason))
            continue
        task = candidates[0]
        status = status_by_task.get(task["task_id"])
        if status is None:
            reason = "Tier-1 task status is missing"
            if require_clean:
                raise GeometryResolutionError(f"{request.geometry_key}: {reason}")
            results.append(failed_row(request, reason=reason, task=task))
            continue
        if status["run_id"] != run_id:
            raise GeometryResolutionError(
                f"{request.geometry_key}: task belongs to {status['run_id']}, expected {run_id}"
            )
        qc_ok = (
            status["status"] == "accepted"
            and status["geometry_ok"] == "true"
            and status["normal_termination"] == "true"
            and status["input_hash_ok"] == "true"
        )
        if not qc_ok:
            reason = (
                f"Tier-1 task is not clean: status={status['status']} "
                f"reason={status['reason']}"
            )
            if require_clean:
                raise GeometryResolutionError(f"{request.geometry_key}: {reason}")
            results.append(
                failed_row(request, reason=reason, task=task, status=status)
            )
            continue
        source = tier1_run / status["output_dir"] / "xtbopt.xyz"
        try:
            source_symbols = Counter(atom.symbol for atom in read_xyz(source))
            expected_symbols = Counter(
                atom.GetSymbol()
                for atom in Chem.AddHs(
                    molecule_from_smiles(state["smiles_or_generator"])
                ).GetAtoms()
            )
        except (OSError, ValueError) as exc:
            if require_clean:
                raise GeometryResolutionError(
                    f"{request.geometry_key}: Tier-1 XYZ cannot be read: {exc}"
                ) from exc
            results.append(
                failed_row(
                    request,
                    reason=f"Tier-1 XYZ cannot be read: {exc}",
                    task=task,
                    status=status,
                )
            )
            continue
        if source_symbols != expected_symbols:
            reason = (
                f"Tier-1 XYZ composition {source_symbols} differs from registered "
                f"state {expected_symbols}"
            )
            if require_clean:
                raise GeometryResolutionError(f"{request.geometry_key}: {reason}")
            results.append(
                failed_row(request, reason=reason, task=task, status=status)
            )
            continue
        destination = (
            run_dir / "resolved" / "tier1" / request.state_id / f"{request.solvent_id}.xyz"
        )
        try:
            copied_sha = _copy_hashed(
                source, destination, status["optimized_xyz_sha256"]
            )
        except GeometryResolutionError as exc:
            if require_clean:
                raise
            results.append(
                failed_row(request, reason=str(exc), task=task, status=status)
            )
            continue
        results.append(
            {
                "geometry_key": request.geometry_key,
                "state_id": request.state_id,
                "solvent_id": request.solvent_id,
                "formal_charge": request.formal_charge,
                "multiplicity": request.multiplicity,
                "status": "resolved",
                "xyz_path": _relative_or_absolute(destination, repository_root),
                "xyz_sha256": copied_sha,
                "source_kind": "tier1_same_run_redox",
                "source_run_id": run_id,
                "source_task_id": task["task_id"],
                "source_xyz_sha256": status["optimized_xyz_sha256"],
                "source_output_sha256": status["output_sha256"],
                "source_qc_status": status["qc_status"],
                "connectivity_status": "pass",
                "bonds_broken": status["bonds_broken"],
                "bonds_formed": status["bonds_formed"],
                "monomer_source_sha256": status["source_sha256"],
                "topology_sha256": "",
                "preopt_method_id": "Tier1_GFN2-xTB_default_opt_ddCOSMO",
                "preopt_epsilon": task["epsilon_r"],
                "reason": "",
            }
        )
    return results


def sigma_reference_bonds(topology: SigmaTopology) -> set[tuple[int, int]]:
    neutral_dimer = build_repeat_chain(
        topology.monomer_smiles,
        topology.site_a_atom_index_0based,
        topology.site_b_atom_index_0based,
        copies=2,
    )
    explicit_hydrogen_dimer = Chem.AddHs(Chem.Mol(neutral_dimer))
    bonds = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in explicit_hydrogen_dimer.GetBonds()
    }
    first_restored_h = explicit_hydrogen_dimer.GetNumAtoms()
    bonds.add((topology.junction_copy1_atom_index_0based, first_restored_h))
    bonds.add((topology.junction_copy2_atom_index_0based, first_restored_h + 1))
    return bonds


def _sigma_raw_path(run_dir: Path, state_id: str) -> Path:
    return run_dir / "raw_sigma" / f"{state_id}.xyz"


def _sigma_output_dir(run_dir: Path, state_id: str, solvent_id: str) -> Path:
    return run_dir / "sigma_preopt" / f"{state_id}__{solvent_id}"


def prepare_and_resolve_sigma_requests(
    requests: Sequence[GeometryRequest],
    *,
    spec_dir: Path,
    run_dir: Path,
    n_conformers: int = N_CONFORMERS,
    reuse_existing_inputs: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    topology_by_state = {
        topology.sigma_state_id: topology
        for topology in load_sigma_topologies(spec_dir / "sigma_coupling_topology.csv")
    }
    solvent_by_id = {
        row["solvent_id"]: row
        for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
    }
    monomer_sha256 = sha256_file(spec_dir / "source_fullspace_monomers.csv")
    repository_root = spec_dir.parent
    prepared_states: set[str] = set()
    manifest: list[dict[str, object]] = []
    results: list[dict[str, object]] = []

    for request in requests:
        if not request.is_sigma:
            continue
        topology = topology_by_state.get(request.state_id)
        solvent = solvent_by_id.get(request.solvent_id)
        if topology is None or solvent is None:
            raise GeometryResolutionError(
                f"{request.geometry_key}: sigma topology or solvent row is missing"
            )
        raw_path = _sigma_raw_path(run_dir, request.state_id)
        if request.state_id not in prepared_states:
            if reuse_existing_inputs:
                if not raw_path.is_file():
                    raise GeometryResolutionError(
                        f"{request.geometry_key}: reusable sigma input is missing: {raw_path}"
                    )
            else:
                sigma = build_sigma_complex(topology, n_conformers=n_conformers)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(sigma.xyz_text(), encoding="utf-8")
            prepared_states.add(request.state_id)
        raw_sha256 = sha256_file(raw_path)
        output_dir = _sigma_output_dir(
            run_dir, request.state_id, request.solvent_id
        )
        task_id = f"sigma-preopt-{request.state_id.split('_', 1)[0]}-{request.solvent_id}"
        epsilon = solvent["epsilon"]
        manifest.append(
            {
                "task_id": task_id,
                "geometry_key": request.geometry_key,
                "state_id": request.state_id,
                "parent_id": topology.parent_id,
                "solvent_id": request.solvent_id,
                "solvent_name": solvent["solvent_name"],
                "source_xyz": str(raw_path.relative_to(run_dir)),
                "source_xyz_sha256": raw_sha256,
                "output_dir": str(output_dir.relative_to(run_dir)),
                "formal_charge": topology.charge,
                "multiplicity": topology.multiplicity,
                "uhf": topology.multiplicity - 1,
                "epsilon": epsilon,
                "preopt_method_id": SIGMA_PREOPT_METHOD_ID,
                "monomer_source_sha256": monomer_sha256,
                "topology_sha256": topology.topology_sha256,
                "xtb_command": (
                    f"xtb in.xyz --chrg {topology.charge} --uhf "
                    f"{topology.multiplicity - 1} --opt --cosmo {epsilon}"
                ),
            }
        )

        optimized = output_dir / "xtbopt.xyz"
        xtb_output = output_dir / "xtb.out"
        status = "pending"
        reason = "sigma target-medium preoptimization output is absent"
        xyz_path = ""
        xyz_sha256 = ""
        output_sha256 = ""
        connectivity_status = "pending"
        bonds_broken: int | str = ""
        bonds_formed: int | str = ""
        if optimized.is_file() and xtb_output.is_file():
            output_text = xtb_output.read_text(encoding="utf-8", errors="replace").lower()
            input_copy = output_dir / "in.xyz"
            if "normal termination of xtb" not in output_text:
                status = "failed"
                reason = "xTB normal termination marker is missing"
            elif not input_copy.is_file() or sha256_file(input_copy) != raw_sha256:
                status = "failed"
                reason = "preoptimization input XYZ hash mismatch"
            else:
                try:
                    connectivity = check_connectivity(
                        raw_path,
                        optimized,
                        reference_bonds=sigma_reference_bonds(topology),
                    )
                    bonds_broken = connectivity.bonds_broken
                    bonds_formed = connectivity.bonds_formed
                    connectivity_status = "pass" if connectivity.ok else "fail"
                    if connectivity.ok:
                        materialized = (
                            run_dir
                            / "resolved"
                            / "sigma"
                            / request.state_id
                            / f"{request.solvent_id}.xyz"
                        )
                        optimized_sha = sha256_file(optimized)
                        xyz_sha256 = _copy_hashed(
                            optimized, materialized, optimized_sha
                        )
                        xyz_path = _relative_or_absolute(
                            materialized, repository_root
                        )
                        output_sha256 = sha256_file(xtb_output)
                        status = "resolved"
                        reason = ""
                    else:
                        status = "failed"
                        reason = "sigma connectivity changed during target-medium preoptimization"
                except (OSError, ValueError) as exc:
                    status = "failed"
                    reason = f"sigma connectivity QC error: {exc}"

        results.append(
            {
                "geometry_key": request.geometry_key,
                "state_id": request.state_id,
                "solvent_id": request.solvent_id,
                "formal_charge": request.formal_charge,
                "multiplicity": request.multiplicity,
                "status": status,
                "xyz_path": xyz_path,
                "xyz_sha256": xyz_sha256,
                "source_kind": "sigma_target_medium_preopt",
                "source_run_id": SIGMA_PREOPT_RUN_ID,
                "source_task_id": task_id,
                "source_xyz_sha256": raw_sha256,
                "source_output_sha256": output_sha256,
                "source_qc_status": connectivity_status,
                "connectivity_status": connectivity_status,
                "bonds_broken": bonds_broken,
                "bonds_formed": bonds_formed,
                "monomer_source_sha256": monomer_sha256,
                "topology_sha256": topology.topology_sha256,
                "preopt_method_id": SIGMA_PREOPT_METHOD_ID,
                "preopt_epsilon": epsilon,
                "reason": reason,
            }
        )

    write_csv_deterministic(
        run_dir / "sigma_preopt_manifest.csv",
        SIGMA_PREOPT_MANIFEST_FIELDS,
        manifest,
        sort_by=("task_id",),
    )
    array_path = run_dir / "sigma_preopt_array.tsv"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\t".join(
            (
                str(row["task_id"]),
                str(row["source_xyz"]),
                str(row["source_xyz_sha256"]),
                str(row["output_dir"]),
                str(row["formal_charge"]),
                str(row["uhf"]),
                str(row["epsilon"]),
                str(row["topology_sha256"]),
            )
        )
        for row in sorted(manifest, key=lambda item: str(item["task_id"]))
    ]
    array_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results, manifest


def validate_sigma_preopt_files(
    *,
    spec_dir: Path,
    run_dir: Path,
    launcher_path: Path,
    include_fullspace_inference: bool = False,
) -> dict[str, object]:
    """Inspect the exact xTB task files that will be submitted to Lop."""

    manifest_path = run_dir / "sigma_preopt_manifest.csv"
    array_path = run_dir / "sigma_preopt_array.tsv"
    rows = read_csv_rows(manifest_path)
    request_by_key = {
        request.geometry_key: request
        for request in geometry_requests(
            spec_dir, include_fullspace_inference=include_fullspace_inference
        )
        if request.is_sigma
    }
    topology_by_state = {
        topology.sigma_state_id: topology
        for topology in load_sigma_topologies(spec_dir / "sigma_coupling_topology.csv")
    }
    solvent_by_id = {
        row["solvent_id"]: row
        for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
    }
    issues: list[str] = []
    exact_commands = 0
    exact_parameters = 0
    exact_sources = 0
    unique_outputs: set[str] = set()
    for row in rows:
        request = request_by_key.get(row["geometry_key"])
        topology = topology_by_state.get(row["state_id"])
        solvent = solvent_by_id.get(row["solvent_id"])
        if request is None or topology is None or solvent is None:
            issues.append(f"{row['task_id']}: missing request, topology or solvent authority")
            continue
        expected_command = (
            f"xtb in.xyz --chrg 2 --uhf 0 --opt --cosmo {solvent['epsilon']}"
        )
        if row["xtb_command"] == expected_command:
            exact_commands += 1
        else:
            issues.append(f"{row['task_id']}: xTB command differs from frozen settings")
        parameter_match = (
            row["formal_charge"] == str(request.formal_charge) == "2"
            and row["multiplicity"] == str(request.multiplicity) == "1"
            and row["uhf"] == "0"
            and row["epsilon"] == solvent["epsilon"]
            and row["preopt_method_id"] == SIGMA_PREOPT_METHOD_ID
            and row["topology_sha256"] == topology.topology_sha256
        )
        if parameter_match:
            exact_parameters += 1
        else:
            issues.append(f"{row['task_id']}: charge/spin/epsilon/method/topology drift")
        source = run_dir / row["source_xyz"]
        if source.is_file() and sha256_file(source) == row["source_xyz_sha256"]:
            exact_sources += 1
        else:
            issues.append(f"{row['task_id']}: source XYZ is absent or hash-mismatched")
        if row["output_dir"] in unique_outputs:
            issues.append(f"{row['task_id']}: duplicate output directory {row['output_dir']}")
        unique_outputs.add(row["output_dir"])

    expected_array_rows = [
        "\t".join(
            (
                row["task_id"],
                row["source_xyz"],
                row["source_xyz_sha256"],
                row["output_dir"],
                row["formal_charge"],
                row["uhf"],
                row["epsilon"],
                row["topology_sha256"],
            )
        )
        for row in sorted(rows, key=lambda item: item["task_id"])
    ]
    actual_array_rows = array_path.read_text(encoding="utf-8").splitlines()
    if actual_array_rows != expected_array_rows:
        issues.append("sigma_preopt_array.tsv differs from the inspected manifest")
    if set(request_by_key) != {row["geometry_key"] for row in rows}:
        issues.append(
            "sigma preoptimization manifest does not cover every requested sigma key"
        )
    if len(rows) != len(request_by_key):
        issues.append(
            f"expected {len(request_by_key)} sigma preoptimization rows, found {len(rows)}"
        )
    if not launcher_path.is_file():
        issues.append(f"missing Lop launcher: {launcher_path}")

    report: dict[str, object] = {
        "status": "PASS" if not issues else "FAIL",
        "task_count": len(rows),
        "unique_geometry_keys": len({row["geometry_key"] for row in rows}),
        "unique_sigma_states": len({row["state_id"] for row in rows}),
        "charge_counts": dict(sorted(Counter(row["formal_charge"] for row in rows).items())),
        "multiplicity_counts": dict(sorted(Counter(row["multiplicity"] for row in rows).items())),
        "uhf_counts": dict(sorted(Counter(row["uhf"] for row in rows).items())),
        "unique_media": len({row["solvent_id"] for row in rows}),
        "exact_parameter_rows": exact_parameters,
        "exact_command_rows": exact_commands,
        "source_hash_rows": exact_sources,
        "array_rows": len(actual_array_rows),
        "manifest_sha256": sha256_file(manifest_path),
        "array_sha256": sha256_file(array_path),
        "launcher_sha256": sha256_file(launcher_path) if launcher_path.is_file() else "",
        "issues": issues,
    }
    return report


def resolve_geometries(
    *,
    spec_dir: Path,
    tier1_run: Path,
    run_dir: Path,
    index_path: Path,
    n_conformers: int = N_CONFORMERS,
    include_fullspace_inference: bool = False,
    reuse_existing_sigma_inputs: bool = False,
) -> GeometryResolutionSummary:
    requests = geometry_requests(
        spec_dir, include_fullspace_inference=include_fullspace_inference
    )
    tier1_rows = resolve_tier1_requests(
        requests,
        spec_dir=spec_dir,
        tier1_run=tier1_run,
        run_dir=run_dir,
        require_clean=not include_fullspace_inference,
    )
    sigma_rows, _manifest = prepare_and_resolve_sigma_requests(
        requests,
        spec_dir=spec_dir,
        run_dir=run_dir,
        n_conformers=n_conformers,
        reuse_existing_inputs=reuse_existing_sigma_inputs,
    )
    preflight = validate_sigma_preopt_files(
        spec_dir=spec_dir,
        run_dir=run_dir,
        launcher_path=spec_dir.parent
        / "hpc"
        / (
            "run_sigma_preopt_budgeted.sh"
            if include_fullspace_inference
            else "run_sigma_preopt.sh"
        ),
        include_fullspace_inference=include_fullspace_inference,
    )
    if preflight["status"] != "PASS":
        raise GeometryResolutionError(
            f"sigma preoptimization file inspection failed: {preflight['issues']}"
        )
    preflight_name = (
        "fullspace_sigma_preopt_preflight.json"
        if include_fullspace_inference
        else "sigma_preopt_preflight.json"
    )
    preflight_path = index_path.parent / preflight_name
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = sorted(tier1_rows + sigma_rows, key=lambda row: str(row["geometry_key"]))
    if len(rows) != len(requests):
        raise GeometryResolutionError(
            f"resolved index rows {len(rows)} != geometry requests {len(requests)}"
        )
    write_csv_deterministic(
        index_path,
        GEOMETRY_INDEX_FIELDS,
        rows,
        sort_by=("geometry_key",),
    )
    statuses = Counter(str(row["status"]) for row in rows)
    return GeometryResolutionSummary(
        total=len(rows),
        resolved=statuses["resolved"],
        pending=statuses["pending"],
        failed=statuses["failed"],
        tier1_resolved=sum(row["status"] == "resolved" for row in tier1_rows),
        sigma_resolved=sum(
            row["status"] == "resolved" for row in sigma_rows
        ),
    )
