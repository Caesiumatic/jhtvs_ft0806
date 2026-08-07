"""Assemble state and reaction labels from frozen stoichiometry and parsed ORCA data."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping

from jhtvs_ft0806.orca.parser import HARTREE_TO_EV, KCAL_PER_HARTREE
from jhtvs_ft0806.schemas import (
    csv_fieldnames,
    read_csv_rows,
    write_csv_deterministic,
)
from jhtvs_ft0806.spec_validation import validate_spec


PINNED_REFERENCE_CONVERSION_RELATIVE_PATH = Path("tier2/src/parse_orca.py")
PINNED_REFERENCE_CONVERSION_SHA256 = (
    "09582db38f81d8992bc83ea8ef87499334ee66b154ae91af0e461d13156de525"
)

STATE_SP_FIELDS = (
    "source_job_id",
    "state_id",
    "solvent_id",
    "geometry_hash",
    "formal_charge",
    "multiplicity",
    "split",
    "method_id",
    "workflow_revision",
    "E_base_MACE_eV",
    "E_T2_SMD_SP_Eh",
    "E_T2_SMD_SP_eV",
    "CPCM_dielectric_Eh",
    "SMD_CDS_Eh",
    "echoed_smd_vector",
    "normal_termination",
    "qc_status",
    "qc_reasons",
    "output_sha256",
)
REACTION_SP_FIELDS = (
    "reaction_id",
    "reaction_class",
    "role",
    "parent_id",
    "solvent_id",
    "split",
    "stoichiometry",
    "deltaE_base_MACE_rxn_eV",
    "deltaE_T2_SMD_SP_rxn_eV",
    "sp_residual_eV",
    "delta_CPCM_dielectric_rxn_eV",
    "delta_SMD_CDS_rxn_eV",
    "complete_tuple",
    "qc_status",
    "qc_reasons",
    "source_job_ids",
)
REACTION_FINAL_FIELDS = (
    "reaction_id",
    "reaction_class",
    "role",
    "parent_id",
    "solvent_id",
    "split",
    "stoichiometry",
    "deltaE_base_MACE_rxn_eV",
    "deltaE_T2_SMD_SP_rxn_eV",
    "deltaG_T2_SMD_min_rxn_eV",
    "sp_residual_eV",
    "rt_correction_eV",
    "final_residual_eV",
    "Eox_vs_AgAgCl_V",
    "deltaG_sigma_kcal_mol",
    "complete_tuple",
    "qc_status",
    "qc_reasons",
    "source_job_ids",
)
ECHO_RESULT_FIELDS = (
    "echo_epsilon",
    "echo_soln",
    "echo_soln25",
    "echo_sola",
    "echo_solb",
    "echo_solg",
    "echo_solc",
    "echo_solh",
)


class LabelAssemblyError(ValueError):
    """Raised when label inputs conflict with the frozen schemas or registries."""


@dataclass(frozen=True, slots=True)
class AssemblySummary:
    state_sp_rows: int
    state_sp_clean: int
    state_sp_flagged: int
    state_sp_missing: int
    reaction_sp_rows: int
    reaction_sp_clean: int
    reaction_sp_flagged: int
    reaction_sp_missing: int
    reaction_final_rows: int
    reaction_final_clean: int
    reaction_final_flagged: int
    reaction_final_missing: int
    baseline_state_rows: int
    scientific_stops: int
    reference_conversion_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": "SCIENTIFIC_STOP" if self.scientific_stops else "PASS",
            "state_sp_rows": self.state_sp_rows,
            "state_sp_clean": self.state_sp_clean,
            "state_sp_flagged": self.state_sp_flagged,
            "state_sp_missing": self.state_sp_missing,
            "reaction_sp_rows": self.reaction_sp_rows,
            "reaction_sp_clean": self.reaction_sp_clean,
            "reaction_sp_flagged": self.reaction_sp_flagged,
            "reaction_sp_missing": self.reaction_sp_missing,
            "reaction_final_rows": self.reaction_final_rows,
            "reaction_final_clean": self.reaction_final_clean,
            "reaction_final_flagged": self.reaction_final_flagged,
            "reaction_final_missing": self.reaction_final_missing,
            "baseline_state_rows": self.baseline_state_rows,
            "scientific_stops": self.scientific_stops,
            "reference_conversion_sha256": self.reference_conversion_sha256,
        }


def parse_stoichiometry(text: str) -> tuple[tuple[str, int], ...]:
    terms: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw_term in text.split(";"):
        try:
            state_id, coefficient_text = raw_term.rsplit(":", 1)
            coefficient = int(coefficient_text)
        except ValueError as exc:
            raise LabelAssemblyError(f"invalid stoichiometric term: {raw_term!r}") from exc
        if not state_id or coefficient == 0 or state_id in seen:
            raise LabelAssemblyError(f"invalid stoichiometric term: {raw_term!r}")
        seen.add(state_id)
        terms.append((state_id, coefficient))
    if not terms:
        raise LabelAssemblyError("empty stoichiometry")
    return tuple(terms)


def aggregate_stoichiometric(
    stoichiometry: tuple[tuple[str, int], ...], values: Mapping[str, float | None]
) -> float | None:
    if any(values.get(state_id) is None for state_id, _ in stoichiometry):
        return None
    return sum(
        coefficient * float(values[state_id])
        for state_id, coefficient in stoichiometry
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_conversion(
    source_path: Path,
    *,
    expected_sha256: str = PINNED_REFERENCE_CONVERSION_SHA256,
) -> Callable[[float | None], float | None]:
    """Load and call the exact pinned project conversion without copying its constants."""

    if not source_path.is_file() or source_path.is_symlink():
        raise LabelAssemblyError(f"reference conversion source is missing: {source_path}")
    observed_sha256 = _sha256(source_path)
    if observed_sha256 != expected_sha256:
        raise LabelAssemblyError(
            "reference conversion source hash mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    required_assignments = {"SHE_ABS_V", "AGCL_VS_SHE_V"}
    selected: list[ast.stmt] = []
    found_assignments: set[str] = set()
    found_function = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & required_assignments:
                selected.append(node)
                found_assignments.update(names & required_assignments)
        elif isinstance(node, ast.FunctionDef) and node.name == "v_vs_agcl":
            selected.append(node)
            found_function = True
    if found_assignments != required_assignments or not found_function:
        raise LabelAssemblyError("pinned project conversion definition is incomplete")
    namespace: dict[str, object] = {}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)
    conversion = namespace.get("v_vs_agcl")
    if not callable(conversion) or conversion(None) is not None:
        raise LabelAssemblyError("pinned project conversion contract is invalid")
    return conversion  # type: ignore[return-value]


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float | None) -> str:
    return "" if value is None else format(value, ".15g")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _source_reasons(job_id: str, row: Mapping[str, object]) -> list[str]:
    raw = str(row.get("qc_reasons", ""))
    return [f"{job_id}:{reason}" for reason in raw.split(";") if reason]


def _qc_status(*, complete: bool, statuses: list[str], reasons: list[str]) -> str:
    if not complete or "missing" in statuses:
        return "missing"
    if "flagged" in statuses or reasons:
        return "flagged"
    return "clean"


def _echo_vector(result: Mapping[str, str]) -> str:
    if any(not result.get(field, "") for field in ECHO_RESULT_FIELDS):
        return ""
    names = ("epsilon", "soln", "soln25", "sola", "solb", "solg", "solc", "solh")
    return ";".join(
        f"{name}={result[field]}" for name, field in zip(names, ECHO_RESULT_FIELDS)
    )


def _load_baselines(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str], set[str]],
    int,
]:
    if not path.is_file():
        return {}, {}, 0
    required = {"state_id", "solvent_id", "geometry_hash", "E_base_MACE_eV"}
    fields = set(csv_fieldnames(path))
    if not required <= fields:
        raise LabelAssemblyError(
            f"baseline table is missing fields: {sorted(required - fields)}"
        )
    values: dict[tuple[str, str, str], float] = {}
    geometries: dict[tuple[str, str], set[str]] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    rows = read_csv_rows(path)
    for row in rows:
        key = (row["state_id"], row["solvent_id"], row["geometry_hash"])
        value = _float(row["E_base_MACE_eV"])
        if key in seen_keys:
            raise LabelAssemblyError(f"duplicate baseline key: {key}")
        seen_keys.add(key)
        if value is not None:
            values[key] = value
        geometries.setdefault(key[:2], set()).add(key[2])
    return values, geometries, len(rows)


def _assert_output_schemas(spec_dir: Path) -> None:
    expected = {
        "state_sp_labels.csv": STATE_SP_FIELDS,
        "reaction_sp_labels.csv": REACTION_SP_FIELDS,
        "reaction_final_labels.csv": REACTION_FINAL_FIELDS,
    }
    for filename, fields in expected.items():
        observed = csv_fieldnames(spec_dir / filename)
        if observed != fields:
            raise LabelAssemblyError(f"frozen output schema drift: {filename}")


def assemble_labels(
    *,
    spec_dir: Path,
    state_results_path: Path,
    baseline_state_energies_path: Path,
    state_sp_output_path: Path,
    reaction_sp_output_path: Path,
    reaction_final_output_path: Path,
    reference_conversion_path: Path,
    reference_conversion_sha256: str = PINNED_REFERENCE_CONVERSION_SHA256,
) -> AssemblySummary:
    spec_dir = spec_dir.resolve()
    validation = validate_spec(spec_dir)
    if not validation.ok:
        raise LabelAssemblyError("scientific specification validation failed")
    _assert_output_schemas(spec_dir)
    conversion = load_reference_conversion(
        reference_conversion_path, expected_sha256=reference_conversion_sha256
    )
    if not state_results_path.is_file():
        raise LabelAssemblyError(f"parsed state result table is missing: {state_results_path}")

    result_rows = read_csv_rows(state_results_path)
    results_by_job: dict[str, dict[str, str]] = {}
    for row in result_rows:
        job_id = row["job_id"]
        if job_id in results_by_job:
            raise LabelAssemblyError(f"duplicate parsed result job_id: {job_id}")
        results_by_job[job_id] = row
    scientific_stops = sum(
        row.get("scientific_stop_required", "false").lower() == "true"
        for row in result_rows
    )

    baseline_values, baseline_geometries, baseline_count = _load_baselines(
        baseline_state_energies_path
    )
    sp_jobs = read_csv_rows(spec_dir / "sp_job_manifest.csv")
    optfreq_jobs = read_csv_rows(spec_dir / "optfreq_job_manifest.csv")
    state_rows: list[dict[str, object]] = []
    state_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for job in sp_jobs:
        if job["job_class"] != "smd_energy_sp":
            continue
        job_id = job["job_id"]
        result = results_by_job.get(job_id, {})
        geometry_hash = result.get("geometry_sha256", "") or job.get("geometry_hash", "")
        baseline_key = (job["state_id"], job["solvent_id"], geometry_hash)
        baseline = baseline_values.get(baseline_key)
        reasons = _source_reasons(job_id, result)
        source_status = result.get("qc_status", "missing")
        if not result:
            _append_reason(reasons, f"{job_id}:result_row_missing")
        if baseline is None:
            if baseline_geometries.get(baseline_key[:2]):
                _append_reason(reasons, "base_geometry_hash_mismatch")
            else:
                _append_reason(reasons, "base_energy_missing")
        energy_eh = _float(result.get("final_energy_Eh"))
        cpcm_eh = _float(result.get("cpcm_dielectric_Eh"))
        cds_eh = _float(result.get("smd_cds_Eh"))
        complete = all(value is not None for value in (baseline, energy_eh, cpcm_eh, cds_eh))
        status = _qc_status(
            complete=complete,
            statuses=[source_status, "missing" if baseline is None else "clean"],
            reasons=reasons,
        )
        row: dict[str, object] = {
            "source_job_id": job_id,
            "state_id": job["state_id"],
            "solvent_id": job["solvent_id"],
            "geometry_hash": geometry_hash,
            "formal_charge": job["formal_charge"],
            "multiplicity": job["multiplicity"],
            "split": job["split"],
            "method_id": job["method_id"],
            "workflow_revision": job["workflow_revision"],
            "E_base_MACE_eV": _fmt(baseline),
            "E_T2_SMD_SP_Eh": _fmt(energy_eh),
            "E_T2_SMD_SP_eV": _fmt(
                None if energy_eh is None else energy_eh * HARTREE_TO_EV
            ),
            "CPCM_dielectric_Eh": _fmt(cpcm_eh),
            "SMD_CDS_Eh": _fmt(cds_eh),
            "echoed_smd_vector": _echo_vector(result),
            "normal_termination": result.get("normal_termination", "false"),
            "qc_status": status,
            "qc_reasons": ";".join(reasons),
            "output_sha256": result.get("output_sha256", ""),
        }
        key = (job["state_id"], job["solvent_id"])
        if key in state_by_key:
            raise LabelAssemblyError(f"duplicate SMD SP state-medium key: {key}")
        state_by_key[key] = row
        state_rows.append(row)

    reaction_registry = {
        row["reaction_id"]: row
        for row in read_csv_rows(spec_dir / "fullspace_reaction_registry.csv")
    }
    design_rows = read_csv_rows(spec_dir / "calibration_tuple_design.csv")
    reaction_pairs: list[tuple[dict[str, str], dict[str, str], str]] = []
    for design in design_rows:
        reaction = reaction_registry[design["reaction_id"]]
        solvents = design["solvent_ids"].split(";")
        if len(solvents) != int(design["solvent_count"]):
            raise LabelAssemblyError(
                f"{design['reaction_id']}: solvent count changed after validation"
            )
        reaction_pairs.extend((design, reaction, solvent) for solvent in solvents)

    reaction_sp_rows: list[dict[str, object]] = []
    reaction_sp_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for design, reaction, solvent_id in reaction_pairs:
        stoichiometry = parse_stoichiometry(reaction["stoichiometry"])
        source_rows: list[dict[str, object]] = []
        missing_states: list[str] = []
        for state_id, _ in stoichiometry:
            state = state_by_key.get((state_id, solvent_id))
            if state is None:
                missing_states.append(state_id)
            else:
                source_rows.append(state)
        reasons: list[str] = []
        for state_id in missing_states:
            _append_reason(reasons, f"{state_id}:sp_state_missing")
        statuses = [str(row["qc_status"]) for row in source_rows]
        for row in source_rows:
            for reason in str(row["qc_reasons"]).split(";"):
                _append_reason(reasons, reason)

        by_state = {str(row["state_id"]): row for row in source_rows}
        base = aggregate_stoichiometric(
            stoichiometry,
            {state: _float(row["E_base_MACE_eV"]) for state, row in by_state.items()},
        )
        t2_eh = aggregate_stoichiometric(
            stoichiometry,
            {state: _float(row["E_T2_SMD_SP_Eh"]) for state, row in by_state.items()},
        )
        cpcm_eh = aggregate_stoichiometric(
            stoichiometry,
            {state: _float(row["CPCM_dielectric_Eh"]) for state, row in by_state.items()},
        )
        cds_eh = aggregate_stoichiometric(
            stoichiometry,
            {state: _float(row["SMD_CDS_Eh"]) for state, row in by_state.items()},
        )
        t2_ev = None if t2_eh is None else t2_eh * HARTREE_TO_EV
        complete = not missing_states and all(
            value is not None for value in (base, t2_ev, cpcm_eh, cds_eh)
        )
        status = _qc_status(complete=complete, statuses=statuses, reasons=reasons)
        row = {
            "reaction_id": reaction["reaction_id"],
            "reaction_class": reaction["reaction_class"],
            "role": reaction["role"],
            "parent_id": reaction["parent_id"],
            "solvent_id": solvent_id,
            "split": design["split"],
            "stoichiometry": reaction["stoichiometry"],
            "deltaE_base_MACE_rxn_eV": _fmt(base),
            "deltaE_T2_SMD_SP_rxn_eV": _fmt(t2_ev),
            "sp_residual_eV": _fmt(
                None if base is None or t2_ev is None else t2_ev - base
            ),
            "delta_CPCM_dielectric_rxn_eV": _fmt(
                None if cpcm_eh is None else cpcm_eh * HARTREE_TO_EV
            ),
            "delta_SMD_CDS_rxn_eV": _fmt(
                None if cds_eh is None else cds_eh * HARTREE_TO_EV
            ),
            "complete_tuple": str(complete).lower(),
            "qc_status": status,
            "qc_reasons": ";".join(dict.fromkeys(reasons)),
            "source_job_ids": ";".join(
                str(by_state[state_id]["source_job_id"])
                for state_id, _ in stoichiometry
                if state_id in by_state
            ),
        }
        key = (reaction["reaction_id"], solvent_id)
        if key in reaction_sp_by_key:
            raise LabelAssemblyError(f"duplicate reaction-medium key: {key}")
        reaction_sp_by_key[key] = row
        reaction_sp_rows.append(row)

    optfreq_by_key: dict[tuple[str, str], dict[str, str]] = {}
    final_pair_keys: set[tuple[str, str]] = set()
    for job in optfreq_jobs:
        state_key = (job["state_id"], job["solvent_id"])
        if state_key in optfreq_by_key:
            raise LabelAssemblyError(f"duplicate Opt/Freq state-medium key: {state_key}")
        optfreq_by_key[state_key] = job
        for reaction_id in job["reaction_ids"].split(";"):
            final_pair_keys.add((reaction_id, job["solvent_id"]))

    design_by_reaction = {row["reaction_id"]: row for row in design_rows}
    reaction_final_rows: list[dict[str, object]] = []
    for reaction_id, solvent_id in sorted(final_pair_keys):
        reaction = reaction_registry[reaction_id]
        design = design_by_reaction[reaction_id]
        stoichiometry = parse_stoichiometry(reaction["stoichiometry"])
        source_jobs: list[dict[str, str]] = []
        source_results: list[dict[str, str]] = []
        reasons: list[str] = []
        for state_id, _ in stoichiometry:
            job = optfreq_by_key.get((state_id, solvent_id))
            if job is None:
                _append_reason(reasons, f"{state_id}:optfreq_job_missing")
                continue
            source_jobs.append(job)
            result = results_by_job.get(job["job_id"])
            if result is None:
                _append_reason(reasons, f"{job['job_id']}:result_row_missing")
            else:
                source_results.append(result)
                reasons.extend(_source_reasons(job["job_id"], result))
        result_by_state = {
            job["state_id"]: results_by_job.get(job["job_id"], {})
            for job in source_jobs
        }
        delta_g_eh = aggregate_stoichiometric(
            stoichiometry,
            {
                state: _float(result.get("G_composite_1M_Eh"))
                for state, result in result_by_state.items()
            },
        )
        delta_g_ev = None if delta_g_eh is None else delta_g_eh * HARTREE_TO_EV
        sp_row = reaction_sp_by_key[(reaction_id, solvent_id)]
        for reason in str(sp_row["qc_reasons"]).split(";"):
            _append_reason(reasons, reason)
        base = _float(sp_row["deltaE_base_MACE_rxn_eV"])
        t2_sp = _float(sp_row["deltaE_T2_SMD_SP_rxn_eV"])
        sp_residual = _float(sp_row["sp_residual_eV"])
        complete = all(value is not None for value in (base, t2_sp, delta_g_ev))
        statuses = [str(sp_row["qc_status"])] + [
            result.get("qc_status", "missing") for result in source_results
        ]
        status = _qc_status(complete=complete, statuses=statuses, reasons=reasons)
        all_job_ids = str(sp_row["source_job_ids"]).split(";") + [
            job["job_id"] for job in source_jobs
        ]
        row = {
            "reaction_id": reaction_id,
            "reaction_class": reaction["reaction_class"],
            "role": reaction["role"],
            "parent_id": reaction["parent_id"],
            "solvent_id": solvent_id,
            "split": design["split"],
            "stoichiometry": reaction["stoichiometry"],
            "deltaE_base_MACE_rxn_eV": _fmt(base),
            "deltaE_T2_SMD_SP_rxn_eV": _fmt(t2_sp),
            "deltaG_T2_SMD_min_rxn_eV": _fmt(delta_g_ev),
            "sp_residual_eV": _fmt(sp_residual),
            "rt_correction_eV": _fmt(
                None if delta_g_ev is None or t2_sp is None else delta_g_ev - t2_sp
            ),
            "final_residual_eV": _fmt(
                None if delta_g_ev is None or base is None else delta_g_ev - base
            ),
            "Eox_vs_AgAgCl_V": _fmt(
                conversion(delta_g_ev)
                if reaction["reaction_class"] == "redox" and delta_g_ev is not None
                else None
            ),
            "deltaG_sigma_kcal_mol": _fmt(
                delta_g_eh * KCAL_PER_HARTREE
                if reaction["reaction_class"] == "sigma_dimerization"
                and delta_g_eh is not None
                else None
            ),
            "complete_tuple": str(complete).lower(),
            "qc_status": status,
            "qc_reasons": ";".join(dict.fromkeys(reasons)),
            "source_job_ids": ";".join(dict.fromkeys(job for job in all_job_ids if job)),
        }
        reaction_final_rows.append(row)

    write_csv_deterministic(
        state_sp_output_path,
        STATE_SP_FIELDS,
        state_rows,
        sort_by=("source_job_id",),
    )
    write_csv_deterministic(
        reaction_sp_output_path,
        REACTION_SP_FIELDS,
        reaction_sp_rows,
        sort_by=("reaction_id", "solvent_id"),
    )
    write_csv_deterministic(
        reaction_final_output_path,
        REACTION_FINAL_FIELDS,
        reaction_final_rows,
        sort_by=("reaction_id", "solvent_id"),
    )

    state_counts = Counter(str(row["qc_status"]) for row in state_rows)
    sp_counts = Counter(str(row["qc_status"]) for row in reaction_sp_rows)
    final_counts = Counter(str(row["qc_status"]) for row in reaction_final_rows)
    return AssemblySummary(
        state_sp_rows=len(state_rows),
        state_sp_clean=state_counts["clean"],
        state_sp_flagged=state_counts["flagged"],
        state_sp_missing=state_counts["missing"],
        reaction_sp_rows=len(reaction_sp_rows),
        reaction_sp_clean=sp_counts["clean"],
        reaction_sp_flagged=sp_counts["flagged"],
        reaction_sp_missing=sp_counts["missing"],
        reaction_final_rows=len(reaction_final_rows),
        reaction_final_clean=final_counts["clean"],
        reaction_final_flagged=final_counts["flagged"],
        reaction_final_missing=final_counts["missing"],
        baseline_state_rows=baseline_count,
        scientific_stops=scientific_stops,
        reference_conversion_sha256=reference_conversion_sha256,
    )
