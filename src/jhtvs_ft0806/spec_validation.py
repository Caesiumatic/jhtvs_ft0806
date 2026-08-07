"""Validation of the immutable scientific specification bundle."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import (
    SchemaError,
    ValidationIssue,
    ValidationReport,
    csv_fieldnames,
    read_csv_rows,
)

EXPECTED_COUNTS = {
    "solvent_smd_registry.csv": 25,
    "fullspace_state_registry.csv": 372,
    "fullspace_reaction_registry.csv": 236,
    "calibration_tuple_design.csv": 88,
    "sp_job_manifest.csv": 735,
    "optfreq_job_manifest.csv": 80,
    "training_config.csv": 57,
    "compute_budget.csv": 5,
}

EXPECTED_FULL25_ANCHORS = {
    "RXN_MOX_M022": ("train", "train_full25"),
    "RXN_AOX_A005": ("train", "train_full25"),
    "RXN_SIG_M022": ("train", "train_full25"),
    "RXN_MOX_M084": ("test", "test_full25"),
    "RXN_AOX_A002": ("test", "test_full25"),
    "RXN_SIG_M027": ("test", "test_full25"),
}

MODEL_VECTOR_FIELDS = (
    "log_epsilon_natural",
    "soln_293K",
    "soln25_298K",
    "sola",
    "solb",
    "solg",
    "solc",
    "solh",
)


class _Validator:
    def __init__(self, spec_dir: Path) -> None:
        self.spec_dir = spec_dir.resolve()
        self.repository_root = self.spec_dir.parent
        self.report = ValidationReport(spec_dir=str(self.spec_dir))
        self._tables: dict[str, list[dict[str, str]]] = {}

    def issue(self, code: str, message: str) -> None:
        self.report.issues.append(ValidationIssue(code=code, message=message))

    def require(self, condition: bool, code: str, message: str) -> None:
        if not condition:
            self.issue(code, message)

    def rows(self, filename: str) -> list[dict[str, str]]:
        if filename not in self._tables:
            self._tables[filename] = read_csv_rows(self.spec_dir / filename)
        return self._tables[filename]

    def check_count(self, filename: str, expected: int) -> None:
        actual = len(self.rows(filename))
        self.report.checks[f"rows:{filename}"] = actual
        self.require(
            actual == expected,
            "row_count",
            f"{filename}: expected {expected} data rows, found {actual}",
        )

    def check_unique(
        self,
        rows: Sequence[Mapping[str, str]],
        fields: Sequence[str],
        table: str,
    ) -> None:
        keys = [tuple(row[field] for field in fields) for row in rows]
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        self.require(
            not duplicates,
            "duplicate_key",
            f"{table}: duplicate key {tuple(fields)}: {duplicates[:5]}",
        )

    def validate_manifest(self) -> None:
        manifest = self.rows("package_manifest.csv")
        self.report.checks["manifest_entries"] = len(manifest)
        self.require(len(manifest) == 26, "manifest_count", "package manifest must contain 26 entries")
        self.check_unique(manifest, ("filename",), "package_manifest.csv")
        for row in manifest:
            filename = row["filename"]
            path = (
                self.repository_root / filename
                if filename == "AGENTS.md"
                else self.spec_dir / filename
            )
            if not path.is_file():
                self.issue("manifest_missing", f"manifest file is missing: {path}")
                continue
            actual_bytes = path.stat().st_size
            actual_hash = sha256_file(path)
            self.require(
                actual_bytes == int(row["bytes"]),
                "manifest_bytes",
                f"{filename}: expected {row['bytes']} bytes, found {actual_bytes}",
            )
            self.require(
                actual_hash == row["sha256"],
                "manifest_sha256",
                f"{filename}: SHA-256 mismatch",
            )
            if row["format"] == "csv":
                actual_rows = len(read_csv_rows(path))
                self.require(
                    actual_rows == int(row["rows"]),
                    "manifest_csv_rows",
                    f"{filename}: expected {row['rows']} rows, found {actual_rows}",
                )

    def validate_sources(self) -> None:
        sources = self.rows("source_registry.csv")
        self.check_unique(sources, ("source_id",), "source_registry.csv")
        known = {row["source_id"] for row in sources}
        referenced: set[str] = set()
        for path in sorted(self.spec_dir.glob("*.csv")):
            if path.name in {"source_registry.csv", "package_manifest.csv"}:
                continue
            columns = csv_fieldnames(path)
            source_columns = [
                column
                for column in columns
                if column == "source_id"
                or column.endswith("_source_id")
                or column.endswith("_source_ids")
            ]
            for row in self.rows(path.name):
                for column in source_columns:
                    referenced.update(
                        token.strip()
                        for token in row[column].split(";")
                        if token.strip().startswith("SRC_")
                    )
        missing = sorted(referenced - known)
        self.report.checks["source_ids_referenced"] = len(referenced)
        self.report.checks["source_ids_registered"] = len(known)
        self.require(not missing, "source_id_closure", f"unregistered source IDs: {missing}")

    def validate_solvents(self) -> tuple[set[str], dict[str, dict[str, str]]]:
        rows = self.rows("solvent_smd_registry.csv")
        self.check_unique(rows, ("solvent_id",), "solvent_smd_registry.csv")
        by_id = {row["solvent_id"]: row for row in rows}
        modes = Counter(row["orca_smd_mode"] for row in rows)
        self.report.checks["smd_mode_counts"] = dict(sorted(modes.items()))
        self.require(
            modes == Counter({"native_orca_smd": 14, "custom_smd": 11}),
            "smd_mode_counts",
            f"expected 14 native and 11 custom SMD rows, found {dict(modes)}",
        )
        expected_order = "log_epsilon|soln|soln25|sola|solb|solg|solc|solh"
        for row in rows:
            solvent_id = row["solvent_id"]
            try:
                epsilon = float(row["epsilon"])
                values = [float(row[field]) for field in MODEL_VECTOR_FIELDS]
                encoded = [float(value) for value in row["resolved_model_vector"].split("|")]
            except ValueError as exc:
                self.issue("smd_numeric", f"{solvent_id}: non-numeric SMD field: {exc}")
                continue
            self.require(len(encoded) == 8, "smd_vector_length", f"{solvent_id}: model vector is not 8-D")
            self.require(
                len(encoded) == len(values)
                and all(math.isclose(a, b, rel_tol=0.0, abs_tol=5e-11) for a, b in zip(encoded, values)),
                "smd_vector_values",
                f"{solvent_id}: resolved vector does not match scalar fields",
            )
            self.require(
                math.isclose(values[0], math.log(epsilon), rel_tol=0.0, abs_tol=5e-11),
                "smd_log_epsilon",
                f"{solvent_id}: log_epsilon_natural != ln(epsilon)",
            )
            self.require(
                epsilon > 1
                and 1 < values[1] < 2
                and 1 < values[2] < 2
                and all(value >= 0 for value in values[3:6])
                and all(0 <= value <= 1 for value in values[6:]),
                "smd_ranges",
                f"{solvent_id}: resolved SMD field is outside its accepted range",
            )
            self.require(row["model_vector_order"] == expected_order, "smd_vector_order", f"{solvent_id}: wrong model vector order")
            self.require(row["numeric_vector_complete_in_this_file"] == "1", "smd_incomplete", f"{solvent_id}: vector is not marked complete")
            self.require(row["parameter_status"] == "resolved_and_frozen", "smd_status", f"{solvent_id}: parameter status is not frozen")
            self.require(bool(row["orca_parameter_payload_resolved"]), "smd_payload", f"{solvent_id}: resolved ORCA payload is empty")
        water = next((row for row in rows if row["solvent_name"].startswith("Water")), None)
        self.require(water is not None, "water_missing", "water row is missing")
        if water is not None:
            self.require(water["orca_smd_mode"] == "native_orca_smd", "water_mode", "water must execute with native SMD")
            self.require("SMD(Water)" in water["orca_smd_input_from_source"], "water_keyword", "water must use native SMD(Water)")
        return set(by_id), by_id

    @staticmethod
    def parse_stoichiometry(value: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for token in value.split(";"):
            state_id, coefficient = token.rsplit(":", 1)
            parsed = int(coefficient)
            if parsed == 0 or state_id in result:
                raise ValueError(f"invalid stoichiometric token: {token}")
            result[state_id] = parsed
        return result

    def validate_states_reactions(
        self, solvent_ids: set[str]
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, int]]]:
        states = self.rows("fullspace_state_registry.csv")
        reactions = self.rows("fullspace_reaction_registry.csv")
        self.check_unique(states, ("state_id",), "fullspace_state_registry.csv")
        self.check_unique(states, ("reference_geometry_key",), "fullspace_state_registry.csv")
        self.check_unique(reactions, ("reaction_id",), "fullspace_reaction_registry.csv")
        state_by_id = {row["state_id"]: row for row in states}
        reaction_by_id = {row["reaction_id"]: row for row in reactions}
        stoichiometries: dict[str, dict[str, int]] = {}
        for row in reactions:
            reaction_id = row["reaction_id"]
            try:
                stoichiometry = self.parse_stoichiometry(row["stoichiometry"])
            except (ValueError, TypeError) as exc:
                self.issue("stoichiometry", f"{reaction_id}: {exc}")
                continue
            stoichiometries[reaction_id] = stoichiometry
            missing = sorted(set(stoichiometry) - set(state_by_id))
            self.require(not missing, "state_id_closure", f"{reaction_id}: unknown states {missing}")
            self.require(row["internal_unit"] == "eV", "reaction_unit", f"{reaction_id}: internal unit must be eV")
        role_counts = Counter(row["role"] for row in reactions)
        expected_roles = Counter({"monomer": 100, "solvent": 25, "anion": 11, "monomer_sigma": 100})
        self.report.checks["reaction_role_counts"] = dict(sorted(role_counts.items()))
        self.require(role_counts == expected_roles, "reaction_roles", f"reaction role counts differ: {dict(role_counts)}")
        expected_predictions = (
            role_counts["monomer"] * len(solvent_ids)
            + role_counts["solvent"]
            + role_counts["anion"] * len(solvent_ids)
            + role_counts["monomer_sigma"] * len(solvent_ids)
        )
        self.report.checks["expected_fullspace_predictions"] = expected_predictions
        self.require(expected_predictions == 5300, "fullspace_count", f"expected 5300 inference rows, derived {expected_predictions}")
        return state_by_id, reaction_by_id, stoichiometries

    def validate_calibration(
        self,
        solvent_ids: set[str],
        reaction_by_id: Mapping[str, Mapping[str, str]],
    ) -> dict[tuple[str, str], str]:
        design = self.rows("calibration_tuple_design.csv")
        self.check_unique(design, ("reaction_id",), "calibration_tuple_design.csv")
        splits = Counter(row["split"] for row in design)
        self.report.checks["calibration_split_counts"] = dict(sorted(splits.items()))
        self.require(splits == Counter({"train": 60, "val": 14, "test": 14}), "split_counts", f"calibration split counts differ: {dict(splits)}")
        parent_splits: dict[str, set[str]] = defaultdict(set)
        assigned: dict[tuple[str, str], str] = {}
        for row in design:
            reaction_id = row["reaction_id"]
            self.require(reaction_id in reaction_by_id, "reaction_id_closure", f"calibration references unknown reaction {reaction_id}")
            parent_splits[row["parent_id"]].add(row["split"])
            listed = [item for item in row["solvent_ids"].split(";") if item]
            self.require(len(listed) == int(row["solvent_count"]), "solvent_count", f"{reaction_id}: solvent_count does not match solvent_ids")
            self.require(not (set(listed) - solvent_ids), "solvent_id_closure", f"{reaction_id}: unknown solvent IDs {sorted(set(listed)-solvent_ids)}")
            self.require(len(listed) == len(set(listed)), "duplicate_solvent", f"{reaction_id}: duplicate assigned solvent")
            for solvent_id in listed:
                assigned[(reaction_id, solvent_id)] = row["split"]
            registry_row = reaction_by_id.get(reaction_id)
            if registry_row is not None:
                for field in ("reaction_class", "role", "parent_id", "split", "anchor_type"):
                    self.require(row[field] == registry_row[field], "calibration_registry_drift", f"{reaction_id}: {field} differs between calibration and reaction registry")
                self.require(row["solvent_ids"] == registry_row["assigned_solvent_ids"], "calibration_registry_drift", f"{reaction_id}: assigned solvent IDs differ from reaction registry")
        leaked = {parent: sorted(values) for parent, values in parent_splits.items() if len(values) > 1}
        self.report.checks["parent_split_leakage"] = leaked
        self.require(not leaked, "parent_split_leakage", f"parents occur in multiple splits: {leaked}")
        anchors = {row["reaction_id"]: (row["split"], row["anchor_type"]) for row in design if row["anchor_type"].endswith("full25")}
        self.report.checks["full25_anchors"] = sorted(anchors)
        self.require(anchors == EXPECTED_FULL25_ANCHORS, "full25_anchors", f"full-25 anchor set differs: {anchors}")
        by_id = {row["reaction_id"]: row for row in design}
        for reaction_id in EXPECTED_FULL25_ANCHORS:
            row = by_id.get(reaction_id)
            if row is not None:
                self.require(int(row["solvent_count"]) == 25 and set(row["solvent_ids"].split(";")) == solvent_ids, "full25_coverage", f"{reaction_id}: not assigned all 25 media")
        sparse = by_id.get("RXN_SIG_M084")
        if sparse is not None:
            expected_sparse = {"S004", "S011", "S016", "S024"}
            self.require(int(sparse["solvent_count"]) == 4 and set(sparse["solvent_ids"].split(";")) == expected_sparse, "sigma_m084_assignment", "RXN_SIG_M084 must be the frozen four-medium sparse tuple")
        self.report.checks["calibration_reaction_medium_cells"] = len(assigned)
        self.require(len(assigned) == 403, "calibration_cells", f"expected 403 calibration reaction-medium cells, found {len(assigned)}")
        return assigned

    def validate_manifests(
        self,
        solvent_ids: set[str],
        solvent_by_id: Mapping[str, Mapping[str, str]],
        state_by_id: Mapping[str, Mapping[str, str]],
        reaction_by_id: Mapping[str, Mapping[str, str]],
        stoichiometries: Mapping[str, Mapping[str, int]],
        assigned_cells: Mapping[tuple[str, str], str],
    ) -> None:
        sp_rows = self.rows("sp_job_manifest.csv")
        optfreq_rows = self.rows("optfreq_job_manifest.csv")
        self.check_unique(sp_rows, ("job_id",), "sp_job_manifest.csv")
        self.check_unique(optfreq_rows, ("job_id",), "optfreq_job_manifest.csv")
        self.check_unique(
            sp_rows,
            ("job_class", "state_id", "solvent_id", "geometry_key", "workflow_revision", "method_id"),
            "sp_job_manifest.csv scientific key",
        )
        self.check_unique(
            optfreq_rows,
            ("state_id", "solvent_id", "start_geometry_key", "workflow_revision", "method_id"),
            "optfreq_job_manifest.csv scientific key",
        )
        sp_classes = Counter(row["job_class"] for row in sp_rows)
        self.report.checks["sp_job_classes"] = dict(sorted(sp_classes.items()))
        self.require(sp_classes == Counter({"smd_energy_sp": 705, "diagnostic_gas_sp": 30}), "sp_job_counts", f"SP job classes differ: {dict(sp_classes)}")
        sp_cells: set[tuple[str, str]] = set()
        reaction_sp_states: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in sp_rows:
            job_id = row["job_id"]
            state = state_by_id.get(row["state_id"])
            self.require(state is not None, "state_id_closure", f"{job_id}: unknown state {row['state_id']}")
            if state is None:
                continue
            self.require(row["formal_charge"] == state["formal_charge"], "state_charge_drift", f"{job_id}: charge differs from state registry")
            self.require(row["multiplicity"] == state["multiplicity"], "state_multiplicity_drift", f"{job_id}: multiplicity differs from state registry")
            geometry_solvent = row["paired_solvent_id"] if row["job_class"] == "diagnostic_gas_sp" else row["solvent_id"]
            expected_geometry_key = state["reference_geometry_key"].format(solvent_id=geometry_solvent)
            self.require(row["geometry_key"] == expected_geometry_key, "geometry_key_drift", f"{job_id}: geometry key differs from state registry template")
            if row["job_class"] == "smd_energy_sp":
                solvent = solvent_by_id.get(row["solvent_id"])
                self.require(solvent is not None, "solvent_id_closure", f"{job_id}: unknown solvent {row['solvent_id']}")
                if solvent is not None:
                    self.require(row["smd_mode"] == solvent["orca_smd_mode"], "smd_mode_drift", f"{job_id}: SMD mode differs from registry")
                    self.require(row["smd_parameter_temperature_K"] == solvent["smd_parameter_temperature_K"], "smd_temperature_drift", f"{job_id}: SMD temperature differs from registry")
                self.require(row["paired_solvent_id"] == row["solvent_id"], "paired_solvent", f"{job_id}: SMD paired solvent differs")
                sp_cells.add((row["state_id"], row["solvent_id"]))
                for reaction_id in row["reaction_ids"].split(";"):
                    self.require(reaction_id in reaction_by_id, "reaction_id_closure", f"{job_id}: unknown reaction {reaction_id}")
                    reaction_sp_states[(reaction_id, row["solvent_id"])].add(row["state_id"])
            else:
                self.require(row["solvent_id"] == "GAS", "gas_solvent", f"{job_id}: diagnostic gas job solvent_id must be GAS")
                self.require(row["paired_solvent_id"] in solvent_ids, "paired_solvent", f"{job_id}: unknown paired solvent")
                self.require(row["reaction_ids"] == "DIAGNOSTIC_GAS", "gas_reaction", f"{job_id}: gas reaction marker differs")
        missing_sp_tuples = []
        for cell in assigned_cells:
            reaction_id, solvent_id = cell
            required = set(stoichiometries[reaction_id])
            present = reaction_sp_states.get(cell, set())
            if not required <= present:
                missing_sp_tuples.append((reaction_id, solvent_id, sorted(required - present)))
        self.report.checks["complete_sp_reaction_medium_cells"] = len(assigned_cells) - len(missing_sp_tuples)
        self.require(not missing_sp_tuples, "incomplete_sp_tuple", f"incomplete SP reaction tuples: {missing_sp_tuples[:5]}")

        optfreq_states: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in optfreq_rows:
            job_id = row["job_id"]
            state = state_by_id.get(row["state_id"])
            solvent = solvent_by_id.get(row["solvent_id"])
            self.require(state is not None, "state_id_closure", f"{job_id}: unknown state {row['state_id']}")
            self.require(solvent is not None, "solvent_id_closure", f"{job_id}: unknown solvent {row['solvent_id']}")
            if state is None or solvent is None:
                continue
            self.require((row["state_id"], row["solvent_id"]) in sp_cells, "optfreq_sp_coverage", f"{job_id}: no matching fixed-geometry SMD SP cell")
            self.require(row["formal_charge"] == state["formal_charge"], "state_charge_drift", f"{job_id}: charge differs from state registry")
            self.require(row["multiplicity"] == state["multiplicity"], "state_multiplicity_drift", f"{job_id}: multiplicity differs from state registry")
            expected_geometry_key = state["reference_geometry_key"].format(solvent_id=row["solvent_id"])
            self.require(row["start_geometry_key"] == expected_geometry_key, "geometry_key_drift", f"{job_id}: start geometry key differs from state registry template")
            expected_basis = "ma-def2-TZVP" if int(row["formal_charge"]) < 0 else "def2-TZVP"
            self.require(row["optfreq_basis"] == expected_basis, "optfreq_basis", f"{job_id}: basis violates charge rule")
            self.require(row["final_sp_basis"] == "def2-TZVPD", "final_sp_basis", f"{job_id}: final SP basis must be def2-TZVPD")
            self.require(row["smd_parameter_temperature_K"] == solvent["smd_parameter_temperature_K"], "smd_temperature_drift", f"{job_id}: SMD temperature differs from registry")
            for reaction_id in row["reaction_ids"].split(";"):
                self.require(reaction_id in reaction_by_id, "reaction_id_closure", f"{job_id}: unknown reaction {reaction_id}")
                optfreq_states[(reaction_id, row["solvent_id"])].add(row["state_id"])
        complete_final_cells = {
            cell
            for cell, present in optfreq_states.items()
            if cell[0] in stoichiometries and set(stoichiometries[cell[0]]) <= present
        }
        self.report.checks["complete_optfreq_reaction_medium_cells"] = len(complete_final_cells)
        self.require(len(complete_final_cells) == 50, "optfreq_complete_labels", f"expected 50 complete Opt/Freq reaction-medium labels, found {len(complete_final_cells)}")
        sp_cost = sum(Decimal(row["planning_core_h"]) for row in sp_rows)
        optfreq_cost = sum(Decimal(row["planning_core_h"]) for row in optfreq_rows)
        self.report.checks["planned_core_hours"] = {
            "sp": str(sp_cost.quantize(Decimal("0.01"))),
            "optfreq": str(optfreq_cost.quantize(Decimal("0.01"))),
            "total": str((sp_cost + optfreq_cost).quantize(Decimal("0.01"))),
        }
        self.require(sp_cost == Decimal("1601.30"), "sp_budget", f"SP planned cost is {sp_cost}, expected 1601.30")
        self.require(optfreq_cost == Decimal("2289.20"), "optfreq_budget", f"Opt/Freq planned cost is {optfreq_cost}, expected 2289.20")
        self.require(sp_cost + optfreq_cost == Decimal("3890.50"), "total_budget", f"initial planned cost is {sp_cost + optfreq_cost}, expected 3890.50")

    def validate_budget_and_training(self) -> None:
        budget = self.rows("compute_budget.csv")
        by_phase = {row["phase"]: row for row in budget}
        self.require(by_phase.get("initial_manifest", {}).get("planned_core_h") == "3890.50", "budget_table", "initial manifest budget must be 3890.50 core-h")
        self.require(by_phase.get("first_round", {}).get("hard_stop_core_h") == "8000", "budget_guard", "first-round hard stop must be 8000 core-h")
        self.require(by_phase.get("whole_project", {}).get("project_cap_core_h") == "12000", "budget_cap", "whole-project cap must be 12000 core-h")
        training = self.rows("training_config.csv")
        self.check_unique(training, ("category", "key"), "training_config.csv")
        config = {(row["category"], row["key"]): row["value"] for row in training}
        required = {
            ("environment", "mace_torch_version"): "0.3.16",
            ("environment", "mace_source_commit"): "4d2da09413ac1407f37cdbb6b81fa28e4c15655e",
            ("foundation", "checkpoint"): "polar-1-l",
            ("foundation", "default_dtype"): "float64",
            ("training", "head_warmup_epochs"): "50",
            ("training", "lora_rank"): "4",
            ("training", "lora_alpha"): "1.0",
            ("training", "seeds"): "17;29;43;71;101",
            ("budget", "first_round_hard_stop"): "8000",
            ("budget", "project_total_cap"): "12000",
            ("output", "expected_fullspace_rows"): "5300",
        }
        for key, expected in required.items():
            self.require(config.get(key) == expected, "training_config", f"{key}: expected {expected!r}, found {config.get(key)!r}")

    def run(self) -> ValidationReport:
        try:
            self.validate_manifest()
            for filename, expected in EXPECTED_COUNTS.items():
                self.check_count(filename, expected)
            self.validate_sources()
            solvent_ids, solvent_by_id = self.validate_solvents()
            state_by_id, reaction_by_id, stoichiometries = self.validate_states_reactions(solvent_ids)
            assigned = self.validate_calibration(solvent_ids, reaction_by_id)
            self.validate_manifests(
                solvent_ids,
                solvent_by_id,
                state_by_id,
                reaction_by_id,
                stoichiometries,
                assigned,
            )
            self.validate_budget_and_training()
        except (OSError, SchemaError, KeyError, ValueError, csv.Error) as exc:
            self.issue("validation_exception", f"{type(exc).__name__}: {exc}")
        self.report.checks = dict(sorted(self.report.checks.items()))
        return self.report


def validate_spec(spec_dir: Path) -> ValidationReport:
    """Validate bundle integrity, table closure, and frozen scientific invariants."""

    return _Validator(spec_dir).run()
