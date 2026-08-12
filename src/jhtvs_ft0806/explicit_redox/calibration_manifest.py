from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Mapping, Sequence

from .alignment import assert_disjoint_keys
from .manifest import (
    MANIFEST_FIELDS,
    _ANION_SMILES,
    _lookup_name,
    _name_smiles_registry,
    _read_csv,
    _system_row,
    _write_csv,
    canonical_smiles,
)


CALIBRATION_FIELDS = MANIFEST_FIELDS + ("experimental_value_V_vs_AgAgCl", "selection_reason")
AUDIT_FIELDS = (
    "registry_row",
    "class",
    "species_id",
    "species_name",
    "environment",
    "canonical_key",
    "experimental_value",
    "decision",
    "reason",
)

_MONOMER_ALIASES = {
    "EDOT": "3,4-Ethylenedioxythiophene (EDOT)",
    "3-methylthiophene": "3-Methylthiophene",
    "2,2′-bithiophene": "2,2′-Bithiophene",
    "CPDT": "Cyclopenta[2,1-b:3,4-b′]dithiophene (CPDT)",
    "Terthiophene": "2,2′:5′,2″-Terthiophene",
    "Nmethylpyrrole": "N-Methylpyrrole",
}


def _scalar(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip()))


def _value_supported(value: float, original: str, conversion: str) -> bool:
    numbers = [float(item) for item in re.findall(r"(?<![A-Za-z])([+-]?\d+(?:\.\d+)?)", original + " " + conversion)]
    return any(abs(candidate - value) <= 0.055 for candidate in numbers)


def _resolve_species(
    row: Mapping[str, str], monomers: Mapping[str, tuple[str, str]]
) -> str:
    class_name = row["channel"].split("_", maxsplit=1)[0]
    if class_name == "monomer":
        name = _MONOMER_ALIASES.get(row["species"], row["species"])
        return _lookup_name(monomers, name)[1]
    if class_name == "solvent":
        return canonical_smiles(row["solvent_smiles"])
    if row["species_id"] in _ANION_SMILES:
        return canonical_smiles(_ANION_SMILES[row["species_id"]])
    anion_by_id = {
        "CE02": "F[B-](F)(F)F",
        "CE12": "O=S(=O)([O-])C(F)(F)F",
    }
    return canonical_smiles(anion_by_id[row["species_id"]])


def build_calibration(
    *,
    calibration_registry: Path,
    validation_manifest: Path,
    monomer_registry: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registry = _read_csv(calibration_registry)
    validation = _read_csv(validation_manifest)
    validation_keys = [row["canonical_key"] for row in validation]
    monomers = _name_smiles_registry(monomer_registry)
    included: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, row in enumerate(registry, start=2):
        if row["source_space"] != "calibration" or row["channel"] not in {
            "monomer_eox",
            "solvent_eox",
            "anion_eox",
        }:
            continue
        class_name = row["channel"].split("_", maxsplit=1)[0]
        value = row["experimental_or_reference_value_context_only"].strip()
        reason = "included_objective_scalar"
        decision = "include"
        canonical_key = ""
        species_smiles = ""
        if not _scalar(value):
            decision, reason = "exclude", "non_scalar_or_missing_experiment"
        elif not row["reference_not_fit"].strip() or not row["original_literature_data_not_fit"].strip():
            decision, reason = "exclude", "insufficient_source_provenance"
        else:
            text = " ".join(
                (
                    row["original_literature_data_not_fit"],
                    row["conversion_or_normalization_not_fit"],
                    row["reference_not_fit"],
                )
            ).casefold()
            if any(
                marker in text
                for marker in (
                    "no defensible conversion",
                    "not comparable",
                    "different process",
                    "total potential window",
                    "full potential window",
                    "lower bound",
                    "secondary citation",
                    "patent",
                    "not verified here",
                )
            ):
                decision, reason = "exclude", "incompatible_or_insufficient_observable"
            elif "propylene carbonate" in text and row["solvent_or_environment"] == "Acetonitrile":
                decision, reason = "exclude", "experimental_medium_mismatch"
            elif "existing 1.50 is onset-scale" in row["conversion_or_normalization_not_fit"]:
                decision, reason = "exclude", "registry_value_provenance_conflict"
            elif not re.search(r"(?:19|20)\d{2}", row["reference_not_fit"]):
                decision, reason = "exclude", "insufficient_source_provenance"
            elif not _value_supported(
                float(value),
                row["original_literature_data_not_fit"],
                row["conversion_or_normalization_not_fit"],
            ):
                decision, reason = "exclude", "registry_value_not_supported_by_cited_normalization"
        try:
            species_smiles = _resolve_species(row, monomers)
            solvent_smiles = canonical_smiles(row["solvent_smiles"])
            canonical_key = f"{class_name}|{species_smiles}|{solvent_smiles}"
        except (KeyError, ValueError):
            if decision == "include":
                decision, reason = "exclude", "canonical_identity_unresolved"
        if decision == "include" and canonical_key in validation_keys:
            decision, reason = "exclude", "validation_key_overlap"
        if decision == "include" and canonical_key in seen:
            decision, reason = "exclude", "duplicate_calibration_system"
        if decision == "include":
            seen.add(canonical_key)
            system = _system_row(
                class_name=class_name,
                species_id=row["species_id"],
                species_name=row["species"],
                species_smiles=species_smiles,
                solvent_id=row["solvent_or_environment"],
                solvent_name=row["solvent_or_environment"],
                solvent_smiles=row["solvent_smiles"],
                source_records=[f"calib_data.csv:{index}"],
                scope="cal",
            )
            system["experimental_value_V_vs_AgAgCl"] = value
            system["selection_reason"] = reason
            included.append(system)
        audit.append(
            {
                "registry_row": index,
                "class": class_name,
                "species_id": row["species_id"],
                "species_name": row["species"],
                "environment": row["solvent_or_environment"],
                "canonical_key": canonical_key,
                "experimental_value": value,
                "decision": decision,
                "reason": reason,
            }
        )
    assert_disjoint_keys([str(row["canonical_key"]) for row in included], validation_keys)
    included.sort(key=lambda row: str(row["canonical_key"]))
    return included, audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-registry", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--monomer-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    included, audit = build_calibration(
        calibration_registry=args.calibration_registry,
        validation_manifest=args.validation_manifest,
        monomer_registry=args.monomer_registry,
    )
    _write_csv(args.output / "calibration_manifest.csv", CALIBRATION_FIELDS, included)
    _write_csv(args.output / "calibration_audit.csv", AUDIT_FIELDS, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
