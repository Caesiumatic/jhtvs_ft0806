from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem


EXPECTED_VALIDATION_COUNTS = {"monomer": 9, "solvent": 7, "anion": 5}
STATE_MATRIX = {
    "monomer": {"lower_charge": 0, "lower_spin": 1, "oxidized_charge": 1, "oxidized_spin": 2},
    "solvent": {"lower_charge": 0, "lower_spin": 1, "oxidized_charge": 1, "oxidized_spin": 2},
    "anion": {"lower_charge": -1, "lower_spin": 1, "oxidized_charge": 0, "oxidized_spin": 2},
}

MANIFEST_FIELDS = (
    "system_id",
    "class",
    "species_id",
    "species_name",
    "canonical_smiles",
    "solvent_id",
    "solvent_name",
    "solvent_canonical_smiles",
    "canonical_key",
    "lower_charge",
    "lower_spin",
    "oxidized_charge",
    "oxidized_spin",
    "shell_seed_ids",
    "conformer_seed",
    "source_records",
)

OBSERVATION_FIELDS = (
    "observation_id",
    "system_id",
    "class",
    "species_id",
    "experimental_value_V_vs_AgAgCl",
    "protocol_id",
    "source_citation",
    "source_url",
)

_ANION_SMILES = {
    "CE03": "[O-][Cl+3]([O-])([O-])[O-]",
    "CE04": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "JVA002": "[O-][Cl+3]([O-])([O-])[O-]",
    "JVA004": "O=S(=O)([O-])C(F)(F)F",
    "JVA005": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
}

_SOLVENT_NAME_ALIASES = {
    "Acetonitrile": "Acetonitrile (MeCN)",
    "Propylene carbonate": "Propylene carbonate (PC)",
    "γ-Butyrolactone": "γ-Butyrolactone (GBL)",
    "N,N-Dimethylformamide": "N,N-Dimethylformamide (DMF)",
    "Nitromethane": "Nitromethane",
    "Sulfolane": "Sulfolane",
    "Dimethyl sulfoxide": "Dimethyl sulfoxide (DMSO)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def formal_charge(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _seed(label: str, index: int) -> int:
    raw = hashlib.sha256(f"mace-polar-5solv-v1|{label}|{index}".encode()).digest()
    return 1_000_000 + int.from_bytes(raw[:4], "big") % 8_000_000


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold().replace("α", "alpha").replace("γ", "gamma"))


def _decimal_text(value: str) -> str:
    return format(Decimal(value).normalize(), "f")


def _name_smiles_registry(path: Path) -> dict[str, tuple[str, str]]:
    rows = _read_csv(path)
    registry: dict[str, tuple[str, str]] = {}
    for row in rows:
        name = row["input"]
        registry[_normalized_name(name)] = (name, canonical_smiles(row["smiles"]))
    return registry


def _lookup_name(registry: Mapping[str, tuple[str, str]], name: str) -> tuple[str, str]:
    key = _normalized_name(name)
    if key in registry:
        return registry[key]
    matches = [value for candidate, value in registry.items() if key in candidate or candidate in key]
    if len(matches) != 1:
        raise ValueError(f"name does not resolve uniquely in registry: {name!r}")
    return matches[0]


def _system_row(
    *,
    class_name: str,
    species_id: str,
    species_name: str,
    species_smiles: str,
    solvent_id: str,
    solvent_name: str,
    solvent_smiles: str,
    source_records: Sequence[str],
    scope: str = "val",
) -> dict[str, object]:
    species_canonical = canonical_smiles(species_smiles)
    solvent_canonical = canonical_smiles(solvent_smiles)
    expected_charge = -1 if class_name == "anion" else 0
    if formal_charge(species_canonical) != expected_charge:
        raise ValueError(f"charge mismatch for {species_id}: expected {expected_charge}")
    canonical_key = f"{class_name}|{species_canonical}|{solvent_canonical}"
    system_id = f"{scope}-{class_name[:3]}-{species_id.lower()}-{hashlib.sha256(canonical_key.encode()).hexdigest()[:8]}"
    state = STATE_MATRIX[class_name]
    return {
        "system_id": system_id,
        "class": class_name,
        "species_id": species_id,
        "species_name": species_name,
        "canonical_smiles": species_canonical,
        "solvent_id": solvent_id,
        "solvent_name": solvent_name,
        "solvent_canonical_smiles": solvent_canonical,
        "canonical_key": canonical_key,
        **state,
        "shell_seed_ids": ";".join(str(_seed(canonical_key, index)) for index in range(5)),
        "conformer_seed": _seed(canonical_key, 50),
        "source_records": ";".join(sorted(source_records)),
    }


def build_validation(
    *,
    legacy_audit: Path,
    primary_core: Path,
    calibration_registry: Path,
    monomer_registry: Path,
    solvent_registry: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    legacy = _read_csv(legacy_audit)
    primary = _read_csv(primary_core)
    calibration = _read_csv(calibration_registry)
    monomers = _name_smiles_registry(monomer_registry)
    solvents = _name_smiles_registry(solvent_registry)
    by_id = {row["species_id"]: row for row in calibration if row["species_id"] and row["species_smiles"]}

    systems: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []

    monomer_rows = [
        row for row in legacy if row["property"] == "Monomer Eox" and row["included"].casefold() == "yes"
    ]
    for row in monomer_rows:
        identity = by_id.get(row["species_id"])
        if identity is None:
            _, smiles = _lookup_name(monomers, row["species"])
        else:
            smiles = identity["species_smiles"]
        solvent_name, solvent_smiles = _lookup_name(solvents, row["environment"])
        system = _system_row(
            class_name="monomer",
            species_id=row["species_id"],
            species_name=row["species"],
            species_smiles=smiles,
            solvent_id=solvent_name,
            solvent_name=solvent_name,
            solvent_smiles=solvent_smiles,
            source_records=[row["species_id"]],
        )
        systems.append(system)
        observations.append(
            {
                "observation_id": row["species_id"],
                "system_id": system["system_id"],
                "class": "monomer",
                "species_id": row["species_id"],
                    "experimental_value_V_vs_AgAgCl": _decimal_text(row["experimental_value"]),
                "protocol_id": "legacy-monomer-validation",
                "source_citation": row["reference"],
                "source_url": row["source_url"],
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in primary:
        class_name = row["Domain"].casefold()
        if class_name not in {"solvent", "anion"} or row["Comparison_Type"] != "point":
            continue
        environment = row["Implicit_Environment"]
        grouped.setdefault((class_name, row["Tier2_ID"], environment), []).append(row)

    for (class_name, species_id, _), rows in sorted(grouped.items()):
        first = rows[0]
        if class_name == "solvent":
            source_name = _SOLVENT_NAME_ALIASES[first["Species"]]
            solvent_name, smiles = _lookup_name(solvents, source_name)
            species_smiles = smiles
            solvent_id = species_id
        else:
            species_smiles = _ANION_SMILES[species_id]
            solvent_name, smiles = _lookup_name(solvents, first["Implicit_Environment"])
            solvent_id = solvent_name
        system = _system_row(
            class_name=class_name,
            species_id=species_id,
            species_name=first["Species"],
            species_smiles=species_smiles,
            solvent_id=solvent_id,
            solvent_name=solvent_name,
            solvent_smiles=smiles,
            source_records=[row["Record_ID"] for row in rows],
        )
        systems.append(system)
        for row in rows:
            observations.append(
                {
                    "observation_id": row["Record_ID"],
                    "system_id": system["system_id"],
                    "class": class_name,
                    "species_id": species_id,
                    "experimental_value_V_vs_AgAgCl": _decimal_text(row["Experimental_AgAgCl_V"]),
                    "protocol_id": row["Protocol_ID"],
                    "source_citation": row["Source_Citation"],
                    "source_url": row["Source_URL"],
                }
            )

    systems.sort(key=lambda row: (str(row["class"]), str(row["species_id"]), str(row["canonical_key"])))
    observations.sort(key=lambda row: str(row["observation_id"]))
    validate_manifest(systems)
    return systems, observations


def validate_manifest(rows: Sequence[Mapping[str, object]]) -> None:
    counts = Counter(str(row["class"]) for row in rows)
    if dict(counts) != EXPECTED_VALIDATION_COUNTS:
        raise ValueError(f"validation class counts {dict(counts)} != {EXPECTED_VALIDATION_COUNTS}")
    if len(rows) != 21:
        raise ValueError(f"expected 21 validation systems, found {len(rows)}")
    keys = [str(row["canonical_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("validation canonical keys are not unique")
    for row in rows:
        class_name = str(row["class"])
        expected = STATE_MATRIX[class_name]
        for field, value in expected.items():
            if int(row[field]) != value:
                raise ValueError(f"state mismatch for {row['system_id']}: {field}")
        seeds = str(row["shell_seed_ids"]).split(";")
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError(f"invalid shell seeds for {row['system_id']}")


def build_and_write(args: argparse.Namespace) -> None:
    systems, observations = build_validation(
        legacy_audit=args.legacy_audit,
        primary_core=args.primary_core,
        calibration_registry=args.calibration_registry,
        monomer_registry=args.monomer_registry,
        solvent_registry=args.solvent_registry,
    )
    _write_csv(args.output / "validation_manifest.csv", MANIFEST_FIELDS, systems)
    _write_csv(args.output / "validation_observations.csv", OBSERVATION_FIELDS, observations)
    provenance = {
        "base_commit": args.base_commit,
        "sources": {
            "legacy_validation_points_audit.csv": sha256_file(args.legacy_audit),
            "primary_audited_v2_core_primary_eox.csv": sha256_file(args.primary_core),
            "calibration_registry.csv": sha256_file(args.calibration_registry),
            "source_fullspace_monomers.csv": sha256_file(args.monomer_registry),
            "source_fullspace_solvents.csv": sha256_file(args.solvent_registry),
        },
        "validation_counts": dict(Counter(str(row["class"]) for row in systems)),
        "validation_system_count": len(systems),
        "observation_count": len(observations),
    }
    (args.output / "manifest_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--legacy-audit", type=Path, required=True)
    result.add_argument("--primary-core", type=Path, required=True)
    result.add_argument("--calibration-registry", type=Path, required=True)
    result.add_argument("--monomer-registry", type=Path, required=True)
    result.add_argument("--solvent-registry", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--base-commit", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    build_and_write(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
