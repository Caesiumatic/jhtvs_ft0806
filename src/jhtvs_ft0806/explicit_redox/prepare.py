from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .manifest import formal_charge
from .structures import write_conformer_set


STRUCTURE_FIELDS = (
    "entity_id",
    "canonical_smiles",
    "formal_charge",
    "roles",
    "source_system_ids",
    "etkdg_seed",
    "requested_conformers",
    "deduplicated_conformers",
    "directory",
    "metadata_sha256",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_seed(smiles: str) -> int:
    return 1_000_000 + int.from_bytes(
        hashlib.sha256(f"mace-polar-5solv-v1|isolated|{smiles}".encode()).digest()[:4], "big"
    ) % 8_000_000


def collect_entities(manifests: Sequence[Path]) -> list[dict[str, object]]:
    by_smiles: dict[str, dict[str, object]] = {}
    for manifest in manifests:
        for row in _read(manifest):
            entries = (
                (row["canonical_smiles"], "target"),
                (row["solvent_canonical_smiles"], "solvent"),
            )
            for smiles, role in entries:
                record = by_smiles.setdefault(
                    smiles,
                    {"canonical_smiles": smiles, "roles": set(), "source_system_ids": set()},
                )
                record["roles"].add(role)  # type: ignore[union-attr]
                record["source_system_ids"].add(row["system_id"])  # type: ignore[union-attr]
    result = []
    for smiles, record in sorted(by_smiles.items()):
        entity_id = f"iso-{hashlib.sha256(smiles.encode()).hexdigest()[:12]}"
        result.append(
            {
                "entity_id": entity_id,
                "canonical_smiles": smiles,
                "formal_charge": formal_charge(smiles),
                "roles": ";".join(sorted(record["roles"])),  # type: ignore[arg-type]
                "source_system_ids": ";".join(sorted(record["source_system_ids"])),  # type: ignore[arg-type]
                "etkdg_seed": _stable_seed(smiles),
            }
        )
    return result


def build_structure_candidates(
    *, manifests: Sequence[Path], output_root: Path, conformer_count: int = 50
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entity in collect_entities(manifests):
        directory = output_root / "isolated" / str(entity["entity_id"])
        metadata = directory / "conformers.json"
        if metadata.is_file():
            records = json.loads(metadata.read_text(encoding="utf-8"))
        else:
            records = write_conformer_set(
                str(entity["canonical_smiles"]),
                species_id=str(entity["entity_id"]),
                seed=int(entity["etkdg_seed"]),
                output_dir=directory,
                count=conformer_count,
            )
        rows.append(
            {
                **entity,
                "requested_conformers": conformer_count,
                "deduplicated_conformers": len(records),
                "directory": directory.relative_to(output_root).as_posix(),
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            }
        )
    manifest_path = output_root / "structure_candidates.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def structure_summary(rows: Sequence[Mapping[str, object]], raw_root: Path) -> dict[str, object]:
    manifest = raw_root / "structure_candidates.csv"
    return {
        "status": "PASS",
        "entity_count": len(rows),
        "requested_conformer_count": sum(int(row["requested_conformers"]) for row in rows),
        "deduplicated_conformer_count": sum(int(row["deduplicated_conformers"]) for row in rows),
        "structure_candidates_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "entities": [
            {
                "entity_id": row["entity_id"],
                "canonical_smiles": row["canonical_smiles"],
                "formal_charge": row["formal_charge"],
                "etkdg_seed": row["etkdg_seed"],
                "deduplicated_conformers": row["deduplicated_conformers"],
                "metadata_sha256": row["metadata_sha256"],
            }
            for row in rows
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conformer-count", type=int, default=50)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args(argv)
    rows = build_structure_candidates(
        manifests=args.manifest,
        output_root=args.output_root,
        conformer_count=args.conformer_count,
    )
    summary = structure_summary(rows, args.output_root)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": "PASS", "entities": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
