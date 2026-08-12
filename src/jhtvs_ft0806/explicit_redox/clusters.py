from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .packing import run_packmol


CLUSTER_FIELDS = (
    "system_id",
    "class",
    "seed_index",
    "shell_seed",
    "target_entity_id",
    "target_conformer_rank",
    "target_geometry_sha256",
    "solvent_entity_id",
    "solvent_geometry_sha256",
    "target_atoms",
    "solvent_atoms",
    "solvent_count",
    "containment_radius_A",
    "minimum_intermolecular_distance_A",
    "packmol_input_sha256",
    "packmol_log_sha256",
    "geometry_sha256",
    "geometry_path",
    "lower_charge",
    "lower_spin",
    "oxidized_charge",
    "oxidized_spin",
    "status",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _entity_by_smiles(raw_root: Path) -> dict[str, dict[str, str]]:
    return {row["canonical_smiles"]: row for row in _read(raw_root / "structure_candidates.csv")}


def _selected_by_entity(raw_root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in _read(raw_root / "isolated_selected.csv"):
        result.setdefault(row["entity_id"], []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return result


def pack_systems(
    *,
    manifest: Path,
    raw_root: Path,
    executable: str = "packmol",
    seed_limit: int = 5,
) -> list[dict[str, object]]:
    systems = _read(manifest)
    entities = _entity_by_smiles(raw_root)
    selected = _selected_by_entity(raw_root)
    rows: list[dict[str, object]] = []
    for system in systems:
        target_entity = entities[system["canonical_smiles"]]
        solvent_entity = entities[system["solvent_canonical_smiles"]]
        target_options = selected[target_entity["entity_id"]]
        solvent_geometry = selected[solvent_entity["entity_id"]][0]
        seeds = system["shell_seed_ids"].split(";")[:seed_limit]
        for seed_index, shell_seed in enumerate(seeds):
            target = target_options[seed_index % len(target_options)]
            output_dir = raw_root / "clusters" / system["system_id"] / f"seed-{seed_index}"
            if (output_dir / "cluster.json").is_file():
                payload = json.loads((output_dir / "cluster.json").read_text(encoding="utf-8"))
                rows.append(payload)
                continue
            packed = run_packmol(
                target_path=raw_root / target["optimized_xyz"],
                solvent_path=raw_root / solvent_geometry["optimized_xyz"],
                solvent_smiles=system["solvent_canonical_smiles"],
                seed=int(shell_seed),
                output_dir=output_dir,
                executable=executable,
            )
            payload = {
                "system_id": system["system_id"],
                "class": system["class"],
                "seed_index": seed_index,
                "shell_seed": shell_seed,
                "target_entity_id": target_entity["entity_id"],
                "target_conformer_rank": target["rank"],
                "target_geometry_sha256": target["geometry_sha256"],
                "solvent_entity_id": solvent_entity["entity_id"],
                "solvent_geometry_sha256": solvent_geometry["geometry_sha256"],
                "target_atoms": packed.target_atoms,
                "solvent_atoms": packed.solvent_atoms,
                "solvent_count": packed.solvent_count,
                "containment_radius_A": format(packed.containment_radius_A, ".8f"),
                "minimum_intermolecular_distance_A": format(
                    packed.minimum_intermolecular_distance_A, ".8f"
                ),
                "packmol_input_sha256": packed.input_sha256,
                "packmol_log_sha256": hashlib.sha256(packed.log_path.read_bytes()).hexdigest(),
                "geometry_sha256": packed.geometry_sha256,
                "geometry_path": packed.geometry_path.relative_to(raw_root).as_posix(),
                "lower_charge": system["lower_charge"],
                "lower_spin": system["lower_spin"],
                "oxidized_charge": system["oxidized_charge"],
                "oxidized_spin": system["oxidized_spin"],
                "status": "clean",
            }
            (output_dir / "cluster.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            rows.append(payload)
    output = raw_root / "cluster_manifest.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLUSTER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--packmol", default="packmol")
    parser.add_argument("--seed-limit", type=int, default=5)
    args = parser.parse_args(argv)
    rows = pack_systems(
        manifest=args.manifest,
        raw_root=args.raw_root,
        executable=args.packmol,
        seed_limit=args.seed_limit,
    )
    print(json.dumps({"status": "PASS", "clusters": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
