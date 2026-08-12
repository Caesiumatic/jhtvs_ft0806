from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from jhtvs_ft0806.geometry.xyz import XYZAtom, inferred_bonds

from .analysis import _gaps
from .marcus import block_statistics
from .trajectory import _read_tsv


def _fragment_bonds(symbols: Sequence[str], positions: np.ndarray) -> set[frozenset[int]]:
    atoms = tuple(
        XYZAtom(symbol, float(position[0]), float(position[1]), float(position[2]))
        for symbol, position in zip(symbols, positions, strict=True)
    )
    return inferred_bonds(atoms)


def connectivity_changes(
    *,
    initial_symbols: Sequence[str],
    initial_positions: np.ndarray,
    final_symbols: Sequence[str],
    final_positions: np.ndarray,
    target_atoms: int,
    solvent_atoms: int,
) -> dict[str, int]:
    if tuple(initial_symbols) != tuple(final_symbols):
        raise RuntimeError("trajectory atom order or composition changed")
    groups = [(0, target_atoms)] + [
        (target_atoms + index * solvent_atoms, target_atoms + (index + 1) * solvent_atoms)
        for index in range(5)
    ]
    if groups[-1][1] != len(initial_symbols):
        raise RuntimeError("QC atom grouping does not close")
    changes = {
        "target_bonds_broken": 0,
        "target_bonds_formed": 0,
        "solvent_bonds_broken": 0,
        "solvent_bonds_formed": 0,
    }
    for group_index, (start, stop) in enumerate(groups):
        initial = _fragment_bonds(initial_symbols[start:stop], initial_positions[start:stop])
        final = _fragment_bonds(final_symbols[start:stop], final_positions[start:stop])
        prefix = "target" if group_index == 0 else "solvent"
        changes[f"{prefix}_bonds_broken"] += len(initial - final)
        changes[f"{prefix}_bonds_formed"] += len(final - initial)
    return changes


def final_shell_distances_A(atoms: Any, *, target_atoms: int, solvent_atoms: int) -> list[float]:
    symbols = atoms.get_chemical_symbols()
    target_heavy = [index for index, symbol in enumerate(symbols[:target_atoms]) if symbol != "H"]
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    centroid = positions[target_heavy].mean(axis=0)
    distances = []
    for index in range(5):
        start = target_atoms + index * solvent_atoms
        stop = start + solvent_atoms
        com = np.average(positions[start:stop], axis=0, weights=masses[start:stop])
        distances.append(float(np.linalg.norm(com - centroid)))
    return distances


def localization_diagnostics(paths: Sequence[Path], *, target_atoms: int) -> dict[str, float]:
    target_density = solvent_density = target_spin = solvent_spin = 0.0
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            lower_density = np.asarray(payload["lower_density_coefficients"], dtype=np.float64)
            oxidized_density = np.asarray(payload["oxidized_density_coefficients"], dtype=np.float64)
            oxidized_spin = np.asarray(payload["oxidized_spin_density"], dtype=np.float64)
        density_change = np.abs(oxidized_density[:, :, 0] - lower_density[:, :, 0])
        spin_density = np.abs(oxidized_spin[:, :, 0])
        target_density += float(density_change[:, :target_atoms].sum())
        solvent_density += float(density_change[:, target_atoms:].sum())
        target_spin += float(spin_density[:, :target_atoms].sum())
        solvent_spin += float(spin_density[:, target_atoms:].sum())
    density_total = target_density + solvent_density
    spin_total = target_spin + solvent_spin
    return {
        "target_density_change_fraction": target_density / density_total if density_total else 0.5,
        "target_spin_density_fraction": target_spin / spin_total if spin_total else 0.5,
    }


def localization_flag(diagnostics: dict[str, float], *, connectivity_clean: bool) -> str:
    if not connectivity_clean:
        return "localization_ambiguous"
    density_target = diagnostics["target_density_change_fraction"] > 0.5
    spin_target = diagnostics["target_spin_density_fraction"] > 0.5
    if density_target and spin_target:
        return "oxidation_localization_target"
    if not density_target and not spin_target:
        return "oxidation_localization_solvent_competing"
    return "localization_ambiguous"


def collect_trajectory_qc(*, raw_root: Path, mode: str) -> list[dict[str, Any]]:
    try:
        from ase.io import read
    except ImportError as exc:  # pragma: no cover - execution dependency
        raise RuntimeError("ASE is required for trajectory QC") from exc
    rows: list[dict[str, Any]] = []
    for task in _read_tsv(raw_root / f"{mode}_trajectory_tasks.tsv"):
        trajectory_dir = raw_root / "trajectories" / task["logical_trajectory_id"]
        trajectory_path = trajectory_dir / "trajectory.json"
        gap_path = raw_root / "gaps" / task["logical_trajectory_id"] / "gaps.json"
        if not trajectory_path.is_file() or not gap_path.is_file():
            rows.append(
                {
                    "logical_trajectory_id": task["logical_trajectory_id"],
                    "system_id": task["system_id"],
                    "seed_index": task["seed_index"],
                    "state": task["state"],
                    "qc_status": "incomplete",
                    "flags": "incomplete_sampling",
                }
            )
            continue
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        gap = json.loads(gap_path.read_text(encoding="utf-8"))
        initial = read(raw_root / task["cluster_geometry_path"])
        final = read(trajectory_dir / "md" / "latest.extxyz")
        changes = connectivity_changes(
            initial_symbols=initial.get_chemical_symbols(),
            initial_positions=np.asarray(initial.positions),
            final_symbols=final.get_chemical_symbols(),
            final_positions=np.asarray(final.positions),
            target_atoms=int(task["target_atoms"]),
            solvent_atoms=int(task["solvent_atoms"]),
        )
        flags: list[str] = []
        if changes["target_bonds_broken"] or changes["target_bonds_formed"]:
            flags.append("target_fragmented")
        if changes["solvent_bonds_broken"] or changes["solvent_bonds_formed"]:
            flags.append("solvent_fragmented")
        md = trajectory["md"]
        complete_sampling = (
            trajectory["status"] == "complete"
            and gap["status"] == "complete"
            and int(gap["sample_count"]) == int(md["expected_production_samples"])
        )
        if not complete_sampling:
            flags.append("incomplete_sampling")
        shell_distances = final_shell_distances_A(
            final,
            target_atoms=int(task["target_atoms"]),
            solvent_atoms=int(task["solvent_atoms"]),
        )
        shell_retained = max(shell_distances) <= float(task["R0_A"])
        if not shell_retained:
            flags.append("shell_escape")
        gap_values = _gaps(raw_root, gap)
        if mode == "production":
            stats = block_statistics(gap_values)
            if stats.multimodal:
                flags.append("multimodal_gap")
            gap_stats = {
                "gap_mean_eV": stats.mean_eV,
                "gap_sd_eV": stats.standard_deviation_eV,
                "gap_skewness": stats.skewness,
                "gap_block_se_eV": stats.block_standard_error_eV,
                "gap_multimodal": stats.multimodal,
            }
        else:
            centered = gap_values - gap_values.mean()
            scale = float(np.sqrt(np.mean(centered**2)))
            gap_stats = {
                "gap_mean_eV": float(gap_values.mean()),
                "gap_sd_eV": float(gap_values.std(ddof=1)) if gap_values.size > 1 else 0.0,
                "gap_skewness": float(np.mean(centered**3) / scale**3) if scale else 0.0,
                "gap_block_se_eV": "",
                "gap_multimodal": False,
            }
        diagnostics = localization_diagnostics(
            [raw_root / chunk["path"] for chunk in gap["gap_chunks"]],
            target_atoms=int(task["target_atoms"]),
        )
        localization = localization_flag(diagnostics, connectivity_clean=not any(
            flag in flags for flag in ("target_fragmented", "solvent_fragmented")
        ))
        rows.append(
            {
                "logical_trajectory_id": task["logical_trajectory_id"],
                "system_id": task["system_id"],
                "seed_index": int(task["seed_index"]),
                "state": task["state"],
                "qc_status": "clean" if not flags else "flagged",
                "flags": ";".join(flags) if flags else "clean",
                **changes,
                "shell_retained_at_final": shell_retained,
                "final_maximum_solvent_com_distance_A": max(shell_distances),
                "restraint_activation_samples": md["restraint_activation_samples"],
                "maximum_excursion_A": md["maximum_excursion_A"],
                "temperature_mean_K": md["temperature_mean_K"],
                "temperature_sd_K": md["temperature_sd_K"],
                "temperature_min_K": md["temperature_min_K"],
                "temperature_max_K": md["temperature_max_K"],
                "maximum_force_eV_A": md["maximum_force_eV_A"],
                "maximum_absolute_chunk_energy_change_eV_ps": md[
                    "maximum_absolute_chunk_energy_change_eV_ps"
                ],
                "expected_samples": md["expected_production_samples"],
                "completed_samples": gap["sample_count"],
                **gap_stats,
                **diagnostics,
                "localization_flag": localization,
            }
        )
    output = raw_root / f"{mode}_trajectory_qc.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "production"), required=True)
    args = parser.parse_args(argv)
    rows = collect_trajectory_qc(raw_root=args.raw_root, mode=args.mode)
    print(json.dumps({"status": "PASS", "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
