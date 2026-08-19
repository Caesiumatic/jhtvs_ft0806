from __future__ import annotations

from pathlib import Path

try:
    from .common import (
        BIAS_ALPHA_ANGSTROM_INV,
        LIBXC_CORRELATION,
        LIBXC_EXCHANGE,
        atom_distance,
        load_metadata,
        orca_bias_depth_kcal_mol,
        read_xyz,
    )
except ImportError:
    from common import (
        BIAS_ALPHA_ANGSTROM_INV,
        LIBXC_CORRELATION,
        LIBXC_EXCHANGE,
        atom_distance,
        load_metadata,
        orca_bias_depth_kcal_mol,
        read_xyz,
    )


def bias_payload(row: dict[str, str], xyz_path: Path) -> list[dict[str, float | int]]:
    if row["restraint"] == "none":
        return []
    if row["restraint"] != "two_adjacent_anchor_distances":
        raise ValueError(f"unsupported restraint: {row['restraint']}")
    metadata = load_metadata(xyz_path)
    atoms, _ = read_xyz(xyz_path)
    anchors = metadata["anchor_indices_zero_based"]
    topology = row["topology"]
    if sorted(topology) != ["A", "C", "S"]:
        raise ValueError(f"invalid triad topology: {topology}")
    force = float(row["restraint_force_constant_eh_bohr2"])
    depth = orca_bias_depth_kcal_mol(force)
    output = []
    for left, right in zip(topology, topology[1:]):
        left_index, right_index = int(anchors[left]), int(anchors[right])
        output.append({
            "left_index": left_index,
            "right_index": right_index,
            "reference_distance_ang": atom_distance(atoms[left_index], atoms[right_index]),
            "depth_kcal_mol": depth,
            "alpha_angstrom_inv": BIAS_ALPHA_ANGSTROM_INV,
        })
    return output


def build_input(row: dict[str, str], state: str, xyz_path: Path, nprocs: int = 8, maxcore_mb: int = 3000) -> str:
    if state not in {"reduced_opt", "reduced_sp", "oxidized_sp"}:
        raise ValueError(f"unknown state: {state}")
    optimize = state == "reduced_opt"
    reduced = state != "oxidized_sp"
    charge = row["charge_reduced"] if reduced else row["charge_oxidized"]
    multiplicity = row["multiplicity_reduced"] if reduced else row["multiplicity_oxidized"]
    keywords = ["aug-cc-pVTZ", "RIJCOSX", "AutoAux", "TightSCF", "DEFGRID3", "SlowConv", "MULLIKEN"]
    if optimize:
        keywords.append("Opt")
    lines = [
        f"# task_id: {row['task_id']}",
        f"# state: {state}",
        f"# method_id: {row['method_id']}",
        "# M06-HF is supplied by LibXC as its published exchange and correlation components.",
        f"! {' '.join(keywords)}",
        "%method",
        "  Method DFT",
        f"  Exchange {LIBXC_EXCHANGE}",
        f"  Correlation {LIBXC_CORRELATION}",
        "end",
        f"%pal nprocs {nprocs} end",
        f"%maxcore {maxcore_mb}",
        "%scf",
        "  MaxIter 500",
        "  AutoTRAH true",
        "end",
        "%output",
        "  Print[P_AtCharges_M] 1",
        "end",
    ]
    if row["benchmark"] == "chauhan":
        lines.extend([
            "%cpcm",
            f"  epsilon {float(row['epsilon']):.8f}",
            "  fepstype cpcm",
            "  surfacetype vdw_gaussian",
            "end",
        ])
    elif row["benchmark"] != "fadel" or row["environment"] != "vacuum":
        raise ValueError("invalid benchmark environment")
    if optimize:
        biases = bias_payload(row, xyz_path)
        lines.extend(["%geom", "  MaxIter 300"])
        if biases:
            lines.append("  BIAS")
            for bias in biases:
                lines.append(
                    "    { B %d %d %.10f %.10f %.8f }"
                    % (
                        bias["left_index"], bias["right_index"], bias["reference_distance_ang"],
                        bias["depth_kcal_mol"], bias["alpha_angstrom_inv"],
                    )
                )
            lines.append("  END")
        lines.append("end")
    lines.extend([f"* xyzfile {charge} {multiplicity} in.xyz", ""])
    text = "\n".join(lines)
    if state != "reduced_opt" and "%geom" in text:
        raise AssertionError("geometry bias leaked into single-point input")
    if row["benchmark"] == "fadel" and "%cpcm" in text:
        raise AssertionError("solvation leaked into Fadel vacuum input")
    return text
