from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from jhtvs_ft0806.geometry.resolution import (
    GeometryRequest,
    geometry_requests,
    prepare_and_resolve_sigma_requests,
    resolve_tier1_requests,
    sigma_reference_bonds,
)
from jhtvs_ft0806.geometry.sigma import build_sigma_complex, load_sigma_topologies
from jhtvs_ft0806.geometry.topology import molecule_from_smiles
from jhtvs_ft0806.geometry.xyz import check_connectivity
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import write_csv_deterministic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"


def _xyz_for_smiles(smiles: str) -> str:
    molecule = Chem.AddHs(molecule_from_smiles(smiles))
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    lines = [str(len(symbols)), "synthetic geometry-resolution fixture"]
    lines.extend(
        f"{symbol:<2s} {index * 2.0:18.10f} {0.0:18.10f} {0.0:18.10f}"
        for index, symbol in enumerate(symbols)
    )
    return "\n".join(lines) + "\n"


def test_manifest_geometry_requests_have_frozen_unique_coverage() -> None:
    requests = geometry_requests(SPEC_DIR)

    assert len(requests) == 705
    assert sum(not request.is_sigma for request in requests) == 583
    assert sum(request.is_sigma for request in requests) == 122
    assert len({request.state_id for request in requests if request.is_sigma}) == 20
    assert len({request.geometry_key for request in requests}) == len(requests)


def test_tier1_resolution_requires_clean_same_run_hash_and_composition(
    tmp_path: Path,
) -> None:
    tier1_run = tmp_path / "tier1-run"
    output_dir = tier1_run / "raw" / "redox" / "task-1"
    output_dir.mkdir(parents=True)
    source_xyz = output_dir / "xtbopt.xyz"
    source_xyz.write_text(_xyz_for_smiles("c1ccsc1"), encoding="utf-8")
    source_sha = sha256_file(source_xyz)
    (tier1_run / "manifests").mkdir()
    (tier1_run / "descriptors").mkdir()
    (tier1_run / "manifests" / "run_manifest.json").write_text(
        json.dumps({"run_id": "same-run-fixture"}), encoding="utf-8"
    )
    write_csv_deterministic(
        tier1_run / "manifests" / "redox_tasks.csv",
        (
            "task_id",
            "species_id",
            "solvent_name",
            "charge",
            "multiplicity",
            "epsilon_r",
        ),
        [
            {
                "task_id": "task-1",
                "species_id": "monomer-001",
                "solvent_name": "Acetonitrile (MeCN)",
                "charge": "0",
                "multiplicity": "1",
                "epsilon_r": "35.688",
            }
        ],
    )
    write_csv_deterministic(
        tier1_run / "descriptors" / "task_status.csv",
        (
            "batch",
            "task_id",
            "run_id",
            "status",
            "geometry_ok",
            "normal_termination",
            "input_hash_ok",
            "reason",
            "output_dir",
            "optimized_xyz_sha256",
            "output_sha256",
            "qc_status",
            "bonds_broken",
            "bonds_formed",
            "source_sha256",
        ),
        [
            {
                "batch": "redox",
                "task_id": "task-1",
                "run_id": "same-run-fixture",
                "status": "accepted",
                "geometry_ok": "true",
                "normal_termination": "true",
                "input_hash_ok": "true",
                "reason": "",
                "output_dir": "raw/redox/task-1",
                "optimized_xyz_sha256": source_sha,
                "output_sha256": "fixture-output-sha",
                "qc_status": "pass",
                "bonds_broken": "0",
                "bonds_formed": "0",
                "source_sha256": "fixture-source-sha",
            }
        ],
    )
    request = GeometryRequest(
        geometry_key="tier1:redox:M001:S001:q0:m1",
        state_id="M001_Q0_M1",
        solvent_id="S001",
        formal_charge=0,
        multiplicity=1,
        geometry_source="same-run Tier-1",
        required_by_job_ids=("fixture-job",),
    )

    rows = resolve_tier1_requests(
        [request],
        spec_dir=SPEC_DIR,
        tier1_run=tier1_run,
        run_dir=tmp_path / "geometry-run",
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "resolved"
    assert rows[0]["xyz_sha256"] == source_sha
    assert rows[0]["source_run_id"] == "same-run-fixture"


def test_sigma_preopt_manifest_uses_frozen_charge_spin_epsilon_and_indices(
    tmp_path: Path,
) -> None:
    request = next(
        request
        for request in geometry_requests(SPEC_DIR)
        if request.state_id == "D001_QP2_M1" and request.solvent_id == "S001"
    )

    rows, manifest = prepare_and_resolve_sigma_requests(
        [request], spec_dir=SPEC_DIR, run_dir=tmp_path, n_conformers=2
    )

    assert rows[0]["status"] == "pending"
    assert manifest[0]["formal_charge"] == 2
    assert manifest[0]["multiplicity"] == 1
    assert manifest[0]["uhf"] == 0
    assert manifest[0]["epsilon"] == "35.688"
    assert manifest[0]["xtb_command"] == (
        "xtb in.xyz --chrg 2 --uhf 0 --opt --cosmo 35.688"
    )
    assert (tmp_path / str(manifest[0]["source_xyz"])).is_file()
    assert len((tmp_path / "sigma_preopt_array.tsv").read_text().splitlines()) == 1


def test_sigma_connectivity_qc_uses_authored_graph_and_detects_broken_junction(
    tmp_path: Path,
) -> None:
    topology = load_sigma_topologies(SPEC_DIR / "sigma_coupling_topology.csv")[0]
    sigma = build_sigma_complex(topology, n_conformers=2)
    input_xyz = tmp_path / "input.xyz"
    unchanged_xyz = tmp_path / "unchanged.xyz"
    broken_xyz = tmp_path / "broken.xyz"
    input_xyz.write_text(sigma.xyz_text(), encoding="utf-8")
    unchanged_xyz.write_text(sigma.xyz_text(), encoding="utf-8")

    coordinates = list(sigma.coordinates_angstrom)
    junction = sigma.junction_atom_indices[0]
    x, y, z = coordinates[junction]
    coordinates[junction] = (x + 10.0, y, z)
    lines = [str(len(sigma.symbols)), "broken junction fixture"]
    lines.extend(
        f"{symbol:<2s} {cx:18.10f} {cy:18.10f} {cz:18.10f}"
        for symbol, (cx, cy, cz) in zip(sigma.symbols, coordinates, strict=True)
    )
    broken_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reference = sigma_reference_bonds(topology)

    unchanged = check_connectivity(
        input_xyz, unchanged_xyz, reference_bonds=reference
    )
    broken = check_connectivity(input_xyz, broken_xyz, reference_bonds=reference)

    assert unchanged.ok
    assert unchanged.bonds_broken == unchanged.bonds_formed == 0
    assert not broken.ok
    assert broken.bonds_broken > 0
