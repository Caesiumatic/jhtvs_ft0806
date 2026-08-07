from __future__ import annotations

import csv
from pathlib import Path

import pytest

from jhtvs_ft0806.orca.decks import (
    DeckGenerationError,
    build_exact_reuse_key,
    build_decks,
    render_optfreq_deck,
    render_sp_deck,
)
from jhtvs_ft0806.orca.smd import CUSTOM_SMD_FIELDS, render_smd_block
from jhtvs_ft0806.orca.preflight import audit_decks
from jhtvs_ft0806.provenance import csv_record_sha256, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "orca"


def _solvents() -> dict[str, dict[str, str]]:
    with (SPEC_DIR / "solvent_smd_registry.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return {row["solvent_id"]: row for row in csv.DictReader(handle)}


def _geometry() -> dict[str, str]:
    return {
        "geometry_key": "fixture:geometry",
        "xyz_sha256": "fixture-geometry-sha",
    }


def test_native_sp_deck_matches_golden_fixture() -> None:
    job = {
        "workflow_revision": "sp-revision",
        "job_id": "SPG001",
        "job_class": "smd_energy_sp",
        "state_id": "FIX_Q0_M1",
        "solvent_id": "S001",
        "formal_charge": "0",
        "multiplicity": "1",
        "method_id": "sp-method",
        "functional": "wB97X-D3",
        "basis": "def2-TZVPD",
        "nprocs": "8",
        "maxcore_mb_per_rank": "3000",
    }

    rendered = render_sp_deck(
        job,
        _geometry(),
        FIXTURE_DIR / "small.xyz",
        _solvents()["S001"],
        registry_sha256="registry-sha",
    )

    assert rendered == (FIXTURE_DIR / "sp_native.inp").read_text(encoding="utf-8")


def test_custom_optfreq_deck_matches_golden_fixture() -> None:
    job = {
        "workflow_revision": "optfreq-revision",
        "job_id": "OFG001",
        "state_id": "FIX_QM1_M1",
        "solvent_id": "S003",
        "formal_charge": "-1",
        "multiplicity": "1",
        "method_id": "optfreq-method",
        "functional": "wB97X-D3",
        "optfreq_basis": "ma-def2-TZVP",
        "final_sp_basis": "def2-TZVPD",
        "nprocs": "8",
        "maxcore_mb_per_rank": "3000",
    }

    rendered = render_optfreq_deck(
        job,
        _geometry(),
        FIXTURE_DIR / "small.xyz",
        _solvents()["S003"],
        registry_sha256="registry-sha",
    )

    assert rendered == (FIXTURE_DIR / "optfreq_custom.inp").read_text(
        encoding="utf-8"
    )
    assert rendered.count("%cpcm") == 2
    assert rendered.count('SMDsolvent "water"') == 2
    for field in CUSTOM_SMD_FIELDS:
        assert rendered.count(f"  {field} ") == 2


def test_gas_sp_has_no_smd_block_and_water_keeps_native_route() -> None:
    gas_job = {
        "workflow_revision": "gas-revision",
        "job_id": "SPGAS",
        "job_class": "diagnostic_gas_sp",
        "state_id": "FIX_Q0_M1",
        "solvent_id": "GAS",
        "formal_charge": "0",
        "multiplicity": "1",
        "method_id": "gas-method",
        "functional": "wB97X-D3",
        "basis": "def2-TZVPD",
        "nprocs": "8",
        "maxcore_mb_per_rank": "3000",
    }

    gas = render_sp_deck(
        gas_job,
        _geometry(),
        FIXTURE_DIR / "small.xyz",
        None,
        registry_sha256="registry-sha",
    )
    water = render_smd_block(_solvents()["S010"])

    assert "%cpcm" not in gas
    assert "SMDsolvent" not in gas
    assert 'SMDsolvent "Water"' in water
    assert "  epsilon " not in water


@pytest.mark.parametrize(
    ("solvent_id", "expected"),
    [
        (
            "S007",
            "%cpcm\n"
            "  smd true\n"
            "  smdsolvent \"DMSO\"\n"
            "  epsilon 46.826\n"
            "  refrac 1.4783\n"
            "  soln 1.4783\n"
            "  soln25 1.4783\n"
            "  sola 0.0\n"
            "  solb 0.88\n"
            "  solg 61.78\n"
            "  solc 0.0\n"
            "  solh 0.0\n"
            "  smd18 false\n"
            "  draco false\n"
            "end\n",
        ),
        (
            "S012",
            "%cpcm\n"
            "  smd true\n"
            "  smdsolvent \"Sulfolane\"\n"
            "  epsilon 43.962\n"
            "  refrac 1.4825\n"
            "  soln 1.4833\n"
            "  soln25 1.4825\n"
            "  sola 0.0\n"
            "  solb 0.88\n"
            "  solg 87.49\n"
            "  solc 0.0\n"
            "  solh 0.0\n"
            "  smd18 false\n"
            "  draco false\n"
            "end\n",
        ),
    ],
)
def test_registry_exact_custom_blocks_use_only_the_approved_self_seed(
    solvent_id: str, expected: str
) -> None:
    rendered = render_smd_block(_solvents()[solvent_id])

    assert rendered == expected
    assert 'SMDsolvent "water"' not in rendered
    assert "! SMD(" not in rendered


def test_registry_exact_optfreq_reuses_the_echo_payload_in_both_steps() -> None:
    solvent = _solvents()["S007"]
    job = {
        "workflow_revision": "jhtvs-ft0806-optfreq-v3",
        "job_id": "OF-S007-TEST",
        "state_id": "A001_QM1_M1",
        "solvent_id": "S007",
        "formal_charge": "-1",
        "multiplicity": "1",
        "method_id": "T2_wB97X-D3_OptFreq_TZVPD-SP_SMD_v4",
        "functional": "wB97X-D3",
        "optfreq_basis": "ma-def2-TZVP",
        "final_sp_basis": "def2-TZVPD",
        "nprocs": "8",
        "maxcore_mb_per_rank": "3000",
    }

    rendered = render_optfreq_deck(
        job,
        _geometry(),
        FIXTURE_DIR / "small.xyz",
        solvent,
        registry_sha256="registry-sha",
        registry_row_sha256="row-sha",
    )
    echo_payload = render_smd_block(solvent)

    assert rendered.count(echo_payload) == 2
    assert rendered.count('smdsolvent "DMSO"') == 2
    assert 'SMDsolvent "water"' not in rendered
    assert "! SMD(" not in rendered


def test_self_seed_name_is_read_from_the_registry_row() -> None:
    solvent = dict(_solvents()["S007"])
    solvent["orca_smd_input_from_source"] = (
        'custom self-seed: SMDsolvent("REGISTRY_TEST")'
    )

    rendered = render_smd_block(solvent)

    assert 'smdsolvent "REGISTRY_TEST"' in rendered


def test_exact_reuse_key_binds_medium_row_payload_method_and_workflow() -> None:
    job = {
        "state_id": "FIX_Q0_M1",
        "solvent_id": "S007",
        "method_id": "method-v1",
        "workflow_revision": "workflow-v1",
    }
    arguments = {
        "job_class": "smd_energy_sp",
        "smd_registry_row_sha256": "row-sha",
        "smd_payload_sha256": "payload-sha",
    }
    reference = build_exact_reuse_key(job, _geometry(), **arguments)

    for field, changed in {
        "solvent_id": "S012",
        "method_id": "method-v2",
        "workflow_revision": "workflow-v2",
    }.items():
        changed_job = dict(job)
        changed_job[field] = changed
        assert build_exact_reuse_key(changed_job, _geometry(), **arguments) != reference
    for field, changed in {
        "smd_registry_row_sha256": "other-row-sha",
        "smd_payload_sha256": "other-payload-sha",
    }.items():
        changed_arguments = dict(arguments)
        changed_arguments[field] = changed
        assert build_exact_reuse_key(job, _geometry(), **changed_arguments) != reference


def test_registry_row_hash_is_exact_record_hash() -> None:
    registry = SPEC_DIR / "solvent_smd_registry.csv"
    digest = csv_record_sha256(
        registry, key_field="solvent_id", key_value="S007"
    )
    record = next(
        line for line in registry.read_bytes().splitlines(keepends=True)
        if line.startswith(b"S007,")
    )

    import hashlib

    assert digest == hashlib.sha256(record).hexdigest()


def test_selected_deck_build_is_hash_bound_and_rejects_unknown_job(
    tmp_path: Path,
) -> None:
    xyz_path = tmp_path / "source.xyz"
    xyz_path.write_bytes((FIXTURE_DIR / "small.xyz").read_bytes())
    geometry_index = tmp_path / "geometry_index.csv"
    write_csv_deterministic(
        geometry_index,
        ("geometry_key", "status", "reason", "xyz_path", "xyz_sha256"),
        [
            {
                "geometry_key": "tier1:redox:A002:S001:q0:m2",
                "status": "resolved",
                "reason": "",
                "xyz_path": str(xyz_path),
                "xyz_sha256": sha256_file(xyz_path),
            }
        ],
    )
    run_dir = tmp_path / "orca"
    manifest_path = tmp_path / "deck_manifest.csv"

    summary = build_decks(
        spec_dir=SPEC_DIR,
        geometry_index_path=geometry_index,
        run_dir=run_dir,
        manifest_path=manifest_path,
        selected_job_ids={"SP0001"},
    )
    row = read_csv_rows(manifest_path)[0]
    input_path = run_dir / "sp" / "SP0001" / "SP0001.inp"

    assert summary.total == summary.ready == 1
    assert row["status"] == "ready"
    assert row["geometry_sha256"] == sha256_file(xyz_path)
    assert row["smd_registry_row_sha256"] == ""
    assert row["exact_reuse_key"]
    assert row["input_sha256"] == sha256_file(input_path)
    assert "# job_id: SP0001" in input_path.read_text(encoding="utf-8")

    audit = audit_decks(
        spec_dir=SPEC_DIR,
        geometry_index_path=geometry_index,
        deck_manifest_path=manifest_path,
        selected_job_ids={"SP0001"},
        report_path=tmp_path / "preflight.json",
    )
    assert audit.ok
    assert audit.checks["audited_jobs"] == 1
    assert audit.checks["nprocs"] == ["8"]
    assert audit.checks["maxcore_mb_per_rank"] == ["3000"]

    input_path.write_text(
        input_path.read_text(encoding="utf-8").replace("TightSCF", "LooseSCF"),
        encoding="utf-8",
    )
    tampered = audit_decks(
        spec_dir=SPEC_DIR,
        geometry_index_path=geometry_index,
        deck_manifest_path=manifest_path,
        selected_job_ids={"SP0001"},
    )
    assert not tampered.ok
    assert any("differs from exact re-render" in issue for issue in tampered.issues)
    assert any("input SHA-256 mismatch" in issue for issue in tampered.issues)

    with pytest.raises(DeckGenerationError, match="unknown selected job IDs"):
        build_decks(
            spec_dir=SPEC_DIR,
            geometry_index_path=geometry_index,
            run_dir=run_dir,
            manifest_path=manifest_path,
            selected_job_ids={"NOT_A_JOB"},
        )
