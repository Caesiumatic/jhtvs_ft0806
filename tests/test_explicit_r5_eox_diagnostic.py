from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path

import pytest

from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import read_csv_rows


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "diagnostics/explicit_solvation_eox_r5/run_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("explicit_r5_eox_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_input_snapshots_match_all_frozen_sha256_values() -> None:
    protocol = DIAGNOSTIC._protocol()  # noqa: SLF001
    observed = {
        path.name: sha256_file(path)
        for path in (SCRIPT.parent / "input_files").glob("*.csv")
    }

    assert observed == protocol["input_files"]


def test_record_id_merge_is_strict() -> None:
    left = [{"record_id": "r1", "property": "p", "species": "s", "environment": "e", "experimental_value": "1", "unit": "V", "calibrated_tier1": "2", "tier2_dft": "3", "source": "x", "source_url": "u", "source_file": "f", "source_location": "l", "protocol_or_conditions": "c", "normalization": "n"}]
    right = [dict(left[0])]

    assert DIAGNOSTIC._strict_merge_by_record_id(left, right) == left  # noqa: SLF001
    right[0]["species"] = "different"
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="identity mismatch"):
        DIAGNOSTIC._strict_merge_by_record_id(left, right)  # noqa: SLF001


def test_phase_a_intersection_and_calculation_key_deduplication() -> None:
    records = read_csv_rows(SCRIPT.parent / "benchmark_records.csv")
    registry = read_csv_rows(SCRIPT.parent / "calculation_registry.csv")

    assert len(records) == 15
    assert len(registry) == 6
    assert {row["phase"] for row in registry} == {"A"}
    assert {row["calculation_key"] for row in records} == {
        row["calculation_key"] for row in registry
    }


def test_packmol_clusters_have_one_target_and_exactly_five_shell_molecules() -> None:
    rows = read_csv_rows(SCRIPT.parent / "cluster_manifest.csv")

    assert len(rows) == 6
    assert all(row["n_shell"] == "5" for row in rows)
    assert all(row["molecule_count"] == "6" for row in rows)
    assert all(row["status"] == "clean" for row in rows)
    assert all(float(row["minimum_intermolecular_distance_A"]) >= 1.99 for row in rows)
    assert all(row["containment_qc"] == "pass" for row in rows)
    assert all(float(row["max_shell_box_violation_A"]) <= 0.1 for row in rows)


def test_seeds_and_boxes_are_derived_from_stable_keys_and_source_geometry() -> None:
    protocol = DIAGNOSTIC._protocol()  # noqa: SLF001
    volumes = protocol["packmol"]["molecular_volumes_A3"]
    for system in protocol["systems"]:
        shell = DIAGNOSTIC.read_xyz(DIAGNOSTIC._source_path(system["shell_geometry_source"]))  # noqa: SLF001
        expected_box = DIAGNOSTIC.molecular_volume_box_side(
            volumes[system["target_geometry_source"]],
            5,
            volumes[system["shell_geometry_source"]],
            DIAGNOSTIC._molecular_span(shell),  # noqa: SLF001
        )
        assert system["seed"] == DIAGNOSTIC.deterministic_seed(system["calculation_key"])
        assert system["box_side_A"] == expected_box


def test_charge_multiplicity_basis_smd_and_compound_workflow_routing() -> None:
    manifests = read_csv_rows(SCRIPT.parent / "orca/job_manifest.csv")
    systems = DIAGNOSTIC._system_map()  # noqa: SLF001
    by_key: dict[str, list[dict[str, str]]] = {}
    for row in manifests:
        by_key.setdefault(row["calculation_key"], []).append(row)
        system = systems[row["calculation_key"]]
        state_index = 0 if row["state_role"] == "reduced" else 1
        assert (int(row["formal_charge"]), int(row["multiplicity"])) == tuple(
            system["states"][state_index]
        )
        assert row["optfreq_basis"] == system[f"optfreq_basis_{row['state_role']}"]
        deck = (REPOSITORY_ROOT / row["input_path"]).read_text(encoding="utf-8")
        assert " Opt Freq\n" in deck
        assert f"* xyzfile {row['formal_charge']} {row['multiplicity']} {row['job_id']}_Compound_1.xyz" in deck
        assert deck.count("%cpcm") == 2
        assert ("Hirshfeld" in deck) == (row["state_role"] == "oxidized")
    assert len(manifests) == 12
    assert all(len(pair) == 2 for pair in by_key.values())
    assert all(len({row["geometry_sha256"] for row in pair}) == 1 for pair in by_key.values())
    assert all(len({row["coordinate_payload_sha256"] for row in pair}) == 1 for pair in by_key.values())


def test_composite_gap_and_pinned_conversion_arithmetic() -> None:
    conversion = lambda value: value - 4.477
    delta, primary, independent = DIAGNOSTIC._explicit_eox_from_raw(  # noqa: SLF001
        "-100.123456789012", "-99.923456789012", conversion
    )

    assert delta == Decimal("0.200000000000") * DIAGNOSTIC.HARTREE_TO_EV_DECIMAL
    assert abs(primary - independent) < Decimal("1e-12")


def test_pinned_agagcl_callable_loads_only_when_readonly_checkout_is_present() -> None:
    checkout = Path("/Users/shichen/GitHub/20260707")
    if not checkout.is_dir():
        pytest.skip("read-only 20260707 checkout is not mounted")

    conversion = DIAGNOSTIC._pinned_conversion(checkout)  # noqa: SLF001

    assert callable(conversion)


def test_anion_fragment_spin_aggregation_requires_target_dominance() -> None:
    ranges = (range(0, 2), range(2, 4), range(4, 6))
    target_spins = [Decimal("0.35"), Decimal("0.35"), Decimal("0.10"), Decimal("0.10"), Decimal("0.05"), Decimal("0.05")]
    _, target_identity = DIAGNOSTIC.aggregate_fragment_spins(target_spins, ranges, anion=True)
    solvent_spins = [Decimal("0.05"), Decimal("0.05"), Decimal("0.40"), Decimal("0.40"), Decimal("0.05"), Decimal("0.05")]
    _, solvent_identity = DIAGNOSTIC.aggregate_fragment_spins(solvent_spins, ranges, anion=True)

    assert target_identity["oxidation_identity_status"] == "clean"
    assert solvent_identity["oxidation_identity_status"] == "oxidation_identity_mismatch"


def test_hirshfeld_parser_uses_last_analysis_and_full_atom_coverage() -> None:
    text = """HIRSHFELD ANALYSIS
  ATOM     CHARGE      SPIN
   0 C   -0.100000    0.700000
   1 O    0.100000    0.300000
  TOTAL   0.000000    1.000000
"""

    assert DIAGNOSTIC.parse_hirshfeld_spins(text, 2) == [Decimal("0.700000"), Decimal("0.300000")]


def test_record_and_macro_metrics_and_qc_propagation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(DIAGNOSTIC, "DIAGNOSTIC_ROOT", tmp_path)
    rows = [
        {"record_id": "r1", "property": "anion_eox", "calculation_key": "a", "experimental_Eox_V": "2.0", "implicit_calibrated_xTB_Eox_V": "2.3", "implicit_DFT_Eox_V": "2.4", "explicit_R5_DFT_Eox_V": "2.1", "calculation_qc": "clean"},
        {"record_id": "r2", "property": "anion_eox", "calculation_key": "a", "experimental_Eox_V": "2.2", "implicit_calibrated_xTB_Eox_V": "2.3", "implicit_DFT_Eox_V": "2.4", "explicit_R5_DFT_Eox_V": "2.1", "calculation_qc": "clean"},
        {"record_id": "r3", "property": "anion_eox", "calculation_key": "b", "experimental_Eox_V": "3.0", "implicit_calibrated_xTB_Eox_V": "3.2", "implicit_DFT_Eox_V": "3.3", "explicit_R5_DFT_Eox_V": "3.1", "calculation_qc": "clean"},
    ]

    metrics, _ = DIAGNOSTIC.build_metrics(rows)

    assert {row["aggregation"] for row in metrics} == {"record", "unique_calculation_macro"}
    assert DIAGNOSTIC.pair_qc_status(["clean", "clean"]) == "clean"
    assert DIAGNOSTIC.pair_qc_status(["clean", "flagged"]) == "flagged"
    assert DIAGNOSTIC.pair_qc_status(["flagged", "missing"]) == "missing"
    assert DIAGNOSTIC.optimized_geometries_independent("a", "b")
    assert not DIAGNOSTIC.optimized_geometries_independent("a", "a")


def test_no_mace_or_torch_import_and_runner_path_is_narrowly_allowlisted() -> None:
    imports = [
        line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    ]
    runner = (REPOSITORY_ROOT / "hpc/run_orca.sh").read_text(encoding="utf-8")

    assert not any("mace" in line.lower() or "torch" in line.lower() for line in imports)
    assert "optfreq:diagnostics/explicit_solvation_eox_r5/orca/jobs/*/*.inp" in runner
    assert "diagnostics/explicit_solvation_eox_r5/orca/jobs/*/*.out" in runner


def test_effective_manifest_uses_only_the_single_audited_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    originals = read_csv_rows(SCRIPT.parent / "orca/job_manifest.csv")
    original = next(row for row in originals if row["job_id"] == "R5EOX_A01_RED")
    continuation = {field: "" for field in DIAGNOSTIC.CONTINUATION_FIELDS}
    continuation.update(
        {
            "logical_job_id": original["job_id"],
            "attempt": "1",
            "attempt_job_id": "R5EOX_A01_RED_CONT1",
            "calculation_key": original["calculation_key"],
            "state_role": original["state_role"],
            "input_path": "diagnostics/retry.inp",
            "input_sha256": "1" * 64,
            "output_path": "diagnostics/retry.out",
            "geometry_key": "continuation1:R5EOX_A01_RED",
            "source_geometry_path": "diagnostics/retry_source.xyz",
            "source_geometry_sha256": "2" * 64,
            "coordinate_payload_sha256": "3" * 64,
            "exact_reuse_key": "4" * 64,
        }
    )
    path = tmp_path / "continuation_manifest.csv"
    DIAGNOSTIC.write_csv_deterministic(
        path, DIAGNOSTIC.CONTINUATION_FIELDS, [continuation]
    )
    monkeypatch.setattr(DIAGNOSTIC, "CONTINUATION_MANIFEST_PATH", path)

    effective = DIAGNOSTIC._effective_manifests()  # noqa: SLF001
    replacement = next(
        row for row in effective
        if row.get("logical_job_id") == original["job_id"]
    )

    assert len(effective) == len(originals)
    assert replacement["job_id"] == "R5EOX_A01_RED_CONT1"
    assert replacement["input_geometry_path"] == "diagnostics/retry_source.xyz"
    assert replacement["method_id"] == original["method_id"]

    continuation["attempt"] = "2"
    DIAGNOSTIC.write_csv_deterministic(
        path, DIAGNOSTIC.CONTINUATION_FIELDS, [continuation]
    )
    with pytest.raises(DIAGNOSTIC.DiagnosticError, match="only one continuation"):
        DIAGNOSTIC._continuation_rows()  # noqa: SLF001


def test_reduced_state_continuation_serially_preserves_unattempted_oxidized_state() -> None:
    continuation_task = {
        field: "continuation-value" for field in DIAGNOSTIC.TASK_FIELDS
    }
    continuation_task["job_id"] = "R5EOX_A01_RED_CONT1"

    bundled = DIAGNOSTIC._continuation_bundle_rows(  # noqa: SLF001
        "R5EOX_A01_RED", continuation_task
    )

    assert [row["job_id"] for row in bundled] == [
        "R5EOX_A01_RED_CONT1",
        "R5EOX_A01_OX",
    ]
    assert [row["array_task"] for row in bundled] == ["1", "1"]
    assert [row["sequence"] for row in bundled] == ["1", "2"]


def test_scheduler_accounting_core_hours_and_maxiter_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = next(
        row for row in read_csv_rows(SCRIPT.parent / "orca/job_manifest.csv")
        if row["job_id"] == "R5EOX_A01_RED"
    )
    output = tmp_path / "failed.out"
    output.write_text(
        "ERROR !!!\nThe optimization did not converge but reached the maximum\n"
        "****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "job_manifest.csv"
    patched = dict(original)
    patched["output_path"] = output.name
    DIAGNOSTIC.write_csv_deterministic(
        manifest, DIAGNOSTIC.ORCA_FIELDS, [patched]
    )
    status = tmp_path / "execution_status.json"
    status.write_text('{"scheduler_job_ids": []}\n', encoding="utf-8")
    monkeypatch.setattr(DIAGNOSTIC, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(DIAGNOSTIC, "DIAGNOSTIC_ROOT", tmp_path)
    monkeypatch.setattr(DIAGNOSTIC, "ORCA_MANIFEST_PATH", manifest)
    monkeypatch.setattr(DIAGNOSTIC, "ACCOUNTING_PATH", tmp_path / "accounting.csv")
    monkeypatch.setattr(
        DIAGNOSTIC, "CONTINUATION_MANIFEST_PATH", tmp_path / "absent.csv"
    )

    row = DIAGNOSTIC.record_scheduler_accounting(
        scheduler_job_id="423864", task_id="1", logical_job_id="R5EOX_A01_RED",
        attempt="0", start_time_iso="2026-08-12T15:00:23-05:00",
        end_time_iso="2026-08-13T01:47:58-05:00", wallclock_s="38855",
        slots="8", failed="0", exit_status="2",
        outcome="optimization_maxiter_200",
        recorded_at_utc="2026-08-13T17:20:00+00:00",
    )

    assert row["core_hours"] == "86.344444444444"
    assert row["output_sha256"] == sha256_file(output)
    assert read_csv_rows(tmp_path / "accounting.csv") == [row]


def test_prepared_artifacts_pass_fail_closed_validation() -> None:
    report = DIAGNOSTIC.validate_prepared()

    assert report["status"] == "PASS"
    assert report["checks"]["eligible_record_count"] == 15
    assert report["checks"]["phase_a_unique_key_count"] == 6
    assert report["checks"]["orca_job_count"] == 12
