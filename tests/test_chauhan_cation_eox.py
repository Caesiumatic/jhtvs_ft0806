from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "chauhan_cation_eox"
sys.path.insert(0, str(WORKFLOW))

import aggregate_results  # noqa: E402
import analyze_cation_effects  # noqa: E402
import analyze_pair_only  # noqa: E402
import analyze_unconstrained_results  # noqa: E402
import build_structures  # noqa: E402
import common  # noqa: E402
import make_manifest  # noqa: E402
import make_unconstrained_manifest  # noqa: E402
import parse_results  # noqa: E402
import parse_unconstrained_results  # noqa: E402
import run_calculation  # noqa: E402


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("chauhan_structures")
    structures = build_structures.generate(output)
    calculations = make_manifest.generate(output / "structure_manifest.csv", output / "calculation_manifest.csv")
    return output, structures, calculations


def _fake_xtb(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import pathlib
import shutil
import sys

if '--version' in sys.argv:
    print('xtb version 6.7.1')
    raise SystemExit(0)
args = sys.argv[1:]
cwd = pathlib.Path.cwd()
atoms = int((cwd / 'in.xyz').read_text().splitlines()[0])
charge = int(args[args.index('--chrg') + 1])
task_id = cwd.parent.name
base = -20.0 if task_id.startswith('cation__') else -30.0
if cwd.name == 'reduced_opt':
    energy = base - 0.5
    shutil.copy2(cwd / 'in.xyz', cwd / 'xtbopt.xyz')
elif cwd.name == 'reduced_sp':
    energy = base
else:
    energy = base + (0.2 if task_id.startswith('cation__') else 0.1)
(cwd / 'charges').write_text('\\n'.join([str(charge / atoms)] * atoms) + '\\n')
print(f'| TOTAL ENERGY {energy:.12f} Eh |')
print('normal termination of xtb')
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _triad_and_cation(calculations):
    triad = next(r for r in calculations if r["kind"] == "triad" and r["cation"] == "EMIM" and r["anion"] == "NTF2" and r["solvent"] == "PC" and r["topology"] == "CAS")
    cation = next(r for r in calculations if r["kind"] == "cation" and r["cation"] == "EMIM" and r["solvent"] == "PC")
    return triad, cation


def test_exact_species_and_composition_matrix():
    species = common.species_table()
    assert {key: float(species[key]["epsilon"]) for key in common.SOLVENTS} == {"PC": 65.0, "EG": 37.0, "THF": 7.6}
    assert common.CATIONS_BY_ANION == {
        "NTF2": ("EMIM", "BMIM", "HMIM"),
        "OTF": ("EMIM", "HMIM", "BMPYRR"),
        "PF6": ("BMIM", "HMIM"),
    }
    assert len(common.composition_keys()) == 24
    assert len(common.cation_solvent_keys()) == 12
    assert len(common.benchmark_rows()) == 24


def test_real_smoke_summary_is_corrected_full_composition():
    rows = common.read_csv(ROOT / "data" / "chauhan_cation_eox" / "smoke_test_summary.csv")
    assert {(row["cation"], row["anion"], row["solvent"], row["topology"]) for row in rows} == {
        ("EMIM", "NTF2", "PC", topology) for topology in common.TOPOLOGIES
    }
    assert all(row["status"] == "complete" for row in rows)
    assert all(row["topology_preserved"] == "True" for row in rows)
    assert all(row["same_geometry_reduced_sp"] == "True" for row in rows)
    assert all(row["same_geometry_oxidized_sp"] == "True" for row in rows)


def test_structure_task_and_invocation_counts(generated):
    _, structures, calculations = generated
    assert len([r for r in structures if r["kind"] == "triad"]) == 72
    assert len([r for r in structures if r["kind"] == "as_pair"]) == 9
    assert len([r for r in structures if r["kind"] == "solvent"]) == 3
    assert len([r for r in structures if r["kind"] == "anion"]) == 9
    assert len([r for r in structures if r["kind"] == "cation"]) == 12
    assert len(calculations) == 105
    assert len(calculations) * 3 == 315
    assert all(int(r["formal_charge"]) == 0 for r in structures if r["kind"] == "triad")
    assert all(int(r["formal_charge"]) == -1 for r in structures if r["kind"] == "as_pair")
    assert all(int(r["formal_charge"]) == 1 for r in structures if r["kind"] == "cation")


def test_vertical_state_assignments_and_cation_reuse(generated):
    _, _, calculations = generated
    cation_tasks = [row for row in calculations if row["kind"] == "cation"]
    assert len(cation_tasks) == 12
    assert len({(row["cation"], row["solvent"]) for row in cation_tasks}) == 12
    for row in calculations:
        if row["kind"] in {"triad", "solvent"}:
            assert (int(row["charge_reduced"]), int(row["uhf_reduced"]), int(row["charge_oxidized"]), int(row["uhf_oxidized"])) == (0, 0, 1, 1)
        elif row["kind"] == "cation":
            assert (int(row["charge_reduced"]), int(row["uhf_reduced"]), int(row["charge_oxidized"]), int(row["uhf_oxidized"])) == (1, 0, 2, 1)
        else:
            assert (int(row["charge_reduced"]), int(row["uhf_reduced"]), int(row["charge_oxidized"]), int(row["uhf_oxidized"])) == (-1, 0, 0, 1)
    for cation, _, solvent in common.composition_keys():
        assert sum(row["cation"] == cation and row["solvent"] == solvent for row in cation_tasks) == 1


def test_geometry_generation_is_deterministic():
    first = build_structures.build_component("HMIM")
    second = build_structures.build_component("HMIM")
    assert first.anchor == second.anchor
    assert first.atoms == second.atoms
    pf6 = build_structures.build_component("PF6")
    p = pf6.atoms[pf6.anchor]
    assert (p.x, p.y, p.z) == (0.0, 0.0, 0.0)
    assert sorted(round(common.distance(p, atom), 8) for atom in pf6.atoms if atom.element == "F") == [1.58] * 6


def test_requested_middle_fragment_is_initially_placed_in_middle(generated):
    output, structures, _ = generated
    for row in structures:
        if row["kind"] != "triad":
            continue
        xyz = output / "initial" / f"{row['structure_id']}.xyz"
        atoms, _ = common.read_xyz(xyz)
        metadata = common.load_metadata(xyz)
        anchors = metadata["anchor_indices_zero_based"]
        order = "".join(sorted(anchors, key=lambda label: atoms[anchors[label]].x))
        assert order == row["topology"]
        assert parse_results.geometry_metrics(atoms, metadata)["inferred_topology"] == row["topology"]
        assert metadata["minimum_heavy_distance_ang"] >= build_structures.MIN_HEAVY_DISTANCE_ANG


def test_commands_limit_restraint_to_optimization_and_hashes_match(generated, tmp_path):
    _, _, calculations = generated
    triad, _ = _triad_and_cation(calculations)
    provenance = run_calculation.run_task(triad, str(_fake_xtb(tmp_path / "xtb")), tmp_path / "runs")
    assert "--opt" in provenance["optimization_command"]
    assert provenance["optimization_command"][-2:] == ["--input", "xcontrol.inp"]
    for command in (provenance["reduced_sp_command"], provenance["oxidized_sp_command"]):
        assert "--opt" not in command
        assert "--input" not in command
    assert provenance["optimized_geometry_sha256"] == provenance["reduced_sp_input_geometry_sha256"]
    assert provenance["optimized_geometry_sha256"] == provenance["oxidized_sp_input_geometry_sha256"]
    assert provenance["same_geometry_reduced_sp"] is True
    assert provenance["same_geometry_oxidized_sp"] is True


def test_vertical_ip_uses_sp_minus_sp_and_cation_delta(generated, tmp_path):
    _, _, calculations = generated
    triad, cation = _triad_and_cation(calculations)
    fake = _fake_xtb(tmp_path / "xtb")
    run_root = tmp_path / "runs"
    run_calculation.run_task(triad, str(fake), run_root)
    run_calculation.run_task(cation, str(fake), run_root)
    parsed = parse_results.parse_triad(triad, calculations, run_root)
    assert parsed["energy_neutral_opt_eh"] == -30.5
    assert parsed["energy_neutral_sp_eh"] == -30.0
    assert parsed["energy_oxidized_sp_eh"] == -29.9
    assert parsed["ip_vertical_ev"] == pytest.approx(0.1 * common.HARTREE_TO_EV)
    assert parsed["ip_vertical_ev"] != pytest.approx(0.6 * common.HARTREE_TO_EV)
    assert parsed["ip_cation_ev"] == pytest.approx(0.2 * common.HARTREE_TO_EV)
    assert parsed["delta_ip_vs_isolated_cation_ev"] == pytest.approx(-0.1 * common.HARTREE_TO_EV)


def test_resume_requires_and_detects_all_three_states(generated, tmp_path):
    _, _, calculations = generated
    triad, _ = _triad_and_cation(calculations)
    fake = _fake_xtb(tmp_path / "xtb")
    run_root = tmp_path / "runs"
    run_calculation.run_task(triad, str(fake), run_root)
    resumed = run_calculation.run_task(triad, str(fake), run_root)
    assert resumed["optimization_executed_this_invocation"] is False
    assert resumed["reduced_sp_executed_this_invocation"] is False
    assert resumed["oxidized_sp_executed_this_invocation"] is False
    (run_root / triad["task_id"] / "reduced_sp" / "charges").unlink()
    repaired = run_calculation.run_task(triad, str(fake), run_root)
    assert repaired["optimization_executed_this_invocation"] is False
    assert repaired["reduced_sp_executed_this_invocation"] is True
    assert repaired["oxidized_sp_executed_this_invocation"] is False


def test_parser_rejects_geometry_mismatch(generated, tmp_path):
    _, _, calculations = generated
    triad, cation = _triad_and_cation(calculations)
    fake = _fake_xtb(tmp_path / "xtb")
    run_root = tmp_path / "runs"
    run_calculation.run_task(triad, str(fake), run_root)
    run_calculation.run_task(cation, str(fake), run_root)
    reduced_sp_input = run_root / triad["task_id"] / "reduced_sp" / "in.xyz"
    reduced_sp_input.write_text(reduced_sp_input.read_text() + "\n", encoding="utf-8")
    parsed = parse_results.parse_triad(triad, calculations, run_root)
    assert parsed["status"] == "not_run_or_incomplete"
    assert "geometry hash differs" in parsed["note"]


def test_hartree_to_ev_conversion():
    assert common.ip_ev(-9.75, -10.0) == pytest.approx(6.802846561497)


def test_xtb_fixture_energy_and_atomic_charge_parser():
    fixture = ROOT / "tests" / "fixtures" / "chauhan_cation_eox"
    assert run_calculation.parse_energy((fixture / "xtb.out").read_text()) == pytest.approx(-123.456789012345)
    assert run_calculation.parse_charges(fixture / "charges", 3) == pytest.approx([0.125, -0.375, 0.25])


def test_failed_calculation_records_provenance(generated, tmp_path):
    _, _, calculations = generated
    fake_xtb = tmp_path / "xtb"
    fake_xtb.write_text("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'xtb version 6.7.1'; exit 0; fi\nexit 7\n")
    fake_xtb.chmod(0o755)
    triad, _ = _triad_and_cation(calculations)
    run_root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="exit code 7"):
        run_calculation.run_task(triad, str(fake_xtb), run_root)
    provenance = json.loads((run_root / triad["task_id"] / "provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["optimization_executed_this_invocation"] is True
    assert provenance["reduced_sp_executed_this_invocation"] is False
    assert provenance["oxidized_sp_executed_this_invocation"] is False


def test_composition_aggregation_formulas(tmp_path):
    triad_path = tmp_path / "triads.csv"
    reference_path = tmp_path / "references.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    output = tmp_path / "summary.csv"
    with triad_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cation", "anion", "solvent", "topology", "ip_vertical_ev", "ip_cation_ev", "oxidized_fragment"])
        writer.writeheader()
        for topology, value, fragment in [("CAS", 6.0, "A"), ("CSA", 5.5, "S"), ("ACS", 6.5, "C")]:
            writer.writerow({"cation": "EMIM", "anion": "NTF2", "solvent": "PC", "topology": topology, "ip_vertical_ev": value, "ip_cation_ev": 7.0, "oxidized_fragment": fragment})
    with reference_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anion", "solvent", "ip_as_direct_ev", "ip_fadel_2p8_ev"])
        writer.writeheader()
        writer.writerow({"anion": "NTF2", "solvent": "PC", "ip_as_direct_ev": 5.0, "ip_fadel_2p8_ev": 4.5})
    with benchmark_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cation", "anion", "solvent", "eox_exp_v", "eox_sd_v"])
        writer.writeheader()
        writer.writerow({"cation": "EMIM", "anion": "NTF2", "solvent": "PC", "eox_exp_v": 2.8, "eox_sd_v": 0.15})
    row = aggregate_results.aggregate(triad_path, reference_path, benchmark_path, output)[0]
    assert row["ip_min_ev"] == 5.5
    assert row["ip_mean_ev"] == 6.0
    assert row["ip_span_ev"] == 1.0
    assert row["topology_of_min_ip"] == "CSA"
    assert row["delta_ip_min_vs_AS"] == 0.5
    assert row["ip_fadel_2p8_ev"] == 4.5
    assert row["ip_cation_ev"] == 7.0
    assert row["status"] == "complete"


def _analysis_fixture_rows():
    rows = []
    cations = {"NTF2": ("EMIM", "BMIM", "HMIM"), "OTF": ("EMIM", "HMIM", "BMPYRR"), "PF6": ("BMIM", "HMIM")}
    for anion_index, (anion, members) in enumerate(cations.items()):
        for solvent_index, solvent in enumerate(("PC", "EG", "THF")):
            for cation_index, cation in enumerate(members):
                experimental = anion_index + solvent_index / 10 + cation_index / 5
                base = 5 + anion_index + solvent_index / 10 + cation_index / 5
                rows.append({
                    "cation": cation, "anion": anion, "solvent": solvent,
                    "eox_exp_v": str(experimental), "eox_sd_v": "0.1",
                    "ip_cation_ev": str(base), "ip_CAS_ev": str(base + 0.1),
                    "ip_CSA_ev": str(base + 0.2), "ip_ACS_ev": str(base + 0.3),
                    "ip_min_ev": str(base + 0.1), "ip_mean_ev": str(base + 0.2),
                    "ip_span_ev": "0.2", "status": "complete",
                })
    return rows


def test_cation_analysis_cardinalities_and_zero_baseline():
    rows = _analysis_fixture_rows()
    sensitivity = analyze_cation_effects.build_sensitivity(rows)
    contrasts = analyze_cation_effects.build_pairwise(rows)
    metrics = analyze_cation_effects.build_metrics(rows, contrasts)
    assert len(sensitivity) == 9
    assert len(contrasts) == 21
    assert all(row["delta_fadel_as_zero_ev"] == 0 for row in contrasts)
    isolated = next(row for row in metrics if row["descriptor"] == "isolated_cation")
    assert float(isolated["mae_pairwise_v"]) == pytest.approx(0.0)
    assert float(isolated["pearson_pairwise_r"]) == pytest.approx(1.0)
    assert float(isolated["spearman_pairwise_rho"]) == pytest.approx(1.0)


def test_group_centered_descriptor_sums_are_zero():
    rows = _analysis_fixture_rows()
    for _, field in analyze_cation_effects.DESCRIPTORS:
        if field is None:
            continue
        observed, predicted = analyze_cation_effects._centered_vectors(rows, field)
        assert len(observed) == len(predicted) == 24
        offset = 0
        for size in (3, 3, 3, 3, 3, 3, 2, 2, 2):
            assert sum(observed[offset:offset + size]) == pytest.approx(0.0, abs=1e-12)
            assert sum(predicted[offset:offset + size]) == pytest.approx(0.0, abs=1e-12)
            offset += size


def _pair_only_fixture_records():
    records = []
    for index, (anion, solvent) in enumerate((
        (anion, solvent) for anion in common.ANIONS for solvent in common.SOLVENTS
    )):
        ip = 10.0 + index / 10
        records.append({
            "anion": anion, "solvent": solvent,
            "epsilon": common.species_table()[solvent]["epsilon"],
            "energy_reduced_opt_eh": -10.1, "energy_reduced_sp_eh": -10.0,
            "energy_oxidized_sp_eh": -10.0 + ip / common.HARTREE_TO_EV,
            "ip_as_direct_ev": ip,
            "eox_as_calc_vs_agagcl_v": ip - analyze_pair_only.AGAGCL_SHIFT_V,
            "dq_A": 0.75, "dq_S": 0.25, "oxidized_fragment_as": "A",
            "same_geometry_pass": True, "status": "complete", "note": "",
        })
    return records


def test_pair_only_tables_have_exact_mapping_and_fixed_conversion():
    comparison, unique = analyze_pair_only.build_tables(_pair_only_fixture_records(), common.benchmark_rows())
    assert len(comparison) == 24
    assert len(unique) == 9
    for row in comparison:
        assert float(row["eox_as_calc_vs_agagcl_v"]) == pytest.approx(float(row["ip_as_direct_ev"]) - 4.477)
    for key in {(row["anion"], row["solvent"]) for row in comparison}:
        repeated = [row for row in comparison if (row["anion"], row["solvent"]) == key]
        assert len({(row["ip_as_direct_ev"], row["eox_as_calc_vs_agagcl_v"]) for row in repeated}) == 1


def test_pair_only_inputs_contain_only_anion_and_solvent_fragments():
    tasks = common.read_csv(ROOT / "data" / "chauhan_cation_eox" / "calculation_manifest.csv")
    pair_tasks = [row for row in tasks if row["kind"] == "as_pair"]
    assert len(pair_tasks) == 9
    for row in pair_tasks:
        xyz = ROOT / row["input_xyz"]
        atoms, _ = common.read_xyz(xyz)
        metadata = common.load_metadata(xyz)
        assert set(metadata["fragments"]) == {"A", "S"}
        covered = sorted(index for fragment in metadata["fragments"].values() for index in fragment["atom_indices_zero_based"])
        assert covered == list(range(len(atoms)))


def test_pair_only_raw_protocol_and_same_geometry_validation(generated, tmp_path):
    _, _, calculations = generated
    pair = next(row for row in calculations if row["kind"] == "as_pair" and row["anion"] == "NTF2" and row["solvent"] == "PC")
    run_root = tmp_path / "runs"
    run_calculation.run_task(pair, str(_fake_xtb(tmp_path / "xtb")), run_root)
    record = analyze_pair_only.load_pair_record(pair, run_root)
    assert record["same_geometry_pass"] is True
    assert record["status"] == "complete"
    assert record["ip_as_direct_ev"] == pytest.approx(0.1 * common.HARTREE_TO_EV)


def test_unconstrained_manifest_reuses_exact_triads_without_references(generated, tmp_path):
    output, _, _ = generated
    rows = make_unconstrained_manifest.generate(output / "calculation_manifest.csv", tmp_path / "unconstrained.csv")
    assert len(rows) == 72
    assert all(row["kind"] == "triad" and row["restraint"] == "none" for row in rows)
    assert all(row["restraint_force_constant_eh_bohr2"] == "" for row in rows)
    assert {row["task_id"] for row in rows} == {
        row["task_id"] for row in common.read_csv(output / "calculation_manifest.csv") if row["kind"] == "triad"
    }
    assert all(row["source_initial_xyz_sha256"] == common.sha256_file(Path(row["input_xyz"])) for row in rows)


def test_unconstrained_optimization_has_no_xcontrol_and_sp_hashes_match(generated, tmp_path):
    output, _, _ = generated
    rows = make_unconstrained_manifest.generate(output / "calculation_manifest.csv", tmp_path / "unconstrained.csv")
    provenance = run_calculation.run_task(rows[0], str(_fake_xtb(tmp_path / "xtb")), tmp_path / "runs")
    assert "--opt" in provenance["optimization_command"]
    assert "--input" not in provenance["optimization_command"]
    assert not (tmp_path / "runs" / rows[0]["task_id"] / "reduced_opt" / "xcontrol.inp").exists()
    assert provenance["restraint"]["form"] == "none"
    assert provenance["optimized_geometry_sha256"] == provenance["reduced_sp_input_geometry_sha256"]
    assert provenance["optimized_geometry_sha256"] == provenance["oxidized_sp_input_geometry_sha256"]


def test_unconstrained_parser_rejects_restraints_and_uses_sp_minus_sp(generated, tmp_path):
    output, _, _ = generated
    rows = make_unconstrained_manifest.generate(output / "calculation_manifest.csv", tmp_path / "unconstrained.csv")
    run_root = tmp_path / "runs"
    run_calculation.run_task(rows[0], str(_fake_xtb(tmp_path / "xtb")), run_root)
    parsed = parse_unconstrained_results.parse_row(rows[0], run_root)
    assert parsed["status"] == "complete"
    assert parsed["same_geometry_pass"] is True
    assert parsed["ip_vertical_ev"] == pytest.approx(0.1 * common.HARTREE_TO_EV)


def test_heavy_atom_rmsd_and_geometry_classification_are_deterministic():
    import numpy as np

    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.3, 0.4]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = coordinates @ rotation + np.array([4.0, -2.0, 1.0])
    assert analyze_unconstrained_results.heavy_atom_rmsd(coordinates, transformed) == pytest.approx(0.0, abs=1e-12)
    assert analyze_unconstrained_results.classify_geometry((0.10, 0.20, 0.24)) == "all_same"
    assert analyze_unconstrained_results.classify_geometry((0.10, 0.40, 0.45)) == "two_same_one_distinct"
    assert analyze_unconstrained_results.classify_geometry((0.30, 0.40, 0.45)) == "three_distinct"


def test_free_min_and_lowest_energy_selection_are_independent():
    group = {
        "CAS": {"ip_vertical_ev": "5.0", "energy_neutral_sp_eh": "-10.0"},
        "CSA": {"ip_vertical_ev": "4.0", "energy_neutral_sp_eh": "-9.8"},
        "ACS": {"ip_vertical_ev": "4.5", "energy_neutral_sp_eh": "-10.2"},
    }
    minimum_ip, lowest_energy = analyze_unconstrained_results.select_topologies(group)
    assert minimum_ip == "CSA"
    assert lowest_energy == "ACS"


def test_global_offset_fit_keeps_unit_slope():
    observed = [1.0, 2.0, 4.0]
    calculated = [5.0, 7.0, 8.0]
    metric, fitted = analyze_unconstrained_results.offset_metric_row("fixture", observed, calculated)
    assert metric["slope_fixed"] == 1.0
    assert all((fit - calc) == pytest.approx(metric["offset_v"]) for fit, calc in zip(fitted, calculated))
