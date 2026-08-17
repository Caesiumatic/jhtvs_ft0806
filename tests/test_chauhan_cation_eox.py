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
import build_structures  # noqa: E402
import common  # noqa: E402
import make_manifest  # noqa: E402
import parse_results  # noqa: E402
import run_calculation  # noqa: E402


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("chauhan_structures")
    structures = build_structures.generate(output)
    calculations = make_manifest.generate(output / "structure_manifest.csv", output / "calculation_manifest.csv")
    return output, structures, calculations


def test_exact_species_and_composition_matrix():
    species = common.species_table()
    assert {key: float(species[key]["epsilon"]) for key in common.SOLVENTS} == {"PC": 65.0, "EG": 37.0, "THF": 7.6}
    assert common.CATIONS_BY_ANION == {
        "NTF2": ("EMIM", "BMIM", "HMIM"),
        "OTF": ("EMIM", "HMIM", "BMPYRR"),
        "PF6": ("BMIM", "HMIM"),
    }
    assert len(common.composition_keys()) == 24
    assert len(common.benchmark_rows()) == 24
    assert len({(r["cation"], r["anion"], r["solvent"]) for r in common.benchmark_rows()}) == 24


def test_structure_and_calculation_counts_and_charges(generated):
    _, structures, calculations = generated
    assert len([r for r in structures if r["kind"] == "triad"]) == 72
    assert len([r for r in structures if r["kind"] == "as_pair"]) == 9
    assert len([r for r in structures if r["kind"] == "solvent"]) == 3
    assert len([r for r in structures if r["kind"] == "anion"]) == 9
    assert len(calculations) == 93
    assert len(calculations) * 2 == 186
    assert all(int(r["formal_charge"]) == 0 for r in structures if r["kind"] == "triad")
    assert all(int(r["formal_charge"]) == -1 for r in structures if r["kind"] == "as_pair")


def test_vertical_state_assignments(generated):
    _, _, calculations = generated
    for row in calculations:
        if row["kind"] in {"triad", "solvent"}:
            assert (int(row["charge_reduced"]), int(row["uhf_reduced"])) == (0, 0)
            assert (int(row["charge_oxidized"]), int(row["uhf_oxidized"])) == (1, 1)
        else:
            assert (int(row["charge_reduced"]), int(row["uhf_reduced"])) == (-1, 0)
            assert (int(row["charge_oxidized"]), int(row["uhf_oxidized"])) == (0, 1)


def test_geometry_generation_is_deterministic(tmp_path):
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
        metrics = parse_results.geometry_metrics(atoms, metadata)
        assert metrics["inferred_topology"] == row["topology"]
        assert metadata["minimum_heavy_distance_ang"] >= build_structures.MIN_HEAVY_DISTANCE_ANG


def test_restraint_targets_only_two_adjacent_anchors(generated):
    output, structures, calculations = generated
    row = next(r for r in calculations if r["kind"] == "triad" and r["topology"] == "CSA")
    metadata = common.load_metadata(ROOT / row["input_xyz"] if (ROOT / row["input_xyz"]).exists() else output / "initial" / f"{row['task_id']}.xyz")
    text = run_calculation.restraint_text(metadata, "CSA", make_manifest.RESTRAINT_FORCE)
    assert "force constant=0.02000000" in text
    assert text.count("distance:") == 2
    assert "atoms:" not in text


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
    row = next(r for r in calculations if r["kind"] == "triad")
    run_root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="exit code 7"):
        run_calculation.run_task(row, str(fake_xtb), run_root)
    provenance = json.loads((run_root / row["task_id"] / "provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["reduced_executed_this_invocation"] is True
    assert provenance["oxidized_executed_this_invocation"] is False
    assert provenance["reduced_command"][-2:] == ["--input", "xcontrol.inp"]


def test_aggregation_formulas(tmp_path):
    triad_path = tmp_path / "triads.csv"
    reference_path = tmp_path / "references.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    output = tmp_path / "summary.csv"
    with triad_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cation", "anion", "solvent", "topology", "ip_vertical_ev", "oxidized_fragment"])
        writer.writeheader()
        for topology, value, fragment in [("CAS", 6.0, "A"), ("CSA", 5.5, "S"), ("ACS", 6.5, "C")]:
            writer.writerow({"cation": "EMIM", "anion": "NTF2", "solvent": "PC", "topology": topology, "ip_vertical_ev": value, "oxidized_fragment": fragment})
    with reference_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anion", "solvent", "ip_as_direct_ev"])
        writer.writeheader()
        writer.writerow({"anion": "NTF2", "solvent": "PC", "ip_as_direct_ev": 5.0})
    with benchmark_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cation", "anion", "solvent", "eox_exp_v", "eox_sd_v"])
        writer.writeheader()
        writer.writerow({"cation": "EMIM", "anion": "NTF2", "solvent": "PC", "eox_exp_v": 2.8, "eox_sd_v": 0.15})
    rows = aggregate_results.aggregate(triad_path, reference_path, benchmark_path, output)
    row = rows[0]
    assert row["ip_min_ev"] == 5.5
    assert row["ip_mean_ev"] == 6.0
    assert row["ip_span_ev"] == 1.0
    assert row["topology_of_min_ip"] == "CSA"
    assert row["delta_ip_min_vs_AS"] == 0.5
    assert row["status"] == "complete"
