from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.dft_cluster_benchmark import analyze_results, common, make_manifest, orca_input, parse_results, run_case  # noqa: E402


@pytest.fixture(scope="module")
def manifest_rows(tmp_path_factory):
    output = tmp_path_factory.mktemp("dft_manifest") / "manifest.csv"
    return make_manifest.generate(
        ROOT / "data/chauhan_cation_eox/calculation_manifest.csv",
        ROOT / "data/fadel_benchmark/calculation_manifest.csv",
        output,
    )


def test_manifest_has_exact_authorized_case_counts(manifest_rows):
    counts = Counter((row["benchmark"], row["kind"]) for row in manifest_rows)
    assert len(manifest_rows) == 145
    assert counts == {("chauhan", "as_pair"): 9, ("chauhan", "triad"): 72, ("fadel", "as_pair"): 16, ("fadel", "triad"): 48}

    def charge_spin(row):
        return tuple(int(row[field]) for field in ("charge_reduced", "multiplicity_reduced", "charge_oxidized", "multiplicity_oxidized"))

    assert all(charge_spin(row) == (-1, 1, 0, 2) for row in manifest_rows if row["kind"] == "as_pair")
    assert all(charge_spin(row) == (0, 1, 1, 2) for row in manifest_rows if row["kind"] == "triad")


def test_orca_inputs_preserve_method_environment_and_restraint_scope(manifest_rows):
    chauhan_triad = next(row for row in manifest_rows if row["benchmark"] == "chauhan" and row["kind"] == "triad")
    fadel_pair = next(row for row in manifest_rows if row["benchmark"] == "fadel" and row["kind"] == "as_pair")
    triad_xyz = ROOT / chauhan_triad["input_xyz"]
    pair_xyz = ROOT / fadel_pair["input_xyz"]
    opt = orca_input.build_input(chauhan_triad, "reduced_opt", triad_xyz)
    reduced = orca_input.build_input(chauhan_triad, "reduced_sp", triad_xyz)
    oxidized = orca_input.build_input(chauhan_triad, "oxidized_sp", triad_xyz)
    vacuum = orca_input.build_input(fadel_pair, "reduced_opt", pair_xyz)
    assert "! aug-cc-pVTZ RIJCOSX AutoAux" in opt
    assert "Exchange hyb_mgga_x_m06_hf" in opt
    assert "Correlation mgga_c_m06_hf" in opt
    assert "! M06-HF" not in opt
    assert chauhan_triad["functional_implementation"] == "LibXC"
    assert "%cpcm" in opt and "epsilon 65.00000000" in opt
    assert "%cpcm" in reduced and "%cpcm" in oxidized
    assert "BIAS" in opt and "BIAS" not in reduced and "BIAS" not in oxidized
    assert "* xyzfile 0 1 in.xyz" in opt
    assert "* xyzfile 1 2 in.xyz" in oxidized
    assert "%cpcm" not in vacuum
    assert "BIAS" not in vacuum
    assert "* xyzfile -1 1 in.xyz" in vacuum


def test_orca_bias_matches_original_local_harmonic_curvature(manifest_rows):
    row = next(row for row in manifest_rows if row["kind"] == "triad")
    payload = orca_input.bias_payload(row, ROOT / row["input_xyz"])
    assert len(payload) == 2
    expected = common.harmonic_force_kcal_mol_angstrom2(0.005)
    for bias in payload:
        recovered = 2 * float(bias["depth_kcal_mol"]) * float(bias["alpha_angstrom_inv"]) ** 2
        assert recovered == pytest.approx(expected, rel=1e-12)


def test_parse_orca_energy_mulliken_and_spin():
    text = """
Program Version 6.1.0
SCF CONVERGED AFTER 12 CYCLES
FINAL SINGLE POINT ENERGY     -100.250000000000
MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS
---------------------------------------------
  0 C :   -0.250000   0.500000
  1 H :    0.250000   0.500000
Sum of atomic charges: 0.000000
Expectation value of <S**2> : 0.752000
ORCA TERMINATED NORMALLY
"""
    assert parse_results.parse_energy(text) == pytest.approx(-100.25)
    assert parse_results.parse_mulliken_charges(text, 2) == pytest.approx([-0.25, 0.25])
    assert parse_results.parse_s2(text, 2) == pytest.approx(0.752)
    assert parse_results.parse_s2("", 1) == 0.0


def test_metadata_fragments_cover_each_case_without_overlap(manifest_rows):
    for row in manifest_rows:
        metadata = common.load_metadata(ROOT / row["input_xyz"])
        indices = [index for fragment in metadata["fragments"].values() for index in fragment["atom_indices_zero_based"]]
        atoms, _ = common.read_xyz(ROOT / row["input_xyz"])
        assert sorted(indices) == list(range(len(atoms)))
        assert set(metadata["fragments"]) == ({"A", "S"} if row["kind"] == "as_pair" else {"C", "A", "S"})


def test_runner_uses_identical_optimized_geometry_for_both_single_points(manifest_rows, tmp_path):
    fake_orca = tmp_path / "orca"
    fake_orca.write_text(
        """#!/usr/bin/env python3
import pathlib
import shutil
import sys

text = pathlib.Path(sys.argv[1]).read_text()
print('Program Version 6.1.0')
print('SCF CONVERGED AFTER 8 CYCLES')
if any(line.startswith('! ') and ' Opt' in line for line in text.splitlines()):
    shutil.copy2('in.xyz', 'orca.xyz')
    print('THE OPTIMIZATION HAS CONVERGED')
print('FINAL SINGLE POINT ENERGY     -100.000000000000')
print('ORCA TERMINATED NORMALLY')
""",
        encoding="utf-8",
    )
    fake_orca.chmod(0o755)
    row = next(row for row in manifest_rows if row["benchmark"] == "fadel" and row["kind"] == "as_pair")
    provenance = run_case.run_case(row, str(fake_orca), tmp_path / "runs")
    assert provenance["status"] == "complete"
    assert provenance["same_sp_geometry"] is True
    assert provenance["optimized_geometry_sha256"] == provenance["reduced_sp_input_geometry_sha256"] == provenance["oxidized_sp_input_geometry_sha256"]


def _synthetic_task_results(manifest_rows):
    topology_shift = {"AS": 0.0, "CAS": 0.3, "CSA": 0.2, "ACS": 0.1}
    rows = []
    for index, source in enumerate(manifest_rows):
        rows.append({
            **source,
            "ip_vertical_ev": 6.0 + 0.01 * index + topology_shift[source["topology"]],
            "oxidized_fragment": "A" if source["topology"] in {"AS", "ACS"} else "S",
            "status": "complete",
        })
    return rows


def test_analysis_aggregation_has_exact_compositions_and_minimum(manifest_rows):
    tasks = _synthetic_task_results(manifest_rows)
    chauhan = analyze_results.build_chauhan_summary(tasks, common.read_csv(ROOT / "workflows/chauhan_cation_eox/benchmark_chauhan.csv"))
    fadel = analyze_results.build_fadel_summary(tasks, common.read_csv(ROOT / "data/fadel_benchmark/fadel_table2_reference.csv"))
    assert len(chauhan) == 24
    assert len(fadel) == 16
    for row in chauhan + fadel:
        minimum = min(common.TOPOLOGIES, key=lambda topology: float(row[f"dft_ip_{topology}_ev"]))
        assert row["topology_of_min"] == minimum
        assert float(row["dft_ip_min_ev"]) == pytest.approx(float(row[f"dft_ip_{minimum}_ev"]))


def test_metric_tables_use_one_fixed_unit_slope_offset():
    rows = [
        {"reference": 5.0, **{f"prediction_{key}": value for _, key in analyze_results.DESCRIPTORS}}
        for value in (6.0, 7.0, 9.0, 10.0)
    ]
    for index, row in enumerate(rows):
        row["reference"] = (5.0, 6.0, 8.0, 9.0)[index]
    raw, offsets, corrected = analyze_results.metric_tables(rows, "reference", "prediction_")
    assert len(raw) == len(offsets) == 6
    assert all(row["slope_fixed"] == 1.0 and row["offset"] == pytest.approx(-1.0) for row in offsets)
    assert corrected["AS"] == pytest.approx([5.0, 6.0, 8.0, 9.0])
