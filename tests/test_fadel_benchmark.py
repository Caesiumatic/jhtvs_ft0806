from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows.fadel_benchmark import analyze_results, build_structures, common, make_manifest, run_calculation  # noqa: E402


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("fadel_benchmark")
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
cwd = pathlib.Path.cwd()
atoms = int((cwd / 'in.xyz').read_text().splitlines()[0])
charge = int(sys.argv[sys.argv.index('--chrg') + 1])
if cwd.name == 'reduced_opt':
    energy = -100.5
    shutil.copy2(cwd / 'in.xyz', cwd / 'xtbopt.xyz')
elif cwd.name == 'reduced_sp':
    energy = -100.0
else:
    energy = -99.75
(cwd / 'charges').write_text('\\n'.join([str(charge / atoms)] * atoms) + '\\n')
print(f'| TOTAL ENERGY {energy:.12f} Eh |')
print('normal termination of xtb')
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _fake_xtb_with_recovery(path: Path) -> Path:
    executable = _fake_xtb(path)
    text = executable.read_text(encoding="utf-8")
    text = text.replace(
        "cwd = pathlib.Path.cwd()\n",
        "cwd = pathlib.Path.cwd()\n"
        "if '--opt' in sys.argv and '--restart' not in sys.argv and not (cwd / '.failed_once').exists():\n"
        "    (cwd / '.failed_once').write_text('1')\n"
        "    print('SCC did not converge')\n"
        "    raise SystemExit(1)\n",
    )
    executable.write_text(text, encoding="utf-8")
    return executable


def test_exact_fadel_table2_reference_values():
    rows = common.read_csv(ROOT / "data" / "fadel_benchmark" / "fadel_table2_reference.csv")
    observed = {(row["anion"], row["solvent"]): float(row["fadel_ip_dscf_mean_ev"]) for row in rows}
    assert len(rows) == 16
    assert observed == {
        ("TDI", "DMSO"): 5.85, ("TFSI", "DMSO"): 6.66, ("BF4", "DMSO"): 6.24, ("PF6", "DMSO"): 6.41,
        ("TDI", "DME"): 5.76, ("TFSI", "DME"): 7.28, ("BF4", "DME"): 7.20, ("PF6", "DME"): 7.44,
        ("TDI", "PC"): 5.90, ("TFSI", "PC"): 7.52, ("BF4", "PC"): 8.75, ("PF6", "PC"): 9.00,
        ("TDI", "ACN"): 5.77, ("TFSI", "ACN"): 7.38, ("BF4", "ACN"): 9.28, ("PF6", "ACN"): 9.54,
    }
    assert all(row["reference_environment"] == "vacuum" for row in rows)


def test_exact_task_counts_and_li_formal_charge(generated):
    _, structures, calculations = generated
    pairs = [row for row in calculations if row["kind"] == "as_pair"]
    triads = [row for row in calculations if row["kind"] == "triad"]
    assert len(pairs) == 16
    assert len(triads) == 48
    assert len(structures) == 64
    assert all(row["environment"] == "vacuum" for row in calculations)
    assert all((row["charge_reduced"], row["uhf_reduced"], row["charge_oxidized"], row["uhf_oxidized"]) == (-1, 0, 0, 1) for row in pairs)
    assert all((row["charge_reduced"], row["uhf_reduced"], row["charge_oxidized"], row["uhf_oxidized"]) == (0, 0, 1, 1) for row in triads)
    assert all(row["formal_charge"] == 0 and row["cation"] == "Li" for row in structures if row["kind"] == "triad")
    assert common.species_table()["TDI"]["smiles"] == "C(#N)C1=C(N=C([N-]1)C(F)(F)F)C#N"


def test_vacuum_commands_and_optimization_only_triad_restraint(generated, tmp_path):
    _, _, calculations = generated
    fake_xtb = str(_fake_xtb(tmp_path / "xtb"))
    for row in (next(row for row in calculations if row["kind"] == "as_pair"), next(row for row in calculations if row["kind"] == "triad")):
        provenance = run_calculation.run_task(row, fake_xtb, tmp_path / "runs")
        commands = (provenance["optimization_command"], provenance["reduced_sp_command"], provenance["oxidized_sp_command"])
        assert all(not any(flag in command for flag in ("--cosmo", "--alpb", "--gbsa")) for command in commands)
        assert all(command[command.index("--iterations") + 1] == "500" for command in commands)
        assert "--opt" in commands[0] and all("--opt" not in command for command in commands[1:])
        assert provenance["same_geometry_reduced_sp"] is True
        assert provenance["same_geometry_oxidized_sp"] is True
        if row["kind"] == "as_pair":
            assert all("--input" not in command for command in commands)
        else:
            assert "--input" in commands[0] and all("--input" not in command for command in commands[1:])


def test_scc_recovery_returns_to_default_temperature_for_final_optimization(generated, tmp_path):
    _, _, calculations = generated
    row = next(row for row in calculations if row["kind"] == "triad")
    provenance = run_calculation.run_task(row, str(_fake_xtb_with_recovery(tmp_path / "xtb")), tmp_path / "runs")
    recovery = provenance["optimization_recovery"]
    assert recovery["used"] is True
    assert recovery["warmstart_command"][-2:] == ["--etemp", "1000"]
    assert "--restart" in recovery["restart_optimization_command"]
    assert "--etemp" not in recovery["restart_optimization_command"]
    assert provenance["same_geometry_reduced_sp"] is True
    assert provenance["same_geometry_oxidized_sp"] is True


def test_triad_min_error_sign_and_paired_improvement():
    reference = [{"anion": "TDI", "solvent": "DMSO", "fadel_ip_dscf_mean_ev": "6.0"}]
    tasks = [
        {"anion": "TDI", "solvent": "DMSO", "initial_topology": "AS", "ip_vertical_ev": "7.0", "oxidized_fragment": "A", "status": "complete"},
        {"anion": "TDI", "solvent": "DMSO", "initial_topology": "CAS", "ip_vertical_ev": "6.8", "oxidized_fragment": "A", "status": "complete"},
        {"anion": "TDI", "solvent": "DMSO", "initial_topology": "CSA", "ip_vertical_ev": "6.4", "oxidized_fragment": "S", "status": "complete"},
        {"anion": "TDI", "solvent": "DMSO", "initial_topology": "ACS", "ip_vertical_ev": "6.6", "oxidized_fragment": "C", "status": "complete"},
    ]
    with pytest.raises(ValueError, match="64 complete tasks"):
        analyze_results.build_comparison(tasks, reference)
    padded_tasks, padded_reference = [], []
    for solvent_index, solvent in enumerate(common.SOLVENTS):
        for anion_index, anion in enumerate(common.ANIONS):
            shift = 0.01 * (4 * solvent_index + anion_index)
            padded_reference.append({"anion": anion, "solvent": solvent, "fadel_ip_dscf_mean_ev": str(6.0 + shift)})
            for row in tasks:
                padded_tasks.append({**row, "anion": anion, "solvent": solvent, "ip_vertical_ev": str(float(row["ip_vertical_ev"]) + shift)})
    rows = analyze_results.build_comparison(padded_tasks, padded_reference)
    first = rows[0]
    assert first["xtb_triad_min_ev"] == pytest.approx(6.4)
    assert first["triad_min_topology"] == "CSA"
    assert first["error_as_ev"] == pytest.approx(1.0)
    assert first["error_triad_min_ev"] == pytest.approx(0.4)
    assert first["triad_improvement_over_as_ev"] == pytest.approx(0.6)


def test_offset_only_fit_has_fixed_unit_slope():
    observed = [5.0, 6.0, 8.0, 9.0]
    predicted = [6.0, 7.0, 9.0, 10.0]
    metric, corrected = analyze_results.offset_metric_row("fixture", observed, predicted)
    assert metric["slope_fixed"] == 1.0
    assert metric["offset_ev"] == pytest.approx(-1.0)
    assert corrected == pytest.approx(observed)
    assert metric["MAE_after_offset_ev"] == pytest.approx(0.0)
