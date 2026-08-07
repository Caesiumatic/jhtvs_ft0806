from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jhtvs_ft0806.labels.assembly import (
    HARTREE_TO_EV,
    KCAL_PER_HARTREE,
    aggregate_stoichiometric,
    assemble_labels,
    load_reference_conversion,
    parse_stoichiometry,
)
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"
RESULT_FIELDS = (
    "job_id",
    "geometry_sha256",
    "final_energy_Eh",
    "cpcm_dielectric_Eh",
    "smd_cds_Eh",
    "G_composite_1M_Eh",
    "normal_termination",
    "output_sha256",
    "qc_status",
    "qc_reasons",
    "scientific_stop_required",
)
REFERENCE_SOURCE = """\
SHE_ABS_V = 4.28
AGCL_VS_SHE_V = 0.197

def v_vs_agcl(delta_g_ev: float | None) -> float | None:
    return None if delta_g_ev is None else delta_g_ev - SHE_ABS_V - AGCL_VS_SHE_V
"""


def _result(
    job_id: str,
    geometry_hash: str,
    *,
    energy: float | None = None,
    cpcm: float | None = None,
    cds: float | None = None,
    gibbs: float | None = None,
    status: str = "clean",
    reasons: str = "",
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "geometry_sha256": geometry_hash,
        "final_energy_Eh": "" if energy is None else energy,
        "cpcm_dielectric_Eh": "" if cpcm is None else cpcm,
        "smd_cds_Eh": "" if cds is None else cds,
        "G_composite_1M_Eh": "" if gibbs is None else gibbs,
        "normal_termination": "true",
        "output_sha256": f"output-{job_id}",
        "qc_status": status,
        "qc_reasons": reasons,
        "scientific_stop_required": "false",
    }


def _assemble_fixture(tmp_path: Path, *, flagged_radical: bool = False):
    reference_path = tmp_path / "parse_orca.py"
    reference_path.write_text(REFERENCE_SOURCE, encoding="utf-8")
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    state_results_path = tmp_path / "state_results.csv"
    radical_status = "flagged" if flagged_radical else "clean"
    radical_reasons = "significant_imaginary_frequency" if flagged_radical else ""
    write_csv_deterministic(
        state_results_path,
        RESULT_FIELDS,
        [
            _result("SP0041", "geom-d", energy=-199.4, cpcm=-0.2, cds=0.01),
            _result("SP0045", "geom-m0", energy=-100.0, cpcm=-0.1, cds=0.005),
            _result(
                "SP0046",
                "geom-mp",
                energy=-99.8,
                cpcm=-0.09,
                cds=0.006,
                status=radical_status,
                reasons=radical_reasons,
            ),
            _result("OF003", "geom-d", gibbs=-199.5),
            _result("OF004", "geom-m0", gibbs=-100.1),
            _result("OF005", "geom-mp", gibbs=-99.85),
        ],
        sort_by=("job_id",),
    )
    baseline_path = tmp_path / "base_state_energies.csv"
    write_csv_deterministic(
        baseline_path,
        ("state_id", "solvent_id", "geometry_hash", "E_base_MACE_eV"),
        [
            {
                "state_id": "D001_QP2_M1",
                "solvent_id": "S001",
                "geometry_hash": "geom-d",
                "E_base_MACE_eV": -17.0,
            },
            {
                "state_id": "M001_Q0_M1",
                "solvent_id": "S001",
                "geometry_hash": "geom-m0",
                "E_base_MACE_eV": -10.0,
            },
            {
                "state_id": "M001_QP1_M2",
                "solvent_id": "S001",
                "geometry_hash": "geom-mp",
                "E_base_MACE_eV": -9.0,
            },
        ],
        sort_by=("state_id",),
    )
    outputs = {
        "state": tmp_path / "state_sp_labels.csv",
        "sp": tmp_path / "reaction_sp_labels.csv",
        "final": tmp_path / "reaction_final_labels.csv",
    }
    summary = assemble_labels(
        spec_dir=SPEC_DIR,
        state_results_path=state_results_path,
        baseline_state_energies_path=baseline_path,
        state_sp_output_path=outputs["state"],
        reaction_sp_output_path=outputs["sp"],
        reaction_final_output_path=outputs["final"],
        reference_conversion_path=reference_path,
        reference_conversion_sha256=reference_sha,
    )
    return summary, outputs


def _row(path: Path, reaction_id: str, solvent_id: str) -> dict[str, str]:
    return next(
        row
        for row in read_csv_rows(path)
        if row["reaction_id"] == reaction_id and row["solvent_id"] == solvent_id
    )


def test_stoichiometry_signs_and_energy_units() -> None:
    redox = parse_stoichiometry("M001_Q0_M1:-1;M001_QP1_M2:+1")
    sigma = parse_stoichiometry("M001_QP1_M2:-2;D001_QP2_M1:+1")

    assert aggregate_stoichiometric(
        redox, {"M001_Q0_M1": -100.0, "M001_QP1_M2": -99.8}
    ) == pytest.approx(0.2)
    assert aggregate_stoichiometric(
        sigma, {"M001_QP1_M2": -99.8, "D001_QP2_M1": -199.4}
    ) == pytest.approx(0.2)
    assert 0.2 * HARTREE_TO_EV == pytest.approx(5.4422772491976)
    assert 0.2 * KCAL_PER_HARTREE == pytest.approx(125.50189481262)


def test_reference_conversion_is_called_from_hash_pinned_source(tmp_path: Path) -> None:
    source = tmp_path / "parse_orca.py"
    source.write_text(REFERENCE_SOURCE, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    conversion = load_reference_conversion(source, expected_sha256=digest)

    assert conversion(None) is None
    assert conversion(5.0) == pytest.approx(0.523)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reference_conversion(source, expected_sha256="0" * 64)


def test_assemble_labels_uses_exact_reaction_stoichiometry_and_schemas(
    tmp_path: Path,
) -> None:
    summary, outputs = _assemble_fixture(tmp_path)

    assert summary.state_sp_rows == 705
    assert summary.reaction_sp_rows == 403
    assert summary.reaction_final_rows == 50
    assert summary.state_sp_clean == 3
    assert summary.reaction_sp_clean == 2
    assert summary.reaction_final_clean == 2

    redox_sp = _row(outputs["sp"], "RXN_MOX_M001", "S001")
    sigma_sp = _row(outputs["sp"], "RXN_SIG_M001", "S001")
    assert float(redox_sp["deltaE_base_MACE_rxn_eV"]) == pytest.approx(1.0)
    assert float(redox_sp["deltaE_T2_SMD_SP_rxn_eV"]) == pytest.approx(
        0.2 * HARTREE_TO_EV
    )
    assert float(sigma_sp["deltaE_T2_SMD_SP_rxn_eV"]) == pytest.approx(
        0.2 * HARTREE_TO_EV
    )

    redox_final = _row(outputs["final"], "RXN_MOX_M001", "S001")
    sigma_final = _row(outputs["final"], "RXN_SIG_M001", "S001")
    redox_delta_g = 0.25 * HARTREE_TO_EV
    assert float(redox_final["deltaG_T2_SMD_min_rxn_eV"]) == pytest.approx(
        redox_delta_g
    )
    assert float(redox_final["rt_correction_eV"]) == pytest.approx(
        0.05 * HARTREE_TO_EV
    )
    assert float(redox_final["final_residual_eV"]) == pytest.approx(
        redox_delta_g - 1.0
    )
    assert float(redox_final["Eox_vs_AgAgCl_V"]) == pytest.approx(
        redox_delta_g - 4.28 - 0.197
    )
    assert float(sigma_final["deltaG_sigma_kcal_mol"]) == pytest.approx(
        0.2 * KCAL_PER_HARTREE
    )
    assert redox_final["complete_tuple"] == "true"
    assert sigma_final["complete_tuple"] == "true"


def test_flagged_source_retains_values_outside_clean_labels(tmp_path: Path) -> None:
    summary, outputs = _assemble_fixture(tmp_path, flagged_radical=True)

    redox_sp = _row(outputs["sp"], "RXN_MOX_M001", "S001")
    redox_final = _row(outputs["final"], "RXN_MOX_M001", "S001")
    assert summary.reaction_sp_flagged == 2
    assert summary.reaction_final_flagged == 2
    assert redox_sp["complete_tuple"] == "true"
    assert redox_sp["sp_residual_eV"] != ""
    assert redox_sp["qc_status"] == "flagged"
    assert "SP0046:significant_imaginary_frequency" in redox_sp["qc_reasons"]
    assert redox_final["complete_tuple"] == "true"
    assert redox_final["final_residual_eV"] != ""
    assert redox_final["qc_status"] == "flagged"
