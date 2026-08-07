from __future__ import annotations

from pathlib import Path

import pytest

from jhtvs_ft0806.orca.parser import (
    STANDARD_STATE_1M_CORRECTION_EH,
    _audit_echo,
    parse_job_result,
)
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import read_csv_rows


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPOSITORY_ROOT / "spec"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "orca"


def _solvent(solvent_id: str) -> dict[str, str]:
    return next(
        row
        for row in read_csv_rows(SPEC_DIR / "solvent_smd_registry.csv")
        if row["solvent_id"] == solvent_id
    )


def _parse_real_excerpt(
    tmp_path: Path, text: str, *, include_identity: bool = True
) -> dict[str, object]:
    repository_root = tmp_path
    run_dir = repository_root / "runs" / "orca" / "optfreq" / "OFGOLD"
    run_dir.mkdir(parents=True)
    input_path = run_dir / "OFGOLD.inp"
    output_path = run_dir / "OFGOLD.out"
    geometry_path = repository_root / "runs" / "geometry" / "source.xyz"
    geometry_path.parent.mkdir(parents=True)
    geometry_path.write_bytes((FIXTURE_DIR / "small.xyz").read_bytes())
    input_path.write_text("# fixture input\n", encoding="utf-8")
    input_sha256 = sha256_file(input_path)
    (run_dir / "OFGOLD_Compound_1.xyz").write_bytes(geometry_path.read_bytes())
    manifest = {
        "job_class": "optfreq",
        "geometry_key": "fixture:geometry",
        "geometry_sha256": sha256_file(geometry_path),
        "input_path": "runs/orca/optfreq/OFGOLD/OFGOLD.inp",
        "input_sha256": input_sha256,
        "status": "ready",
    }
    job = {
        "job_id": "OFGOLD",
        "state_id": "FIX_QM1_M1",
        "solvent_id": "S001",
        "workflow_revision": "jhtvs-ft0806-optfreq-v3",
        "method_id": "T2_wB97X-D3_OptFreq_TZVPD-SP_SMD_v4",
    }
    identity = (
        f"# job_id: {job['job_id']}\n"
        f"# input_sha256: {input_sha256}\n"
        f"# workflow_revision: {job['workflow_revision']}\n"
        f"# method_id: {job['method_id']}\n"
    )
    output_path.write_text(
        (identity if include_identity else "") + text,
        encoding="utf-8",
    )
    geometry = {
        "geometry_key": "fixture:geometry",
        "xyz_path": "runs/geometry/source.xyz",
        "xyz_sha256": sha256_file(geometry_path),
    }
    return parse_job_result(
        manifest=manifest,
        job=job,
        geometry=geometry,
        solvent=_solvent("S001"),
        repository_root=repository_root,
    )


def test_real_20260707_excerpt_parses_composite_gibbs_echo_and_qc(
    tmp_path: Path,
) -> None:
    text = (FIXTURE_DIR / "optfreq_native_real_excerpt.out").read_text(
        encoding="utf-8"
    )

    result = _parse_real_excerpt(tmp_path, text)

    expected_1atm = -424.715536005354 - 0.01149123
    assert result["qc_status"] == "clean"
    assert float(result["E_freq_Eh"]) == pytest.approx(-424.71495837)
    assert float(result["E_final_SP_Eh"]) == pytest.approx(-424.715536005354)
    assert float(result["G_composite_1atm_Eh"]) == pytest.approx(expected_1atm)
    assert float(result["G_composite_1M_Eh"]) == pytest.approx(
        expected_1atm + STANDARD_STATE_1M_CORRECTION_EH
    )
    assert result["frequency_count"] == 15
    assert result["significant_imaginary_count"] == 0
    assert result["echo_qc"] == "pass"
    assert result["echo_refrac"] == "1.3442"
    assert result["connectivity_status"] == "pass"


def test_echo_mismatch_retains_value_but_requires_scientific_stop(
    tmp_path: Path,
) -> None:
    text = (FIXTURE_DIR / "optfreq_native_real_excerpt.out").read_text(
        encoding="utf-8"
    ).replace("35.6880", "36.6880")

    result = _parse_real_excerpt(tmp_path, text)

    assert result["qc_status"] == "flagged"
    assert result["scientific_stop_required"] == "true"
    assert "smd_echo_mismatch" in result["qc_reasons"]
    assert result["G_composite_1M_Eh"] != ""


def test_registry_exact_custom_refrac_is_a_required_echo_field() -> None:
    text = (
        "  Epsilon ... 46.8260\n"
        "  Refrac ... 1.4170\n"
        "  Soln ... 1.4783\n"
        "  Soln25 ... 1.4783\n"
        "  Sola ... 0.0000\n"
        "  Solb ... 0.8800\n"
        "  Solg ... 61.7800\n"
        "  Solc ... 0.0000\n"
        "  Solh ... 0.0000\n"
    )

    echoed, max_delta, echo_qc, mismatch = _audit_echo(text, _solvent("S007"))

    assert echoed["refrac"] == "1.417"
    assert float(max_delta) == pytest.approx(0.0613)
    assert echo_qc == "mismatch"
    assert mismatch


def test_missing_output_identity_is_not_accepted(tmp_path: Path) -> None:
    text = (FIXTURE_DIR / "optfreq_native_real_excerpt.out").read_text(
        encoding="utf-8"
    )
    result = _parse_real_excerpt(tmp_path, text, include_identity=False)

    assert result["qc_status"] == "missing"
    assert "output_identity_mismatch" in result["qc_reasons"]


def test_significant_imaginary_frequency_retains_composite_value(
    tmp_path: Path,
) -> None:
    text = (FIXTURE_DIR / "optfreq_native_real_excerpt.out").read_text(
        encoding="utf-8"
    ).replace("6:     341.38 cm**-1", "6:     -75.00 cm**-1")

    result = _parse_real_excerpt(tmp_path, text)

    assert result["qc_status"] == "flagged"
    assert result["significant_imaginary_count"] == 1
    assert result["most_negative_frequency_cm1"] == "-75"
    assert "significant_imaginary_frequency" in result["qc_reasons"]
    assert result["G_composite_1M_Eh"] != ""


@pytest.mark.parametrize(
    ("job_class", "solvent_id", "smd_text"),
    [
        ("diagnostic_gas_sp", "GAS", ""),
        (
            "smd_energy_sp",
            "S001",
            "  Epsilon ... 35.6880\n"
            "  Refrac ... 1.3442\n"
            "  Soln ... 1.3442\n"
            "  Soln25 ... 1.3416\n"
            "  Sola ... 0.0700\n"
            "  Solb ... 0.3200\n"
            "  Solg ... 41.2500\n"
            "  Solc ... 0.0000\n"
            "  Solh ... 0.0000\n"
            "CPCM Dielectric : -0.0123 Eh\n"
            "SMD CDS (Gcds) : 0.0004 Eh\n",
        ),
        (
            "smd_energy_sp",
            "S007",
            "  Epsilon ... 46.8260\n"
            "  Refrac ... 1.4783\n"
            "  Soln ... 1.4783\n"
            "  Soln25 ... 1.4783\n"
            "  Sola ... 0.0000\n"
            "  Solb ... 0.8800\n"
            "  Solg ... 61.7800\n"
            "  Solc ... 0.0000\n"
            "  Solh ... 0.0000\n"
            "CPCM Dielectric : -0.0123 Eh\n"
            "SMD CDS (Gcds) : 0.0004 Eh\n",
        ),
    ],
)
def test_sp_parser_distinguishes_gas_and_smd_contracts(
    tmp_path: Path, job_class: str, solvent_id: str, smd_text: str
) -> None:
    run_dir = tmp_path / "runs" / "orca" / "sp" / "SPGOLD"
    run_dir.mkdir(parents=True)
    input_path = run_dir / "SPGOLD.inp"
    output_path = run_dir / "SPGOLD.out"
    input_path.write_text("# fixture SP input\n", encoding="utf-8")
    input_sha256 = sha256_file(input_path)
    job = {
        "job_id": "SPGOLD",
        "state_id": "FIX_Q0_M1",
        "solvent_id": solvent_id,
        "workflow_revision": "jhtvs-ft0806-sp-v3",
        "method_id": "fixture-sp-method",
    }
    output_path.write_text(
        f"# job_id: {job['job_id']}\n"
        f"# input_sha256: {input_sha256}\n"
        f"# workflow_revision: {job['workflow_revision']}\n"
        f"# method_id: {job['method_id']}\n"
        + smd_text
        + "Program Version 6.1.0\n"
        + "FINAL SINGLE POINT ENERGY -40.123456789\n"
        + "ORCA TERMINATED NORMALLY\n",
        encoding="utf-8",
    )
    manifest = {
        "job_class": job_class,
        "geometry_key": "fixture:geometry",
        "geometry_sha256": "fixture-geometry-sha",
        "input_path": "runs/orca/sp/SPGOLD/SPGOLD.inp",
        "input_sha256": input_sha256,
        "status": "ready",
    }

    result = parse_job_result(
        manifest=manifest,
        job=job,
        geometry=None,
        solvent=None if solvent_id == "GAS" else _solvent(solvent_id),
        repository_root=tmp_path,
    )

    assert result["qc_status"] == "clean"
    assert float(result["final_energy_Eh"]) == pytest.approx(-40.123456789)
    assert result["orca_version"] == "6.1.0"
    assert result["echo_qc"] == (
        "not_applicable" if solvent_id == "GAS" else "pass"
    )
