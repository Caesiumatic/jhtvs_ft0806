from __future__ import annotations

import json

from jhtvs_ft0806.explicit_redox.device_probe import compare_probes


def test_device_comparison_requires_same_coordinates_and_tight_float64_agreement(tmp_path) -> None:
    base = {
        "geometry_sha256": "same",
        "calculator": {"checkpoint_sha256": "checkpoint"},
        "energy_eV": -10.0,
        "forces_eV_A": [[1.0, 2.0, 3.0]],
        "evaluation_seconds": 5.0,
    }
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(json.dumps(base), encoding="utf-8")
    gpu.write_text(
        json.dumps(
            {
                **base,
                "energy_eV": -10.0 + 1e-10,
                "forces_eV_A": [[1.0, 2.0 + 1e-9, 3.0]],
                "evaluation_seconds": 0.5,
            }
        ),
        encoding="utf-8",
    )
    result = compare_probes(cpu, gpu)
    assert result["status"] == "PASS"
    assert result["gpu_evaluation_seconds"] == 0.5


def test_device_comparison_fails_closed_on_different_coordinates(tmp_path) -> None:
    base = {
        "geometry_sha256": "cpu-geometry",
        "calculator": {"checkpoint_sha256": "checkpoint"},
        "energy_eV": -10.0,
        "forces_eV_A": [[1.0, 2.0, 3.0]],
        "evaluation_seconds": 1.0,
    }
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(json.dumps(base), encoding="utf-8")
    gpu.write_text(json.dumps({**base, "geometry_sha256": "gpu-geometry"}), encoding="utf-8")
    try:
        compare_probes(cpu, gpu)
    except RuntimeError as exc:
        assert "different coordinates" in str(exc)
    else:
        raise AssertionError("different device-probe coordinates must fail closed")
