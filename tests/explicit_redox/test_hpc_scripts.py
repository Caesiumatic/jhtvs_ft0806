from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_isolated_launcher_is_fail_closed_and_uses_array_identity() -> None:
    text = (ROOT / "workflows" / "mace_polar_5solv_redox" / "hpc" / "run_isolated.sh").read_text()
    assert "set -euo pipefail" in text
    assert "TASK_TABLE_SHA256" in text
    assert '"$SGE_TASK_ID"' in text
    assert "polar-1-l" in text
    assert "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" in text
