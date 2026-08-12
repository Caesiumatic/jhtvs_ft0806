from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]


def test_isolated_launcher_is_fail_closed_and_uses_array_identity() -> None:
    text = (ROOT / "workflows" / "mace_polar_5solv_redox" / "hpc" / "run_isolated.sh").read_text()
    assert "set -euo pipefail" in text
    assert "TASK_TABLE_SHA256" in text
    assert '"$SGE_TASK_ID"' in text
    assert "polar-1-l" in text
    assert "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" in text


def test_trajectory_launcher_is_fail_closed_and_mode_scoped() -> None:
    text = (ROOT / "workflows" / "mace_polar_5solv_redox" / "hpc" / "run_trajectory.sh").read_text()
    assert "set -euo pipefail" in text
    assert "TRAJECTORY_MODE" in text
    assert "MACE_DEVICE" in text
    assert "MD_CHUNKS_PER_JOB" in text
    assert "--max-md-chunks" in text
    assert "CONDA_ENV_NAME" in text
    assert '${TRAJECTORY_MODE}_trajectory_tasks.tsv' in text
    assert "TASK_TABLE_SHA256" in text
    assert "REPOSITORY_COMMIT" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "flock -n" in text
    assert '"$SGE_TASK_ID"' in text
    assert "polar-1-l" in text
    assert "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" in text


def test_gap_launcher_uses_same_frozen_trajectory_identity() -> None:
    text = (ROOT / "workflows" / "mace_polar_5solv_redox" / "hpc" / "run_gap.sh").read_text()
    assert "set -euo pipefail" in text
    assert '${TRAJECTORY_MODE}_trajectory_tasks.tsv' in text
    assert "TASK_TABLE_SHA256" in text
    assert "REPOSITORY_COMMIT" in text
    assert '"$SGE_TASK_ID"' in text
    assert "evaluate-gaps" in text
    assert "MACE_DEVICE" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "flock -n" in text


def test_device_probe_launcher_activates_requested_isolated_environment() -> None:
    text = (ROOT / "workflows" / "mace_polar_5solv_redox" / "hpc" / "run_device_probe.sh").read_text()
    assert "set -euo pipefail" in text
    assert "CONDA_ENV_NAME" in text
    assert "PROBE_DEVICE" in text
    assert 'LD_LIBRARY_PATH="$CONDA_PREFIX/lib:' in text
    assert "device_probe run" in text


def test_isolated_submission_matches_frozen_task_scope() -> None:
    payload = json.loads(
        (ROOT / "workflows" / "mace_polar_5solv_redox" / "isolated_submission.json").read_text()
    )
    assert payload["array_task_count"] == 120
    assert payload["scheduler_job_id"].isdecimal()
    assert payload["task_table_sha256"] == "2d156affd73f8518ea978c87f0d761ea0d7cfcae7c963348009875ceca0f08cc"
