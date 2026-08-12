from __future__ import annotations

import csv
import json
from pathlib import Path

from jhtvs_ft0806.explicit_redox.workflow import submit_array, workflow_status


def _tasks(raw: Path) -> None:
    fields = ["task_index", "logical_trajectory_id"]
    with (raw / "pilot_trajectory_tasks.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {"task_index": 1, "logical_trajectory_id": "one"},
                {"task_index": 2, "logical_trajectory_id": "two"},
            ]
        )


def test_status_and_submission_preflight_are_task_hash_scoped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _tasks(raw)
    (raw / "trajectories" / "one").mkdir(parents=True)
    (raw / "trajectories" / "one" / "trajectory.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    status = workflow_status(raw_root=raw, scope="pilot")
    assert status["trajectory_complete"] == 1
    assert status["pending_trajectory_indices"] == [2]
    prepared = submit_array(raw_root=raw, scope="pilot", stage="trajectory", execute=False)
    assert prepared["status"] == "PREPARED"
    assert prepared["task_count"] == 2
    assert "TASK_TABLE_SHA256=" + prepared["task_table_sha256"] in prepared["command"]
    gpu = submit_array(
        raw_root=raw,
        scope="pilot",
        stage="gap",
        execute=False,
        max_concurrent=4,
        device="cuda",
    )
    assert "-l gpu=1,slots_gpu=1" in gpu["command"]
    assert "MACE_DEVICE=cuda" in gpu["command"]
