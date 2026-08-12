from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert prepared["scheduler_wave_count"] == 1
    assert "MD_CHUNKS_PER_JOB=10" in prepared["command"]
    assert "REPOSITORY_COMMIT=" + prepared["repository_commit"] in prepared["command"]


def test_full_trajectory_submission_is_an_idempotent_twenty_wave_dependency_chain(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _tasks(raw)
    (raw / "pilot_trajectory_tasks.tsv").rename(raw / "validation_trajectory_tasks.tsv")
    commands = []

    def fake_run(command, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        commands.append(command)
        return SimpleNamespace(stdout=f"{1000 + len(commands)}.1-2:1\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    submitted = submit_array(
        raw_root=raw,
        scope="validation",
        stage="trajectory",
        execute=True,
        device="cuda",
    )
    assert submitted["status"] == "SUBMITTED"
    assert submitted["scheduler_wave_count"] == 20
    assert len(submitted["scheduler_job_ids"]) == 20
    assert "-hold_jid_ad" not in commands[0]
    assert commands[1][commands[1].index("-hold_jid_ad") + 1] == "1001"
    again = submit_array(
        raw_root=raw,
        scope="validation",
        stage="trajectory",
        execute=True,
        device="cuda",
    )
    assert again["status"] == "ALREADY_SUBMITTED"
    assert len(commands) == 20
