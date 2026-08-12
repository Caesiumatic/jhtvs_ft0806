from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from jhtvs_ft0806.explicit_redox.workflow import (
    _active_array_indices,
    _array_range,
    _expand_sge_task_spec,
    resume_array,
    submit_array,
    workflow_status,
)


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
    assert "gpu1@compute-1-21.local" in gpu["command"]
    assert "gpu2@compute-1-22.local" in gpu["command"]
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


def test_resume_submits_only_pending_indices_once_on_safe_gpus(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _tasks(raw)
    digest = __import__("hashlib").sha256(
        (raw / "pilot_trajectory_tasks.tsv").read_bytes()
    ).hexdigest()
    submissions = raw / "submissions"
    submissions.mkdir()
    (submissions / "pilot_trajectory.json").write_text(
        json.dumps({"status": "SUBMITTED", "task_table_sha256": digest}),
        encoding="utf-8",
    )
    complete = raw / "trajectories" / "one"
    complete.mkdir(parents=True)
    (complete / "trajectory.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    commands = []

    def fake_run(command, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="b" * 40 + "\n", returncode=0)
        if command[:2] == ["qstat", "-j"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        commands.append(command)
        return SimpleNamespace(stdout="2001.2\n", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    submitted = resume_array(
        raw_root=raw,
        scope="pilot",
        stage="trajectory",
        execute=True,
    )
    assert submitted["pending_indices"] == [2]
    assert commands[0][commands[0].index("-t") + 1] == "1-1"
    assert any("TASK_INDEX_MAP=2" in part for part in commands[0])
    assert any("gpu1@compute-1-21.local" in part for part in commands[0])
    again = resume_array(
        raw_root=raw,
        scope="pilot",
        stage="trajectory",
        execute=True,
    )
    assert again["status"] == "ALREADY_SUBMITTED"
    assert len(commands) == 1


def test_scheduler_array_range_compresses_only_contiguous_indices() -> None:
    assert _array_range([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"


def test_scheduler_task_spec_expansion() -> None:
    assert _expand_sge_task_spec("1,3-7:2,9-10") == {1, 3, 5, 7, 9, 10}


def test_active_array_indices_are_read_from_qstat_xml(monkeypatch) -> None:
    xml = """<?xml version='1.0'?>
<job_info><queue_info>
<job_list><JB_job_number>123</JB_job_number><tasks>1</tasks></job_list>
<job_list><JB_job_number>123</JB_job_number><tasks>3-5:2</tasks></job_list>
<job_list><JB_job_number>999</JB_job_number><tasks>7</tasks></job_list>
</queue_info><job_info /></job_info>
"""

    def fake_run(command, **_kwargs):
        assert command == ["qstat", "-xml"]
        return SimpleNamespace(stdout=xml, stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _active_array_indices(["123"]) == {1, 3, 5}


def test_resume_excludes_indices_still_active_in_base_array(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _tasks(raw)
    digest = __import__("hashlib").sha256(
        (raw / "pilot_trajectory_tasks.tsv").read_bytes()
    ).hexdigest()
    submissions = raw / "submissions"
    submissions.mkdir()
    (submissions / "pilot_trajectory.json").write_text(
        json.dumps(
            {
                "status": "SUBMITTED",
                "task_table_sha256": digest,
                "scheduler_job_ids": ["123"],
            }
        ),
        encoding="utf-8",
    )
    commands = []
    xml = """<?xml version='1.0'?>
<job_info><queue_info>
<job_list><JB_job_number>123</JB_job_number><tasks>1</tasks></job_list>
</queue_info><job_info /></job_info>
"""

    def fake_run(command, **_kwargs):
        if command[:2] == ["qstat", "-xml"]:
            return SimpleNamespace(stdout=xml, stderr="", returncode=0)
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="c" * 40 + "\n", returncode=0)
        commands.append(command)
        return SimpleNamespace(stdout="2002.2\n", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    submitted = resume_array(
        raw_root=raw,
        scope="pilot",
        stage="trajectory",
        execute=True,
    )
    assert submitted["pending_indices"] == [2]
    assert submitted["active_indices_excluded"] == [1]
    assert commands[0][commands[0].index("-t") + 1] == "1-1"
    assert any("TASK_INDEX_MAP=2" in part for part in commands[0])
