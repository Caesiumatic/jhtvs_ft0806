from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import xml.etree.ElementTree as ET
from math import ceil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .trajectory import TRAJECTORY_SCOPES, _read_tsv


MD_CHUNKS_PER_SCHEDULER_JOB = 10
SAFE_GPU_QUEUES = (
    "gpu1@compute-1-21.local,"
    "gpu2@compute-1-22.local,"
    "gpu2@compute-1-23.local"
)


def _task_table(raw_root: Path, scope: str) -> Path:
    return raw_root / f"{scope}_trajectory_tasks.tsv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("could not resolve a full repository commit")
    return commit


def workflow_status(*, raw_root: Path, scope: str) -> dict[str, Any]:
    tasks = _read_tsv(_task_table(raw_root, scope))
    trajectory_complete = []
    gap_complete = []
    for task in tasks:
        logical_id = task["logical_trajectory_id"]
        trajectory_path = raw_root / "trajectories" / logical_id / "trajectory.json"
        gap_path = raw_root / "gaps" / logical_id / "gaps.json"
        if trajectory_path.is_file() and json.loads(trajectory_path.read_text())["status"] == "complete":
            trajectory_complete.append(int(task["task_index"]))
        if gap_path.is_file() and json.loads(gap_path.read_text())["status"] == "complete":
            gap_complete.append(int(task["task_index"]))
    all_indices = {int(task["task_index"]) for task in tasks}
    return {
        "status": "complete" if len(gap_complete) == len(tasks) else "incomplete",
        "scope": scope,
        "task_count": len(tasks),
        "trajectory_complete": len(trajectory_complete),
        "gap_complete": len(gap_complete),
        "pending_trajectory_indices": sorted(all_indices - set(trajectory_complete)),
        "pending_gap_indices": sorted(all_indices - set(gap_complete)),
        "task_table_sha256": _sha256(_task_table(raw_root, scope)),
    }


def _array_range(indices: Sequence[int]) -> str:
    if not indices:
        raise ValueError("cannot render an empty scheduler task range")
    groups: list[str] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def _expand_sge_task_spec(specification: str) -> set[int]:
    indices: set[int] = set()
    for group in specification.split(","):
        bounds, _, stride_text = group.partition(":")
        stride = int(stride_text) if stride_text else 1
        start_text, separator, end_text = bounds.partition("-")
        start = int(start_text)
        end = int(end_text) if separator else start
        indices.update(range(start, end + 1, stride))
    return indices


def _active_array_indices(job_ids: Sequence[str]) -> set[int]:
    if not job_ids:
        return set()
    result = subprocess.run(
        ["qstat", "-xml"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"qstat -xml failed: {result.stderr.strip()}")
    requested = set(job_ids)
    active: set[int] = set()
    for job in ET.fromstring(result.stdout).iter("job_list"):
        job_number = job.findtext("JB_job_number")
        task_specification = job.findtext("tasks")
        if job_number in requested and task_specification:
            active.update(_expand_sge_task_spec(task_specification))
    return active


def _resource_arguments(device: str, slots: int) -> tuple[list[str], str]:
    if device == "cuda":
        return (
            [
                "-q",
                SAFE_GPU_QUEUES,
                "-l",
                "gpu=1,slots_gpu=1",
                "-pe",
                "cuda",
                "1",
            ],
            "jhtvs-ft0806-gpu",
        )
    return ["-pe", "smp", str(slots)], "jhtvs-ft0806"


def submit_array(
    *,
    raw_root: Path,
    scope: str,
    stage: str,
    execute: bool,
    max_concurrent: int = 8,
    slots: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    if stage not in {"trajectory", "gap"}:
        raise ValueError("submission stage must be trajectory or gap")
    if device not in {"cpu", "cuda"}:
        raise ValueError("submission device must be cpu or cuda")
    table = _task_table(raw_root, scope)
    tasks = _read_tsv(table)
    if not tasks:
        raise RuntimeError("cannot submit an empty task table")
    digest = _sha256(table)
    ledger_dir = raw_root / "submissions"
    ledger_path = ledger_dir / f"{scope}_{stage}.json"
    prior_ledger: dict[str, Any] | None = None
    if ledger_path.is_file():
        prior_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if prior_ledger["task_table_sha256"] != digest:
            raise RuntimeError("submission ledger task-table hash drift")
        if prior_ledger["status"] == "SUBMITTED":
            return {**prior_ledger, "status": "ALREADY_SUBMITTED"}
    repository = Path(__file__).resolve().parents[3]
    repository_commit = _repository_commit(repository)
    script = repository / "workflows" / "mace_polar_5solv_redox" / "hpc" / (
        "run_trajectory.sh" if stage == "trajectory" else "run_gap.sh"
    )
    log_dir = raw_root / "scheduler_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"mp_{scope[:3]}_{'md' if stage == 'trajectory' else 'gap'}"
    command = [
        "qsub",
        "-terse",
        "-cwd",
        "-t",
        f"1-{len(tasks)}",
        "-tc",
        str(max_concurrent),
    ]
    resource_arguments, conda_environment = _resource_arguments(device, slots)
    command.extend(resource_arguments)
    command.extend([
        "-N",
        job_name,
        "-o",
        str(log_dir),
        "-v",
        f"RAW_ROOT={raw_root.resolve()},TRAJECTORY_MODE={scope},TASK_TABLE_SHA256={digest},REPOSITORY_COMMIT={repository_commit},CONDA_ENV_NAME={conda_environment},MACE_DEVICE={device},MD_CHUNKS_PER_JOB={MD_CHUNKS_PER_SCHEDULER_JOB}",
        str(script),
    ])
    wave_count = (
        1
        if stage == "gap" or scope == "pilot"
        else 200 // MD_CHUNKS_PER_SCHEDULER_JOB
    )
    payload: dict[str, Any] = {
        "status": "PREPARED",
        "scope": scope,
        "stage": stage,
        "task_count": len(tasks),
        "task_table_sha256": digest,
        "repository_commit": repository_commit,
        "max_concurrent": max_concurrent,
        "slots": slots,
        "device": device,
        "md_chunks_per_scheduler_job": MD_CHUNKS_PER_SCHEDULER_JOB
        if stage == "trajectory"
        else None,
        "scheduler_wave_count": wave_count,
        "command": shlex.join(command),
    }
    if not execute:
        return payload
    ledger_dir.mkdir(parents=True, exist_ok=True)
    scheduler_job_ids = list(prior_ledger.get("scheduler_job_ids", [])) if prior_ledger else []
    scheduler_receipts = list(prior_ledger.get("scheduler_receipts", [])) if prior_ledger else []
    payload.update(
        {
            "status": "SUBMITTING",
            "scheduler_job_ids": scheduler_job_ids,
            "scheduler_receipts": scheduler_receipts,
            "submitted_at_utc": prior_ledger.get("submitted_at_utc")
            if prior_ledger
            else datetime.now(UTC).isoformat(),
        }
    )
    ledger_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for wave_index in range(len(scheduler_job_ids), wave_count):
        wave_command = list(command)
        if scheduler_job_ids:
            wave_command[-1:-1] = ["-hold_jid_ad", scheduler_job_ids[-1]]
        result = subprocess.run(wave_command, check=True, capture_output=True, text=True)
        scheduler_text = result.stdout.strip()
        scheduler_job_id = scheduler_text.split(".", maxsplit=1)[0]
        if not scheduler_job_id.isdecimal():
            raise RuntimeError(f"could not parse qsub job id: {scheduler_text!r}")
        scheduler_job_ids.append(scheduler_job_id)
        scheduler_receipts.append(scheduler_text)
        payload.update(
            {
                "scheduler_job_ids": scheduler_job_ids,
                "scheduler_receipts": scheduler_receipts,
                "submitted_scheduler_waves": wave_index + 1,
            }
        )
        ledger_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    payload.update(
        {
            "status": "SUBMITTED",
            "scheduler_job_id": scheduler_job_ids[0],
            "scheduler_receipt": scheduler_receipts[0],
        }
    )
    ledger_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _remaining_trajectory_waves(
    *, raw_root: Path, scope: str, pending_indices: Sequence[int]
) -> int:
    tasks = {
        int(row["task_index"]): row for row in _read_tsv(_task_table(raw_root, scope))
    }
    total_chunks = 3 if scope == "pilot" else 200
    maximum_remaining = 0
    for index in pending_indices:
        logical_id = tasks[index]["logical_trajectory_id"]
        completed = 0
        for path in (raw_root / "trajectories" / logical_id / "md").glob("chunk-*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            completed += int(payload.get("status") == "complete")
        maximum_remaining = max(maximum_remaining, total_chunks - completed)
    return max(1, ceil(maximum_remaining / MD_CHUNKS_PER_SCHEDULER_JOB))


def resume_array(
    *,
    raw_root: Path,
    scope: str,
    stage: str,
    execute: bool,
    max_concurrent: int = 3,
    slots: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    if stage not in {"trajectory", "gap"}:
        raise ValueError("resume stage must be trajectory or gap")
    if device not in {"cpu", "cuda"}:
        raise ValueError("resume device must be cpu or cuda")
    table = _task_table(raw_root, scope)
    digest = _sha256(table)
    status = workflow_status(raw_root=raw_root, scope=scope)
    pending_indices = status[
        "pending_trajectory_indices" if stage == "trajectory" else "pending_gap_indices"
    ]
    if not pending_indices:
        return {"status": "complete", "scope": scope, "stage": stage, "pending_indices": []}
    ledger_dir = raw_root / "submissions"
    base_ledger_path = ledger_dir / f"{scope}_{stage}.json"
    if not base_ledger_path.is_file():
        raise RuntimeError("cannot resume before the initial stage submission")
    base_ledger = json.loads(base_ledger_path.read_text(encoding="utf-8"))
    if base_ledger["task_table_sha256"] != digest:
        raise RuntimeError("resume task-table hash drift")
    active_base_indices = (
        _active_array_indices(base_ledger.get("scheduler_job_ids", [])) if execute else set()
    )
    pending_indices = [
        index for index in pending_indices if index not in active_base_indices
    ]
    if not pending_indices:
        return {
            "status": "ACTIVE_TASKS",
            "scope": scope,
            "stage": stage,
            "pending_indices": [],
            "active_indices_excluded": sorted(active_base_indices),
        }
    prior_retries = sorted(ledger_dir.glob(f"{scope}_{stage}_retry-*.json"))
    prior_ledger: dict[str, Any] | None = None
    if prior_retries:
        latest = json.loads(prior_retries[-1].read_text(encoding="utf-8"))
        if latest["status"] == "SUBMITTING":
            prior_ledger = latest
            retry_index = int(latest["retry_index"])
            ledger_path = prior_retries[-1]
        else:
            active = any(
                subprocess.run(
                    ["qstat", "-j", job_id], capture_output=True, text=True
                ).returncode
                == 0
                for job_id in latest["scheduler_job_ids"]
            )
            if active:
                return {**latest, "status": "ALREADY_SUBMITTED"}
            retry_index = int(latest["retry_index"]) + 1
            ledger_path = ledger_dir / f"{scope}_{stage}_retry-{retry_index:03d}.json"
    else:
        retry_index = 1
        ledger_path = ledger_dir / f"{scope}_{stage}_retry-{retry_index:03d}.json"
    repository = Path(__file__).resolve().parents[3]
    repository_commit = _repository_commit(repository)
    script = repository / "workflows" / "mace_polar_5solv_redox" / "hpc" / (
        "run_trajectory.sh" if stage == "trajectory" else "run_gap.sh"
    )
    log_dir = raw_root / "scheduler_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "qsub",
        "-terse",
        "-cwd",
        "-t",
        f"1-{len(pending_indices)}",
        "-tc",
        str(max_concurrent),
    ]
    resource_arguments, conda_environment = _resource_arguments(device, slots)
    command.extend(resource_arguments)
    command.extend(
        [
            "-N",
            f"mp_{scope[:3]}_{'md' if stage == 'trajectory' else 'gap'}r{retry_index}",
            "-o",
            str(log_dir),
            "-v",
            f"RAW_ROOT={raw_root.resolve()},TRAJECTORY_MODE={scope},TASK_TABLE_SHA256={digest},REPOSITORY_COMMIT={repository_commit},CONDA_ENV_NAME={conda_environment},MACE_DEVICE={device},MD_CHUNKS_PER_JOB={MD_CHUNKS_PER_SCHEDULER_JOB},TASK_INDEX_MAP={':'.join(map(str, pending_indices))}",
            str(script),
        ]
    )
    wave_count = (
        _remaining_trajectory_waves(
            raw_root=raw_root, scope=scope, pending_indices=pending_indices
        )
        if stage == "trajectory"
        else 1
    )
    payload: dict[str, Any] = {
        "status": "PREPARED",
        "scope": scope,
        "stage": stage,
        "retry_index": retry_index,
        "pending_indices": pending_indices,
        "active_indices_excluded": sorted(active_base_indices),
        "task_table_sha256": digest,
        "repository_commit": repository_commit,
        "device": device,
        "max_concurrent": max_concurrent,
        "slots": slots,
        "scheduler_wave_count": wave_count,
        "command": shlex.join(command),
    }
    if not execute:
        return payload
    ledger_dir.mkdir(parents=True, exist_ok=True)
    scheduler_job_ids = list(prior_ledger.get("scheduler_job_ids", [])) if prior_ledger else []
    scheduler_receipts = list(prior_ledger.get("scheduler_receipts", [])) if prior_ledger else []
    payload.update(
        {
            "status": "SUBMITTING",
            "scheduler_job_ids": scheduler_job_ids,
            "scheduler_receipts": scheduler_receipts,
            "submitted_at_utc": prior_ledger.get("submitted_at_utc")
            if prior_ledger
            else datetime.now(UTC).isoformat(),
        }
    )
    ledger_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for wave_index in range(len(scheduler_job_ids), wave_count):
        wave_command = list(command)
        if scheduler_job_ids:
            wave_command[-1:-1] = ["-hold_jid_ad", scheduler_job_ids[-1]]
        result = subprocess.run(wave_command, capture_output=True, text=True)
        if result.returncode != 0:
            payload.update(
                {
                    "status": "SUBMISSION_FAILED",
                    "failed_command": shlex.join(wave_command),
                    "scheduler_stdout": result.stdout.strip(),
                    "scheduler_stderr": result.stderr.strip(),
                }
            )
            ledger_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"retry qsub failed with exit {result.returncode}: {result.stderr.strip()}"
            )
        scheduler_text = result.stdout.strip()
        scheduler_job_id = scheduler_text.split(".", maxsplit=1)[0]
        if not scheduler_job_id.isdecimal():
            raise RuntimeError(f"could not parse retry qsub job id: {scheduler_text!r}")
        scheduler_job_ids.append(scheduler_job_id)
        scheduler_receipts.append(scheduler_text)
        payload.update(
            {
                "scheduler_job_ids": scheduler_job_ids,
                "scheduler_receipts": scheduler_receipts,
                "submitted_scheduler_waves": wave_index + 1,
            }
        )
        ledger_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    payload.update({"status": "SUBMITTED", "scheduler_job_id": scheduler_job_ids[0]})
    ledger_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--raw-root", type=Path, required=True)
    status.add_argument("--scope", choices=TRAJECTORY_SCOPES, required=True)
    resume = sub.add_parser("resume")
    resume.add_argument("--raw-root", type=Path, required=True)
    resume.add_argument("--scope", choices=TRAJECTORY_SCOPES, required=True)
    resume.add_argument("--stage", choices=("trajectory", "gap"), default="trajectory")
    resume.add_argument("--max-concurrent", type=int, default=3)
    resume.add_argument("--slots", type=int, default=8)
    resume.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    resume.add_argument("--execute", action="store_true")
    submit = sub.add_parser("submit-md")
    submit.add_argument("--raw-root", type=Path, required=True)
    submit.add_argument("--scope", choices=TRAJECTORY_SCOPES, required=True)
    submit.add_argument("--stage", choices=("trajectory", "gap"), default="trajectory")
    submit.add_argument("--max-concurrent", type=int, default=8)
    submit.add_argument("--slots", type=int, default=8)
    submit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    submit.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        payload = workflow_status(raw_root=args.raw_root, scope=args.scope)
    elif args.command == "resume":
        payload = resume_array(
            raw_root=args.raw_root,
            scope=args.scope,
            stage=args.stage,
            execute=args.execute,
            max_concurrent=args.max_concurrent,
            slots=args.slots,
            device=args.device,
        )
    else:
        payload = submit_array(
            raw_root=args.raw_root,
            scope=args.scope,
            stage=args.stage,
            execute=args.execute,
            max_concurrent=args.max_concurrent,
            slots=args.slots,
            device=args.device,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
