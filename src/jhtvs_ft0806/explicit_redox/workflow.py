from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .trajectory import TRAJECTORY_SCOPES, _read_tsv


def _task_table(raw_root: Path, scope: str) -> Path:
    return raw_root / f"{scope}_trajectory_tasks.tsv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def submit_array(
    *,
    raw_root: Path,
    scope: str,
    stage: str,
    execute: bool,
    max_concurrent: int = 8,
    slots: int = 8,
) -> dict[str, Any]:
    if stage not in {"trajectory", "gap"}:
        raise ValueError("submission stage must be trajectory or gap")
    table = _task_table(raw_root, scope)
    tasks = _read_tsv(table)
    if not tasks:
        raise RuntimeError("cannot submit an empty task table")
    digest = _sha256(table)
    ledger_dir = raw_root / "submissions"
    ledger_path = ledger_dir / f"{scope}_{stage}.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger["task_table_sha256"] != digest:
            raise RuntimeError("submission ledger task-table hash drift")
        return {**ledger, "status": "ALREADY_SUBMITTED"}
    repository = Path(__file__).resolve().parents[3]
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
        "-pe",
        "smp",
        str(slots),
        "-N",
        job_name,
        "-o",
        str(log_dir),
        "-v",
        f"RAW_ROOT={raw_root.resolve()},TRAJECTORY_MODE={scope},TASK_TABLE_SHA256={digest}",
        str(script),
    ]
    payload: dict[str, Any] = {
        "status": "PREPARED",
        "scope": scope,
        "stage": stage,
        "task_count": len(tasks),
        "task_table_sha256": digest,
        "max_concurrent": max_concurrent,
        "slots": slots,
        "command": shlex.join(command),
    }
    if not execute:
        return payload
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    scheduler_text = result.stdout.strip()
    scheduler_job_id = scheduler_text.split(".", maxsplit=1)[0]
    if not scheduler_job_id.isdecimal():
        raise RuntimeError(f"could not parse qsub job id: {scheduler_text!r}")
    payload.update(
        {
            "status": "SUBMITTED",
            "scheduler_job_id": scheduler_job_id,
            "scheduler_receipt": scheduler_text,
            "submitted_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--raw-root", type=Path, required=True)
        command.add_argument("--scope", choices=TRAJECTORY_SCOPES, required=True)
    submit = sub.add_parser("submit-md")
    submit.add_argument("--raw-root", type=Path, required=True)
    submit.add_argument("--scope", choices=TRAJECTORY_SCOPES, required=True)
    submit.add_argument("--stage", choices=("trajectory", "gap"), default="trajectory")
    submit.add_argument("--max-concurrent", type=int, default=8)
    submit.add_argument("--slots", type=int, default=8)
    submit.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"status", "resume"}:
        payload = workflow_status(raw_root=args.raw_root, scope=args.scope)
    else:
        payload = submit_array(
            raw_root=args.raw_root,
            scope=args.scope,
            stage=args.stage,
            execute=args.execute,
            max_concurrent=args.max_concurrent,
            slots=args.slots,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
