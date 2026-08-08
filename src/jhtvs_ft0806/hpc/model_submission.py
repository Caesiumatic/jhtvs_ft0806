"""Immutable, budget-guarded SGE preparation for five-seed model training."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from jhtvs_ft0806.hpc.submission import (
    ACTIVE_SUBMISSION_STATUSES,
    SubmissionError,
    SubmissionPlan,
    TASK_FIELDS,
    check_budget,
)
from jhtvs_ft0806.ml.features import EXPECTED_CHECKPOINT_SHA256
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows
from jhtvs_ft0806.spec_validation import validate_spec


MODEL_JOB_ID = "MACE-ENSEMBLE"
MODEL_JOB_CLASS = "mace_training"
MODEL_WORKFLOW_REVISION = "jhtvs-ft0806-five-seed-lora-training-v1"
MODEL_METHOD_ID = "MACE_POLAR_1_L_reaction_residual_LoRA_r4_a1_v1"
MODEL_NPROCS = 8

TRAINING_INPUT_PATHS = (
    "data/resolved/geometry_index.csv",
    "data/resolved/reaction_sp_labels.csv",
    "data/resolved/reaction_final_labels.csv",
    "data/resolved/base_feature_index.csv",
    "data/resolved/base_feature_completion.json",
    "spec/01_SCIENTIFIC_SPEC.md",
    "spec/calibration_tuple_design.csv",
    "spec/solvent_smd_registry.csv",
    "spec/training_config.csv",
    "config/ml_environment_lock.json",
)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SubmissionError(f"model submission path is outside repository: {path}") from exc


def _write_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SubmissionError(f"immutable model submission file changed: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _task_text(row: dict[str, str]) -> str:
    return "\n".join(("\t".join(TASK_FIELDS), "\t".join(row[field] for field in TASK_FIELDS))) + "\n"


def _active_job_ids(ledger_path: Path, repository_root: Path) -> set[str]:
    active: set[str] = set()
    if not ledger_path.is_file():
        return active
    for ledger in read_csv_rows(ledger_path):
        if ledger["status"] not in ACTIVE_SUBMISSION_STATUSES:
            continue
        task_path = Path(ledger["task_table_path"])
        if not task_path.is_absolute():
            task_path = repository_root / task_path
        if not task_path.is_file() or sha256_file(task_path) != ledger["task_table_sha256"]:
            raise SubmissionError("active submission task table is missing or hash-mismatched")
        text = task_path.read_text(encoding="utf-8").splitlines()
        active.update(line.split("\t")[2] for line in text[1:] if line)
    return active


def prepare_model_submission(
    *,
    submission_id: str,
    spec_dir: Path,
    submissions_root: Path,
    accounting_path: Path,
    ledger_path: Path,
    runner_path: Path,
    planning_core_h: Decimal,
    queue: str = "amd16smt",
    parallel_environment: str = "orte",
    budget_scope: str = "first_round",
) -> SubmissionPlan:
    if not submission_id or any(character.isspace() for character in submission_id):
        raise SubmissionError("invalid model submission ID")
    if planning_core_h <= 0:
        raise SubmissionError("model planning core-hours must be positive")
    spec_dir = spec_dir.resolve()
    repository_root = spec_dir.parent
    runner_path = runner_path.resolve()
    if not validate_spec(spec_dir).ok:
        raise SubmissionError("scientific specification validation failed")
    if not runner_path.is_file() or runner_path.is_symlink():
        raise SubmissionError("model runner is missing or unsafe")

    inputs: dict[str, str] = {}
    for relative in TRAINING_INPUT_PATHS:
        path = repository_root / relative
        if not path.is_file() or path.is_symlink():
            raise SubmissionError(f"model training input is missing or unsafe: {relative}")
        inputs[relative] = sha256_file(path)
    if len(read_csv_rows(repository_root / "data/resolved/reaction_sp_labels.csv")) != 403:
        raise SubmissionError("model training requires 403 reaction SP rows")
    if len(read_csv_rows(repository_root / "data/resolved/reaction_final_labels.csv")) != 50:
        raise SubmissionError("model training requires 50 reaction final rows")
    feature_rows = read_csv_rows(repository_root / "data/resolved/base_feature_index.csv")
    if len(feature_rows) != 705 or {row["checkpoint_sha256"] for row in feature_rows} != {
        EXPECTED_CHECKPOINT_SHA256
    }:
        raise SubmissionError("model training feature index is incomplete or checkpoint-drifted")
    feature_receipt = json.loads(
        (repository_root / "data/resolved/base_feature_completion.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        feature_receipt.get("status") != "PASS"
        or feature_receipt.get("total") != 705
        or feature_receipt.get("missing") != 0
        or feature_receipt.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
    ):
        raise SubmissionError("base feature completion receipt is not clean")

    submission_dir = (submissions_root / submission_id).resolve()
    task_table_path = submission_dir / "tasks.tsv"
    input_manifest_path = submission_dir / "training_inputs.json"
    preflight_path = submission_dir / "preflight.json"
    plan_path = submission_dir / "submission_plan.json"
    completion_path = submission_dir / "training_completion.json"
    model_manifest_path = repository_root / "data/resolved/model_training_manifest.json"
    if completion_path.exists() or model_manifest_path.exists():
        raise SubmissionError("model production outputs already exist; reconcile before submission")

    input_payload = {
        "workflow_revision": MODEL_WORKFLOW_REVISION,
        "method_id": MODEL_METHOD_ID,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "expected_reaction_sp_rows": 403,
        "expected_reaction_final_rows": 50,
        "expected_feature_rows": 705,
        "head_warmup_epochs": 50,
        "max_lora_epochs": 300,
        "online_batch_size": 4,
        "lora_rank": 4,
        "lora_alpha": 1.0,
        "seeds": [17, 29, 43, 71, 101],
        "inputs": inputs,
    }
    _write_exact(
        input_manifest_path,
        json.dumps(input_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    task = {
        "array_task": "1",
        "sequence": "1",
        "job_id": MODEL_JOB_ID,
        "job_class": MODEL_JOB_CLASS,
        "input_path": _relative(input_manifest_path, repository_root),
        "input_sha256": sha256_file(input_manifest_path),
        "output_path": _relative(completion_path, repository_root),
        "nprocs": str(MODEL_NPROCS),
        "planning_core_h": str(planning_core_h),
        "workflow_revision": MODEL_WORKFLOW_REVISION,
        "method_id": MODEL_METHOD_ID,
    }
    _write_exact(task_table_path, _task_text(task))
    preflight = {
        "status": "PASS",
        "job_id": MODEL_JOB_ID,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "runner_sha256": sha256_file(runner_path),
        "task_table_sha256": sha256_file(task_table_path),
        "planning_core_h": str(planning_core_h),
        "nprocs": MODEL_NPROCS,
        "outputs_absent": True,
        **input_payload,
    }
    _write_exact(
        preflight_path,
        json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    budget = check_budget(
        spec_dir=spec_dir,
        accounting_path=accounting_path,
        ledger_path=ledger_path,
        proposed_core_h=planning_core_h,
        budget_scope=budget_scope,
    )
    plan_payload = {
        "submission_id": submission_id,
        "job_ids": [MODEL_JOB_ID],
        "job_count": 1,
        "array_task_count": 1,
        "planned_core_h": str(planning_core_h),
        "task_table_sha256": sha256_file(task_table_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "runner_sha256": sha256_file(runner_path),
        "budget_scope": budget_scope,
        "budget": budget.to_dict(),
        "queue": queue,
        "parallel_environment": parallel_environment,
        "nprocs": MODEL_NPROCS,
        "max_concurrent": 1,
    }
    submission_sha = content_hash(plan_payload)
    _write_exact(
        plan_path,
        json.dumps(
            {**plan_payload, "submission_sha256": submission_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    if MODEL_JOB_ID in _active_job_ids(ledger_path, repository_root):
        matches = [
            row
            for row in read_csv_rows(ledger_path)
            if row["submission_id"] == submission_id
            and row["submission_sha256"] == submission_sha
            and row["status"] in ACTIVE_SUBMISSION_STATUSES
        ]
        if len(matches) != 1:
            raise SubmissionError(
                "five-seed model training already belongs to another active submission"
            )
    return SubmissionPlan(
        submission_id=submission_id,
        submission_dir=submission_dir,
        task_table_path=task_table_path,
        preflight_report_path=preflight_path,
        plan_path=plan_path,
        task_table_sha256=sha256_file(task_table_path),
        submission_sha256=submission_sha,
        job_count=1,
        array_task_count=1,
        planned_core_h=planning_core_h,
        budget=budget,
        queue=queue,
        parallel_environment=parallel_environment,
        nprocs=MODEL_NPROCS,
        max_concurrent=1,
    )
