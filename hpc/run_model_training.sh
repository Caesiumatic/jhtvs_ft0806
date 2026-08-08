#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${TASK_FILE:?set TASK_FILE to the immutable task table}"
: "${TASK_FILE_SHA256:?set TASK_FILE_SHA256 to its SHA256}"
: "${SGE_TASK_ID:?submit as a one-task SGE array}"

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
[ "${NSLOTS:-1}" = "8" ]
[ -f "$TASK_FILE" ] && [ ! -L "$TASK_FILE" ]
[ "$(sha256sum "$TASK_FILE" | awk '{print $1}')" = "$TASK_FILE_SHA256" ]

TASK_ROW="$(awk -F '\t' -v task="$SGE_TASK_ID" 'NR > 1 && $1 == task' "$TASK_FILE")"
[ -n "$TASK_ROW" ]
IFS=$'\t' read -r ARRAY_TASK SEQUENCE LOGICAL_JOB_ID JOB_CLASS INPUT_REL INPUT_SHA \
  OUTPUT_REL NPROCS PLANNING_CORE_H WORKFLOW_REVISION METHOD_ID <<< "$TASK_ROW"
[ "$ARRAY_TASK" = "1" ] && [ "$SEQUENCE" = "1" ]
[ "$LOGICAL_JOB_ID" = "MACE-ENSEMBLE" ]
[ "$JOB_CLASS" = "mace_training" ]
[ "$NPROCS" = "8" ]
[ "$WORKFLOW_REVISION" = "jhtvs-ft0806-five-seed-lora-training-v1" ]
[ "$METHOD_ID" = "MACE_POLAR_1_L_reaction_residual_LoRA_r4_a1_v1" ]

INPUT="$REPO_ROOT/$INPUT_REL"
OUTPUT="$REPO_ROOT/$OUTPUT_REL"
[ -f "$INPUT" ] && [ ! -L "$INPUT" ]
[ "$(sha256sum "$INPUT" | awk '{print $1}')" = "$INPUT_SHA" ]
[ ! -e "$OUTPUT" ]

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate jhtvs-ft0806
set -u
cd "$REPO_ROOT"

export OMP_NUM_THREADS="$NPROCS"
export MKL_NUM_THREADS="$NPROCS"
export OPENBLAS_NUM_THREADS="$NPROCS"
export NUMEXPR_NUM_THREADS="$NPROCS"

CHECKPOINT="$HOME/.cache/mace/MACEPOLAR1Lmodel"
[ -f "$CHECKPOINT" ]
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" ]

INPUT="$INPUT" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["INPUT"]).read_text())
if manifest["workflow_revision"] != "jhtvs-ft0806-five-seed-lora-training-v1":
    raise SystemExit("training workflow revision drift")
if manifest["seeds"] != [17, 29, 43, 71, 101]:
    raise SystemExit("training seeds drift")
if (manifest["head_warmup_epochs"], manifest["max_lora_epochs"], manifest["online_batch_size"]) != (50, 300, 4):
    raise SystemExit("training schedule drift")
for relative, expected in manifest["inputs"].items():
    path = Path.cwd() / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing training input: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"training input hash mismatch: {relative}")
PY

SUMMARY="${OUTPUT}.summary.tmp"
PYTHONPATH=src python -m jhtvs_ft0806.cli train \
  --device cpu \
  --max-lora-epochs 300 \
  --online-batch-size 4 > "$SUMMARY"

OUTPUT="$OUTPUT" SUMMARY="$SUMMARY" INPUT_SHA="$INPUT_SHA" \
  PLANNING_CORE_H="$PLANNING_CORE_H" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

summary_text = Path(os.environ["SUMMARY"]).read_text()
start = summary_text.find("{")
if start < 0:
    raise SystemExit("training summary has no JSON payload")
summary, end = json.JSONDecoder().raw_decode(summary_text[start:])
if summary_text[start + end :].strip():
    raise SystemExit("training summary has unexpected trailing output")
if summary.get("status") != "PASS" or summary.get("member_count") != 5:
    raise SystemExit("training summary is incomplete")
root = Path.cwd()
manifest = root / "data/resolved/model_training_manifest.json"
receipt = {
    **summary,
    "job_id": "MACE-ENSEMBLE",
    "job_class": "mace_training",
    "input_sha256": os.environ["INPUT_SHA"],
    "model_manifest_path": "data/resolved/model_training_manifest.json",
    "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "planning_core_h": os.environ["PLANNING_CORE_H"],
    "workflow_revision": "jhtvs-ft0806-five-seed-lora-training-v1",
    "method_id": "MACE_POLAR_1_L_reaction_residual_LoRA_r4_a1_v1",
    "scheduler_job_id": os.environ.get("JOB_ID", "unknown"),
    "scheduler_array_task": os.environ.get("SGE_TASK_ID", "unknown"),
}
target = Path(os.environ["OUTPUT"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
PY
rm -f -- "$SUMMARY"
echo "completed: $LOGICAL_JOB_ID"
