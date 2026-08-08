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
[ "${NSLOTS:-1}" = "1" ]
[ -f "$TASK_FILE" ] && [ ! -L "$TASK_FILE" ]
[ "$(sha256sum "$TASK_FILE" | awk '{print $1}')" = "$TASK_FILE_SHA256" ]

TASK_ROW="$(awk -F '\t' -v task="$SGE_TASK_ID" 'NR > 1 && $1 == task' "$TASK_FILE")"
[ -n "$TASK_ROW" ]
IFS=$'\t' read -r ARRAY_TASK SEQUENCE LOGICAL_JOB_ID JOB_CLASS INPUT_REL INPUT_SHA \
  OUTPUT_REL NPROCS PLANNING_CORE_H WORKFLOW_REVISION METHOD_ID <<< "$TASK_ROW"
[ "$ARRAY_TASK" = "1" ] && [ "$SEQUENCE" = "1" ]
[ "$LOGICAL_JOB_ID" = "MACEBASE-CALIBRATION" ]
[ "$JOB_CLASS" = "mace_base_features" ]
[ "$NPROCS" = "1" ]
[ "$WORKFLOW_REVISION" = "jhtvs-ft0806-polar1l-base-features-v1" ]
[ "$METHOD_ID" = "MACE_POLAR_1_L_raw_invariant_base_v1" ]
[ "$INPUT_REL" = "data/resolved/geometry_index.csv" ]

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

CHECKPOINT="$HOME/.cache/mace/MACEPOLAR1Lmodel"
[ -f "$CHECKPOINT" ]
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" ]

SUMMARY="${OUTPUT}.summary.tmp"
PYTHONPATH=src python -m jhtvs_ft0806.cli extract-base-features \
  --geometry-index "$INPUT" \
  --cache-dir artifacts/base_features \
  --feature-index data/resolved/base_feature_index.csv \
  --baseline-output data/resolved/base_state_energies.csv \
  --checkpoint polar-1-l \
  --device cpu \
  --require-complete > "$SUMMARY"

OUTPUT="$OUTPUT" SUMMARY="$SUMMARY" INPUT_SHA="$INPUT_SHA" \
  PLANNING_CORE_H="$PLANNING_CORE_H" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

summary_path = Path(os.environ["SUMMARY"])
summary_text = summary_path.read_text()
json_start = summary_text.find("{")
if json_start < 0:
    raise SystemExit("feature extraction summary has no JSON payload")
summary, remainder_index = json.JSONDecoder().raw_decode(summary_text[json_start:])
if summary_text[json_start + remainder_index :].strip():
    raise SystemExit("feature extraction summary has unexpected trailing output")
if summary.get("status") != "PASS" or summary.get("total") != 705:
    raise SystemExit("feature extraction summary is incomplete")
root = Path.cwd()
feature_index = root / "data/resolved/base_feature_index.csv"
baseline = root / "data/resolved/base_state_energies.csv"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    **summary,
    "job_id": "MACEBASE-CALIBRATION",
    "job_class": "mace_base_features",
    "input_sha256": os.environ["INPUT_SHA"],
    "feature_index_path": "data/resolved/base_feature_index.csv",
    "feature_index_sha256": sha(feature_index),
    "baseline_path": "data/resolved/base_state_energies.csv",
    "baseline_sha256": sha(baseline),
    "planning_core_h": os.environ["PLANNING_CORE_H"],
    "workflow_revision": "jhtvs-ft0806-polar1l-base-features-v1",
    "method_id": "MACE_POLAR_1_L_raw_invariant_base_v1",
    "scheduler_job_id": os.environ.get("JOB_ID", "unknown"),
    "scheduler_array_task": os.environ.get("SGE_TASK_ID", "unknown"),
}
target = Path(os.environ["OUTPUT"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
PY
rm -f -- "$SUMMARY"
echo "completed: $LOGICAL_JOB_ID"
