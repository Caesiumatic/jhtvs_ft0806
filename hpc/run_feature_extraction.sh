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
[ "$JOB_CLASS" = "mace_base_features" ]
[ "$METHOD_ID" = "MACE_POLAR_1_L_raw_invariant_base_v1" ]

case "$LOGICAL_JOB_ID" in
  MACEBASE-CALIBRATION)
    EXPECTED_ROWS=705
    EXPECTED_INPUT_REL="data/resolved/geometry_index.csv"
    FEATURE_INDEX_REL="data/resolved/base_feature_index.csv"
    BASELINE_REL="data/resolved/base_state_energies.csv"
    EXPECTED_NPROCS=1
    EXPECTED_WORKFLOW_REVISION="jhtvs-ft0806-polar1l-base-features-v1"
    ;;
  MACEBASE-FULLSPACE)
    EXPECTED_ROWS=8100
    EXPECTED_INPUT_REL="data/resolved/fullspace_geometry_index.csv"
    FEATURE_INDEX_REL="data/resolved/fullspace_feature_index.csv"
    BASELINE_REL="data/resolved/fullspace_base_state_energies.csv"
    EXPECTED_NPROCS=8
    EXPECTED_WORKFLOW_REVISION="jhtvs-ft0806-polar1l-fullspace-base-features-v1"
    ;;
  *)
    echo "unsupported feature dataset job: $LOGICAL_JOB_ID" >&2
    exit 1
    ;;
esac
[ "$INPUT_REL" = "$EXPECTED_INPUT_REL" ]
[ "$NPROCS" = "$EXPECTED_NPROCS" ]
[ "${NSLOTS:-1}" = "$EXPECTED_NPROCS" ]
[ "$WORKFLOW_REVISION" = "$EXPECTED_WORKFLOW_REVISION" ]

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

SUMMARY="${OUTPUT}.summary.tmp"
PYTHONPATH=src python -m jhtvs_ft0806.cli extract-base-features \
  --geometry-index "$INPUT" \
  --cache-dir artifacts/base_features \
  --feature-index "$FEATURE_INDEX_REL" \
  --baseline-output "$BASELINE_REL" \
  --checkpoint polar-1-l \
  --device cpu \
  --require-complete > "$SUMMARY"

OUTPUT="$OUTPUT" SUMMARY="$SUMMARY" INPUT_SHA="$INPUT_SHA" \
  PLANNING_CORE_H="$PLANNING_CORE_H" EXPECTED_ROWS="$EXPECTED_ROWS" \
  LOGICAL_JOB_ID="$LOGICAL_JOB_ID" FEATURE_INDEX_REL="$FEATURE_INDEX_REL" \
  BASELINE_REL="$BASELINE_REL" WORKFLOW_REVISION="$WORKFLOW_REVISION" \
  METHOD_ID="$METHOD_ID" python - <<'PY'
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
if summary.get("status") != "PASS" or summary.get("total") != int(os.environ["EXPECTED_ROWS"]):
    raise SystemExit("feature extraction summary is incomplete")
root = Path.cwd()
feature_index = root / os.environ["FEATURE_INDEX_REL"]
baseline = root / os.environ["BASELINE_REL"]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    **summary,
    "job_id": os.environ["LOGICAL_JOB_ID"],
    "job_class": "mace_base_features",
    "input_sha256": os.environ["INPUT_SHA"],
    "feature_index_path": os.environ["FEATURE_INDEX_REL"],
    "feature_index_sha256": sha(feature_index),
    "baseline_path": os.environ["BASELINE_REL"],
    "baseline_sha256": sha(baseline),
    "planning_core_h": os.environ["PLANNING_CORE_H"],
    "workflow_revision": os.environ["WORKFLOW_REVISION"],
    "method_id": os.environ["METHOD_ID"],
    "scheduler_job_id": os.environ.get("JOB_ID", "unknown"),
    "scheduler_array_task": os.environ.get("SGE_TASK_ID", "unknown"),
}
target = Path(os.environ["OUTPUT"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
PY
rm -f -- "$SUMMARY"
echo "completed: $LOGICAL_JOB_ID"
