#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the absolute geometry run directory}"
: "${ARRAY_SHA256:?set ARRAY_SHA256 to the inspected native array hash}"
: "${TASK_FILE:?set TASK_FILE to the immutable common task table}"
: "${TASK_FILE_SHA256:?set TASK_FILE_SHA256 to its hash}"
: "${SGE_TASK_ID:?submit as an SGE array job}"

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate screen2026
set -u
command -v xtb >/dev/null
export OMP_NUM_THREADS="${NSLOTS:-1}"
[ "${NSLOTS:-1}" = "1" ]

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
NATIVE_ARRAY="$RUN_ROOT/sigma_preopt_array.tsv"
[ -f "$NATIVE_ARRAY" ] && [ ! -L "$NATIVE_ARRAY" ]
[ "$(sha256sum "$NATIVE_ARRAY" | awk '{print $1}')" = "$ARRAY_SHA256" ]
[ -f "$TASK_FILE" ] && [ ! -L "$TASK_FILE" ]
[ "$(sha256sum "$TASK_FILE" | awk '{print $1}')" = "$TASK_FILE_SHA256" ]

COMMON_ROW="$(awk -F '\t' -v task="$SGE_TASK_ID" 'NR > 1 && $1 == task' "$TASK_FILE")"
NATIVE_ROW="$(sed -n "${SGE_TASK_ID}p" "$NATIVE_ARRAY")"
[ -n "$COMMON_ROW" ] && [ -n "$NATIVE_ROW" ]
IFS=$'\t' read -r ARRAY_TASK SEQUENCE LOGICAL_JOB_ID JOB_CLASS INPUT_REL INPUT_SHA \
  OUTPUT_REL NPROCS PLANNING_CORE_H WORKFLOW_REVISION METHOD_ID <<< "$COMMON_ROW"
IFS=$'\t' read -r TASK_ID SOURCE_REL SOURCE_SHA OUTPUT_DIR_REL CHARGE UHF EPSILON \
  TOPOLOGY_SHA <<< "$NATIVE_ROW"
[ "$ARRAY_TASK" = "$SGE_TASK_ID" ] && [ "$SEQUENCE" = "1" ]
[ "$LOGICAL_JOB_ID" = "$TASK_ID" ] && [ "$JOB_CLASS" = "sigma_preopt" ]
[ "$NPROCS" = "1" ] && [ "$CHARGE" = "2" ] && [ "$UHF" = "0" ]
[ "$WORKFLOW_REVISION" = "jhtvs-ft0806-fullspace-sigma-preopt-v1" ]
[ "$METHOD_ID" = "GFN2-xTB_default_opt_ddCOSMO_v1" ]
[ "$INPUT_REL" = "runs/geometry_fullspace/$SOURCE_REL" ]
[ "$OUTPUT_REL" = "runs/geometry_fullspace/$OUTPUT_DIR_REL/task_status.tsv" ]
[ "$INPUT_SHA" = "$SOURCE_SHA" ] && [ -n "$EPSILON" ] && [ -n "$TOPOLOGY_SHA" ]

SOURCE_XYZ="$REPO_ROOT/$INPUT_REL"
OUTPUT_DIR="$RUN_ROOT/$OUTPUT_DIR_REL"
[ -f "$SOURCE_XYZ" ] && [ ! -L "$SOURCE_XYZ" ]
[ "$(sha256sum "$SOURCE_XYZ" | awk '{print $1}')" = "$SOURCE_SHA" ]

if [ -d "$OUTPUT_DIR" ]; then
  if [ -s "$OUTPUT_DIR/xtbopt.xyz" ] && [ -s "$OUTPUT_DIR/task_status.tsv" ] \
      && grep -Fq "$TASK_ID" "$OUTPUT_DIR/task_status.tsv" \
      && grep -Fq "$SOURCE_SHA" "$OUTPUT_DIR/task_status.tsv"; then
    echo "already complete: $TASK_ID"
    exit 0
  fi
  echo "refusing incomplete existing output: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
WORK_DIR="$(mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX")"
cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT
cp "$SOURCE_XYZ" "$WORK_DIR/in.xyz"
(
  cd "$WORK_DIR"
  xtb in.xyz --chrg "$CHARGE" --uhf "$UHF" --opt --cosmo "$EPSILON" > xtb.out 2>&1
  [ -s xtbopt.xyz ]
  grep -qi "normal termination of xtb" xtb.out
  OUTPUT_SHA="$(sha256sum xtbopt.xyz | awk '{print $1}')"
  printf 'task_id\tsource_xyz_sha256\toptimized_xyz_sha256\ttopology_sha256\tcharge\tuhf\tepsilon\n' > task_status.tsv
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TASK_ID" "$SOURCE_SHA" "$OUTPUT_SHA" "$TOPOLOGY_SHA" "$CHARGE" "$UHF" "$EPSILON" \
    >> task_status.tsv
)
mv "$WORK_DIR" "$OUTPUT_DIR"
trap - EXIT
echo "completed: $TASK_ID"
