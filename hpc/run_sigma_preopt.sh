#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the absolute geometry run directory}"
: "${SGE_TASK_ID:?submit as an SGE array job}"

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate screen2026
set -u

command -v xtb >/dev/null
command -v sha256sum >/dev/null
export OMP_NUM_THREADS="${NSLOTS:-1}"

RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
TASK_TABLE="$RUN_ROOT/sigma_preopt_array.tsv"
[ -f "$TASK_TABLE" ] && [ ! -L "$TASK_TABLE" ]

TASK_ROW="$(sed -n "${SGE_TASK_ID}p" "$TASK_TABLE")"
[ -n "$TASK_ROW" ]
IFS=$'\t' read -r TASK_ID SOURCE_XYZ SOURCE_SHA OUTPUT_DIR CHARGE UHF EPSILON TOPOLOGY_SHA <<< "$TASK_ROW"

SOURCE_XYZ="$RUN_ROOT/$SOURCE_XYZ"
OUTPUT_DIR="$RUN_ROOT/$OUTPUT_DIR"
[ -f "$SOURCE_XYZ" ] && [ ! -L "$SOURCE_XYZ" ]
[ "$(sha256sum "$SOURCE_XYZ" | awk '{print $1}')" = "$SOURCE_SHA" ]
[ "$CHARGE" = "2" ]
[ "$UHF" = "0" ]
[ -n "$EPSILON" ]
[ -n "$TOPOLOGY_SHA" ]

if [ -d "$OUTPUT_DIR" ]; then
  if [ -s "$OUTPUT_DIR/xtbopt.xyz" ] && [ -s "$OUTPUT_DIR/task_status.tsv" ] \
      && grep -Fq "$TASK_ID" "$OUTPUT_DIR/task_status.tsv" \
      && grep -Fq "$SOURCE_SHA" "$OUTPUT_DIR/task_status.tsv"; then
    echo "already complete: $TASK_ID"
    exit 0
  fi
  echo "refusing non-complete existing output directory: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
WORK_DIR="$(mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX")"
cleanup() {
  rm -rf -- "$WORK_DIR"
}
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
