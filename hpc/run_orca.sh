#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${TASK_FILE:?set TASK_FILE to the absolute prepared task table}"
: "${TASK_FILE_SHA256:?set TASK_FILE_SHA256 to the prepared table hash}"
: "${SGE_TASK_ID:?submit as an SGE array job}"

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
TASK_FILE="$(cd "$(dirname "$TASK_FILE")" && pwd)/$(basename "$TASK_FILE")"
[ -f "$TASK_FILE" ] && [ ! -L "$TASK_FILE" ] || {
  echo "missing or unsafe task table: $TASK_FILE" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

[ "$(sha256_file "$TASK_FILE")" = "$TASK_FILE_SHA256" ] || {
  echo "task table hash mismatch" >&2
  exit 2
}
[ "${NSLOTS:-}" = "8" ] || {
  echo "ORCA submission requires exactly 8 allocated ranks" >&2
  exit 2
}

source /etc/profile.d/modules.sh
module load openmpi/4.1.8 orca/6.1.0-418
ORCA="$(command -v orca)"
[ -n "$ORCA" ] || { echo "orca not found" >&2; exit 2; }
export OMPI_MCA_btl="^sm"
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

mapfile -t TASK_ROWS < <(awk -F '\t' -v task="$SGE_TASK_ID" 'NR > 1 && $1 == task' "$TASK_FILE")
[ "${#TASK_ROWS[@]}" -gt 0 ] || {
  echo "no logical jobs for array task $SGE_TASK_ID" >&2
  exit 2
}

for TASK_ROW in "${TASK_ROWS[@]}"; do
  IFS=$'\t' read -r ARRAY_TASK SEQUENCE LOGICAL_JOB_ID JOB_CLASS INPUT_REL INPUT_SHA \
    OUTPUT_REL NPROCS PLANNING_CORE_H WORKFLOW_REVISION METHOD_ID <<< "$TASK_ROW"
  [ "$ARRAY_TASK" = "$SGE_TASK_ID" ]
  [ "$NPROCS" = "8" ]
  case "$JOB_CLASS:$INPUT_REL" in
    diagnostic_gas_sp:runs/orca/sp/*/*.inp|smd_energy_sp:runs/orca/sp/*/*.inp) ;;
    optfreq:runs/orca/optfreq/*/*.inp) ;;
    *) echo "unsafe ORCA input path for $LOGICAL_JOB_ID: $INPUT_REL" >&2; exit 2 ;;
  esac
  case "$OUTPUT_REL" in
    runs/orca/sp/*/*.out|runs/orca/optfreq/*/*.out) ;;
    *) echo "unsafe ORCA output path for $LOGICAL_JOB_ID: $OUTPUT_REL" >&2; exit 2 ;;
  esac
  [ "$OUTPUT_REL" = "${INPUT_REL%.inp}.out" ] || {
    echo "input/output identity mismatch for $LOGICAL_JOB_ID" >&2
    exit 2
  }
  INPUT="$REPO_ROOT/$INPUT_REL"
  OUTPUT="$REPO_ROOT/$OUTPUT_REL"
  [ -f "$INPUT" ] && [ ! -L "$INPUT" ] || {
    echo "missing or unsafe ORCA input for $LOGICAL_JOB_ID" >&2
    exit 2
  }
  [ "$(sha256_file "$INPUT")" = "$INPUT_SHA" ] || {
    echo "input hash mismatch for $LOGICAL_JOB_ID" >&2
    exit 2
  }
  grep -Fqx "# job_id: $LOGICAL_JOB_ID" "$INPUT"
  grep -Fqx "# workflow_revision: $WORKFLOW_REVISION" "$INPUT"
  grep -Fqx "# method_id: $METHOD_ID" "$INPUT"
  grep -Fqx "%pal nprocs 8 end" "$INPUT"
  grep -Fqx "%maxcore 3000" "$INPUT"

  if [ -e "$OUTPUT" ]; then
    if grep -Fqx "# input_sha256: $INPUT_SHA" "$OUTPUT" \
        && grep -Fq "ORCA TERMINATED NORMALLY" "$OUTPUT" \
        && ! grep -Fq "ERROR !!!" "$OUTPUT"; then
      echo "already complete: $LOGICAL_JOB_ID"
      continue
    fi
    echo "refusing existing incomplete or mismatched output: $OUTPUT" >&2
    exit 3
  fi

  RUN_DIR="$(dirname "$INPUT")"
  BASE="$(basename "${INPUT%.inp}")"
  (
    cd "$RUN_DIR"
    set -o noclobber
    {
      printf '# job_id: %s\n' "$LOGICAL_JOB_ID"
      printf '# input_sha256: %s\n' "$INPUT_SHA"
      printf '# workflow_revision: %s\n' "$WORKFLOW_REVISION"
      printf '# method_id: %s\n' "$METHOD_ID"
      printf '# scheduler_job_id: %s\n' "${JOB_ID:-unknown}"
      printf '# scheduler_array_task: %s\n' "$SGE_TASK_ID"
      printf '# bundle_sequence: %s\n' "$SEQUENCE"
      printf '# planned_core_h: %s\n' "$PLANNING_CORE_H"
    } > "$BASE.out"
    ORCA_STATUS=0
    "$ORCA" "$BASE.inp" >> "$BASE.out" || ORCA_STATUS=$?
    if [ "$ORCA_STATUS" -ne 0 ]; then
      echo "ORCA process exited with status $ORCA_STATUS for $LOGICAL_JOB_ID" >&2
      exit 2
    fi
    if grep -Fq "ERROR !!!" "$BASE.out"; then
      echo "ORCA reported ERROR !!! for $LOGICAL_JOB_ID" >&2
      exit 2
    fi
    if ! grep -Fq "ORCA TERMINATED NORMALLY" "$BASE.out"; then
      echo "ORCA normal-termination footer is missing for $LOGICAL_JOB_ID" >&2
      exit 2
    fi
  )
  echo "completed: $LOGICAL_JOB_ID"
done
