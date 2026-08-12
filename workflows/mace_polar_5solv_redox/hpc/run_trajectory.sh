#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${SGE_TASK_ID:?submit as an SGE array}"
: "${RAW_ROOT:?set RAW_ROOT to the non-Git workflow data root}"
: "${TRAJECTORY_MODE:?set TRAJECTORY_MODE to pilot or production}"
: "${TASK_TABLE_SHA256:?set the frozen task-table SHA256}"
: "${REPOSITORY_COMMIT:?set the frozen repository commit}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jhtvs-ft0806}"
MACE_DEVICE="${MACE_DEVICE:-cpu}"
MD_CHUNKS_PER_JOB="${MD_CHUNKS_PER_JOB:-10}"

case "$TRAJECTORY_MODE" in
  pilot|calibration|validation|production) ;;
  *) exit 64 ;;
esac
case "$MACE_DEVICE" in
  cpu|cuda) ;;
  *) exit 64 ;;
esac

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
RAW_ROOT="$(cd "$RAW_ROOT" && pwd)"
TASK_TABLE="$RAW_ROOT/${TRAJECTORY_MODE}_trajectory_tasks.tsv"
[ -f "$TASK_TABLE" ] && [ ! -L "$TASK_TABLE" ]
[ "$(sha256sum "$TASK_TABLE" | awk '{print $1}')" = "$TASK_TABLE_SHA256" ]
[ -n "$(awk -F '\t' -v task="$SGE_TASK_ID" 'NR > 1 && $1 == task {print $0}' "$TASK_TABLE")" ]

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "$CONDA_ENV_NAME"
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"
[ "$(git rev-parse --verify HEAD)" = "$REPOSITORY_COMMIT" ]

export OMP_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

CHECKPOINT="$HOME/.cache/mace/MACEPOLAR1Lmodel"
[ -f "$CHECKPOINT" ]
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" ]

PYTHONPATH=src python -m jhtvs_ft0806.explicit_redox.trajectory run-task \
  --raw-root "$RAW_ROOT" \
  --mode "$TRAJECTORY_MODE" \
  --task-index "$SGE_TASK_ID" \
  --checkpoint polar-1-l \
  --device "$MACE_DEVICE" \
  --max-md-chunks "$MD_CHUNKS_PER_JOB"
