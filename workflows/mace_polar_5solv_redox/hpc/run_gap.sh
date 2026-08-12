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
TASK_INDEX="$SGE_TASK_ID"
if [ -n "${TASK_INDEX_MAP:-}" ]; then
  TASK_INDEX="$(printf '%s\n' "$TASK_INDEX_MAP" | awk -F ':' -v n="$SGE_TASK_ID" '{print $n}')"
  [ -n "$TASK_INDEX" ]
fi

case "$TRAJECTORY_MODE" in
  pilot|calibration|validation|production) ;;
  *) exit 64 ;;
esac
case "$MACE_DEVICE" in
  cpu|cuda) ;;
  *) exit 64 ;;
esac

if [ "$MACE_DEVICE" = "cuda" ]; then
  GPU_LOCK_FD=""
  for gpu_index in 0 1 2 3; do
    lock_file="/tmp/jhtvs-mace-gpu-${HOSTNAME}-${gpu_index}.lock"
    exec {candidate_fd}>"$lock_file"
    if flock -n "$candidate_fd"; then
      memory_used_mib="$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits)"
      if [ "$memory_used_mib" -le 128 ]; then
        export CUDA_VISIBLE_DEVICES="$gpu_index"
        GPU_LOCK_FD="$candidate_fd"
        break
      fi
      flock -u "$candidate_fd"
    fi
    eval "exec ${candidate_fd}>&-"
  done
  [ -n "$GPU_LOCK_FD" ]
  echo "MACE CUDA assignment host=$HOSTNAME physical_gpu=$CUDA_VISIBLE_DEVICES lock_fd=$GPU_LOCK_FD"
fi

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
RAW_ROOT="$(cd "$RAW_ROOT" && pwd)"
TASK_TABLE="$RAW_ROOT/${TRAJECTORY_MODE}_trajectory_tasks.tsv"
[ -f "$TASK_TABLE" ] && [ ! -L "$TASK_TABLE" ]
[ "$(sha256sum "$TASK_TABLE" | awk '{print $1}')" = "$TASK_TABLE_SHA256" ]
[ -n "$(awk -F '\t' -v task="$TASK_INDEX" 'NR > 1 && $1 == task {print $0}' "$TASK_TABLE")" ]

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

PYTHONPATH=src python -m jhtvs_ft0806.explicit_redox.analysis evaluate-gaps \
  --raw-root "$RAW_ROOT" \
  --mode "$TRAJECTORY_MODE" \
  --task-index "$TASK_INDEX" \
  --checkpoint polar-1-l \
  --device "$MACE_DEVICE"
