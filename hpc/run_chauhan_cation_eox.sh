#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -o runs/chauhan_cation_eox/scheduler_logs/$JOB_ID.$TASK_ID.out
#$ -t 1-105

set -euo pipefail

REPO="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
mkdir -p runs/chauhan_cation_eox/scheduler_logs

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate screen2026
set -u

export OMP_NUM_THREADS="${NSLOTS:-1}"
TASK_INDEX="${SGE_TASK_ID:?submit as an SGE array or set SGE_TASK_ID}"
python workflows/chauhan_cation_eox/run_calculation.py \
  --manifest data/chauhan_cation_eox/calculation_manifest.csv \
  --task-index "$TASK_INDEX"
