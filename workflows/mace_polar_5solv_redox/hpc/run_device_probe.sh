#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${PROBE_DEVICE:?set PROBE_DEVICE to cpu or cuda}"
: "${PROBE_GEOMETRY:?set PROBE_GEOMETRY to a repository XYZ}"
: "${PROBE_OUTPUT:?set PROBE_OUTPUT to a non-Git JSON path}"
: "${PROBE_CHARGE:?set PROBE_CHARGE}"
: "${PROBE_SPIN:?set PROBE_SPIN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-jhtvs-ft0806}"
PROBE_EVALUATIONS="${PROBE_EVALUATIONS:-1}"
case "$PROBE_DEVICE" in
  cpu|cuda) ;;
  *) exit 64 ;;
esac

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "$CONDA_ENV_NAME"
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO_ROOT"

PYTHONPATH=src python -m jhtvs_ft0806.explicit_redox.device_probe run \
  --geometry "$PROBE_GEOMETRY" \
  --charge "$PROBE_CHARGE" \
  --spin "$PROBE_SPIN" \
  --device "$PROBE_DEVICE" \
  --output "$PROBE_OUTPUT" \
  --evaluations "$PROBE_EVALUATIONS"
