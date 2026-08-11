#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
set -euo pipefail

: "${CLUSTER_MANIFEST_SHA256:?set the immutable cluster-manifest SHA-256}"

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
MANIFEST="$REPO_ROOT/diagnostics/explicit_solvation_sp/cluster_manifest.csv"
OUTPUT="$REPO_ROOT/diagnostics/explicit_solvation_sp/mace/raw_results.json"

[ -f "$MANIFEST" ] && [ ! -L "$MANIFEST" ]
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$CLUSTER_MANIFEST_SHA256" ]
[ ! -e "$OUTPUT" ] || {
  echo "refusing to replace existing MACE result: $OUTPUT" >&2
  exit 2
}

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate jhtvs-ft0806
set -u
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${NSLOTS:-8}"
export MKL_NUM_THREADS="${NSLOTS:-8}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-8}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-8}"

CHECKPOINT="$HOME/.cache/mace/MACEPOLAR1Lmodel"
[ -f "$CHECKPOINT" ]
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b" ]

PYTHONPATH=src python diagnostics/explicit_solvation_sp/run_diagnostic.py \
  mace --checkpoint polar-1-l --device cpu

[ -f "$OUTPUT" ]
echo "completed: explicit-solvation MACE-POLAR-1-L SPE matrix"
