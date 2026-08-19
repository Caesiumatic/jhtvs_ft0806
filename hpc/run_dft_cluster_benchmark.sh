#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe mpi 8
set -euo pipefail

: "${SGE_TASK_ID:?submit as an SGE array job}"
: "${DFT_MANIFEST_SHA256:?set DFT_MANIFEST_SHA256 at submission}"

REPO_ROOT="${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
MANIFEST="$REPO_ROOT/data/dft_cluster_benchmark/calculation_manifest.csv"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

[ -f "$MANIFEST" ] && [ "$(sha256_file "$MANIFEST")" = "$DFT_MANIFEST_SHA256" ] || {
  echo "DFT manifest is missing or changed" >&2
  exit 2
}
[ "${NSLOTS:-}" = "8" ] || { echo "expected 8 SGE slots" >&2; exit 2; }

source /etc/profile.d/modules.sh
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate screen2026
set -u
module load openmpi/4.1.8 orca/6.1.0-418
export OMP_NUM_THREADS=1
export OMPI_MCA_btl="^sm"
export OMPI_MCA_mpi_yield_when_idle=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

python workflows/dft_cluster_benchmark/run_case.py \
  --manifest "$MANIFEST" \
  --task-index "$SGE_TASK_ID" \
  --orca "$(command -v orca)" \
  --run-root "$REPO_ROOT/runs/dft_cluster_benchmark"
