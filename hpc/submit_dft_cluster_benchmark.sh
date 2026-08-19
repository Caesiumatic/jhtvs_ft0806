#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
MANIFEST="data/dft_cluster_benchmark/calculation_manifest.csv"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST" >&2; exit 2; }

source /etc/profile.d/modules.sh
module load openmpi/4.1.8 orca/6.1.0-418
ORCA_LIBXC_FUNCTIONALS="$(orca -libxcfunctionals)"
grep -qi "hyb_mgga_x_m06_hf" <<< "$ORCA_LIBXC_FUNCTIONALS" || {
  echo "ORCA LibXC exchange component hyb_mgga_x_m06_hf is unavailable" >&2
  exit 2
}
grep -qi "mgga_c_m06_hf" <<< "$ORCA_LIBXC_FUNCTIONALS" || {
  echo "ORCA LibXC correlation component mgga_c_m06_hf is unavailable" >&2
  exit 2
}

mkdir -p runs/dft_cluster_benchmark/scheduler_logs
if command -v sha256sum >/dev/null 2>&1; then
  MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
else
  MANIFEST_SHA="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"
fi
qsub -N DFTCAS145 -t 1-145 -tc 24 -v "DFT_MANIFEST_SHA256=$MANIFEST_SHA" \
  -o runs/dft_cluster_benchmark/scheduler_logs hpc/run_dft_cluster_benchmark.sh
