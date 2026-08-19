#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
MANIFEST="data/dft_cluster_benchmark/calculation_manifest.csv"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST" >&2; exit 2; }
mkdir -p runs/dft_cluster_benchmark/scheduler_logs
if command -v sha256sum >/dev/null 2>&1; then
  MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
else
  MANIFEST_SHA="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"
fi
qsub -N DFTCAS145 -t 1-145 -tc 24 -v "DFT_MANIFEST_SHA256=$MANIFEST_SHA" \
  -o runs/dft_cluster_benchmark/scheduler_logs hpc/run_dft_cluster_benchmark.sh
