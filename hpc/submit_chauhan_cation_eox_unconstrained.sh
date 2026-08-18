#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p runs/chauhan_cation_eox_unconstrained/scheduler_logs
qsub hpc/run_chauhan_cation_eox_unconstrained.sh
