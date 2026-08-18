#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p runs/fadel_benchmark/scheduler_logs
qsub hpc/run_fadel_benchmark.sh
