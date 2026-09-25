#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/experiments/coverage_history_v1/background"
mkdir -p "$OUT"
systemd-run --user --unit=football-hgt-v4-coverage-history --collect \
  --property="WorkingDirectory=$ROOT" \
  --property="StandardOutput=append:$OUT/pipeline.log" \
  --property="StandardError=append:$OUT/pipeline.log" \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=PYTHONUNBUFFERED=1 \
  /home/li/anaconda3/bin/python "$ROOT/scripts/run_coverage_history.py" \
  --stage pipeline --devices 0 1 2 3 --device cuda:0
