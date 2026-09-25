#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/experiments/spatiotemporal_edge_v1/background"
mkdir -p "$OUT"
systemd-run --user --unit=football-hgt-v4-spatiotemporal-edge --collect \
  --property="WorkingDirectory=$ROOT" \
  --property="StandardOutput=append:$OUT/pipeline.log" \
  --property="StandardError=append:$OUT/pipeline.log" \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=PYTHONUNBUFFERED=1 \
  /home/li/anaconda3/bin/python "$ROOT/scripts/run_spatiotemporal_edge_matrix.py" \
  --stage pipeline --devices 0 1 2 3 --slots-per-gpu 2 --device cuda:0
