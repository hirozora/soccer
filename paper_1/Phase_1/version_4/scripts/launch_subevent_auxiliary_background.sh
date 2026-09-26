#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/experiments/subevent_auxiliary_v1/background"
test "$(loginctl show-user "$USER" -p Linger --value)" = yes
mkdir -p "$OUT"
systemd-run --user --unit=football-hgt-v4-subevent-auxiliary --collect \
  --property="WorkingDirectory=$ROOT" \
  --property="StandardOutput=append:$OUT/pipeline.log" \
  --property="StandardError=append:$OUT/pipeline.log" \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=PYTHONUNBUFFERED=1 \
  /home/li/anaconda3/bin/python "$ROOT/scripts/run_subevent_auxiliary.py" \
  --stage pipeline --devices 0 1 2 3 --device cuda:0

