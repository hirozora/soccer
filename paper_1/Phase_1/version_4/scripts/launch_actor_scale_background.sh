#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="football-hgt-actor-scale-v1"
LOG_DIR="$ROOT/experiments/actor_scale_v1/background"
mkdir -p "$LOG_DIR"

systemctl --user stop "$UNIT.service" >/dev/null 2>&1 || true
systemd-run --user --unit "$UNIT" --collect \
  --property=WorkingDirectory="$ROOT" \
  --setenv=PYTHONPATH="$ROOT/src:$ROOT/../benchmark_unified_v1/src" \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  bash -lc "python scripts/run_actor_scale_matrix.py --stage pipeline > '$LOG_DIR/pipeline.log' 2>&1"
systemctl --user show "$UNIT.service" -p MainPID --value > "$LOG_DIR/pid"
echo "$UNIT.service"

