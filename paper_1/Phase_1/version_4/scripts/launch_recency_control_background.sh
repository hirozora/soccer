#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/li/anaconda3/bin/python}"
OUT="$ROOT/experiments/possession_recency_control_v1/background"
UNIT="football-hgt-v4-recency-control"
mkdir -p "$OUT"

if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "Recency-control study is already running as $UNIT.service"
  exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
systemd-run --user --unit "$UNIT" --collect \
  --property "WorkingDirectory=$ROOT" \
  --property "StandardOutput=append:$OUT/pipeline.log" \
  --property "StandardError=append:$OUT/pipeline.log" \
  "$PYTHON" "$ROOT/scripts/run_recency_control_matrix.py" \
    --stage pipeline --devices 0 1 2 3 --slots-per-gpu 2 >/dev/null
PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
printf '%s\n' "$PID" >"$OUT/pipeline.pid"
printf '{"service":"%s","main_pid":%s,"state":"started"}\n' \
  "$UNIT.service" "$PID" >"$OUT/launcher.json"
echo "Started $UNIT.service with PID $PID"
