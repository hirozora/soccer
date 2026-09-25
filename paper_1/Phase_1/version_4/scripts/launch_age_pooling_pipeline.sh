#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/experiments/task_age_pooling_v1/background"
UNIT="football-hgt-v4-age-pooling"
mkdir -p "$OUT"
if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "$UNIT.service is already active"
  exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
systemd-run --user --unit "$UNIT" --collect \
  --property "WorkingDirectory=$ROOT" \
  --property "StandardOutput=append:$OUT/pipeline.log" \
  --property "StandardError=append:$OUT/pipeline.log" \
  /home/li/anaconda3/bin/python "$ROOT/scripts/run_age_pooling_matrix.py" \
    --stage pipeline --devices 0 1 2 3 --slots-per-gpu 2 >/dev/null
PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
printf '%s\n' "$PID" >"$OUT/pid"
printf '{"service":"%s","main_pid":%s}\n' "$UNIT.service" "$PID" >"$OUT/launcher.json"
echo "Started $UNIT.service with PID $PID"
