#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/experiments/task_receptive_field_v1/background"
UNIT="football-hgt-v4-task-rf"
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
  "$ROOT/scripts/wait_then_run_receptive_field.sh" >/dev/null
PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
printf '%s\n' "$PID" >"$OUT/pid"
printf '{"service":"%s","main_pid":%s,"waiting_for":"football-hgt-v4-recency-control.service"}\n' "$UNIT.service" "$PID" >"$OUT/launcher.json"
echo "Started $UNIT.service with PID $PID"
