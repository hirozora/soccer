#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/li/anaconda3/bin/python}"
BACKGROUND="$ROOT/experiments/five_task_fixed_budget_v1/background"
UNIT="football-hgt-targets-v4-fixed-budget-conflict"
mkdir -p "$BACKGROUND"
if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "Fixed-budget conflict study is already running as $UNIT.service"
  exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
systemd-run --user --unit "$UNIT" --collect \
  --property "WorkingDirectory=$ROOT" \
  --property "StandardOutput=append:$BACKGROUND/pipeline.log" \
  --property "StandardError=append:$BACKGROUND/pipeline.log" \
  "$PYTHON" "$ROOT/scripts/run_fixed_budget_matrix.py" \
    --stage pipeline --devices 0 1 2 3 --slots-per-gpu 2 >/dev/null
PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
printf '%s\n' "$PID" > "$BACKGROUND/pid"
printf '{"service":"%s","main_pid":%s,"state":"started"}\n' "$UNIT.service" "$PID" > "$BACKGROUND/launcher.json"
echo "Started $UNIT.service with PID $PID"
