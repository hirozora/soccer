#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/li/anaconda3/bin/python}"
BACKGROUND="$ROOT/experiments/loss_balance/background"
UNIT="football-hgt-targets-v4-loss-balance"
mkdir -p "$BACKGROUND"

if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "Loss-balance pipeline is already running as $UNIT.service"
  exit 0
fi

systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
systemd-run --user --unit "$UNIT" --collect \
  --property "WorkingDirectory=$ROOT" \
  --property "StandardOutput=append:$BACKGROUND/pipeline.log" \
  --property "StandardError=append:$BACKGROUND/pipeline.log" \
  "$PYTHON" "$ROOT/scripts/run_matrix.py" \
    --stage balance_pipeline --devices 0 1 2 3 --slots-per-gpu 3 >/dev/null
PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
printf '%s\n' "$PID" > "$BACKGROUND/pid"
printf '{"service": "%s", "main_pid": %s, "state": "started"}\n' \
  "$UNIT.service" "$PID" > "$BACKGROUND/launcher.json"
echo "Started loss-balance pipeline as $UNIT.service with PID $PID"
