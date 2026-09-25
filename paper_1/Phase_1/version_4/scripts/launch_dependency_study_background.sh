#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/li/anaconda3/bin/python}"
BACKGROUND="$ROOT/experiments/target_dependency_v1/background"
UNIT="football-hgt-target-dependency-v1"
mkdir -p "$BACKGROUND"

if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "Dependency study is already running as $UNIT.service"
  exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
systemd-run --user --unit "$UNIT" --collect \
  --property "WorkingDirectory=$ROOT" \
  --property "StandardOutput=append:$BACKGROUND/pipeline.log" \
  --property "StandardError=append:$BACKGROUND/pipeline.log" \
  "$PYTHON" "$ROOT/scripts/run_dependency_matrix.py" \
    --stage pipeline --devices 0 1 2 3 --slots-per-gpu 3 >/dev/null
PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
printf '%s\n' "$PID" > "$BACKGROUND/pid"
printf '{"service":"%s","main_pid":%s,"state":"started"}\n' "$UNIT.service" "$PID" > "$BACKGROUND/launcher.json"
echo "Started $UNIT.service with PID $PID"
