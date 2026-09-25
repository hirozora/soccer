#!/usr/bin/env bash
set -euo pipefail

CURRENT_UNIT="football-hgt-v4-recency-control.service"
while systemctl --user is-active --quiet "$CURRENT_UNIT"; do
  sleep 30
done

exec /home/li/anaconda3/bin/python \
  /home/li/football/paper_1/Phase_1/version_4/scripts/run_receptive_field_matrix.py \
  --stage pipeline --devices 0 1 2 3 --slots-per-gpu 2
