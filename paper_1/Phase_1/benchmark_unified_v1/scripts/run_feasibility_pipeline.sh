#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_PATH="${ROOT}/experiments/feasibility/background/status.json"
DEVICES=(cuda:0 cuda:1 cuda:2 cuda:3)
WORKERS_PER_DEVICE="${WORKERS_PER_DEVICE:-3}"

mkdir -p "$(dirname "${STATUS_PATH}")"

write_status() {
    local status="$1"
    local stage="$2"
    printf '{"status":"%s","stage":"%s","pid":%d,"updated_at":"%s"}\n' \
        "${status}" "${stage}" "$$" "$(date --iso-8601=seconds)" > "${STATUS_PATH}"
}

on_exit() {
    local code=$?
    if [[ ${code} -eq 0 ]]; then
        write_status "completed" "summary"
    else
        write_status "failed" "pipeline"
    fi
}
trap on_exit EXIT

cd "${ROOT}"
write_status "running" "tune"
python -u scripts/run_feasibility_matrix.py \
    --stage tune --devices "${DEVICES[@]}" \
    --workers-per-device "${WORKERS_PER_DEVICE}"

write_status "running" "final"
python -u scripts/run_feasibility_matrix.py \
    --stage final --devices "${DEVICES[@]}" \
    --workers-per-device "${WORKERS_PER_DEVICE}"

write_status "running" "summary"
python -u scripts/summarize.py --profile feasibility --replicates 10000
