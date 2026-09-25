#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKGROUND_DIR="${ROOT}/experiments/feasibility/unified_lem_repair/background"
STATUS_PATH="${BACKGROUND_DIR}/status.json"
mkdir -p "${BACKGROUND_DIR}"

write_status() {
    printf '{"status":"%s","stage":"%s","pid":%d,"updated_at":"%s"}\n' \
        "$1" "$2" "$$" "$(date --iso-8601=seconds)" > "${STATUS_PATH}"
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
python -u scripts/run_unified_repair_matrix.py \
    --stage tune --devices cuda:0 cuda:1 cuda:2

write_status "running" "final"
python -u scripts/run_unified_repair_matrix.py \
    --stage final --devices cuda:0 cuda:1 cuda:2

write_status "running" "summary"
python -u scripts/summarize.py \
    --profile feasibility --unified-repair --replicates 10000
