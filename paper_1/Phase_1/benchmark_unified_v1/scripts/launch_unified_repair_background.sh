#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKGROUND_DIR="${ROOT}/experiments/feasibility/unified_lem_repair/background"
PID_PATH="${BACKGROUND_DIR}/pipeline.pid"
LOG_PATH="${BACKGROUND_DIR}/pipeline.log"
mkdir -p "${BACKGROUND_DIR}"

if [[ -f "${PID_PATH}" ]]; then
    previous_pid="$(cat "${PID_PATH}")"
    if kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Unified repair pipeline is already running with PID ${previous_pid}" >&2
        exit 1
    fi
fi

nohup setsid bash "${ROOT}/scripts/run_unified_repair_pipeline.sh" \
    > "${LOG_PATH}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_PATH}"
echo "PID=${pid}"
echo "LOG=${LOG_PATH}"
echo "STATUS=${BACKGROUND_DIR}/status.json"
