#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKGROUND_DIR="${ROOT}/experiments/feasibility/semantic_hgt_v2/background"
STATUS_PATH="${BACKGROUND_DIR}/status.json"
mkdir -p "${BACKGROUND_DIR}"

write_status() {
    printf '{"status":"%s","stage":"%s","pid":%d,"updated_at":"%s"}\n' \
        "$1" "$2" "$$" "$(date --iso-8601=seconds)" > "${STATUS_PATH}"
}

on_exit() {
    local code=$?
    if [[ ${code} -eq 0 ]]; then
        write_status "completed" "ablation"
    else
        write_status "failed" "pipeline"
    fi
}
trap on_exit EXIT

cd "${ROOT}"
write_status "running" "graph_audit"
python -u scripts/build_semantic_graphs.py

write_status "waiting" "gpu_idle"
python -u scripts/wait_for_gpus.py --gpus 0 1 2 3

write_status "running" "smoke"
python -u scripts/run_semantic_hgt_matrix.py \
    --stage smoke --devices cuda:0 cuda:1 cuda:2

read -ra GPU_SLOTS <<< "$(python scripts/semantic_gpu_slots.py --gpus 0 1 2 3)"

write_status "running" "tune"
python -u scripts/run_semantic_hgt_matrix.py \
    --stage tune --devices "${GPU_SLOTS[@]}"

write_status "running" "final"
python -u scripts/run_semantic_hgt_matrix.py \
    --stage final --devices "${GPU_SLOTS[@]}"

write_status "running" "summary"
python -u scripts/summarize.py \
    --profile feasibility --unified-repair --semantic-hgt --replicates 10000

write_status "running" "ablation"
python -u scripts/run_semantic_relation_ablation.py \
    --contract seq2event --device cuda:0 \
    > "${BACKGROUND_DIR}/ablation_seq2event.log" 2>&1 &
pid_seq=$!
python -u scripts/run_semantic_relation_ablation.py \
    --contract unified_lem --device cuda:1 \
    > "${BACKGROUND_DIR}/ablation_unified_lem.log" 2>&1 &
pid_unified=$!
python -u scripts/run_semantic_relation_ablation.py \
    --contract nmstpp --device cuda:2 \
    > "${BACKGROUND_DIR}/ablation_nmstpp.log" 2>&1 &
pid_nmstpp=$!
wait "${pid_seq}" "${pid_unified}" "${pid_nmstpp}"
python -u scripts/run_semantic_relation_ablation.py --merge
