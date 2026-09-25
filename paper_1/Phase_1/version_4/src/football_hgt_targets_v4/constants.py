"""Immutable paths and experiment definitions."""

from __future__ import annotations

from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = VERSION_ROOT.parent
BENCHMARK_ROOT = PHASE_ROOT / "benchmark_unified_v1"
EVENT_TABLE_ROOT = PHASE_ROOT / "data/whyscout/processed/event_tables/v1/England"
SPLIT_PATH = PHASE_ROOT / "version_1/data_splits/temporal_match_split_v1.csv"
SEMANTIC_GRAPH_ROOT = PHASE_ROOT / "data/whyscout/processed/heterogeneous_graphs/semantic_v2"
POSSESSION_GRAPH_ROOT = (
    PHASE_ROOT
    / "data/whyscout/processed/heterogeneous_graphs/semantic_v3_possession"
)
FEASIBILITY_ARTIFACT = BENCHMARK_ROOT / "artifacts/feasibility/protocol.pt"
FULL_ARTIFACT = BENCHMARK_ROOT / "artifacts/protocol.pt"
SAMPLE_PLAN = BENCHMARK_ROOT / "artifacts/feasibility/sample_plan.json"
EXPERIMENT_ROOT = VERSION_ROOT / "experiments"
ANALYSIS_ROOT = VERSION_ROOT / "artifacts/target_analysis"

WINDOW_SIZE = 80
LEARNING_RATE = 9e-4
SCREEN_SEED = 20260715
CONFIRMATION_SEEDS = (20260715, 20260716, 20260717)
EVENT_METHODS = ("ce", "inverse_ce", "sqrt_capped_ce", "balanced_softmax")
TIME_METHODS = ("current_huber", "log1p_huber", "bucket_offset")
POSITION_METHODS = ("xy", "zone", "zone_residual")
METHODS_BY_TASK = {
    "event": EVENT_METHODS,
    "time": TIME_METHODS,
    "position": POSITION_METHODS,
}
CONFIRMATION_METHODS = {
    "event": ("inverse_ce", "ce", "sqrt_capped_ce"),
    "time": ("current_huber", "log1p_huber"),
    "position": ("xy", "zone_residual"),
}
ORIGINAL_JOINT_METHODS = {
    "event": "inverse_ce",
    "time": "current_huber",
    "position": "xy",
}
LOSS_BALANCE_SCALES = (0.05, 0.10, 0.20)
OPTIMIZED_JOINT_METHODS = {
    "event": "ce",
    "time": "current_huber",
    "position": "xy",
}
TIME_BOUNDARIES = (0.0, 2.0, 5.0, 15.0, 60.0)
POSSESSION_TOPOLOGIES = ("none", "membership", "owner", "transition")
POSSESSION_FEATURE_LEVELS = ("topology", "categorical", "dynamic")
RAW_EVENT_NAMES = (
    "Duel",
    "Foul",
    "Free Kick",
    "Goalkeeper leaving line",
    "Interruption",
    "Offside",
    "Others on the ball",
    "Pass",
    "Save attempt",
    "Shot",
)
