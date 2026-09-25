"""Immutable label and spatial constants used by every benchmark model."""

from __future__ import annotations

from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = BENCHMARK_ROOT.parent
GRAPH_ROOT = (
    PHASE_ROOT
    / "data/whyscout/processed/heterogeneous_graphs/v1"
)
SEMANTIC_GRAPH_ROOT = (
    PHASE_ROOT
    / "data/whyscout/processed/heterogeneous_graphs/semantic_v2"
)
POSSESSION_GRAPH_ROOT = (
    PHASE_ROOT
    / "data/whyscout/processed/heterogeneous_graphs/semantic_v3_possession"
)
POSSESSION_V2_ROOT = (
    PHASE_ROOT
    / "data/whyscout/processed/inferred_possessions/v2/England"
)
SPLIT_PATH = PHASE_ROOT / "version_1/data_splits/temporal_match_split_v1.csv"
VOCAB_PATH = GRAPH_ROOT / "metadata/vocabularies.json"
DEFAULT_ARTIFACT_PATH = BENCHMARK_ROOT / "artifacts/protocol.pt"
FEASIBILITY_ARTIFACT_PATH = BENCHMARK_ROOT / "artifacts/feasibility/protocol.pt"
FEASIBILITY_SAMPLE_PLAN_PATH = (
    BENCHMARK_ROOT / "artifacts/feasibility/sample_plan.json"
)

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
RAW_EVENT_ID_TO_INDEX = {index + 1: index for index in range(len(RAW_EVENT_NAMES))}

ACTION_NAMES = ("pass", "dribble", "cross", "shot")
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}

# This order is copied from the official Unified LEM tokenizer.
FINE_EVENT_NAMES = (
    "pass",
    "long_pass",
    "cross",
    "touch",
    "aerial_duel",
    "clearance",
    "interception",
    "loose_ball_duel",
    "defensive_duel",
    "offensive_duel",
    "dribble",
    "carry",
    "game_interruption",
    "own_goal",
    "throw_in",
    "free_kick",
    "goal_kick",
    "infraction",
    "corner",
    "acceleration",
    "offside",
    "right_foot_shot",
    "left_foot_shot",
    "head_shot",
    "goalkeeper_exit",
    "save",
    "shot_against",
    "fairplay",
    "yellow_card",
    "red_card",
    "first_half_end",
    "game_end",
)
FINE_EVENT_TO_INDEX = {name: index for index, name in enumerate(FINE_EVENT_NAMES)}

# Official NMSTPP Juego de Posicion centres in Wyscout 0..100 coordinates.
ZONE_CENTERS_100 = (
    (8.5, 89.45),
    (25.25, 89.45),
    (41.75, 89.45),
    (58.25, 89.45),
    (74.75, 89.45),
    (91.5, 89.45),
    (8.5, 10.55),
    (25.25, 10.55),
    (41.75, 10.55),
    (58.25, 10.55),
    (74.75, 10.55),
    (91.5, 10.55),
    (33.5, 71.05),
    (66.5, 71.05),
    (33.5, 50.0),
    (66.5, 50.0),
    (33.5, 28.95),
    (66.5, 28.95),
    (8.5, 50.0),
    (91.5, 50.0),
)

ACTION_TAGS = frozenset({501, 502, 503, 504})
LEFT_FOOT_TAG = 401
RIGHT_FOOT_TAG = 402
HEAD_BODY_TAG = 403
OWN_GOAL_TAG = 102
INTERCEPTION_TAG = 1401
FAIRPLAY_TAG = 1001
RED_CARD_TAG = 1701
YELLOW_CARD_TAGS = frozenset({1702, 1703})

TIME_CAP_SECONDS = 60.0

SEMANTIC_GRAPH_VERSION = "2.0.0"
POSSESSION_GRAPH_VERSION = "3.0.0"
TEMPORAL_RELATIONS = (
    "gap_0_2s",
    "gap_2_5s",
    "gap_5_15s",
    "gap_15_60s",
    "gap_60plus",
    "period_break",
)
PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0

CONTRACT_TASKS = {
    "seq2event": ("event", "position"),
    "unified_lem": ("event", "time", "position"),
    "nmstpp": ("event", "time", "position"),
}

NATIVE_WINDOWS = {"seq2event": 40, "unified_lem": 3, "nmstpp": 40}
FINAL_SEEDS = tuple(range(20260715, 20260720))
