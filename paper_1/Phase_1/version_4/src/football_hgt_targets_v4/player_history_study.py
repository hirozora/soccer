"""Registered paths and definitions for Player-history Stage A."""

from __future__ import annotations

from pathlib import Path

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT
from .partial_sharing_study import training_dir as partial_training_dir


PLAYER_HISTORY_ROOT = EXPERIMENT_ROOT / "player_history_stage_a_v1"
CONDITIONS = ("ph_null", "ph_hist_k5", "ph_shuffled_team")
HISTORY_LENGTH = 5
HISTORY_STATS_DIM = 23
SELECTOR_SEED = 20260815


def source_checkpoint(seed: int) -> Path:
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError(f"Unsupported seed: {seed}")
    return partial_training_dir(seed) / "best_guarded_core.pt"


def training_dir(condition: str, seed: int) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown Player-history condition: {condition}")
    return PLAYER_HISTORY_ROOT / "training" / condition / f"seed{seed}"


def test_dir(condition: str, seed: int) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown Player-history condition: {condition}")
    return PLAYER_HISTORY_ROOT / "test" / condition / f"seed{seed}"


def decision_path() -> Path:
    return PLAYER_HISTORY_ROOT / "selection/player_history_decision.json"

