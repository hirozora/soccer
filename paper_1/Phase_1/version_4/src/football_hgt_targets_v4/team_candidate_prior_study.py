"""Registered Team-aware Player candidate-prior experiment."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import training_dir as partial_training_dir


TEAM_CANDIDATE_PRIOR_ROOT = EXPERIMENT_ROOT / "team_candidate_prior_v1"
LAMBDA_GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
EPSILON = 1e-6
FOLD_COUNT = 5
SELECTOR_SEED = 20260815


def source_checkpoint(seed: int) -> Path:
    return partial_training_dir(seed) / "best_guarded_core.pt"


def cache_path(seed: int, split: str) -> Path:
    return TEAM_CANDIDATE_PRIOR_ROOT / "cache" / f"seed{seed}" / f"{split}.pt"


def validation_prediction_path(method: str) -> Path:
    return TEAM_CANDIDATE_PRIOR_ROOT / "validation" / f"{method}.parquet"


def test_prediction_path(method: str) -> Path:
    return TEAM_CANDIDATE_PRIOR_ROOT / "test" / f"{method}.parquet"


def lock_path() -> Path:
    return TEAM_CANDIDATE_PRIOR_ROOT / "selection" / "team_candidate_prior_lock.json"

