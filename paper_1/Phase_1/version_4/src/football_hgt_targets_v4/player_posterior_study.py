"""Registered Player posterior conditioning Stage B experiment."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT


PLAYER_POSTERIOR_ROOT = EXPERIMENT_ROOT / "player_posterior_stage_b_v1"
FAMILIES = ("event", "position")
CONDITIONS = ("null", "base_post", "team_post")
TEAM_PRIOR_LAMBDA = 1.5
EPSILON = 1e-6


def condition_cache_path(seed: int, split: str) -> Path:
    return PLAYER_POSTERIOR_ROOT / "cache" / f"seed{seed}" / f"{split}.pt"


def training_dir(family: str, condition: str, seed: int) -> Path:
    return PLAYER_POSTERIOR_ROOT / "training" / family / condition / f"seed{seed}"


def test_dir(family: str, condition: str, seed: int) -> Path:
    return PLAYER_POSTERIOR_ROOT / "test" / family / condition / f"seed{seed}"


def decision_path() -> Path:
    return PLAYER_POSTERIOR_ROOT / "selection" / "player_posterior_dependency.json"

