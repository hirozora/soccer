"""Registered TrueTeam and TruePlayerState dependency probes."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import training_dir as partial_l2_training_dir


ORACLE_DEPENDENCY_ROOT = EXPERIMENT_ROOT / "oracle_dependency_probe_v1"
FAMILIES = ("player_team", "position_player_state", "event_player_state")
CONDITIONS = ("null", "oracle", "shuffled")


def source_checkpoint(seed: int) -> Path:
    return partial_l2_training_dir(seed) / "best_guarded_core.pt"


def cache_path(seed: int, split: str) -> Path:
    return ORACLE_DEPENDENCY_ROOT / "cache" / f"seed{seed}" / f"{split}.pt"


def training_dir(family: str, condition: str, seed: int) -> Path:
    return ORACLE_DEPENDENCY_ROOT / "training" / family / condition / f"seed{seed}"


def test_dir(family: str, condition: str, seed: int) -> Path:
    return ORACLE_DEPENDENCY_ROOT / "test" / family / condition / f"seed{seed}"


def validation_lock_path() -> Path:
    return ORACLE_DEPENDENCY_ROOT / "selection" / "dependency_decision.json"
