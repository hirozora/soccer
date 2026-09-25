"""Registered layer-wise partial-sharing experiment."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .fixed_budget_study import FIXED_BUDGET_ROOT


PARTIAL_SHARING_ROOT = EXPERIMENT_ROOT / "layerwise_partial_sharing_v1"
REFERENCE_ROOT = FIXED_BUDGET_ROOT
CONFIGURATION = "partial_l2"
REFERENCE_CONFIGURATIONS = (
    "three_f80",
    "t4_team",
    "five_f80",
    "t5_player_adapter",
)


def training_dir(seed: int) -> Path:
    return PARTIAL_SHARING_ROOT / "training" / CONFIGURATION / f"seed{seed}"


def test_dir(seed: int) -> Path:
    return PARTIAL_SHARING_ROOT / "test" / CONFIGURATION / f"seed{seed}"
