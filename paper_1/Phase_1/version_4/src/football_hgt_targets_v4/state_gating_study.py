"""Paths and constants for state-aware relation gating."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import PARTIAL_SHARING_ROOT


STATE_GATING_ROOT = EXPERIMENT_ROOT / "state_aware_relation_gating_v1"
CONFIGURATION = "state_gated_partial_l2"
G0_ROOT = PARTIAL_SHARING_ROOT


def training_dir(seed: int) -> Path:
    return STATE_GATING_ROOT / "training" / "g1" / f"seed{seed}"


def test_dir(seed: int) -> Path:
    return STATE_GATING_ROOT / "test" / "g1" / f"seed{seed}"

