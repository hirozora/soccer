"""Paths and definitions for Partial-L2 task-specific context validation."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import test_dir as f80_test_dir
from .partial_sharing_study import training_dir as f80_training_dir


PARTIAL_CONTEXT_ROOT = EXPERIMENT_ROOT / "partial_l2_task_context_v1"
TRAINED_CONFIGURATIONS = ("partial_l2_hard", "partial_l2_soft")
ALL_CONFIGURATIONS = ("partial_l2", *TRAINED_CONFIGURATIONS)
COST_ORDER = {"partial_l2": 0, "partial_l2_hard": 1, "partial_l2_soft": 2}


def training_dir(configuration: str, seed: int) -> Path:
    if configuration == "partial_l2":
        return f80_training_dir(seed)
    return PARTIAL_CONTEXT_ROOT / "training" / configuration / f"seed{seed}"


def test_dir(configuration: str, seed: int) -> Path:
    if configuration == "partial_l2":
        return f80_test_dir(seed)
    return PARTIAL_CONTEXT_ROOT / "test" / configuration / f"seed{seed}"


def checkpoint_path(configuration: str, seed: int) -> Path:
    return training_dir(configuration, seed) / "best_guarded_core.pt"


def final_lock_path() -> Path:
    return PARTIAL_CONTEXT_ROOT / "selection" / "final_context_lock.json"
