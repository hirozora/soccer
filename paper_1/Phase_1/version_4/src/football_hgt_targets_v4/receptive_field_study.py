"""Paths and registered comparisons for strict task receptive fields."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import test_dir as f80_test_dir
from .partial_sharing_study import training_dir as f80_training_dir


RF_ROOT = EXPERIMENT_ROOT / "task_receptive_field_v1"
TRAINED = ("rf_core_u5", "rf_core_u10", "rf_task")
ALL = ("rf_f80", *TRAINED)


def training_dir(configuration: str, seed: int) -> Path:
    if configuration == "rf_f80":
        return f80_training_dir(seed)
    return RF_ROOT / "training" / configuration / f"seed{seed}"


def test_dir(configuration: str, seed: int) -> Path:
    if configuration == "rf_f80":
        return f80_test_dir(seed)
    return RF_ROOT / "test" / configuration / f"seed{seed}"


def checkpoint_path(configuration: str, seed: int) -> Path:
    return training_dir(configuration, seed) / "best_guarded_core.pt"


def lock_path() -> Path:
    return RF_ROOT / "selection" / "validation_lock.json"
