"""Paths and registered comparisons for task-specific age-aware readout."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import test_dir as mean_test_dir
from .partial_sharing_study import training_dir as mean_training_dir


AGE_POOL_ROOT = EXPERIMENT_ROOT / "task_age_pooling_v1"
TRAINED = ("ap_shared", "ap_task")
ALL = ("ap_mean", *TRAINED)
COMPARISONS = (
    ("ap_mean", "ap_shared"),
    ("ap_shared", "ap_task"),
    ("ap_mean", "ap_task"),
)


def training_dir(configuration: str, seed: int) -> Path:
    if configuration == "ap_mean":
        return mean_training_dir(seed)
    return AGE_POOL_ROOT / "training" / configuration / f"seed{seed}"


def test_dir(configuration: str, seed: int) -> Path:
    if configuration == "ap_mean":
        return mean_test_dir(seed)
    return AGE_POOL_ROOT / "test" / configuration / f"seed{seed}"


def checkpoint_path(configuration: str, seed: int) -> Path:
    return training_dir(configuration, seed) / "best_guarded_core.pt"


def lock_path() -> Path:
    return AGE_POOL_ROOT / "selection" / "validation_lock.json"
