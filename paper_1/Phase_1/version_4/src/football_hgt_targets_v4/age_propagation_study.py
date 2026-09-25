"""Paths and comparisons for task-conditioned age propagation."""

from __future__ import annotations

from pathlib import Path

from .age_pooling_study import test_dir as ap_test_dir
from .age_pooling_study import training_dir as ap_training_dir
from .constants import EXPERIMENT_ROOT
from .partial_sharing_study import test_dir as f80_test_dir
from .partial_sharing_study import training_dir as f80_training_dir


AGE_PROPAGATION_ROOT = EXPERIMENT_ROOT / "task_age_propagation_v1"
TRAINED = ("pg_constant", "pg_shared", "pg_task")
ALL = ("partial_l2_f80", "ap_task", *TRAINED)
COMPARISONS = (
    ("partial_l2_f80", "pg_constant"),
    ("pg_constant", "pg_shared"),
    ("pg_shared", "pg_task"),
    ("partial_l2_f80", "pg_shared"),
    ("partial_l2_f80", "pg_task"),
    ("ap_task", "pg_task"),
)
COST_ORDER = {
    "partial_l2_f80": 0,
    "pg_constant": 1,
    "pg_shared": 2,
    "pg_task": 3,
}


def training_dir(configuration: str, seed: int) -> Path:
    if configuration == "partial_l2_f80":
        return f80_training_dir(seed)
    if configuration == "ap_task":
        return ap_training_dir("ap_task", seed)
    return AGE_PROPAGATION_ROOT / "training" / configuration / f"seed{seed}"


def test_dir(configuration: str, seed: int) -> Path:
    if configuration == "partial_l2_f80":
        return f80_test_dir(seed)
    if configuration == "ap_task":
        return ap_test_dir("ap_task", seed)
    return AGE_PROPAGATION_ROOT / "test" / configuration / f"seed{seed}"


def checkpoint_path(configuration: str, seed: int) -> Path:
    return training_dir(configuration, seed) / "best_guarded_core.pt"


def lock_path() -> Path:
    return AGE_PROPAGATION_ROOT / "selection" / "validation_lock.json"
