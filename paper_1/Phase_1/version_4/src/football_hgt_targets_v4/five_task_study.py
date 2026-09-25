"""Registered Semantic V3 five-task view configurations."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT


FIVE_TASK_EXPERIMENT_ROOT = EXPERIMENT_ROOT / "five_task_view_v1"
FIVE_TASK_MODES = ("five_f80", "five_hard", "five_soft")
FIVE_TASK_NAMES = ("event", "time", "position", "team", "player")
FIVE_TASK_WEIGHTS = {
    "event": 0.2,
    "time": 1.0,
    "position": 1.0,
    "team": 0.05,
    "player": 0.4,
}
FIVE_TASK_LOSS_DIVISOR = 3.0


def validation_dir(mode: str, seed: int) -> Path:
    return FIVE_TASK_EXPERIMENT_ROOT / "validation" / mode / f"seed{seed}"


def test_dir(mode: str, seed: int) -> Path:
    return FIVE_TASK_EXPERIMENT_ROOT / "test" / mode / f"seed{seed}"

