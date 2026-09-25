"""Registered paths and protocol for Team/Player context-scale probes."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT


ACTOR_EXPERIMENT_ROOT = EXPERIMENT_ROOT / "actor_scale_v1"
ACTOR_TASKS = ("team", "player")
ACTOR_VIEWS = ("f80", "p1", "p2")


def validation_dir(task: str, view: str, seed: int) -> Path:
    return ACTOR_EXPERIMENT_ROOT / "validation" / task / view / f"seed{seed}"


def test_dir(task: str, view: str, seed: int) -> Path:
    return ACTOR_EXPERIMENT_ROOT / "test" / task / view / f"seed{seed}"

