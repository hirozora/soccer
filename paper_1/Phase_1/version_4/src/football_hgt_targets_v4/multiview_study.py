"""Paths and registered configurations for static task-view fusion."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT


MULTIVIEW_EXPERIMENT_ROOT = EXPERIMENT_ROOT / "task_view_fusion_v1"
EXPERIMENT_MODES = ("fixed_a", "fixed_b", "sf_a", "sf_b")


def validation_dir(mode: str, seed: int) -> Path:
    return MULTIVIEW_EXPERIMENT_ROOT / "validation" / mode / f"seed{seed}"


def test_dir(mode: str, seed: int) -> Path:
    return MULTIVIEW_EXPERIMENT_ROOT / "test" / mode / f"seed{seed}"

