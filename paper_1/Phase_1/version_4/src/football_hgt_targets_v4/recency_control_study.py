"""Registered paths and comparisons for possession/recency controls."""

from __future__ import annotations

from pathlib import Path

from .constants import EXPERIMENT_ROOT
from .multiview_study import test_dir as multiview_test_dir
from .multiview_study import validation_dir as multiview_validation_dir
from .subgraph_study import f80_validation_dir, test_dir as subgraph_test_dir
from .subgraph_study import validation_dir as subgraph_validation_dir


RECENCY_CONTROL_ROOT = EXPERIMENT_ROOT / "possession_recency_control_v1"
NEW_CONFIGS = ("lp1", "lp2", "recency_sf_b")
ALL_CONFIGS = ("f80", "p1", "lp1", "p2", "lp2", "semantic_sf_b", "recency_sf_b")


def validation_dir(config: str, seed: int) -> Path:
    if config == "f80":
        return f80_validation_dir(seed)
    if config in {"p1", "p2"}:
        return subgraph_validation_dir(config, seed)
    if config == "semantic_sf_b":
        return multiview_validation_dir("sf_b", seed)
    if config in NEW_CONFIGS:
        return RECENCY_CONTROL_ROOT / "validation" / config / f"seed{seed}"
    raise ValueError(f"Unknown recency-control config {config!r}")


def test_dir(config: str, seed: int) -> Path:
    if config in {"f80", "p1", "p2"}:
        return subgraph_test_dir(config, seed)
    if config == "semantic_sf_b":
        return multiview_test_dir("sf_b", seed)
    if config in NEW_CONFIGS:
        return RECENCY_CONTROL_ROOT / "test" / config / f"seed{seed}"
    raise ValueError(f"Unknown recency-control config {config!r}")


def lock_path() -> Path:
    return RECENCY_CONTROL_ROOT / "selection" / "validation_locked.json"
