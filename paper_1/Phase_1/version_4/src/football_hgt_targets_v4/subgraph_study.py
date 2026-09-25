"""Paths and fixed protocol definitions for the subgraph scale study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT
from .subgraph_views import ROUND_A_VIEWS, ROUND_B_BY_FAMILY, resolve_view_spec


SUBGRAPH_EXPERIMENT_ROOT = EXPERIMENT_ROOT / "subgraph_scale_v1"
F80_VALIDATION_ROOT = EXPERIMENT_ROOT / "possession_regression/validation/features/d2_dynamic"
F80_TEST_ROOT = EXPERIMENT_ROOT / "possession_regression/test/d2_dynamic"


def validation_dir(view: str, seed: int) -> Path:
    normalized = view.lower()
    if normalized in ROUND_A_VIEWS:
        stage = "round_a"
    elif normalized in {item for values in ROUND_B_BY_FAMILY.values() for item in values}:
        stage = "round_b"
    elif normalized.startswith("random_"):
        stage = "controls"
    else:
        raise ValueError(f"Unknown trainable view {view!r}")
    return SUBGRAPH_EXPERIMENT_ROOT / "validation" / stage / normalized / f"seed{seed}"


def f80_validation_dir(seed: int) -> Path:
    return F80_VALIDATION_ROOT / f"seed{seed}"


def test_dir(view: str, seed: int) -> Path:
    if view == "f80":
        return F80_TEST_ROOT / f"seed{seed}"
    return SUBGRAPH_EXPERIMENT_ROOT / "test" / view / f"seed{seed}"


def read_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def metric_row(view: str, seed: int, result: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = result[split]
    return {
        "view": view,
        "family": resolve_view_spec(view).family,
        "seed": seed,
        "split": split,
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
    }


def completed_views() -> list[str]:
    values = ["f80"]
    for view in (*ROUND_A_VIEWS, *(item for values in ROUND_B_BY_FAMILY.values() for item in values)):
        if all((validation_dir(view, seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS):
            values.append(view)
    return values
