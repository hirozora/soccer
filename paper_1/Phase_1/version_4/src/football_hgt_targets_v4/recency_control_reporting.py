"""Validation-locked reporting for equal-size possession/recency controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .multiview_reporting import PRACTICAL_THRESHOLDS, TASK_METRICS
from .recency_control_study import ALL_CONFIGS, RECENCY_CONTROL_ROOT, lock_path, test_dir, validation_dir
from .reporting import _match_statistics, _paired_hierarchical_bootstrap
from .subgraph_reporting import _paired_frame
from .subgraph_study import read_result


COMPARISONS = (
    ("lp1", "p1"),
    ("lp2", "p2"),
    ("recency_sf_b", "semantic_sf_b"),
    ("f80", "p1"),
    ("f80", "p2"),
    ("f80", "semantic_sf_b"),
    ("f80", "recency_sf_b"),
)


def _predictions(config: str, seed: int, split: str) -> pd.DataFrame:
    root = validation_dir(config, seed) if split == "validation" else test_dir(config, seed)
    return pd.read_parquet(root / f"{split}_predictions.parquet")


def _metric_row(config: str, seed: int, split: str) -> dict[str, Any]:
    root = validation_dir(config, seed) if split == "validation" else test_dir(config, seed)
    result = read_result(root)
    metrics = result[split]
    return {
        "config": config,
        "seed": seed,
        "split": split,
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
    }


def _bootstrap(reference: str, candidate: str, split: str) -> dict[str, Any]:
    left_stats, right_stats = [], []
    for seed in CONFIRMATION_SEEDS:
        left, right = _paired_frame(
            _predictions(reference, seed, split),
            _predictions(candidate, seed, split),
        )
        matches = sorted(left.match_id.astype(int).unique().tolist())
        left_stats.append(_match_statistics(left, matches))
        right_stats.append(_match_statistics(right, matches))
    return _paired_hierarchical_bootstrap(
        np.stack(left_stats), np.stack(right_stats), iterations=10_000
    )


def _semantic_decision(reference: str, candidate: str, split: str, bootstrap: dict[str, Any]) -> dict[str, Any]:
    differences = []
    for seed in CONFIRMATION_SEEDS:
        left = _metric_row(reference, seed, split)
        right = _metric_row(candidate, seed, split)
        differences.append({key: right[key] - left[key] for key in TASK_METRICS.values()})
    frame = pd.DataFrame(differences)
    tasks = {}
    for task, metric in TASK_METRICS.items():
        values = frame[metric]
        low, high = bootstrap[metric]["ci95"]
        lower_better = task in {"time", "position"}
        improving = int((values < 0).sum() if lower_better else (values > 0).sum())
        practical = float(values.mean()) <= -PRACTICAL_THRESHOLDS[task] if lower_better else float(values.mean()) >= PRACTICAL_THRESHOLDS[task]
        directional_ci = high < 0 if lower_better else low > 0
        tasks[task] = {
            "mean_candidate_minus_reference": float(values.mean()),
            "ci95": [float(low), float(high)],
            "improving_seeds": improving,
            "semantic_advantage": improving >= 2 and directional_ci and practical,
        }
    return {"reference": reference, "candidate": candidate, "tasks": tasks}


def validate_and_lock() -> Path:
    missing = [
        str(validation_dir(config, seed) / "result.json")
        for config in ALL_CONFIGS
        for seed in CONFIRMATION_SEEDS
        if not (validation_dir(config, seed) / "result.json").exists()
    ]
    if missing:
        raise RuntimeError(f"Cannot lock incomplete validation results: {missing}")
    bootstraps = {}
    decisions = {}
    for reference, candidate in COMPARISONS[:3]:
        key = f"{candidate}_minus_{reference}"
        bootstraps[key] = _bootstrap(reference, candidate, "validation")
        decisions[key] = _semantic_decision(reference, candidate, "validation", bootstraps[key])
    state = {
        "seeds": list(CONFIRMATION_SEEDS),
        "locked_from": "validation_only",
        "comparisons": decisions,
        "test_accessed": False,
    }
    output = lock_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2), encoding="utf-8")
    report_root = RECENCY_CONTROL_ROOT / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "validation_bootstrap.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    return output


def build_report() -> Path:
    if not lock_path().exists():
        raise RuntimeError("Validation lock is required before test reporting")
    output = RECENCY_CONTROL_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = [_metric_row(config, seed, split) for split in ("validation", "test") for config in ALL_CONFIGS for seed in CONFIRMATION_SEEDS]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    frame.groupby(["split", "config"]).agg(
        event_accuracy_mean=("event_accuracy", "mean"),
        event_accuracy_std=("event_accuracy", "std"),
        event_macro_f1_mean=("event_macro_f1", "mean"),
        event_macro_f1_std=("event_macro_f1", "std"),
        time_mae_mean=("time_mae_seconds", "mean"),
        time_mae_std=("time_mae_seconds", "std"),
        position_error_mean=("position_distance_mae_m", "mean"),
        position_error_std=("position_distance_mae_m", "std"),
    ).reset_index().to_csv(output / "summary.csv", index=False)
    bootstraps = {f"{candidate}_minus_{reference}": _bootstrap(reference, candidate, "test") for reference, candidate in COMPARISONS}
    (output / "test_bootstrap.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    validation_lock = json.loads(lock_path().read_text(encoding="utf-8"))
    final = {
        "question": "Does possession-aware selection outperform an equal-size contiguous recent window?",
        "validation_decisions": validation_lock["comparisons"],
        "test_only_confirms_locked_hypotheses": True,
        "efficiency_note": "Fusion cost is the sum of both encoded views; it is never reported as either constituent alone.",
    }
    (output / "report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return output
