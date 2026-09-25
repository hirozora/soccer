"""Validation lock and test reporting for layer-wise partial sharing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .fixed_budget_study import test_dir as reference_test_dir
from .fixed_budget_study import training_dir as reference_training_dir
from .partial_sharing_study import (
    PARTIAL_SHARING_ROOT,
    REFERENCE_CONFIGURATIONS,
    test_dir,
    training_dir,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def lock_guarded_checkpoints() -> Path:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        root = training_dir(seed)
        result = _read(root)
        checkpoint = root / "best_guarded_core.pt"
        predictions = root / "validation_predictions_guarded_core.parquet"
        if not checkpoint.exists() or not predictions.exists():
            raise RuntimeError(f"Missing guarded checkpoint for seed {seed}")
        if result.get("test_accessed") or result.get("test") is not None:
            raise RuntimeError(f"Validation accessed test data: {root}")
        reference = _read(reference_training_dir("five_f80", seed))
        if result["initial_common_sha256"] != reference["initial_common_sha256"]:
            raise RuntimeError(f"Shared initialization differs for seed {seed}")
        metrics = result["validation_guarded_core"]
        guards = result["guard_reference"]
        team_gap = float(metrics["team"]["accuracy"]) - float(guards["team_accuracy"])
        player_gap = float(metrics["player"]["top1_accuracy"]) - float(guards["player_top1"])
        if team_gap < -0.01 - 1e-12 or player_gap < -0.01 - 1e-12:
            raise RuntimeError(f"Stored guarded checkpoint violates constraints for seed {seed}")
        rows.append({
            "seed": seed,
            "checkpoint": str(checkpoint),
            "epoch": int(torch_checkpoint_epoch(checkpoint)),
            "team_gap_vs_five_f80": team_gap,
            "player_top1_gap_vs_five_f80": player_gap,
        })
    output = PARTIAL_SHARING_ROOT / "selection/locked.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "selection_split": "validation",
        "selection_metric": "guarded_core",
        "configurations": rows,
        "test_accessed": False,
    }, indent=2), encoding="utf-8")
    return output


def torch_checkpoint_epoch(path: Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def _metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
        "position_median_m": float(metrics["position"]["distance_median_m"]),
        "team_accuracy": float(metrics["team"]["accuracy"]),
        "team_macro_f1": float(metrics["team"]["macro_f1"]),
        "player_top1": float(metrics["player"]["top1_accuracy"]),
        "player_top3": float(metrics["player"]["top3_accuracy"]),
        "player_top5": float(metrics["player"]["top5_accuracy"]),
        "player_mrr": float(metrics["player"]["mrr"]),
        "player_coverage": float(metrics["player"]["coverage"]),
    }


def _prediction(configuration: str, seed: int) -> pd.DataFrame:
    path = (
        test_dir(seed) / "test_predictions.parquet"
        if configuration == "partial_l2"
        else reference_test_dir(configuration, seed) / "test_predictions.parquet"
    )
    return pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)


def _paired_bootstrap(reference: str) -> dict[str, Any]:
    left_stats, right_stats = [], []
    expected_ids = None
    for seed in CONFIRMATION_SEEDS:
        left = _prediction(reference, seed)
        right = _prediction("partial_l2", seed)
        if left.sample_id.tolist() != right.sample_id.tolist():
            raise RuntimeError(f"Test samples differ: {reference}/partial_l2")
        if expected_ids is None:
            expected_ids = left.sample_id.tolist()
        elif expected_ids != left.sample_id.tolist():
            raise RuntimeError("Seeds use different test samples")
        matches = sorted(left.match_id.astype(int).unique().tolist())
        left_stats.append(_match_statistics(left, matches))
        right_stats.append(_match_statistics(right, matches))
    return _bootstrap(np.stack(left_stats), np.stack(right_stats), iterations=10_000)


def build_partial_sharing_report() -> Path:
    lock = PARTIAL_SHARING_ROOT / "selection/locked.json"
    if not lock.exists():
        raise RuntimeError("Guarded checkpoints must be locked before reporting")
    output = PARTIAL_SHARING_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    configurations = (*REFERENCE_CONFIGURATIONS, "partial_l2")
    for configuration in configurations:
        for seed in CONFIRMATION_SEEDS:
            result = _read(
                test_dir(seed)
                if configuration == "partial_l2"
                else reference_test_dir(configuration, seed)
            )
            row = {
                "configuration": configuration,
                "seed": seed,
                "best_epoch": int(result["best_epoch"]),
                **_metrics(result["test"]),
            }
            if configuration == "partial_l2":
                training = _read(training_dir(seed))
                row.update({
                    "parameters": int(training["parameters"]),
                    "training_seconds": float(training["elapsed_seconds"]),
                    "peak_cuda_memory_bytes": int(training["peak_cuda_memory_bytes"]),
                })
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    numeric = [name for name in frame.columns if name not in {"configuration", "seed"}]
    frame.groupby("configuration")[numeric].agg(["mean", "std"]).to_csv(
        output / "test_summary.csv"
    )
    comparisons = {
        f"partial_l2_minus_{reference}": _paired_bootstrap(reference)
        for reference in REFERENCE_CONFIGURATIONS
    }
    (output / "paired_bootstrap.json").write_text(
        json.dumps({"iterations": 10_000, "comparisons": comparisons}, indent=2),
        encoding="utf-8",
    )
    means = frame.groupby("configuration").mean(numeric_only=True)
    partial = means.loc["partial_l2"]
    three = means.loc["three_f80"]
    five = means.loc["five_f80"]
    checks = {
        "event": float(partial.event_macro_f1 - three.event_macro_f1) > -0.02,
        "time": float(partial.time_mae_seconds - three.time_mae_seconds) < 0.05,
        "position": float(partial.position_distance_mae_m - three.position_distance_mae_m) < 0.50,
        "team": float(partial.team_accuracy - five.team_accuracy) >= -0.01,
        "player": float(partial.player_top1 - five.player_top1) >= -0.01,
    }
    event_denominator = float(three.event_macro_f1 - five.event_macro_f1)
    position_denominator = float(five.position_distance_mae_m - three.position_distance_mae_m)
    report = {
        "checks": checks,
        "accepted": all(checks.values()),
        "event_recovery_fraction": (
            float(partial.event_macro_f1 - five.event_macro_f1) / event_denominator
            if abs(event_denominator) > 1e-12 else None
        ),
        "position_recovery_fraction": (
            float(five.position_distance_mae_m - partial.position_distance_mae_m)
            / position_denominator
            if abs(position_denominator) > 1e-12 else None
        ),
        "conclusion": (
            "shared low-level structure with a private Player high-level representation is supported"
            if all(checks.values())
            else "Partial-L2 did not establish a stable five-task baseline; test a fully private two-layer Player HGT"
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
