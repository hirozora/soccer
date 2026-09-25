"""Locked test reporting for the five-task context-view comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS

from .constants import CONFIRMATION_SEEDS
from .five_task_study import FIVE_TASK_EXPERIMENT_ROOT, FIVE_TASK_MODES, test_dir, validation_dir


METRICS = (
    "event_accuracy", "event_macro_f1", "time_mae_seconds",
    "position_distance_mae_m", "team_accuracy", "team_macro_f1",
    "player_top1", "player_top3", "player_top5", "player_mrr",
)
LOWER_IS_BETTER = {"time_mae_seconds", "position_distance_mae_m"}
PRACTICAL = {
    "event_macro_f1": 0.005,
    "time_mae_seconds": 0.01,
    "position_distance_mae_m": 0.25,
    "team_macro_f1": 0.005,
    "player_top1": 0.01,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def _metric_row(mode: str, seed: int, split: str) -> dict[str, Any]:
    root = validation_dir(mode, seed) if split == "validation" else test_dir(mode, seed)
    result = _read(root)
    metrics = result[split]
    if metrics is None:
        raise RuntimeError(f"Missing {split} metrics: {root}")
    return {
        "mode": mode,
        "seed": seed,
        "split": split,
        "event_accuracy": metrics["event"]["accuracy"],
        "event_macro_f1": metrics["event"]["macro_f1"],
        "time_mae_seconds": metrics["time"]["mae_seconds"],
        "position_distance_mae_m": metrics["position"]["distance_mae_m"],
        "position_median_m": metrics["position"]["distance_median_m"],
        "team_accuracy": metrics["team"]["accuracy"],
        "team_macro_f1": metrics["team"]["macro_f1"],
        "player_top1": metrics["player"]["top1_accuracy"],
        "player_top3": metrics["player"]["top3_accuracy"],
        "player_top5": metrics["player"]["top5_accuracy"],
        "player_mrr": metrics["player"]["mrr"],
        "player_coverage": metrics["player"]["coverage"],
        "parameters": result["parameters"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
    }


def lock_five_task_configurations() -> Path:
    hashes: dict[int, str] = {}
    rows = []
    for seed in CONFIRMATION_SEEDS:
        for mode in FIVE_TASK_MODES:
            root = validation_dir(mode, seed)
            result = _read(root)
            if result.get("test_accessed") or result.get("test") is not None:
                raise RuntimeError(f"Validation accessed test data: {root}")
            value = result["initial_common_sha256"]
            if seed in hashes and hashes[seed] != value:
                raise RuntimeError(f"Five-task common initialization differs for seed {seed}")
            hashes[seed] = value
            rows.append({"mode": mode, "seed": seed, "checkpoint": str(root / "best.pt")})
    output = FIVE_TASK_EXPERIMENT_ROOT / "selection/locked.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "selection_split": "validation",
        "configurations": rows,
        "initial_common_sha256": hashes,
        "test_accessed": False,
    }, indent=2), encoding="utf-8")
    return output


def _prediction(mode: str, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(test_dir(mode, seed) / "test_predictions.parquet").sort_values("sample_id")
    return frame.reset_index(drop=True)


def _match_statistics(frame: pd.DataFrame, matches: list[int]) -> np.ndarray:
    # Event 10x10, Team 2x2, time sum/count, position sum/count,
    # Player valid/top1/top3/top5/reciprocal-rank.
    result = np.zeros((len(matches), 113), dtype=np.float64)
    grouped = {int(key): value for key, value in frame.groupby("match_id")}
    for row, match_id in enumerate(matches):
        group = grouped[match_id]
        event_flat = group.event_true.to_numpy(np.int64) * 10 + group.event_pred.to_numpy(np.int64)
        result[row, :100] = np.bincount(event_flat, minlength=100)
        team_flat = group.team_true.to_numpy(np.int64) * 2 + group.team_pred.to_numpy(np.int64)
        result[row, 100:104] = np.bincount(team_flat, minlength=4)
        active_time = group[group.time_mask]
        result[row, 104] = np.abs(active_time.time_pred - active_time.time_true).sum()
        result[row, 105] = len(active_time)
        active_position = group[group.position_mask]
        dx = (active_position.position_pred_x - active_position.position_true_x) * PITCH_LENGTH_METERS
        dy = (active_position.position_pred_y - active_position.position_true_y) * PITCH_WIDTH_METERS
        result[row, 106] = np.sqrt(dx**2 + dy**2).sum()
        result[row, 107] = len(active_position)
        active_player = group[group.player_mask]
        rank = active_player.player_rank.to_numpy(np.float64)
        result[row, 108] = len(rank)
        result[row, 109] = np.sum(rank <= 1)
        result[row, 110] = np.sum(rank <= 3)
        result[row, 111] = np.sum(rank <= 5)
        result[row, 112] = np.sum(1.0 / rank)
    return result


def _confusion_metrics(values: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    confusion = values.reshape(*values.shape[:-1], classes, classes)
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    total = confusion.sum(axis=(-2, -1))
    accuracy = np.divide(tp.sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)
    denominator = 2 * tp + confusion.sum(axis=-2) - tp + confusion.sum(axis=-1) - tp
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return accuracy, f1.mean(axis=-1)


def _to_metrics(statistics: np.ndarray) -> dict[str, np.ndarray]:
    event_accuracy, event_macro = _confusion_metrics(statistics[..., :100], 10)
    team_accuracy, team_macro = _confusion_metrics(statistics[..., 100:104], 2)
    return {
        "event_accuracy": event_accuracy,
        "event_macro_f1": event_macro,
        "time_mae_seconds": statistics[..., 104] / statistics[..., 105],
        "position_distance_mae_m": statistics[..., 106] / statistics[..., 107],
        "team_accuracy": team_accuracy,
        "team_macro_f1": team_macro,
        "player_top1": statistics[..., 109] / statistics[..., 108],
        "player_top3": statistics[..., 110] / statistics[..., 108],
        "player_top5": statistics[..., 111] / statistics[..., 108],
        "player_mrr": statistics[..., 112] / statistics[..., 108],
    }


def _bootstrap(reference: np.ndarray, candidate: np.ndarray, iterations: int = 10_000) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError("Five-task bootstrap arrays differ")
    rng = np.random.default_rng(20260715)
    seeds, matches, width = reference.shape
    differences = {name: np.empty(iterations) for name in METRICS}
    for start in range(0, iterations, 500):
        stop = min(iterations, start + 500)
        size = stop - start
        seed_draws = rng.integers(0, seeds, size=(size, seeds))
        left = np.zeros((size, width))
        right = np.zeros((size, width))
        for draw in range(seeds):
            weights = rng.multinomial(matches, np.full(matches, 1 / matches), size=size)
            selected = seed_draws[:, draw]
            left += np.einsum("bm,bmd->bd", weights, reference[selected])
            right += np.einsum("bm,bmd->bd", weights, candidate[selected])
        left_metrics, right_metrics = _to_metrics(left), _to_metrics(right)
        for name in METRICS:
            differences[name][start:stop] = right_metrics[name] - left_metrics[name]
    observed_left = _to_metrics(reference.sum(axis=(0, 1)))
    observed_right = _to_metrics(candidate.sum(axis=(0, 1)))
    return {name: {
        "reference": float(observed_left[name]),
        "candidate": float(observed_right[name]),
        "difference": float(observed_right[name] - observed_left[name]),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    } for name, values in differences.items()}


def _decision(metric: str, seed_differences: np.ndarray, bootstrap: dict[str, Any]) -> dict[str, Any]:
    lower = metric in LOWER_IS_BETTER
    mean = float(seed_differences.mean())
    low, high = bootstrap[metric]["ci95"]
    improving = int(np.sum(seed_differences < 0 if lower else seed_differences > 0))
    practical = mean <= -PRACTICAL[metric] if lower else mean >= PRACTICAL[metric]
    directional_ci = high < 0 if lower else low > 0
    return {
        "mean_difference": mean,
        "ci95": [low, high],
        "improving_seeds": improving,
        "effective": improving >= 2 and directional_ci and practical,
    }


def build_five_task_report() -> Path:
    locked = FIVE_TASK_EXPERIMENT_ROOT / "selection/locked.json"
    if not locked.exists():
        raise RuntimeError("Five-task configurations are not locked")
    output = FIVE_TASK_EXPERIMENT_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = [_metric_row(mode, seed, split) for split in ("validation", "test") for mode in FIVE_TASK_MODES for seed in CONFIRMATION_SEEDS]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics_by_seed.csv", index=False)
    metrics[metrics.split == "test"].groupby("mode").agg(
        **{f"{name}_mean": (name, "mean") for name in METRICS},
        **{f"{name}_std": (name, "std") for name in METRICS},
    ).to_csv(output / "test_summary.csv")

    predictions: dict[str, list[pd.DataFrame]] = {mode: [] for mode in FIVE_TASK_MODES}
    statistics: dict[str, list[np.ndarray]] = {mode: [] for mode in FIVE_TASK_MODES}
    expected_ids = None
    matches = None
    for seed in CONFIRMATION_SEEDS:
        for mode in FIVE_TASK_MODES:
            frame = _prediction(mode, seed)
            ids = frame.sample_id.tolist()
            if expected_ids is None:
                expected_ids = ids
                matches = sorted(frame.match_id.astype(int).unique())
            elif ids != expected_ids:
                raise RuntimeError("Five-task test sample IDs are not paired")
            predictions[mode].append(frame)
            statistics[mode].append(_match_statistics(frame, matches))
    arrays = {name: np.stack(value) for name, value in statistics.items()}
    pairs = (("five_f80", "five_hard"), ("five_hard", "five_soft"), ("five_f80", "five_soft"))
    bootstraps = {f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right]) for left, right in pairs}
    (output / "paired_bootstrap.json").write_text(json.dumps({"iterations": 10_000, "comparisons": bootstraps}, indent=2), encoding="utf-8")

    test_rows = metrics[metrics.split == "test"].set_index(["mode", "seed"])
    decisions: dict[str, Any] = {}
    for left, right in pairs[:2]:
        key = f"{right}_minus_{left}"
        differences = test_rows.loc[right][list(METRICS)] - test_rows.loc[left][list(METRICS)]
        task_metrics = {
            "event": "event_macro_f1", "time": "time_mae_seconds",
            "position": "position_distance_mae_m", "team": "team_macro_f1",
            "player": "player_top1",
        }
        task_decisions = {task: _decision(metric, differences[metric].to_numpy(), bootstraps[key]) for task, metric in task_metrics.items()}
        guards = {
            "event_accuracy": float(differences.event_accuracy.mean()) >= -0.01,
            "time": float(differences.time_mae_seconds.mean()) <= 0.05,
            "position": float(differences.position_distance_mae_m.mean()) <= 0.5,
            "team_accuracy": float(differences.team_accuracy.mean()) >= -0.01,
            "player_top1": float(differences.player_top1.mean()) >= -0.01,
        }
        decisions[key] = {"tasks": task_decisions, "guardrails": guards, "guardrails_passed": all(guards.values())}

    soft_results = [_read(validation_dir("five_soft", seed)) for seed in CONFIRMATION_SEEDS]
    fusion = {
        task: {
            "per_seed": [result["fusion"]["weights"][task] for result in soft_results],
            "mean": np.mean([result["fusion"]["weights"][task] for result in soft_results], axis=0).tolist(),
        }
        for task in ("event", "time", "position")
    }
    external = []
    for name, path in (
        ("subgraph_scale", FIVE_TASK_EXPERIMENT_ROOT.parent / "subgraph_scale_v1/report/test_summary.csv"),
        ("actor_scale", FIVE_TASK_EXPERIMENT_ROOT.parent / "actor_scale_v1/report/test_summary.csv"),
    ):
        external.append({"study": name, "path": str(path), "exists": path.exists()})
    report = {
        "test_samples": len(expected_ids or []),
        "test_matches": len(matches or []),
        "comparisons": decisions,
        "soft_fusion_weights": fusion,
        "external_scale_evidence": external,
        "interpretation_contract": {
            "team_player_use_f80": True,
            "team_player_changes_are_shared_backbone_effects": True,
            "soft_weights_are_not_effectiveness_criteria": True,
        },
        "test_was_not_used_for_configuration_selection": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
