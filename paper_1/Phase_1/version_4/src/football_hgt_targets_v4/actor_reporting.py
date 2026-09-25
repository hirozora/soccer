"""Locked validation selection and paired reporting for actor-scale probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .actor_study import ACTOR_EXPERIMENT_ROOT, ACTOR_TASKS, ACTOR_VIEWS, test_dir, validation_dir
from .constants import CONFIRMATION_SEEDS


def _read_result(root: Path) -> dict[str, Any]:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _metric_row(task: str, view: str, seed: int, split: str) -> dict[str, Any]:
    root = validation_dir(task, view, seed) if split == "validation" else test_dir(task, view, seed)
    metrics = _read_result(root)[split][task]
    row = {"task": task, "view": view, "seed": seed, "split": split}
    if task == "team":
        row.update(team_accuracy=metrics["accuracy"], team_macro_f1=metrics["macro_f1"])
    else:
        row.update(
            player_top1=metrics["top1_accuracy"],
            player_top3=metrics["top3_accuracy"],
            player_top5=metrics["top5_accuracy"],
            player_mrr=metrics["mrr"],
            player_coverage=metrics["coverage"],
            mean_candidate_count=metrics["mean_candidate_count"],
        )
    return row


def lock_actor_winners() -> dict[str, Any]:
    missing = [
        str(validation_dir(task, view, seed) / "result.json")
        for task in ACTOR_TASKS for view in ACTOR_VIEWS for seed in CONFIRMATION_SEEDS
        if not (validation_dir(task, view, seed) / "result.json").exists()
    ]
    if missing:
        raise RuntimeError(f"Incomplete actor validation matrix: {missing}")
    rows = [
        _metric_row(task, view, seed, "validation")
        for task in ACTOR_TASKS for view in ACTOR_VIEWS for seed in CONFIRMATION_SEEDS
    ]
    frame = pd.DataFrame(rows)
    output = ACTOR_EXPERIMENT_ROOT / "selection"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "validation_by_seed.csv", index=False)
    winners = {
        "team": str(frame[frame.task == "team"].groupby("view").team_macro_f1.mean().idxmax()),
        "player": str(frame[frame.task == "player"].groupby("view").player_top1.mean().idxmax()),
    }
    state = {
        "selection_split": "validation",
        "selection_metrics": {"team": "macro_f1", "player": "top1_accuracy"},
        "winners": winners,
        "candidate_contract": "same match-local pre-match roster for F80/P1/P2",
        "test_accessed": False,
    }
    (output / "winners.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _paired(left: pd.DataFrame, right: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.sort_values("sample_id").reset_index(drop=True)
    right = right.sort_values("sample_id").reset_index(drop=True)
    if left.sample_id.tolist() != right.sample_id.tolist():
        raise RuntimeError("Actor sample IDs are not paired")
    fields = ("team_true",) if task == "team" else ("player_true_raw", "player_mask", "candidate_count")
    for field in fields:
        if not np.array_equal(left[field].to_numpy(), right[field].to_numpy()):
            raise RuntimeError(f"Actor target/candidate field differs: {field}")
    return left, right


def _match_stats(frame: pd.DataFrame, task: str, matches: list[int]) -> np.ndarray:
    rows = []
    for match_id in matches:
        part = frame[frame.match_id == match_id]
        if task == "team":
            matrix = np.zeros((2, 2), dtype=np.float64)
            for true, pred in zip(part.team_true.astype(int), part.team_pred.astype(int)):
                matrix[true, pred] += 1
            rows.append(matrix.reshape(-1))
        else:
            active = part[part.player_mask.astype(bool)]
            rank = active.player_rank.to_numpy()
            rows.append(np.asarray([
                len(active), np.sum(rank <= 1), np.sum(rank <= 3), np.sum(rank <= 5), np.sum(1.0 / rank)
            ], dtype=np.float64))
    return np.stack(rows)


def _metrics_from_stats(values: np.ndarray, task: str) -> dict[str, float]:
    total = values.sum(axis=0)
    if task == "team":
        matrix = total.reshape(2, 2)
        accuracy = np.trace(matrix) / max(matrix.sum(), 1)
        f1 = []
        for label in range(2):
            tp = matrix[label, label]
            fp = matrix[:, label].sum() - tp
            fn = matrix[label, :].sum() - tp
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        return {"accuracy": float(accuracy), "macro_f1": float(np.mean(f1))}
    valid = max(total[0], 1)
    return {
        "top1_accuracy": float(total[1] / valid),
        "top3_accuracy": float(total[2] / valid),
        "top5_accuracy": float(total[3] / valid),
        "mrr": float(total[4] / valid),
    }


def _bootstrap(task: str, view: str, iterations: int = 10_000) -> dict[str, Any]:
    reference_stats, candidate_stats = [], []
    for seed in CONFIRMATION_SEEDS:
        left, right = _paired(
            pd.read_parquet(test_dir(task, "f80", seed) / "test_predictions.parquet"),
            pd.read_parquet(test_dir(task, view, seed) / "test_predictions.parquet"),
            task,
        )
        matches = sorted(left.match_id.astype(int).unique())
        reference_stats.append(_match_stats(left, task, matches))
        candidate_stats.append(_match_stats(right, task, matches))
    reference = np.stack(reference_stats)
    candidate = np.stack(candidate_stats)
    rng = np.random.default_rng(20260715)
    metric_names = ("accuracy", "macro_f1") if task == "team" else ("top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr")
    differences = {name: np.empty(iterations) for name in metric_names}
    seed_count, match_count = reference.shape[:2]
    for iteration in range(iterations):
        seed_indices = rng.integers(0, seed_count, size=seed_count)
        sampled_left, sampled_right = [], []
        for seed_index in seed_indices:
            match_indices = rng.integers(0, match_count, size=match_count)
            sampled_left.append(reference[seed_index, match_indices])
            sampled_right.append(candidate[seed_index, match_indices])
        left_metrics = _metrics_from_stats(np.concatenate(sampled_left), task)
        right_metrics = _metrics_from_stats(np.concatenate(sampled_right), task)
        for name in metric_names:
            differences[name][iteration] = right_metrics[name] - left_metrics[name]
    return {
        name: {
            "mean_difference": float(values.mean()),
            "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        }
        for name, values in differences.items()
    }


def build_actor_report() -> Path:
    selection = lock_actor_winners()
    missing = [
        str(test_dir(task, view, seed) / "result.json")
        for task in ACTOR_TASKS for view in ACTOR_VIEWS for seed in CONFIRMATION_SEEDS
        if not (test_dir(task, view, seed) / "result.json").exists()
    ]
    if missing:
        raise RuntimeError(f"Incomplete actor test matrix: {missing}")
    validation_rows = [
        _metric_row(task, view, seed, "validation")
        for task in ACTOR_TASKS for view in ACTOR_VIEWS for seed in CONFIRMATION_SEEDS
    ]
    test_rows = [
        _metric_row(task, view, seed, "test")
        for task in ACTOR_TASKS for view in ACTOR_VIEWS for seed in CONFIRMATION_SEEDS
    ]
    output = ACTOR_EXPERIMENT_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    validation = pd.DataFrame(validation_rows)
    test = pd.DataFrame(test_rows)
    validation.to_csv(output / "validation_by_seed.csv", index=False)
    test.to_csv(output / "test_by_seed.csv", index=False)
    summaries = []
    for task, part in test.groupby("task"):
        numeric = [
            column for column in part.columns
            if column not in {"task", "view", "seed", "split"}
            and bool(part[column].notna().any())
        ]
        for view, view_part in part.groupby("view"):
            row: dict[str, Any] = {"task": task, "view": view}
            for column in numeric:
                row[f"{column}_mean"] = float(view_part[column].mean())
                row[f"{column}_std"] = float(view_part[column].std(ddof=1))
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / "test_summary.csv", index=False)
    bootstrap = {
        task: {view: _bootstrap(task, view) for view in ("p1", "p2")}
        for task in ACTOR_TASKS
    }
    report = {
        "validation_winners": selection["winners"],
        "selection_metrics": selection["selection_metrics"],
        "bootstrap_vs_f80": bootstrap,
        "interpretation_contract": (
            "A validation winner is a scale preference; a stable advantage claim additionally requires a paired CI excluding zero."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output / "report.json"
