"""Final paired comparison and multitask-interference report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def _metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
    }


def _match_statistics(frame: pd.DataFrame, match_ids: list[int]) -> np.ndarray:
    # 100 confusion cells + time absolute-error sum/count + position distance sum/count.
    result = np.zeros((len(match_ids), 104), dtype=np.float64)
    grouped = {int(match_id): group for match_id, group in frame.groupby("match_id")}
    for index, match_id in enumerate(match_ids):
        group = grouped[match_id]
        flat = group.event_true.to_numpy(np.int64) * 10 + group.event_pred.to_numpy(np.int64)
        result[index, :100] = np.bincount(flat, minlength=100)
        time = group[group.time_mask]
        result[index, 100] = np.abs(time.time_pred - time.time_true).sum()
        result[index, 101] = len(time)
        position = group[group.position_mask]
        dx = (position.position_pred_x - position.position_true_x) * PITCH_LENGTH_METERS
        dy = (position.position_pred_y - position.position_true_y) * PITCH_WIDTH_METERS
        result[index, 102] = np.sqrt(dx**2 + dy**2).sum()
        result[index, 103] = len(position)
    return result


def _statistics_to_metrics(statistics: np.ndarray) -> dict[str, np.ndarray]:
    confusion = statistics[..., :100].reshape(*statistics.shape[:-1], 10, 10)
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    false_positive = confusion.sum(axis=-2) - true_positive
    false_negative = confusion.sum(axis=-1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = confusion.sum(axis=(-2, -1))
    return {
        "event_accuracy": true_positive.sum(axis=-1) / total,
        "event_macro_f1": f1.mean(axis=-1),
        "time_mae_seconds": statistics[..., 100] / statistics[..., 101],
        "position_distance_mae_m": statistics[..., 102] / statistics[..., 103],
    }


def _paired_hierarchical_bootstrap(
    original: np.ndarray,
    optimized: np.ndarray,
    iterations: int = 10_000,
    seed: int = 20260715,
) -> dict[str, Any]:
    if original.shape != optimized.shape:
        raise ValueError("Original and optimized sufficient statistics differ in shape")
    num_seeds, num_matches, width = original.shape
    rng = np.random.default_rng(seed)
    differences = {
        name: np.empty(iterations, dtype=np.float64)
        for name in _statistics_to_metrics(original.sum(axis=(0, 1)))
    }
    chunk_size = 500
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        size = stop - start
        seed_draws = rng.integers(0, num_seeds, size=(size, num_seeds))
        original_sum = np.zeros((size, width), dtype=np.float64)
        optimized_sum = np.zeros((size, width), dtype=np.float64)
        for draw in range(num_seeds):
            weights = rng.multinomial(
                num_matches,
                np.full(num_matches, 1.0 / num_matches),
                size=size,
            )
            selected = seed_draws[:, draw]
            original_sum += np.einsum("bm,bmd->bd", weights, original[selected])
            optimized_sum += np.einsum("bm,bmd->bd", weights, optimized[selected])
        original_metrics = _statistics_to_metrics(original_sum)
        optimized_metrics = _statistics_to_metrics(optimized_sum)
        for name in differences:
            differences[name][start:stop] = optimized_metrics[name] - original_metrics[name]

    observed_original = _statistics_to_metrics(original.sum(axis=(0, 1)))
    observed_optimized = _statistics_to_metrics(optimized.sum(axis=(0, 1)))
    return {
        name: {
            "original": float(observed_original[name]),
            "optimized": float(observed_optimized[name]),
            "difference_optimized_minus_original": float(
                observed_optimized[name] - observed_original[name]
            ),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for name, values in differences.items()
    }


def build_final_report(output_dir: Path | None = None) -> Path:
    root = EXPERIMENT_ROOT / "final"
    output = Path(output_dir) if output_dir is not None else root / "report"
    output.mkdir(parents=True, exist_ok=True)
    winners_state = json.loads(
        (EXPERIMENT_ROOT / "confirmation/winners.json").read_text(encoding="utf-8")
    )
    winners = winners_state["winners"]

    result_rows: list[dict[str, Any]] = []
    statistics: dict[str, list[np.ndarray]] = {
        "joint_original": [],
        "joint_optimized": [],
    }
    expected_sample_ids: list[str] | None = None
    expected_matches: list[int] | None = None
    for seed in CONFIRMATION_SEEDS:
        seed_results: dict[str, dict[str, Any]] = {}
        for variant in ("joint_original", "joint_optimized"):
            path = root / variant / f"seed{seed}"
            result = _read_result(path)
            seed_results[variant] = result
            if not result.get("test_accessed") or result.get("test") is None:
                raise RuntimeError(f"Final result did not evaluate test: {path}")
            for split in ("validation", "test"):
                result_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "split": split,
                        **_metric_values(result[split]),
                    }
                )
            frame = pd.read_parquet(path / "test_predictions.parquet").sort_values(
                "sample_id"
            )
            sample_ids = frame.sample_id.tolist()
            if expected_sample_ids is None:
                expected_sample_ids = sample_ids
                expected_matches = sorted(frame.match_id.astype(int).unique().tolist())
            elif sample_ids != expected_sample_ids:
                raise RuntimeError("Final prediction sample IDs are not paired")
            statistics[variant].append(_match_statistics(frame, expected_matches))
        if (
            seed_results["joint_original"]["initial_backbone_sha256"]
            != seed_results["joint_optimized"]["initial_backbone_sha256"]
        ):
            raise RuntimeError(f"J0/J1 shared HGT initialization differs for seed {seed}")

    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(output / "joint_comparison_by_seed.csv", index=False)
    aggregate = (
        comparison.groupby(["variant", "split"])
        .agg(
            event_accuracy_mean=("event_accuracy", "mean"),
            event_accuracy_std=("event_accuracy", "std"),
            event_macro_f1_mean=("event_macro_f1", "mean"),
            event_macro_f1_std=("event_macro_f1", "std"),
            time_mae_mean=("time_mae_seconds", "mean"),
            time_mae_std=("time_mae_seconds", "std"),
            position_distance_mean=("position_distance_mae_m", "mean"),
            position_distance_std=("position_distance_mae_m", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(output / "joint_comparison_summary.csv", index=False)

    validation = comparison[comparison.split == "validation"].set_index(
        ["variant", "seed"]
    )
    original = validation.loc["joint_original"].mean(numeric_only=True)
    optimized = validation.loc["joint_optimized"].mean(numeric_only=True)
    deltas = optimized - original
    acceptance = {
        "event_macro_f1_improved": bool(deltas.event_macro_f1 > 0),
        "event_accuracy_within_1pp": bool(deltas.event_accuracy >= -0.01),
        "time_mae_within_0.05s": bool(deltas.time_mae_seconds <= 0.05),
        "position_within_0.5m": bool(deltas.position_distance_mae_m <= 0.5),
    }
    acceptance["adopt_optimized"] = all(acceptance.values())
    adoption = {
        "selection_split": "validation",
        "winners": winners,
        "validation_difference_optimized_minus_original": {
            name: float(value) for name, value in deltas.items()
        },
        "criteria": acceptance,
    }
    (output / "adoption.json").write_text(json.dumps(adoption, indent=2), encoding="utf-8")

    bootstrap = _paired_hierarchical_bootstrap(
        np.stack(statistics["joint_original"]),
        np.stack(statistics["joint_optimized"]),
    )
    (output / "paired_bootstrap_test.json").write_text(
        json.dumps({"iterations": 10_000, "metrics": bootstrap}, indent=2),
        encoding="utf-8",
    )

    gap_rows: list[dict[str, Any]] = []
    optimized_test = comparison[
        (comparison.variant == "joint_optimized") & (comparison.split == "test")
    ].set_index("seed")
    metric_name = {
        "event": "event_macro_f1",
        "time": "time_mae_seconds",
        "position": "position_distance_mae_m",
    }
    for task, method in winners.items():
        for seed in CONFIRMATION_SEEDS:
            result = _read_result(
                root / "single_task_test" / task / method / f"seed{seed}"
            )
            test = result["test"]
            if task == "event":
                single_value = test["event"]["macro_f1"]
            elif task == "time":
                single_value = test["time"]["mae_seconds"]
            else:
                single_value = test["position"]["distance_mae_m"]
            joint_value = float(optimized_test.loc[seed, metric_name[task]])
            gap_rows.append(
                {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "single_task": single_value,
                    "joint_optimized": joint_value,
                    "joint_minus_single": joint_value - single_value,
                }
            )
    pd.DataFrame(gap_rows).to_csv(output / "single_vs_joint.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(
            {
                "winners": winners,
                "adopt_optimized": acceptance["adopt_optimized"],
                "test_sample_count": len(expected_sample_ids or []),
                "test_match_count": len(expected_matches or []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
