"""Locked reporting and decision rules for task-view fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .metrics import compute_metrics
from .multiview_study import (
    EXPERIMENT_MODES,
    MULTIVIEW_EXPERIMENT_ROOT,
    test_dir,
    validation_dir,
)
from .reporting import _match_statistics, _paired_hierarchical_bootstrap
from .subgraph_reporting import _add_source_context, _group_masks, _paired_frame
from .subgraph_study import (
    f80_validation_dir,
    read_result,
    test_dir as subgraph_test_dir,
)


TASK_METRICS = {
    "event": "event_macro_f1",
    "time": "time_mae_seconds",
    "position": "position_distance_mae_m",
}
PRACTICAL_THRESHOLDS = {
    "event": 0.005,
    "time": 0.01,
    "position": 0.25,
}
GROUP_THRESHOLDS = {"event": 0.01, "time": 0.05, "position": 0.50}
LOWER_IS_BETTER = {"time", "position"}
GUARDRAILS = {
    "event_accuracy": 0.01,
    "time_mae_seconds": 0.05,
    "position_distance_mae_m": 0.50,
}
REGISTERED_GROUPS = (
    "CONTROL",
    "CONTESTED",
    "restart",
    "switch",
    "boundary",
    "long_possession",
)


def _metric_row(mode: str, seed: int, result: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = result[split]
    if metrics is None:
        raise RuntimeError(f"Missing {split} metrics for {mode}/seed{seed}")
    return {
        "view": mode,
        "family": "full" if mode == "f80" else "task_view_fusion",
        "seed": seed,
        "split": split,
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
    }


def _result_root(mode: str, seed: int, split: str) -> Path:
    if mode == "f80":
        return f80_validation_dir(seed) if split == "validation" else subgraph_test_dir("f80", seed)
    return validation_dir(mode, seed) if split == "validation" else test_dir(mode, seed)


def _prediction(mode: str, seed: int, split: str) -> pd.DataFrame:
    return pd.read_parquet(
        _result_root(mode, seed, split) / f"{split}_predictions.parquet"
    )


def _bootstrap(reference_mode: str, candidate_mode: str, split: str) -> dict[str, Any]:
    reference_stats = []
    candidate_stats = []
    for seed in CONFIRMATION_SEEDS:
        reference, candidate = _paired_frame(
            _prediction(reference_mode, seed, split),
            _prediction(candidate_mode, seed, split),
        )
        matches = sorted(reference.match_id.astype(int).unique().tolist())
        reference_stats.append(_match_statistics(reference, matches))
        candidate_stats.append(_match_statistics(candidate, matches))
    return _paired_hierarchical_bootstrap(
        np.stack(reference_stats), np.stack(candidate_stats), iterations=10_000
    )


def _metric_rows(split: str) -> pd.DataFrame:
    rows = []
    for mode in ("f80", *EXPERIMENT_MODES):
        for seed in CONFIRMATION_SEEDS:
            result = read_result(_result_root(mode, seed, split))
            row = _metric_row(mode, seed, result, split)
            row.update(
                {
                    "parameters": result.get("parameters"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "peak_cuda_memory_bytes": result.get("peak_cuda_memory_bytes"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _state_group_rows() -> pd.DataFrame:
    rows = []
    for mode in ("f80", *EXPERIMENT_MODES):
        for seed in CONFIRMATION_SEEDS:
            frame = _prediction(mode, seed, "test")
            context = _add_source_context(frame, "p1")
            masks = _group_masks(context)
            for group in REGISTERED_GROUPS:
                mask = masks[group].to_numpy()
                metrics = compute_metrics(frame[mask])
                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "group": group,
                        "samples": int(mask.sum()),
                        "event_accuracy": metrics["event"]["accuracy"],
                        "event_macro_f1": metrics["event"]["macro_f1"],
                        "time_mae_seconds": metrics["time"]["mae_seconds"],
                        "position_distance_mae_m": metrics["position"]["distance_mae_m"],
                    }
                )
    return pd.DataFrame(rows)


def _seed_differences(
    reference_mode: str, candidate_mode: str, split: str
) -> pd.DataFrame:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        reference = _metric_row(
            reference_mode,
            seed,
            read_result(_result_root(reference_mode, seed, split)),
            split,
        )
        candidate = _metric_row(
            candidate_mode,
            seed,
            read_result(_result_root(candidate_mode, seed, split)),
            split,
        )
        rows.append(
            {
                "seed": seed,
                **{
                    metric: candidate[metric] - reference[metric]
                    for metric in (
                        "event_accuracy",
                        "event_macro_f1",
                        "time_mae_seconds",
                        "position_distance_mae_m",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def _task_effective(
    task: str, differences: pd.DataFrame, bootstrap: dict[str, Any]
) -> dict[str, Any]:
    metric = TASK_METRICS[task]
    values = differences[metric]
    mean = float(values.mean())
    low, high = bootstrap[metric]["ci95"]
    if task in LOWER_IS_BETTER:
        same_direction = int((values < 0).sum())
        directional_ci = high < 0
        practical = mean <= -PRACTICAL_THRESHOLDS[task]
    else:
        same_direction = int((values > 0).sum())
        directional_ci = low > 0
        practical = mean >= PRACTICAL_THRESHOLDS[task]
    return {
        "metric": metric,
        "mean_difference": mean,
        "ci95": [float(low), float(high)],
        "improving_seeds": same_direction,
        "direction_consistent": same_direction >= 2,
        "ci_excludes_zero": directional_ci,
        "practical_threshold_met": practical,
        "effective": same_direction >= 2 and directional_ci and practical,
    }


def _guardrails(differences: pd.DataFrame) -> dict[str, Any]:
    means = differences.mean(numeric_only=True).to_dict()
    checks = {
        "event_accuracy": means["event_accuracy"] >= -GUARDRAILS["event_accuracy"],
        "time_mae_seconds": means["time_mae_seconds"] <= GUARDRAILS["time_mae_seconds"],
        "position_distance_mae_m": means["position_distance_mae_m"]
        <= GUARDRAILS["position_distance_mae_m"],
    }
    return {"differences": means, "checks": checks, "passed": all(checks.values())}


def _fusion_decisions(bootstraps: dict[str, Any]) -> dict[str, Any]:
    specifications = {
        "sf_a": ("fixed_a", ("time", "position")),
        "sf_b": ("fixed_b", ("event", "time", "position")),
    }
    result = {}
    for mode, (reference, tasks) in specifications.items():
        differences = _seed_differences(reference, mode, "test")
        bootstrap = bootstraps[f"{mode}_minus_{reference}"]
        task_results = {
            task: _task_effective(task, differences, bootstrap) for task in tasks
        }
        guards = _guardrails(differences)
        result[mode] = {
            "reference": reference,
            "tasks": task_results,
            "guardrails": guards,
            "model_effective": any(value["effective"] for value in task_results.values())
            and guards["passed"],
        }
    return result


def _weight_report() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in ("sf_a", "sf_b"):
        tasks: dict[str, list[list[float]]] = {}
        contexts = []
        for seed in CONFIRMATION_SEEDS:
            result = read_result(validation_dir(mode, seed))
            for task, weights in result["fusion"]["weights"].items():
                tasks.setdefault(task, []).append(weights)
            best_epoch = int(result["best_epoch"])
            record = next(row for row in result["history"] if int(row["epoch"]) == best_epoch)
            contexts.append({"seed": seed, **record["fusion"]["context"]})
        task_output = {}
        for task, values in tasks.items():
            array = np.asarray(values, dtype=float)
            means = array.mean(axis=0)
            dominant = int(means.argmax())
            seed_dominant = array.argmax(axis=1)
            stable = int((seed_dominant == dominant).sum()) >= 2 and float(means[dominant]) > 0.60
            task_output[task] = {
                "per_seed": values,
                "mean": means.tolist(),
                "std": array.std(axis=0, ddof=1).tolist(),
                "dominant_view": ("p1", "p2")[dominant],
                "stable_view_preference": stable,
            }
        output[mode] = {"tasks": task_output, "context_diagnostics": contexts}
    return output


def _state_heterogeneity() -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    grouped_frames: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for seed in CONFIRMATION_SEEDS:
        p1, p2 = _paired_frame(
            pd.read_parquet(subgraph_test_dir("p1", seed) / "test_predictions.parquet"),
            pd.read_parquet(subgraph_test_dir("p2", seed) / "test_predictions.parquet"),
        )
        context = _add_source_context(p1, "p1")
        masks = _group_masks(context)
        for group in REGISTERED_GROUPS:
            mask = masks[group].to_numpy()
            left, right = p1[mask], p2[mask]
            grouped_frames[(seed, group)] = (left, right)
            p1_metrics = compute_metrics(left)
            p2_metrics = compute_metrics(right)
            seed_rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "samples": len(left),
                    "event": p2_metrics["event"]["macro_f1"] - p1_metrics["event"]["macro_f1"],
                    "time": p2_metrics["time"]["mae_seconds"] - p1_metrics["time"]["mae_seconds"],
                    "position": p2_metrics["position"]["distance_mae_m"]
                    - p1_metrics["position"]["distance_mae_m"],
                }
            )
    differences = pd.DataFrame(seed_rows)
    decisions: dict[str, Any] = {task: {"preferences": {}} for task in TASK_METRICS}
    for group in REGISTERED_GROUPS:
        values = differences[differences.group == group]
        minimum_samples = int(values.samples.min())
        reference_stats = []
        candidate_stats = []
        for seed in CONFIRMATION_SEEDS:
            p1, p2 = grouped_frames[(seed, group)]
            matches = sorted(p1.match_id.astype(int).unique().tolist())
            reference_stats.append(_match_statistics(p1, matches))
            candidate_stats.append(_match_statistics(p2, matches))
        bootstrap = _paired_hierarchical_bootstrap(
            np.stack(reference_stats), np.stack(candidate_stats), iterations=10_000
        )
        for task, metric in TASK_METRICS.items():
            series = values[task]
            mean = float(series.mean())
            low, high = bootstrap[metric]["ci95"]
            threshold = GROUP_THRESHOLDS[task]
            if task in LOWER_IS_BETTER:
                p2_preferred = int((series < 0).sum()) >= 2 and high < 0 and mean <= -threshold
                p1_preferred = int((series > 0).sum()) >= 2 and low > 0 and mean >= threshold
            else:
                p2_preferred = int((series > 0).sum()) >= 2 and low > 0 and mean >= threshold
                p1_preferred = int((series < 0).sum()) >= 2 and high < 0 and mean <= -threshold
            preference = "p2" if p2_preferred else "p1" if p1_preferred else "none"
            decisions[task]["preferences"][group] = {
                "samples_min": minimum_samples,
                "mean_p2_minus_p1": mean,
                "ci95": [float(low), float(high)],
                "preference": preference if minimum_samples >= 500 else "none",
            }
    for task, state in decisions.items():
        preferences = {
            value["preference"] for value in state["preferences"].values()
        }
        state["router_entry_b"] = {"p1", "p2"}.issubset(preferences)
    return {
        "minimum_group_samples": 500,
        "thresholds": GROUP_THRESHOLDS,
        "tasks": decisions,
        "seed_differences": seed_rows,
    }


def build_multiview_report() -> Path:
    output = MULTIVIEW_EXPERIMENT_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    validation = _metric_rows("validation")
    test = _metric_rows("test")
    validation.to_csv(output / "validation_by_seed.csv", index=False)
    test.to_csv(output / "test_by_seed.csv", index=False)
    test.groupby("view").agg(
        event_accuracy_mean=("event_accuracy", "mean"),
        event_accuracy_std=("event_accuracy", "std"),
        event_macro_f1_mean=("event_macro_f1", "mean"),
        event_macro_f1_std=("event_macro_f1", "std"),
        time_mae_mean=("time_mae_seconds", "mean"),
        time_mae_std=("time_mae_seconds", "std"),
        position_error_mean=("position_distance_mae_m", "mean"),
        position_error_std=("position_distance_mae_m", "std"),
    ).to_csv(output / "test_summary.csv")
    _state_group_rows().to_csv(output / "state_groups.csv", index=False)

    comparison_pairs = (
        ("f80", "fixed_a"),
        ("fixed_a", "fixed_b"),
        ("fixed_a", "sf_a"),
        ("fixed_b", "sf_b"),
        ("sf_a", "sf_b"),
    )
    bootstraps = {
        f"{candidate}_minus_{reference}": _bootstrap(reference, candidate, "test")
        for reference, candidate in comparison_pairs
    }
    (output / "paired_bootstrap.json").write_text(
        json.dumps({"iterations": 10_000, "comparisons": bootstraps}, indent=2),
        encoding="utf-8",
    )
    fusion = _fusion_decisions(bootstraps)
    weights = _weight_report()
    states = _state_heterogeneity()
    (output / "state_heterogeneity.json").write_text(
        json.dumps(states, indent=2), encoding="utf-8"
    )
    entry_a = any(value["model_effective"] for value in fusion.values())
    entry_b = any(
        value["router_entry_b"] for value in states["tasks"].values()
    )
    report = {
        "test_samples": int(
            len(_prediction("fixed_a", CONFIRMATION_SEEDS[0], "test"))
        ),
        "task_level_soft_fusion": fusion,
        "weight_interpretation": weights,
        "router_entry": {
            "entry_a_static_complementarity": entry_a,
            "entry_b_state_heterogeneity": entry_b,
            "proceed_to_sample_conditioned_fusion": entry_a or entry_b,
        },
        "test_was_not_used_for_configuration_selection": True,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return output
