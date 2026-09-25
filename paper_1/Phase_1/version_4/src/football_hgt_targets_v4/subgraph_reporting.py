"""Validation selection and locked test reporting for subgraph scales."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .constants import CONFIRMATION_SEEDS, POSSESSION_GRAPH_ROOT, RAW_EVENT_NAMES
from .metrics import compute_metrics
from .reporting import (
    _match_statistics,
    _paired_hierarchical_bootstrap,
)
from .subgraph_study import (
    SUBGRAPH_EXPERIMENT_ROOT,
    completed_views,
    f80_validation_dir,
    metric_row,
    read_result,
    test_dir,
    validation_dir,
)
from .subgraph_views import ROUND_A_VIEWS, ROUND_B_BY_FAMILY, resolve_view_spec, select_event_indices


PRACTICAL_THRESHOLDS = {
    "event_macro_f1": 0.005,
    "time_mae_seconds": -0.01,
    "position_distance_mae_m": -0.25,
}
GROUP_THRESHOLDS = {
    "event_macro_f1": 0.01,
    "time_mae_seconds": -0.05,
    "position_distance_mae_m": -0.50,
}
LOWER_IS_BETTER = {"time_mae_seconds", "position_distance_mae_m"}


def _prediction_path(view: str, seed: int, split: str = "validation") -> Path:
    root = f80_validation_dir(seed) if view == "f80" else validation_dir(view, seed)
    return root / f"{split}_predictions.parquet"


def _result_path(view: str, seed: int) -> Path:
    return f80_validation_dir(seed) if view == "f80" else validation_dir(view, seed)


def _paired_frame(reference: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = reference.sort_values("sample_id").reset_index(drop=True)
    right = candidate.sort_values("sample_id").reset_index(drop=True)
    if left.sample_id.tolist() != right.sample_id.tolist():
        raise RuntimeError("Subgraph prediction sample IDs are not paired")
    for field in ("event_true", "time_true", "time_mask", "position_true_x", "position_true_y", "position_mask"):
        if field in left and not np.allclose(left[field], right[field], equal_nan=True):
            raise RuntimeError(f"Subgraph prediction targets differ: {field}")
    return left, right


def _bootstrap_for_view(view: str) -> dict[str, Any]:
    reference_stats = []
    candidate_stats = []
    for seed in CONFIRMATION_SEEDS:
        reference, candidate = _paired_frame(
            pd.read_parquet(_prediction_path("f80", seed)),
            pd.read_parquet(_prediction_path(view, seed)),
        )
        matches = sorted(reference.match_id.astype(int).unique().tolist())
        reference_stats.append(_match_statistics(reference, matches))
        candidate_stats.append(_match_statistics(candidate, matches))
    return _paired_hierarchical_bootstrap(
        np.stack(reference_stats), np.stack(candidate_stats), iterations=10_000
    )


def _metric_means(view: str) -> dict[str, float]:
    rows = [
        metric_row(view, seed, read_result(_result_path(view, seed)), "validation")
        for seed in CONFIRMATION_SEEDS
    ]
    frame = pd.DataFrame(rows)
    return {
        key: float(frame[key].mean())
        for key in ("event_accuracy", "event_macro_f1", "time_mae_seconds", "position_distance_mae_m")
    }


def _seed_differences(view: str) -> pd.DataFrame:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        reference = metric_row("f80", seed, read_result(f80_validation_dir(seed)), "validation")
        candidate = metric_row(view, seed, read_result(validation_dir(view, seed)), "validation")
        rows.append(
            {
                "seed": seed,
                **{
                    name: candidate[name] - reference[name]
                    for name in ("event_accuracy", "event_macro_f1", "time_mae_seconds", "position_distance_mae_m")
                },
            }
        )
    return pd.DataFrame(rows)


def _passes_trigger(metric: str, differences: pd.Series, bootstrap: dict[str, Any], threshold: float) -> bool:
    mean = float(differences.mean())
    same_direction = int((differences < 0).sum()) if metric in LOWER_IS_BETTER else int((differences > 0).sum())
    low, high = bootstrap[metric]["ci95"]
    directional_ci = high < 0 if metric in LOWER_IS_BETTER else low > 0
    practical = mean <= threshold if metric in LOWER_IS_BETTER else mean >= threshold
    return same_direction >= 2 and directional_ci and practical


def _graph_paths() -> dict[int, Path]:
    index = pd.read_csv(POSSESSION_GRAPH_ROOT / "metadata/match_index.csv")
    return {
        int(row.match_id): POSSESSION_GRAPH_ROOT / row.graph_path
        for row in index.itertuples()
    }


def _add_source_context(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    result = frame.copy()
    result["control_state"] = -1
    result["event_role"] = -1
    result["switch_confirmed"] = False
    result["possession_event_count"] = 0.0
    result["transition_marker"] = "not_transition_view"
    paths = _graph_paths()
    view_spec = resolve_view_spec(view)
    marker_view = view
    if view_spec.family == "random" and view_spec.reference_view is not None:
        reference = resolve_view_spec(view_spec.reference_view)
        marker_view = reference.name
        is_transition = reference.family == "transition"
    else:
        is_transition = view_spec.family == "transition"
    for match_id, row_indices in result.groupby("match_id").groups.items():
        graph = torch.load(paths[int(match_id)], map_location="cpu", weights_only=True)
        event = graph["node_stores"]["event"]
        rows = np.asarray(list(row_indices), dtype=np.int64)
        anchors = result.loc[rows, "current_event_index"].to_numpy(np.int64)
        result.loc[rows, "control_state"] = event["control_state_after_index"][anchors].numpy()
        result.loc[rows, "event_role"] = event["event_role_index"][anchors].numpy()
        result.loc[rows, "switch_confirmed"] = event["switch_confirmed"][anchors].numpy()
        result.loc[rows, "possession_event_count"] = event["possession_event_count_so_far"][anchors].numpy()
        if is_transition:
            result.loc[rows, "transition_marker"] = [
                select_event_indices(graph, int(anchor), marker_view).marker_type
                for anchor in anchors
            ]
    return result


def _group_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "CONTROL": frame.control_state == 2,
        "CONTESTED": frame.control_state == 3,
        "restart": frame.event_role == 2,
        "switch": frame.switch_confirmed.astype(bool),
        "boundary": frame.event_role == 4,
        "long_possession": frame.possession_event_count >= 11,
        "Shot": frame.event_true == RAW_EVENT_NAMES.index("Shot"),
        "Foul": frame.event_true == RAW_EVENT_NAMES.index("Foul"),
        "Interruption": frame.event_true == RAW_EVENT_NAMES.index("Interruption"),
        "TR-confirmed": frame.transition_marker == "confirmed_transition",
        "TR-boundary": frame.transition_marker == "boundary_context",
        "TR-fallback": frame.transition_marker == "fallback",
    }


def _group_seed_rows(view: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        reference, candidate = _paired_frame(
            pd.read_parquet(_prediction_path("f80", seed)),
            pd.read_parquet(_prediction_path(view, seed)),
        )
        context = _add_source_context(candidate, view)
        for group, mask in _group_masks(context).items():
            if not bool(mask.any()):
                continue
            ref_metrics = compute_metrics(reference[mask.to_numpy()])
            candidate_metrics = compute_metrics(candidate[mask.to_numpy()])
            rows.append(
                {
                    "view": view,
                    "seed": seed,
                    "group": group,
                    "samples": int(mask.sum()),
                    "event_accuracy_difference": candidate_metrics["event"]["accuracy"] - ref_metrics["event"]["accuracy"],
                    "event_macro_f1_difference": candidate_metrics["event"]["macro_f1"] - ref_metrics["event"]["macro_f1"],
                    "time_mae_seconds_difference": candidate_metrics["time"]["mae_seconds"] - ref_metrics["time"]["mae_seconds"],
                    "position_distance_mae_m_difference": candidate_metrics["position"]["distance_mae_m"] - ref_metrics["position"]["distance_mae_m"],
                }
            )
    return pd.DataFrame(rows)


def _group_bootstrap(view: str, group: str) -> dict[str, Any]:
    reference_stats = []
    candidate_stats = []
    for seed in CONFIRMATION_SEEDS:
        reference, candidate = _paired_frame(
            pd.read_parquet(_prediction_path("f80", seed)),
            pd.read_parquet(_prediction_path(view, seed)),
        )
        context = _add_source_context(candidate, view)
        mask = _group_masks(context)[group].to_numpy()
        reference = reference[mask]
        candidate = candidate[mask]
        matches = sorted(set(reference.match_id.astype(int)) | set(candidate.match_id.astype(int)))
        # The registered groups are represented across all validation matches.
        if set(matches) != set(reference.match_id.astype(int)) or set(matches) != set(candidate.match_id.astype(int)):
            raise RuntimeError(f"Unpaired grouped predictions for {view}/{group}")
        reference_stats.append(_match_statistics(reference, matches))
        candidate_stats.append(_match_statistics(candidate, matches))
    return _paired_hierarchical_bootstrap(
        np.stack(reference_stats), np.stack(candidate_stats), iterations=10_000
    )


def select_round_b() -> dict[str, Any]:
    output = SUBGRAPH_EXPERIMENT_ROOT / "selection"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    decisions: dict[str, Any] = {}
    family_triggers: dict[str, bool] = {family: False for family in ROUND_B_BY_FAMILY}
    for view in ROUND_A_VIEWS:
        differences = _seed_differences(view)
        bootstrap = _bootstrap_for_view(view)
        overall = {
            metric: _passes_trigger(metric, differences[metric], bootstrap, threshold)
            for metric, threshold in PRACTICAL_THRESHOLDS.items()
        }
        group_rows = _group_seed_rows(view)
        group_triggers: dict[str, dict[str, bool]] = {}
        for group, values in group_rows.groupby("group"):
            checks: dict[str, bool] = {}
            for metric, threshold in GROUP_THRESHOLDS.items():
                column = f"{metric}_difference"
                series = values[column]
                mean_practical = float(series.mean()) <= threshold if metric in LOWER_IS_BETTER else float(series.mean()) >= threshold
                same_direction = int((series < 0).sum()) >= 2 if metric in LOWER_IS_BETTER else int((series > 0).sum()) >= 2
                if mean_practical and same_direction:
                    grouped_bootstrap = _group_bootstrap(view, str(group))
                    checks[metric] = _passes_trigger(metric, series, grouped_bootstrap, threshold)
                else:
                    checks[metric] = False
            group_triggers[str(group)] = checks
        triggered = any(overall.values()) or any(
            any(checks.values()) for checks in group_triggers.values()
        )
        family = resolve_view_spec(view).family
        family_triggers[family] = family_triggers[family] or triggered
        decisions[view] = {
            "family": family,
            "overall_triggers": overall,
            "group_triggers": group_triggers,
            "triggered": triggered,
            "bootstrap": bootstrap,
        }
        for row in differences.to_dict(orient="records"):
            rows.append({"view": view, **row})
    pd.DataFrame(rows).to_csv(output / "round_a_differences_by_seed.csv", index=False)
    selected_views = [
        view
        for family, views in ROUND_B_BY_FAMILY.items()
        if family_triggers[family]
        for view in views
    ]
    state = {
        "selection_split": "validation",
        "practical_thresholds": PRACTICAL_THRESHOLDS,
        "group_thresholds": GROUP_THRESHOLDS,
        "family_triggers": family_triggers,
        "round_b_views": selected_views,
        "decisions": decisions,
        "test_accessed": False,
    }
    (output / "round_b.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def select_task_winners() -> dict[str, Any]:
    views = completed_views()
    rows = []
    for view in views:
        for seed in CONFIRMATION_SEEDS:
            rows.append(metric_row(view, seed, read_result(_result_path(view, seed)), "validation"))
    frame = pd.DataFrame(rows)
    means = frame.groupby("view").mean(numeric_only=True)
    f80_accuracy = float(means.loc["f80", "event_accuracy"])
    eligible_event = means[means.event_accuracy >= f80_accuracy - 0.01]
    winners = {
        "event": str(eligible_event.event_macro_f1.idxmax()),
        "time": str(means.time_mae_seconds.idxmin()),
        "position": str(means.position_distance_mae_m.idxmin()),
    }
    controls = sorted(
        {
            f"random_{view}"
            for view in winners.values()
            if resolve_view_spec(view).family in {"possession", "transition", "spatial"}
        }
    )
    output = SUBGRAPH_EXPERIMENT_ROOT / "selection"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_validation_by_seed.csv", index=False)
    means.reset_index().to_csv(output / "all_validation_summary.csv", index=False)
    state = {
        "selection_split": "validation",
        "event_rule": "maximize Macro-F1 subject to Accuracy >= F80 Accuracy - 0.01",
        "winners": winners,
        "size_matched_controls": controls,
        "available_views": views,
        "test_accessed": False,
    }
    (output / "winners.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def build_test_report() -> Path:
    selection = json.loads(
        (SUBGRAPH_EXPERIMENT_ROOT / "selection/winners.json").read_text(encoding="utf-8")
    )
    views = ["f80", *sorted(set(selection["winners"].values())), *selection["size_matched_controls"]]
    rows = []
    statistics: dict[str, list[np.ndarray]] = {view: [] for view in views}
    group_rows: list[dict[str, Any]] = []
    expected_ids: list[str] | None = None
    for seed in CONFIRMATION_SEEDS:
        for view in views:
            root = test_dir(view, seed)
            result = read_result(root)
            if not result.get("test_accessed") or result.get("test") is None:
                raise RuntimeError(f"Missing locked test result: {root}")
            rows.append(metric_row(view, seed, result, "test"))
            frame = pd.read_parquet(root / "test_predictions.parquet").sort_values("sample_id").reset_index(drop=True)
            ids = frame.sample_id.tolist()
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError("Locked subgraph test IDs are not paired")
            matches = sorted(frame.match_id.astype(int).unique().tolist())
            statistics[view].append(_match_statistics(frame, matches))
            context = _add_source_context(frame, view)
            for group, mask in _group_masks(context).items():
                if not bool(mask.any()):
                    continue
                metrics = compute_metrics(frame[mask.to_numpy()])
                group_rows.append(
                    {
                        "view": view,
                        "seed": seed,
                        "group": group,
                        "samples": int(mask.sum()),
                        "event_accuracy": metrics["event"]["accuracy"],
                        "event_macro_f1": metrics["event"]["macro_f1"],
                        "time_mae_seconds": metrics["time"]["mae_seconds"],
                        "position_distance_mae_m": metrics["position"]["distance_mae_m"],
                    }
                )
    output = SUBGRAPH_EXPERIMENT_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "test_by_seed.csv", index=False)
    by_seed.groupby("view").agg(
        event_accuracy_mean=("event_accuracy", "mean"),
        event_accuracy_std=("event_accuracy", "std"),
        event_macro_f1_mean=("event_macro_f1", "mean"),
        event_macro_f1_std=("event_macro_f1", "std"),
        time_mae_mean=("time_mae_seconds", "mean"),
        time_mae_std=("time_mae_seconds", "std"),
        position_error_mean=("position_distance_mae_m", "mean"),
        position_error_std=("position_distance_mae_m", "std"),
    ).to_csv(output / "test_summary.csv")
    pd.DataFrame(group_rows).to_csv(output / "state_groups.csv", index=False)
    bootstrap = {
        f"{view}_minus_f80": _paired_hierarchical_bootstrap(
            np.stack(statistics["f80"]), np.stack(statistics[view]), iterations=10_000
        )
        for view in views
        if view != "f80"
    }
    (output / "paired_bootstrap.json").write_text(
        json.dumps({"iterations": 10_000, "comparisons": bootstrap}, indent=2), encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(
            {
                "winners": selection["winners"],
                "controls": selection["size_matched_controls"],
                "test_samples": len(expected_ids or []),
                "conclusion_target": "task-specific optimal history scales and semantic views",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
