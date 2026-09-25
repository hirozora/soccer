"""Paired test reporting for Semantic V3 Possession variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT, POSSESSION_GRAPH_ROOT
from .metrics import compute_metrics
from .possession_study import (
    FEATURE_VARIANTS,
    POSSESSION_EXPERIMENT_ROOT,
    TOPOLOGY_VARIANTS,
    metric_row,
)
from .reporting import _match_statistics, _paired_hierarchical_bootstrap


def test_dir(variant: str, seed: int) -> Path:
    if variant == "b0":
        return EXPERIMENT_ROOT / "loss_balance_test/j2_020" / f"seed{seed}"
    return POSSESSION_EXPERIMENT_ROOT / "test" / variant / f"seed{seed}"


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def _add_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["control_state"] = -1
    result["event_role"] = -1
    result["switch_confirmed"] = False
    result["possession_event_count"] = 0.0
    result["possession_pass_ratio_k80"] = np.nan
    graph_paths = {
        int(row.match_id): POSSESSION_GRAPH_ROOT / row.graph_path
        for row in pd.read_csv(POSSESSION_GRAPH_ROOT / "metadata/match_index.csv").itertuples()
    }
    for match_id, row_indices in result.groupby("match_id").groups.items():
        graph = torch.load(graph_paths[int(match_id)], map_location="cpu", weights_only=True)
        event = graph["node_stores"]["event"]
        rows = np.asarray(list(row_indices), dtype=np.int64)
        indices = result.loc[rows, "current_event_index"].to_numpy(np.int64)
        result.loc[rows, "control_state"] = event["control_state_after_index"][indices].numpy()
        result.loc[rows, "event_role"] = event["event_role_index"][indices].numpy()
        result.loc[rows, "switch_confirmed"] = event["switch_confirmed"][indices].numpy()
        result.loc[rows, "possession_event_count"] = event[
            "possession_event_count_so_far"
        ][indices].numpy()
        for row, anchor in zip(rows, indices):
            possession_index = int(event["possession_local_index"][anchor])
            if possession_index < 0:
                continue
            start = max(0, int(anchor) - 79)
            in_possession = (
                event["possession_local_index"][start : anchor + 1] == possession_index
            )
            if bool(in_possession.any()):
                event_types = event["event_type_index"][start : anchor + 1][in_possession]
                result.loc[row, "possession_pass_ratio_k80"] = float(
                    (event_types == 7).float().mean()
                )
    return result


def _group_rows(variant: str, seed: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    masks = {
        "CONTROL": frame.control_state == 2,
        "CONTESTED": frame.control_state == 3,
        "restart": frame.event_role == 2,
        "switch": frame.switch_confirmed.astype(bool),
        "boundary": frame.event_role == 4,
        "possession_1_3_events": frame.possession_event_count.between(1, 3),
        "possession_4_10_events": frame.possession_event_count.between(4, 10),
        "possession_11plus_events": frame.possession_event_count >= 11,
        "pass_heavy_possession": frame.possession_pass_ratio_k80 >= 0.7,
    }
    rows = []
    for name, mask in masks.items():
        selected = frame[mask]
        if selected.empty:
            continue
        metrics = compute_metrics(selected)
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "group": name,
                "samples": len(selected),
                "event_accuracy": metrics["event"]["accuracy"],
                "event_macro_f1": metrics["event"]["macro_f1"],
                "time_mae_seconds": metrics["time"]["mae_seconds"],
                "position_distance_mae_m": metrics["position"]["distance_mae_m"],
            }
        )
    return rows


def build_possession_report() -> Path:
    selection = json.loads(
        (POSSESSION_EXPERIMENT_ROOT / "selection/final_v3.json").read_text(encoding="utf-8")
    )
    variants = ["b0", *TOPOLOGY_VARIANTS, *FEATURE_VARIANTS, "v3_j0"]
    rows: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    statistics: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    expected_ids: list[str] | None = None
    match_ids: list[int] | None = None
    for seed in CONFIRMATION_SEEDS:
        for variant in variants:
            path = test_dir(variant, seed)
            result = _read_result(path)
            if not result.get("test_accessed") or result.get("test") is None:
                raise RuntimeError(f"Missing locked test result: {path}")
            rows.append(metric_row(variant, seed, result, "test"))
            frame = pd.read_parquet(path / "test_predictions.parquet").sort_values("sample_id")
            ids = frame.sample_id.tolist()
            if expected_ids is None:
                expected_ids = ids
                match_ids = sorted(frame.match_id.astype(int).unique().tolist())
            elif ids != expected_ids:
                raise RuntimeError("Possession test sample IDs are not paired")
            statistics[variant].append(_match_statistics(frame, match_ids or []))
            grouped.extend(_group_rows(variant, seed, _add_context(frame)))

    output = POSSESSION_EXPERIMENT_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "test_by_seed.csv", index=False)
    frame.groupby("variant").agg(
        event_accuracy_mean=("event_accuracy", "mean"),
        event_accuracy_std=("event_accuracy", "std"),
        event_macro_f1_mean=("event_macro_f1", "mean"),
        event_macro_f1_std=("event_macro_f1", "std"),
        time_mae_mean=("time_mae_seconds", "mean"),
        time_mae_std=("time_mae_seconds", "std"),
        position_error_mean=("position_distance_mae_m", "mean"),
        position_error_std=("position_distance_mae_m", "std"),
    ).to_csv(output / "test_summary.csv")
    pd.DataFrame(grouped).to_csv(output / "state_groups.csv", index=False)
    bootstrap = {
        f"{variant}_minus_b0": _paired_hierarchical_bootstrap(
            np.stack(statistics["b0"]), np.stack(statistics[variant])
        )
        for variant in variants
        if variant != "b0"
    }
    (output / "paired_bootstrap.json").write_text(
        json.dumps({"iterations": 10_000, "comparisons": bootstrap}, indent=2),
        encoding="utf-8",
    )
    selected = selection["selected_variant"]
    selected_metrics = frame[frame.variant == selected].mean(numeric_only=True)
    b0_metrics = frame[frame.variant == "b0"].mean(numeric_only=True)
    delta = selected_metrics - b0_metrics
    acceptance = {
        "event_accuracy_within_1pp": bool(delta.event_accuracy >= -0.01),
        "time_mae_within_0.05s": bool(delta.time_mae_seconds <= 0.05),
        "position_within_0.5m": bool(delta.position_distance_mae_m <= 0.5),
    }
    acceptance["adopt_v3"] = all(acceptance.values())
    summary = {
        "selected": selection,
        "acceptance": acceptance,
        "test_difference_selected_minus_b0": {
            name: float(delta[name])
            for name in (
                "event_accuracy",
                "event_macro_f1",
                "time_mae_seconds",
                "position_distance_mae_m",
            )
        },
        "test_samples": len(expected_ids or []),
    }
    (output / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
