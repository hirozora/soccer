"""Paired attribution of the J2 versus NMSTPP position-error gap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from football_benchmark.constants import (
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
    ZONE_CENTERS_100,
)
from football_benchmark.data import load_records

from .constants import CONFIRMATION_SEEDS, RAW_EVENT_NAMES, SEMANTIC_GRAPH_ROOT
from .dependency_study import DEPENDENCY_ROOT


NMSTPP_ROOT = Path(__file__).resolve().parents[3] / "benchmark_unified_v1/experiments/feasibility/final/nmstpp/nmstpp"
J2_TEST_ROOT = Path(__file__).resolve().parents[2] / "experiments/loss_balance_test/j2_020"
TIME_LABELS = ("0-2s", "2-5s", "5-15s", "15-60s", "period_break")
DISTANCE_LABELS = ("0-5m", "5-10m", "10-20m", "20-40m", "40-60m", "60m+")


def _distance(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.sqrt(((x1 - x2) * PITCH_LENGTH_METERS) ** 2 + ((y1 - y2) * PITCH_WIDTH_METERS) ** 2)


def _load_seed(seed: int) -> pd.DataFrame:
    hgt = pd.read_parquet(J2_TEST_ROOT / f"seed{seed}/test_predictions.parquet").sort_values("sample_id").reset_index(drop=True)
    nm = pd.read_parquet(NMSTPP_ROOT / f"seed{seed}/test_predictions.parquet").sort_values("sample_id").reset_index(drop=True)
    if hgt.sample_id.tolist() != nm.sample_id.tolist():
        raise RuntimeError(f"J2 and NMSTPP sample IDs differ for seed {seed}")
    for axis in ("x", "y"):
        if not np.allclose(hgt[f"position_true_{axis}"], nm[f"position_true_{axis}"], atol=1e-6):
            raise RuntimeError(f"Position targets differ for seed {seed}")
    if not np.array_equal(hgt.position_mask, nm.position_mask):
        raise RuntimeError(f"Position masks differ for seed {seed}")
    result = hgt[[
        "sample_id", "match_id", "current_event_index", "event_true", "time_true",
        "time_mask", "position_true_x", "position_true_y", "position_mask", "zone_true",
    ]].copy()
    result["hgt_error_m"] = _distance(
        hgt.position_pred_x.to_numpy(), hgt.position_pred_y.to_numpy(),
        hgt.position_true_x.to_numpy(), hgt.position_true_y.to_numpy(),
    )
    result["nmstpp_error_m"] = _distance(
        nm.position_pred_x.to_numpy(), nm.position_pred_y.to_numpy(),
        nm.position_true_x.to_numpy(), nm.position_true_y.to_numpy(),
    )
    centers = np.asarray(ZONE_CENTERS_100, dtype=np.float32) / 100
    hgt_predicted_centers = centers[hgt.zone_pred.to_numpy(int)]
    true_centers = centers[hgt.zone_true.to_numpy(int)]
    result["hgt_equal_error_m"] = _distance(
        hgt_predicted_centers[:, 0], hgt_predicted_centers[:, 1],
        true_centers[:, 0], true_centers[:, 1],
    )
    result["nmstpp_equal_error_m"] = _distance(
        nm.position_pred_x.to_numpy(), nm.position_pred_y.to_numpy(),
        nm.position_equal_true_x.to_numpy(), nm.position_equal_true_y.to_numpy(),
    )
    result["hgt_zone_correct"] = (hgt.zone_pred == hgt.zone_true).astype(float)
    result["nmstpp_zone_correct"] = (nm.zone_pred == nm.zone_true).astype(float)
    result["hgt_dx_m"] = (hgt.position_pred_x - hgt.position_true_x) * PITCH_LENGTH_METERS
    result["hgt_dy_m"] = (hgt.position_pred_y - hgt.position_true_y) * PITCH_WIDTH_METERS
    result["nmstpp_dx_m"] = (nm.position_pred_x - nm.position_true_x) * PITCH_LENGTH_METERS
    result["nmstpp_dy_m"] = (nm.position_pred_y - nm.position_true_y) * PITCH_WIDTH_METERS
    return result


def _anchor_positions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    records = {record.match_id: record for record in load_records("test", graph_root=SEMANTIC_GRAPH_ROOT)}
    x = np.empty(len(frame), dtype=np.float32)
    y = np.empty(len(frame), dtype=np.float32)
    for match_id, indices in frame.groupby("match_id").groups.items():
        graph = torch.load(records[int(match_id)].graph_path, map_location="cpu", weights_only=False)
        event = graph["node_stores"]["event"]
        current = frame.loc[indices, "current_event_index"].to_numpy(np.int64)
        start = event["start_position"][current].numpy()
        end = event["end_position"][current].numpy()
        end_mask = event["end_position_mask"][current].numpy().astype(bool)
        representative = np.where(end_mask[:, None], end, start)
        x[indices] = representative[:, 0]
        y[indices] = representative[:, 1]
        target_types = event["event_type_index"][current + 1].numpy()
        if not np.array_equal(target_types, frame.loc[indices, "event_true"].to_numpy()):
            raise RuntimeError(f"Target event mismatch in match {match_id}")
    return x, y


def _cluster_ci(
    frame: pd.DataFrame,
    column: str = "gap_m",
    iterations: int = 2_000,
    seed: int = 20260715,
) -> list[float]:
    by_match = frame.groupby("match_id").agg(total=(column, "sum"), count=(column, "size"))
    values = by_match[["total", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for index in range(iterations):
        selected = values[rng.integers(0, len(values), len(values))]
        estimates[index] = selected[:, 0].sum() / selected[:, 1].sum()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _group_summary(frame: pd.DataFrame, column: str, order: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(frame)
    for value in order:
        selected = frame[frame[column] == value]
        if selected.empty:
            continue
        rows.append({
            column: value,
            "samples": len(selected),
            "proportion": len(selected) / total,
            "hgt_mean_error_m": selected.hgt_error_m.mean(),
            "hgt_median_error_m": selected.hgt_error_m.median(),
            "hgt_p90_error_m": selected.hgt_error_m.quantile(0.9),
            "nmstpp_mean_error_m": selected.nmstpp_error_m.mean(),
            "nmstpp_median_error_m": selected.nmstpp_error_m.median(),
            "nmstpp_p90_error_m": selected.nmstpp_error_m.quantile(0.9),
            "mean_gap_hgt_minus_nmstpp_m": selected.gap_m.mean(),
            "gap_ci95_low": _cluster_ci(selected)[0],
            "gap_ci95_high": _cluster_ci(selected)[1],
            "gap_contribution_m": len(selected) / total * selected.gap_m.mean(),
            "hgt_equal_mean_error_m": selected.hgt_equal_error_m.mean(),
            "nmstpp_equal_mean_error_m": selected.nmstpp_equal_error_m.mean(),
            "equal_gap_hgt_minus_nmstpp_m": selected.equal_gap_m.mean(),
            "equal_gap_contribution_m": len(selected) / total * selected.equal_gap_m.mean(),
        })
    return pd.DataFrame(rows)


def run_position_attribution(output_dir: Path | None = None) -> Path:
    output = output_dir or DEPENDENCY_ROOT / "position_attribution"
    output.mkdir(parents=True, exist_ok=True)
    seeds = [_load_seed(seed) for seed in CONFIRMATION_SEEDS]
    reference = seeds[0]
    for frame in seeds[1:]:
        columns = ["sample_id", "event_true", "time_true", "time_mask", "position_true_x", "position_true_y", "zone_true"]
        if not frame[columns].equals(reference[columns]):
            raise RuntimeError("Targets differ across seeds")
    frame = reference[[
        "sample_id", "match_id", "current_event_index", "event_true", "time_true",
        "time_mask", "position_true_x", "position_true_y", "position_mask", "zone_true",
    ]].copy()
    averaged = (pd.concat([
        value[["hgt_error_m", "nmstpp_error_m", "hgt_equal_error_m", "nmstpp_equal_error_m", "hgt_zone_correct", "nmstpp_zone_correct", "hgt_dx_m", "hgt_dy_m", "nmstpp_dx_m", "nmstpp_dy_m"]]
        for value in seeds
    ], keys=range(len(seeds))).groupby(level=1).mean())
    frame[averaged.columns] = averaged.to_numpy()
    frame["gap_m"] = frame.hgt_error_m - frame.nmstpp_error_m
    frame["equal_gap_m"] = frame.hgt_equal_error_m - frame.nmstpp_equal_error_m
    anchor_x, anchor_y = _anchor_positions(frame)
    frame["movement_distance_m"] = _distance(anchor_x, anchor_y, frame.position_true_x.to_numpy(), frame.position_true_y.to_numpy())
    frame["event_name"] = [RAW_EVENT_NAMES[int(value)] for value in frame.event_true]
    frame["time_group"] = pd.cut(
        frame.time_true, [-1e-9, 2, 5, 15, 60.0001], labels=TIME_LABELS[:4], right=False
    ).astype(object)
    frame.loc[~frame.time_mask, "time_group"] = "period_break"
    frame["movement_group"] = pd.cut(
        frame.movement_distance_m, [0, 5, 10, 20, 40, 60, np.inf], labels=DISTANCE_LABELS, right=False, include_lowest=True
    ).astype(str)
    centers = np.asarray(ZONE_CENTERS_100, dtype=np.float32) / 100
    true_centers = centers[frame.zone_true.to_numpy(int)]
    frame["zone_quantization_error_m"] = _distance(
        true_centers[:, 0], true_centers[:, 1], frame.position_true_x.to_numpy(), frame.position_true_y.to_numpy()
    )
    frame["zone_outcome"] = np.select(
        [
            (frame.hgt_zone_correct == 1) & (frame.nmstpp_zone_correct == 1),
            (frame.hgt_zone_correct == 1) & (frame.nmstpp_zone_correct < 1),
            (frame.hgt_zone_correct < 1) & (frame.nmstpp_zone_correct == 1),
        ],
        ["both_correct", "hgt_only", "nmstpp_only"],
        default="both_wrong",
    )
    summaries = {
        "event": _group_summary(frame, "event_name", list(RAW_EVENT_NAMES)),
        "zone": _group_summary(frame, "zone_true", list(range(20))),
        "time": _group_summary(frame, "time_group", list(TIME_LABELS)),
        "movement": _group_summary(frame, "movement_group", list(DISTANCE_LABELS)),
        "zone_outcome": _group_summary(frame, "zone_outcome", ["both_correct", "hgt_only", "nmstpp_only", "both_wrong"]),
    }
    overall_gap = float(frame.gap_m.mean())
    equal_gap = float(frame.equal_gap_m.mean())
    for name, summary in summaries.items():
        if not np.isclose(summary.gap_contribution_m.sum(), overall_gap, atol=1e-9):
            raise RuntimeError(f"Gap contributions do not reproduce overall gap for {name}")
        if not np.isclose(summary.equal_gap_contribution_m.sum(), equal_gap, atol=1e-9):
            raise RuntimeError(f"Equal-granularity contributions do not reproduce gap for {name}")
        summary.to_csv(output / f"by_{name}.csv", index=False)
    frame.to_parquet(output / "paired_samples.parquet", index=False)
    heatmap = frame.groupby([
        pd.cut(frame.position_true_y, np.linspace(0, 1, 9), labels=False, include_lowest=True),
        pd.cut(frame.position_true_x, np.linspace(0, 1, 11), labels=False, include_lowest=True),
    ]).gap_m.mean().unstack()
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap, cmap="coolwarm", center=0, ax=axis)
    axis.set(title="J2 minus NMSTPP position error (m)", xlabel="Target x bin", ylabel="Target y bin")
    fig.tight_layout(); fig.savefig(output / "error_gap_heatmap.png", dpi=180); plt.close(fig)
    summary: dict[str, Any] = {
        "samples": len(frame),
        "seeds": list(CONFIRMATION_SEEDS),
        "hgt_mean_error_m": float(frame.hgt_error_m.mean()),
        "nmstpp_mean_error_m": float(frame.nmstpp_error_m.mean()),
        "overall_gap_m": overall_gap,
        "overall_gap_ci95": _cluster_ci(frame),
        "comparison_warning": (
            "The published NMSTPP 14.37m metric is zone-center to zone-center and "
            "must not be compared with J2 continuous-to-true-coordinate error."
        ),
        "hgt_equal_granularity_error_m": float(frame.hgt_equal_error_m.mean()),
        "nmstpp_equal_granularity_error_m": float(frame.nmstpp_equal_error_m.mean()),
        "equal_granularity_gap_m": equal_gap,
        "equal_granularity_gap_ci95": _cluster_ci(frame, "equal_gap_m"),
        "zone_quantization_floor_m": float(frame.zone_quantization_error_m.mean()),
        "hgt_zone_accuracy": float(frame.hgt_zone_correct.mean()),
        "nmstpp_zone_accuracy": float(frame.nmstpp_zone_correct.mean()),
        "hgt_bias_xy_m": [float(frame.hgt_dx_m.mean()), float(frame.hgt_dy_m.mean())],
        "nmstpp_bias_xy_m": [float(frame.nmstpp_dx_m.mean()), float(frame.nmstpp_dy_m.mean())],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
