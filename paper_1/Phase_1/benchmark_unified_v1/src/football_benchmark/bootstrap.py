"""Hierarchical paired bootstrap over seeds and test matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS


@dataclass(frozen=True)
class MatchStatistics:
    event_confusion: np.ndarray
    time_abs_sum: float
    time_sq_sum: float
    time_count: int
    position_distance_sum: float
    position_distance_sq_sum: float
    position_count: int
    zone_confusion: np.ndarray


def _confusion(frame: pd.DataFrame, true: str, predicted: str, mask: str, classes: int) -> np.ndarray:
    selected = frame[frame[mask]]
    flat = selected[true].to_numpy(dtype=int) * classes + selected[predicted].to_numpy(dtype=int)
    return np.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def match_statistics(frame: pd.DataFrame, event_classes: int, contract: str) -> MatchStatistics:
    event = _confusion(frame, "event_true", "event_pred", "event_mask", event_classes)
    if contract == "seq2event":
        time_abs_sum = time_sq_sum = 0.0
        time_count = 0
    else:
        selected_time = frame[frame.time_mask]
        errors = selected_time.time_pred.to_numpy() - selected_time.time_true.to_numpy()
        time_abs_sum = float(np.abs(errors).sum())
        time_sq_sum = float((errors**2).sum())
        time_count = len(errors)
    selected_position = frame[frame.position_mask]
    true_x = "position_equal_true_x" if contract == "nmstpp" else "position_true_x"
    true_y = "position_equal_true_y" if contract == "nmstpp" else "position_true_y"
    distances = np.sqrt(
        ((selected_position.position_pred_x.to_numpy() - selected_position[true_x].to_numpy()) * PITCH_LENGTH_METERS) ** 2
        + ((selected_position.position_pred_y.to_numpy() - selected_position[true_y].to_numpy()) * PITCH_WIDTH_METERS) ** 2
    )
    zone = (
        _confusion(frame, "zone_true", "zone_pred", "position_mask", 20)
        if contract == "nmstpp"
        else np.zeros((20, 20), dtype=np.int64)
    )
    return MatchStatistics(
        event,
        time_abs_sum,
        time_sq_sum,
        time_count,
        float(distances.sum()),
        float((distances**2).sum()),
        len(distances),
        zone,
    )


def _f1_from_confusion(confusion: np.ndarray) -> tuple[float, float, float]:
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    accuracy = float(true_positive.sum() / max(confusion.sum(), 1))
    macro = float(f1.mean())
    weighted = float((f1 * support).sum() / max(support.sum(), 1))
    return accuracy, macro, weighted


def _aggregate(values: list[MatchStatistics]) -> dict[str, float]:
    event = np.sum([value.event_confusion for value in values], axis=0)
    zone = np.sum([value.zone_confusion for value in values], axis=0)
    time_count = sum(value.time_count for value in values)
    position_count = sum(value.position_count for value in values)
    event_accuracy, event_macro, event_weighted = _f1_from_confusion(event)
    result = {
        "event_accuracy": event_accuracy,
        "event_macro_f1": event_macro,
        "event_weighted_f1": event_weighted,
        "position_distance_mae_m": sum(value.position_distance_sum for value in values) / max(position_count, 1),
        "position_distance_rmse_m": np.sqrt(sum(value.position_distance_sq_sum for value in values) / max(position_count, 1)),
    }
    if time_count:
        result["time_mae_seconds"] = sum(value.time_abs_sum for value in values) / time_count
        result["time_rmse_seconds"] = np.sqrt(sum(value.time_sq_sum for value in values) / time_count)
    if zone.sum():
        zone_accuracy, zone_macro, _ = _f1_from_confusion(zone)
        result["zone_accuracy"] = zone_accuracy
        result["zone_macro_f1"] = zone_macro
    return result


def hierarchical_paired_bootstrap(
    hgt_frames: list[pd.DataFrame],
    baseline_frames: list[pd.DataFrame],
    contract: str,
    replicates: int = 10_000,
    seed: int = 20260723,
) -> dict[str, dict[str, float]]:
    if len(hgt_frames) != len(baseline_frames):
        raise ValueError("HGT and baseline seed counts differ")
    event_classes = 10 if contract == "unified_lem" else 4
    match_ids = sorted(hgt_frames[0].match_id.unique().tolist())
    hgt_stats: list[list[MatchStatistics]] = []
    baseline_stats: list[list[MatchStatistics]] = []
    for hgt, baseline in zip(hgt_frames, baseline_frames):
        if not hgt.sample_id.equals(baseline.sample_id):
            raise ValueError("Paired prediction sample IDs differ")
        for column in ("match_id", "event_true", "event_mask", "position_mask"):
            if not hgt[column].equals(baseline[column]):
                raise ValueError(f"Paired prediction column differs: {column}")
        if contract != "seq2event" and not hgt.time_mask.equals(baseline.time_mask):
            raise ValueError("Paired time masks differ")
        hgt_stats.append(
            [match_statistics(hgt[hgt.match_id == match_id], event_classes, contract) for match_id in match_ids]
        )
        baseline_stats.append(
            [match_statistics(baseline[baseline.match_id == match_id], event_classes, contract) for match_id in match_ids]
        )
    observed_hgt = _aggregate([value for seed_values in hgt_stats for value in seed_values])
    observed_baseline = _aggregate([value for seed_values in baseline_stats for value in seed_values])
    metric_names = sorted(set(observed_hgt).intersection(observed_baseline))
    samples = {name: np.empty(replicates, dtype=float) for name in metric_names}
    rng = np.random.default_rng(seed)
    num_seeds = len(hgt_stats)
    num_matches = len(match_ids)
    for replicate in range(replicates):
        selected_hgt: list[MatchStatistics] = []
        selected_baseline: list[MatchStatistics] = []
        for seed_index in rng.integers(0, num_seeds, size=num_seeds):
            for match_index in rng.integers(0, num_matches, size=num_matches):
                selected_hgt.append(hgt_stats[int(seed_index)][int(match_index)])
                selected_baseline.append(baseline_stats[int(seed_index)][int(match_index)])
        hgt_metric = _aggregate(selected_hgt)
        baseline_metric = _aggregate(selected_baseline)
        for name in metric_names:
            samples[name][replicate] = hgt_metric[name] - baseline_metric[name]
    return {
        name: {
            "hgt": float(observed_hgt[name]),
            "baseline": float(observed_baseline[name]),
            "delta_hgt_minus_baseline": float(observed_hgt[name] - observed_baseline[name]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }
