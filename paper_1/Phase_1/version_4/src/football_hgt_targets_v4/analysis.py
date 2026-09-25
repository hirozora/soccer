"""Leak-free descriptive analysis of the three next-event targets."""

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

from football_benchmark.data import load_records
from football_benchmark.mappings import position_to_zone

from .constants import ANALYSIS_ROOT, RAW_EVENT_NAMES, SEMANTIC_GRAPH_ROOT


def build_training_targets() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in load_records("train", graph_root=SEMANTIC_GRAPH_ROOT):
        graph = torch.load(record.graph_path, map_location="cpu", weights_only=False)
        event = graph["node_stores"]["event"]
        count = int(event["num_nodes"]) - 1
        target_types = event["event_type_index"][1:].numpy()
        target_positions = event["start_position"][1:].numpy()
        frame = pd.DataFrame(
            {
                "match_id": record.match_id,
                "current_event_index": np.arange(count, dtype=np.int64),
                "current_period": event["period_index"][:-1].numpy(),
                "current_seconds": event["absolute_seconds"][:-1].numpy(),
                "event_type": target_types,
                "target_period": event["period_index"][1:].numpy(),
                "target_seconds": event["absolute_seconds"][1:].numpy(),
                "x": target_positions[:, 0],
                "y": target_positions[:, 1],
            }
        )
        frame["event_name"] = [RAW_EVENT_NAMES[index] for index in target_types]
        frames.append(frame)
    targets = pd.concat(frames, ignore_index=True)
    targets["period_break"] = targets.current_period != targets.target_period
    targets["delta_seconds_raw"] = targets.target_seconds - targets.current_seconds
    targets["delta_seconds_60"] = targets.delta_seconds_raw.clip(0.0, 60.0)
    targets["time_mask"] = ~targets.period_break
    positions = torch.tensor(targets[["x", "y"]].to_numpy(), dtype=torch.float32)
    targets["zone"] = position_to_zone(positions).numpy()
    targets["time_bucket"] = np.digitize(
        targets.delta_seconds_60.to_numpy(), (2.0, 5.0, 15.0), right=False
    )
    return targets


def _mutual_information(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    counts = np.zeros((na, nb), dtype=np.float64)
    np.add.at(counts, (a, b), 1)
    probability = counts / counts.sum()
    expected = probability.sum(1, keepdims=True) @ probability.sum(0, keepdims=True)
    active = probability > 0
    return float(
        np.sum(probability[active] * np.log2(probability[active] / expected[active]))
    )


def _entropy(values: np.ndarray, cardinality: int) -> float:
    counts = np.bincount(values, minlength=cardinality).astype(float)
    probability = counts[counts > 0] / counts.sum()
    return float(-np.sum(probability * np.log2(probability)))


def _dependency_metrics(frame: pd.DataFrame) -> dict[str, float]:
    active = frame[frame.time_mask]
    event = active.event_type.to_numpy(int)
    time = active.time_bucket.to_numpy(int)
    zone = active.zone.to_numpy(int)
    event_time = _mutual_information(event, time, 10, 4)
    event_zone = _mutual_information(event, zone, 10, 20)
    time_zone = _mutual_information(time, zone, 4, 20)
    conditional = 0.0
    for event_type in range(10):
        selected = event == event_type
        conditional += float(selected.mean()) * _mutual_information(
            time[selected], zone[selected], 4, 20
        )
    return {
        "event_time_mi_bits": event_time,
        "event_zone_mi_bits": event_zone,
        "time_zone_mi_bits": time_zone,
        "time_zone_given_event_cmi_bits": conditional,
        "event_time_normalized_mi": event_time
        / min(_entropy(event, 10), _entropy(time, 4)),
        "event_zone_normalized_mi": event_zone
        / min(_entropy(event, 10), _entropy(zone, 20)),
        "conditional_time_zone_fraction_of_zone_entropy": conditional
        / max(_entropy(zone, 20) - event_zone, 1e-12),
    }


def _plots(frame: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    counts = frame.event_name.value_counts().reindex(RAW_EVENT_NAMES)
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=axis, color="#3378a8")
    axis.tick_params(axis="x", rotation=35)
    axis.set(xlabel="Next event", ylabel="Training samples", title="Next-event class distribution")
    fig.tight_layout()
    fig.savefig(output / "event_distribution.png", dpi=180)
    plt.close(fig)

    active = frame[frame.time_mask]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(active.delta_seconds_raw.clip(upper=120), bins=80, ax=axes[0])
    sns.histplot(np.log1p(active.delta_seconds_raw), bins=80, ax=axes[1])
    axes[0].set_title("Raw gap (display capped at 120s)")
    axes[1].set_title("log1p(raw gap)")
    fig.tight_layout()
    fig.savefig(output / "time_distributions.png", dpi=180)
    plt.close(fig)

    matrix = pd.crosstab(frame.event_name, frame.zone, normalize="index").reindex(RAW_EVENT_NAMES)
    fig, axis = plt.subplots(figsize=(13, 5))
    sns.heatmap(matrix, cmap="viridis", ax=axis)
    axis.set_title("P(Zone | Event Type)")
    fig.tight_layout()
    fig.savefig(output / "event_zone_heatmap.png", dpi=180)
    plt.close(fig)


def run_analysis(output_dir: Path = ANALYSIS_ROOT) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = build_training_targets()
    active_time = frame[frame.time_mask]
    event_counts = (
        frame.groupby(["event_type", "event_name"]).size().rename("count").reset_index()
    )
    event_counts["proportion"] = event_counts["count"] / len(frame)
    event_counts.to_csv(output / "event_distribution.csv", index=False)

    quantiles = active_time.delta_seconds_raw.quantile(
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    )
    pd.DataFrame(
        {"quantile": quantiles.index, "raw_seconds": quantiles.values}
    ).assign(log1p=lambda value: np.log1p(value.raw_seconds)).to_csv(
        output / "time_quantiles.csv", index=False
    )
    active_time.groupby("event_name").delta_seconds_raw.agg(
        count="count",
        mean="mean",
        median="median",
        p90=lambda values: values.quantile(0.9),
        p99=lambda values: values.quantile(0.99),
    ).reindex(RAW_EVENT_NAMES).to_csv(output / "time_by_event.csv")

    zone_counts = frame.zone.value_counts().sort_index().rename_axis("zone").rename("count")
    zone_counts.to_frame().assign(proportion=lambda value: value["count"] / len(frame)).to_csv(
        output / "zone_distribution.csv"
    )
    event_zone = pd.crosstab(frame.event_name, frame.zone).reindex(RAW_EVENT_NAMES)
    event_zone.to_csv(output / "event_zone_counts.csv")
    conditional = (
        active_time.groupby(["event_type", "event_name", "time_bucket", "zone"])
        .size()
        .rename("count")
        .reset_index()
    )
    conditional["probability"] = conditional["count"] / conditional.groupby(
        ["event_type", "time_bucket"]
    )["count"].transform("sum")
    conditional.to_csv(output / "zone_given_event_time.csv", index=False)

    dependency = _dependency_metrics(frame)
    summary: dict[str, Any] = {
        "training_matches": int(frame.match_id.nunique()),
        "transitions": len(frame),
        "time_samples": int(frame.time_mask.sum()),
        "period_break_samples": int(frame.period_break.sum()),
        "position_samples": int(frame[["x", "y"]].notna().all(axis=1).sum()),
        "raw_time_mean": float(active_time.delta_seconds_raw.mean()),
        "raw_time_median": float(active_time.delta_seconds_raw.median()),
        "raw_time_over_60": int((active_time.delta_seconds_raw > 60).sum()),
        "dependency": dependency,
    }
    expected_counts = [123561, 5752, 25634, 899, 19221, 1093, 35126, 229414, 2329, 5996]
    observed = event_counts.sort_values("event_type")["count"].tolist()
    if len(frame) != 449025 or int(frame.time_mask.sum()) != 448759:
        raise AssertionError("Training target totals do not match the frozen protocol")
    if observed != expected_counts:
        raise AssertionError(f"Event counts differ from the frozen protocol: {observed}")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plots(frame, output)
    return summary
