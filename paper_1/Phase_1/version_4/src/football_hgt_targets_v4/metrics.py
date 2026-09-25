"""Per-sample outputs and task-specific metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from football_benchmark.mappings import position_to_zone

from .constants import RAW_EVENT_NAMES
from .model import time_bucket_ids


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _distance(frame: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        ((frame.position_pred_x - frame.position_true_x) * PITCH_LENGTH_METERS) ** 2
        + ((frame.position_pred_y - frame.position_true_y) * PITCH_WIDTH_METERS) ** 2
    ).to_numpy()


@dataclass
class PredictionAccumulator:
    task: str
    chunks: list[pd.DataFrame] = field(default_factory=list)
    loss_sum: float = 0.0
    sample_count: int = 0

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, Any],
        loss: torch.Tensor,
    ) -> None:
        targets = batch["targets"]
        size = len(batch["sample_ids"])
        values: dict[str, Any] = {
            "sample_id": batch["sample_ids"],
            "match_id": _cpu(batch["match_ids"]),
            "current_event_index": _cpu(batch["current_event_indices"]),
            "event_true": _cpu(targets["raw_event_10"]),
        }
        if "event_logits" in predictions:
            probabilities = predictions["event_logits"].softmax(dim=-1)
            values["event_pred"] = _cpu(probabilities.argmax(dim=-1))
            values["event_confidence"] = _cpu(probabilities.max(dim=-1).values)
            for index, name in enumerate(RAW_EVENT_NAMES):
                values[f"event_probability_{index}_{name}"] = _cpu(probabilities[:, index])
        if "time_seconds" in predictions:
            values["time_true"] = _cpu(targets["delta_seconds_60"])
            values["time_pred"] = _cpu(predictions["time_seconds"])
            values["time_mask"] = _cpu(targets["time_mask"].bool())
            values["time_bucket_true"] = _cpu(
                time_bucket_ids(targets["delta_seconds_60"])
            )
            if "time_bucket_logits" in predictions:
                values["time_bucket_pred"] = _cpu(
                    predictions["time_bucket_logits"].argmax(dim=-1)
                )
        if "position_xy" in predictions:
            values["position_true_x"] = _cpu(targets["position_xy"][:, 0])
            values["position_true_y"] = _cpu(targets["position_xy"][:, 1])
            values["position_pred_x"] = _cpu(predictions["position_xy"][:, 0])
            values["position_pred_y"] = _cpu(predictions["position_xy"][:, 1])
            values["position_mask"] = _cpu(targets["position_mask"].bool())
            values["zone_true"] = _cpu(targets["zone_20"])
            zone_pred = predictions.get("zone_pred")
            if zone_pred is None:
                zone_pred = position_to_zone(predictions["position_xy"])
            values["zone_pred"] = _cpu(zone_pred)
        frame = pd.DataFrame(values)
        self.chunks.append(frame)
        self.loss_sum += float(loss.detach()) * size
        self.sample_count += size

    def frame(self) -> pd.DataFrame:
        return pd.concat(self.chunks, ignore_index=True) if self.chunks else pd.DataFrame()

    def compute(self) -> dict[str, Any]:
        return compute_metrics(
            self.frame(), self.loss_sum / max(self.sample_count, 1)
        )


def compute_metrics(frame: pd.DataFrame, mean_loss: float = 0.0) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot compute empty predictions")
    result: dict[str, Any] = {"loss": float(mean_loss), "samples": len(frame)}
    if "event_pred" in frame:
        labels = list(range(10))
        precision, recall, f1, support = precision_recall_fscore_support(
            frame.event_true, frame.event_pred, labels=labels, zero_division=0
        )
        result["event"] = {
            "accuracy": float(accuracy_score(frame.event_true, frame.event_pred)),
            "macro_f1": float(
                f1_score(frame.event_true, frame.event_pred, labels=labels, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(frame.event_true, frame.event_pred, labels=labels, average="weighted", zero_division=0)
            ),
            "per_class": {
                RAW_EVENT_NAMES[index]: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index in labels
            },
            "confusion_matrix": confusion_matrix(
                frame.event_true, frame.event_pred, labels=labels
            ).tolist(),
        }
    if "time_pred" in frame:
        active = frame[frame.time_mask].copy()
        errors = (active.time_pred - active.time_true).to_numpy()
        absolute = np.abs(errors)
        result["time"] = {
            "samples": len(active),
            "mae_seconds": float(absolute.mean()),
            "median_ae_seconds": float(np.median(absolute)),
            "rmse_seconds": float(np.sqrt(np.mean(errors**2))),
            "by_interval": {},
            "by_event": {},
        }
        for bucket in range(4):
            selected = active[active.time_bucket_true == bucket]
            result["time"]["by_interval"][str(bucket)] = {
                "samples": len(selected),
                "mae_seconds": float(np.abs(selected.time_pred - selected.time_true).mean()),
            }
        for index, name in enumerate(RAW_EVENT_NAMES):
            selected = active[active.event_true == index]
            result["time"]["by_event"][name] = {
                "samples": len(selected),
                "mae_seconds": float(np.abs(selected.time_pred - selected.time_true).mean()),
            }
        if "time_bucket_pred" in active:
            result["time"]["bucket_accuracy"] = float(
                accuracy_score(active.time_bucket_true, active.time_bucket_pred)
            )
            result["time"]["bucket_macro_f1"] = float(
                f1_score(
                    active.time_bucket_true,
                    active.time_bucket_pred,
                    labels=list(range(4)),
                    average="macro",
                    zero_division=0,
                )
            )
    if "position_pred_x" in frame:
        active = frame[frame.position_mask].copy()
        distances = _distance(active)
        result["position"] = {
            "samples": len(active),
            "distance_mae_m": float(distances.mean()),
            "distance_median_m": float(np.median(distances)),
            "distance_rmse_m": float(np.sqrt(np.mean(distances**2))),
            "zone_accuracy": float(accuracy_score(active.zone_true, active.zone_pred)),
            "zone_macro_f1": float(
                f1_score(
                    active.zone_true,
                    active.zone_pred,
                    labels=list(range(20)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "by_zone": {},
            "by_event": {},
        }
        active["distance_m"] = distances
        for zone in range(20):
            selected = active[active.zone_true == zone]
            result["position"]["by_zone"][str(zone)] = {
                "samples": len(selected),
                "distance_mae_m": float(selected.distance_m.mean()),
            }
        for index, name in enumerate(RAW_EVENT_NAMES):
            selected = active[active.event_true == index]
            result["position"]["by_event"][name] = {
                "samples": len(selected),
                "distance_mae_m": float(selected.distance_m.mean()),
            }
    return result
