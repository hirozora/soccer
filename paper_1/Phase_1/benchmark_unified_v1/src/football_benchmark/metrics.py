"""Common prediction adapters, metrics, and per-sample output tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .constants import (
    ACTION_NAMES,
    FINE_EVENT_NAMES,
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
    RAW_EVENT_NAMES,
)
from .mappings import fold_fine_probabilities, position_to_zone, zones_to_centers
from .protocol import ProtocolArtifacts


def _cpu(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def evaluation_outputs(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    family: str,
    contract: str,
    artifacts: ProtocolArtifacts,
) -> dict[str, np.ndarray]:
    targets = batch["targets"]
    result: dict[str, np.ndarray] = {}

    if contract == "unified_lem":
        result["event_true"] = _cpu(targets["raw_event_10"])
        result["event_mask"] = np.ones(len(result["event_true"]), dtype=bool)
        if family == "unified_lem":
            fine_probabilities = predictions["fine_event_logits"].softmax(dim=-1)
            raw_probabilities = fold_fine_probabilities(
                fine_probabilities, artifacts.fold_matrix.to(fine_probabilities.device)
            )
            result["event_pred"] = _cpu(raw_probabilities.argmax(dim=-1))
            result["fine_true"] = _cpu(targets["fine_event_32"])
            result["fine_pred"] = _cpu(fine_probabilities.argmax(dim=-1))
        else:
            result["event_pred"] = _cpu(predictions["event_logits"].argmax(dim=-1))
    else:
        result["event_true"] = _cpu(targets["action_4"])
        result["event_pred"] = _cpu(predictions["event_logits"].argmax(dim=-1))
        result["event_mask"] = _cpu(targets["action_4_mask"].bool())

    result["position_true_x"] = _cpu(targets["position_xy"][:, 0])
    result["position_true_y"] = _cpu(targets["position_xy"][:, 1])
    result["position_mask"] = _cpu(targets["position_mask"].bool())
    result["zone_true"] = _cpu(targets["zone_20"])
    if family == "nmstpp":
        zone_pred = predictions["zone_logits"].argmax(dim=-1)
        position_pred = zones_to_centers(zone_pred)
    else:
        position_pred = predictions["position_xy"]
        zone_pred = position_to_zone(position_pred)
    result["zone_pred"] = _cpu(zone_pred)
    if contract == "nmstpp":
        # Quantize both sides to the same official zone granularity.
        equal_position = zones_to_centers(zone_pred)
        result["position_pred_x"] = _cpu(equal_position[:, 0])
        result["position_pred_y"] = _cpu(equal_position[:, 1])
        true_centers = zones_to_centers(targets["zone_20"].clamp_min(0))
        result["position_equal_true_x"] = _cpu(true_centers[:, 0])
        result["position_equal_true_y"] = _cpu(true_centers[:, 1])
        if family == "hgt":
            result["position_continuous_pred_x"] = _cpu(predictions["position_xy"][:, 0])
            result["position_continuous_pred_y"] = _cpu(predictions["position_xy"][:, 1])
    else:
        result["position_pred_x"] = _cpu(position_pred[:, 0])
        result["position_pred_y"] = _cpu(position_pred[:, 1])

    if contract != "seq2event":
        result["time_true"] = _cpu(targets["delta_seconds_60"])
        result["time_pred"] = _cpu(predictions["time_seconds"])
        result["time_mask"] = _cpu(targets["time_mask"].bool())
    return result


@dataclass
class MetricAccumulator:
    family: str
    contract: str
    artifacts: ProtocolArtifacts
    chunks: list[pd.DataFrame] = field(default_factory=list)
    loss_sum: float = 0.0
    sample_count: int = 0

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, Any],
        loss: torch.Tensor,
    ) -> None:
        outputs = evaluation_outputs(
            predictions, batch, self.family, self.contract, self.artifacts
        )
        frame = pd.DataFrame(outputs)
        frame.insert(0, "sample_id", batch["sample_ids"])
        frame.insert(1, "match_id", _cpu(batch["match_ids"]))
        self.chunks.append(frame)
        count = len(frame)
        self.loss_sum += float(loss.detach()) * count
        self.sample_count += count

    def frame(self) -> pd.DataFrame:
        if not self.chunks:
            return pd.DataFrame()
        return pd.concat(self.chunks, ignore_index=True)

    def compute(self) -> dict[str, Any]:
        return compute_metrics(self.frame(), self.contract, self.family, self.loss_sum, self.sample_count)


def _classification_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    labels: list[int],
    names: tuple[str, ...],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(true, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
    }


def _distance_meters(
    true_x: np.ndarray, true_y: np.ndarray, pred_x: np.ndarray, pred_y: np.ndarray
) -> np.ndarray:
    return np.sqrt(
        ((pred_x - true_x) * PITCH_LENGTH_METERS) ** 2
        + ((pred_y - true_y) * PITCH_WIDTH_METERS) ** 2
    )


def compute_metrics(
    frame: pd.DataFrame,
    contract: str,
    family: str,
    loss_sum: float = 0.0,
    sample_count: int | None = None,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot compute metrics from no predictions")
    event_frame = frame[frame.event_mask]
    event_names = RAW_EVENT_NAMES if contract == "unified_lem" else ACTION_NAMES
    metrics: dict[str, Any] = {
        "loss": float(loss_sum / max(sample_count or len(frame), 1)),
        "samples": int(len(frame)),
        "event_samples": int(len(event_frame)),
        "event": _classification_metrics(
            event_frame.event_true.to_numpy(),
            event_frame.event_pred.to_numpy(),
            list(range(len(event_names))),
            event_names,
        ),
    }
    position_frame = frame[frame.position_mask]
    metrics["position_samples"] = int(len(position_frame))
    if len(position_frame):
        true_x_column = (
            "position_equal_true_x" if contract == "nmstpp" else "position_true_x"
        )
        true_y_column = (
            "position_equal_true_y" if contract == "nmstpp" else "position_true_y"
        )
        distances = _distance_meters(
            position_frame[true_x_column].to_numpy(),
            position_frame[true_y_column].to_numpy(),
            position_frame.position_pred_x.to_numpy(),
            position_frame.position_pred_y.to_numpy(),
        )
        metrics["position"] = {
            "x_mae_100": float(
                np.abs(
                    position_frame.position_pred_x.to_numpy()
                    - position_frame[true_x_column].to_numpy()
                ).mean()
                * 100.0
            ),
            "y_mae_100": float(
                np.abs(
                    position_frame.position_pred_y.to_numpy()
                    - position_frame[true_y_column].to_numpy()
                ).mean()
                * 100.0
            ),
            "distance_mae_m": float(distances.mean()),
            "distance_rmse_m": float(np.sqrt(np.mean(distances**2))),
        }
        if contract == "nmstpp":
            metrics["position"]["zone_accuracy"] = float(
                accuracy_score(position_frame.zone_true, position_frame.zone_pred)
            )
            metrics["position"]["zone_macro_f1"] = float(
                f1_score(
                    position_frame.zone_true,
                    position_frame.zone_pred,
                    labels=list(range(20)),
                    average="macro",
                    zero_division=0,
                )
            )
            if family == "hgt":
                continuous_distances = _distance_meters(
                    position_frame.position_true_x.to_numpy(),
                    position_frame.position_true_y.to_numpy(),
                    position_frame.position_continuous_pred_x.to_numpy(),
                    position_frame.position_continuous_pred_y.to_numpy(),
                )
                metrics["position"]["hgt_continuous_distance_mae_m"] = float(
                    continuous_distances.mean()
                )

    if contract != "seq2event":
        time_frame = frame[frame.time_mask]
        errors = time_frame.time_pred.to_numpy() - time_frame.time_true.to_numpy()
        metrics["time_samples"] = int(len(time_frame))
        metrics["time"] = {
            "mae_seconds": float(np.abs(errors).mean()),
            "rmse_seconds": float(np.sqrt(np.mean(errors**2))),
        }
    else:
        metrics["time"] = None

    if family == "unified_lem" and "fine_true" in frame:
        metrics["fine32_diagnostic"] = _classification_metrics(
            frame.fine_true.to_numpy(),
            frame.fine_pred.to_numpy(),
            list(range(len(FINE_EVENT_NAMES))),
            FINE_EVENT_NAMES,
        )
    return metrics

