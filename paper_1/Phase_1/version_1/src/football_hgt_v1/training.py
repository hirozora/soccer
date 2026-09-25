"""Training metrics and reproducibility helpers for Version 1."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .model import compute_multitask_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(confusion: torch.Tensor) -> float:
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    score = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return float(score.mean())


class MetricAccumulator:
    def __init__(self, num_event_types: int) -> None:
        self.num_event_types = num_event_types
        self.event_confusion = torch.zeros(
            (num_event_types, num_event_types), dtype=torch.long
        )
        self.side_correct = 0
        self.side_count = 0
        self.advantage_correct = 0
        self.advantage_count = 0
        self.player_correct = 0
        self.player_top3 = 0
        self.player_count = 0
        self.time_absolute_error = 0.0
        self.time_squared_error = 0.0
        self.position_absolute_error = 0.0
        self.position_squared_error = 0.0
        self.position_euclidean_error = 0.0
        self.position_count = 0
        self.loss_sum = 0.0
        self.sample_count = 0

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        batch: Any,
        loss: torch.Tensor,
    ) -> None:
        batch_size = int(batch.y_event_type.numel())
        event_prediction = predictions["event_logits"].argmax(dim=-1)
        flat = batch.y_event_type * self.num_event_types + event_prediction
        self.event_confusion += torch.bincount(
            flat.detach().cpu(), minlength=self.num_event_types**2
        ).reshape(self.num_event_types, self.num_event_types)

        predicted_seconds = torch.expm1(predictions["log_delta"]).clamp_min(0.0)
        time_error = predicted_seconds - batch.y_delta_seconds
        self.time_absolute_error += float(time_error.abs().sum())
        self.time_squared_error += float(time_error.square().sum())

        position_mask = batch.y_position_mask.bool()
        if bool(position_mask.any()):
            position_error = (
                predictions["position"][position_mask]
                - batch.y_position[position_mask]
            ) * 100.0
            self.position_absolute_error += float(position_error.abs().sum())
            self.position_squared_error += float(position_error.square().sum())
            self.position_euclidean_error += float(
                torch.linalg.vector_norm(position_error, dim=-1).sum()
            )
            self.position_count += int(position_mask.sum())

        side_prediction = predictions["side_logits"].argmax(dim=-1)
        self.side_correct += int((side_prediction == batch.y_side).sum())
        self.side_count += batch_size

        advantage_mask = batch.y_advantage_mask.bool()
        if bool(advantage_mask.any()):
            advantage_prediction = predictions["advantage_logits"].argmax(dim=-1)
            self.advantage_correct += int(
                (
                    advantage_prediction[advantage_mask]
                    == batch.y_advantage[advantage_mask]
                ).sum()
            )
            self.advantage_count += int(advantage_mask.sum())

        player_ptr = batch["player"].ptr
        for index in torch.nonzero(batch.y_player_mask, as_tuple=False).flatten().tolist():
            start = int(player_ptr[index])
            stop = int(player_ptr[index + 1])
            local_scores = predictions["player_scores"][start:stop]
            target = int(batch.y_player_local[index])
            ranking = torch.topk(local_scores, k=min(3, local_scores.numel())).indices
            self.player_correct += int(int(ranking[0]) == target)
            self.player_top3 += int(bool((ranking == target).any()))
            self.player_count += 1

        self.loss_sum += float(loss) * batch_size
        self.sample_count += batch_size

    def compute(self) -> dict[str, float]:
        count = max(self.sample_count, 1)
        position_count = max(self.position_count, 1)
        return {
            "loss": self.loss_sum / count,
            "event_accuracy": float(self.event_confusion.diag().sum()) / count,
            "event_macro_f1": macro_f1(self.event_confusion),
            "time_mae_seconds": self.time_absolute_error / count,
            "time_rmse_seconds": (self.time_squared_error / count) ** 0.5,
            "position_coordinate_mae": self.position_absolute_error
            / (2 * position_count),
            "position_coordinate_rmse": (
                self.position_squared_error / (2 * position_count)
            )
            ** 0.5,
            "position_euclidean_distance": self.position_euclidean_error
            / position_count,
            "side_accuracy": self.side_correct / max(self.side_count, 1),
            "player_accuracy": self.player_correct / max(self.player_count, 1),
            "player_top3_accuracy": self.player_top3 / max(self.player_count, 1),
            "advantage_accuracy": self.advantage_correct
            / max(self.advantage_count, 1),
            "num_samples": self.sample_count,
            "num_position_targets": self.position_count,
            "num_player_targets": self.player_count,
            "num_advantage_targets": self.advantage_count,
        }


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_event_types: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator(num_event_types)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            loss, _ = compute_multitask_loss(predictions, batch)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            metrics.update(predictions, batch, loss.detach())
    return metrics.compute()
