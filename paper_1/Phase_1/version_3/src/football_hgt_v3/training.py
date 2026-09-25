"""Training, evaluation, and diagnostics for Version 3."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from football_hgt_v1.training import macro_f1

from .experts import EXPERT_NAMES, OVERLAP_KEYS
from .model import TASK_NAMES, compute_multitask_loss


def move_payload(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "graph": payload["graph"].to(device),
        "selections": {
            name: {key: value.to(device) for key, value in selection.items()}
            for name, selection in payload["selections"].items()
        },
        "candidate_features": payload["candidate_features"].to(device),
        "overlap": payload["overlap"],
        "selection_sizes": payload["selection_sizes"],
    }


def _confusion_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "class_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(confusion[index].sum()),
            }
            for index in range(confusion.shape[0])
        ],
    }


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    squares: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if "residual_outputs" in name:
            group = "residual_output"
        elif "residual_stems" in name or "readouts" in name:
            group = "expert_stem"
        elif "router" in name or "embedding" in name:
            group = "router"
        elif "position_" in name:
            group = "position_correction"
        elif "player_delta" in name:
            group = "player_correction"
        else:
            group = "other"
        squares[group] += float(parameter.grad.detach().square().sum())
    return {name: value**0.5 for name, value in squares.items()}


class MetricAccumulator:
    def __init__(self, num_event_types: int) -> None:
        self.num_event_types = num_event_types
        self.event_confusion = torch.zeros((num_event_types, num_event_types), dtype=torch.long)
        self.base_event_confusion = torch.zeros_like(self.event_confusion)
        self.side_confusion = torch.zeros((2, 2), dtype=torch.long)
        self.advantage_confusion = torch.zeros((2, 2), dtype=torch.long)
        self.time_absolute = 0.0
        self.time_squared = 0.0
        self.position_absolute = 0.0
        self.position_squared = 0.0
        self.position_euclidean = 0.0
        self.position_count = 0
        self.player_correct = 0
        self.player_top3 = 0
        self.player_reciprocal_rank = 0.0
        self.player_count = 0
        self.loss_sum = 0.0
        self.loss_parts = {name: 0.0 for name in TASK_NAMES}
        self.sample_count = 0
        shape = (len(TASK_NAMES), len(EXPERT_NAMES))
        self.routing_probability = torch.zeros(shape, dtype=torch.float64)
        self.routing_argmax = torch.zeros(shape, dtype=torch.float64)
        self.routing_entropy = torch.zeros(len(TASK_NAMES), dtype=torch.float64)
        self.routing_count = torch.zeros(len(TASK_NAMES), dtype=torch.float64)
        self.residual_norm = torch.zeros(shape, dtype=torch.float64)
        self.weighted_residual_norm = torch.zeros(len(TASK_NAMES), dtype=torch.float64)
        self.overlap_sum = torch.zeros(len(OVERLAP_KEYS), dtype=torch.float64)
        self.selection_size_sum = torch.zeros(len(EXPERT_NAMES), dtype=torch.float64)
        self.structure_count = 0
        self.gradient_sums: dict[str, float] = defaultdict(float)
        self.gradient_batches = 0

    @staticmethod
    def _update_confusion(
        confusion: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        classes = confusion.shape[0]
        prediction = logits.argmax(dim=-1)
        flat = targets * classes + prediction
        confusion += torch.bincount(flat.detach().cpu(), minlength=classes**2).reshape(
            classes, classes
        )

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        graph: Any,
        overlap: torch.Tensor,
        selection_sizes: torch.Tensor,
        loss: torch.Tensor,
        loss_parts: dict[str, torch.Tensor],
        gradients: dict[str, float] | None,
    ) -> None:
        batch_size = int(graph.y_event_type.numel())
        self._update_confusion(self.event_confusion, predictions["event_logits"], graph.y_event_type)
        self._update_confusion(
            self.base_event_confusion, predictions["base_event_logits"], graph.y_event_type
        )
        self._update_confusion(self.side_confusion, predictions["side_logits"], graph.y_side)
        advantage_mask = graph.y_advantage_mask.bool()
        if bool(advantage_mask.any()):
            self._update_confusion(
                self.advantage_confusion,
                predictions["advantage_logits"][advantage_mask],
                graph.y_advantage[advantage_mask],
            )

        seconds = torch.expm1(predictions["log_delta"]).clamp_min(0.0)
        time_error = seconds - graph.y_delta_seconds
        self.time_absolute += float(time_error.abs().sum())
        self.time_squared += float(time_error.square().sum())

        position_mask = graph.y_position_mask.bool()
        if bool(position_mask.any()):
            error = (predictions["position"][position_mask] - graph.y_position[position_mask]) * 100.0
            self.position_absolute += float(error.abs().sum())
            self.position_squared += float(error.square().sum())
            self.position_euclidean += float(torch.linalg.vector_norm(error, dim=-1).sum())
            self.position_count += int(position_mask.sum())

        player_mask = graph.y_player_mask.bool()
        player_ptr = graph["player"].ptr
        for index in torch.nonzero(player_mask, as_tuple=False).flatten().tolist():
            start = int(player_ptr[index])
            stop = int(player_ptr[index + 1])
            scores = predictions["player_scores"][start:stop]
            target = int(graph.y_player_local[index])
            ranking = torch.argsort(scores, descending=True)
            rank = int(torch.nonzero(ranking == target, as_tuple=False)[0]) + 1
            self.player_correct += int(rank == 1)
            self.player_top3 += int(rank <= 3)
            self.player_reciprocal_rank += 1.0 / rank
            self.player_count += 1

        task_masks = (
            torch.ones(batch_size, dtype=torch.bool, device=graph.y_event_type.device),
            torch.ones(batch_size, dtype=torch.bool, device=graph.y_event_type.device),
            position_mask,
            torch.ones(batch_size, dtype=torch.bool, device=graph.y_event_type.device),
            player_mask,
            advantage_mask,
        )
        routing = predictions["routing_weights"].detach()
        residuals = predictions["task_residuals"].detach()
        weighted = torch.einsum("bqm,bqmh->bqh", routing, residuals)
        for task_index, mask in enumerate(task_masks):
            count = int(mask.sum())
            if count == 0:
                continue
            selected = routing[mask, task_index]
            self.routing_probability[task_index] += selected.sum(dim=0).cpu().double()
            self.routing_argmax[task_index] += F.one_hot(
                selected.argmax(dim=-1), num_classes=len(EXPERT_NAMES)
            ).sum(dim=0).cpu().double()
            entropy = -(selected * selected.clamp_min(1e-12).log()).sum(dim=-1)
            self.routing_entropy[task_index] += float(entropy.sum())
            self.routing_count[task_index] += count
            norms = torch.linalg.vector_norm(residuals[mask, task_index], dim=-1)
            self.residual_norm[task_index] += norms.sum(dim=0).cpu().double()
            self.weighted_residual_norm[task_index] += float(
                torch.linalg.vector_norm(weighted[mask, task_index], dim=-1).sum()
            )

        self.overlap_sum += overlap.sum(dim=0).double()
        self.selection_size_sum += selection_sizes.sum(dim=0).double()
        self.structure_count += batch_size
        self.loss_sum += float(loss) * batch_size
        for name, value in loss_parts.items():
            self.loss_parts[name] += float(value) * batch_size
        self.sample_count += batch_size
        if gradients is not None:
            for name, value in gradients.items():
                self.gradient_sums[name] += value
            self.gradient_batches += 1

    def compute(self) -> dict[str, Any]:
        count = max(self.sample_count, 1)
        position_count = max(self.position_count, 1)
        event = _confusion_metrics(self.event_confusion)
        base_event = _confusion_metrics(self.base_event_confusion)
        side = _confusion_metrics(self.side_confusion)
        advantage = _confusion_metrics(self.advantage_confusion)
        routing_count = self.routing_count.clamp_min(1.0)
        probability = self.routing_probability / routing_count[:, None]
        frequency = self.routing_argmax / routing_count[:, None]
        residual = self.residual_norm / routing_count[:, None]
        weighted = self.weighted_residual_norm / routing_count
        return {
            "loss": self.loss_sum / count,
            "loss_parts": {name: value / count for name, value in self.loss_parts.items()},
            "event_accuracy": float(self.event_confusion.diag().sum()) / count,
            "event_macro_f1": event["macro_f1"],
            "event_per_class": event["per_class"],
            "base_event_accuracy": float(self.base_event_confusion.diag().sum()) / count,
            "base_event_macro_f1": base_event["macro_f1"],
            "residual_event_macro_f1_gain": event["macro_f1"] - base_event["macro_f1"],
            "time_mae_seconds": self.time_absolute / count,
            "time_rmse_seconds": (self.time_squared / count) ** 0.5,
            "position_coordinate_mae": self.position_absolute / (2 * position_count),
            "position_coordinate_rmse": (self.position_squared / (2 * position_count)) ** 0.5,
            "position_euclidean_distance": self.position_euclidean / position_count,
            "side_accuracy": float(self.side_confusion.diag().sum()) / count,
            "side_macro_f1": side["macro_f1"],
            "player_accuracy": self.player_correct / max(self.player_count, 1),
            "player_top3_accuracy": self.player_top3 / max(self.player_count, 1),
            "player_mrr": self.player_reciprocal_rank / max(self.player_count, 1),
            "advantage_accuracy": float(self.advantage_confusion.diag().sum())
            / max(int(self.advantage_confusion.sum()), 1),
            "advantage_macro_f1": advantage["macro_f1"],
            "routing": {
                "mean_probability": {
                    task: {
                        expert: float(probability[task_index, expert_index])
                        for expert_index, expert in enumerate(EXPERT_NAMES)
                    }
                    for task_index, task in enumerate(TASK_NAMES)
                },
                "argmax_frequency": {
                    task: {
                        expert: float(frequency[task_index, expert_index])
                        for expert_index, expert in enumerate(EXPERT_NAMES)
                    }
                    for task_index, task in enumerate(TASK_NAMES)
                },
                "entropy": {
                    task: float(self.routing_entropy[index] / routing_count[index])
                    for index, task in enumerate(TASK_NAMES)
                },
            },
            "mean_residual_norm": {
                task: {
                    expert: float(residual[task_index, expert_index])
                    for expert_index, expert in enumerate(EXPERT_NAMES)
                }
                for task_index, task in enumerate(TASK_NAMES)
            },
            "mean_weighted_residual_norm": {
                task: float(weighted[index]) for index, task in enumerate(TASK_NAMES)
            },
            "mean_selection_size": {
                expert: float(self.selection_size_sum[index] / max(self.structure_count, 1))
                for index, expert in enumerate(EXPERT_NAMES)
            },
            "overlap": {
                key: float(self.overlap_sum[index] / max(self.structure_count, 1))
                for index, key in enumerate(OVERLAP_KEYS)
            },
            "gradient_norms": {
                name: value / max(self.gradient_batches, 1)
                for name, value in self.gradient_sums.items()
            },
            "num_samples": self.sample_count,
            "num_position_targets": self.position_count,
            "num_player_targets": self.player_count,
            "num_advantage_targets": int(self.advantage_confusion.sum()),
        }


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_event_types: int,
    optimizer: torch.optim.Optimizer | None = None,
    disabled_expert: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator(num_event_types)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for raw_payload in loader:
            payload = move_payload(raw_payload, device)
            graph = payload["graph"]
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(
                graph,
                payload["selections"],
                payload["candidate_features"],
                disabled_expert=disabled_expert,
            )
            loss, loss_parts = compute_multitask_loss(predictions, graph)
            gradients = None
            if training:
                loss.backward()
                gradients = _gradient_norms(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            with torch.no_grad():
                metrics.update(
                    predictions,
                    graph,
                    payload["overlap"],
                    payload["selection_sizes"],
                    loss.detach(),
                    {name: value.detach() for name, value in loss_parts.items()},
                    gradients,
                )
            del predictions, loss, loss_parts, payload, graph
    return metrics.compute()
