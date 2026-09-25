"""Training and diagnostics for the Version 3 residual graph MoE."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from football_hgt_v1.training import macro_f1

from .experts import BRANCH_NAMES, EXPERT_NAMES
from .model import TASK_NAMES, compute_residual_moe_loss


class ResidualMoEMetricAccumulator:
    def __init__(self, num_event_types: int) -> None:
        self.num_event_types = num_event_types
        shape = (num_event_types, num_event_types)
        self.event_confusion = torch.zeros(shape, dtype=torch.long)
        self.base_event_confusion = torch.zeros(shape, dtype=torch.long)
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
        self.loss_part_sums = {name: 0.0 for name in TASK_NAMES}
        self.sample_count = 0
        self.routing_probability_sum = torch.zeros(
            (len(TASK_NAMES), len(EXPERT_NAMES)), dtype=torch.float64
        )
        self.routing_argmax_count = torch.zeros_like(self.routing_probability_sum)
        self.routing_entropy_sum = torch.zeros(len(TASK_NAMES), dtype=torch.float64)
        self.routing_count = 0
        self.overlap_sum: torch.Tensor | None = None
        self.overlap_high_count: torch.Tensor | None = None
        self.overlap_count = 0
        self.view_size_sum = torch.zeros(len(BRANCH_NAMES), dtype=torch.float64)
        self.residual_norm_sum = torch.zeros(len(EXPERT_NAMES), dtype=torch.float64)

    def _update_confusion(
        self,
        confusion: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        prediction = logits.argmax(dim=-1)
        flat = targets * self.num_event_types + prediction
        confusion += torch.bincount(
            flat.detach().cpu(), minlength=self.num_event_types**2
        ).reshape(self.num_event_types, self.num_event_types)

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        graph: Any,
        overlap: torch.Tensor,
        view_sizes: torch.Tensor,
        batch_size: int,
        loss: torch.Tensor,
        loss_parts: dict[str, torch.Tensor],
    ) -> None:
        target = slice(0, batch_size)
        event_target = graph.y_event_type[target]
        self._update_confusion(
            self.event_confusion, predictions["event_logits"], event_target
        )
        self._update_confusion(
            self.base_event_confusion,
            predictions["base_event_logits"],
            event_target,
        )

        predicted_seconds = torch.expm1(predictions["log_delta"]).clamp_min(0.0)
        time_error = predicted_seconds - graph.y_delta_seconds[target]
        self.time_absolute_error += float(time_error.abs().sum())
        self.time_squared_error += float(time_error.square().sum())

        position_mask = graph.y_position_mask[target].bool()
        if bool(position_mask.any()):
            position_error = (
                predictions["position"][position_mask]
                - graph.y_position[target][position_mask]
            ) * 100.0
            self.position_absolute_error += float(position_error.abs().sum())
            self.position_squared_error += float(position_error.square().sum())
            self.position_euclidean_error += float(
                torch.linalg.vector_norm(position_error, dim=-1).sum()
            )
            self.position_count += int(position_mask.sum())

        side_prediction = predictions["side_logits"].argmax(dim=-1)
        self.side_correct += int((side_prediction == graph.y_side[target]).sum())
        self.side_count += batch_size

        advantage_mask = graph.y_advantage_mask[target].bool()
        if bool(advantage_mask.any()):
            advantage_prediction = predictions["advantage_logits"].argmax(dim=-1)
            self.advantage_correct += int(
                (
                    advantage_prediction[advantage_mask]
                    == graph.y_advantage[target][advantage_mask]
                ).sum()
            )
            self.advantage_count += int(advantage_mask.sum())

        player_ptr = graph["player"].ptr[: batch_size + 1]
        player_mask = graph.y_player_mask[target].bool()
        for index in torch.nonzero(player_mask, as_tuple=False).flatten().tolist():
            start = int(player_ptr[index])
            stop = int(player_ptr[index + 1])
            local_scores = predictions["player_scores"][start:stop]
            player_target = int(graph.y_player_local[index])
            ranking = torch.topk(
                local_scores, k=min(3, local_scores.numel())
            ).indices
            self.player_correct += int(int(ranking[0]) == player_target)
            self.player_top3 += int(bool((ranking == player_target).any()))
            self.player_count += 1

        routing = predictions["routing_weights"].detach().cpu().double()
        self.routing_probability_sum += routing.sum(dim=0)
        selected = F.one_hot(
            routing.argmax(dim=-1), num_classes=len(EXPERT_NAMES)
        ).double()
        self.routing_argmax_count += selected.sum(dim=0)
        entropy = -(routing * routing.clamp_min(1e-12).log()).sum(dim=-1)
        self.routing_entropy_sum += entropy.sum(dim=0)
        self.routing_count += batch_size
        residual_norm = torch.linalg.vector_norm(
            predictions["expert_residuals"].detach(), dim=-1
        )
        self.residual_norm_sum += residual_norm.sum(dim=0).cpu().double()
        self.view_size_sum += view_sizes.sum(dim=0).cpu().double()

        overlap_cpu = overlap.detach().cpu().double()
        if self.overlap_sum is None:
            self.overlap_sum = torch.zeros(overlap_cpu.shape[1], dtype=torch.float64)
            self.overlap_high_count = torch.zeros_like(self.overlap_sum)
        self.overlap_sum += overlap_cpu.sum(dim=0)
        self.overlap_high_count += (overlap_cpu >= 0.8).sum(dim=0)
        self.overlap_count += batch_size

        self.loss_sum += float(loss) * batch_size
        for name, value in loss_parts.items():
            self.loss_part_sums[name] += float(value) * batch_size
        self.sample_count += batch_size

    def compute(self, overlap_keys: tuple[str, ...]) -> dict[str, Any]:
        count = max(self.sample_count, 1)
        position_count = max(self.position_count, 1)
        routing_count = max(self.routing_count, 1)
        overlap_count = max(self.overlap_count, 1)
        probability = self.routing_probability_sum / routing_count
        frequency = self.routing_argmax_count / routing_count
        entropy = self.routing_entropy_sum / routing_count
        overlap_mean = (
            self.overlap_sum / overlap_count
            if self.overlap_sum is not None
            else torch.zeros(len(overlap_keys), dtype=torch.float64)
        )
        overlap_high = (
            self.overlap_high_count / overlap_count
            if self.overlap_high_count is not None
            else torch.zeros(len(overlap_keys), dtype=torch.float64)
        )
        event_macro_f1 = macro_f1(self.event_confusion)
        base_event_macro_f1 = macro_f1(self.base_event_confusion)
        return {
            "loss": self.loss_sum / count,
            "loss_parts": {
                name: value / count for name, value in self.loss_part_sums.items()
            },
            "event_accuracy": float(self.event_confusion.diag().sum()) / count,
            "event_macro_f1": event_macro_f1,
            "base_event_accuracy": float(self.base_event_confusion.diag().sum())
            / count,
            "base_event_macro_f1": base_event_macro_f1,
            "residual_event_macro_f1_gain": event_macro_f1 - base_event_macro_f1,
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
                    task: float(entropy[index])
                    for index, task in enumerate(TASK_NAMES)
                },
                "maximum_argmax_share": {
                    task: float(frequency[index].max())
                    for index, task in enumerate(TASK_NAMES)
                },
            },
            "mean_view_size": {
                name: float(self.view_size_sum[index] / count)
                for index, name in enumerate(BRANCH_NAMES)
            },
            "mean_residual_norm": {
                name: float(self.residual_norm_sum[index] / count)
                for index, name in enumerate(EXPERT_NAMES)
            },
            "overlap": {
                key: {
                    "mean_jaccard": float(overlap_mean[index]),
                    "fraction_at_least_0_8": float(overlap_high[index]),
                }
                for index, key in enumerate(overlap_keys)
            },
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
    overlap_keys: tuple[str, ...],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    metrics = ResidualMoEMetricAccumulator(num_event_types)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for payload in loader:
            graph = payload["graph"].to(device)
            overlap = payload["overlap"]
            view_sizes = payload["view_sizes"]
            batch_size = int(payload["batch_size"])
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(graph, batch_size)
            loss, loss_parts = compute_residual_moe_loss(
                predictions, graph, batch_size
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            metrics.update(
                predictions,
                graph,
                overlap,
                view_sizes,
                batch_size,
                loss.detach(),
                {name: value.detach() for name, value in loss_parts.items()},
            )
    return metrics.compute(overlap_keys)
