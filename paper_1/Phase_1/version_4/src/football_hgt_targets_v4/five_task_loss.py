"""Fixed five-task objective for the controlled view comparison."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from football_benchmark.protocol import ProtocolArtifacts

from .actor_training import player_cross_entropy
from .five_task_study import FIVE_TASK_LOSS_DIVISOR, FIVE_TASK_WEIGHTS
from .losses import event_loss, position_loss, time_loss


def five_task_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    artifacts: ProtocolArtifacts,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the preregistered objective and unweighted task components."""

    targets = batch["targets"]
    components = {
        "event": event_loss(
            predictions["event_logits"], targets["raw_event_10"], "ce", artifacts
        ),
        "time": time_loss(predictions, targets, "current_huber"),
        "position": position_loss(predictions, targets, "xy"),
        "team": F.cross_entropy(predictions["team_logits"], targets["team_actor"]),
        "player": player_cross_entropy(
            predictions["player_scores"],
            batch["graphs"]["f80"]["player"].ptr,
            targets["player_local"],
            targets["player_mask"].bool(),
        ),
    }
    total = sum(
        components[name] * FIVE_TASK_WEIGHTS[name] for name in components
    ) / FIVE_TASK_LOSS_DIVISOR
    return total, components

