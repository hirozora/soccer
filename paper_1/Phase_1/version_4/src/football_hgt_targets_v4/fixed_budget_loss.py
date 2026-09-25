"""Active-task objective with invariant five-task loss scaling."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from football_benchmark.protocol import ProtocolArtifacts

from .five_task_loss import five_task_loss
from .five_task_study import FIVE_TASK_LOSS_DIVISOR, FIVE_TASK_WEIGHTS


def fixed_budget_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    artifacts: ProtocolArtifacts,
    active_tasks: Iterable[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    _, components = five_task_loss(predictions, batch, artifacts)
    active = tuple(active_tasks)
    unknown = set(active) - set(components)
    if unknown or not {"event", "time", "position"}.issubset(active):
        raise ValueError(f"Invalid active tasks: {active}")
    total = sum(components[name] * FIVE_TASK_WEIGHTS[name] for name in active) / FIVE_TASK_LOSS_DIVISOR
    core = sum(components[name] * FIVE_TASK_WEIGHTS[name] for name in ("event", "time", "position")) / FIVE_TASK_LOSS_DIVISOR
    return total, components, core
