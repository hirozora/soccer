"""Five-task shared-backbone gradient diagnostics."""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch

from football_benchmark.protocol import ProtocolArtifacts

from .five_task_loss import five_task_loss
from .five_task_study import FIVE_TASK_NAMES, FIVE_TASK_WEIGHTS
from .model import FiveTaskViewHGT


FIVE_TASK_HEAD_PREFIXES = (
    "event_head.",
    "time_head.",
    "position_head.",
    "team_actor_head.",
    "player_actor_scorer.",
)


def _shared_parameters(model: FiveTaskViewHGT) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(FIVE_TASK_HEAD_PREFIXES)
        and not name.startswith("fusion_logits.")
    ]


def _dot(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    values = [
        (a * b).sum()
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if not values:
        raise RuntimeError("No shared gradients were produced")
    return torch.stack(values).sum()


def five_task_gradient_diagnostics(
    model: FiveTaskViewHGT,
    batch: dict[str, Any],
    artifacts: ProtocolArtifacts,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    _, components = five_task_loss(model(batch), batch, artifacts)
    parameters = _shared_parameters(model)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for index, name in enumerate(FIVE_TASK_NAMES):
        gradients[name] = torch.autograd.grad(
            components[name],
            parameters,
            retain_graph=index + 1 < len(FIVE_TASK_NAMES),
            allow_unused=True,
        )
    norms = {
        name: float(torch.sqrt(_dot(values, values)).detach())
        for name, values in gradients.items()
    }
    cosines: dict[str, float] = {}
    for left, right in itertools.combinations(FIVE_TASK_NAMES, 2):
        denominator = max(norms[left] * norms[right], torch.finfo(torch.float32).eps)
        value = float((_dot(gradients[left], gradients[right]) / denominator).detach())
        if not math.isfinite(value):
            raise RuntimeError("Non-finite five-task gradient cosine")
        cosines[f"{left}_{right}"] = max(-1.0, min(1.0, value))
    result = {
        "sample_ids": list(batch["sample_ids"]),
        "raw_loss": {name: float(value.detach()) for name, value in components.items()},
        "weights": dict(FIVE_TASK_WEIGHTS),
        "raw_gradient_norm": norms,
        "effective_gradient_norm": {
            name: norms[name] * FIVE_TASK_WEIGHTS[name] for name in norms
        },
        "gradient_cosine": cosines,
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return result

