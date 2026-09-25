"""Pair-matched, dimensionless multitask objectives."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from .protocol import ProtocolArtifacts


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    safe_targets = targets.clamp_min(0)
    values = F.cross_entropy(
        logits, safe_targets, weight=weights.to(logits.device), reduction="none"
    )
    active_classes = int((weights > 0).sum())
    normalizer = math.log(max(active_classes, 2))
    return _masked_mean(values, mask) / normalizer


def stable_sqrt_class_weights(
    counts: torch.Tensor,
    max_ratio: float = 5.0,
) -> torch.Tensor:
    """Return bounded inverse-sqrt weights with sample-weighted mean one."""

    counts = counts.to(torch.float64)
    active = counts > 0
    result = torch.zeros_like(counts)
    if active.any():
        largest = counts[active].max()
        result[active] = torch.sqrt(largest / counts[active]).clamp(max=max_ratio)
        sample_mean = (result[active] * counts[active]).sum() / counts[active].sum()
        result[active] /= sample_mean
    return result.float()


def _normalized_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Normalize weighted CE by selected sample weights, as PyTorch mean does."""

    if not mask.any():
        return logits.sum() * 0.0
    safe_targets = targets.clamp_min(0)
    device_weights = weights.to(logits.device)
    values = F.cross_entropy(
        logits,
        safe_targets,
        weight=device_weights,
        reduction="none",
    )
    selected_weights = device_weights[safe_targets][mask]
    active_classes = int((weights > 0).sum())
    return (
        values[mask].sum()
        / selected_weights.sum().clamp_min(torch.finfo(values.dtype).eps)
        / math.log(max(active_classes, 2))
    )


def compute_benchmark_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    family: str,
    contract: str,
    artifacts: ProtocolArtifacts,
    unified_event_loss_mode: str = "legacy_balanced",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = batch["targets"]
    device = next(iter(predictions.values())).device
    losses: dict[str, torch.Tensor] = {}

    if contract == "unified_lem" and family == "unified_lem":
        event_mask = torch.ones_like(targets["fine_event_32"], dtype=torch.bool)
        if unified_event_loss_mode == "legacy_balanced":
            losses["event"] = _classification_loss(
                predictions["fine_event_logits"],
                targets["fine_event_32"],
                event_mask,
                artifacts.class_weights["fine32"],
            )
        else:
            if unified_event_loss_mode == "unweighted":
                event_weights = (artifacts.class_counts["fine32"] > 0).float()
            elif unified_event_loss_mode == "sqrt_capped":
                event_weights = stable_sqrt_class_weights(
                    artifacts.class_counts["fine32"]
                )
            else:
                raise ValueError(
                    f"Unknown Unified event loss mode: {unified_event_loss_mode}"
                )
            losses["event"] = _normalized_classification_loss(
                predictions["fine_event_logits"],
                targets["fine_event_32"],
                event_mask,
                event_weights,
            )
    else:
        event_target = (
            targets["raw_event_10"]
            if contract == "unified_lem"
            else targets["action_4"]
        )
        event_mask = (
            torch.ones_like(event_target, dtype=torch.bool)
            if contract == "unified_lem"
            else targets["action_4_mask"].bool()
        )
        weight_name = "raw10" if contract == "unified_lem" else "action4"
        losses["event"] = _classification_loss(
            predictions["event_logits"],
            event_target,
            event_mask,
            artifacts.class_weights[weight_name],
        )

    if contract != "seq2event":
        time_mask = targets["time_mask"].bool()
        if family == "unified_lem":
            time_target = torch.floor(targets["delta_seconds_60"]).long().clamp(0, 60)
            time_values = F.cross_entropy(
                predictions["time_logits"], time_target, reduction="none"
            ) / math.log(61)
            losses["time"] = _masked_mean(time_values, time_mask)
        else:
            time_values = F.smooth_l1_loss(
                predictions["time_seconds"] / 60.0,
                targets["delta_seconds_60"] / 60.0,
                reduction="none",
                beta=1.0 / 60.0,
            )
            losses["time"] = _masked_mean(time_values, time_mask)

    position_mask = targets["position_mask"].bool()
    if family == "unified_lem":
        x_target = torch.floor(targets["position_xy"][:, 0] * 100).long().clamp(0, 100)
        y_target = torch.floor(targets["position_xy"][:, 1] * 100).long().clamp(0, 100)
        x_values = F.cross_entropy(predictions["x_logits"], x_target, reduction="none")
        y_values = F.cross_entropy(predictions["y_logits"], y_target, reduction="none")
        losses["position"] = _masked_mean(
            (x_values + y_values) / (2.0 * math.log(101)), position_mask
        )
    elif family == "nmstpp":
        losses["position"] = _classification_loss(
            predictions["zone_logits"],
            targets["zone_20"],
            position_mask,
            artifacts.class_weights["zone20"],
        )
    else:
        position_values = F.smooth_l1_loss(
            predictions["position_xy"], targets["position_xy"], reduction="none"
        ).mean(dim=-1)
        losses["position"] = _masked_mean(position_values, position_mask)

    total = torch.stack(tuple(losses.values())).mean() if losses else torch.zeros((), device=device)
    return total, losses
