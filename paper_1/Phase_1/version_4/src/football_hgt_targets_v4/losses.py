"""Losses for controlled target-formulation experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from football_benchmark.protocol import ProtocolArtifacts

from .model import time_bucket_ids, zone_centers


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0


def sqrt_capped_weights(counts: torch.Tensor, max_ratio: float = 5.0) -> torch.Tensor:
    counts = counts.double()
    active = counts > 0
    result = torch.zeros_like(counts)
    largest = counts[active].max()
    result[active] = torch.sqrt(largest / counts[active]).clamp(max=max_ratio)
    result[active] /= result[active].mean()
    return result.float()


def event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    artifacts: ProtocolArtifacts,
) -> torch.Tensor:
    if method == "ce":
        return F.cross_entropy(logits, targets) / math.log(10)
    if method == "inverse_ce":
        values = F.cross_entropy(
            logits,
            targets,
            weight=artifacts.class_weights["raw10"].to(logits.device),
            reduction="none",
        )
        return values.mean() / math.log(10)
    if method == "sqrt_capped_ce":
        weights = sqrt_capped_weights(artifacts.class_counts["raw10"]).to(logits.device)
        return F.cross_entropy(logits, targets, weight=weights) / math.log(10)
    if method == "balanced_softmax":
        counts = artifacts.class_counts["raw10"].to(logits.device, logits.dtype)
        return F.cross_entropy(logits + counts.clamp_min(1).log(), targets) / math.log(10)
    raise ValueError(f"Unknown Event method {method!r}")


def time_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    method: str,
) -> torch.Tensor:
    mask = targets["time_mask"].bool()
    seconds = targets["delta_seconds_60"]
    if method == "current_huber":
        values = F.smooth_l1_loss(
            predictions["time_seconds"] / 60.0,
            seconds / 60.0,
            reduction="none",
            beta=1.0 / 60.0,
        )
        return masked_mean(values, mask)
    if method == "log1p_huber":
        values = F.huber_loss(
            predictions["log_delta"], torch.log1p(seconds), reduction="none"
        )
        return masked_mean(values, mask)
    if method == "bucket_offset":
        bucket = time_bucket_ids(seconds)
        class_values = F.cross_entropy(
            predictions["time_bucket_logits"], bucket, reduction="none"
        ) / math.log(4)
        bounds = torch.tensor((0.0, 2.0, 5.0, 15.0, 60.0), device=seconds.device)
        low, high = bounds[bucket], bounds[bucket + 1]
        target_fraction = (seconds - low) / (high - low).clamp_min(1e-6)
        rows = torch.arange(bucket.shape[0], device=bucket.device)
        predicted_fraction = torch.sigmoid(
            predictions["time_raw_offsets"][rows, bucket]
        )
        offset_values = F.huber_loss(
            predicted_fraction, target_fraction, reduction="none"
        )
        return 0.5 * (
            masked_mean(class_values, mask) + masked_mean(offset_values, mask)
        )
    raise ValueError(f"Unknown Time method {method!r}")


def position_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    method: str,
) -> torch.Tensor:
    mask = targets["position_mask"].bool()
    position = targets["position_xy"]
    if method == "xy":
        values = F.smooth_l1_loss(
            predictions["position_xy"], position, reduction="none"
        ).mean(dim=-1)
        return masked_mean(values, mask)
    zone = targets["zone_20"].clamp_min(0)
    class_values = F.cross_entropy(predictions["zone_logits"], zone, reduction="none")
    if method == "zone":
        return masked_mean(class_values, mask) / math.log(20)
    if method == "zone_residual":
        rows = torch.arange(zone.shape[0], device=zone.device)
        predicted = 0.25 * torch.tanh(
            predictions["zone_raw_residuals"][rows, zone]
        )
        target = position - zone_centers(position.device, position.dtype)[zone]
        residual_values = F.huber_loss(
            predicted / 0.25, target / 0.25, reduction="none"
        ).mean(dim=-1)
        return 0.5 * (
            masked_mean(class_values, mask) / math.log(20)
            + masked_mean(residual_values, mask)
        )
    raise ValueError(f"Unknown Position method {method!r}")


def compute_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    task: str,
    method: str,
    artifacts: ProtocolArtifacts,
    joint_methods: dict[str, str] | None = None,
    joint_loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = batch["targets"]
    methods = joint_methods if task == "joint" else {task: method}
    components: dict[str, torch.Tensor] = {}
    if "event" in methods:
        components["event"] = event_loss(
            predictions["event_logits"],
            targets["raw_event_10"],
            methods["event"],
            artifacts,
        )
    if "time" in methods:
        components["time"] = time_loss(predictions, targets, methods["time"])
    if "position" in methods:
        components["position"] = position_loss(
            predictions, targets, methods["position"]
        )
    if task != "joint":
        if joint_loss_weights is not None:
            raise ValueError("Joint loss weights are only valid for joint training")
        return torch.stack(tuple(components.values())).mean(), components

    weights = joint_loss_weights or {name: 1.0 for name in components}
    if set(weights) != set(components):
        raise ValueError("Joint loss weights must define event, time, and position")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in weights.values()):
        raise ValueError("Joint loss weights must be finite and positive")
    weighted = torch.stack(
        tuple(components[name] * float(weights[name]) for name in components)
    )
    return weighted.sum() / len(components), components
