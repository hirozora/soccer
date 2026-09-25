from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from football_benchmark.constants import ZONE_CENTERS_100
from football_benchmark.protocol import ProtocolArtifacts
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT
from football_hgt_targets_v4.losses import (
    event_loss,
    position_loss,
    sqrt_capped_weights,
    time_loss,
)
from football_hgt_targets_v4.model import (
    decode_bucket_offsets,
    decode_zone_residuals,
    time_bucket_ids,
)


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


def test_time_bucket_boundaries_are_left_closed() -> None:
    seconds = torch.tensor([0.0, 1.9999, 2.0, 4.9999, 5.0, 14.9999, 15.0, 60.0])
    assert time_bucket_ids(seconds).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_bucket_offset_decode_stays_inside_selected_bucket() -> None:
    buckets = torch.tensor([0, 1, 2, 3])
    offsets = torch.zeros((4, 4))
    decoded = decode_bucket_offsets(buckets, offsets)
    assert torch.allclose(decoded, torch.tensor([1.0, 3.5, 10.0, 37.5]))


def test_sqrt_weights_are_finite_and_bounded() -> None:
    weights = sqrt_capped_weights(torch.tensor([10000, 100, 1, 0]))
    active = weights[weights > 0]
    assert torch.isfinite(weights).all()
    assert float(active.max() / active.min()) <= 5.0 + 1e-6
    assert float(weights[-1]) == 0.0


def test_balanced_softmax_matches_definition(artifacts: ProtocolArtifacts) -> None:
    torch.manual_seed(3)
    logits = torch.randn(7, 10)
    targets = torch.tensor([0, 1, 2, 3, 4, 7, 9])
    expected = F.cross_entropy(
        logits + artifacts.class_counts["raw10"].clamp_min(1).log(), targets
    ) / math.log(10)
    assert torch.allclose(
        event_loss(logits, targets, "balanced_softmax", artifacts), expected
    )


def test_legacy_losses_are_exact(artifacts: ProtocolArtifacts) -> None:
    torch.manual_seed(7)
    event_logits = torch.randn(6, 10)
    event_target = torch.tensor([0, 1, 2, 4, 7, 9])
    expected_event = F.cross_entropy(
        event_logits,
        event_target,
        weight=artifacts.class_weights["raw10"],
        reduction="none",
    ).mean() / math.log(10)
    assert torch.allclose(
        event_loss(event_logits, event_target, "inverse_ce", artifacts),
        expected_event,
        atol=0,
        rtol=0,
    )

    seconds = torch.tensor([0.0, 1.5, 5.0, 17.0, 60.0, 3.0])
    mask = torch.tensor([True, True, False, True, True, False])
    predicted_seconds = torch.tensor([0.4, 2.0, 1.0, 20.0, 49.0, 3.0])
    predictions = {
        "time_seconds": predicted_seconds,
        "log_delta": torch.log1p(predicted_seconds),
    }
    targets = {"delta_seconds_60": seconds, "time_mask": mask}
    expected_time = F.smooth_l1_loss(
        predicted_seconds / 60.0,
        seconds / 60.0,
        reduction="none",
        beta=1.0 / 60.0,
    )[mask].mean()
    assert torch.allclose(
        time_loss(predictions, targets, "current_huber"),
        expected_time,
        atol=0,
        rtol=0,
    )

    predicted_xy = torch.tensor([[0.2, 0.4], [0.7, 0.3], [0.8, 0.9]])
    target_xy = torch.tensor([[0.1, 0.5], [0.8, 0.1], [0.4, 0.2]])
    position_mask = torch.tensor([True, False, True])
    position_targets = {
        "position_xy": target_xy,
        "position_mask": position_mask,
        "zone_20": torch.tensor([1, -1, 4]),
    }
    expected_position = F.smooth_l1_loss(
        predicted_xy, target_xy, reduction="none"
    ).mean(-1)[position_mask].mean()
    assert torch.allclose(
        position_loss({"position_xy": predicted_xy}, position_targets, "xy"),
        expected_position,
        atol=0,
        rtol=0,
    )


def test_zone_residual_decode_uses_predicted_zone_and_clips() -> None:
    logits = torch.full((2, 20), -5.0)
    logits[0, 3] = 5.0
    logits[1, 19] = 5.0
    residuals = torch.zeros((2, 20, 2))
    residuals[0, 3] = torch.tensor([100.0, -100.0])
    position, zones = decode_zone_residuals(logits, residuals)
    centers = torch.tensor(ZONE_CENTERS_100) / 100.0
    expected = (centers[3] + torch.tensor([0.25, -0.25])).clamp(0, 1)
    assert zones.tolist() == [3, 19]
    assert torch.allclose(position[0], expected)
    assert bool(((position >= 0) & (position <= 1)).all())

