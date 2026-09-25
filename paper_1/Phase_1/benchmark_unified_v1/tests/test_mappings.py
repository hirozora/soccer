from __future__ import annotations

import torch

from football_benchmark.constants import ACTION_TO_INDEX, FINE_EVENT_TO_INDEX
from football_benchmark.mappings import (
    action4_label,
    build_fold_matrix,
    fold_fine_probabilities,
    position_to_zone,
    unified_fine_label,
    zones_to_centers,
)


def test_action4_mapping_uses_only_real_actions() -> None:
    assert action4_label(8, 85, set()) == (ACTION_TO_INDEX["pass"], True)
    assert action4_label(8, 80, set()) == (ACTION_TO_INDEX["cross"], True)
    assert action4_label(3, 30, set()) == (ACTION_TO_INDEX["cross"], True)
    assert action4_label(3, 35, set()) == (ACTION_TO_INDEX["shot"], True)
    assert action4_label(1, 11, {501}) == (ACTION_TO_INDEX["dribble"], True)
    assert action4_label(1, 11, {702}) == (-1, False)
    assert action4_label(5, 50, set()) == (-1, False)


def test_unified_mapping_follows_official_override_order() -> None:
    assert unified_fine_label(8, 84, set()) == FINE_EVENT_TO_INDEX["long_pass"]
    assert unified_fine_label(3, 32, set()) == FINE_EVENT_TO_INDEX["cross"]
    assert unified_fine_label(10, 100, {401}) == FINE_EVENT_TO_INDEX["left_foot_shot"]
    assert unified_fine_label(1, 11, {504}) == FINE_EVENT_TO_INDEX["dribble"]
    # Card overrides the underlying event type last.
    assert unified_fine_label(2, 20, {1701}) == FINE_EVENT_TO_INDEX["red_card"]


def test_fold_matrix_is_train_conditional_distribution() -> None:
    raw = torch.tensor([0, 0, 1, 1, 1])
    fine = torch.tensor([0, 0, 0, 1, 1])
    matrix, active = build_fold_matrix(raw, fine, num_raw=2)
    assert active[:2].tolist() == [True, True]
    assert torch.allclose(matrix[:, 0], torch.tensor([2 / 3, 1 / 3]))
    assert torch.allclose(matrix[:, 1], torch.tensor([0.0, 1.0]))
    probabilities = torch.zeros((1, 32))
    probabilities[0, 0] = 1.0
    folded = fold_fine_probabilities(probabilities, matrix)
    assert torch.allclose(folded[0], torch.tensor([2 / 3, 1 / 3]))


def test_zone_round_trip_at_official_centres() -> None:
    zones = torch.arange(20)
    centers = zones_to_centers(zones)
    assert torch.equal(position_to_zone(centers), zones)

