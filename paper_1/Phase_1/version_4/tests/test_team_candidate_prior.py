from __future__ import annotations

import math

import torch

from football_hgt_targets_v4.team_candidate_prior import (
    centered_log_prior,
    prediction_frame,
)
from football_hgt_targets_v4.team_candidate_prior_reporting import _fold_map


def cache() -> dict:
    return {
        "seed": 20260715,
        "sample_ids": ["1:0", "2:0"],
        "match_ids": torch.tensor([1, 2]),
        "current_event_indices": torch.tensor([0, 0]),
        "event_true": torch.tensor([7, 2]),
        "team_true": torch.tensor([0, 1]),
        "player_local": torch.tensor([2, 0]),
        "player_mask": torch.tensor([True, True]),
        "team_logits": torch.log(torch.tensor([[0.2, 0.8], [0.3, 0.7]])),
        "candidate_ptr": torch.tensor([0, 3, 6]),
        "candidate_scores": torch.tensor([2.0, 1.0, 0.0, 0.0, 2.0, 1.0]),
        "candidate_raw": torch.tensor([1, 2, 3, 4, 5, 6]),
        "candidate_team_raw": torch.tensor([10, 10, 11, 20, -1, 21]),
        "candidate_team_valid": torch.tensor([True, True, True, True, False, True]),
        "candidate_same_anchor": torch.tensor([True, True, False, True, False, False]),
        "anchor_team_raw": torch.tensor([10, 20]),
        "event_role": torch.tensor([1, 4]),
        "control_state": torch.tensor([2, 3]),
        "switch_confirmed": torch.tensor([False, True]),
    }


def test_centered_prior_is_neutral_for_unmapped_candidates() -> None:
    values = centered_log_prior(cache())
    assert torch.isclose(values[0], torch.tensor(math.log(0.8) - math.log(0.5)))
    assert torch.isclose(values[2], torch.tensor(math.log(0.2) - math.log(0.5)))
    assert float(values[4]) == 0.0


def test_lambda_zero_strictly_reproduces_base_ranks() -> None:
    base = prediction_frame(cache(), "base")
    soft = prediction_frame(cache(), "soft", lambda_value=0.0)
    assert base.player_rank.tolist() == soft.player_rank.tolist()
    assert base.reciprocal_rank.tolist() == soft.reciprocal_rank.tolist()


def test_hard_wrong_team_counts_as_failure_and_incomplete_mapping_falls_back() -> None:
    hard = prediction_frame(cache(), "hard")
    first = hard.iloc[0]
    assert first.hard_applied
    assert first.target_excluded
    assert first.player_rank == 4
    assert first.reciprocal_rank == 0.0
    second = hard.iloc[1]
    assert not second.hard_applied
    assert not second.target_excluded
    assert second.player_rank == 3


def test_structural_unknown_candidate_does_not_disable_hard_mask() -> None:
    values = cache()
    values["candidate_raw"][4] = 0
    hard = prediction_frame(values, "hard")
    second = hard.iloc[1]
    assert second.hard_applied
    assert not second.target_excluded
    assert second.player_rank == 1


def test_match_folds_are_deterministic_balanced_and_disjoint() -> None:
    matches = set(range(57))
    left, right = _fold_map(matches), _fold_map(matches)
    assert left == right
    assert set(left) == matches
    counts = [list(left.values()).count(fold) for fold in range(5)]
    assert max(counts) - min(counts) <= 1
