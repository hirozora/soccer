from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from football_hgt_targets_v4.oracle_dependency import (
    OracleHead,
    candidate_batch,
    hard_true_team_frame,
    head_inputs,
    load_roster_team_map,
    player_state_condition,
    player_team_condition,
    within_match_donors,
)
from football_hgt_targets_v4.oracle_dependency_reporting import cluster_bootstrap


def cache() -> dict:
    counts = torch.tensor([3, 3, 3, 3])
    states = torch.arange(12 * 64, dtype=torch.float32).reshape(12, 64)
    return {
        "sample_ids": ["a", "b", "c", "d"],
        "seed": 20260715,
        "split": "validation",
        "match_ids": torch.tensor([1, 1, 2, 2]),
        "current_event_indices": torch.arange(4),
        "main_context": torch.zeros(4, 64),
        "player_context": torch.ones(4, 64),
        "candidate_states": states,
        "candidate_scores": torch.arange(12, dtype=torch.float32),
        "candidate_ptr": torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))),
        "candidate_raw": torch.tensor([10, 11, 12, 10, 11, 12, 20, 21, 22, 20, 21, 22]),
        "candidate_team_raw": torch.tensor([100, 100, 101, 100, 100, 101, 200, 201, 201, 200, 201, 201]),
        "candidate_team_valid": torch.ones(12, dtype=torch.bool),
        "player_local": torch.tensor([1, 2, 0, 1]),
        "player_raw": torch.tensor([11, 12, 20, 21]),
        "player_mask": torch.ones(4, dtype=torch.bool),
        "team_true": torch.tensor([1, 0, 1, 0]),
        "target_team_raw": torch.tensor([100, 101, 200, 201]),
        "team_mapping_valid": torch.ones(4, dtype=torch.bool),
        "event_true": torch.tensor([0, 1, 0, 1]),
        "position_true": torch.zeros(4, 2),
        "position_mask": torch.ones(4, dtype=torch.bool),
        "zone_true": torch.zeros(4, dtype=torch.long),
    }


def test_heads_are_parameter_matched_with_common_seed() -> None:
    for family in ("player_team", "position_player_state", "event_player_state"):
        torch.manual_seed(7); first = OracleHead(family)
        torch.manual_seed(7); second = OracleHead(family)
        assert sum(value.numel() for value in first.parameters()) == sum(value.numel() for value in second.parameters())
        for left, right in zip(first.state_dict().values(), second.state_dict().values()):
            assert torch.equal(left, right)


def test_true_player_state_gather_and_null() -> None:
    values = cache(); rows = torch.arange(4)
    oracle = player_state_condition(values, "oracle", rows)
    expected = values["candidate_states"][values["candidate_ptr"][:-1] + values["player_local"]]
    assert torch.equal(oracle, expected)
    assert torch.count_nonzero(player_state_condition(values, "null", rows)) == 0


def test_within_match_shuffle_is_deterministic_and_never_crosses_match() -> None:
    values = cache()
    first = within_match_donors(values, "player_state").clone()
    second = within_match_donors(values, "player_state").clone()
    assert torch.equal(first, second)
    assert torch.equal(values["match_ids"][first], values["match_ids"])
    assert not torch.equal(first, torch.arange(4))


def test_team_condition_recomputes_candidate_match() -> None:
    values = cache()
    oracle = player_team_condition(values, "oracle")
    assert oracle.shape == (12, 2)
    assert oracle[:3, 1].tolist() == [1.0, 1.0, 0.0]
    assert oracle[3:6, 1].tolist() == [0.0, 0.0, 1.0]
    assert torch.count_nonzero(player_team_condition(values, "null")) == 0


def test_candidate_batch_and_player_input_alignment() -> None:
    values = cache(); rows = torch.tensor([1, 3])
    candidates, owners, ptr = candidate_batch(values, rows)
    assert candidates.tolist() == [3, 4, 5, 9, 10, 11]
    assert owners.tolist() == [0, 0, 0, 1, 1, 1]
    assert ptr.tolist() == [0, 3, 6]
    inputs, actual_ptr = head_inputs(values, "player_team", "oracle", rows)
    assert inputs.shape == (6, 130)
    assert torch.equal(ptr, actual_ptr)


def test_hard_mask_uses_one_identical_eligible_set_and_preserves_target() -> None:
    values = cache()
    frame = hard_true_team_frame(values, values["candidate_scores"], "base")
    assert len(frame) == 4
    assert frame.sample_id.tolist() == values["sample_ids"]
    assert (frame.hard_candidate_count > 0).all()
    assert np.isfinite(frame.hard_rank).all()


def test_match_cluster_bootstrap_uses_shared_match_draws() -> None:
    reference = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"], "match_id": [1, 1, 2, 2],
        "seed": [1, 1, 1, 1], "player_mask": [True] * 4,
        "player_rank": [2, 2, 1, 1],
    })
    candidate = reference.copy(); candidate["player_rank"] = [1, 1, 1, 1]
    draws = np.array([[1, 1], [2, 0], [0, 2]], dtype=int)
    result = cluster_bootstrap(reference, candidate, "player_team", draws)
    assert result["resampling_unit"] == "match"
    assert result["replicates"] == 3
    assert result["improvement"] == 0.5


def test_england_roster_mapping_is_complete_and_unambiguous() -> None:
    mapping = load_roster_team_map()
    assert len(mapping) == 380
    assert all(players and all(player > 0 and team > 0 for player, team in players.items()) for players in mapping.values())
