from __future__ import annotations

import torch

from football_hgt_targets_v4.player_posterior import (
    PosteriorProbeHead,
    _expected_states,
    condition_values,
    head_inputs,
)
from football_hgt_targets_v4.training import set_seed


def candidate_cache() -> dict:
    states = torch.zeros((4, 64))
    states[0, 0], states[1, 1], states[2, 2], states[3, 3] = 1, 1, 1, 1
    return {
        "sample_ids": ["1:0", "1:1"],
        "candidate_ptr": torch.tensor([0, 2, 4]),
        "candidate_scores": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "candidate_states": states,
        "candidate_team_raw": torch.tensor([10, 11, 10, -1]),
        "candidate_team_valid": torch.tensor([True, True, True, False]),
        "player_local": torch.tensor([0, 1]),
    }


def test_expected_states_are_segmented_weighted_sums_and_unmapped_is_neutral() -> None:
    cache = candidate_cache()
    logits = torch.log(torch.tensor([[0.2, 0.8], [0.7, 0.3]]))
    values = _expected_states(cache, logits, torch.tensor([10, 10]))
    assert torch.allclose(values["base_post_condition"][0, :2], torch.tensor([0.5, 0.5]))
    assert values["team_post_condition"][0, 0] > values["team_post_condition"][0, 1]
    base_second = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    assert torch.allclose(
        values["base_post_condition"][1, 2:4], base_second, atol=1e-6
    )
    assert torch.isfinite(values["team_post_condition"]).all()


def test_null_condition_is_canonical_zero_and_has_same_shape() -> None:
    cache = {
        "main_context": torch.randn(3, 64),
        "base_post_condition": torch.randn(3, 64),
        "team_post_condition": torch.randn(3, 64),
    }
    rows = torch.tensor([0, 2])
    null = condition_values(cache, "null", rows)
    assert null.shape == (2, 64)
    assert torch.count_nonzero(null) == 0
    assert head_inputs(cache, "null", rows).shape == (2, 128)


def test_all_conditions_use_identically_initialized_capacity() -> None:
    hashes = []
    for _condition in ("null", "base_post", "team_post"):
        set_seed(20260715)
        model = PosteriorProbeHead("event")
        hashes.append([value.detach().clone() for value in model.state_dict().values()])
    for left, right in zip(hashes[0], hashes[1]):
        assert torch.equal(left, right)
    for left, right in zip(hashes[0], hashes[2]):
        assert torch.equal(left, right)


def test_posterior_probe_backward_only_updates_probe() -> None:
    set_seed(20260715)
    model = PosteriorProbeHead("position")
    context = torch.randn(4, 64, requires_grad=False)
    condition = torch.randn(4, 64, requires_grad=False)
    output = model(torch.cat((context, condition), dim=-1))
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert context.grad is None
    assert condition.grad is None

