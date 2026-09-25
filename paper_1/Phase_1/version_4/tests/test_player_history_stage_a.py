from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from football_benchmark.protocol import ProtocolArtifacts
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT
from football_hgt_targets_v4.model import build_partial_l2_model
from football_hgt_targets_v4.oracle_dependency import load_roster_team_map
from football_hgt_targets_v4.player_history_data import (
    _apply_condition,
    _history_for_sample,
    _team_derangement,
)
from football_hgt_targets_v4.player_history_training import (
    PlayerHistoryTrainingConfig,
    _load_source_model,
    _loader,
    _loss,
)


@pytest.fixture(scope="module")
def real_batch_and_sample():
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    config = PlayerHistoryTrainingConfig(
        "ph_hist_k5",
        Path("/tmp/player-history-test"),
        20260715,
        "cpu",
        batch_size=2,
        num_workers=0,
        max_validation_samples=2,
    )
    loader = _loader("validation", config, artifacts, False)
    return artifacts, loader.dataset[1], next(iter(loader))


def test_history_uses_no_future_events(real_batch_and_sample):
    _, sample, _ = real_batch_and_sample
    reference = _history_for_sample(sample)
    graph = dict(sample.graph)
    stores = dict(graph["node_stores"])
    event = dict(stores["event"])
    event["event_type_index"] = event["event_type_index"].clone()
    event["event_type_index"][sample.current_event_index + 1 :] = 9
    event["absolute_seconds"] = event["absolute_seconds"].clone()
    event["absolute_seconds"][sample.current_event_index + 1 :] += 10000
    stores["event"] = event
    graph["node_stores"] = stores
    changed = _history_for_sample(replace(sample, graph=graph))
    for field in ("event_type", "numeric", "sequence_mask", "lengths", "statistics"):
        assert torch.equal(getattr(reference, field), getattr(changed, field))


def test_null_is_canonical_no_history(real_batch_and_sample):
    _, sample, _ = real_batch_and_sample
    raw_ids = sample.graph["node_stores"]["player"]["raw_id"]
    value = _apply_condition(
        _history_for_sample(sample), "ph_null", sample.match_id, raw_ids, 20260815
    )
    assert not value.event_type.any()
    assert not value.numeric.any()
    assert not value.sequence_mask.any()
    assert not value.lengths.any()
    assert not value.statistics.any()
    assert not value.shuffle_eligible.any()


def test_within_team_derangement_is_fixed_and_has_no_self(real_batch_and_sample):
    _, sample, _ = real_batch_and_sample
    raw_ids = tuple(
        int(value) for value in sample.graph["node_stores"]["player"]["raw_id"]
    )
    first = _team_derangement(sample.match_id, raw_ids, 20260815)
    second = _team_derangement(sample.match_id, raw_ids, 20260815)
    assert first == second
    donors, eligible = first
    roster = load_roster_team_map()[sample.match_id]
    for index, enabled in enumerate(eligible):
        if enabled:
            assert donors[index] != index
            assert roster[raw_ids[index]] == roster[raw_ids[donors[index]]]


def test_candidate_history_aligns_with_player_ptr(real_batch_and_sample):
    _, _, batch = real_batch_and_sample
    count = int(batch["graphs"]["f80"]["player"].ptr[-1])
    assert batch["player_history"]["lengths"].numel() == count
    assert batch["player_history"]["statistics"].shape == (count, 23)
    assert batch["player_history"]["numeric"].shape == (count, 5, 7)


def test_zero_adapter_reproduces_partial_l2(real_batch_and_sample):
    artifacts, _, batch = real_batch_and_sample
    model, source = _load_source_model(artifacts, 20260715, torch.device("cpu"))
    baseline = build_partial_l2_model(artifacts)
    baseline.load_state_dict(source["model"])
    model.eval(); baseline.eval()
    with torch.no_grad():
        left = model(batch)["player_scores"]
        right = baseline(batch)["player_scores"]
    assert torch.max(torch.abs(left - right)).item() < 1e-6


def test_gradients_stay_out_of_shared_layer(real_batch_and_sample):
    artifacts, _, batch = real_batch_and_sample
    model, _ = _load_source_model(artifacts, 20260715, torch.device("cpu"))
    model.train()
    loss = _loss(model(batch), batch)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.convolutions[0].parameters())
    assert any(parameter.grad is not None for parameter in model.player_convolution.parameters())
    assert model.player_history_adapter[-1].weight.grad is not None
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.player_history_encoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    _loss(model(batch), batch).backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.player_history_encoder.parameters()
    )
