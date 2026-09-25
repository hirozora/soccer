from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.actor_data import collate_actor_hgt
from football_hgt_targets_v4.actor_training import actor_backbone_hash, actor_loss
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from football_hgt_targets_v4.model import ActorScaleHGT


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def samples(artifacts: ProtocolArtifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=1,
        selected_currents=selected,
    )
    return [dataset[0]]


def _batch(samples, artifacts, view):
    return next(iter(DataLoader(
        samples,
        batch_size=1,
        collate_fn=partial(
            collate_actor_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            context_view=view,
        ),
    )))


def test_actor_targets_are_view_invariant(samples, artifacts) -> None:
    batches = {view: _batch(samples, artifacts, view) for view in ("f80", "p1", "p2")}
    reference = batches["f80"]
    for view, batch in batches.items():
        assert batch["sample_ids"] == reference["sample_ids"]
        for field in ("team_actor", "player_local", "player_mask", "player_raw"):
            assert torch.equal(batch["targets"][field], reference["targets"][field]), (view, field)


def test_player_candidate_roster_is_identical_across_views(samples, artifacts) -> None:
    batches = {view: _batch(samples, artifacts, view) for view in ("f80", "p1", "p2")}
    reference = batches["f80"]["graph"]["player"]
    for view in ("p1", "p2"):
        candidate = batches[view]["graph"]["player"]
        assert torch.equal(candidate.ptr, reference.ptr)
        assert torch.equal(candidate.raw_id, reference.raw_id)
        assert torch.equal(candidate.vocab_index, reference.vocab_index)
    assert all(
        int(reference.ptr[row]) + int(batches["f80"]["targets"]["player_local"][row])
        < int(reference.ptr[row + 1])
        for row in range(1)
    )


@pytest.mark.parametrize("task", ["team", "player"])
@pytest.mark.parametrize("view", ["f80", "p1", "p2"])
def test_actor_forward_backward(samples, artifacts, task, view) -> None:
    torch.manual_seed(7)
    model = ActorScaleHGT(artifacts, task, dropout=0.0)
    batch = _batch(samples, artifacts, view)
    predictions = model(batch)
    loss = actor_loss(predictions, batch, task)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.possession_base.grad is not None
    head = model.team_actor_head if task == "team" else model.player_actor_scorer[-1]
    assert head.weight.grad is not None


def test_actor_backbone_initialization_is_task_and_view_independent(artifacts) -> None:
    torch.manual_seed(11)
    team = ActorScaleHGT(artifacts, "team", dropout=0.0)
    torch.manual_seed(11)
    player = ActorScaleHGT(artifacts, "player", dropout=0.0)
    assert actor_backbone_hash(team) == actor_backbone_hash(player)


def test_actor_checkpoint_round_trip(samples, artifacts, tmp_path) -> None:
    torch.manual_seed(13)
    source = ActorScaleHGT(artifacts, "team", dropout=0.0).eval()
    batch = _batch(samples, artifacts, "p1")
    with torch.no_grad():
        expected = source(batch)["team_logits"]
    path = tmp_path / "actor.pt"
    torch.save(source.state_dict(), path)
    restored = ActorScaleHGT(artifacts, "team", dropout=0.0).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        actual = restored(batch)["team_logits"]
    assert torch.equal(actual, expected)
