from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from football_hgt_targets_v4.five_task_data import collate_five_task_hgt
from football_hgt_targets_v4.five_task_diagnostics import five_task_gradient_diagnostics
from football_hgt_targets_v4.five_task_loss import five_task_loss
from football_hgt_targets_v4.five_task_study import FIVE_TASK_LOSS_DIVISOR, FIVE_TASK_WEIGHTS
from football_hgt_targets_v4.five_task_training import five_task_common_initialization_hash
from football_hgt_targets_v4.model import build_five_task_model


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


def _batch(samples, artifacts, views=("f80", "p1", "p2")):
    return next(iter(DataLoader(
        samples,
        batch_size=1,
        collate_fn=partial(
            collate_five_task_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            context_views=views,
        ),
    )))


def test_five_task_batch_aligns_views_targets_and_roster(samples, artifacts) -> None:
    batch = _batch(samples, artifacts)
    assert set(batch["graphs"]) == {"f80", "p1", "p2"}
    assert len(batch["sample_ids"]) == 1
    for field in ("raw_event_10", "delta_seconds_60", "position_xy", "team_actor", "player_local", "player_mask"):
        assert batch["targets"][field].shape[0] == 1
    reference = batch["graphs"]["f80"]["player"]
    for view in ("p1", "p2"):
        player = batch["graphs"][view]["player"]
        assert torch.equal(player.ptr, reference.ptr)
        assert torch.equal(player.raw_id, reference.raw_id)
    assert int(batch["targets"]["player_local"][0]) < int(reference.ptr[1] - reference.ptr[0])


def test_models_have_one_shared_hgt_and_only_soft_adds_six_scalars(artifacts) -> None:
    models = {}
    for mode in ("five_f80", "five_hard", "five_soft"):
        torch.manual_seed(9)
        models[mode] = build_five_task_model(artifacts, mode, dropout=0.0)
        assert len(models[mode].convolutions) == 2
    f80_parameters = sum(value.numel() for value in models["five_f80"].parameters())
    hard_parameters = sum(value.numel() for value in models["five_hard"].parameters())
    soft_parameters = sum(value.numel() for value in models["five_soft"].parameters())
    assert f80_parameters == hard_parameters
    assert soft_parameters == hard_parameters + 6
    hashes = {five_task_common_initialization_hash(model) for model in models.values()}
    assert len(hashes) == 1


def test_fixed_routes_and_soft_initialization(artifacts) -> None:
    hard = build_five_task_model(artifacts, "five_hard", dropout=0.0)
    dummy = {name: torch.full((1, 64), value) for name, value in (("f80", 80.0), ("p1", 1.0), ("p2", 2.0))}
    contexts = hard.task_contexts(dummy)
    assert contexts["event"].eq(2).all() and contexts["time"].eq(2).all()
    assert contexts["position"].eq(1).all()
    assert contexts["team"].eq(80).all() and contexts["player"].eq(80).all()
    soft = build_five_task_model(artifacts, "five_soft", dropout=0.0)
    weights = soft.fusion_weights()
    assert torch.allclose(weights["event"], torch.tensor([0.1, 0.9]))
    assert torch.allclose(weights["time"], torch.tensor([0.1, 0.9]))
    assert torch.allclose(weights["position"], torch.tensor([0.9, 0.1]))
    assert all(torch.allclose(value.sum(), torch.tensor(1.0)) for value in weights.values())


@pytest.mark.parametrize("mode", ["five_f80", "five_hard", "five_soft"])
def test_five_task_forward_backward(samples, artifacts, mode) -> None:
    torch.manual_seed(11)
    model = build_five_task_model(artifacts, mode, dropout=0.0)
    views = ("f80",) if mode == "five_f80" else ("f80", "p1", "p2")
    batch = _batch(samples, artifacts, views)
    predictions = model(batch)
    total, components = five_task_loss(predictions, batch, artifacts)
    expected = sum(components[name] * FIVE_TASK_WEIGHTS[name] for name in components) / FIVE_TASK_LOSS_DIVISOR
    assert torch.allclose(total, expected)
    total.backward()
    assert torch.isfinite(total)
    assert model.convolutions[0].kqv_lin.lins["event"].weight.grad is not None
    assert model.team_actor_head.weight.grad is not None
    assert model.player_actor_scorer[-1].weight.grad is not None
    if mode == "five_soft":
        assert all(value.grad is not None for value in model.fusion_logits.values())


def test_each_required_view_is_encoded_once(samples, artifacts) -> None:
    model = build_five_task_model(artifacts, "five_hard", dropout=0.0)
    batch = _batch(samples, artifacts)
    original = model.encode_context_and_states
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    model.encode_context_and_states = counted
    model(batch)
    assert calls == 3


def test_common_state_produces_exact_outputs_when_routes_are_identical(samples, artifacts) -> None:
    torch.manual_seed(13)
    baseline = build_five_task_model(artifacts, "five_f80", dropout=0.0).eval()
    torch.manual_seed(13)
    hard = build_five_task_model(artifacts, "five_hard", dropout=0.0).eval()
    hard.load_state_dict(baseline.state_dict(), strict=True)
    f80_batch = _batch(samples, artifacts, ("f80",))
    shared_graph = f80_batch["graphs"]["f80"]
    hard_batch = {**f80_batch, "graphs": {"f80": shared_graph, "p1": shared_graph, "p2": shared_graph}}
    with torch.no_grad():
        left, right = baseline(f80_batch), hard(hard_batch)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_gradient_diagnostics_cover_all_pairs(samples, artifacts) -> None:
    model = build_five_task_model(artifacts, "five_f80", dropout=0.0)
    diagnostics = five_task_gradient_diagnostics(model, _batch(samples, artifacts, ("f80",)), artifacts)
    assert len(diagnostics["gradient_cosine"]) == 10
    assert all(-1.0 <= value <= 1.0 for value in diagnostics["gradient_cosine"].values())


def test_checkpoint_round_trip(samples, artifacts, tmp_path) -> None:
    torch.manual_seed(17)
    source = build_five_task_model(artifacts, "five_soft", dropout=0.0).eval()
    batch = _batch(samples, artifacts)
    with torch.no_grad():
        expected = source(batch)
    path = tmp_path / "five.pt"
    torch.save(source.state_dict(), path)
    restored = build_five_task_model(artifacts, "five_soft", dropout=0.0).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        actual = restored(batch)
    for name in expected:
        assert torch.equal(expected[name], actual[name])
