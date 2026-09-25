from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import (
    FEASIBILITY_ARTIFACT,
    POSSESSION_GRAPH_ROOT,
    SAMPLE_PLAN,
    WINDOW_SIZE,
)
from football_hgt_targets_v4.five_task_data import collate_five_task_hgt
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import ALL_TASKS, CONFIGURATIONS
from football_hgt_targets_v4.fixed_budget_training import guarded_core_eligible
from football_hgt_targets_v4.fixed_budget_training import _common_hash
from football_hgt_targets_v4.model import (
    build_five_task_model,
    build_partial_l2_model,
)


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def batch(artifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=32,
        selected_currents=selected,
    )
    samples = [
        value
        for value in dataset
        if bool(value.graph["targets"]["player_known_mask"][value.current_event_index])
    ][:1]
    return next(iter(DataLoader(
        samples,
        batch_size=1,
        collate_fn=partial(
            collate_five_task_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            context_views=("f80",),
        ),
    )))


def test_partial_l2_is_registered() -> None:
    definition = CONFIGURATIONS["partial_l2"]
    assert definition["active_tasks"] == ALL_TASKS
    assert definition["checkpoint_metric"] == "guarded_core"
    assert definition["partial_l2"] is True


def test_private_layer_is_equal_but_not_shared(artifacts) -> None:
    torch.manual_seed(101)
    model = build_partial_l2_model(artifacts, dropout=0.0)
    main = dict(model.convolutions[1].named_parameters())
    private = dict(model.player_convolution.named_parameters())
    assert main.keys() == private.keys()
    for name in main:
        assert torch.equal(main[name], private[name])
        assert main[name].data_ptr() != private[name].data_ptr()
    for main_value, private_value in zip(
        model.context_projection.parameters(),
        model.player_context_projection.parameters(),
    ):
        assert torch.equal(main_value, private_value)
        assert main_value.data_ptr() != private_value.data_ptr()


def test_shared_initialization_hash_matches_five_f80(artifacts) -> None:
    torch.manual_seed(102)
    baseline = build_five_task_model(artifacts, "five_f80", dropout=0.0)
    left = _common_hash(baseline)
    del baseline
    torch.manual_seed(102)
    partial_l2 = build_partial_l2_model(artifacts, dropout=0.0)
    assert _common_hash(partial_l2) == left


def test_initial_outputs_match_fully_shared(batch, artifacts) -> None:
    torch.manual_seed(103)
    baseline = build_five_task_model(artifacts, "five_f80", dropout=0.0).eval()
    with torch.no_grad():
        left = {name: value.clone() for name, value in baseline(batch).items()}
    del baseline
    torch.manual_seed(103)
    partial_l2 = build_partial_l2_model(artifacts, dropout=0.0).eval()
    with torch.no_grad():
        right = partial_l2(batch)
    assert left.keys() == right.keys()
    assert max(float((left[name] - right[name]).abs().max()) for name in left) < 1e-6


def test_layer_call_counts(batch, artifacts) -> None:
    model = build_partial_l2_model(artifacts, dropout=0.0).eval()
    counts = {"shared": 0, "main": 0, "player": 0}
    hooks = [
        model.convolutions[0].register_forward_hook(
            lambda *_: counts.__setitem__("shared", counts["shared"] + 1)
        ),
        model.convolutions[1].register_forward_hook(
            lambda *_: counts.__setitem__("main", counts["main"] + 1)
        ),
        model.player_convolution.register_forward_hook(
            lambda *_: counts.__setitem__("player", counts["player"] + 1)
        ),
    ]
    with torch.no_grad():
        model(batch)
    for hook in hooks:
        hook.remove()
    assert counts == {"shared": 1, "main": 1, "player": 1}


def _has_gradient(parameters) -> bool:
    return any(
        value.grad is not None and float(value.grad.norm()) > 0
        for value in parameters
    )


def test_branch_gradients_are_isolated(batch, artifacts) -> None:
    model = build_partial_l2_model(artifacts, dropout=0.0)
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    components["event"].backward()
    assert _has_gradient(model.convolutions[0].parameters())
    assert _has_gradient(model.convolutions[1].parameters())
    assert not _has_gradient(model.player_convolution.parameters())

    model.zero_grad(set_to_none=True)
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    components["player"].backward()
    assert _has_gradient(model.convolutions[0].parameters())
    assert not _has_gradient(model.convolutions[1].parameters())
    assert _has_gradient(model.player_convolution.parameters())


def test_guarded_core_thresholds() -> None:
    reference = {"team_accuracy": 0.86, "player_top1": 0.42}
    valid = {
        "team": {"accuracy": 0.85},
        "player": {"top1_accuracy": 0.41},
    }
    assert guarded_core_eligible(valid, reference)
    invalid_team = {**valid, "team": {"accuracy": 0.849}}
    invalid_player = {**valid, "player": {"top1_accuracy": 0.409}}
    assert not guarded_core_eligible(invalid_team, reference)
    assert not guarded_core_eligible(invalid_player, reference)


def test_partial_l2_state_dict_restores_outputs(batch, artifacts) -> None:
    torch.manual_seed(107)
    model = build_partial_l2_model(artifacts, dropout=0.0).eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = {name: value.clone() for name, value in model(batch).items()}
        next(model.player_convolution.parameters()).add_(1.0)
    model.load_state_dict(state)
    with torch.no_grad():
        actual = model(batch)
    assert max(float((expected[name] - actual[name]).abs().max()) for name in expected) < 1e-6
