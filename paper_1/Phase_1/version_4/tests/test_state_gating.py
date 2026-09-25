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
from football_hgt_targets_v4.fixed_budget_training import _common_hash
from football_hgt_targets_v4.model import (
    build_partial_l2_model,
    build_state_aware_partial_l2_model,
)
from football_hgt_targets_v4.state_gating import StateAwareHGTConv


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


def _has_gradient(parameters) -> bool:
    return any(
        value.grad is not None and float(value.grad.norm()) > 0
        for value in parameters
    )


def test_state_gating_configuration_is_registered() -> None:
    definition = CONFIGURATIONS["state_gated_partial_l2"]
    assert definition["state_gating"] is True
    assert definition["partial_l2"] is True
    assert definition["active_tasks"] == ALL_TASKS


def test_anchor_state_has_registered_shape_and_finite_values(batch) -> None:
    state = batch["anchor_state"]
    assert state.shape == (1, 8)
    assert state.dtype == torch.float32
    assert torch.isfinite(state).all()
    assert torch.all((state[:, :5] == 0) | (state[:, :5] == 1))
    assert torch.all((state[:, 7] == 0) | (state[:, 7] == 1))


def test_gate_is_sample_layer_relation_scalar(batch, artifacts) -> None:
    model = build_state_aware_partial_l2_model(artifacts, dropout=0.0).eval()
    with torch.no_grad():
        model(batch)
    graph = batch["graphs"]["f80"]
    convolution = model.convolutions[0]
    assert isinstance(convolution, StateAwareHGTConv)
    gates = convolution.last_gate_matrix
    assert gates is not None
    assert gates.shape == (1, len(convolution.edge_types))
    assert torch.equal(gates, torch.ones_like(gates))
    edge_gates = convolution.edge_gate_vector(
        dict(graph.edge_index_dict),
        {node_type: graph[node_type].batch for node_type in graph.node_types},
        gates,
    )
    assert edge_gates.ndim == 1
    assert torch.equal(edge_gates, torch.ones_like(edge_gates))


def test_original_initialization_hash_and_outputs_match(batch, artifacts) -> None:
    torch.manual_seed(510)
    baseline = build_partial_l2_model(artifacts, dropout=0.0).eval()
    baseline_hash = _common_hash(baseline)
    with torch.no_grad():
        expected = {name: value.clone() for name, value in baseline(batch).items()}
    del baseline

    torch.manual_seed(510)
    gated = build_state_aware_partial_l2_model(artifacts, dropout=0.0).eval()
    assert _common_hash(gated) == baseline_hash
    with torch.no_grad():
        actual = gated(batch)
    assert expected.keys() == actual.keys()
    assert max(
        float((expected[name] - actual[name]).abs().max()) for name in expected
    ) < 1e-6


def test_controller_copy_is_equal_but_independent(artifacts) -> None:
    model = build_state_aware_partial_l2_model(artifacts, dropout=0.0)
    main = dict(model.convolutions[1].gate_controller.named_parameters())
    player = dict(model.player_convolution.gate_controller.named_parameters())
    assert main.keys() == player.keys()
    for name in main:
        assert torch.equal(main[name], player[name])
        assert main[name].data_ptr() != player[name].data_ptr()


def test_gate_gradient_and_branch_isolation(batch, artifacts) -> None:
    model = build_state_aware_partial_l2_model(artifacts, dropout=0.0)
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    components["event"].backward()
    assert _has_gradient(model.convolutions[0].gate_controller.parameters())
    assert _has_gradient(model.convolutions[1].gate_controller.parameters())
    assert not _has_gradient(model.player_convolution.gate_controller.parameters())

    model.zero_grad(set_to_none=True)
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    components["player"].backward()
    assert _has_gradient(model.convolutions[0].gate_controller.parameters())
    assert not _has_gradient(model.convolutions[1].gate_controller.parameters())
    assert _has_gradient(model.player_convolution.gate_controller.parameters())


def test_state_dict_round_trip(batch, artifacts) -> None:
    model = build_state_aware_partial_l2_model(artifacts, dropout=0.0).eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = {name: value.clone() for name, value in model(batch).items()}
        model.convolutions[0].gate_controller.network[-1].bias.add_(0.5)
    model.load_state_dict(state)
    with torch.no_grad():
        actual = model(batch)
    assert max(
        float((expected[name] - actual[name]).abs().max()) for name in expected
    ) < 1e-6
