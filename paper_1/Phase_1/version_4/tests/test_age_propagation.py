from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.age_propagation import local_event_ages
from football_hgt_targets_v4.constants import (
    FEASIBILITY_ARTIFACT,
    POSSESSION_GRAPH_ROOT,
    SAMPLE_PLAN,
    WINDOW_SIZE,
)
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import ALL_TASKS, CONFIGURATIONS
from football_hgt_targets_v4.fixed_budget_training import _common_hash
from football_hgt_targets_v4.five_task_data import collate_five_task_hgt
from football_hgt_targets_v4.model import (
    POSSESSION_HGT_METADATA,
    build_age_propagation_model,
    build_partial_l2_model,
)
from football_hgt_targets_v4.training import _move_batch_to_device


@pytest.fixture(scope="module")
def artifacts():
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def samples(artifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=128,
        selected_currents=selected,
    )
    return [value for value in dataset if value.current_event_index >= 20][:2]


def make_batch(samples, artifacts):
    return next(iter(DataLoader(
        samples,
        batch_size=2,
        collate_fn=partial(
            collate_five_task_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            context_views=("f80",),
        ),
    )))


def cuda_batch(samples, artifacts):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for HGT propagation tests")
    return _move_batch_to_device(make_batch(samples, artifacts), torch.device("cuda:0"))


def test_configurations_are_guarded_partial_l2():
    for name, mode in (
        ("pg_constant", "constant"),
        ("pg_shared", "shared"),
        ("pg_task", "task"),
    ):
        assert CONFIGURATIONS[name]["partial_l2"] is True
        assert CONFIGURATIONS[name]["checkpoint_metric"] == "guarded_core"
        assert CONFIGURATIONS[name]["age_propagation"] == mode


def test_common_initialization_and_rng_isolation(artifacts):
    torch.manual_seed(313)
    baseline = build_partial_l2_model(artifacts, dropout=0.0)
    baseline_hash = _common_hash(baseline)
    next_baseline = torch.rand(4)

    models = {}
    next_values = {}
    for mode in ("constant", "shared", "task"):
        torch.manual_seed(313)
        models[mode] = build_age_propagation_model(artifacts, mode, dropout=0.0)
        next_values[mode] = torch.rand(4)
        assert _common_hash(models[mode]) == baseline_hash
        assert torch.equal(next_values[mode], next_baseline)

    for left, right in zip(
        models["constant"].propagation_residuals,
        models["shared"].propagation_residuals,
    ):
        for left_value, right_value in zip(left.parameters(), right.parameters()):
            assert torch.equal(left_value, right_value)
    shared_gate = models["shared"].propagation_gate
    task_gate = models["task"].propagation_gate
    assert torch.equal(shared_gate.age_embedding.weight, task_gate.age_embedding.weight)
    assert torch.equal(shared_gate.pooling_embeddings[0], task_gate.pooling_embeddings[0])
    assert torch.equal(task_gate.pooling_embeddings[0], task_gate.pooling_embeddings[1])
    assert torch.equal(task_gate.pooling_embeddings[1], task_gate.pooling_embeddings[2])
    for shared_scorer, task_scorer in zip(shared_gate.scorers, task_gate.scorers):
        for left_value, right_value in zip(shared_scorer.parameters(), task_scorer.parameters()):
            assert torch.equal(left_value, right_value)


def test_initial_outputs_and_loss_equal_partial_l2(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    torch.manual_seed(401)
    baseline = build_partial_l2_model(artifacts, dropout=0.0).cuda().eval()
    with torch.no_grad():
        expected = baseline(batch)
        expected_loss = fixed_budget_loss(expected, batch, artifacts, ALL_TASKS)[0]
    for mode in ("constant", "shared", "task"):
        torch.manual_seed(401)
        model = build_age_propagation_model(artifacts, mode, dropout=0.0).cuda().eval()
        with torch.no_grad():
            actual = model(batch)
        assert max(float((expected[key] - actual[key]).abs().max()) for key in expected) < 1e-6
        actual_loss = fixed_budget_loss(actual, batch, artifacts, ALL_TASKS)[0]
        assert float((actual_loss - expected_loss).abs()) < 1e-6


def test_event_endpoint_age_and_global_degree(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    graph = batch["graphs"]["f80"]
    model = build_age_propagation_model(artifacts, "shared", dropout=0.0).cuda().eval()
    ages = local_event_ages(graph)
    assert model.source_age_mismatch_count(graph) == 0
    layer = model.propagation_residuals[0]
    for edge_type in POSSESSION_HGT_METADATA[1]:
        edge_index = graph.edge_index_dict[edge_type]
        mapped = layer._edge_ages(edge_type, edge_index, ages)
        if edge_type[0] == "event":
            assert torch.equal(mapped, ages[edge_index[0]])
        elif edge_type[-1] == "event":
            assert torch.equal(mapped, ages[edge_index[1]])
        else:
            assert mapped is None

    layer.capture_diagnostics = True
    states = model._encode_nodes(graph)
    layer(states, dict(graph.edge_index_dict), ages, model.propagation_gate, 0, "event")
    for node_type in POSSESSION_HGT_METADATA[0]:
        expected = torch.zeros(graph[node_type].num_nodes, device=ages.device)
        for edge_type, edge_index in graph.edge_index_dict.items():
            if edge_type[-1] == node_type:
                expected.index_add_(
                    0, edge_index[1], torch.ones(edge_index.shape[1], device=ages.device)
                )
        assert torch.equal(layer.last_degree_by_type[node_type], expected)


def test_single_backbone_and_residual_call_counts(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    for mode, expected_residual_calls in (("constant", 1), ("shared", 1), ("task", 3)):
        model = build_age_propagation_model(artifacts, mode, dropout=0.0).cuda().eval()
        calls = {"shared": 0, "main": 0, "player": 0}
        hooks = [
            model.convolutions[0].register_forward_hook(
                lambda *_: calls.__setitem__("shared", calls["shared"] + 1)
            ),
            model.convolutions[1].register_forward_hook(
                lambda *_: calls.__setitem__("main", calls["main"] + 1)
            ),
            model.player_convolution.register_forward_hook(
                lambda *_: calls.__setitem__("player", calls["player"] + 1)
            ),
        ]
        with torch.no_grad():
            model(batch)
        for hook in hooks:
            hook.remove()
        assert calls == {"shared": 1, "main": 1, "player": 1}
        assert [layer.forward_calls for layer in model.propagation_residuals] == [
            expected_residual_calls,
            expected_residual_calls,
        ]


def test_task_state_only_enters_second_residual(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    model = build_age_propagation_model(artifacts, "shared", dropout=0.0).cuda().eval()
    for up in model.propagation_residuals[0].up.values():
        torch.nn.init.normal_(up.weight, std=0.1)
    observed = {}

    def main_pre(_module, args):
        observed["main"] = {key: value.detach().clone() for key, value in args[0].items()}

    def residual_pre(_module, args):
        observed["residual"] = {key: value.detach().clone() for key, value in args[0].items()}

    hooks = [
        model.convolutions[1].register_forward_pre_hook(main_pre),
        model.propagation_residuals[1].register_forward_pre_hook(residual_pre),
    ]
    with torch.no_grad():
        model(batch)
    for hook in hooks:
        hook.remove()
    assert any(
        not torch.equal(observed["main"][node_type], observed["residual"][node_type])
        for node_type in observed["main"]
    )


def test_team_player_paths_ignore_nonzero_propagation_residual(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    torch.manual_seed(419)
    baseline = build_partial_l2_model(artifacts, dropout=0.0).cuda().eval()
    torch.manual_seed(419)
    model = build_age_propagation_model(artifacts, "task", dropout=0.0).cuda().eval()
    for layer in model.propagation_residuals:
        for up in layer.up.values():
            torch.nn.init.normal_(up.weight, std=0.1)
    with torch.no_grad():
        expected = baseline(batch)
        actual = model(batch)
    assert float((expected["team_logits"] - actual["team_logits"]).abs().max()) < 1e-6
    assert float((expected["player_scores"] - actual["player_scores"]).abs().max()) < 1e-6
    assert any(
        not torch.equal(expected[key], actual[key])
        for key in ("event_logits", "time_seconds", "position_xy")
    )


def test_state_dict_checkpoint_roundtrip(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    torch.manual_seed(431)
    model = build_age_propagation_model(artifacts, "task", dropout=0.0).cuda().eval()
    for layer in model.propagation_residuals:
        for up in layer.up.values():
            torch.nn.init.normal_(up.weight, std=0.01)
    with torch.no_grad():
        expected = {key: value.clone() for key, value in model(batch).items()}
    restored = build_age_propagation_model(artifacts, "task", dropout=0.0).cuda().eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        actual = restored(batch)
    assert all(
        float((expected[key] - actual[key]).abs().max()) < 1e-6
        for key in expected
    )


def test_main_player_and_residual_gradients_are_isolated(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    model = build_age_propagation_model(artifacts, "task", dropout=0.0).cuda()
    _, components, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    main = list(model.convolutions[1].parameters())
    player = list(model.player_convolution.parameters())
    residual = list(model.propagation_residuals.parameters())
    event_to_player = torch.autograd.grad(
        components["event"], player, retain_graph=True, allow_unused=True
    )
    player_to_main = torch.autograd.grad(
        components["player"], main, retain_graph=True, allow_unused=True
    )
    player_to_residual = torch.autograd.grad(
        components["player"], residual, allow_unused=True
    )
    assert all(value is None for value in event_to_player)
    assert all(value is None for value in player_to_main)
    assert all(value is None for value in player_to_residual)


def _grad_sum(parameters):
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in parameters
        if parameter.grad is not None
    )


def test_three_stage_zero_init_gradient_contract(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    model = build_age_propagation_model(artifacts, "task", dropout=0.0).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    up = [parameter for layer in model.propagation_residuals for module in layer.up.values() for parameter in module.parameters()]
    down = [parameter for layer in model.propagation_residuals for module in layer.down for parameter in module.parameters()]
    gate_final = [parameter for scorer in model.propagation_gate.scorers for parameter in scorer[-1].parameters()]
    gate_upstream = [parameter for scorer in model.propagation_gate.scorers for parameter in scorer[0].parameters()]

    fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)[0].backward()
    assert _grad_sum(up) > 0
    assert _grad_sum(down) == 0
    assert _grad_sum(gate_final) == 0
    assert _grad_sum(gate_upstream) == 0
    assert _grad_sum([model.propagation_gate.age_embedding.weight, model.propagation_gate.pooling_embeddings]) == 0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)

    fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)[0].backward()
    assert _grad_sum(down) > 0
    assert _grad_sum(gate_final) > 0
    assert _grad_sum(gate_upstream) == 0
    assert _grad_sum([model.propagation_gate.age_embedding.weight, model.propagation_gate.pooling_embeddings]) == 0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)

    fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)[0].backward()
    assert _grad_sum(gate_upstream) > 0
    assert _grad_sum([model.propagation_gate.age_embedding.weight, model.propagation_gate.pooling_embeddings]) > 0
    assert all(float(row.abs().sum()) > 0 for row in model.propagation_gate.pooling_embeddings.grad)
