from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import ALL_TASKS, CONFIGURATIONS
from football_hgt_targets_v4.fixed_budget_training import _common_hash
from football_hgt_targets_v4.five_task_data import collate_five_task_hgt
from football_hgt_targets_v4.model import build_age_pooling_model, build_partial_l2_model
from football_hgt_targets_v4.training import _move_batch_to_device


@pytest.fixture(scope="module")
def artifacts():
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def samples(artifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT), artifacts,
        WINDOW_SIZE, max_samples=128, selected_currents=selected,
    )
    return [value for value in dataset if value.current_event_index >= 20][:2]


def make_batch(samples, artifacts):
    return next(iter(DataLoader(samples, batch_size=2, collate_fn=partial(
        collate_five_task_hgt,
        artifacts=artifacts, window_size=WINDOW_SIZE, context_views=("f80",),
    ))))


def cuda_batch(samples, artifacts):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for HGT age-pooling forward tests")
    return _move_batch_to_device(make_batch(samples, artifacts), torch.device("cuda:0"))


def test_configurations_are_guarded_partial_l2():
    for name, mode in (("ap_shared", "shared"), ("ap_task", "task")):
        assert CONFIGURATIONS[name]["partial_l2"] is True
        assert CONFIGURATIONS[name]["checkpoint_metric"] == "guarded_core"
        assert CONFIGURATIONS[name]["age_pooling"] == mode


def test_strict_common_initialization_and_rng(artifacts):
    torch.manual_seed(31)
    baseline = build_partial_l2_model(artifacts, dropout=0.0)
    baseline_hash = _common_hash(baseline)
    baseline_next = torch.rand(4)
    del baseline

    torch.manual_seed(31)
    shared = build_age_pooling_model(artifacts, "shared", dropout=0.0)
    shared_next = torch.rand(4)
    torch.manual_seed(31)
    task = build_age_pooling_model(artifacts, "task", dropout=0.0)
    task_next = torch.rand(4)

    assert _common_hash(shared) == baseline_hash == _common_hash(task)
    assert torch.equal(baseline_next, shared_next)
    assert torch.equal(shared_next, task_next)
    assert torch.equal(shared.age_embedding.weight, task.age_embedding.weight)
    assert torch.equal(shared.age_scorer[0].weight, task.age_scorer[0].weight)
    assert torch.equal(shared.pooling_embeddings[0], task.pooling_embeddings[0])
    assert torch.equal(task.pooling_embeddings[0], task.pooling_embeddings[1])
    assert torch.equal(task.pooling_embeddings[1], task.pooling_embeddings[2])

    ages = torch.arange(80)
    shared_input = torch.cat((shared.age_embedding(ages), shared.pooling_embeddings[0].expand(80, -1)), dim=-1)
    task_input = torch.cat((task.age_embedding(ages), task.pooling_embeddings[2].expand(80, -1)), dim=-1)
    assert torch.equal(shared.age_scorer[:2](shared_input), task.age_scorer[:2](task_input))
    assert torch.equal(shared.raw_age_scores("event", ages), task.raw_age_scores("position", ages))


def test_initial_outputs_are_exact_mean_regression(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    torch.manual_seed(47)
    baseline = build_partial_l2_model(artifacts, dropout=0.0).cuda().eval()
    with torch.no_grad():
        expected = baseline(batch)
        expected_loss = fixed_budget_loss(expected, batch, artifacts, ALL_TASKS)[0]
    for mode in ("shared", "task"):
        torch.manual_seed(47)
        model = build_age_pooling_model(artifacts, mode, dropout=0.0).cuda().eval()
        with torch.no_grad():
            actual = model(batch)
        assert max(float((expected[key] - actual[key]).abs().max()) for key in expected) < 1e-6
        assert float((fixed_budget_loss(actual, batch, artifacts, ALL_TASKS)[0] - expected_loss).abs()) < 1e-6


def test_local_ages_weights_and_shared_identity(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    graph = batch["graphs"]["f80"]
    model = build_age_pooling_model(artifacts, "shared", dropout=0.0).cuda().eval()
    assert model.source_age_mismatch_count(graph) == 0
    ages, event_weight = model.age_weights(graph, "event")
    _, time_weight = model.age_weights(graph, "time")
    assert torch.equal(event_weight, time_weight)
    ptr = graph["event"].ptr
    for start, stop in zip(ptr[:-1], ptr[1:]):
        sample_age = ages[int(start):int(stop)]
        assert int(sample_age[-1]) == 0
        assert int(sample_age.max()) <= 79
        assert torch.equal(sample_age, torch.arange(len(sample_age) - 1, -1, -1, device=sample_age.device))
        assert torch.allclose(event_weight[int(start):int(stop)].sum(), torch.tensor(1.0, device=event_weight.device))


def test_zero_init_two_stage_gradient_contract(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    model = build_age_pooling_model(artifacts, "task", dropout=0.0).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    loss, _, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    loss.backward()
    assert float(model.age_scorer[-1].weight.grad.abs().sum()) > 0
    assert float(model.age_scorer[0].weight.grad.abs().sum()) == 0
    assert float(model.age_embedding.weight.grad.abs().sum()) == 0
    assert float(model.pooling_embeddings.grad.abs().sum()) == 0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in model.convolutions[0].parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    loss, _, _ = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)
    loss.backward()
    assert float(model.age_scorer[0].weight.grad.abs().sum()) > 0
    assert float(model.age_embedding.weight.grad.abs().sum()) > 0
    assert float(model.pooling_embeddings.grad.abs().sum()) > 0
    row_gradients = model.pooling_embeddings.grad
    assert row_gradients.shape == (3, 16)
    assert all(float(row_gradients[index].abs().sum()) > 0 for index in range(3))


def test_single_hgt_pass_per_branch(samples, artifacts):
    batch = cuda_batch(samples, artifacts)
    model = build_age_pooling_model(artifacts, "task", dropout=0.0).cuda().eval()
    calls = {"shared": 0, "main": 0, "player": 0}
    hooks = [
        model.convolutions[0].register_forward_hook(lambda *_: calls.__setitem__("shared", calls["shared"] + 1)),
        model.convolutions[1].register_forward_hook(lambda *_: calls.__setitem__("main", calls["main"] + 1)),
        model.player_convolution.register_forward_hook(lambda *_: calls.__setitem__("player", calls["player"] + 1)),
    ]
    with torch.no_grad():
        model(batch)
    for hook in hooks:
        hook.remove()
    assert calls == {"shared": 1, "main": 1, "player": 1}
