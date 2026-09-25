from __future__ import annotations

import random
from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT, SAMPLE_PLAN, WINDOW_SIZE
from football_hgt_targets_v4.five_task_data import collate_five_task_hgt
from football_hgt_targets_v4.five_task_study import FIVE_TASK_WEIGHTS
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import CONFIGURATIONS, CORE_TASKS
from football_hgt_targets_v4.fixed_budget_training import (
    FixedBudgetConfig,
    _common_hash,
    _optimizer,
    _restore_rng,
    _rng_state,
)
from football_hgt_targets_v4.model import build_five_task_model


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def sample(artifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=32,
        selected_currents=selected,
    )
    return next(
        value
        for value in dataset
        if bool(value.graph["targets"]["player_known_mask"][value.current_event_index])
    )


def _batch(sample, artifacts, views=("f80",)):
    return next(iter(DataLoader(
        [sample], batch_size=1,
        collate_fn=partial(
            collate_five_task_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            context_views=views,
        ),
    )))


def test_registered_stage_definitions_are_decision_complete() -> None:
    assert CONFIGURATIONS["three_f80"]["active_tasks"] == CORE_TASKS
    assert CONFIGURATIONS["five_f80"]["checkpoint_metric"] == "joint_five"
    assert CONFIGURATIONS["t4_team"]["active_tasks"] == (*CORE_TASKS, "team")
    assert CONFIGURATIONS["t4_player"]["active_tasks"] == (*CORE_TASKS, "player")
    assert CONFIGURATIONS["t5_player_adapter"]["player_adapter"] is True


def test_active_loss_keeps_fixed_divisor(sample, artifacts) -> None:
    model = build_five_task_model(artifacts, "five_f80", dropout=0.0)
    batch = _batch(sample, artifacts)
    predictions = model(batch)
    three, components, core = fixed_budget_loss(predictions, batch, artifacts, CORE_TASKS)
    five, _, _ = fixed_budget_loss(predictions, batch, artifacts, CONFIGURATIONS["five_f80"]["active_tasks"])
    expected_three = sum(components[name] * FIVE_TASK_WEIGHTS[name] for name in CORE_TASKS) / 3
    expected_five = sum(components[name] * FIVE_TASK_WEIGHTS[name] for name in components) / 3
    assert torch.allclose(three, expected_three)
    assert torch.allclose(core, expected_three)
    assert torch.allclose(five, expected_five)


def test_common_initialization_matches_three_and_five_modes(artifacts) -> None:
    hashes = []
    for mode in ("five_f80", "five_hard", "five_soft"):
        torch.manual_seed(31)
        hashes.append(_common_hash(build_five_task_model(artifacts, mode, dropout=0.0)))
    assert len(set(hashes)) == 1


def test_inactive_heads_receive_no_gradient_or_optimizer_update(sample, artifacts, tmp_path) -> None:
    torch.manual_seed(37)
    model = build_five_task_model(artifacts, "five_f80", dropout=0.0)
    config = FixedBudgetConfig("three_f80", tmp_path, 37, "cpu", training_budget=1)
    optimizer = _optimizer(model, config)
    before_team = {name: value.detach().clone() for name, value in model.team_actor_head.state_dict().items()}
    before_player = {name: value.detach().clone() for name, value in model.player_actor_scorer.state_dict().items()}
    batch = _batch(sample, artifacts)
    loss, _, _ = fixed_budget_loss(model(batch), batch, artifacts, config.active_tasks)
    loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in model.team_actor_head.parameters())
    assert all(parameter.grad is None for parameter in model.player_actor_scorer.parameters())
    assert all(torch.equal(value, before_team[name]) for name, value in model.team_actor_head.state_dict().items())
    assert all(torch.equal(value, before_player[name]) for name, value in model.player_actor_scorer.state_dict().items())


def test_player_adapter_is_exactly_output_equivalent_and_trainable(sample, artifacts) -> None:
    torch.manual_seed(41)
    baseline = build_five_task_model(artifacts, "five_f80", dropout=0.0).eval()
    torch.manual_seed(41)
    adapted = build_five_task_model(artifacts, "five_f80", dropout=0.0, player_adapter=True).eval()
    batch = _batch(sample, artifacts)
    with torch.no_grad():
        left, right = baseline(batch), adapted(batch)
    assert max(float((left[name] - right[name]).abs().max()) for name in left) < 1e-6
    adapted.train()
    loss, _, _ = fixed_budget_loss(adapted(batch), batch, artifacts, CONFIGURATIONS["t5_player_adapter"]["active_tasks"])
    loss.backward()
    assert adapted.player_adapter[-1].weight.grad is not None
    assert float(adapted.player_adapter[-1].weight.grad.norm()) > 0


class _GeneratorHolder:
    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)


class _FakeLoader:
    def __init__(self):
        self.sampler = _GeneratorHolder(51)
        self.generator = torch.Generator().manual_seed(52)


def test_rng_and_sampler_state_round_trip() -> None:
    loader = _FakeLoader()
    random.seed(53); np.random.seed(54); torch.manual_seed(55)
    state = _rng_state(loader)
    expected = (
        random.random(), float(np.random.random()), float(torch.rand(())),
        float(torch.rand((), generator=loader.sampler.generator)),
        float(torch.rand((), generator=loader.generator)),
    )
    _restore_rng(state, loader)
    actual = (
        random.random(), float(np.random.random()), float(torch.rand(())),
        float(torch.rand((), generator=loader.sampler.generator)),
        float(torch.rand((), generator=loader.generator)),
    )
    assert actual == expected
