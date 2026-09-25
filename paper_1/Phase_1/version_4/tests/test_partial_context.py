from __future__ import annotations

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
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import ALL_TASKS, CONFIGURATIONS
from football_hgt_targets_v4.fixed_budget_training import _common_hash, guarded_core_eligible
from football_hgt_targets_v4.model import build_partial_l2_model
from football_hgt_targets_v4.partial_context_reporting import _task_decision


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def batches(artifacts):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=32,
        selected_currents=selected,
    )
    sample = next(
        value for value in dataset
        if bool(value.graph["targets"]["player_known_mask"][value.current_event_index])
    )

    def collate(views):
        return next(iter(DataLoader(
            [sample],
            batch_size=1,
            collate_fn=partial(
                collate_five_task_hgt,
                artifacts=artifacts,
                window_size=WINDOW_SIZE,
                context_views=views,
            ),
        )))

    return collate(("f80",)), collate(("f80", "p1", "p2"))


def _has_gradient(parameters) -> bool:
    return any(value.grad is not None and float(value.grad.norm()) > 0 for value in parameters)


def test_partial_context_configurations_are_registered() -> None:
    for configuration, mode in (
        ("partial_l2_hard", "five_hard"),
        ("partial_l2_soft", "five_soft"),
    ):
        assert CONFIGURATIONS[configuration] == {
            "mode": mode,
            "active_tasks": ALL_TASKS,
            "checkpoint_metric": "guarded_core",
            "partial_l2": True,
        }


def test_f80_regression_is_exact(batches, artifacts) -> None:
    f80, _ = batches
    torch.manual_seed(501)
    original = build_partial_l2_model(artifacts, dropout=0.0).eval()
    torch.manual_seed(501)
    context = build_partial_l2_model(artifacts, "five_f80", dropout=0.0).eval()
    with torch.no_grad():
        left, right = original(f80), context(f80)
    assert max(float((left[name] - right[name]).abs().max()) for name in left) < 1e-6


def test_common_initialization_matches_all_context_modes(artifacts) -> None:
    hashes = set()
    for mode in ("five_f80", "five_hard", "five_soft"):
        torch.manual_seed(502)
        hashes.add(_common_hash(build_partial_l2_model(artifacts, mode, dropout=0.0)))
    assert len(hashes) == 1


def test_routes_and_soft_initialization(artifacts) -> None:
    hard = build_partial_l2_model(artifacts, "five_hard", dropout=0.0)
    values = {name: torch.full((1, 64), value) for name, value in (("f80", 80.0), ("p1", 1.0), ("p2", 2.0))}
    routed = hard.task_contexts(values)
    assert routed["event"].eq(2).all() and routed["time"].eq(2).all()
    assert routed["position"].eq(1).all()
    assert routed["team"].eq(80).all() and routed["player"].eq(80).all()
    weights = build_partial_l2_model(artifacts, "five_soft", dropout=0.0).fusion_weights()
    assert torch.allclose(weights["event"], torch.tensor([0.1, 0.9]))
    assert torch.allclose(weights["time"], torch.tensor([0.1, 0.9]))
    assert torch.allclose(weights["position"], torch.tensor([0.9, 0.1]))
    assert all(torch.allclose(value.sum(), torch.tensor(1.0)) for value in weights.values())


@pytest.mark.parametrize("mode", ["five_hard", "five_soft"])
def test_view_calls_and_branch_isolation(batches, artifacts, mode) -> None:
    _, multiview = batches
    model = build_partial_l2_model(artifacts, mode, dropout=0.0)
    counts = {"shared": 0, "main": 0, "player": 0}
    hooks = [
        model.convolutions[0].register_forward_hook(lambda *_: counts.__setitem__("shared", counts["shared"] + 1)),
        model.convolutions[1].register_forward_hook(lambda *_: counts.__setitem__("main", counts["main"] + 1)),
        model.player_convolution.register_forward_hook(lambda *_: counts.__setitem__("player", counts["player"] + 1)),
    ]
    fixed_budget_loss(model(multiview), multiview, artifacts, ALL_TASKS)[0].backward()
    for hook in hooks:
        hook.remove()
    assert counts == {"shared": 3, "main": 3, "player": 1}
    if mode == "five_soft":
        assert all(value.grad is not None for value in model.fusion_logits.values())

    model.zero_grad(set_to_none=True)
    _, components, _ = fixed_budget_loss(model(multiview), multiview, artifacts, ALL_TASKS)
    components["event"].backward()
    assert _has_gradient(model.convolutions[1].parameters())
    assert not _has_gradient(model.player_convolution.parameters())
    model.zero_grad(set_to_none=True)
    _, components, _ = fixed_budget_loss(model(multiview), multiview, artifacts, ALL_TASKS)
    components["player"].backward()
    assert not _has_gradient(model.convolutions[1].parameters())
    assert _has_gradient(model.player_convolution.parameters())


def test_dual_actor_guard_thresholds() -> None:
    metrics = {"team": {"accuracy": 0.855}, "player": {"top1_accuracy": 0.415}}
    five = {"team_accuracy": 0.86, "player_top1": 0.42}
    partial = {"team_accuracy": 0.86, "player_top1": 0.42}
    assert guarded_core_eligible(metrics, five, margin=0.01)
    assert guarded_core_eligible(metrics, partial, margin=0.005)
    assert not guarded_core_eligible({**metrics, "team": {"accuracy": 0.8549}}, partial, margin=0.005)


def test_effective_task_requires_direction_ci_and_practical_gain() -> None:
    comparison = {"position_distance_mae_m": {"ci95": [-0.40, -0.10]}}
    effective = _task_decision("position", np.array([-0.5, -0.4, 0.1]), comparison)
    assert effective["effective"]
    too_small = _task_decision("position", np.array([-0.1, -0.1, -0.1]), comparison)
    assert not too_small["effective"]
    crosses_zero = _task_decision(
        "position",
        np.array([-0.3, -0.3, -0.3]),
        {"position_distance_mae_m": {"ci95": [-0.5, 0.01]}},
    )
    assert not crosses_zero["effective"]
