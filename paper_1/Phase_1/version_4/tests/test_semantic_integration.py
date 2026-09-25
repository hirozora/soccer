from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import (
    CanonicalEventDataset,
    collate_semantic_hgt,
    load_records,
)
from football_benchmark.models import SemanticBenchmarkHGT
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import (
    EVENT_METHODS,
    FEASIBILITY_ARTIFACT,
    POSITION_METHODS,
    SAMPLE_PLAN,
    SEMANTIC_GRAPH_ROOT,
    TIME_METHODS,
    WINDOW_SIZE,
)
from football_hgt_targets_v4.losses import compute_loss
from football_hgt_targets_v4.diagnostics import compute_gradient_diagnostics
from football_hgt_targets_v4.model import TargetStudyHGT
from football_hgt_targets_v4.training import backbone_state_hash


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def batch(artifacts: ProtocolArtifacts) -> dict:
    records = load_records("validation", graph_root=SEMANTIC_GRAPH_ROOT)
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        records,
        artifacts,
        WINDOW_SIZE,
        max_samples=2,
        selected_currents=selected,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(
            collate_semantic_hgt, artifacts=artifacts, window_size=WINDOW_SIZE
        ),
    )
    result = next(iter(loader))
    event_counts = result["graph"]["event"].ptr[1:] - result["graph"]["event"].ptr[:-1]
    assert bool((event_counts <= WINDOW_SIZE).all())
    assert all(
        int(sample_id.rsplit(":", 1)[1]) == int(current)
        for sample_id, current in zip(result["sample_ids"], result["current_event_indices"])
    )
    return result


@pytest.mark.parametrize(
    ("task", "method", "key"),
    [
        ("event", "inverse_ce", "event_logits"),
        ("time", "current_huber", "time_seconds"),
        ("position", "xy", "position_xy"),
    ],
)
def test_legacy_forward_is_exact(
    artifacts: ProtocolArtifacts,
    batch: dict,
    task: str,
    method: str,
    key: str,
) -> None:
    torch.manual_seed(20260715)
    baseline = SemanticBenchmarkHGT(artifacts, "unified_lem", dropout=0.1)
    candidate = TargetStudyHGT(artifacts, task, method, dropout=0.1)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        expected = baseline(batch)[key]
        observed = candidate(batch)[key]
    assert torch.max(torch.abs(expected - observed)).item() < 1e-6


@pytest.mark.parametrize(
    ("task", "method"),
    [
        *(("event", method) for method in EVENT_METHODS),
        *(("time", method) for method in TIME_METHODS),
        *(("position", method) for method in POSITION_METHODS),
    ],
)
def test_all_formulations_forward_backward(
    artifacts: ProtocolArtifacts, batch: dict, task: str, method: str
) -> None:
    torch.manual_seed(11)
    model = TargetStudyHGT(artifacts, task, method, dropout=0.0)
    predictions = model(batch)
    loss, _ = compute_loss(predictions, batch, task, method, artifacts)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_joint_weight_changes_only_total_loss(
    artifacts: ProtocolArtifacts, batch: dict
) -> None:
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    torch.manual_seed(31)
    model = TargetStudyHGT(artifacts, "joint", "joint_test", methods, dropout=0.0)
    predictions = model(batch)
    default_total, default_components = compute_loss(
        predictions, batch, "joint", "joint_test", artifacts, methods
    )
    explicit_total, explicit_components = compute_loss(
        predictions,
        batch,
        "joint",
        "joint_test",
        artifacts,
        methods,
        {"event": 1.0, "time": 1.0, "position": 1.0},
    )
    scaled_total, scaled_components = compute_loss(
        predictions,
        batch,
        "joint",
        "joint_test",
        artifacts,
        methods,
        {"event": 0.1, "time": 1.0, "position": 1.0},
    )
    assert torch.equal(default_total, explicit_total)
    for task in methods:
        assert torch.equal(default_components[task], explicit_components[task])
        assert torch.equal(default_components[task], scaled_components[task])
    expected = (
        0.1 * default_components["event"]
        + default_components["time"]
        + default_components["position"]
    ) / 3
    assert torch.allclose(scaled_total, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "weights",
    [
        {"event": 0.0, "time": 1.0, "position": 1.0},
        {"event": -0.1, "time": 1.0, "position": 1.0},
        {"event": float("nan"), "time": 1.0, "position": 1.0},
        {"event": 1.0, "time": 1.0},
    ],
)
def test_joint_weight_validation(
    artifacts: ProtocolArtifacts, batch: dict, weights: dict[str, float]
) -> None:
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    model = TargetStudyHGT(artifacts, "joint", "joint_test", methods, dropout=0.0)
    with pytest.raises(ValueError):
        compute_loss(
            model(batch), batch, "joint", "joint_test", artifacts, methods, weights
        )


def test_gradient_diagnostics_are_weighted_and_bounded(
    artifacts: ProtocolArtifacts, batch: dict
) -> None:
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    weights = {"event": 0.1, "time": 1.0, "position": 1.0}
    torch.manual_seed(41)
    model = TargetStudyHGT(artifacts, "joint", "joint_test", methods, dropout=0.0)
    result = compute_gradient_diagnostics(
        model, batch, artifacts, "joint_test", methods, weights
    )
    assert result["weights"] == weights
    assert result["effective_gradient_norm"]["event"] == pytest.approx(
        result["raw_gradient_norm"]["event"] * 0.1
    )
    assert result["effective_gradient_norm"]["time"] == pytest.approx(
        result["raw_gradient_norm"]["time"]
    )
    assert all(value >= 0 for value in result["raw_gradient_norm"].values())
    assert all(-1.0 <= value <= 1.0 for value in result["gradient_cosine"].values())


def test_joint_loss_scales_share_backbone_initialization(
    artifacts: ProtocolArtifacts,
) -> None:
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    hashes = []
    for _ in (0.05, 0.10, 0.20):
        torch.manual_seed(20260715)
        model = TargetStudyHGT(artifacts, "joint", "joint_test", methods)
        hashes.append(backbone_state_hash(model))
    assert len(set(hashes)) == 1
