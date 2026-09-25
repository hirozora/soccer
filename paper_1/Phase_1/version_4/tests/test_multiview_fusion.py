from __future__ import annotations

from functools import partial

import pandas as pd
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
from football_hgt_targets_v4.losses import compute_loss
from football_hgt_targets_v4.model import build_multiview_model
from football_hgt_targets_v4.multiview_reporting import _task_effective
from football_hgt_targets_v4.possession_data import collate_multiview_possession_hgt
from football_hgt_targets_v4.training import backbone_state_hash


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


@pytest.fixture(scope="module")
def multiview_batch(artifacts: ProtocolArtifacts) -> dict:
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=2,
        selected_currents=selected,
    )
    collate = partial(
        collate_multiview_possession_hgt,
        artifacts=artifacts,
        window_size=WINDOW_SIZE,
        topology="membership",
        feature_level="dynamic",
        snapshot_scope="selected_events",
        context_views=("f80", "p1", "p2", "lp1", "lp2"),
    )
    return next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))


def test_multiview_batch_aligns_samples_targets_and_anchors(multiview_batch) -> None:
    assert set(multiview_batch["graphs"]) == {"f80", "p1", "p2", "lp1", "lp2"}
    assert len(multiview_batch["sample_ids"]) == 2
    assert multiview_batch["targets"]["raw_event_10"].shape[0] == 2
    for graph in multiview_batch["graphs"].values():
        assert int(graph.num_graphs) == 2


def test_modes_share_one_backbone_and_only_soft_modes_add_scalars(artifacts) -> None:
    torch.manual_seed(19)
    baseline = build_multiview_model(artifacts, "f80", dropout=0.0)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    baseline_hash = backbone_state_hash(baseline)
    expected_extra = {
        "fixed_a": 0, "fixed_b": 0, "sf_a": 4, "sf_b": 6,
        "recency_sf_b": 6,
    }
    for mode, extra in expected_extra.items():
        torch.manual_seed(19)
        model = build_multiview_model(artifacts, mode, dropout=0.0)
        assert sum(parameter.numel() for parameter in model.parameters()) == baseline_count + extra
        assert backbone_state_hash(model) == baseline_hash
        assert len(model.convolutions) == 2


def test_fixed_routing_and_soft_initialization(artifacts) -> None:
    contexts = {
        "f80": torch.full((2, 64), 80.0),
        "p1": torch.full((2, 64), 1.0),
        "p2": torch.full((2, 64), 2.0),
    }
    fixed_a = build_multiview_model(artifacts, "fixed_a", dropout=0.0)
    routed = fixed_a.task_contexts(contexts)
    assert routed["event"] is contexts["f80"]
    assert routed["time"] is contexts["p2"]
    assert routed["position"] is contexts["p1"]
    sf_b = build_multiview_model(artifacts, "sf_b", dropout=0.0)
    weights = sf_b.fusion_weights()
    assert weights["event"].tolist() == pytest.approx([0.1, 0.9])
    assert weights["time"].tolist() == pytest.approx([0.1, 0.9])
    assert weights["position"].tolist() == pytest.approx([0.9, 0.1])
    assert all(float(value.sum()) == pytest.approx(1.0) for value in weights.values())
    recency = build_multiview_model(artifacts, "recency_sf_b", dropout=0.0)
    assert recency.required_views == ("lp1", "lp2")
    for task, value in recency.fusion_weights().items():
        assert value.tolist() == pytest.approx(weights[task].tolist())


@pytest.mark.parametrize(
    "mode", ["fixed_a", "fixed_b", "sf_a", "sf_b", "recency_sf_b"]
)
def test_multiview_forward_backward(
    artifacts: ProtocolArtifacts, multiview_batch: dict, mode: str
) -> None:
    torch.manual_seed(5)
    model = build_multiview_model(artifacts, mode, dropout=0.0)
    batch = {
        **multiview_batch,
        "graphs": {
            view: multiview_batch["graphs"][view]
            for view in model.required_views
        },
    }
    predictions = model(batch)
    loss, _ = compute_loss(
        predictions,
        batch,
        "joint",
        mode,
        artifacts,
        {"event": "ce", "time": "current_huber", "position": "xy"},
        {"event": 0.2, "time": 1.0, "position": 1.0},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.possession_base.grad is not None
    for parameter in model.fusion_logits.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_task_effectiveness_is_task_local() -> None:
    differences = pd.DataFrame(
        {
            "time_mae_seconds": [-0.02, -0.015, 0.001],
            "position_distance_mae_m": [-0.1, -0.08, -0.05],
        }
    )
    bootstrap = {
        "time_mae_seconds": {"ci95": [-0.03, -0.002]},
        "position_distance_mae_m": {"ci95": [-0.2, -0.01]},
    }
    assert _task_effective("time", differences, bootstrap)["effective"]
    assert not _task_effective("position", differences, bootstrap)["effective"]
