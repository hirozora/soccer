from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from football_benchmark.data import CanonicalEventDataset, collate_semantic_hgt, load_records
from football_benchmark.protocol import ProtocolArtifacts
from football_benchmark.sampling import TargetSamplePlan
from football_hgt_targets_v4.constants import (
    FEASIBILITY_ARTIFACT,
    POSSESSION_GRAPH_ROOT,
    SAMPLE_PLAN,
    SEMANTIC_GRAPH_ROOT,
    WINDOW_SIZE,
)
from football_hgt_targets_v4.losses import compute_loss
from football_hgt_targets_v4.model import build_target_model
from football_hgt_targets_v4.possession_data import collate_possession_hgt


@pytest.fixture(scope="module")
def artifacts() -> ProtocolArtifacts:
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)


def _batch(root, artifacts, collate):
    selected = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=root),
        artifacts,
        WINDOW_SIZE,
        max_samples=2,
        selected_currents=selected,
    )
    return next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))


def _v3_batch(artifacts, topology: str, feature_level: str):
    return _batch(
        POSSESSION_GRAPH_ROOT,
        artifacts,
        partial(
            collate_possession_hgt,
            artifacts=artifacts,
            window_size=WINDOW_SIZE,
            topology=topology,
            feature_level=feature_level,
            snapshot_scope="selected_events",
        ),
    )


def test_n0_is_exact_semantic_v2_regression(artifacts) -> None:
    v2 = _batch(
        SEMANTIC_GRAPH_ROOT,
        artifacts,
        partial(collate_semantic_hgt, artifacts=artifacts, window_size=WINDOW_SIZE),
    )
    n0 = _v3_batch(artifacts, "none", "topology")
    assert v2["sample_ids"] == n0["sample_ids"]
    assert set(v2["graph"].edge_index_dict) == set(n0["graph"].edge_index_dict)
    for edge_type in v2["graph"].edge_index_dict:
        assert torch.equal(
            v2["graph"].edge_index_dict[edge_type],
            n0["graph"].edge_index_dict[edge_type],
        )
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    torch.manual_seed(20260715)
    baseline = build_target_model(
        artifacts,
        "joint",
        "j2_020",
        methods,
        graph_variant="semantic_v2",
        possession_topology="none",
        possession_feature_level="topology",
        dropout=0.0,
    )
    torch.manual_seed(20260715)
    candidate = build_target_model(
        artifacts,
        "joint",
        "j2_020",
        methods,
        graph_variant="semantic_v3_possession",
        possession_topology="none",
        possession_feature_level="topology",
        dropout=0.0,
    )
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        expected = baseline(v2)
        observed = candidate(n0)
    for key in expected:
        assert torch.max(torch.abs(expected[key] - observed[key])).item() < 1e-6


@pytest.mark.parametrize(
    ("topology", "owner_enabled", "next_enabled"),
    [
        ("membership", False, False),
        ("owner", True, False),
        ("transition", True, True),
    ],
)
def test_topology_relations_are_incremental(
    artifacts, topology: str, owner_enabled: bool, next_enabled: bool
) -> None:
    batch = _v3_batch(artifacts, topology, "topology")
    graph = batch["graph"]
    assert graph[("event", "belongs_to", "possession")].edge_index.shape[1] > 0
    assert (
        graph[("possession", "owned_by", "team")].edge_index.shape[1] > 0
    ) is owner_enabled
    assert (
        graph[("possession", "next", "possession")].edge_index.shape[1] > 0
    ) is next_enabled


@pytest.mark.parametrize("feature_level", ["topology", "categorical", "dynamic"])
def test_possession_variants_forward_backward(artifacts, feature_level: str) -> None:
    batch = _v3_batch(artifacts, "transition", feature_level)
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    torch.manual_seed(7)
    model = build_target_model(
        artifacts,
        "joint",
        "j2_020",
        methods,
        graph_variant="semantic_v3_possession",
        possession_topology="transition",
        possession_feature_level=feature_level,
        dropout=0.0,
    )
    predictions = model(batch)
    loss, _ = compute_loss(
        predictions,
        batch,
        "joint",
        "j2_020",
        artifacts,
        methods,
        {"event": 0.2, "time": 1.0, "position": 1.0},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.possession_base.grad is not None
    assert torch.isfinite(model.possession_base.grad).all()
    if feature_level == "dynamic":
        assert any(
            parameter.grad is not None
            for parameter in model.possession_dynamic_projection.parameters()
        )


def test_unassigned_window_has_fixed_empty_possession_features(artifacts) -> None:
    record = next(
        item
        for item in load_records("train", graph_root=POSSESSION_GRAPH_ROOT)
        if item.match_id == 2499725
    )
    dataset = CanonicalEventDataset(
        [record], artifacts, WINDOW_SIZE, selected_currents={record.match_id: [0]}
    )
    batch = collate_possession_hgt(
        [dataset[0]],
        artifacts,
        WINDOW_SIZE,
        topology="membership",
        feature_level="topology",
        snapshot_scope="selected_events",
    )
    possession = batch["graph"]["possession"]
    assert possession.num_nodes == 0
    assert possession.current_position.shape == (0, 2)
    assert possession.duration_so_far_seconds.shape == (0,)
    methods = {"event": "ce", "time": "current_huber", "position": "xy"}
    model = build_target_model(
        artifacts,
        "joint",
        "j2_020",
        methods,
        graph_variant="semantic_v3_possession",
        possession_topology="membership",
        possession_feature_level="topology",
        dropout=0.0,
    )
    predictions = model(batch)
    assert predictions["event_logits"].shape == (1, 10)
