from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from football_benchmark.data import (
    CanonicalEventDataset,
    _relative_event_features,
    collate_semantic_hgt,
    load_records,
)
from football_benchmark.losses import compute_benchmark_loss
from football_benchmark.models import ModelSpec, build_model
from football_benchmark.semantic_graph import (
    SEMANTIC_EDGE_TYPES,
    TEMPORAL_RELATIONS,
    convert_graph,
    edge_key,
    temporal_relation,
    validate_semantic_graph,
)


def _semantic_samples(artifacts, indices=(20, 21), window_size=8):
    dataset = CanonicalEventDataset(
        load_records("train")[:1], artifacts, window_size=window_size
    )
    source = dataset[0].graph
    semantic = convert_graph(source)
    return [replace(dataset[index], graph=semantic) for index in indices], source


@pytest.mark.parametrize(
    ("delta", "period_changed", "expected"),
    [
        (0.0, False, "gap_0_2s"),
        (2.0, False, "gap_2_5s"),
        (5.0, False, "gap_5_15s"),
        (15.0, False, "gap_15_60s"),
        (61.0, False, "gap_60plus"),
        (1.0, True, "period_break"),
    ],
)
def test_temporal_relation_boundaries(delta, period_changed, expected) -> None:
    assert temporal_relation(delta, period_changed) == expected


def test_semantic_conversion_preserves_targets_and_partitions_next(artifacts) -> None:
    samples, source = _semantic_samples(artifacts, indices=(20,))
    graph = samples[0].graph
    assert validate_semantic_graph(graph, source) == []
    assert set(graph["node_stores"]) == {
        "event",
        "player",
        "team",
        "event_type",
        "tag",
        "zone",
    }
    assert torch.equal(graph["targets"]["event_type_index"], source["targets"]["event_type_index"])
    next_edges = graph["edge_stores"][edge_key(("event", "next", "event"))]["edge_index"]
    temporal_edges = torch.cat(
        [
            graph["edge_stores"][edge_key(("event", relation, "event"))]["edge_index"]
            for relation in TEMPORAL_RELATIONS
        ],
        dim=1,
    )
    assert sorted(map(tuple, temporal_edges.t().tolist())) == sorted(
        map(tuple, next_edges.t().tolist())
    )


def test_semantic_window_is_causal_and_anchor_features_are_zero(artifacts) -> None:
    samples, _ = _semantic_samples(artifacts, indices=(20, 21), window_size=8)
    batch = collate_semantic_hgt(samples, artifacts, window_size=8)
    graph = batch["graph"]
    anchors = graph["event"].relative_features[graph["event"].ptr[1:] - 1]
    assert torch.allclose(anchors[:, :6], torch.zeros_like(anchors[:, :6]))
    assert torch.all(anchors[:, 6] == 1)
    assert batch["sample_ids"] == [f"{samples[0].match_id}:20", f"{samples[1].match_id}:21"]
    for edge_type in SEMANTIC_EDGE_TYPES:
        index = graph[edge_type].edge_index
        if "event" in (edge_type[0], edge_type[2]) and index.numel():
            assert int(index.max()) < max(graph[edge_type[0]].num_nodes, graph[edge_type[2]].num_nodes)


def test_relative_position_uses_current_team_view() -> None:
    event = {
        "absolute_seconds": torch.tensor([1.0, 2.0]),
        "start_position": torch.tensor([[0.2, 0.3], [0.8, 0.7]]),
        "start_position_mask": torch.tensor([True, True]),
        "end_position": torch.zeros((2, 2)),
        "end_position_mask": torch.tensor([False, False]),
        "team_local_index": torch.tensor([0, 1]),
    }
    features = _relative_event_features(event, 0, 2, 80)
    assert torch.allclose(features[:, 3:6], torch.zeros((2, 3)))
    assert features[:, 6].tolist() == [1.0, 1.0]


def test_semantic_hgt_updates_every_entity_path(artifacts) -> None:
    pytest.importorskip("torch_geometric")
    samples, _ = _semantic_samples(artifacts, indices=(20, 21), window_size=8)
    batch = collate_semantic_hgt(samples, artifacts, window_size=8)
    model = build_model(
        ModelSpec("hgt", "unified_lem", 8, graph_variant="semantic_v2"), artifacts
    )
    predictions = model(batch)
    loss, _ = compute_benchmark_loss(
        predictions, batch, "hgt", "unified_lem", artifacts
    )
    loss.backward()
    embeddings = (
        model.player_embedding,
        model.team_embedding,
        model.type_node_embedding,
        model.tag_embedding,
        model.zone_embedding,
    )
    assert all(
        embedding.weight.grad is not None
        and float(embedding.weight.grad.norm()) > 0.0
        for embedding in embeddings
    )
    assert predictions["event_logits"].shape == (2, 10)
    assert predictions["position_xy"].shape == (2, 2)
    assert predictions["time_seconds"].shape == (2,)

    model.eval()
    with torch.no_grad():
        full = model(batch)["event_logits"]
        batch["disabled_relation_families"] = ("zone",)
        without_zone = model(batch)["event_logits"]
    assert not torch.allclose(full, without_zone)
