from __future__ import annotations

from functools import partial

import pytest
import pandas as pd
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
from football_hgt_targets_v4.possession_data import collate_possession_hgt
from football_hgt_targets_v4.subgraph_views import (
    align_to_anchor_team,
    metric_distances,
    representative_positions,
    select_event_indices,
)
from football_hgt_targets_v4.subgraph_reporting import _passes_trigger


def _toy_graph() -> dict:
    count = 16
    possession = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    periods = torch.tensor([0] * 8 + [1] * 8)
    starts = torch.stack((torch.linspace(0.0, 0.75, count), torch.full((count,), 0.5)), dim=1)
    return {
        "match_id": 1,
        "node_stores": {
            "event": {
                "num_nodes": count,
                "absolute_seconds": torch.arange(count, dtype=torch.float32),
                "period_index": periods,
                "start_position": starts,
                "start_position_mask": torch.ones(count, dtype=torch.bool),
                "end_position": starts.clone(),
                "end_position_mask": torch.zeros(count, dtype=torch.bool),
                "team_local_index": torch.tensor([0, 0, 1, 0, 0, 1, 0, 1] * 2),
                "possession_local_index": possession,
                "active_possession_after_local_index": possession,
                "event_role_index": torch.tensor([1, 1, 4, 2, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1]),
                "switch_confirmed": torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool),
            },
            "possession": {
                "num_nodes": 5,
                "period_index": torch.tensor([0, 0, 0, 1, 1]),
                "start_event_index": torch.tensor([0, 3, 6, 8, 12]),
            },
        },
        "edge_stores": {
            "possession__next__possession": {
                "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
                "transition_event_index": torch.tensor([3, 6, 8, 12]),
                "cross_period": torch.tensor([0, 0, 1, 0], dtype=torch.bool),
            }
        },
    }


def test_possession_views_do_not_cross_period() -> None:
    graph = _toy_graph()
    p2 = select_event_indices(graph, 12, "p2")
    assert p2.event_indices.tolist() == [8, 9, 10, 11, 12]
    assert not bool((graph["node_stores"]["event"]["period_index"][p2.event_indices] == 0).any())


def test_transition_marker_types_and_fallback() -> None:
    graph = _toy_graph()
    boundary = select_event_indices(graph, 10, "tr5")
    assert boundary.marker_type == "boundary_context"
    assert 9 in boundary.event_indices
    confirmed = select_event_indices(graph, 13, "tr5")
    assert confirmed.marker_type == "confirmed_transition"
    assert 11 in confirmed.event_indices
    graph["node_stores"]["event"]["event_role_index"][8:] = 1
    fallback = select_event_indices(graph, 15, "tr5")
    assert fallback.marker_type == "fallback"
    assert fallback.fallback_reason == "no_transition_marker"


def test_spatial_geometry_transform_and_metric_boundaries() -> None:
    position = torch.tensor([[0.2, 0.3], [0.8, 0.7]])
    teams = torch.tensor([0, 1])
    aligned = align_to_anchor_team(position, teams, anchor_team=0)
    assert torch.allclose(aligned[0], position[0])
    assert torch.allclose(aligned[1], position[0])
    assert torch.allclose(
        align_to_anchor_team(1.0 - position[1:], torch.tensor([1]), anchor_team=0),
        position[1:],
    )
    assert metric_distances(aligned, aligned[0]).tolist() == pytest.approx(
        [0.0, 0.0], abs=1e-5
    )
    assert metric_distances(torch.tensor([[1.0, 0.0]]), torch.tensor([0.0, 0.0])).item() == pytest.approx(105.0)
    assert metric_distances(torch.tensor([[0.0, 1.0]]), torch.tensor([0.0, 0.0])).item() == pytest.approx(68.0)
    for radius in (15.0, 30.0, 45.0):
        point = torch.tensor([[radius / 105.0, 0.0]])
        assert metric_distances(point, torch.zeros(2)).item() <= radius + 1e-5


def test_representative_position_prefers_end_and_falls_back_to_start() -> None:
    event = {
        "start_position": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "start_position_mask": torch.tensor([1, 1], dtype=torch.bool),
        "end_position": torch.tensor([[0.8, 0.9], [0.7, 0.6]]),
        "end_position_mask": torch.tensor([1, 0], dtype=torch.bool),
    }
    positions, valid = representative_positions(event)
    assert torch.allclose(positions, torch.tensor([[0.8, 0.9], [0.3, 0.4]]))
    assert valid.tolist() == [True, True]


def test_random_control_is_deterministic_and_size_matched() -> None:
    graph = _toy_graph()
    semantic = select_event_indices(graph, 15, "p2")
    first = select_event_indices(graph, 15, "random_p2")
    second = select_event_indices(graph, 15, "random_p2")
    assert first.event_count == semantic.event_count
    assert torch.equal(first.event_indices, second.event_indices)
    assert int(first.event_indices[-1]) == 15


@pytest.mark.parametrize(("semantic", "recency"), [("p1", "lp1"), ("p2", "lp2")])
def test_matched_recency_is_contiguous_same_period_and_size_matched(
    semantic: str, recency: str
) -> None:
    graph = _toy_graph()
    possession = select_event_indices(graph, 15, semantic)
    recent = select_event_indices(graph, 15, recency)
    assert recent.event_count == possession.event_count
    assert recent.event_indices.tolist() == list(
        range(16 - recent.event_count, 16)
    )
    periods = graph["node_stores"]["event"]["period_index"][recent.event_indices]
    assert torch.unique(periods).tolist() == [1]
    assert recent.gap_rate == 0.0
    assert recent.fallback_reason == f"event_count_matched:{semantic}"


@pytest.fixture(scope="module")
def real_batch() -> dict:
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    plan = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("validation")
    dataset = CanonicalEventDataset(
        load_records("validation", graph_root=POSSESSION_GRAPH_ROOT),
        artifacts,
        WINDOW_SIZE,
        max_samples=2,
        selected_currents=plan,
    )
    collate = partial(
        collate_possession_hgt,
        artifacts=artifacts,
        window_size=WINDOW_SIZE,
        topology="membership",
        feature_level="dynamic",
        snapshot_scope="selected_events",
        context_view="sp30",
    )
    return next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))


def test_sparse_collate_preserves_source_gaps_and_real_next_edges(real_batch) -> None:
    graph = real_batch["graph"]
    source = graph["event"].source_index
    edge = graph[("event", "next", "event")].edge_index
    assert bool((source[edge[1]] - source[edge[0]] == 1).all())
    ptr = graph["event"].ptr
    for start, stop in zip(ptr[:-1], ptr[1:]):
        local_source = source[start:stop]
        denominator = max(79, int(local_source[-1] - local_source[0]))
        expected = (local_source.float() - float(local_source[-1])) / denominator
        assert torch.allclose(graph["event"].relative_features[start:stop, 0], expected)


def test_round_b_requires_direction_ci_and_practical_effect() -> None:
    significant = {"event_macro_f1": {"ci95": [0.001, 0.010]}}
    assert _passes_trigger(
        "event_macro_f1",
        pd.Series([0.006, 0.006, 0.004]),
        significant,
        0.005,
    )
    assert not _passes_trigger(
        "event_macro_f1",
        pd.Series([0.001, 0.002, 0.001]),
        significant,
        0.005,
    )
    crossing = {"event_macro_f1": {"ci95": [-0.001, 0.010]}}
    assert not _passes_trigger(
        "event_macro_f1",
        pd.Series([0.010, 0.008, 0.006]),
        crossing,
        0.005,
    )
