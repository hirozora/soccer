from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from football_benchmark.constants import POSSESSION_V2_ROOT
from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.possession_graph import (
    BINARY_MASK_FIELDS,
    CONTINUOUS_FEATURE_FIELDS,
    REFERENCE_ONLY_FIELDS,
    build_possession_vocabularies,
    convert_match_to_possession_graph,
    edge_key,
    extract_causal_subgraph,
    validate_possession_graph,
)
from football_benchmark import possession_graph as possession_graph_module
from football_benchmark.semantic_graph import convert_graph


def _representative(event: dict, index: int) -> tuple[float, float]:
    if bool(event["end_position_mask"][index]):
        return tuple(float(value) for value in event["end_position"][index])
    return tuple(float(value) for value in event["start_position"][index])


def _synthetic_possession_graph(artifacts):
    dataset = CanonicalEventDataset(
        load_records("train")[:1], artifacts, window_size=8
    )
    source = dataset[0].graph
    semantic = convert_graph(source)
    event = semantic["node_stores"]["event"]
    team_raw = semantic["node_stores"]["team"]["raw_id"].tolist()
    num_events = int(event["num_nodes"])

    event_to_possession: list[int] = []
    starts: list[int] = []
    current_possession = -1
    previous_period = None
    count_in_possession = 4
    for index in range(num_events):
        period = int(event["period_index"][index])
        if period != previous_period or count_in_possession >= 4:
            current_possession += 1
            starts.append(index)
            count_in_possession = 0
        event_to_possession.append(current_possession)
        count_in_possession += 1
        previous_period = period
    num_possessions = current_possession + 1

    owner_by_possession: list[int] = []
    for possession_index, start in enumerate(starts):
        if possession_index == 0:
            selected_actor = int(event["team_local_index"][min(start + 1, num_events - 1)])
            owner_local = 1 - selected_actor
        else:
            owner_local = possession_index % len(team_raw)
        owner_by_possession.append(int(team_raw[owner_local]))

    end_by_possession = [num_events - 1] * num_possessions
    for possession_index in range(num_possessions - 1):
        end_by_possession[possession_index] = starts[possession_index + 1] - 1

    state_rows = []
    for index, possession_index in enumerate(event_to_possession):
        start = starts[possession_index]
        end = end_by_possession[possession_index]
        uid = f"{semantic['match_id']}:{possession_index}"
        owner = owner_by_possession[possession_index]
        actor = int(team_raw[int(event["team_local_index"][index])])
        start_xy = _representative(event, start)
        current_xy = _representative(event, index)
        closed = index == end
        role = "contest" if index == 1 else "control"
        state_rows.append(
            {
                "match_id": semantic["match_id"],
                "event_uid": int(event["raw_id"][index]),
                "event_index": index,
                "period": "1H" if int(event["period_index"][index]) == 0 else "2H",
                "event_team_id": actor,
                "possession_uid": uid,
                "possession_index": possession_index,
                "possession_owner_team_id": owner,
                "event_role": role,
                "actor_relation_to_owner": "owner" if actor == owner else "opponent",
                "control_state_before": "CONTROL" if index > start else "DEAD_BALL",
                "control_state_after": "DEAD_BALL" if closed else "CONTROL",
                "candidate_status_before_event": "none",
                "candidate_status_after_event": "conflicting" if index == 1 else "none",
                "active_possession_after_event": None if closed else uid,
                "closed_possession_before_event": None,
                "closed_possession_after_event": uid if closed else None,
                "switch_confirmed": index == start and possession_index > 0,
                "duration_so_far_seconds": max(
                    float(event["absolute_seconds"][index])
                    - float(event["absolute_seconds"][start]),
                    0.0,
                ),
                "event_count_so_far": index - start + 1,
                "start_x": start_xy[0],
                "start_y": start_xy[1],
                "current_x": current_xy[0],
                "current_y": current_xy[1],
                "is_closed_as_of_event": closed,
            }
        )
    possession_rows = []
    for possession_index, start in enumerate(starts):
        start_xy = _representative(event, start)
        possession_rows.append(
            {
                "match_id": semantic["match_id"],
                "possession_uid": f"{semantic['match_id']}:{possession_index}",
                "possession_index": possession_index,
                "owner_team_id": owner_by_possession[possession_index],
                "period": "1H" if int(event["period_index"][start]) == 0 else "2H",
                "start_event_uid": int(event["raw_id"][start]),
                "start_event_index": start,
                "start_seconds": float(event["absolute_seconds"][start]),
                "start_x": start_xy[0],
                "start_y": start_xy[1],
                "start_reason": "control:Simple pass",
            }
        )
    transition_rows = []
    for possession_index in range(1, num_possessions):
        start = starts[possession_index]
        period_changed = (
            int(event["period_index"][start])
            != int(event["period_index"][starts[possession_index - 1]])
        )
        transition_rows.append(
            {
                "match_id": semantic["match_id"],
                "previous_possession_uid": f"{semantic['match_id']}:{possession_index - 1}",
                "next_possession_uid": f"{semantic['match_id']}:{possession_index}",
                "transition_event_uid": int(event["raw_id"][start]),
                "transition_event_index": start,
                "transition_reason": "period_break" if period_changed else "confirmed_opponent_control",
            }
        )
    vocabularies = build_possession_vocabularies(
        Path(POSSESSION_V2_ROOT) / "possession_rules_v2.csv"
    )
    states = pd.DataFrame(state_rows)
    possessions = pd.DataFrame(possession_rows)
    transitions = pd.DataFrame(transition_rows)
    graph = convert_match_to_possession_graph(
        semantic, states, possessions, transitions, vocabularies
    )
    return graph, semantic, states, possessions, transitions


def test_schema_separates_reference_continuous_and_binary_fields() -> None:
    assert set(REFERENCE_ONLY_FIELDS).isdisjoint(CONTINUOUS_FEATURE_FIELDS)
    assert set(REFERENCE_ONLY_FIELDS).isdisjoint(BINARY_MASK_FIELDS)
    assert set(CONTINUOUS_FEATURE_FIELDS).isdisjoint(BINARY_MASK_FIELDS)
    assert "possession_index" in REFERENCE_ONLY_FIELDS
    assert "dynamic_feature_mask" in BINARY_MASK_FIELDS


def test_model_safe_reader_never_opens_audit_table(monkeypatch, tmp_path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "model_safe_contract.json").write_text(
        """{
          "allowed_tables": [
            "event_possession_states.parquet",
            "possessions.parquet",
            "possession_transitions.parquet"
          ],
          "forbidden_tables": ["possession_audit.parquet"]
        }""",
        encoding="utf-8",
    )
    opened: list[str] = []

    def fake_read(path):
        opened.append(Path(path).name)
        return pd.DataFrame()

    monkeypatch.setattr(possession_graph_module, "_require_pyarrow_24", lambda: None)
    monkeypatch.setattr(possession_graph_module.pd, "read_parquet", fake_read)
    possession_graph_module._read_model_safe_tables(tmp_path)
    assert opened == [
        "event_possession_states.parquet",
        "possessions.parquet",
        "possession_transitions.parquet",
    ]


def test_conversion_preserves_semantic_graph_and_adds_exact_reverse_edges(artifacts) -> None:
    graph, semantic, states, possessions, transitions = _synthetic_possession_graph(artifacts)
    assert validate_possession_graph(
        graph,
        semantic_graph=semantic,
        states=states,
        possessions=possessions,
        transitions=transitions,
    ) == []
    membership = graph["edge_stores"][edge_key(("event", "belongs_to", "possession"))][
        "edge_index"
    ]
    contains = graph["edge_stores"][edge_key(("possession", "contains", "event"))][
        "edge_index"
    ]
    owned = graph["edge_stores"][edge_key(("possession", "owned_by", "team"))][
        "edge_index"
    ]
    reverse_owned = graph["edge_stores"][edge_key(("team", "has_possession", "possession"))][
        "edge_index"
    ]
    assert torch.equal(membership, contains.flip(0))
    assert torch.equal(owned, reverse_owned.flip(0))
    assert torch.equal(graph["targets"]["event_type_index"], semantic["targets"]["event_type_index"])
    transition = graph["edge_stores"][edge_key(("possession", "next", "possession"))]
    possession = graph["node_stores"]["possession"]
    expected_cross_period = (
        possession["period_index"][transition["edge_index"][0]]
        != possession["period_index"][transition["edge_index"][1]]
    )
    assert torch.equal(transition["cross_period"], expected_cross_period)


def test_selected_events_masks_unselected_start_and_dynamic_history(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    selected = extract_causal_subgraph(
        graph,
        [1, 2],
        2,
        snapshot_scope="selected_events",
        include_dynamic_possession_features=True,
    )
    possession = selected["node_stores"]["possession"]
    event = graph["node_stores"]["event"]
    assert possession["source_index"].tolist() == [0]
    assert possession["true_start_visible"].tolist() == [False]
    assert possession["start_reason_index"].tolist() == [0]
    assert torch.isclose(
        possession["start_absolute_seconds"][0], event["absolute_seconds"][1]
    )
    assert possession["event_count_so_far"].tolist() == [2.0]
    expected_duration = float(event["absolute_seconds"][2] - event["absolute_seconds"][1])
    assert float(possession["duration_so_far_seconds"][0]) == pytest.approx(
        max(expected_duration, 0.0), abs=1e-5
    )


def test_anchor_history_uses_full_prefix_snapshot(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    selected = extract_causal_subgraph(
        graph,
        [1, 2],
        2,
        snapshot_scope="anchor_history",
        include_dynamic_possession_features=True,
    )
    possession = selected["node_stores"]["possession"]
    assert possession["true_start_visible"].tolist() == [True]
    assert possession["snapshot_event_index"].tolist() == [2]
    assert possession["event_count_so_far"].tolist() == [3.0]
    assert possession["dynamic_feature_mask"].tolist() == [True]


def test_topology_only_zeros_dynamic_features(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    selected = extract_causal_subgraph(
        graph,
        [1],
        1,
        snapshot_scope="selected_events",
        include_dynamic_possession_features=False,
    )
    possession = selected["node_stores"]["possession"]
    assert possession["dynamic_feature_mask"].tolist() == [False]
    assert possession["duration_so_far_seconds"].tolist() == [0.0]
    assert possession["event_count_so_far"].tolist() == [0.0]
    assert torch.equal(possession["current_position"], torch.zeros((1, 2)))
    assert possession["is_current_active"].tolist() == [False]


def test_relation_closure_keeps_possession_owner_team(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    selected = extract_causal_subgraph(
        graph,
        [1],
        1,
        snapshot_scope="selected_events",
        include_dynamic_possession_features=False,
    )
    event_team = int(graph["node_stores"]["event"]["team_local_index"][1])
    owner_team = int(graph["node_stores"]["possession"]["owner_team_local_index"][0])
    assert event_team != owner_team
    retained_teams = set(selected["node_stores"]["team"]["source_index"].tolist())
    assert retained_teams == {event_team, owner_team}
    owned = selected["edge_stores"][edge_key(("possession", "owned_by", "team"))][
        "edge_index"
    ]
    assert owned.shape[1] == 1


def test_sparse_events_do_not_create_synthetic_next_edges(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    selected = extract_causal_subgraph(
        graph,
        [0, 2],
        2,
        snapshot_scope="selected_events",
        include_dynamic_possession_features=False,
    )
    next_edge = selected["edge_stores"][edge_key(("event", "next", "event"))][
        "edge_index"
    ]
    assert next_edge.shape == (2, 0)


def test_possession_transition_requires_effective_anchor(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    transition = graph["edge_stores"][edge_key(("possession", "next", "possession"))]
    original = int(transition["transition_event_index"][0])
    assert original == 4
    transition["transition_event_index"][0] = 5
    selected = extract_causal_subgraph(
        graph,
        [3, 4],
        4,
        snapshot_scope="selected_events",
        include_dynamic_possession_features=False,
    )
    edge = selected["edge_stores"][edge_key(("possession", "next", "possession"))][
        "edge_index"
    ]
    assert edge.shape == (2, 0)


def test_extractor_requires_explicit_valid_scope_and_anchor(artifacts) -> None:
    graph, *_ = _synthetic_possession_graph(artifacts)
    with pytest.raises(ValueError, match="final selected Event"):
        extract_causal_subgraph(
            graph,
            [0, 1],
            0,
            snapshot_scope="selected_events",
            include_dynamic_possession_features=False,
        )
    with pytest.raises(ValueError, match="snapshot_scope"):
        extract_causal_subgraph(
            graph,
            [0],
            0,
            snapshot_scope="invalid",  # type: ignore[arg-type]
            include_dynamic_possession_features=False,
        )
