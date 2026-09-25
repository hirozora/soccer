"""Construct one portable heterogeneous graph from one Wyscout match."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch

from .catalog import Catalog
from .schema import (
    EDGE_TYPES,
    NODE_TYPES,
    PERIOD_NOMINAL_SECONDS,
    PERIOD_ORDER,
    RESULT_DIRECTION_TO_INDEX,
    SCHEMA_VERSION,
    UNKNOWN_PLAYER_ID,
    UNKNOWN_SUBEVENT_ID,
    classify_result_direction,
    edge_type_key,
)


TAG_CATEGORY_TO_INDEX = {
    "technical_action": 0,
    "event_context": 1,
    "event_result": 2,
}
PLAYER_ROLE_TO_INDEX = {
    "": 0,
    "Goalkeeper": 1,
    "Defender": 2,
    "Midfielder": 3,
    "Forward": 4,
}
PLAYER_FOOT_TO_INDEX = {"": 0, "right": 1, "left": 2, "both": 3}
TEAM_SIDE_TO_INDEX = {"home": 0, "away": 1, "": 2}
TEAM_TYPE_TO_INDEX = {"": 0, "club": 1, "national": 2}
COMPETITION_TYPE_TO_INDEX = {"": 0, "club": 1, "international": 2}
MATCH_DURATION_TO_INDEX = {"Regular": 0, "ExtraTime": 1, "Penalties": 2}


def _tensor(values: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def _edge_index(sources: Iterable[int], destinations: Iterable[int]) -> torch.Tensor:
    source_values = list(sources)
    destination_values = list(destinations)
    if not source_values:
        return torch.empty((2, 0), dtype=torch.long)
    return _tensor([source_values, destination_values], torch.long)


def _ordered_match_teams(match: dict[str, Any], events: list[dict[str, Any]]) -> list[int]:
    teams_data = match.get("teamsData", {})
    side_rank = {"home": 0, "away": 1}
    team_ids = sorted(
        (int(team_id) for team_id in teams_data),
        key=lambda team_id: (
            side_rank.get(teams_data[str(team_id)].get("side", ""), 2),
            team_id,
        ),
    )
    observed = sorted({int(event["teamId"]) for event in events})
    team_ids.extend(team_id for team_id in observed if team_id not in team_ids)
    return team_ids


def _roster_player_ids(match: dict[str, Any]) -> set[int]:
    player_ids: set[int] = set()
    for team_data in match.get("teamsData", {}).values():
        formation = team_data.get("formation") or {}
        for group in ("lineup", "bench"):
            for entry in formation.get(group, []) or []:
                player_id = int(entry.get("playerId") or 0)
                if player_id:
                    player_ids.add(player_id)
    return player_ids


def _ordered_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(events))
    indexed.sort(
        key=lambda pair: (
            PERIOD_ORDER[pair[1]["matchPeriod"]],
            float(pair[1]["eventSec"]),
            pair[0],
        )
    )
    return [event for _, event in indexed]


def _absolute_times(events: list[dict[str, Any]]) -> list[float]:
    """Create a monotonic active-play clock without inserting halftime breaks."""

    max_seconds_by_period: dict[str, float] = defaultdict(float)
    for event in events:
        period = event["matchPeriod"]
        max_seconds_by_period[period] = max(
            max_seconds_by_period[period], float(event["eventSec"])
        )

    offsets: dict[str, float] = {}
    elapsed = 0.0
    for period in PERIOD_ORDER:
        if period not in max_seconds_by_period:
            continue
        offsets[period] = elapsed
        elapsed += max(PERIOD_NOMINAL_SECONDS[period], max_seconds_by_period[period])

    return [offsets[event["matchPeriod"]] + float(event["eventSec"]) for event in events]


def _position(event: dict[str, Any], index: int) -> tuple[list[float], bool]:
    positions = event.get("positions") or []
    if index >= len(positions):
        return [0.0, 0.0], False
    point = positions[index]
    return [float(point["x"]) / 100.0, float(point["y"]) / 100.0], True


def _subevent_key(raw_value: Any) -> str:
    try:
        return str(int(raw_value))
    except (TypeError, ValueError):
        return UNKNOWN_SUBEVENT_ID


def _event_tag_ids(event: dict[str, Any], catalog: Catalog) -> list[int]:
    tag_ids: list[int] = []
    for item in event.get("tags", []):
        tag_id = int(item["id"])
        if tag_id not in catalog.tag_vocab:
            raise ValueError(f"Event references undefined tag {tag_id}")
        override = catalog.tag_overrides.get((event["eventName"], tag_id))
        if override is not None and not override.build_tag_edge:
            continue
        tag_ids.append(tag_id)
    return tag_ids


def build_match_graph(
    match: dict[str, Any],
    raw_events: list[dict[str, Any]],
    catalog: Catalog,
    competition_slug: str,
) -> dict[str, Any]:
    """Build a match graph with node tensors, typed edges, and shifted targets."""

    if not raw_events:
        raise ValueError("Cannot construct a graph from an empty event list")

    match_id = int(match["wyId"])
    if any(int(event["matchId"]) != match_id for event in raw_events):
        raise ValueError(f"Event list contains a matchId other than {match_id}")

    events = _ordered_events(raw_events)
    num_events = len(events)
    absolute_times = _absolute_times(events)

    team_ids = _ordered_match_teams(match, events)
    team_local = {team_id: index for index, team_id in enumerate(team_ids)}

    observed_players = {int(event["playerId"]) for event in events}
    roster_players = _roster_player_ids(match)
    player_ids = sorted((observed_players | roster_players) - {UNKNOWN_PLAYER_ID})
    if UNKNOWN_PLAYER_ID in observed_players:
        player_ids.insert(0, UNKNOWN_PLAYER_ID)
    player_local = {player_id: index for index, player_id in enumerate(player_ids)}

    event_type_ids = [raw_id for raw_id, _ in sorted(catalog.event_type_vocab.items(), key=lambda x: x[1])]
    tag_ids = [raw_id for raw_id, _ in sorted(catalog.tag_vocab.items(), key=lambda x: x[1])]
    tag_local = {tag_id: index for index, tag_id in enumerate(tag_ids)}

    event_raw_ids: list[int] = []
    event_type_indices: list[int] = []
    subevent_type_indices: list[int] = []
    period_indices: list[int] = []
    period_seconds: list[float] = []
    start_positions: list[list[float]] = []
    start_position_masks: list[bool] = []
    end_positions: list[list[float]] = []
    end_position_masks: list[bool] = []
    event_team_local: list[int] = []
    event_player_local: list[int] = []
    event_player_vocab: list[int] = []
    result_directions: list[int] = []
    retained_tag_counts: list[int] = []

    player_sources: list[int] = []
    player_destinations: list[int] = []
    team_sources: list[int] = []
    team_destinations: list[int] = []
    event_tag_sources: list[int] = []
    event_tag_destinations: list[int] = []

    for event_index, event in enumerate(events):
        event_id = int(event["eventId"])
        player_id = int(event["playerId"])
        team_id = int(event["teamId"])
        if event_id not in catalog.event_type_vocab:
            raise ValueError(f"Unknown event type {event_id}")
        if team_id not in team_local:
            raise ValueError(f"Team {team_id} is missing from match-local team nodes")
        if player_id not in player_local:
            raise ValueError(f"Player {player_id} is missing from match-local player nodes")

        raw_tag_ids = {int(item["id"]) for item in event.get("tags", [])}
        allowed_tag_ids = _event_tag_ids(event, catalog)
        start_position, has_start = _position(event, 0)
        end_position, has_end = _position(event, 1)

        event_raw_ids.append(int(event["id"]))
        event_type_indices.append(catalog.event_type_vocab[event_id])
        subevent_type_indices.append(
            catalog.subevent_type_vocab[_subevent_key(event.get("subEventId"))]
        )
        period_indices.append(PERIOD_ORDER[event["matchPeriod"]])
        period_seconds.append(float(event["eventSec"]))
        start_positions.append(start_position)
        start_position_masks.append(has_start)
        end_positions.append(end_position)
        end_position_masks.append(has_end)
        event_team_local.append(team_local[team_id])
        event_player_local.append(player_local[player_id])
        event_player_vocab.append(catalog.player_vocab.get(player_id, 0))
        result_directions.append(
            classify_result_direction(
                event["eventName"], raw_tag_ids, catalog.result_direction_rules
            )
        )
        retained_tag_counts.append(len(allowed_tag_ids))

        player_sources.append(player_local[player_id])
        player_destinations.append(event_index)
        team_sources.append(team_local[team_id])
        team_destinations.append(event_index)
        for tag_id in allowed_tag_ids:
            event_tag_sources.append(event_index)
            event_tag_destinations.append(tag_local[tag_id])

    event_type_destinations = event_type_indices
    event_indices = list(range(num_events))
    match_indices = [0] * num_events

    player_raw_ids: list[int] = []
    player_vocab_indices: list[int] = []
    player_role_indices: list[int] = []
    player_foot_indices: list[int] = []
    player_numeric: list[list[float]] = []
    player_metadata_mask: list[bool] = []
    for player_id in player_ids:
        metadata = catalog.players.get(player_id)
        player_raw_ids.append(player_id)
        player_vocab_indices.append(catalog.player_vocab.get(player_id, 0))
        if metadata is None:
            player_role_indices.append(0)
            player_foot_indices.append(0)
            player_numeric.append([0.0, 0.0])
            player_metadata_mask.append(False)
        else:
            role_name = (metadata.get("role") or {}).get("name", "")
            foot = str(metadata.get("foot") or "").lower()
            player_role_indices.append(PLAYER_ROLE_TO_INDEX.get(role_name, 0))
            player_foot_indices.append(PLAYER_FOOT_TO_INDEX.get(foot, 0))
            player_numeric.append(
                [float(metadata.get("height") or 0.0), float(metadata.get("weight") or 0.0)]
            )
            player_metadata_mask.append(True)

    team_vocab_indices: list[int] = []
    team_side_indices: list[int] = []
    team_type_indices: list[int] = []
    for team_id in team_ids:
        metadata = catalog.teams.get(team_id, {})
        match_team = match.get("teamsData", {}).get(str(team_id), {})
        team_vocab_indices.append(catalog.team_vocab.get(team_id, -1))
        team_side_indices.append(TEAM_SIDE_TO_INDEX.get(match_team.get("side", ""), 2))
        team_type_indices.append(TEAM_TYPE_TO_INDEX.get(metadata.get("type", ""), 0))

    competition_id = int(match["competitionId"])
    competition_metadata = catalog.competitions.get(competition_id, {})

    node_stores = {
        "event": {
            "num_nodes": num_events,
            "raw_id": _tensor(event_raw_ids, torch.long),
            "event_type_index": _tensor(event_type_indices, torch.long),
            "subevent_type_index": _tensor(subevent_type_indices, torch.long),
            "period_index": _tensor(period_indices, torch.long),
            "period_seconds": _tensor(period_seconds, torch.float32),
            "absolute_seconds": _tensor(absolute_times, torch.float32),
            "delta_from_previous": _tensor(
                [0.0] + [b - a for a, b in zip(absolute_times, absolute_times[1:])],
                torch.float32,
            ),
            "start_position": _tensor(start_positions, torch.float32),
            "start_position_mask": _tensor(start_position_masks, torch.bool),
            "end_position": _tensor(end_positions, torch.float32),
            "end_position_mask": _tensor(end_position_masks, torch.bool),
            "team_local_index": _tensor(event_team_local, torch.long),
            "player_local_index": _tensor(event_player_local, torch.long),
            "player_vocab_index": _tensor(event_player_vocab, torch.long),
            "result_direction": _tensor(result_directions, torch.long),
            "retained_tag_count": _tensor(retained_tag_counts, torch.long),
        },
        "player": {
            "num_nodes": len(player_ids),
            "raw_id": _tensor(player_raw_ids, torch.long),
            "vocab_index": _tensor(player_vocab_indices, torch.long),
            "role_index": _tensor(player_role_indices, torch.long),
            "foot_index": _tensor(player_foot_indices, torch.long),
            "height_weight": _tensor(player_numeric, torch.float32),
            "metadata_mask": _tensor(player_metadata_mask, torch.bool),
        },
        "team": {
            "num_nodes": len(team_ids),
            "raw_id": _tensor(team_ids, torch.long),
            "vocab_index": _tensor(team_vocab_indices, torch.long),
            "side_index": _tensor(team_side_indices, torch.long),
            "type_index": _tensor(team_type_indices, torch.long),
        },
        "match": {
            "num_nodes": 1,
            "raw_id": _tensor([match_id], torch.long),
            "vocab_index": _tensor([catalog.match_vocab[match_id]], torch.long),
            "gameweek": _tensor([int(match.get("gameweek") or 0)], torch.long),
            "round_id": _tensor([int(match.get("roundId") or 0)], torch.long),
            "duration_index": _tensor(
                [MATCH_DURATION_TO_INDEX.get(match.get("duration", "Regular"), 0)], torch.long
            ),
        },
        "competition": {
            "num_nodes": 1,
            "raw_id": _tensor([competition_id], torch.long),
            "vocab_index": _tensor([catalog.competition_vocab[competition_id]], torch.long),
            "type_index": _tensor(
                [COMPETITION_TYPE_TO_INDEX.get(competition_metadata.get("type", ""), 0)],
                torch.long,
            ),
        },
        "event_type": {
            "num_nodes": len(event_type_ids),
            "raw_id": _tensor(event_type_ids, torch.long),
            "vocab_index": _tensor(
                [catalog.event_type_vocab[value] for value in event_type_ids], torch.long
            ),
        },
        "tag": {
            "num_nodes": len(tag_ids),
            "raw_id": _tensor(tag_ids, torch.long),
            "vocab_index": _tensor([catalog.tag_vocab[value] for value in tag_ids], torch.long),
            "category_index": _tensor(
                [TAG_CATEGORY_TO_INDEX[catalog.tags[value]["category"]] for value in tag_ids],
                torch.long,
            ),
        },
    }

    edges = {
        edge_type_key(("event", "next", "event")): _edge_index(
            range(num_events - 1), range(1, num_events)
        ),
        edge_type_key(("player", "performs", "event")): _edge_index(
            player_sources, player_destinations
        ),
        edge_type_key(("event", "performed_by", "player")): _edge_index(
            player_destinations, player_sources
        ),
        edge_type_key(("team", "performs", "event")): _edge_index(
            team_sources, team_destinations
        ),
        edge_type_key(("event", "performed_by_team", "team")): _edge_index(
            team_destinations, team_sources
        ),
        edge_type_key(("event", "in_match", "match")): _edge_index(
            event_indices, match_indices
        ),
        edge_type_key(("match", "contains", "event")): _edge_index(
            match_indices, event_indices
        ),
        edge_type_key(("match", "in_competition", "competition")): _edge_index([0], [0]),
        edge_type_key(("competition", "contains", "match")): _edge_index([0], [0]),
        edge_type_key(("event", "has_type", "event_type")): _edge_index(
            event_indices, event_type_destinations
        ),
        edge_type_key(("event_type", "describes", "event")): _edge_index(
            event_type_destinations, event_indices
        ),
        edge_type_key(("event", "has_tag", "tag")): _edge_index(
            event_tag_sources, event_tag_destinations
        ),
        edge_type_key(("tag", "describes", "event")): _edge_index(
            event_tag_destinations, event_tag_sources
        ),
    }

    target_mask = [True] * (num_events - 1) + [False]
    targets = {
        "mask": _tensor(target_mask, torch.bool),
        "event_type_index": _tensor(event_type_indices[1:] + [-1], torch.long),
        "delta_seconds": _tensor(
            [b - a for a, b in zip(absolute_times, absolute_times[1:])] + [0.0],
            torch.float32,
        ),
        "start_position": _tensor(start_positions[1:] + [[0.0, 0.0]], torch.float32),
        "start_position_mask": _tensor(start_position_masks[1:] + [False], torch.bool),
        "end_position": _tensor(end_positions[1:] + [[0.0, 0.0]], torch.float32),
        "end_position_mask": _tensor(end_position_masks[1:] + [False], torch.bool),
        "team_local_index": _tensor(event_team_local[1:] + [-1], torch.long),
        "player_vocab_index": _tensor(event_player_vocab[1:] + [-1], torch.long),
        "player_known_mask": _tensor(
            [
                player_ids[event_player_local[index]] != UNKNOWN_PLAYER_ID
                and player_ids[event_player_local[index]] in catalog.players
                for index in range(1, num_events)
            ]
            + [False],
            torch.bool,
        ),
        "result_direction": _tensor(result_directions[1:] + [-1], torch.long),
    }

    graph = {
        "schema_version": SCHEMA_VERSION,
        "graph_unit": "match",
        "match_id": match_id,
        "competition_id": competition_id,
        "competition_slug": competition_slug,
        "node_types": list(NODE_TYPES),
        "edge_types": [list(edge_type) for edge_type in EDGE_TYPES],
        "node_stores": node_stores,
        "edge_stores": {key: {"edge_index": value} for key, value in edges.items()},
        "targets": targets,
        "causal_contract": {
            "event_order": "matchPeriod,eventSec,source_order",
            "training_requires_prefix_slice": True,
            "next_edge_direction": "past_to_future",
        },
    }

    errors = validate_graph(graph)
    if errors:
        raise ValueError(f"Invalid graph for match {match_id}: {'; '.join(errors)}")
    return graph


def validate_graph(graph: dict[str, Any]) -> list[str]:
    """Return structural validation errors for a serialized match graph."""

    errors: list[str] = []
    if graph.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")

    node_stores = graph.get("node_stores", {})
    edge_stores = graph.get("edge_stores", {})
    for node_type in NODE_TYPES:
        if node_type not in node_stores:
            errors.append(f"missing node store {node_type}")

    if errors:
        return errors

    num_events = int(node_stores["event"]["num_nodes"])
    if num_events < 1:
        errors.append("event node count must be positive")

    for node_type, store in node_stores.items():
        num_nodes = int(store["num_nodes"])
        for name, value in store.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] != num_nodes:
                errors.append(
                    f"{node_type}.{name} has first dimension {value.shape[0]}, expected {num_nodes}"
                )

    expected_edge_keys = {edge_type_key(edge_type) for edge_type in EDGE_TYPES}
    if set(edge_stores) != expected_edge_keys:
        missing = expected_edge_keys - set(edge_stores)
        extra = set(edge_stores) - expected_edge_keys
        if missing:
            errors.append(f"missing edge stores {sorted(missing)}")
        if extra:
            errors.append(f"unexpected edge stores {sorted(extra)}")

    node_counts = {name: int(store["num_nodes"]) for name, store in node_stores.items()}
    for source, relation, destination in EDGE_TYPES:
        key = edge_type_key((source, relation, destination))
        if key not in edge_stores:
            continue
        edge_index = edge_stores[key]["edge_index"]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            errors.append(f"{key}.edge_index must have shape [2, E]")
            continue
        if edge_index.numel():
            if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= node_counts[source]:
                errors.append(f"{key} has out-of-bounds source index")
            if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= node_counts[destination]:
                errors.append(f"{key} has out-of-bounds destination index")

    next_edges = edge_stores[edge_type_key(("event", "next", "event"))]["edge_index"]
    expected_next = _edge_index(range(num_events - 1), range(1, num_events))
    if not torch.equal(next_edges, expected_next):
        errors.append("event next edges do not match chronological adjacency")

    tag_event_edges = edge_stores[
        edge_type_key(("tag", "describes", "event"))
    ]["edge_index"]
    tag_event_destinations = tag_event_edges[1]
    if tag_event_destinations.numel() > 1 and bool(
        torch.any(tag_event_destinations[1:] < tag_event_destinations[:-1])
    ):
        errors.append("tag->event edges are not sorted by event destination")

    absolute_seconds = node_stores["event"]["absolute_seconds"]
    if num_events > 1 and bool(torch.any(absolute_seconds[1:] < absolute_seconds[:-1])):
        errors.append("absolute event time is not monotonic")

    targets = graph.get("targets", {})
    for name, value in targets.items():
        if not isinstance(value, torch.Tensor) or value.shape[0] != num_events:
            errors.append(f"target {name} must be a tensor aligned to event nodes")
    if "mask" in targets:
        prefix = graph.get("prefix")
        is_nonterminal_prefix = bool(
            prefix
            and int(prefix["end_event_index"]) < int(prefix["original_num_events"]) - 1
        )
        expected_supervised = num_events if is_nonterminal_prefix else num_events - 1
        if (
            bool(targets["mask"][-1]) != is_nonterminal_prefix
            or int(targets["mask"].sum()) != expected_supervised
        ):
            errors.append("target mask is inconsistent with graph/prefix boundaries")

    if "result_direction" in targets:
        valid = set(RESULT_DIRECTION_TO_INDEX.values()) | {-1}
        if not set(targets["result_direction"].tolist()).issubset(valid):
            errors.append("target result_direction contains an invalid class")

    return errors
