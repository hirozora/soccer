"""Semantic V3 graph construction and causal possession subgraph extraction."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd
import torch

from .constants import (
    POSSESSION_GRAPH_ROOT,
    POSSESSION_GRAPH_VERSION,
    POSSESSION_V2_ROOT,
    SEMANTIC_GRAPH_ROOT,
)
from .semantic_graph import SEMANTIC_EDGE_TYPES, edge_key, file_sha256


SnapshotScope = Literal["anchor_history", "selected_events"]

POSSESSION_NODE_TYPES = (*("event", "player", "team", "event_type", "tag", "zone"), "possession")
POSSESSION_EDGE_TYPES = (
    *SEMANTIC_EDGE_TYPES,
    ("event", "belongs_to", "possession"),
    ("possession", "contains", "event"),
    ("possession", "owned_by", "team"),
    ("team", "has_possession", "possession"),
    ("possession", "next", "possession"),
)

REFERENCE_ONLY_FIELDS = (
    "raw_id",
    "possession_index",
    "possession_local_index",
    "owner_team_local_index",
    "active_possession_after_local_index",
    "closed_possession_before_local_index",
    "closed_possession_after_local_index",
    "start_event_index",
    "snapshot_event_index",
    "source_index",
    "transition_event_index",
)
CATEGORICAL_FEATURE_FIELDS = (
    "event_role_index",
    "actor_relation_to_owner_index",
    "control_state_before_index",
    "control_state_after_index",
    "candidate_status_before_index",
    "candidate_status_after_index",
    "start_reason_index",
    "transition_reason_index",
)
CONTINUOUS_FEATURE_FIELDS = (
    "duration_so_far_seconds",
    "event_count_so_far",
    "start_absolute_seconds",
    "start_position",
    "current_position",
)
BINARY_MASK_FIELDS = (
    "true_start_visible",
    "dynamic_feature_mask",
    "is_closed_as_of_anchor",
    "is_current_active",
    "start_position_mask",
    "current_position_mask",
    "cross_period",
)
VISIBILITY_SENSITIVE_FIELDS = (
    "start_absolute_seconds",
    "start_position",
    "start_position_mask",
    "start_reason_index",
    "duration_so_far_seconds",
    "event_count_so_far",
    "current_position",
    "current_position_mask",
    "is_closed_as_of_anchor",
    "is_current_active",
)

FIXED_CATEGORIES: dict[str, tuple[str, ...]] = {
    "event_role": ("UNK", "control", "restart", "contest", "boundary", "neutral"),
    "actor_relation_to_owner": ("UNK", "owner", "opponent", "unknown"),
    "control_state": ("UNK", "DEAD_BALL", "CONTROL", "CONTESTED"),
    "candidate_status": ("UNK", "none", "single", "conflicting"),
}

REQUIRED_STATE_COLUMNS = {
    "match_id",
    "event_uid",
    "event_index",
    "period",
    "event_team_id",
    "possession_uid",
    "possession_index",
    "possession_owner_team_id",
    "event_role",
    "actor_relation_to_owner",
    "control_state_before",
    "control_state_after",
    "candidate_status_before_event",
    "candidate_status_after_event",
    "active_possession_after_event",
    "closed_possession_before_event",
    "closed_possession_after_event",
    "switch_confirmed",
    "duration_so_far_seconds",
    "event_count_so_far",
    "start_x",
    "start_y",
    "current_x",
    "current_y",
    "is_closed_as_of_event",
}
REQUIRED_POSSESSION_COLUMNS = {
    "match_id",
    "possession_uid",
    "possession_index",
    "owner_team_id",
    "period",
    "start_event_uid",
    "start_event_index",
    "start_seconds",
    "start_x",
    "start_y",
    "start_reason",
}
REQUIRED_TRANSITION_COLUMNS = {
    "match_id",
    "previous_possession_uid",
    "next_possession_uid",
    "transition_event_uid",
    "transition_event_index",
    "transition_reason",
}


def _category_map(values: Sequence[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


def _rule_label(row: Mapping[str, str]) -> str:
    return row.get("subevent_name") or row.get("event_name") or ""


def build_possession_vocabularies(rules_path: Path) -> dict[str, Any]:
    """Build schema vocabularies from versioned rules, never event instances."""

    with Path(rules_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    starts: set[str] = set()
    transitions = {
        "confirmed_opponent_control",
        "tag_confirmed_close",
        "period_break",
    }
    for row in rows:
        if row["rule_kind"] not in {"event", "subevent"}:
            continue
        label = _rule_label(row)
        signal = row["signal"]
        if not label:
            continue
        if signal in {"control", "restart"}:
            starts.add(f"{signal}:{label}")
        if signal in {"boundary", "restart"}:
            transitions.add(f"{signal}:{label}")
    values = {name: list(categories) for name, categories in FIXED_CATEGORIES.items()}
    values["start_reason"] = ["UNK", *sorted(starts)]
    values["transition_reason"] = ["UNK", *sorted(transitions)]
    return {
        "schema_version": POSSESSION_GRAPH_VERSION,
        "values": values,
        "indices": {name: _category_map(entries) for name, entries in values.items()},
        "source": "versioned_possession_rules",
        "rules_sha256": file_sha256(Path(rules_path)),
    }


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _edge_index(sources: Iterable[int], destinations: Iterable[int]) -> torch.Tensor:
    source = torch.as_tensor(list(sources), dtype=torch.long)
    destination = torch.as_tensor(list(destinations), dtype=torch.long)
    if source.numel() != destination.numel():
        raise ValueError("Edge source and destination counts differ")
    if source.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((source, destination))


def _nullable_uid_indices(values: pd.Series, uid_to_index: Mapping[str, int]) -> torch.Tensor:
    result = torch.full((len(values),), -1, dtype=torch.long)
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        uid = str(value)
        if uid not in uid_to_index:
            raise ValueError(f"Unknown possession UID {uid}")
        result[index] = uid_to_index[uid]
    return result


def _category_tensor(values: pd.Series, mapping: Mapping[str, int], name: str) -> torch.Tensor:
    encoded: list[int] = []
    for value in values:
        key = "UNK" if pd.isna(value) else str(value)
        if key not in mapping:
            raise ValueError(f"Unknown {name} value {key!r}")
        encoded.append(mapping[key])
    return torch.tensor(encoded, dtype=torch.long)


def _float_tensor(values: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    mask = values.notna().to_numpy()
    tensor = torch.tensor(values.fillna(0.0).to_numpy(), dtype=torch.float32)
    return tensor, torch.tensor(mask, dtype=torch.bool)


def _position_tensor(x: pd.Series, y: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (x.notna() & y.notna()).to_numpy()
    positions = torch.tensor(
        list(zip(x.fillna(0.0).astype(float), y.fillna(0.0).astype(float), strict=True)),
        dtype=torch.float32,
    )
    return positions, torch.tensor(mask, dtype=torch.bool)


def _validate_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def convert_match_to_possession_graph(
    semantic_graph: dict[str, Any],
    states: pd.DataFrame,
    possessions: pd.DataFrame,
    transitions: pd.DataFrame,
    vocabularies: Mapping[str, Any],
) -> dict[str, Any]:
    """Add causal possession topology to one immutable semantic_v2 graph."""

    _validate_columns(states, REQUIRED_STATE_COLUMNS, "event_possession_states")
    _validate_columns(possessions, REQUIRED_POSSESSION_COLUMNS, "possessions")
    _validate_columns(transitions, REQUIRED_TRANSITION_COLUMNS, "possession_transitions")
    if semantic_graph.get("schema_version") != "2.0.0":
        raise ValueError("Semantic V3 conversion requires a semantic_v2 source graph")

    match_id = int(semantic_graph["match_id"])
    states = states.sort_values("event_index").reset_index(drop=True)
    possessions = possessions.sort_values("possession_index").reset_index(drop=True)
    transitions = transitions.sort_values("transition_event_index").reset_index(drop=True)
    if any(
        not frame.empty and set(frame["match_id"].astype(int)) != {match_id}
        for frame in (states, possessions, transitions)
    ):
        raise ValueError("Possession tables contain a different match")

    event = _clone(semantic_graph["node_stores"]["event"])
    num_events = int(event["num_nodes"])
    if len(states) != num_events:
        raise ValueError("Possession state count differs from Event count")
    if states["event_index"].astype(int).tolist() != list(range(num_events)):
        raise ValueError("Possession states are not contiguous in Event order")
    if states["event_uid"].astype(int).tolist() != event["raw_id"].tolist():
        raise ValueError("Possession state Event IDs do not match semantic_v2")

    num_possessions = len(possessions)
    expected_indices = list(range(num_possessions))
    if possessions["possession_index"].astype(int).tolist() != expected_indices:
        raise ValueError("Possession indices must be contiguous and zero-based")
    uid_to_index = {
        str(row.possession_uid): int(row.possession_index)
        for row in possessions.itertuples(index=False)
    }
    values = vocabularies["indices"]

    possession_local = torch.full((num_events,), -1, dtype=torch.long)
    assigned = states["possession_index"].notna()
    possession_local[torch.tensor(assigned.to_numpy(), dtype=torch.bool)] = torch.tensor(
        states.loc[assigned, "possession_index"].astype(int).to_numpy(), dtype=torch.long
    )
    if bool((possession_local >= num_possessions).any()):
        raise ValueError("Event references an out-of-range Possession")
    event["possession_local_index"] = possession_local
    event["active_possession_after_local_index"] = _nullable_uid_indices(
        states["active_possession_after_event"], uid_to_index
    )
    event["closed_possession_before_local_index"] = _nullable_uid_indices(
        states["closed_possession_before_event"], uid_to_index
    )
    event["closed_possession_after_local_index"] = _nullable_uid_indices(
        states["closed_possession_after_event"], uid_to_index
    )
    event["event_role_index"] = _category_tensor(
        states["event_role"], values["event_role"], "event_role"
    )
    event["actor_relation_to_owner_index"] = _category_tensor(
        states["actor_relation_to_owner"],
        values["actor_relation_to_owner"],
        "actor_relation_to_owner",
    )
    event["control_state_before_index"] = _category_tensor(
        states["control_state_before"], values["control_state"], "control_state"
    )
    event["control_state_after_index"] = _category_tensor(
        states["control_state_after"], values["control_state"], "control_state"
    )
    event["candidate_status_before_index"] = _category_tensor(
        states["candidate_status_before_event"],
        values["candidate_status"],
        "candidate_status",
    )
    event["candidate_status_after_index"] = _category_tensor(
        states["candidate_status_after_event"],
        values["candidate_status"],
        "candidate_status",
    )
    event["switch_confirmed"] = torch.tensor(
        states["switch_confirmed"].astype(bool).to_numpy(), dtype=torch.bool
    )
    duration, duration_mask = _float_tensor(states["duration_so_far_seconds"])
    count, count_mask = _float_tensor(states["event_count_so_far"])
    start_position, start_mask = _position_tensor(states["start_x"], states["start_y"])
    current_position, current_mask = _position_tensor(
        states["current_x"], states["current_y"]
    )
    event["possession_duration_so_far_seconds"] = duration
    event["possession_event_count_so_far"] = count
    event["possession_start_position"] = start_position
    event["possession_start_position_mask"] = start_mask
    event["possession_current_position"] = current_position
    event["possession_current_position_mask"] = current_mask
    event["possession_snapshot_mask"] = duration_mask & count_mask & current_mask
    event["possession_is_closed_as_of_event"] = torch.tensor(
        states["is_closed_as_of_event"].astype(bool).to_numpy(), dtype=torch.bool
    )

    team_raw = semantic_graph["node_stores"]["team"]["raw_id"]
    team_to_local = {int(raw): index for index, raw in enumerate(team_raw.tolist())}
    owner_local: list[int] = []
    period_index: list[int] = []
    for row in possessions.itertuples(index=False):
        owner = int(row.owner_team_id)
        if owner not in team_to_local:
            raise ValueError(f"Possession owner {owner} is not a match Team")
        start_index = int(row.start_event_index)
        if int(event["raw_id"][start_index]) != int(row.start_event_uid):
            raise ValueError("Possession start Event ID does not match start index")
        owner_local.append(team_to_local[owner])
        period_index.append(int(event["period_index"][start_index]))
    possession_start_position, possession_start_mask = _position_tensor(
        possessions["start_x"], possessions["start_y"]
    )
    possession_store = {
        "num_nodes": num_possessions,
        "raw_id": torch.arange(num_possessions, dtype=torch.long),
        "possession_index": torch.arange(num_possessions, dtype=torch.long),
        "owner_team_local_index": torch.tensor(owner_local, dtype=torch.long),
        "period_index": torch.tensor(period_index, dtype=torch.long),
        "start_event_index": torch.tensor(
            possessions["start_event_index"].astype(int).to_numpy(), dtype=torch.long
        ),
        "start_absolute_seconds": torch.tensor(
            possessions["start_seconds"].astype(float).to_numpy(), dtype=torch.float32
        ),
        "start_position": possession_start_position,
        "start_position_mask": possession_start_mask,
        "start_reason_index": _category_tensor(
            possessions["start_reason"], values["start_reason"], "start_reason"
        ),
    }

    node_stores = {
        name: (_clone(semantic_graph["node_stores"][name]) if name != "event" else event)
        for name in semantic_graph["node_types"]
    }
    node_stores["possession"] = possession_store
    edge_stores = _clone(semantic_graph["edge_stores"])
    assigned_events = torch.nonzero(possession_local >= 0, as_tuple=False).flatten()
    membership = torch.stack((assigned_events, possession_local[assigned_events]))
    edge_stores[edge_key(("event", "belongs_to", "possession"))] = {
        "edge_index": membership
    }
    edge_stores[edge_key(("possession", "contains", "event"))] = {
        "edge_index": membership.flip(0)
    }
    possession_indices = torch.arange(num_possessions, dtype=torch.long)
    owned_by = torch.stack((possession_indices, possession_store["owner_team_local_index"]))
    edge_stores[edge_key(("possession", "owned_by", "team"))] = {
        "edge_index": owned_by
    }
    edge_stores[edge_key(("team", "has_possession", "possession"))] = {
        "edge_index": owned_by.flip(0)
    }

    transition_sources: list[int] = []
    transition_destinations: list[int] = []
    for row in transitions.itertuples(index=False):
        previous_uid = str(row.previous_possession_uid)
        next_uid = str(row.next_possession_uid)
        if previous_uid not in uid_to_index or next_uid not in uid_to_index:
            raise ValueError("Transition references an unknown Possession")
        transition_sources.append(uid_to_index[previous_uid])
        transition_destinations.append(uid_to_index[next_uid])
    transition_reason = _category_tensor(
        transitions["transition_reason"], values["transition_reason"], "transition_reason"
    )
    transition_edge_index = _edge_index(transition_sources, transition_destinations)
    cross_period = (
        possession_store["period_index"][transition_edge_index[0]]
        != possession_store["period_index"][transition_edge_index[1]]
    )
    edge_stores[edge_key(("possession", "next", "possession"))] = {
        "edge_index": transition_edge_index,
        "transition_event_index": torch.tensor(
            transitions["transition_event_index"].astype(int).to_numpy(), dtype=torch.long
        ),
        "transition_reason_index": transition_reason,
        "cross_period": cross_period,
    }

    graph = {
        "schema_version": POSSESSION_GRAPH_VERSION,
        "source_schema_version": semantic_graph["schema_version"],
        "possession_source_schema_version": "2.0.0",
        "graph_unit": "match",
        "match_id": match_id,
        "competition_id": int(semantic_graph["competition_id"]),
        "competition_slug": semantic_graph["competition_slug"],
        "node_types": list(POSSESSION_NODE_TYPES),
        "edge_types": [list(value) for value in POSSESSION_EDGE_TYPES],
        "node_stores": node_stores,
        "edge_stores": edge_stores,
        "targets": _clone(semantic_graph["targets"]),
        "causal_contract": {
            "full_match_possession_features": "static_only",
            "training_requires_causal_subgraph": True,
            "transition_visibility": "transition_event_index <= anchor_event_index",
            "candidate_is_owner": False,
            "audit_table_allowed": False,
        },
    }
    errors = validate_possession_graph(
        graph,
        semantic_graph=semantic_graph,
        states=states,
        possessions=possessions,
        transitions=transitions,
    )
    if errors:
        raise ValueError(f"Invalid Semantic V3 graph {match_id}: {'; '.join(errors)}")
    return graph


def _reverse_equal(forward: torch.Tensor, reverse: torch.Tensor) -> bool:
    return torch.equal(forward, reverse.flip(0))


def validate_possession_graph(
    graph: dict[str, Any],
    *,
    semantic_graph: dict[str, Any] | None = None,
    states: pd.DataFrame | None = None,
    possessions: pd.DataFrame | None = None,
    transitions: pd.DataFrame | None = None,
) -> list[str]:
    errors: list[str] = []
    if graph.get("schema_version") != POSSESSION_GRAPH_VERSION:
        errors.append("schema_version mismatch")
    nodes = graph.get("node_stores", {})
    edges = graph.get("edge_stores", {})
    if set(nodes) != set(POSSESSION_NODE_TYPES):
        return [*errors, "node type set mismatch"]
    expected_edges = {edge_key(value) for value in POSSESSION_EDGE_TYPES}
    if set(edges) != expected_edges:
        return [*errors, "edge type set mismatch"]
    counts = {name: int(store["num_nodes"]) for name, store in nodes.items()}
    for source_type, relation, destination_type in POSSESSION_EDGE_TYPES:
        key = edge_key((source_type, relation, destination_type))
        edge_index = edges[key]["edge_index"]
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            errors.append(f"{key} has invalid edge_index")
            continue
        if edge_index.numel():
            if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= counts[source_type]:
                errors.append(f"{key} source index out of range")
            if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= counts[destination_type]:
                errors.append(f"{key} destination index out of range")
    reverse_pairs = (
        (("event", "belongs_to", "possession"), ("possession", "contains", "event")),
        (("possession", "owned_by", "team"), ("team", "has_possession", "possession")),
    )
    for forward_type, reverse_type in reverse_pairs:
        forward = edges[edge_key(forward_type)]["edge_index"]
        reverse = edges[edge_key(reverse_type)]["edge_index"]
        if not _reverse_equal(forward, reverse):
            errors.append(f"{edge_key(forward_type)} reverse mismatch")
    transition = edges[edge_key(("possession", "next", "possession"))]
    edge_count = transition["edge_index"].shape[1]
    for field in ("transition_event_index", "transition_reason_index", "cross_period"):
        if field not in transition or transition[field].shape != (edge_count,):
            errors.append(f"transition field {field} shape mismatch")
    if "cross_period" in transition and transition["cross_period"].dtype != torch.bool:
        errors.append("transition field cross_period must be bool")
    forbidden_tokens = ("audit", "duration_seconds", "final", "last_event", "close_reason")
    for node_type, store in nodes.items():
        for field in store:
            if any(token in field.lower() for token in forbidden_tokens):
                errors.append(f"forbidden field {node_type}.{field}")
    if semantic_graph is not None:
        for node_type in semantic_graph["node_types"]:
            source = semantic_graph["node_stores"][node_type]
            current = nodes[node_type]
            for field, value in source.items():
                if isinstance(value, torch.Tensor) and not torch.equal(current[field], value):
                    errors.append(f"semantic node field changed: {node_type}.{field}")
        for key, source in semantic_graph["edge_stores"].items():
            if not torch.equal(edges[key]["edge_index"], source["edge_index"]):
                errors.append(f"semantic edge changed: {key}")
        for field, value in semantic_graph["targets"].items():
            if isinstance(value, torch.Tensor) and not torch.equal(graph["targets"][field], value):
                errors.append(f"target changed: {field}")
    if states is not None:
        membership = edges[edge_key(("event", "belongs_to", "possession"))]["edge_index"]
        expected = states["possession_index"].notna().sum()
        if membership.shape[1] != int(expected):
            errors.append("Event-Possession membership count mismatch")
    if possessions is not None:
        if counts["possession"] != len(possessions):
            errors.append("Possession node count mismatch")
        if edges[edge_key(("possession", "owned_by", "team"))]["edge_index"].shape[1] != len(possessions):
            errors.append("Possession-Team edge count mismatch")
        if bool(possessions.groupby("possession_uid")["period"].nunique().gt(1).any()):
            errors.append("Possession crosses period")
    if transitions is not None and edge_count != len(transitions):
        errors.append("Possession transition count mismatch")
    return errors


def _representative_positions(event: Mapping[str, Any], indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    end_mask = event["end_position_mask"][indices].bool()
    start_mask = event["start_position_mask"][indices].bool()
    mask = end_mask | start_mask
    positions = torch.where(
        end_mask.unsqueeze(-1), event["end_position"][indices], event["start_position"][indices]
    )
    positions = torch.where(mask.unsqueeze(-1), positions, torch.zeros_like(positions))
    return positions, mask


def _selected_map(indices: torch.Tensor, total: int) -> torch.Tensor:
    mapping = torch.full((total,), -1, dtype=torch.long)
    mapping[indices] = torch.arange(indices.numel(), dtype=torch.long)
    return mapping


def _referenced_destinations(
    graph: Mapping[str, Any], edge_type: tuple[str, str, str], event_indices: torch.Tensor
) -> torch.Tensor:
    edge_index = graph["edge_stores"][edge_key(edge_type)]["edge_index"]
    event_mask = torch.zeros(int(graph["node_stores"]["event"]["num_nodes"]), dtype=torch.bool)
    event_mask[event_indices] = True
    if edge_type[0] == "event":
        keep = event_mask[edge_index[0]]
        values = edge_index[1, keep]
    elif edge_type[2] == "event":
        keep = event_mask[edge_index[1]]
        values = edge_index[0, keep]
    else:
        raise ValueError("Referenced entity edge must touch Event")
    return torch.unique(values, sorted=True)


def _slice_node_store(store: Mapping[str, Any], selected: torch.Tensor) -> dict[str, Any]:
    total = int(store["num_nodes"])
    result: dict[str, Any] = {"num_nodes": int(selected.numel()), "source_index": selected.clone()}
    for field, value in store.items():
        if field == "num_nodes":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total:
            result[field] = value[selected].clone()
        else:
            result[field] = _clone(value)
    return result


def _dynamic_possession_features(
    graph: Mapping[str, Any],
    event_indices: torch.Tensor,
    possession_indices: torch.Tensor,
    anchor_event_index: int,
    snapshot_scope: SnapshotScope,
    include_dynamic: bool,
) -> dict[str, torch.Tensor]:
    event = graph["node_stores"]["event"]
    possession = graph["node_stores"]["possession"]
    selected_set = set(event_indices.tolist())
    anchor_active = int(event["active_possession_after_local_index"][anchor_event_index])
    fields: dict[str, list[Any]] = {
        "start_absolute_seconds": [],
        "start_position": [],
        "start_position_mask": [],
        "start_reason_index": [],
        "true_start_visible": [],
        "duration_so_far_seconds": [],
        "event_count_so_far": [],
        "current_position": [],
        "current_position_mask": [],
        "control_state_index": [],
        "candidate_status_index": [],
        "is_closed_as_of_anchor": [],
        "is_current_active": [],
        "snapshot_event_index": [],
        "dynamic_feature_mask": [],
    }
    prefix = torch.arange(anchor_event_index + 1, dtype=torch.long)
    for possession_index in possession_indices.tolist():
        all_members = prefix[event["possession_local_index"][: anchor_event_index + 1] == possession_index]
        selected_members = event_indices[event["possession_local_index"][event_indices] == possession_index]
        if selected_members.numel() == 0:
            raise ValueError("Retained Possession has no selected Event")
        scope_members = all_members if snapshot_scope == "anchor_history" else selected_members
        snapshot_index = int(scope_members[-1])
        start_event_index = int(possession["start_event_index"][possession_index])
        true_start_visible = (
            start_event_index <= anchor_event_index
            if snapshot_scope == "anchor_history"
            else start_event_index in selected_set
        )
        if snapshot_scope == "selected_events" and not true_start_visible:
            first_index = int(selected_members[0])
            position, position_mask = _representative_positions(
                event, torch.tensor([first_index], dtype=torch.long)
            )
            start_seconds = float(event["absolute_seconds"][first_index])
            start_position = position[0]
            start_mask = bool(position_mask[0])
            start_reason = 0
        else:
            start_seconds = float(possession["start_absolute_seconds"][possession_index])
            start_position = possession["start_position"][possession_index]
            start_mask = bool(possession["start_position_mask"][possession_index])
            start_reason = int(possession["start_reason_index"][possession_index])

        if snapshot_scope == "anchor_history":
            if possession_index == anchor_active and snapshot_index != anchor_event_index:
                raise ValueError("Open Possession snapshot is not the anchor Event row")
            duration = float(event["possession_duration_so_far_seconds"][snapshot_index])
            count = float(event["possession_event_count_so_far"][snapshot_index])
            current_position = event["possession_current_position"][snapshot_index]
            current_mask = bool(event["possession_current_position_mask"][snapshot_index])
            close_scope = prefix
        else:
            duration = max(
                float(event["absolute_seconds"][selected_members[-1]])
                - float(event["absolute_seconds"][selected_members[0]]),
                0.0,
            )
            count = float(selected_members.numel())
            current_position, current_masks = _representative_positions(
                event, selected_members[-1:].clone()
            )
            current_position = current_position[0]
            current_mask = bool(current_masks[0])
            close_scope = event_indices
        closed = bool(
            event["possession_is_closed_as_of_event"][snapshot_index]
            or (event["closed_possession_before_local_index"][close_scope] == possession_index).any()
            or (event["closed_possession_after_local_index"][close_scope] == possession_index).any()
        )
        dynamic_mask = bool(include_dynamic)
        fields["start_absolute_seconds"].append(start_seconds)
        fields["start_position"].append(start_position)
        fields["start_position_mask"].append(start_mask)
        fields["start_reason_index"].append(start_reason)
        fields["true_start_visible"].append(true_start_visible)
        fields["duration_so_far_seconds"].append(duration if dynamic_mask else 0.0)
        fields["event_count_so_far"].append(count if dynamic_mask else 0.0)
        fields["current_position"].append(
            current_position if dynamic_mask else torch.zeros(2, dtype=torch.float32)
        )
        fields["current_position_mask"].append(current_mask if dynamic_mask else False)
        fields["control_state_index"].append(
            int(event["control_state_after_index"][snapshot_index]) if dynamic_mask else 0
        )
        fields["candidate_status_index"].append(
            int(event["candidate_status_after_index"][snapshot_index]) if dynamic_mask else 0
        )
        fields["is_closed_as_of_anchor"].append(closed if dynamic_mask else False)
        fields["is_current_active"].append(
            possession_index == anchor_active if dynamic_mask else False
        )
        fields["snapshot_event_index"].append(snapshot_index)
        fields["dynamic_feature_mask"].append(dynamic_mask)

    return {
        "start_absolute_seconds": torch.tensor(fields["start_absolute_seconds"], dtype=torch.float32),
        "start_position": torch.stack(fields["start_position"]).float(),
        "start_position_mask": torch.tensor(fields["start_position_mask"], dtype=torch.bool),
        "start_reason_index": torch.tensor(fields["start_reason_index"], dtype=torch.long),
        "true_start_visible": torch.tensor(fields["true_start_visible"], dtype=torch.bool),
        "duration_so_far_seconds": torch.tensor(fields["duration_so_far_seconds"], dtype=torch.float32),
        "event_count_so_far": torch.tensor(fields["event_count_so_far"], dtype=torch.float32),
        "current_position": torch.stack(fields["current_position"]).float(),
        "current_position_mask": torch.tensor(fields["current_position_mask"], dtype=torch.bool),
        "control_state_index": torch.tensor(fields["control_state_index"], dtype=torch.long),
        "candidate_status_index": torch.tensor(fields["candidate_status_index"], dtype=torch.long),
        "is_closed_as_of_anchor": torch.tensor(fields["is_closed_as_of_anchor"], dtype=torch.bool),
        "is_current_active": torch.tensor(fields["is_current_active"], dtype=torch.bool),
        "snapshot_event_index": torch.tensor(fields["snapshot_event_index"], dtype=torch.long),
        "dynamic_feature_mask": torch.tensor(fields["dynamic_feature_mask"], dtype=torch.bool),
    }


def extract_causal_subgraph(
    graph: dict[str, Any],
    event_indices: Sequence[int] | torch.Tensor,
    anchor_event_index: int,
    *,
    snapshot_scope: SnapshotScope,
    include_dynamic_possession_features: bool,
) -> dict[str, Any]:
    """Materialize a compact causal view without inventing edges."""

    if graph.get("schema_version") != POSSESSION_GRAPH_VERSION:
        raise ValueError("Causal extraction requires a Semantic V3 Possession graph")
    if snapshot_scope not in {"anchor_history", "selected_events"}:
        raise ValueError(f"Unknown snapshot_scope {snapshot_scope!r}")
    selected_events = torch.as_tensor(event_indices, dtype=torch.long)
    if selected_events.ndim != 1 or selected_events.numel() == 0:
        raise ValueError("event_indices must be a non-empty one-dimensional sequence")
    if not torch.equal(selected_events, torch.unique(selected_events, sorted=True)):
        raise ValueError("event_indices must be sorted and unique")
    num_events = int(graph["node_stores"]["event"]["num_nodes"])
    if int(selected_events[0]) < 0 or int(selected_events[-1]) >= num_events:
        raise ValueError("event_indices are out of range")
    if int(selected_events[-1]) != int(anchor_event_index):
        raise ValueError("anchor_event_index must be the final selected Event")

    event = graph["node_stores"]["event"]
    possession_values = event["possession_local_index"][selected_events]
    selected_possessions = torch.unique(possession_values[possession_values >= 0], sorted=True)
    anchor_active = int(event["active_possession_after_local_index"][anchor_event_index])
    if anchor_active >= 0 and not bool((selected_possessions == anchor_active).any()):
        selected_possessions = torch.unique(
            torch.cat((selected_possessions, torch.tensor([anchor_active]))), sorted=True
        )

    selected: dict[str, torch.Tensor] = {
        "event": selected_events,
        "possession": selected_possessions,
        "player": _referenced_destinations(
            graph, ("event", "performed_by", "player"), selected_events
        ),
        "team": _referenced_destinations(
            graph, ("event", "performed_by_team", "team"), selected_events
        ),
        "event_type": _referenced_destinations(
            graph, ("event", "has_type", "event_type"), selected_events
        ),
        "tag": _referenced_destinations(graph, ("event", "has_tag", "tag"), selected_events),
        "zone": torch.unique(
            torch.cat(
                (
                    _referenced_destinations(graph, ("event", "starts_in", "zone"), selected_events),
                    _referenced_destinations(graph, ("event", "ends_in", "zone"), selected_events),
                )
            ),
            sorted=True,
        ),
    }
    if selected_possessions.numel():
        owner_teams = graph["node_stores"]["possession"]["owner_team_local_index"][selected_possessions]
        selected["team"] = torch.unique(torch.cat((selected["team"], owner_teams)), sorted=True)

    mappings = {
        node_type: _selected_map(indices, int(graph["node_stores"][node_type]["num_nodes"]))
        for node_type, indices in selected.items()
    }
    node_stores = {
        node_type: _slice_node_store(graph["node_stores"][node_type], indices)
        for node_type, indices in selected.items()
    }
    if selected_possessions.numel():
        node_stores["possession"].update(
            _dynamic_possession_features(
                graph,
                selected_events,
                selected_possessions,
                anchor_event_index,
                snapshot_scope,
                include_dynamic_possession_features,
            )
        )
        node_stores["possession"]["owner_team_local_index"] = mappings["team"][
            graph["node_stores"]["possession"]["owner_team_local_index"][selected_possessions]
        ]

    edge_stores: dict[str, dict[str, torch.Tensor]] = {}
    for source_type, relation, destination_type in POSSESSION_EDGE_TYPES:
        key = edge_key((source_type, relation, destination_type))
        store = graph["edge_stores"][key]
        edge_index = store["edge_index"]
        source_map = mappings[source_type]
        destination_map = mappings[destination_type]
        keep = (source_map[edge_index[0]] >= 0) & (destination_map[edge_index[1]] >= 0)
        if (source_type, relation, destination_type) == ("possession", "next", "possession"):
            keep &= store["transition_event_index"] <= anchor_event_index
        selected_edge = edge_index[:, keep]
        remapped = torch.stack(
            (source_map[selected_edge[0]], destination_map[selected_edge[1]])
        ) if selected_edge.shape[1] else torch.empty((2, 0), dtype=torch.long)
        edge_stores[key] = {"edge_index": remapped}
        for field, value in store.items():
            if field == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == edge_index.shape[1]:
                edge_stores[key][field] = value[keep].clone()
            else:
                edge_stores[key][field] = _clone(value)

    return {
        "schema_version": POSSESSION_GRAPH_VERSION,
        "graph_unit": "causal_subgraph",
        "match_id": int(graph["match_id"]),
        "competition_id": int(graph["competition_id"]),
        "competition_slug": graph["competition_slug"],
        "anchor_event_index": int(anchor_event_index),
        "anchor_event_local_index": int(selected_events.numel() - 1),
        "snapshot_scope": snapshot_scope,
        "include_dynamic_possession_features": bool(include_dynamic_possession_features),
        "node_types": list(POSSESSION_NODE_TYPES),
        "edge_types": [list(value) for value in POSSESSION_EDGE_TYPES],
        "node_stores": node_stores,
        "edge_stores": edge_stores,
        "causal_contract": {
            "targets_included": False,
            "future_events_included": False,
            "synthetic_next_edges": False,
            "selected_events_preserve_inferred_possession_topology": True,
        },
    }


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _require_pyarrow_24() -> None:
    import pyarrow

    major = int(pyarrow.__version__.split(".", maxsplit=1)[0])
    if major < 24 or major >= 26:
        raise RuntimeError(
            f"Semantic V3 build requires pyarrow>=24,<26, found {pyarrow.__version__}"
        )


def _read_model_safe_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _require_pyarrow_24()
    contract_path = root / "metadata/model_safe_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if "possession_audit.parquet" not in contract.get("forbidden_tables", []):
        raise ValueError("Possession V2 contract does not forbid the audit table")
    requested = {
        "event_possession_states.parquet",
        "possessions.parquet",
        "possession_transitions.parquet",
    }
    if not requested.issubset(set(contract.get("allowed_tables", []))):
        raise ValueError("Possession V2 contract does not allow required model-safe tables")
    states = pd.read_parquet(root / "event_possession_states.parquet")
    possessions = pd.read_parquet(root / "possessions.parquet")
    transitions = pd.read_parquet(root / "possession_transitions.parquet")
    return states, possessions, transitions, contract


def _group_matches(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(match_id): group.reset_index(drop=True)
        for match_id, group in frame.groupby("match_id", sort=False)
    }


def build_possession_graph_dataset(
    semantic_root: Path = SEMANTIC_GRAPH_ROOT,
    possession_root: Path = POSSESSION_V2_ROOT,
    output_root: Path = POSSESSION_GRAPH_ROOT,
    competition: str = "England",
    *,
    overwrite: bool = False,
    limit_matches: int | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the independent Semantic V3 dataset."""

    semantic_root = Path(semantic_root).resolve()
    possession_root = Path(possession_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}")
    rules_path = possession_root / "possession_rules_v2.csv"
    vocabularies = build_possession_vocabularies(rules_path)
    states, possessions, transitions, contract = _read_model_safe_tables(possession_root)
    state_groups = _group_matches(states)
    possession_groups = _group_matches(possessions)
    transition_groups = _group_matches(transitions)

    index_path = semantic_root / "metadata/match_index.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["competition_slug"] == competition]
    if limit_matches is not None:
        rows = rows[:limit_matches]
    if not rows:
        raise ValueError(f"No semantic graphs found for {competition}")

    build_root = output_root.parent / f".{output_root.name}.build-{uuid.uuid4().hex}"
    build_root.mkdir(parents=True)
    output_rows: list[dict[str, Any]] = []
    totals = Counter()
    errors: list[str] = []
    try:
        for row in rows:
            match_id = int(row["match_id"])
            if match_id not in state_groups or match_id not in possession_groups:
                raise ValueError(f"Missing Possession V2 rows for match {match_id}")
            source_path = semantic_root / row["graph_path"]
            source = torch.load(source_path, map_location="cpu", weights_only=True)
            match_transitions = transition_groups.get(
                match_id, transitions.iloc[0:0].copy()
            )
            graph = convert_match_to_possession_graph(
                source,
                state_groups[match_id],
                possession_groups[match_id],
                match_transitions,
                vocabularies,
            )
            output_path = build_root / row["graph_path"]
            _atomic_save(graph, output_path)
            num_possessions = int(graph["node_stores"]["possession"]["num_nodes"])
            membership_count = int(
                graph["edge_stores"][edge_key(("event", "belongs_to", "possession"))][
                    "edge_index"
                ].shape[1]
            )
            transition_count = int(
                graph["edge_stores"][edge_key(("possession", "next", "possession"))][
                    "edge_index"
                ].shape[1]
            )
            transition_store = graph["edge_stores"][
                edge_key(("possession", "next", "possession"))
            ]
            cross_period_count = int(transition_store["cross_period"].sum())
            period_break_reason_index = vocabularies["indices"]["transition_reason"][
                "period_break"
            ]
            period_break_reason_count = int(
                (transition_store["transition_reason_index"] == period_break_reason_index).sum()
            )
            output_rows.append(
                {
                    **row,
                    "schema_version": POSSESSION_GRAPH_VERSION,
                    "num_possessions": num_possessions,
                    "num_possession_memberships": membership_count,
                    "num_possession_transitions": transition_count,
                    "num_cross_period_transitions": cross_period_count,
                    "num_period_break_reason_transitions": period_break_reason_count,
                }
            )
            totals.update(
                matches=1,
                events=int(graph["node_stores"]["event"]["num_nodes"]),
                possessions=num_possessions,
                possession_memberships=membership_count,
                possession_team_edges=num_possessions,
                possession_transitions=transition_count,
                cross_period_transitions=cross_period_count,
                period_break_reason_transitions=period_break_reason_count,
            )

        if limit_matches is None and competition == "England":
            expected = {
                "matches": 380,
                "events": 643_150,
                "possessions": 117_607,
                "possession_memberships": 641_040,
                "possession_team_edges": 117_607,
                "possession_transitions": 117_227,
                "cross_period_transitions": 380,
            }
            for field, expected_value in expected.items():
                if totals[field] != expected_value:
                    errors.append(
                        f"{field}: expected {expected_value}, observed {totals[field]}"
                    )
        if errors:
            raise ValueError("Full dataset acceptance failed: " + "; ".join(errors))

        metadata = build_root / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        index_output = metadata / "match_index.csv"
        with index_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        schema = {
            "schema_version": POSSESSION_GRAPH_VERSION,
            "definition": "Semantic V2 + causal Possession topology + optional causal snapshots",
            "node_types": list(POSSESSION_NODE_TYPES),
            "edge_types": [list(value) for value in POSSESSION_EDGE_TYPES],
            "reference_only_fields": list(REFERENCE_ONLY_FIELDS),
            "categorical_feature_fields": list(CATEGORICAL_FEATURE_FIELDS),
            "continuous_feature_fields": list(CONTINUOUS_FEATURE_FIELDS),
            "binary_mask_fields": list(BINARY_MASK_FIELDS),
            "visibility_sensitive_fields": list(VISIBILITY_SENSITIVE_FIELDS),
            "snapshot_scopes": ["anchor_history", "selected_events"],
            "selected_events_contract": (
                "Unselected dynamic summaries are hidden; causally inferred Possession "
                "identity, owner, and topology are retained."
            ),
            "first_model_default": {
                "snapshot_scope": "selected_events",
                "include_dynamic_possession_features": False,
            },
            "transition_reason_storage": "edge_attribute",
            "cross_period_storage": (
                "boolean edge attribute independent of transition_reason precedence"
            ),
        }
        _write_json(schema, metadata / "graph_schema.json")
        _write_json(vocabularies, metadata / "vocabularies.json")
        validation_report = {
            "valid": True,
            "errors": [],
            "totals": dict(totals),
            "forbidden_audit_read": True,
            "source_event_and_target_identity": True,
        }
        _write_json(validation_report, metadata / "validation_report.json")
        source_paths = {
            "semantic_index": index_path,
            "semantic_manifest": semantic_root / "metadata/manifest.json",
            "possession_states": possession_root / "event_possession_states.parquet",
            "possessions": possession_root / "possessions.parquet",
            "possession_transitions": possession_root / "possession_transitions.parquet",
            "model_safe_contract": possession_root / "metadata/model_safe_contract.json",
            "possession_rules": rules_path,
        }
        manifest = {
            "schema_version": POSSESSION_GRAPH_VERSION,
            "status": "complete",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "competition": competition,
            "limit_matches": limit_matches,
            "semantic_root": str(semantic_root),
            "possession_root": str(possession_root),
            "output_root": str(output_root),
            "totals": dict(totals),
            "source_hashes": {
                name: file_sha256(path) for name, path in source_paths.items()
            },
            "model_safe_contract": contract,
        }
        _write_json(manifest, metadata / "manifest.json")
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(build_root, output_root)
        return manifest
    except Exception:
        failure = {
            "schema_version": POSSESSION_GRAPH_VERSION,
            "status": "failed",
            "errors": errors,
        }
        _write_json(failure, build_root / "metadata/manifest.json")
        raise
