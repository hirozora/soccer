"""Deterministic causal graph views for the Version 3 residual model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


FULL_HISTORY_NAME = "full_history"
EXPERT_NAMES = (
    "short_temporal",
    "possession",
    "spatial",
    "actor_relation",
    "transition",
)
BRANCH_NAMES = (FULL_HISTORY_NAME, *EXPERT_NAMES)
EDGE_TYPES = (
    ("event", "next", "event"),
    ("player", "performs", "event"),
    ("team", "performs", "event"),
    ("tag", "describes", "event"),
)
TRANSITION_EVENT_TYPE_IDS = frozenset({1, 2, 3, 6, 9, 10})


@dataclass(frozen=True)
class GraphViewConfig:
    full_history: int = 80
    short_temporal: int = 10
    possession: int = 20
    spatial: int = 16
    actor_relation: int = 18
    transition: int = 12
    spatial_horizon: int = 30
    actor_horizon: int = 80
    actor_players: int = 10


def _chronological(values: list[int], current: int, budget: int) -> list[int]:
    selected = sorted(set(values) | {current})
    if len(selected) > budget:
        selected = selected[-budget:]
    if selected[-1] != current:
        raise ValueError("Expert selection must end at the anchor event")
    return selected


class StructuralExpertGenerator:
    """Generate five bounded event-index views from visible history only."""

    def __init__(self, config: GraphViewConfig | None = None) -> None:
        self.config = config or GraphViewConfig()

    def short_temporal(self, graph: dict[str, Any], current: int) -> list[int]:
        start = max(0, current - self.config.short_temporal + 1)
        return list(range(start, current + 1))

    def possession(self, graph: dict[str, Any], current: int) -> list[int]:
        teams = graph["node_stores"]["event"]["team_local_index"]
        anchor_team = int(teams[current])
        selected = []
        opponent_streak = 0
        for index in range(current, -1, -1):
            if int(teams[index]) == anchor_team:
                opponent_streak = 0
            else:
                opponent_streak += 1
                if opponent_streak >= 2:
                    break
            selected.append(index)
            if len(selected) >= self.config.possession:
                break
        return _chronological(selected, current, self.config.possession)

    def spatial(self, graph: dict[str, Any], current: int) -> list[int]:
        event = graph["node_stores"]["event"]
        start = max(0, current - self.config.spatial_horizon + 1)
        anchor = event["start_position"][current]
        candidates = []
        for index in range(start, current + 1):
            distance = float(
                torch.sum((event["start_position"][index] - anchor).square())
            )
            candidates.append((distance, -index, index))
        closest = [value[2] for value in sorted(candidates)[: self.config.spatial]]
        return _chronological(closest, current, self.config.spatial)

    def actor_relation(self, graph: dict[str, Any], current: int) -> list[int]:
        event = graph["node_stores"]["event"]
        possession = self.possession(graph, current)
        current_player = int(event["player_local_index"][current])

        player_recency: dict[int, int] = {}
        for index in possession:
            player_recency[int(event["player_local_index"][index])] = index
        ranked_players = sorted(
            player_recency,
            key=lambda player: (player != current_player, -player_recency[player]),
        )[: self.config.actor_players]
        active_players = set(ranked_players) | {current_player}

        start = max(0, current - self.config.actor_horizon + 1)
        selected = [
            index
            for index in range(start, current + 1)
            if int(event["player_local_index"][index]) in active_players
        ]
        return _chronological(selected, current, self.config.actor_relation)

    def transition(self, graph: dict[str, Any], current: int) -> list[int]:
        event = graph["node_stores"]["event"]
        raw_type_by_vocab = {
            int(vocab): int(raw)
            for raw, vocab in zip(
                graph["node_stores"]["event_type"]["raw_id"],
                graph["node_stores"]["event_type"]["vocab_index"],
            )
        }
        transition_index = current
        for index in range(current, 0, -1):
            event_type = raw_type_by_vocab[int(event["event_type_index"][index])]
            team_changed = bool(
                event["team_local_index"][index]
                != event["team_local_index"][index - 1]
            )
            if team_changed or event_type in TRANSITION_EVENT_TYPE_IDS:
                transition_index = index
                break

        before = list(range(max(0, transition_index - 3), transition_index + 1))
        after = list(range(transition_index + 1, current + 1))
        if len(before) + len(after) > self.config.transition:
            after_budget = self.config.transition - len(before)
            after = after[-after_budget:] if after_budget > 0 else []
        return _chronological(before + after, current, self.config.transition)

    def generate(self, graph: dict[str, Any], current: int) -> dict[str, list[int]]:
        num_events = int(graph["node_stores"]["event"]["num_nodes"])
        if current < 0 or current >= num_events - 1:
            raise IndexError(f"Anchor event must be in [0, {num_events - 2}]")
        views = {
            "short_temporal": self.short_temporal(graph, current),
            "possession": self.possession(graph, current),
            "spatial": self.spatial(graph, current),
            "actor_relation": self.actor_relation(graph, current),
            "transition": self.transition(graph, current),
        }
        for name, indices in views.items():
            budget = int(getattr(self.config, name))
            if not indices or indices[-1] != current or len(indices) > budget:
                raise ValueError(f"Invalid {name} expert selection")
            if indices != sorted(set(indices)):
                raise ValueError(f"{name} expert indices must be sorted and unique")
        return views


def _selected_event_store(
    store: dict[str, Any], indices: torch.Tensor
) -> dict[str, Any]:
    num_events = int(store["num_nodes"])
    result: dict[str, Any] = {"num_nodes": int(indices.numel())}
    for name, value in store.items():
        if name == "num_nodes":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] != num_events:
                raise ValueError(f"event.{name} is not event-aligned")
            result[name] = value.index_select(0, indices)
        else:
            result[name] = value
    return result


def _selected_tag_edges(
    graph: dict[str, Any], selected: list[int]
) -> torch.Tensor:
    edge_index = graph["edge_stores"]["tag__describes__event"]["edge_index"]
    destinations = edge_index[1]
    sources = []
    local_destinations = []
    for local_index, original_index in enumerate(selected):
        bounds = torch.tensor(
            [original_index, original_index + 1], dtype=destinations.dtype
        )
        left, right = torch.searchsorted(destinations, bounds).tolist()
        count = right - left
        if count:
            sources.append(edge_index[0, left:right])
            local_destinations.append(
                torch.full((count,), local_index, dtype=torch.long)
            )
    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((torch.cat(sources), torch.cat(local_destinations)))


def build_structural_expert_view(
    graph: dict[str, Any],
    selected: list[int],
    current: int,
    expert_name: str,
    targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Materialize one structural view with the full-history branch target."""

    if selected[-1] != current:
        raise ValueError("Selected expert events must end at the anchor")
    index = torch.tensor(selected, dtype=torch.long)
    event_store = _selected_event_store(graph["node_stores"]["event"], index)
    local_events = torch.arange(len(selected), dtype=torch.long)
    next_edges = (
        torch.stack((local_events[:-1], local_events[1:]))
        if len(selected) > 1
        else torch.empty((2, 0), dtype=torch.long)
    )
    return {
        "schema_version": graph["schema_version"],
        "sample_type": "structural_residual_expert",
        "branch_name": expert_name,
        "match_id": graph["match_id"],
        "competition_id": graph["competition_id"],
        "competition_slug": graph["competition_slug"],
        "node_types": ["event", "player", "team", "tag"],
        "edge_types": [list(value) for value in EDGE_TYPES],
        "node_stores": {
            "event": event_store,
            "player": dict(graph["node_stores"]["player"]),
            "team": dict(graph["node_stores"]["team"]),
            "tag": dict(graph["node_stores"]["tag"]),
        },
        "edge_stores": {
            "event__next__event": {"edge_index": next_edges},
            "player__performs__event": {
                "edge_index": torch.stack(
                    (event_store["player_local_index"], local_events)
                )
            },
            "team__performs__event": {
                "edge_index": torch.stack(
                    (event_store["team_local_index"], local_events)
                )
            },
            "tag__describes__event": {
                "edge_index": _selected_tag_edges(graph, selected)
            },
        },
        "targets": targets,
        "selected_event_indices": selected,
        "window": {
            "current_event_index": current,
            "target_event_index": current + 1,
            "num_events": len(selected),
            "query_event_index": len(selected) - 1,
        },
    }


def pairwise_jaccard(views: dict[str, list[int]]) -> dict[str, float]:
    """Return pairwise intersection-over-union for the supplied graph views."""

    result = {}
    names = tuple(views)
    for left_index, left_name in enumerate(names):
        left = set(views[left_name])
        for right_name in names[left_index + 1 :]:
            right = set(views[right_name])
            result[f"{left_name}__{right_name}"] = len(left & right) / len(left | right)
    return result
