"""Causal event selections for post-HGT structural readout experts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log1p, sqrt
from typing import Any

import torch


EXPERT_NAMES = (
    "short_temporal",
    "team_sequence",
    "spatial",
    "actor_relation",
    "transition",
)

STAT_NAMES = (
    "event_ratio",
    "time_span",
    "index_span",
    "mean_time_age",
    "same_team_ratio",
    "team_change_rate",
    "player_coverage",
    "mean_spatial_distance",
    "max_spatial_distance",
    "valid_position_ratio",
    "selection_gap_ratio",
    "window_length_ratio",
)

OVERLAP_KEYS = tuple(f"{left}__{right}" for left, right in combinations(EXPERT_NAMES, 2))

# Wyscout event vocabulary indices for Duel, Foul, Free Kick, Offside,
# Save attempt, and Shot. These are heuristic transition boundaries.
TRANSITION_EVENT_INDICES = frozenset({0, 1, 2, 5, 8, 9})


@dataclass(frozen=True)
class ExpertSelection:
    event_indices: torch.Tensor
    selection_mask: torch.Tensor
    structural_features: torch.Tensor


@dataclass(frozen=True)
class SelectionConfig:
    full_history: int = 80
    short_temporal: int = 10
    team_sequence: int = 20
    spatial: int = 16
    actor_relation: int = 18
    transition: int = 12
    spatial_horizon: int = 30
    actor_players: int = 10

    def budget(self, name: str) -> int:
        return int(getattr(self, name))


def oriented_event_points(event: Any, anchor_team: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid end-or-start points in the anchor team's orientation."""

    start = event["start_position"].float()
    start_mask = event["start_position_mask"].bool()
    end = event["end_position"].float()
    end_mask = event["end_position_mask"].bool()
    points = torch.where(end_mask[:, None], end, start)
    valid = end_mask | start_mask
    opponent = event["team_local_index"].long() != int(anchor_team)
    points = torch.where(opponent[:, None], 1.0 - points, points)
    return points, valid


def _chronological(indices: list[int], anchor: int, budget: int) -> list[int]:
    selected = sorted(set(indices) | {anchor})
    if len(selected) > budget:
        selected = selected[-budget:]
    if not selected or selected[-1] != anchor:
        raise ValueError("Every expert selection must end at the anchor event")
    return selected


class StructuralSelectionGenerator:
    """Generate five deterministic selections within one visible K=80 window."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig()

    def short_temporal(self, sample: dict[str, Any]) -> list[int]:
        length = int(sample["node_stores"]["event"]["num_nodes"])
        start = max(0, length - self.config.short_temporal)
        return list(range(start, length))

    def team_sequence(self, sample: dict[str, Any]) -> list[int]:
        event = sample["node_stores"]["event"]
        teams = event["team_local_index"]
        anchor = int(event["num_nodes"]) - 1
        anchor_team = int(teams[anchor])
        selected: list[int] = []
        opponent_streak = 0
        for index in range(anchor, -1, -1):
            if int(teams[index]) == anchor_team:
                opponent_streak = 0
            else:
                opponent_streak += 1
                if opponent_streak >= 2:
                    break
            selected.append(index)
            if len(selected) >= self.config.team_sequence:
                break
        return _chronological(selected, anchor, self.config.team_sequence)

    def spatial(self, sample: dict[str, Any]) -> list[int]:
        event = sample["node_stores"]["event"]
        anchor = int(event["num_nodes"]) - 1
        anchor_team = int(event["team_local_index"][anchor])
        points, valid = oriented_event_points(event, anchor_team)
        anchor_point = points[anchor]
        start = max(0, anchor - self.config.spatial_horizon + 1)
        candidates = []
        for index in range(start, anchor + 1):
            distance = (
                float(torch.linalg.vector_norm(points[index] - anchor_point))
                if bool(valid[index])
                else float("inf")
            )
            candidates.append((distance, -index, index))
        selected = [row[2] for row in sorted(candidates)[: self.config.spatial]]
        return _chronological(selected, anchor, self.config.spatial)

    def actor_relation(self, sample: dict[str, Any]) -> list[int]:
        event = sample["node_stores"]["event"]
        anchor = int(event["num_nodes"]) - 1
        players = event["player_local_index"]
        current_player = int(players[anchor])
        context = self.team_sequence(sample)
        recency: dict[int, int] = {}
        for index in context:
            recency[int(players[index])] = index
        active = sorted(
            recency,
            key=lambda player: (player != current_player, -recency[player]),
        )[: self.config.actor_players]
        active_players = set(active) | {current_player}
        selected = [
            index
            for index in range(anchor + 1)
            if int(players[index]) in active_players
        ]
        return _chronological(selected, anchor, self.config.actor_relation)

    def transition(self, sample: dict[str, Any]) -> list[int]:
        event = sample["node_stores"]["event"]
        anchor = int(event["num_nodes"]) - 1
        transition_index = anchor
        for index in range(anchor, 0, -1):
            team_changed = bool(
                event["team_local_index"][index]
                != event["team_local_index"][index - 1]
            )
            event_type = int(event["event_type_index"][index])
            if team_changed or event_type in TRANSITION_EVENT_INDICES:
                transition_index = index
                break
        before = list(range(max(0, transition_index - 3), transition_index + 1))
        after = list(range(transition_index + 1, anchor + 1))
        return _chronological(before + after, anchor, self.config.transition)

    def generate(self, sample: dict[str, Any]) -> dict[str, ExpertSelection]:
        length = int(sample["node_stores"]["event"]["num_nodes"])
        if length < 1 or length > self.config.full_history:
            raise ValueError("Window length is outside the configured full history")
        raw = {
            "short_temporal": self.short_temporal(sample),
            "team_sequence": self.team_sequence(sample),
            "spatial": self.spatial(sample),
            "actor_relation": self.actor_relation(sample),
            "transition": self.transition(sample),
        }
        result: dict[str, ExpertSelection] = {}
        for name, indices in raw.items():
            if indices != sorted(set(indices)):
                raise ValueError(f"{name} indices must be sorted and unique")
            if indices[-1] != length - 1 or len(indices) > self.config.budget(name):
                raise ValueError(f"Invalid {name} selection")
            tensor = torch.tensor(indices, dtype=torch.long)
            result[name] = ExpertSelection(
                event_indices=tensor,
                selection_mask=torch.ones(len(indices), dtype=torch.bool),
                structural_features=self._statistics(sample, name, tensor),
            )
        return result

    def _statistics(
        self,
        sample: dict[str, Any],
        name: str,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        event = sample["node_stores"]["event"]
        length = int(event["num_nodes"])
        anchor = length - 1
        anchor_team = int(event["team_local_index"][anchor])
        budget = self.config.budget(name)
        count = int(indices.numel())
        absolute = event["absolute_seconds"][indices].float()
        ages = (event["absolute_seconds"][anchor] - absolute).clamp_min(0.0)
        teams = event["team_local_index"][indices].long()
        players = event["player_local_index"][indices].long()
        points, valid = oriented_event_points(event, anchor_team)
        selected_points = points[indices]
        selected_valid = valid[indices]
        distances = torch.linalg.vector_norm(selected_points - points[anchor], dim=-1)
        valid_distances = distances[selected_valid]
        span = int(indices[-1] - indices[0]) if count > 1 else 0
        team_changes = int((teams[1:] != teams[:-1]).sum()) if count > 1 else 0
        player_nodes = max(int(sample["node_stores"]["player"]["num_nodes"]), 1)
        mean_distance = float(valid_distances.mean()) if valid_distances.numel() else 0.0
        max_distance = float(valid_distances.max()) if valid_distances.numel() else 0.0
        denominator = max(span + 1, 1)
        values = torch.tensor(
            [
                count / max(budget, 1),
                min(log1p(float(absolute[-1] - absolute[0])) / log1p(7200.0), 1.0),
                span / max(self.config.full_history - 1, 1),
                min(float(torch.log1p(ages).mean()) / log1p(7200.0), 1.0),
                float((teams == anchor_team).float().mean()),
                team_changes / max(count - 1, 1),
                int(torch.unique(players).numel()) / player_nodes,
                min(mean_distance / sqrt(2.0), 1.0),
                min(max_distance / sqrt(2.0), 1.0),
                float(selected_valid.float().mean()),
                max(denominator - count, 0) / denominator,
                length / max(self.config.full_history, 1),
            ],
            dtype=torch.float32,
        )
        if values.shape != (len(STAT_NAMES),) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"Invalid structural statistics for {name}")
        return values.clamp(0.0, 1.0)


def selection_overlap(
    selections: dict[str, ExpertSelection],
) -> dict[str, float]:
    sets = {name: set(value.event_indices.tolist()) for name, value in selections.items()}
    result: dict[str, float] = {}
    for left, right in combinations(EXPERT_NAMES, 2):
        union = sets[left] | sets[right]
        result[f"{left}__{right}"] = len(sets[left] & sets[right]) / max(len(union), 1)
    return result
