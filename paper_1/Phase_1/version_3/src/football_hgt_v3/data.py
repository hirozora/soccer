"""Single-graph datasets and batching for Version 3."""

from __future__ import annotations

from math import log1p
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from football_hgt.dataset import FixedWindowDataset, MatchGraphRecord
from football_hgt_v1.data import window_to_heterodata

from .experts import (
    EXPERT_NAMES,
    OVERLAP_KEYS,
    TRANSITION_EVENT_INDICES,
    ExpertSelection,
    SelectionConfig,
    StructuralSelectionGenerator,
    oriented_event_points,
    selection_overlap,
)


CANDIDATE_FEATURE_NAMES = (
    "count_k10",
    "count_k20",
    "count_k80",
    "event_recency",
    "time_recency",
    "is_anchor_actor",
    "seen",
    "last_team_same",
    "last_team_other",
    "last_team_unseen",
)


class DataAwareWindowDataset(Dataset[dict[str, Any]]):
    """Attach causal expert selections to one ordinary V1 K=80 window."""

    def __init__(
        self,
        records: Sequence[MatchGraphRecord],
        selection_config: SelectionConfig | None = None,
        cache_size: int | None = None,
        cache_samples: bool = True,
    ) -> None:
        self.selection_config = selection_config or SelectionConfig()
        self.base = FixedWindowDataset(
            records,
            window_size=self.selection_config.full_history,
            cache_size=cache_size or min(8, len(records)),
            validate_graph_on_load=False,
            validate_samples=False,
        )
        self.generator = StructuralSelectionGenerator(self.selection_config)
        self.cache_samples = cache_samples
        self._sample_cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index in self._sample_cache:
            return self._sample_cache[index]
        window = self.base[index]
        selections = self.generator.generate(window)
        sample = {
            "window": window,
            "selection_config": self.selection_config,
            "selections": selections,
            "selection_features": {
                name: selection_event_features(window, selection)
                for name, selection in selections.items()
            },
            "candidate_features": candidate_player_features(window),
            "overlap": selection_overlap(selections),
        }
        if self.cache_samples:
            self._sample_cache[index] = sample
        return sample


def candidate_player_features(window: dict[str, Any]) -> torch.Tensor:
    """Build prefix-only features for every match-local player candidate."""

    event = window["node_stores"]["event"]
    num_players = int(window["node_stores"]["player"]["num_nodes"])
    length = int(event["num_nodes"])
    anchor = length - 1
    anchor_time = float(event["absolute_seconds"][anchor])
    anchor_team = int(event["team_local_index"][anchor])
    anchor_player = int(event["player_local_index"][anchor])
    result = torch.zeros((num_players, len(CANDIDATE_FEATURE_NAMES)), dtype=torch.float32)
    players = event["player_local_index"].long()
    teams = event["team_local_index"].long()
    absolute = event["absolute_seconds"].float()

    for player in range(num_players):
        occurrences = torch.nonzero(players == player, as_tuple=False).flatten()
        seen = bool(occurrences.numel())
        result[player, 0] = float((players[max(0, length - 10) :] == player).sum()) / 10.0
        result[player, 1] = float((players[max(0, length - 20) :] == player).sum()) / 20.0
        result[player, 2] = float((players == player).sum()) / 80.0
        result[player, 5] = float(player == anchor_player)
        result[player, 6] = float(seen)
        if seen:
            latest = int(occurrences[-1])
            result[player, 3] = min((anchor - latest) / 79.0, 1.0)
            result[player, 4] = min(
                log1p(max(anchor_time - float(absolute[latest]), 0.0)) / log1p(7200.0),
                1.0,
            )
            if int(teams[latest]) == anchor_team:
                result[player, 7] = 1.0
            else:
                result[player, 8] = 1.0
        else:
            result[player, 3:5] = 1.0
            result[player, 9] = 1.0
    return result.clamp(0.0, 1.0)


def selection_event_features(
    window: dict[str, Any], selection: ExpertSelection
) -> dict[str, torch.Tensor]:
    """Precompute per-selected-event inputs once for later padded collation."""

    event = window["node_stores"]["event"]
    local = selection.event_indices
    count = int(local.numel())
    anchor = int(event["num_nodes"]) - 1
    anchor_time = event["absolute_seconds"][anchor].float()
    times = event["absolute_seconds"][local].float()
    relative_index = (anchor - local).float() / 79.0
    relative_time = (
        torch.log1p((anchor_time - times).clamp_min(0.0)) / log1p(7200.0)
    ).clamp(0.0, 1.0)

    anchor_team = int(event["team_local_index"][anchor])
    selected_teams = event["team_local_index"][local].long()
    team_relation = (selected_teams != anchor_team).long()
    current_player = int(event["player_local_index"][anchor])
    selected_players = event["player_local_index"][local].long()
    actor_relation = torch.where(
        selected_players == current_player,
        torch.zeros_like(selected_players),
        torch.where(
            selected_teams == anchor_team,
            torch.ones_like(selected_players),
            torch.full_like(selected_players, 2),
        ),
    )

    all_boundary = torch.zeros(anchor + 1, dtype=torch.bool)
    if anchor > 0:
        all_boundary[1:] = (
            event["team_local_index"][1 : anchor + 1]
            != event["team_local_index"][:anchor]
        )
    boundary = all_boundary[local]

    points, valid = oriented_event_points(event, anchor_team)
    differences = points[local] - points[anchor]
    distances = torch.linalg.vector_norm(differences, dim=-1, keepdim=True)
    spatial = torch.cat((differences, distances, valid[local, None].float()), dim=-1)

    transition_types = torch.tensor(
        [
            float(int(event["event_type_index"][index]) in TRANSITION_EVENT_INDICES)
            for index in local.tolist()
        ],
        dtype=torch.float32,
    )
    index_gaps = torch.zeros(count, dtype=torch.float32)
    time_gaps = torch.zeros(count, dtype=torch.float32)
    if count > 1:
        index_gaps[1:] = (local[1:] - local[:-1] - 1).clamp_min(0).float() / 79.0
        selected_times = event["absolute_seconds"][local].float()
        time_gaps[1:] = (
            torch.log1p((selected_times[1:] - selected_times[:-1]).clamp_min(0.0))
            / log1p(7200.0)
        )
    transition = torch.stack(
        (boundary.float(), transition_types, index_gaps, time_gaps), dim=-1
    )
    return {
        "relative_index": relative_index,
        "relative_time": relative_time,
        "team_relation": team_relation,
        "actor_relation": actor_relation,
        "boundary": boundary,
        "spatial_features": spatial,
        "transition_features": transition,
    }


def _pad_selection(
    samples: Sequence[dict[str, Any]],
    name: str,
    event_offsets: torch.Tensor,
    budget: int,
) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    indices = torch.empty((batch_size, budget), dtype=torch.long)
    mask = torch.zeros((batch_size, budget), dtype=torch.bool)
    relative_index = torch.zeros((batch_size, budget), dtype=torch.float32)
    relative_time = torch.zeros((batch_size, budget), dtype=torch.float32)
    team_relation = torch.zeros((batch_size, budget), dtype=torch.long)
    actor_relation = torch.zeros((batch_size, budget), dtype=torch.long)
    boundary = torch.zeros((batch_size, budget), dtype=torch.bool)
    spatial = torch.zeros((batch_size, budget, 4), dtype=torch.float32)
    transition = torch.zeros((batch_size, budget, 4), dtype=torch.float32)
    statistics = torch.stack(
        [samples[index]["selections"][name].structural_features for index in range(batch_size)]
    )

    for batch_index, sample in enumerate(samples):
        window = sample["window"]
        event = window["node_stores"]["event"]
        selection: ExpertSelection = sample["selections"][name]
        features = sample["selection_features"][name]
        local = selection.event_indices
        count = int(local.numel())
        anchor = int(event["num_nodes"]) - 1
        global_anchor = int(event_offsets[batch_index]) + anchor
        indices[batch_index].fill_(global_anchor)
        indices[batch_index, :count] = local + event_offsets[batch_index]
        mask[batch_index, :count] = selection.selection_mask

        relative_index[batch_index, :count] = features["relative_index"]
        relative_time[batch_index, :count] = features["relative_time"]
        team_relation[batch_index, :count] = features["team_relation"]
        actor_relation[batch_index, :count] = features["actor_relation"]
        boundary[batch_index, :count] = features["boundary"]
        spatial[batch_index, :count] = features["spatial_features"]
        transition[batch_index, :count] = features["transition_features"]

    return {
        "event_indices": indices,
        "selection_mask": mask,
        "structural_features": statistics,
        "relative_index": relative_index,
        "relative_time": relative_time,
        "team_relation": team_relation,
        "actor_relation": actor_relation,
        "boundary": boundary,
        "spatial_features": spatial,
        "transition_features": transition,
    }


def collate_data_aware_windows(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Batch one full graph per sample plus padded post-HGT selections."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    graph = Batch.from_data_list(
        [window_to_heterodata(sample["window"]) for sample in samples]
    )
    event_counts = torch.tensor(
        [sample["window"]["node_stores"]["event"]["num_nodes"] for sample in samples],
        dtype=torch.long,
    )
    event_offsets = torch.cat((torch.zeros(1, dtype=torch.long), event_counts.cumsum(0)))[:-1]
    config: SelectionConfig = samples[0]["selection_config"]
    budgets = {name: config.budget(name) for name in EXPERT_NAMES}
    selections = {
        name: _pad_selection(samples, name, event_offsets, budgets[name])
        for name in EXPERT_NAMES
    }
    overlap = torch.tensor(
        [[sample["overlap"][key] for key in OVERLAP_KEYS] for sample in samples],
        dtype=torch.float32,
    )
    sizes = torch.tensor(
        [
            [int(sample["selections"][name].event_indices.numel()) for name in EXPERT_NAMES]
            for sample in samples
        ],
        dtype=torch.long,
    )
    return {
        "graph": graph,
        "selections": selections,
        "candidate_features": torch.cat(
            [sample["candidate_features"] for sample in samples], dim=0
        ),
        "overlap": overlap,
        "selection_sizes": sizes,
    }


def validate_data_aware_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    window = sample.get("window", {})
    length = int(window.get("node_stores", {}).get("event", {}).get("num_nodes", 0))
    if length < 1 or length > 80:
        errors.append("full window length must be in [1, 80]")
    selections = sample.get("selections", {})
    if tuple(selections) != EXPERT_NAMES:
        errors.append("expert names or order do not match the Version 3 contract")
        return errors
    for name, selection in selections.items():
        indices = selection.event_indices
        if indices.dtype != torch.long or selection.selection_mask.dtype != torch.bool:
            errors.append(f"{name} selection dtype is invalid")
        if not indices.numel() or int(indices[-1]) != length - 1:
            errors.append(f"{name} does not end at the anchor")
        if int(indices.min()) < 0 or int(indices.max()) >= length:
            errors.append(f"{name} contains an out-of-window event")
        stats = selection.structural_features
        if stats.dtype != torch.float32 or stats.shape != (12,):
            errors.append(f"{name} statistics shape or dtype is invalid")
        elif not bool(torch.isfinite(stats).all()) or bool((stats < 0).any()) or bool((stats > 1).any()):
            errors.append(f"{name} statistics are not finite normalized values")
    candidates = sample.get("candidate_features")
    expected_players = int(window["node_stores"]["player"]["num_nodes"])
    if candidates is None or candidates.shape != (expected_players, len(CANDIDATE_FEATURE_NAMES)):
        errors.append("candidate feature shape is invalid")
    return errors
