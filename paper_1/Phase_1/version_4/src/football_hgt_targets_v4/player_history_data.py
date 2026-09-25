"""Causal current-match histories for every match-local Player candidate."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import torch

from football_benchmark.data import CanonicalSample
from football_benchmark.protocol import ProtocolArtifacts

from .five_task_data import collate_five_task_hgt
from .oracle_dependency import load_roster_team_map
from .player_history_study import HISTORY_LENGTH, HISTORY_STATS_DIM, SELECTOR_SEED


@dataclass(frozen=True)
class PlayerHistoryTensors:
    event_type: torch.Tensor
    numeric: torch.Tensor
    sequence_mask: torch.Tensor
    lengths: torch.Tensor
    statistics: torch.Tensor
    shuffle_eligible: torch.Tensor
    history_count: torch.Tensor
    recency_seconds: torch.Tensor
    current_possession_participated: torch.Tensor


_HISTORY_CACHE: dict[tuple[int, int], PlayerHistoryTensors] = {}


def _representative_positions(event: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    start = event["start_position"].float()
    start_mask = event["start_position_mask"].bool()
    end = event["end_position"].float()
    end_mask = event["end_position_mask"].bool()
    return torch.where(end_mask.unsqueeze(-1), end, start), end_mask | start_mask


def _normalized_log(value: float, denominator: float) -> float:
    return math.log1p(max(value, 0.0)) / max(math.log1p(max(denominator, 1.0)), 1e-12)


def _history_for_sample(sample: CanonicalSample) -> PlayerHistoryTensors:
    graph = sample.graph
    event = graph["node_stores"]["event"]
    players = graph["node_stores"]["player"]
    count = int(players["num_nodes"])
    anchor = int(sample.current_event_index)
    anchor_time = float(event["absolute_seconds"][anchor])
    anchor_team = int(event["team_local_index"][anchor])
    player_index = event["player_local_index"][: anchor + 1].long()
    event_types = event["event_type_index"][: anchor + 1].long()
    absolute = event["absolute_seconds"][: anchor + 1].float()
    teams = event["team_local_index"][: anchor + 1].long()
    positions, position_mask = _representative_positions(event)
    positions = positions[: anchor + 1]
    position_mask = position_mask[: anchor + 1]
    positions = torch.where(
        (teams == anchor_team).unsqueeze(-1), positions, 1.0 - positions
    )

    sequence_type = torch.zeros((count, HISTORY_LENGTH), dtype=torch.long)
    sequence_numeric = torch.zeros((count, HISTORY_LENGTH, 7), dtype=torch.float32)
    sequence_mask = torch.zeros((count, HISTORY_LENGTH), dtype=torch.bool)
    lengths = torch.zeros(count, dtype=torch.long)
    statistics = torch.zeros((count, HISTORY_STATS_DIM), dtype=torch.float32)
    history_count = torch.zeros(count, dtype=torch.long)
    recency_seconds = torch.full((count,), float("inf"), dtype=torch.float32)
    possession_participated = torch.zeros(count, dtype=torch.bool)

    active_possession = int(event["active_possession_after_local_index"][anchor])
    active_valid = active_possession >= 0
    if not active_valid:
        active_possession = int(event["possession_local_index"][anchor])
        active_valid = active_possession >= 0
    possession_ids = event["possession_local_index"][: anchor + 1].long()

    for player in range(count):
        indices = torch.nonzero(player_index == player, as_tuple=False).flatten()
        n = int(indices.numel())
        history_count[player] = n
        if n == 0:
            continue
        statistics[player, 0] = 1.0
        statistics[player, 1] = _normalized_log(n, anchor + 1)
        for offset, window in enumerate((10, 20, 80), start=2):
            start = max(0, anchor - window + 1)
            available = anchor - start + 1
            statistics[player, offset] = float((indices >= start).sum()) / available
        last = int(indices[-1])
        event_age = anchor - last
        seconds_age = max(anchor_time - float(absolute[last]), 0.0)
        recency_seconds[player] = seconds_age
        statistics[player, 5] = _normalized_log(event_age, anchor + 1)
        statistics[player, 6] = _normalized_log(seconds_age, anchor_time)
        type_counts = torch.bincount(event_types[indices], minlength=10).float()[:10]
        statistics[player, 7:17] = type_counts / max(n, 1)
        statistics[player, 17] = float(active_valid)
        participated = bool(active_valid and (possession_ids[indices] == active_possession).any())
        statistics[player, 18] = float(participated)
        possession_participated[player] = participated
        valid_positions = indices[position_mask[indices]]
        if valid_positions.numel():
            latest_position = int(valid_positions[-1])
            statistics[player, 19:21] = positions[latest_position]
            statistics[player, 21] = 1.0
        if n > 1:
            intervals = absolute[indices[1:]] - absolute[indices[:-1]]
            recent_intervals = intervals[-HISTORY_LENGTH:].clamp(min=0.0)
            statistics[player, 22] = _normalized_log(
                float(recent_intervals.mean()), 7200.0
            )

        recent = indices[-HISTORY_LENGTH:]
        length = int(recent.numel())
        destination = torch.arange(HISTORY_LENGTH - length, HISTORY_LENGTH)
        lengths[player] = length
        sequence_mask[player, destination] = True
        sequence_type[player, destination] = event_types[recent]
        for local, source in zip(destination.tolist(), recent.tolist()):
            previous = indices[indices < source]
            interval_valid = bool(previous.numel())
            interval = (
                max(float(absolute[source] - absolute[int(previous[-1])]), 0.0)
                if interval_valid
                else 0.0
            )
            sequence_numeric[player, local] = torch.tensor(
                (
                    _normalized_log(anchor - source, anchor + 1),
                    _normalized_log(anchor_time - float(absolute[source]), anchor_time),
                    _normalized_log(interval, 7200.0),
                    float(interval_valid),
                    float(positions[source, 0]) if position_mask[source] else 0.0,
                    float(positions[source, 1]) if position_mask[source] else 0.0,
                    float(position_mask[source]),
                )
            )

    return PlayerHistoryTensors(
        sequence_type,
        sequence_numeric,
        sequence_mask,
        lengths,
        statistics,
        torch.zeros(count, dtype=torch.bool),
        history_count,
        recency_seconds,
        possession_participated,
    )


def _cached_history(sample: CanonicalSample) -> PlayerHistoryTensors:
    key = (int(sample.match_id), int(sample.current_event_index))
    value = _HISTORY_CACHE.get(key)
    if value is None:
        value = _history_for_sample(sample)
        _HISTORY_CACHE[key] = value
    return value


@lru_cache(maxsize=1024)
def _team_derangement(
    match_id: int, raw_ids: tuple[int, ...], selector_seed: int
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    roster = load_roster_team_map().get(match_id, {})
    donor = list(range(len(raw_ids)))
    eligible = [False] * len(raw_ids)
    groups: dict[int, list[int]] = {}
    for index, raw_id in enumerate(raw_ids):
        team = roster.get(int(raw_id))
        if team is not None:
            groups.setdefault(int(team), []).append(index)
    for team, indices in groups.items():
        indices.sort(key=lambda value: raw_ids[value])
        if len(indices) < 2:
            continue
        digest = hashlib.sha256(
            f"{match_id}:{team}:{selector_seed}".encode()
        ).digest()
        shift = int.from_bytes(digest[:8], "little") % (len(indices) - 1) + 1
        for position, index in enumerate(indices):
            donor[index] = indices[(position + shift) % len(indices)]
            eligible[index] = True
    return tuple(donor), tuple(eligible)


def _apply_condition(
    history: PlayerHistoryTensors,
    condition: str,
    match_id: int,
    raw_ids: torch.Tensor,
    selector_seed: int,
) -> PlayerHistoryTensors:
    if condition == "ph_hist_k5":
        return history
    if condition == "ph_null":
        return PlayerHistoryTensors(
            torch.zeros_like(history.event_type),
            torch.zeros_like(history.numeric),
            torch.zeros_like(history.sequence_mask),
            torch.zeros_like(history.lengths),
            torch.zeros_like(history.statistics),
            torch.zeros_like(history.shuffle_eligible),
            history.history_count,
            history.recency_seconds,
            history.current_possession_participated,
        )
    if condition != "ph_shuffled_team":
        raise ValueError(f"Unknown Player-history condition: {condition}")
    donor, eligible = _team_derangement(
        int(match_id), tuple(int(value) for value in raw_ids.tolist()), selector_seed
    )
    donor_tensor = torch.tensor(donor, dtype=torch.long)
    eligible_tensor = torch.tensor(eligible, dtype=torch.bool)

    def shuffled(value: torch.Tensor) -> torch.Tensor:
        result = value[donor_tensor].clone()
        if result.ndim == 1:
            result[~eligible_tensor] = 0
        else:
            result[~eligible_tensor] = 0
        return result

    return PlayerHistoryTensors(
        shuffled(history.event_type),
        shuffled(history.numeric),
        shuffled(history.sequence_mask),
        shuffled(history.lengths),
        shuffled(history.statistics),
        eligible_tensor,
        history.history_count,
        history.recency_seconds,
        history.current_possession_participated,
    )


def collate_player_history_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
    *,
    condition: str,
    selector_seed: int = SELECTOR_SEED,
) -> dict[str, Any]:
    batch = collate_five_task_hgt(
        samples,
        artifacts,
        window_size,
        context_views=("f80",),
        selector_seed=selector_seed,
    )
    values: list[PlayerHistoryTensors] = []
    target_shuffle_eligible = []
    target_team_mapping_valid = []
    roster = load_roster_team_map()
    for sample in samples:
        raw_ids = sample.graph["node_stores"]["player"]["raw_id"].long()
        original = _cached_history(sample)
        conditioned = _apply_condition(
            original, condition, sample.match_id, raw_ids, selector_seed
        )
        values.append(conditioned)
        target = int(sample.graph["node_stores"]["event"]["player_local_index"][sample.current_event_index + 1])
        _, eligibility = _team_derangement(
            int(sample.match_id),
            tuple(int(value) for value in raw_ids.tolist()),
            selector_seed,
        )
        target_shuffle_eligible.append(bool(eligibility[target]))
        target_raw = int(raw_ids[target])
        target_team_mapping_valid.append(target_raw in roster.get(sample.match_id, {}))
    graph = batch["graphs"]["f80"]
    batch["player_history"] = {
        "event_type": torch.cat([value.event_type for value in values]),
        "numeric": torch.cat([value.numeric for value in values]),
        "sequence_mask": torch.cat([value.sequence_mask for value in values]),
        "lengths": torch.cat([value.lengths for value in values]),
        "statistics": torch.cat([value.statistics for value in values]),
        "shuffle_eligible": torch.cat([value.shuffle_eligible for value in values]),
        "target_shuffle_eligible": torch.tensor(target_shuffle_eligible),
        "target_team_mapping_valid": torch.tensor(target_team_mapping_valid),
        "target_history_count": torch.tensor([
            int(value.history_count[int(batch["targets"]["player_local"][row])])
            for row, value in enumerate(values)
        ]),
        "target_recency_seconds": torch.tensor([
            float(value.recency_seconds[int(batch["targets"]["player_local"][row])])
            for row, value in enumerate(values)
        ]),
        "target_current_possession_participated": torch.tensor([
            bool(value.current_possession_participated[int(batch["targets"]["player_local"][row])])
            for row, value in enumerate(values)
        ]),
    }
    if graph["player"].num_nodes != batch["player_history"]["lengths"].numel():
        raise RuntimeError("Candidate histories do not align with batched Player nodes")
    return batch
