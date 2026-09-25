"""Training-only benchmark artifacts and immutable label statistics."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .constants import (
    ACTION_NAMES,
    DEFAULT_ARTIFACT_PATH,
    FINE_EVENT_NAMES,
    GRAPH_ROOT,
    RAW_EVENT_NAMES,
    SPLIT_PATH,
    VOCAB_PATH,
)
from .mappings import action4_label, build_fold_matrix, position_to_zone, unified_fine_label
from .sampling import TargetSamplePlan


TAG_EDGE_KEY = "tag__describes__event"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_weights(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.float64)
    active = counts > 0
    result = torch.zeros_like(counts)
    if active.any():
        total = counts[active].sum()
        result[active] = total / (active.sum() * counts[active])
        result[active] /= result[active].mean()
    return result.float()


def event_tag_sets(graph: dict[str, Any]) -> list[frozenset[int]]:
    """Return raw tag IDs for every event in one portable match graph."""

    num_events = int(graph["node_stores"]["event"]["num_nodes"])
    result: list[set[int]] = [set() for _ in range(num_events)]
    tag_store = graph["node_stores"]["tag"]
    tag_raw_ids = tag_store["raw_id"]
    edges = graph["edge_stores"][TAG_EDGE_KEY]["edge_index"]
    for tag_local, event_index in edges.T.tolist():
        result[event_index].add(int(tag_raw_ids[tag_local]))
    return [frozenset(values) for values in result]


@dataclass(frozen=True)
class ProtocolArtifacts:
    """All train-derived state required by data loaders and evaluators."""

    player_to_index: dict[int, int]
    team_to_index: dict[int, int]
    tag_to_index: dict[int, int]
    event_type_ids: tuple[int, ...]
    subevent_type_ids: tuple[int | None, ...]
    cardinalities: dict[str, int]
    fold_matrix: torch.Tensor
    fine_active_mask: torch.Tensor
    class_counts: dict[str, torch.Tensor]
    class_weights: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    @property
    def num_players(self) -> int:
        return len(self.player_to_index) + 1

    @property
    def num_teams(self) -> int:
        return len(self.team_to_index) + 1

    @property
    def num_tags(self) -> int:
        return len(self.tag_to_index)

    def state_dict(self) -> dict[str, Any]:
        return {
            "player_to_index": self.player_to_index,
            "team_to_index": self.team_to_index,
            "tag_to_index": self.tag_to_index,
            "event_type_ids": self.event_type_ids,
            "subevent_type_ids": self.subevent_type_ids,
            "cardinalities": self.cardinalities,
            "fold_matrix": self.fold_matrix,
            "fine_active_mask": self.fine_active_mask,
            "class_counts": self.class_counts,
            "class_weights": self.class_weights,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ProtocolArtifacts":
        return cls(**state)

    def save(self, path: str | Path = DEFAULT_ARTIFACT_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)
        return target

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "ProtocolArtifacts":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls.from_state_dict(state)


def _train_rows(split_path: Path) -> list[dict[str, str]]:
    with split_path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["competition_slug"] == "England" and row["split"] == "train"
        ]


def build_protocol_artifacts(
    split_path: str | Path = SPLIT_PATH,
    graph_root: str | Path = GRAPH_ROOT,
    vocab_path: str | Path = VOCAB_PATH,
    sample_plan: TargetSamplePlan | None = None,
) -> ProtocolArtifacts:
    """Scan England training matches and build leak-free benchmark state."""

    split_path = Path(split_path)
    graph_root = Path(graph_root)
    vocab_path = Path(vocab_path)
    with vocab_path.open(encoding="utf-8") as handle:
        vocab = json.load(handle)
    event_type_ids = tuple(int(value) for value in vocab["event_type_ids"])
    subevent_type_ids = tuple(
        None if value == "__UNKNOWN_SUBEVENT__" else int(value)
        for value in vocab["subevent_type_ids"]
    )

    players: set[int] = set()
    teams: set[int] = set()
    tags: set[int] = set()
    raw_targets: list[torch.Tensor] = []
    fine_targets: list[torch.Tensor] = []
    raw_counts = torch.zeros(len(RAW_EVENT_NAMES), dtype=torch.long)
    action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
    fine_counts = torch.zeros(len(FINE_EVENT_NAMES), dtype=torch.long)
    zone_counts = torch.zeros(20, dtype=torch.long)
    maxima = {"role": 0, "foot": 0, "team_side": 0, "team_type": 0, "tag_category": 0}
    sample_count = 0
    action_sample_count = 0
    position_sample_count = 0
    time_sample_count = 0

    rows = _train_rows(split_path)
    for row in rows:
        match_id = int(row["match_id"])
        graph = torch.load(
            graph_root / row["graph_path"], map_location="cpu", weights_only=True
        )
        event = graph["node_stores"]["event"]
        player = graph["node_stores"]["player"]
        team = graph["node_stores"]["team"]
        tag = graph["node_stores"]["tag"]
        players.update(int(value) for value in player["raw_id"].tolist() if int(value) > 0)
        teams.update(int(value) for value in team["raw_id"].tolist() if int(value) > 0)
        tags.update(int(value) for value in tag["raw_id"].tolist())
        maxima["role"] = max(maxima["role"], int(player["role_index"].max()))
        maxima["foot"] = max(maxima["foot"], int(player["foot_index"].max()))
        maxima["team_side"] = max(maxima["team_side"], int(team["side_index"].max()))
        maxima["team_type"] = max(maxima["team_type"], int(team["type_index"].max()))
        maxima["tag_category"] = max(
            maxima["tag_category"], int(tag["category_index"].max())
        )

        tag_sets = event_tag_sets(graph)
        match_raw: list[int] = []
        match_fine: list[int] = []
        num_events = int(event["num_nodes"])
        currents = (
            range(num_events - 1)
            if sample_plan is None
            else sample_plan.currents_by_match("train").get(match_id)
        )
        if currents is None:
            raise ValueError(f"Sample plan is missing training match {match_id}")
        for current in currents:
            target = current + 1
            raw_index = int(event["event_type_index"][target])
            event_id = event_type_ids[raw_index]
            subevent_id = subevent_type_ids[int(event["subevent_type_index"][target])]
            target_tags = tag_sets[target]
            fine_index = unified_fine_label(event_id, subevent_id, target_tags)
            action_index, action_mask = action4_label(
                event_id, subevent_id, target_tags
            )
            raw_counts[raw_index] += 1
            fine_counts[fine_index] += 1
            match_raw.append(raw_index)
            match_fine.append(fine_index)
            sample_count += 1
            if action_mask:
                action_counts[action_index] += 1
                action_sample_count += 1
            if bool(event["start_position_mask"][target]):
                zone = position_to_zone(event["start_position"][target]).item()
                zone_counts[int(zone)] += 1
                position_sample_count += 1
            if int(event["period_index"][current]) == int(event["period_index"][target]):
                time_sample_count += 1
        raw_targets.append(torch.tensor(match_raw, dtype=torch.long))
        fine_targets.append(torch.tensor(match_fine, dtype=torch.long))

    fold_matrix, fine_active = build_fold_matrix(
        torch.cat(raw_targets), torch.cat(fine_targets)
    )
    player_to_index = {raw_id: index + 1 for index, raw_id in enumerate(sorted(players))}
    team_to_index = {raw_id: index + 1 for index, raw_id in enumerate(sorted(teams))}
    tag_to_index = {raw_id: index for index, raw_id in enumerate(sorted(tags))}
    cardinalities = {
        "event": len(event_type_ids),
        "subevent": len(subevent_type_ids),
        "period": 5,
        "result": 4,
        "role": maxima["role"] + 1,
        "foot": maxima["foot"] + 1,
        "team_side": maxima["team_side"] + 1,
        "team_type": maxima["team_type"] + 1,
        "tag_category": maxima["tag_category"] + 1,
    }
    counts = {
        "raw10": raw_counts,
        "action4": action_counts,
        "fine32": fine_counts,
        "zone20": zone_counts,
    }
    weights = {name: _balanced_weights(values) for name, values in counts.items()}
    metadata = {
        "competition": "England",
        "split": "train",
        "matches": len(rows),
        "samples": sample_count,
        "action4_samples": action_sample_count,
        "position_samples": position_sample_count,
        "time_samples": time_sample_count,
        "split_sha256": _sha256(split_path),
        "vocab_sha256": _sha256(vocab_path),
        "raw_event_names": RAW_EVENT_NAMES,
        "action_names": ACTION_NAMES,
        "fine_event_names": FINE_EVENT_NAMES,
        "target_sampling": "full" if sample_plan is None else "temporal_stratified",
        "sampling_seed": None if sample_plan is None else sample_plan.sampling_seed,
        "targets_per_match": (
            None if sample_plan is None else sample_plan.targets_per_match
        ),
    }
    return ProtocolArtifacts(
        player_to_index=player_to_index,
        team_to_index=team_to_index,
        tag_to_index=tag_to_index,
        event_type_ids=event_type_ids,
        subevent_type_ids=subevent_type_ids,
        cardinalities=cardinalities,
        fold_matrix=fold_matrix,
        fine_active_mask=fine_active,
        class_counts=counts,
        class_weights=weights,
        metadata=metadata,
    )
