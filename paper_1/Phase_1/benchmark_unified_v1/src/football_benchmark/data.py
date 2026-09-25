"""Canonical immediate-next-event dataset and model-specific collation."""

from __future__ import annotations

import csv
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .constants import GRAPH_ROOT, SPLIT_PATH, TIME_CAP_SECONDS
from .mappings import action4_label, position_to_zone, unified_fine_label
from .protocol import ProtocolArtifacts, event_tag_sets
from .semantic_graph import SEMANTIC_EDGE_TYPES


@dataclass(frozen=True)
class MatchRecord:
    match_id: int
    graph_path: Path
    num_events: int


@dataclass(frozen=True)
class CanonicalTarget:
    raw_event_10: torch.Tensor
    action_4: torch.Tensor
    action_4_mask: torch.Tensor
    fine_event_32: torch.Tensor
    delta_seconds_60: torch.Tensor
    time_mask: torch.Tensor
    position_xy: torch.Tensor
    position_mask: torch.Tensor
    zone_20: torch.Tensor


@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    match_id: int
    current_event_index: int
    target_event_index: int
    graph: dict[str, Any]
    prepared: dict[str, torch.Tensor]
    start: int
    stop: int
    target: CanonicalTarget


def load_records(
    split: str,
    split_path: str | Path = SPLIT_PATH,
    graph_root: str | Path = GRAPH_ROOT,
    competition: str = "England",
) -> list[MatchRecord]:
    result: list[MatchRecord] = []
    with Path(split_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["competition_slug"] != competition or row["split"] != split:
                continue
            result.append(
                MatchRecord(
                    match_id=int(row["match_id"]),
                    graph_path=Path(graph_root) / row["graph_path"],
                    num_events=int(row["num_events"]),
                )
            )
    if not result:
        raise ValueError(f"No records found for {competition}/{split}")
    return result


def _prepare_graph(
    graph: dict[str, Any],
    tag_sets: tuple[frozenset[int], ...],
    artifacts: ProtocolArtifacts,
) -> dict[str, torch.Tensor]:
    """Materialize event-level sequence fields once per loaded match."""

    event = graph["node_stores"]["event"]
    player = graph["node_stores"]["player"]
    team = graph["node_stores"]["team"]
    player_local = event["player_local_index"]
    team_local = event["team_local_index"]
    player_raw = player["raw_id"][player_local]
    team_raw = team["raw_id"][team_local]
    player_vocab = torch.tensor(
        [artifacts.player_to_index.get(int(value), 0) for value in player_raw],
        dtype=torch.long,
    )
    team_vocab = torch.tensor(
        [artifacts.team_to_index.get(int(value), 0) for value in team_raw],
        dtype=torch.long,
    )
    tag_multihot = torch.zeros(
        (int(event["num_nodes"]), artifacts.num_tags), dtype=torch.float32
    )
    for event_index, tags in enumerate(tag_sets):
        for raw_tag in tags:
            tag_index = artifacts.tag_to_index.get(raw_tag)
            if tag_index is not None:
                tag_multihot[event_index, tag_index] = 1.0
    numeric = torch.cat(
        (
            (event["period_seconds"] / 3600.0).unsqueeze(-1),
            (event["absolute_seconds"] / 7200.0).unsqueeze(-1),
            torch.log1p(event["delta_from_previous"].clamp_min(0)).unsqueeze(-1),
            event["start_position"],
            event["start_position_mask"].float().unsqueeze(-1),
            event["end_position"],
            event["end_position_mask"].float().unsqueeze(-1),
            player["height_weight"][player_local, :1] / 200.0,
            player["height_weight"][player_local, 1:] / 100.0,
            player["metadata_mask"][player_local].float().unsqueeze(-1),
        ),
        dim=-1,
    )
    action = torch.full((int(event["num_nodes"]),), -1, dtype=torch.long)
    action_mask = torch.zeros(int(event["num_nodes"]), dtype=torch.bool)
    fine = torch.empty(int(event["num_nodes"]), dtype=torch.long)
    for index in range(int(event["num_nodes"])):
        raw_index = int(event["event_type_index"][index])
        event_id = artifacts.event_type_ids[raw_index]
        subevent_id = artifacts.subevent_type_ids[
            int(event["subevent_type_index"][index])
        ]
        action_value, mask = action4_label(event_id, subevent_id, tag_sets[index])
        action[index] = action_value
        action_mask[index] = mask
        fine[index] = unified_fine_label(event_id, subevent_id, tag_sets[index])
    zone = position_to_zone(event["start_position"])
    return {
        "event": event["event_type_index"],
        "subevent": event["subevent_type_index"],
        "period": event["period_index"],
        "result": event["result_direction"],
        "player": player_vocab,
        "role": player["role_index"][player_local],
        "foot": player["foot_index"][player_local],
        "team": team_vocab,
        "team_side": team["side_index"][team_local],
        "team_type": team["type_index"][team_local],
        "tags": tag_multihot,
        "numeric": numeric,
        "action4": action,
        "action4_mask": action_mask,
        "fine32": fine,
        "zone20": zone,
    }


class _GraphCache:
    def __init__(self, artifacts: ProtocolArtifacts, capacity: int = 3) -> None:
        self.artifacts = artifacts
        self.capacity = capacity
        self.values: OrderedDict[
            Path, tuple[dict[str, Any], dict[str, torch.Tensor]]
        ] = OrderedDict()

    def get(self, path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if path in self.values:
            value = self.values.pop(path)
            self.values[path] = value
            return value
        graph = torch.load(path, map_location="cpu", weights_only=True)
        tag_sets = tuple(event_tag_sets(graph))
        value = (graph, _prepare_graph(graph, tag_sets, self.artifacts))
        self.values[path] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


class CanonicalEventDataset(Dataset[CanonicalSample]):
    """Immediate transitions from a full split or a fixed target sample plan."""

    def __init__(
        self,
        records: Sequence[MatchRecord],
        artifacts: ProtocolArtifacts,
        window_size: int,
        max_samples: int | None = None,
        selected_currents: Mapping[int, Sequence[int]] | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.records = list(records)
        self.artifacts = artifacts
        self.window_size = window_size
        self.currents: list[tuple[int, ...] | None] = []
        counts: list[int] = []
        for record in self.records:
            if selected_currents is None:
                self.currents.append(None)
                counts.append(record.num_events - 1)
                continue
            if record.match_id not in selected_currents:
                raise ValueError(f"Sample plan is missing match {record.match_id}")
            values = tuple(int(value) for value in selected_currents[record.match_id])
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate sampled transition in match {record.match_id}")
            if values != tuple(sorted(values)):
                raise ValueError(f"Sampled transitions must be sorted for match {record.match_id}")
            if any(value < 0 or value >= record.num_events - 1 for value in values):
                raise ValueError(f"Sampled transition is out of range for match {record.match_id}")
            self.currents.append(values)
            counts.append(len(values))
        cumulative: list[int] = []
        total = 0
        for count in counts:
            total += count
            cumulative.append(total)
        self.cumulative = cumulative
        self.total = min(total, max_samples) if max_samples is not None else total
        self.cache = _GraphCache(artifacts)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> CanonicalSample:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        record_index = bisect_right(self.cumulative, index)
        previous = self.cumulative[record_index - 1] if record_index else 0
        local_index = index - previous
        selected = self.currents[record_index]
        current = local_index if selected is None else selected[local_index]
        record = self.records[record_index]
        graph, prepared = self.cache.get(record.graph_path)
        event = graph["node_stores"]["event"]
        target_index = current + 1
        raw_index = int(event["event_type_index"][target_index])
        action_index = int(prepared["action4"][target_index])
        action_mask = bool(prepared["action4_mask"][target_index])
        fine_index = int(prepared["fine32"][target_index])
        position = event["start_position"][target_index].clone()
        position_mask = event["start_position_mask"][target_index].clone().bool()
        zone = prepared["zone20"][target_index] if bool(position_mask) else torch.tensor(-1)
        time_mask = (
            event["period_index"][current] == event["period_index"][target_index]
        ).clone().bool()
        delta = graph["targets"]["delta_seconds"][current].clamp(
            min=0.0, max=TIME_CAP_SECONDS
        )
        start = max(0, current - self.window_size + 1)
        stop = current + 1
        target = CanonicalTarget(
            raw_event_10=torch.tensor(raw_index, dtype=torch.long),
            action_4=torch.tensor(action_index, dtype=torch.long),
            action_4_mask=torch.tensor(action_mask, dtype=torch.bool),
            fine_event_32=torch.tensor(fine_index, dtype=torch.long),
            delta_seconds_60=delta.float(),
            time_mask=time_mask,
            position_xy=position.float(),
            position_mask=position_mask,
            zone_20=zone.long(),
        )
        return CanonicalSample(
            sample_id=f"{record.match_id}:{current}",
            match_id=record.match_id,
            current_event_index=current,
            target_event_index=target_index,
            graph=graph,
            prepared=prepared,
            start=start,
            stop=stop,
            target=target,
        )


class MatchBlockShuffleSampler(Sampler[int]):
    """Shuffle matches and within-match steps while preserving graph-cache locality."""

    def __init__(self, dataset: CanonicalEventDataset, seed: int) -> None:
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        blocks: list[tuple[int, int]] = []
        previous = 0
        for stop in self.dataset.cumulative:
            clipped_stop = min(stop, len(self.dataset))
            if previous < clipped_stop:
                blocks.append((previous, clipped_stop))
            previous = stop
            if previous >= len(self.dataset):
                break
        for block_index in torch.randperm(
            len(blocks), generator=self.generator
        ).tolist():
            start, stop = blocks[block_index]
            local = torch.randperm(stop - start, generator=self.generator).tolist()
            for offset in local:
                yield start + offset


TARGET_FIELDS = tuple(CanonicalTarget.__dataclass_fields__)


def _stack_targets(samples: Sequence[CanonicalSample]) -> dict[str, torch.Tensor]:
    return {
        field: torch.stack([getattr(sample.target, field) for sample in samples])
        for field in TARGET_FIELDS
    }


def _sequence_fields(
    sample: CanonicalSample, artifacts: ProtocolArtifacts
) -> dict[str, torch.Tensor]:
    start, stop = sample.start, sample.stop
    return {field: sample.prepared[field][start:stop] for field in SEQUENCE_FIELDS}


SEQUENCE_FIELDS = (
    "event",
    "subevent",
    "period",
    "result",
    "player",
    "role",
    "foot",
    "team",
    "team_side",
    "team_type",
    "tags",
    "numeric",
)


def collate_sequence(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    padded_width: int | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    values = [_sequence_fields(sample, artifacts) for sample in samples]
    observed_width = max(item["event"].shape[0] for item in values)
    width = padded_width if padded_width is not None else observed_width
    if width < observed_width:
        raise ValueError("padded_width cannot be smaller than an observed history")
    valid = torch.zeros((len(samples), width), dtype=torch.bool)
    batch: dict[str, torch.Tensor] = {}
    for field in SEQUENCE_FIELDS:
        suffix = values[0][field].shape[1:]
        dtype = values[0][field].dtype
        result = torch.zeros((len(samples), width, *suffix), dtype=dtype)
        for row, item in enumerate(values):
            length = item[field].shape[0]
            result[row, width - length :] = item[field]
            valid[row, width - length :] = True
        batch[field] = result
    return {
        "sequence": batch,
        "valid_mask": valid,
        "targets": _stack_targets(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "match_ids": torch.tensor([sample.match_id for sample in samples]),
        "current_event_indices": torch.tensor(
            [sample.current_event_index for sample in samples]
        ),
    }


def _window_tag_edges(sample: CanonicalSample) -> torch.Tensor:
    graph = sample.graph
    edges = graph["edge_stores"]["tag__describes__event"]["edge_index"]
    destinations = edges[1]
    bounds = torch.tensor([sample.start, sample.stop], dtype=destinations.dtype)
    left, right = torch.searchsorted(destinations, bounds).tolist()
    selected = edges[:, left:right]
    if selected.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((selected[0], selected[1] - sample.start))


def sample_to_heterodata(sample: CanonicalSample, artifacts: ProtocolArtifacts) -> Any:
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("torch-geometric is required for HGT experiments") from exc

    graph = sample.graph
    data = HeteroData()
    event = graph["node_stores"]["event"]
    player = graph["node_stores"]["player"]
    team = graph["node_stores"]["team"]
    tag = graph["node_stores"]["tag"]
    start, stop = sample.start, sample.stop
    event_fields = (
        "event_type_index",
        "subevent_type_index",
        "period_index",
        "period_seconds",
        "absolute_seconds",
        "delta_from_previous",
        "start_position",
        "start_position_mask",
        "end_position",
        "end_position_mask",
        "result_direction",
    )
    data["event"].num_nodes = stop - start
    for field in event_fields:
        data["event"][field] = event[field][start:stop]

    data["player"].num_nodes = int(player["num_nodes"])
    for field in ("role_index", "foot_index", "height_weight", "metadata_mask"):
        data["player"][field] = player[field]
    data["player"].vocab_index = torch.tensor(
        [artifacts.player_to_index.get(int(value), 0) for value in player["raw_id"]],
        dtype=torch.long,
    )

    data["team"].num_nodes = int(team["num_nodes"])
    for field in ("side_index", "type_index"):
        data["team"][field] = team[field]
    data["team"].vocab_index = torch.tensor(
        [artifacts.team_to_index.get(int(value), 0) for value in team["raw_id"]],
        dtype=torch.long,
    )

    data["tag"].num_nodes = int(tag["num_nodes"])
    data["tag"].vocab_index = torch.tensor(
        [artifacts.tag_to_index[int(value)] for value in tag["raw_id"]],
        dtype=torch.long,
    )
    data["tag"].category_index = tag["category_index"]

    local_events = torch.arange(stop - start, dtype=torch.long)
    data[("event", "next", "event")].edge_index = (
        torch.stack((local_events[:-1], local_events[1:]))
        if local_events.numel() > 1
        else torch.empty((2, 0), dtype=torch.long)
    )
    data[("player", "performs", "event")].edge_index = torch.stack(
        (event["player_local_index"][start:stop], local_events)
    )
    data[("team", "performs", "event")].edge_index = torch.stack(
        (event["team_local_index"][start:stop], local_events)
    )
    data[("tag", "describes", "event")].edge_index = _window_tag_edges(sample)
    return data


def collate_hgt(
    samples: Sequence[CanonicalSample], artifacts: ProtocolArtifacts
) -> dict[str, Any]:
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch-geometric is required for HGT experiments") from exc
    return {
        "graph": Batch.from_data_list(
            [sample_to_heterodata(sample, artifacts) for sample in samples]
        ),
        "targets": _stack_targets(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "match_ids": torch.tensor([sample.match_id for sample in samples]),
        "current_event_indices": torch.tensor(
            [sample.current_event_index for sample in samples]
        ),
    }


def _representative_positions(event: Any, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    start_position = event["start_position"][start:stop]
    start_mask = event["start_position_mask"][start:stop].bool()
    end_position = event["end_position"][start:stop]
    end_mask = event["end_position_mask"][start:stop].bool()
    positions = torch.where(end_mask.unsqueeze(-1), end_position, start_position)
    return positions, end_mask | start_mask


def _relative_event_features(
    event: Any, start: int, stop: int, window_size: int
) -> torch.Tensor:
    length = stop - start
    absolute = event["absolute_seconds"][start:stop]
    age = (absolute[-1] - absolute).clamp_min(0.0)
    relative_index = (
        torch.arange(length, dtype=torch.float32) - float(length - 1)
    ) / float(max(window_size - 1, 1))
    normalized_age = (age / 7200.0).clamp(max=1.0)
    log_age = (torch.log1p(age) / torch.log1p(torch.tensor(7200.0))).clamp(max=1.0)

    positions, position_mask = _representative_positions(event, start, stop)
    teams = event["team_local_index"][start:stop]
    anchor_team = teams[-1]
    aligned_positions = torch.where(
        (teams == anchor_team).unsqueeze(-1), positions, 1.0 - positions
    )
    anchor_position = positions[-1]
    valid = position_mask & position_mask[-1]
    delta = anchor_position.unsqueeze(0) - aligned_positions
    delta = torch.where(valid.unsqueeze(-1), delta, torch.zeros_like(delta))
    distance = torch.linalg.vector_norm(delta, dim=-1)
    return torch.stack(
        (
            relative_index,
            normalized_age,
            log_age,
            delta[:, 0],
            delta[:, 1],
            distance,
            valid.float(),
        ),
        dim=-1,
    )


def _slice_semantic_edge(
    edge_index: torch.Tensor,
    source_type: str,
    destination_type: str,
    start: int,
    stop: int,
) -> torch.Tensor:
    keep = torch.ones(edge_index.shape[1], dtype=torch.bool)
    if source_type == "event":
        keep &= (edge_index[0] >= start) & (edge_index[0] < stop)
    if destination_type == "event":
        keep &= (edge_index[1] >= start) & (edge_index[1] < stop)
    selected = edge_index[:, keep].clone()
    if source_type == "event":
        selected[0] -= start
    if destination_type == "event":
        selected[1] -= start
    return selected


def sample_to_semantic_heterodata(
    sample: CanonicalSample, artifacts: ProtocolArtifacts, window_size: int
) -> Any:
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch-geometric is required for semantic HGT") from exc

    graph = sample.graph
    if graph.get("schema_version") != "2.0.0":
        raise ValueError("semantic_v2 HGT requires a semantic_v2 graph")
    data = HeteroData()
    start, stop = sample.start, sample.stop
    event = graph["node_stores"]["event"]
    event_fields = (
        "event_type_index",
        "subevent_type_index",
        "period_index",
        "period_seconds",
        "absolute_seconds",
        "delta_from_previous",
        "start_position",
        "start_position_mask",
        "end_position",
        "end_position_mask",
        "result_direction",
    )
    data["event"].num_nodes = stop - start
    for field in event_fields:
        data["event"][field] = event[field][start:stop]
    data["event"].relative_features = _relative_event_features(
        event, start, stop, window_size
    )

    player = graph["node_stores"]["player"]
    data["player"].num_nodes = int(player["num_nodes"])
    for field in ("role_index", "foot_index", "height_weight", "metadata_mask"):
        data["player"][field] = player[field]
    data["player"].vocab_index = torch.tensor(
        [artifacts.player_to_index.get(int(value), 0) for value in player["raw_id"]],
        dtype=torch.long,
    )

    team = graph["node_stores"]["team"]
    data["team"].num_nodes = int(team["num_nodes"])
    for field in ("side_index", "type_index"):
        data["team"][field] = team[field]
    data["team"].vocab_index = torch.tensor(
        [artifacts.team_to_index.get(int(value), 0) for value in team["raw_id"]],
        dtype=torch.long,
    )

    event_type = graph["node_stores"]["event_type"]
    data["event_type"].num_nodes = int(event_type["num_nodes"])
    data["event_type"].vocab_index = event_type["vocab_index"]

    tag = graph["node_stores"]["tag"]
    data["tag"].num_nodes = int(tag["num_nodes"])
    data["tag"].vocab_index = torch.tensor(
        [artifacts.tag_to_index[int(value)] for value in tag["raw_id"]],
        dtype=torch.long,
    )
    data["tag"].category_index = tag["category_index"]

    zone = graph["node_stores"]["zone"]
    data["zone"].num_nodes = int(zone["num_nodes"])
    data["zone"].vocab_index = zone["vocab_index"]
    data["zone"].center_xy = zone["center_xy"]

    for source_type, relation, destination_type in SEMANTIC_EDGE_TYPES:
        key = "__".join((source_type, relation, destination_type))
        data[(source_type, relation, destination_type)].edge_index = _slice_semantic_edge(
            graph["edge_stores"][key]["edge_index"],
            source_type,
            destination_type,
            start,
            stop,
        )
    return data


def collate_semantic_hgt(
    samples: Sequence[CanonicalSample],
    artifacts: ProtocolArtifacts,
    window_size: int,
) -> dict[str, Any]:
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch-geometric is required for semantic HGT") from exc
    return {
        "graph": Batch.from_data_list(
            [
                sample_to_semantic_heterodata(sample, artifacts, window_size)
                for sample in samples
            ]
        ),
        "targets": _stack_targets(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "match_ids": torch.tensor([sample.match_id for sample in samples]),
        "current_event_indices": torch.tensor(
            [sample.current_event_index for sample in samples]
        ),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    if "graph" in result:
        result["graph"] = result["graph"].to(device)
    if "sequence" in result:
        result["sequence"] = {
            key: value.to(device) for key, value in result["sequence"].items()
        }
        result["valid_mask"] = result["valid_mask"].to(device)
    result["targets"] = {
        key: value.to(device) for key, value in result["targets"].items()
    }
    result["match_ids"] = result["match_ids"].to(device)
    result["current_event_indices"] = result["current_event_indices"].to(device)
    return result
