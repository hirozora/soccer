"""Load serialized graphs and create causally valid event-window views."""

from __future__ import annotations

import csv
from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .graph_builder import validate_graph
from .schema import (
    EDGE_TYPES,
    RESULT_DIRECTION_TO_INDEX,
    SCHEMA_VERSION,
    edge_type_key,
)


WINDOW_NODE_TYPES = ("event", "player", "team", "tag")
WINDOW_EDGE_TYPES = (
    ("event", "next", "event"),
    ("player", "performs", "event"),
    ("team", "performs", "event"),
    ("tag", "describes", "event"),
)


@dataclass(frozen=True)
class MatchGraphRecord:
    """One match graph entry required by :class:`FixedWindowDataset`."""

    competition_slug: str
    match_id: int
    graph_path: Path
    num_events: int


def load_match_graph(path: str | Path, validate: bool = True) -> dict[str, Any]:
    graph = torch.load(Path(path), map_location="cpu", weights_only=True)
    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {graph.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if validate:
        errors = validate_graph(graph)
        if errors:
            raise ValueError(f"Invalid graph {path}: {'; '.join(errors)}")
    return graph


def slice_event_prefix(
    graph: dict[str, Any], end_event_index: int, validate: bool = True
) -> dict[str, Any]:
    """Keep events 0..end_event_index and all incident historical edges."""

    original_events = int(graph["node_stores"]["event"]["num_nodes"])
    if end_event_index < 0 or end_event_index >= original_events:
        raise IndexError(
            f"end_event_index must be in [0, {original_events - 1}], got {end_event_index}"
        )
    prefix_size = end_event_index + 1
    result = deepcopy(graph)

    event_store = result["node_stores"]["event"]
    for name, value in list(event_store.items()):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            event_store[name] = value[:prefix_size].clone()
    event_store["num_nodes"] = prefix_size

    for name, value in list(result["targets"].items()):
        result["targets"][name] = value[:prefix_size].clone()

    for source, relation, destination in EDGE_TYPES:
        key = edge_type_key((source, relation, destination))
        edge_index = result["edge_stores"][key]["edge_index"]
        keep = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if source == "event":
            keep &= edge_index[0] < prefix_size
        if destination == "event":
            keep &= edge_index[1] < prefix_size
        result["edge_stores"][key]["edge_index"] = edge_index[:, keep].clone()

    result["prefix"] = {
        "end_event_index": end_event_index,
        "original_num_events": original_events,
    }
    if validate:
        errors = validate_graph(result)
        if errors:
            raise ValueError(f"Invalid prefix graph: {'; '.join(errors)}")
    return result


def _slice_event_store(
    store: dict[str, Any], start: int, stop: int
) -> dict[str, Any]:
    """Create tensor views for one event interval without copying full tensors."""

    num_events = int(store["num_nodes"])
    result: dict[str, Any] = {"num_nodes": stop - start}
    for name, value in store.items():
        if name == "num_nodes":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] != num_events:
                raise ValueError(
                    f"event.{name} has first dimension {value.shape[0]}, "
                    f"expected {num_events}"
                )
            result[name] = value[start:stop]
        else:
            result[name] = value
    return result


def _static_node_store(store: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy a node store while sharing its immutable tensors."""

    return dict(store)


def _window_tag_edges(
    graph: dict[str, Any], start: int, stop: int
) -> torch.Tensor:
    """Slice sorted Tag->Event edges and relabel event destinations."""

    key = edge_type_key(("tag", "describes", "event"))
    edge_index = graph["edge_stores"][key]["edge_index"]
    event_destinations = edge_index[1]
    bounds = torch.tensor(
        [start, stop], dtype=event_destinations.dtype, device=event_destinations.device
    )
    left, right = torch.searchsorted(event_destinations, bounds).tolist()
    selected = edge_index[:, left:right]
    if selected.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.stack((selected[0], selected[1] - start))


def _match_local_player_target(
    player_store: dict[str, Any], target_vocab_index: torch.Tensor, known: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a global player vocabulary target to the match candidate index."""

    device = target_vocab_index.device
    if not known:
        return (
            torch.tensor(-1, dtype=torch.long, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )

    candidates = torch.nonzero(
        player_store["vocab_index"] == target_vocab_index, as_tuple=False
    ).flatten()
    if candidates.numel() != 1:
        raise ValueError(
            "A known player target must map to exactly one match-local candidate; "
            f"found {candidates.numel()} for vocabulary index "
            f"{int(target_vocab_index)}"
        )
    return candidates[0].clone(), torch.tensor(True, dtype=torch.bool, device=device)


def sample_fixed_event_window(
    graph: dict[str, Any],
    current_event_index: int,
    window_size: int = 40,
    validate: bool = False,
) -> dict[str, Any]:
    """Build a causal fixed-length view ending at ``current_event_index``.

    The returned graph contains no target or future event nodes. Event tensors are
    views into the loaded match graph, static entity tensors are shared, and only
    the small selected edge tensors and scalar targets are materialized.
    """

    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")

    event_store = graph["node_stores"]["event"]
    num_events = int(event_store["num_nodes"])
    if current_event_index < 0 or current_event_index >= num_events - 1:
        raise IndexError(
            "current_event_index must identify an event with a next-event target "
            f"in [0, {num_events - 2}], got {current_event_index}"
        )
    if not bool(graph["targets"]["mask"][current_event_index]):
        raise ValueError(f"Event {current_event_index} has no valid next-event target")

    start = max(0, current_event_index - window_size + 1)
    stop = current_event_index + 1
    window_length = stop - start
    device = event_store["raw_id"].device
    local_events = torch.arange(window_length, dtype=torch.long, device=device)

    node_stores = {
        "event": _slice_event_store(event_store, start, stop),
        "player": _static_node_store(graph["node_stores"]["player"]),
        "team": _static_node_store(graph["node_stores"]["team"]),
        "tag": _static_node_store(graph["node_stores"]["tag"]),
    }
    if window_length > 1:
        next_edges = torch.stack((local_events[:-1], local_events[1:]))
    else:
        next_edges = torch.empty((2, 0), dtype=torch.long, device=device)

    edge_stores = {
        edge_type_key(("event", "next", "event")): {"edge_index": next_edges},
        edge_type_key(("player", "performs", "event")): {
            "edge_index": torch.stack(
                (node_stores["event"]["player_local_index"], local_events)
            )
        },
        edge_type_key(("team", "performs", "event")): {
            "edge_index": torch.stack(
                (node_stores["event"]["team_local_index"], local_events)
            )
        },
        edge_type_key(("tag", "describes", "event")): {
            "edge_index": _window_tag_edges(graph, start, stop)
        },
    }

    source_targets = graph["targets"]
    delta_seconds = source_targets["delta_seconds"][current_event_index].clone()
    if float(delta_seconds) < 0.0:
        raise ValueError(f"Negative next-event interval at event {current_event_index}")

    current_team = event_store["team_local_index"][current_event_index]
    target_team = source_targets["team_local_index"][current_event_index].clone()
    acting_side = (target_team == current_team).to(dtype=torch.long)

    player_vocab = source_targets["player_vocab_index"][current_event_index].clone()
    player_known = bool(source_targets["player_known_mask"][current_event_index])
    player_local, player_mask = _match_local_player_target(
        node_stores["player"], player_vocab, player_known
    )

    result_direction = source_targets["result_direction"][current_event_index].clone()
    favorable = RESULT_DIRECTION_TO_INDEX["favorable"]
    unfavorable = RESULT_DIRECTION_TO_INDEX["unfavorable"]
    if int(result_direction) == favorable:
        advantage = torch.tensor(1, dtype=torch.long, device=result_direction.device)
        advantage_mask = torch.tensor(
            True, dtype=torch.bool, device=result_direction.device
        )
    elif int(result_direction) == unfavorable:
        advantage = torch.tensor(0, dtype=torch.long, device=result_direction.device)
        advantage_mask = torch.tensor(
            True, dtype=torch.bool, device=result_direction.device
        )
    else:
        advantage = torch.tensor(-1, dtype=torch.long, device=result_direction.device)
        advantage_mask = torch.tensor(
            False, dtype=torch.bool, device=result_direction.device
        )

    targets = {
        "event_type_index": source_targets["event_type_index"][
            current_event_index
        ].clone(),
        "delta_seconds": delta_seconds,
        "log_delta_seconds": torch.log1p(delta_seconds),
        "start_position": source_targets["start_position"][
            current_event_index
        ].clone(),
        "start_position_mask": source_targets["start_position_mask"][
            current_event_index
        ].clone(),
        "team_local_index": target_team,
        "acting_side_index": acting_side,
        "player_vocab_index": player_vocab,
        "player_local_index": player_local,
        "player_mask": player_mask,
        "result_direction": result_direction,
        "advantage_index": advantage,
        "advantage_mask": advantage_mask,
    }

    sample = {
        "schema_version": graph["schema_version"],
        "sample_type": "fixed_event_window",
        "match_id": graph["match_id"],
        "competition_id": graph["competition_id"],
        "competition_slug": graph["competition_slug"],
        "node_types": list(WINDOW_NODE_TYPES),
        "edge_types": [list(edge_type) for edge_type in WINDOW_EDGE_TYPES],
        "node_stores": node_stores,
        "edge_stores": edge_stores,
        "targets": targets,
        "window": {
            "window_size": window_size,
            "start_event_index": start,
            "current_event_index": current_event_index,
            "target_event_index": current_event_index + 1,
            "num_events": window_length,
            "query_event_index": window_length - 1,
        },
    }
    if validate:
        errors = validate_fixed_event_window(sample)
        if errors:
            raise ValueError(f"Invalid fixed event window: {'; '.join(errors)}")
    return sample


def validate_fixed_event_window(sample: dict[str, Any]) -> list[str]:
    """Return structural errors for a sampled fixed event window."""

    errors: list[str] = []
    if sample.get("sample_type") != "fixed_event_window":
        errors.append("sample_type must be fixed_event_window")

    node_stores = sample.get("node_stores", {})
    edge_stores = sample.get("edge_stores", {})
    if set(node_stores) != set(WINDOW_NODE_TYPES):
        errors.append("window node types do not match the Version 1 contract")

    expected_edge_keys = {edge_type_key(value) for value in WINDOW_EDGE_TYPES}
    if set(edge_stores) != expected_edge_keys:
        errors.append("window edge types do not match the Version 1 contract")
    if errors:
        return errors

    node_counts = {
        node_type: int(node_stores[node_type]["num_nodes"])
        for node_type in WINDOW_NODE_TYPES
    }
    window = sample.get("window", {})
    num_events = node_counts["event"]
    if num_events < 1 or num_events > int(window.get("window_size", 0)):
        errors.append("event count must be in [1, window_size]")
    if int(window.get("num_events", -1)) != num_events:
        errors.append("window num_events does not match the event node store")
    if int(window.get("query_event_index", -1)) != num_events - 1:
        errors.append("query_event_index must identify the last event node")
    if int(window.get("target_event_index", -1)) != int(
        window.get("current_event_index", -2)
    ) + 1:
        errors.append("target_event_index must immediately follow current_event_index")

    for source, relation, destination in WINDOW_EDGE_TYPES:
        key = edge_type_key((source, relation, destination))
        edge_index = edge_stores[key]["edge_index"]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            errors.append(f"{key}.edge_index must have shape [2, E]")
            continue
        if edge_index.numel():
            if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= node_counts[source]:
                errors.append(f"{key} has an out-of-bounds source index")
            if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= node_counts[destination]:
                errors.append(f"{key} has an out-of-bounds destination index")

    event_device = node_stores["event"]["raw_id"].device
    if num_events > 1:
        local_events = torch.arange(num_events, dtype=torch.long, device=event_device)
        expected_next = torch.stack((local_events[:-1], local_events[1:]))
    else:
        expected_next = torch.empty((2, 0), dtype=torch.long, device=event_device)
    actual_next = edge_stores[edge_type_key(("event", "next", "event"))][
        "edge_index"
    ]
    if not torch.equal(actual_next, expected_next):
        errors.append("event next edges do not match window chronology")

    for target_name in (
        "event_type_index",
        "delta_seconds",
        "log_delta_seconds",
        "start_position",
        "start_position_mask",
        "acting_side_index",
        "player_local_index",
        "player_mask",
        "advantage_index",
        "advantage_mask",
    ):
        if target_name not in sample.get("targets", {}):
            errors.append(f"missing target {target_name}")
    return errors


def load_match_index(
    dataset_root: str | Path,
    competitions: Iterable[str] | None = None,
    limit_matches: int | None = None,
) -> list[MatchGraphRecord]:
    """Load graph paths and event counts from the generated match index."""

    root = Path(dataset_root)
    selected = set(competitions) if competitions is not None else None
    records: list[MatchGraphRecord] = []
    with (root / "metadata/match_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if selected is not None and row["competition_slug"] not in selected:
                continue
            records.append(
                MatchGraphRecord(
                    competition_slug=row["competition_slug"],
                    match_id=int(row["match_id"]),
                    graph_path=root / row["graph_path"],
                    num_events=int(row["num_events"]),
                )
            )
            if limit_matches is not None and len(records) >= limit_matches:
                break
    if not records:
        raise ValueError("No match graphs matched the requested index filters")
    return records


class FixedWindowDataset(Dataset[dict[str, Any]]):
    """Index all valid next-event windows across serialized match graphs.

    Loaded match graphs are retained in a small per-process LRU cache. Training
    samplers should group nearby indices by match to benefit from this cache.
    """

    def __init__(
        self,
        records: Sequence[MatchGraphRecord],
        window_size: int = 40,
        cache_size: int = 2,
        validate_graph_on_load: bool = False,
        validate_samples: bool = False,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        if not records:
            raise ValueError("records must contain at least one match")
        if any(record.num_events < 2 for record in records):
            raise ValueError("Every match record must contain at least two events")

        self.records = tuple(records)
        self.window_size = window_size
        self.cache_size = cache_size
        self.validate_graph_on_load = validate_graph_on_load
        self.validate_samples = validate_samples
        self._sample_ends = tuple(
            accumulate(record.num_events - 1 for record in self.records)
        )
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str | Path,
        window_size: int = 40,
        competitions: Iterable[str] | None = None,
        limit_matches: int | None = None,
        **kwargs: Any,
    ) -> "FixedWindowDataset":
        records = load_match_index(dataset_root, competitions, limit_matches)
        return cls(records=records, window_size=window_size, **kwargs)

    def __len__(self) -> int:
        return self._sample_ends[-1]

    def _load_graph(self, record: MatchGraphRecord) -> dict[str, Any]:
        path = record.graph_path
        if path in self._cache:
            graph = self._cache.pop(path)
            self._cache[path] = graph
            return graph

        graph = load_match_graph(path, validate=self.validate_graph_on_load)
        actual_events = int(graph["node_stores"]["event"]["num_nodes"])
        if actual_events != record.num_events:
            raise ValueError(
                f"Index reports {record.num_events} events for {path}, "
                f"but graph contains {actual_events}"
            )
        self._cache[path] = graph
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return graph

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        record_index = bisect_right(self._sample_ends, index)
        previous_end = self._sample_ends[record_index - 1] if record_index else 0
        current_event_index = index - previous_end
        graph = self._load_graph(self.records[record_index])
        return sample_fixed_event_window(
            graph,
            current_event_index=current_event_index,
            window_size=self.window_size,
            validate=self.validate_samples,
        )

    def clear_cache(self) -> None:
        self._cache.clear()


def hgt_inputs(graph: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], torch.Tensor]]:
    """Expose node stores and typed edge indices in an HGT-friendly form."""

    edge_index_dict = {
        edge_type: graph["edge_stores"][edge_type_key(edge_type)]["edge_index"]
        for edge_type in EDGE_TYPES
    }
    return graph["node_stores"], edge_index_dict
