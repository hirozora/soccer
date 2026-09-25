"""Deterministic conversion and validation for semantic spatiotemporal graphs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from .constants import (
    GRAPH_ROOT,
    SEMANTIC_GRAPH_ROOT,
    SEMANTIC_GRAPH_VERSION,
    TEMPORAL_RELATIONS,
    ZONE_CENTERS_100,
)
from .mappings import position_to_zone


SEMANTIC_NODE_TYPES = ("event", "player", "team", "event_type", "tag", "zone")
SEMANTIC_EDGE_TYPES = (
    ("event", "next", "event"),
    *(("event", relation, "event") for relation in TEMPORAL_RELATIONS),
    ("player", "performs", "event"),
    ("event", "performed_by", "player"),
    ("team", "performs", "event"),
    ("event", "performed_by_team", "team"),
    ("event", "has_type", "event_type"),
    ("event_type", "describes", "event"),
    ("event", "has_tag", "tag"),
    ("tag", "describes", "event"),
    ("event", "starts_in", "zone"),
    ("zone", "start_of", "event"),
    ("event", "ends_in", "zone"),
    ("zone", "end_of", "event"),
)


def edge_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


def temporal_relation(delta_seconds: float, period_changed: bool) -> str:
    """Return the exclusive known-history relation for one consecutive pair."""

    if period_changed:
        return "period_break"
    delta = max(float(delta_seconds), 0.0)
    if delta < 2.0:
        return "gap_0_2s"
    if delta < 5.0:
        return "gap_2_5s"
    if delta < 15.0:
        return "gap_5_15s"
    if delta <= 60.0:
        return "gap_15_60s"
    return "gap_60plus"


def _edge_index(sources: Iterable[int], destinations: Iterable[int]) -> torch.Tensor:
    source = torch.as_tensor(list(sources), dtype=torch.long)
    destination = torch.as_tensor(list(destinations), dtype=torch.long)
    if source.numel() != destination.numel():
        raise ValueError("Edge source and destination counts differ")
    if source.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((source, destination))


def _clone_store(store: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in store.items()
    }


def _zone_edges(event: dict[str, Any], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    positions = event[f"{prefix}_position"]
    mask = event[f"{prefix}_position_mask"].bool()
    event_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if event_indices.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, empty.clone()
    zones = position_to_zone(positions[event_indices]).long()
    event_to_zone = torch.stack((event_indices, zones))
    return event_to_zone, event_to_zone.flip(0)


def convert_graph(source: dict[str, Any]) -> dict[str, Any]:
    """Convert one v1 match graph without changing any event or target tensor."""

    source_nodes = source["node_stores"]
    source_edges = source["edge_stores"]
    missing_nodes = set(SEMANTIC_NODE_TYPES[:-1]) - set(source_nodes)
    if missing_nodes:
        raise ValueError(f"Source graph lacks nodes: {sorted(missing_nodes)}")

    node_stores = {
        node_type: _clone_store(source_nodes[node_type])
        for node_type in SEMANTIC_NODE_TYPES[:-1]
    }
    centers = torch.tensor(ZONE_CENTERS_100, dtype=torch.float32) / 100.0
    node_stores["zone"] = {
        "num_nodes": len(ZONE_CENTERS_100),
        "vocab_index": torch.arange(len(ZONE_CENTERS_100), dtype=torch.long),
        "center_xy": centers,
    }

    retained = (
        ("event", "next", "event"),
        ("player", "performs", "event"),
        ("event", "performed_by", "player"),
        ("team", "performs", "event"),
        ("event", "performed_by_team", "team"),
        ("event", "has_type", "event_type"),
        ("event_type", "describes", "event"),
        ("event", "has_tag", "tag"),
        ("tag", "describes", "event"),
    )
    edges: dict[str, torch.Tensor] = {}
    for edge_type in retained:
        key = edge_key(edge_type)
        if key not in source_edges:
            raise ValueError(f"Source graph lacks edge store {key}")
        edges[key] = source_edges[key]["edge_index"].clone()

    event = node_stores["event"]
    num_events = int(event["num_nodes"])
    temporal_sources: dict[str, list[int]] = {name: [] for name in TEMPORAL_RELATIONS}
    temporal_destinations: dict[str, list[int]] = {
        name: [] for name in TEMPORAL_RELATIONS
    }
    periods = event["period_index"]
    absolute = event["absolute_seconds"]
    for source_index in range(num_events - 1):
        destination_index = source_index + 1
        relation = temporal_relation(
            float(absolute[destination_index] - absolute[source_index]),
            bool(periods[destination_index] != periods[source_index]),
        )
        temporal_sources[relation].append(source_index)
        temporal_destinations[relation].append(destination_index)
    for relation in TEMPORAL_RELATIONS:
        edges[edge_key(("event", relation, "event"))] = _edge_index(
            temporal_sources[relation], temporal_destinations[relation]
        )

    start_forward, start_reverse = _zone_edges(event, "start")
    end_forward, end_reverse = _zone_edges(event, "end")
    edges[edge_key(("event", "starts_in", "zone"))] = start_forward
    edges[edge_key(("zone", "start_of", "event"))] = start_reverse
    edges[edge_key(("event", "ends_in", "zone"))] = end_forward
    edges[edge_key(("zone", "end_of", "event"))] = end_reverse

    graph = {
        "schema_version": SEMANTIC_GRAPH_VERSION,
        "source_schema_version": source.get("schema_version"),
        "graph_unit": source.get("graph_unit", "match"),
        "match_id": int(source["match_id"]),
        "competition_id": int(source["competition_id"]),
        "competition_slug": source["competition_slug"],
        "node_types": list(SEMANTIC_NODE_TYPES),
        "edge_types": [list(value) for value in SEMANTIC_EDGE_TYPES],
        "node_stores": node_stores,
        "edge_stores": {
            key: {"edge_index": value} for key, value in edges.items()
        },
        "targets": _clone_store(source["targets"]),
        "causal_contract": {
            "event_order": "preserved_from_v1",
            "training_requires_prefix_slice": True,
            "next_edge_direction": "past_to_future",
            "temporal_buckets_use_observed_history_only": True,
        },
    }
    errors = validate_semantic_graph(graph, source)
    if errors:
        raise ValueError(f"Invalid semantic graph {graph['match_id']}: {'; '.join(errors)}")
    return graph


def _reverse_equal(forward: torch.Tensor, reverse: torch.Tensor) -> bool:
    return torch.equal(forward, reverse.flip(0))


def validate_semantic_graph(
    graph: dict[str, Any], source: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    if graph.get("schema_version") != SEMANTIC_GRAPH_VERSION:
        errors.append("schema_version mismatch")
    nodes = graph.get("node_stores", {})
    edges = graph.get("edge_stores", {})
    if set(nodes) != set(SEMANTIC_NODE_TYPES):
        errors.append("node type set mismatch")
        return errors
    expected_edges = {edge_key(value) for value in SEMANTIC_EDGE_TYPES}
    if set(edges) != expected_edges:
        errors.append("edge type set mismatch")
        return errors

    counts = {name: int(store["num_nodes"]) for name, store in nodes.items()}
    for source_type, relation, destination_type in SEMANTIC_EDGE_TYPES:
        key = edge_key((source_type, relation, destination_type))
        index = edges[key]["edge_index"]
        if index.dtype != torch.long or index.ndim != 2 or index.shape[0] != 2:
            errors.append(f"{key} has invalid edge_index")
            continue
        if index.numel():
            if int(index[0].min()) < 0 or int(index[0].max()) >= counts[source_type]:
                errors.append(f"{key} source index out of range")
            if int(index[1].min()) < 0 or int(index[1].max()) >= counts[destination_type]:
                errors.append(f"{key} destination index out of range")

    reverse_pairs = (
        (("player", "performs", "event"), ("event", "performed_by", "player")),
        (("team", "performs", "event"), ("event", "performed_by_team", "team")),
        (("event", "has_type", "event_type"), ("event_type", "describes", "event")),
        (("event", "has_tag", "tag"), ("tag", "describes", "event")),
        (("event", "starts_in", "zone"), ("zone", "start_of", "event")),
        (("event", "ends_in", "zone"), ("zone", "end_of", "event")),
    )
    for forward_type, reverse_type in reverse_pairs:
        forward = edges[edge_key(forward_type)]["edge_index"]
        reverse = edges[edge_key(reverse_type)]["edge_index"]
        if not _reverse_equal(forward, reverse):
            errors.append(f"{edge_key(forward_type)} reverse mismatch")

    next_edges = edges[edge_key(("event", "next", "event"))]["edge_index"]
    temporal = torch.cat(
        [
            edges[edge_key(("event", relation, "event"))]["edge_index"]
            for relation in TEMPORAL_RELATIONS
        ],
        dim=1,
    )
    if next_edges.shape[1] != temporal.shape[1]:
        errors.append("temporal relation count differs from next count")
    elif next_edges.numel():
        next_pairs = Counter(map(tuple, next_edges.t().tolist()))
        temporal_pairs = Counter(map(tuple, temporal.t().tolist()))
        if next_pairs != temporal_pairs or any(value != 1 for value in temporal_pairs.values()):
            errors.append("temporal relations are not an exclusive partition of next")

    if source is not None:
        source_event = source["node_stores"]["event"]
        event = nodes["event"]
        for name in ("raw_id", "event_type_index", "absolute_seconds"):
            if not torch.equal(event[name], source_event[name]):
                errors.append(f"event field {name} changed")
        for name, value in source["targets"].items():
            if isinstance(value, torch.Tensor) and not torch.equal(graph["targets"][name], value):
                errors.append(f"target field {name} changed")
    return errors


def _atomic_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_semantic_dataset(
    source_root: Path = GRAPH_ROOT,
    output_root: Path = SEMANTIC_GRAPH_ROOT,
    competition: str = "England",
    overwrite: bool = False,
    limit_matches: int | None = None,
) -> dict[str, Any]:
    index_path = source_root / "metadata/match_index.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        source_rows = [
            row for row in csv.DictReader(handle) if row["competition_slug"] == competition
        ]
    if limit_matches is not None:
        source_rows = source_rows[:limit_matches]
    output_rows: list[dict[str, Any]] = []
    totals = Counter()
    built = 0
    reused = 0
    for row in source_rows:
        relative = Path(row["graph_path"])
        source_path = source_root / relative
        output_path = output_root / relative
        source = torch.load(source_path, map_location="cpu", weights_only=True)
        if output_path.exists() and not overwrite:
            graph = torch.load(output_path, map_location="cpu", weights_only=True)
            errors = validate_semantic_graph(graph, source)
            if errors:
                raise ValueError(f"Invalid existing {output_path}: {'; '.join(errors)}")
            reused += 1
        else:
            graph = convert_graph(source)
            _atomic_save(graph, output_path)
            built += 1
        event_count = int(graph["node_stores"]["event"]["num_nodes"])
        zone_edges = sum(
            graph["edge_stores"][edge_key(("event", relation, "zone"))]["edge_index"].shape[1]
            for relation in ("starts_in", "ends_in")
        )
        temporal_counts = {
            relation: int(
                graph["edge_stores"][edge_key(("event", relation, "event"))][
                    "edge_index"
                ].shape[1]
            )
            for relation in TEMPORAL_RELATIONS
        }
        output_rows.append(
            {
                **row,
                "schema_version": SEMANTIC_GRAPH_VERSION,
                "num_zone_edges": zone_edges,
                **{f"num_{key}": value for key, value in temporal_counts.items()},
            }
        )
        totals["matches"] += 1
        totals["events"] += event_count
        totals["zone_edges"] += zone_edges
        totals.update({f"temporal_{key}": value for key, value in temporal_counts.items()})

    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    index_output = metadata / "match_index.csv"
    temporary = index_output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(temporary, index_output)
    schema = {
        "schema_version": SEMANTIC_GRAPH_VERSION,
        "source_schema": "1.1.0",
        "node_types": list(SEMANTIC_NODE_TYPES),
        "edge_types": [list(value) for value in SEMANTIC_EDGE_TYPES],
        "zone_centers_100": ZONE_CENTERS_100,
        "relative_features": "computed causally at window collation",
    }
    _write_json(schema, metadata / "graph_schema.json")
    manifest = {
        "schema_version": SEMANTIC_GRAPH_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "competition": competition,
        "limit_matches": limit_matches,
        "built": built,
        "reused": reused,
        "totals": dict(totals),
        "source_index_sha256": file_sha256(index_path),
    }
    _write_json(manifest, metadata / "manifest.json")
    return manifest
