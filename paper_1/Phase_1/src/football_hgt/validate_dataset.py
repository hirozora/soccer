"""Validate a built heterogeneous graph dataset against its index and manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .graph_builder import validate_graph
from .schema import (
    FAVORABLE_TAGS,
    HIGH_PRIORITY_FAVORABLE_TAGS,
    HIGH_PRIORITY_UNFAVORABLE_TAGS,
    RESULT_DIRECTION_TO_INDEX,
    SCHEMA_VERSION,
    UNFAVORABLE_TAGS,
)


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/whyscout/processed/heterogeneous_graphs/v1"
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    metadata_dir = dataset_root / "metadata"
    manifest = _read_json(metadata_dir / "build_manifest.json")
    with (metadata_dir / "match_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        index_rows = list(csv.DictReader(handle))

    errors: list[str] = []
    warnings: list[str] = []
    indexed_paths = {row["graph_path"] for row in index_rows}
    actual_paths = {
        str(path.relative_to(dataset_root))
        for path in (dataset_root / "graphs").rglob("*.pt")
    }
    if indexed_paths != actual_paths:
        missing = sorted(indexed_paths - actual_paths)
        extra = sorted(actual_paths - indexed_paths)
        if missing:
            errors.append(f"Index references {len(missing)} absent graph files: {missing[:5]}")
        if extra:
            errors.append(f"Found {len(extra)} graph files absent from index: {extra[:5]}")

    temporary_files = sorted(str(path.relative_to(dataset_root)) for path in dataset_root.rglob("*.tmp"))
    if temporary_files:
        errors.append(f"Found unfinished temporary files: {temporary_files[:5]}")

    totals = Counter()
    quality = Counter()
    conflict_patterns: Counter[tuple[int, tuple[int, ...]]] = Counter()
    conflict_resolutions: Counter[str] = Counter()
    result_direction_names = {value: key for key, value in RESULT_DIRECTION_TO_INDEX.items()}
    positive_result_tags = set(FAVORABLE_TAGS) | set(HIGH_PRIORITY_FAVORABLE_TAGS)
    negative_result_tags = set(UNFAVORABLE_TAGS) | set(HIGH_PRIORITY_UNFAVORABLE_TAGS)
    min_events_per_match: int | None = None
    max_events_per_match = 0
    seen_match_ids: set[int] = set()
    for row in index_rows:
        path = dataset_root / row["graph_path"]
        if not path.exists():
            continue
        try:
            graph = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            errors.append(f"{row['graph_path']}: load failed: {type(exc).__name__}: {exc}")
            continue

        graph_errors = validate_graph(graph)
        errors.extend(f"{row['graph_path']}: {message}" for message in graph_errors)
        match_id = int(graph.get("match_id", -1))
        if match_id in seen_match_ids:
            errors.append(f"Duplicate match_id {match_id}")
        seen_match_ids.add(match_id)
        if match_id != int(row["match_id"]):
            errors.append(
                f"{row['graph_path']}: graph match_id {match_id} differs from index {row['match_id']}"
            )

        event_store = graph["node_stores"]["event"]
        num_events = int(event_store["num_nodes"])
        min_events_per_match = (
            num_events
            if min_events_per_match is None
            else min(min_events_per_match, num_events)
        )
        max_events_per_match = max(max_events_per_match, num_events)
        num_tag_edges = int(
            graph["edge_stores"]["event__has_tag__tag"]["edge_index"].shape[1]
        )
        direction_counts = Counter(int(value) for value in event_store["result_direction"].tolist())
        tag_edges = graph["edge_stores"]["event__has_tag__tag"]["edge_index"]
        graph_tag_ids = graph["node_stores"]["tag"]["raw_id"]
        result_tags_by_event: dict[int, set[int]] = defaultdict(set)
        for event_index, tag_index in zip(tag_edges[0].tolist(), tag_edges[1].tolist()):
            tag_id = int(graph_tag_ids[tag_index])
            if tag_id in positive_result_tags or tag_id in negative_result_tags:
                result_tags_by_event[event_index].add(tag_id)
        conflicting_events = {
            event_index
            for event_index, tag_ids in result_tags_by_event.items()
            if tag_ids & positive_result_tags and tag_ids & negative_result_tags
        }
        event_type_raw_ids = graph["node_stores"]["event_type"]["raw_id"]
        for event_index in conflicting_events:
            event_type_index = int(event_store["event_type_index"][event_index])
            event_type_id = int(event_type_raw_ids[event_type_index])
            tag_ids = tuple(sorted(result_tags_by_event[event_index]))
            conflict_patterns[(event_type_id, tag_ids)] += 1
            direction_index = int(event_store["result_direction"][event_index])
            conflict_resolutions[result_direction_names[direction_index]] += 1
        measured = {
            "num_events": num_events,
            "num_players": int(graph["node_stores"]["player"]["num_nodes"]),
            "num_teams": int(graph["node_stores"]["team"]["num_nodes"]),
            "num_tag_edges": num_tag_edges,
            "num_supervised_steps": int(graph["targets"]["mask"].sum()),
            "num_known_player_targets": int(graph["targets"]["player_known_mask"].sum()),
            "num_favorable_events": direction_counts[RESULT_DIRECTION_TO_INDEX["favorable"]],
            "num_unfavorable_events": direction_counts[RESULT_DIRECTION_TO_INDEX["unfavorable"]],
            "num_neutral_or_unknown_events": direction_counts[
                RESULT_DIRECTION_TO_INDEX["neutral_or_unknown"]
            ],
        }
        for field, value in measured.items():
            if int(row[field]) != value:
                errors.append(
                    f"{row['graph_path']}: {field} is {value}, index records {row[field]}"
                )

        totals.update(
            {
                "matches": 1,
                "events": num_events,
                "supervised_steps": measured["num_supervised_steps"],
                "tag_edges": num_tag_edges,
                "known_player_targets": measured["num_known_player_targets"],
                "favorable_events": measured["num_favorable_events"],
                "unfavorable_events": measured["num_unfavorable_events"],
                "neutral_or_unknown_events": measured[
                    "num_neutral_or_unknown_events"
                ],
            }
        )
        quality.update(
            {
                "events_with_start_position": int(
                    event_store["start_position_mask"].sum()
                ),
                "events_with_end_position": int(event_store["end_position_mask"].sum()),
                "events_with_unknown_player_vocab": int(
                    (event_store["player_vocab_index"] == 0).sum()
                ),
                "events_with_unknown_subevent": int(
                    (event_store["subevent_type_index"] == 0).sum()
                ),
                "player_nodes_without_metadata": int(
                    (~graph["node_stores"]["player"]["metadata_mask"]).sum()
                ),
                "matches_without_exactly_two_teams": int(
                    graph["node_stores"]["team"]["num_nodes"] != 2
                ),
                "events_with_conflicting_result_signals": len(
                    conflicting_events
                ),
            }
        )

    expected_totals = {key: int(value) for key, value in manifest["totals"].items()}
    measured_totals = {key: int(totals[key]) for key in expected_totals}
    if measured_totals != expected_totals:
        errors.append(
            f"Measured totals differ from manifest: measured={measured_totals}, "
            f"expected={expected_totals}"
        )

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"Manifest schema version {manifest.get('schema_version')} differs from {SCHEMA_VERSION}"
        )

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "graph_files_checked": len(actual_paths),
        "index_rows_checked": len(index_rows),
        "measured_totals": measured_totals,
        "data_quality": {
            **{key: int(value) for key, value in sorted(quality.items())},
            "masked_player_targets": measured_totals["supervised_steps"]
            - measured_totals["known_player_targets"],
            "min_events_per_match": min_events_per_match,
            "max_events_per_match": max_events_per_match,
        },
        "result_signal_conflicts": {
            "total": int(sum(conflict_patterns.values())),
            "resolved_direction_counts": {
                key: int(value) for key, value in sorted(conflict_resolutions.items())
            },
            "top_patterns": [
                {
                    "event_type_id": event_type_id,
                    "tag_ids": list(tag_ids),
                    "count": int(count),
                }
                for (event_type_id, tag_ids), count in conflict_patterns.most_common(20)
            ],
        },
        "errors": errors,
        "warnings": warnings,
    }
    _write_json(report, metadata_dir / "validation_report.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a built graph dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
