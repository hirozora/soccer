"""Command-line builder for the complete match-level graph dataset."""

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

from .catalog import load_catalog
from .graph_builder import build_match_graph
from .schema import (
    EDGE_TYPES,
    NODE_TYPES,
    PERIOD_NOMINAL_SECONDS,
    RESULT_DIRECTION_TO_INDEX,
    SCHEMA_VERSION,
    edge_type_key,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/whyscout"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed/heterogeneous_graphs/v1"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _source_manifest(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths)):
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return records


def _schema_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_unit": "one_match",
        "serialization": "torch.save portable tensor dictionary",
        "node_types": list(NODE_TYPES),
        "edge_types": [list(edge_type) for edge_type in EDGE_TYPES],
        "edge_store_key_format": "source__relation__destination",
        "event_order": ["matchPeriod order", "eventSec", "source order tie-break"],
        "time_policy": {
            "description": "monotonic active-play clock without halftime breaks",
            "period_nominal_seconds": PERIOD_NOMINAL_SECONDS,
            "period_offset_increment": "max(nominal period length, observed maximum eventSec)",
        },
        "position_policy": {
            "coordinates": "Wyscout x/y divided by 100",
            "start": "positions[0] with mask",
            "end": "positions[1] with mask; zero-filled when absent",
        },
        "unknown_policy": {
            "player_id_zero": "local UNKNOWN_PLAYER node, global player vocab index 0",
            "missing_player_metadata": "global player vocab index 0 and player target loss mask false",
            "empty_subevent_id": "subevent vocabulary index 0",
        },
        "tag_policy": {
            "taxonomy_source": "analysis/tag_taxonomy.csv",
            "overrides_source": "analysis/event_tag_pipeline_overrides.csv",
            "result_direction_rules_source": "analysis/event_result_direction_rules.csv",
            "duel_neutral_edge": "excluded",
            "result_direction_classes": RESULT_DIRECTION_TO_INDEX,
            "conflict_priority": "highest event-specific rule priority",
            "model_target": "binary favorable/unfavorable with neutral_or_unknown masked",
        },
        "causal_policy": {
            "complete_graph_is_storage_only": True,
            "training_requires_event_prefix_slice": True,
            "reason": "entity and tag nodes can otherwise aggregate future events",
        },
        "targets_aligned_to_event_nodes": [
            "next event_type_index",
            "next delta_seconds",
            "next start_position and optional end_position",
            "next team_local_index",
            "next player_vocab_index with player_known_mask",
            "next result_direction",
        ],
    }


def _graph_index_row(graph: dict[str, Any], relative_path: Path) -> dict[str, Any]:
    event_store = graph["node_stores"]["event"]
    direction_counts = Counter(
        int(value) for value in event_store["result_direction"].tolist()
    )
    return {
        "competition_slug": graph["competition_slug"],
        "competition_id": graph["competition_id"],
        "match_id": graph["match_id"],
        "graph_path": str(relative_path),
        "num_events": event_store["num_nodes"],
        "num_players": graph["node_stores"]["player"]["num_nodes"],
        "num_teams": graph["node_stores"]["team"]["num_nodes"],
        "num_tag_edges": graph["edge_stores"]["event__has_tag__tag"]["edge_index"].shape[1],
        "num_supervised_steps": int(graph["targets"]["mask"].sum()),
        "num_known_player_targets": int(graph["targets"]["player_known_mask"].sum()),
        "num_favorable_events": direction_counts[RESULT_DIRECTION_TO_INDEX["favorable"]],
        "num_unfavorable_events": direction_counts[RESULT_DIRECTION_TO_INDEX["unfavorable"]],
        "num_neutral_or_unknown_events": direction_counts[
            RESULT_DIRECTION_TO_INDEX["neutral_or_unknown"]
        ],
    }


def _write_match_index(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty match index")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_dataset(
    data_root: Path,
    output_root: Path,
    competitions: list[str] | None = None,
    limit_matches: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    catalog = load_catalog(data_root)
    event_paths = sorted((data_root / "raw/events").glob("events_*.json"))
    available = {path.stem.removeprefix("events_"): path for path in event_paths}
    selected = sorted(competitions or available)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(
            f"Unknown competitions {sorted(unknown)}; available values are {sorted(available)}"
        )

    metadata_dir = output_root / "metadata"
    graphs_dir = output_root / "graphs"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    _write_json(_schema_document(), metadata_dir / "graph_schema.json")
    _write_json(catalog.to_serializable_vocabularies(), metadata_dir / "vocabularies.json")

    index_rows: list[dict[str, Any]] = []
    source_paths = [
        data_root / "raw/entities/players.json",
        data_root / "raw/entities/teams.json",
        data_root / "raw/metadata/competitions.json",
        data_root / "raw/mappings/eventid2name.csv",
        data_root / "raw/mappings/tags2name.csv",
        data_root / "analysis/tag_taxonomy.csv",
        data_root / "analysis/event_tag_pipeline_overrides.csv",
        data_root / "analysis/event_result_direction_rules.csv",
    ]
    built = 0
    skipped = 0
    processed_matches = 0

    for competition_slug in selected:
        event_path = available[competition_slug]
        match_path = data_root / "raw/matches" / f"matches_{competition_slug}.json"
        source_paths.extend([event_path, match_path])
        events = _read_json(event_path)
        matches = {int(item["wyId"]): item for item in _read_json(match_path)}
        events_by_match: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            events_by_match[int(event["matchId"])].append(event)

        missing_metadata = set(events_by_match) - set(matches)
        if missing_metadata:
            raise ValueError(
                f"{competition_slug} has events for matches absent from metadata: "
                f"{sorted(missing_metadata)[:10]}"
            )

        for match_id in sorted(events_by_match):
            if limit_matches is not None and processed_matches >= limit_matches:
                break
            relative_path = Path("graphs") / competition_slug / f"{match_id}.pt"
            graph_path = output_root / relative_path
            if graph_path.exists() and not overwrite:
                graph = torch.load(graph_path, map_location="cpu", weights_only=True)
                skipped += 1
            else:
                graph = build_match_graph(
                    matches[match_id],
                    events_by_match[match_id],
                    catalog,
                    competition_slug,
                )
                _atomic_torch_save(graph, graph_path)
                built += 1
            index_rows.append(_graph_index_row(graph, relative_path))
            processed_matches += 1
        if limit_matches is not None and processed_matches >= limit_matches:
            break

    index_rows.sort(key=lambda row: (row["competition_slug"], int(row["match_id"])))
    _write_match_index(index_rows, metadata_dir / "match_index.csv")

    totals = {
        "matches": len(index_rows),
        "events": sum(int(row["num_events"]) for row in index_rows),
        "supervised_steps": sum(int(row["num_supervised_steps"]) for row in index_rows),
        "tag_edges": sum(int(row["num_tag_edges"]) for row in index_rows),
        "known_player_targets": sum(
            int(row["num_known_player_targets"]) for row in index_rows
        ),
        "favorable_events": sum(int(row["num_favorable_events"]) for row in index_rows),
        "unfavorable_events": sum(
            int(row["num_unfavorable_events"]) for row in index_rows
        ),
        "neutral_or_unknown_events": sum(
            int(row["num_neutral_or_unknown_events"]) for row in index_rows
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "selected_competitions": selected,
        "limit_matches": limit_matches,
        "graph_files_built": built,
        "graph_files_reused": skipped,
        "totals": totals,
        "vocabulary_sizes": {
            "players_including_unknown": len(catalog.player_vocab),
            "teams": len(catalog.team_vocab),
            "matches": len(catalog.match_vocab),
            "competitions": len(catalog.competition_vocab),
            "event_types": len(catalog.event_type_vocab),
            "subevent_types_including_unknown": len(catalog.subevent_type_vocab),
            "tags": len(catalog.tag_vocab),
        },
        "source_files": _source_manifest(source_paths, data_root),
    }
    _write_json(manifest, metadata_dir / "build_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build match-level heterogeneous football graphs from Wyscout data."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--competitions",
        nargs="+",
        help="Competition suffixes such as World_Cup or England. Default: all.",
    )
    parser.add_argument("--limit-matches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        data_root=args.data_root.resolve(),
        output_root=args.output_root.resolve(),
        competitions=args.competitions,
        limit_matches=args.limit_matches,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
