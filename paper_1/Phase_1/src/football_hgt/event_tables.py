"""Build graph-independent normalized event tables from raw Wyscout JSON."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import PERIOD_NOMINAL_SECONDS, PERIOD_ORDER


EVENT_TABLE_SCHEMA_VERSION = "1.0.0"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/whyscout"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "processed/event_tables/v1"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _position(
    event: dict[str, Any], index: int
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    positions = event.get("positions") or []
    if index >= len(positions):
        return None, None, None, None, False
    point = positions[index]
    raw_x = float(point["x"])
    raw_y = float(point["y"])
    return raw_x, raw_y, raw_x / 100.0, raw_y / 100.0, True


def _absolute_times(events: list[dict[str, Any]]) -> list[float]:
    maximum: dict[str, float] = defaultdict(float)
    for event in events:
        period = str(event["matchPeriod"])
        maximum[period] = max(maximum[period], float(event["eventSec"]))
    offsets: dict[str, float] = {}
    elapsed = 0.0
    for period in PERIOD_ORDER:
        if period not in maximum:
            continue
        offsets[period] = elapsed
        elapsed += max(PERIOD_NOMINAL_SECONDS[period], maximum[period])
    return [offsets[str(event["matchPeriod"])] + float(event["eventSec"]) for event in events]


def _nullable_int(frame: pd.DataFrame, fields: tuple[str, ...]) -> None:
    for field in fields:
        frame[field] = frame[field].astype("Int64")


def build_event_tables(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    competition: str = "England",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize one Wyscout competition without applying graph tag filters."""

    data_root = Path(data_root).resolve()
    competition_root = Path(output_root).resolve() / competition
    manifest_path = competition_root / "metadata/manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    event_path = data_root / "raw/events" / f"events_{competition}.json"
    match_path = data_root / "raw/matches" / f"matches_{competition}.json"
    event_mapping_path = data_root / "raw/mappings/eventid2name.csv"
    tag_mapping_path = data_root / "raw/mappings/tags2name.csv"
    players_path = data_root / "raw/entities/players.json"
    for path in (
        event_path,
        match_path,
        event_mapping_path,
        tag_mapping_path,
        players_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    event_mapping = {
        (int(row["event"]), int(row["subevent"])): row
        for row in _mapping_rows(event_mapping_path)
    }
    tag_mapping = {
        int(row["Tag"]): row for row in _mapping_rows(tag_mapping_path)
    }
    known_players = {int(row["wyId"]) for row in _read_json(players_path)}
    raw_matches = _read_json(match_path)
    matches_by_id = {int(row["wyId"]): row for row in raw_matches}
    raw_events = _read_json(event_path)

    indexed_by_match: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for source_index, event in enumerate(raw_events):
        indexed_by_match[int(event["matchId"])].append((source_index, event))
    missing_matches = set(indexed_by_match) - set(matches_by_id)
    if missing_matches:
        raise ValueError(f"Events reference missing matches: {sorted(missing_matches)[:10]}")

    event_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    for match_id in sorted(indexed_by_match):
        ordered_pairs = sorted(
            indexed_by_match[match_id],
            key=lambda pair: (
                PERIOD_ORDER[str(pair[1]["matchPeriod"])],
                float(pair[1]["eventSec"]),
                pair[0],
            ),
        )
        ordered_events = [event for _, event in ordered_pairs]
        absolute_times = _absolute_times(ordered_events)
        previous_absolute: float | None = None
        previous_period: str | None = None
        for event_index, ((source_index, event), absolute_seconds) in enumerate(
            zip(ordered_pairs, absolute_times, strict=True)
        ):
            event_type_id = int(event["eventId"])
            raw_subevent = event.get("subEventId")
            subevent_id = (
                None if raw_subevent in (None, "") else int(raw_subevent)
            )
            if subevent_id is not None and (event_type_id, subevent_id) not in event_mapping:
                raise ValueError(
                    f"Undefined event/subevent pair {(event_type_id, subevent_id)}"
                )
            period = str(event["matchPeriod"])
            period_break = previous_period is not None and period != previous_period
            delta = (
                0.0
                if previous_absolute is None
                else max(float(absolute_seconds - previous_absolute), 0.0)
            )
            start = _position(event, 0)
            end = _position(event, 1)
            player_id = int(event.get("playerId") or 0)
            raw_tags = event.get("tags") or []
            event_uid = int(event["id"])
            event_rows.append(
                {
                    "competition": competition,
                    "competition_id": int(matches_by_id[match_id]["competitionId"]),
                    "match_id": match_id,
                    "event_uid": event_uid,
                    "source_event_index": source_index,
                    "event_index": event_index,
                    "event_type_id": event_type_id,
                    "event_name": str(event["eventName"]),
                    "subevent_id": subevent_id,
                    "subevent_name": str(event.get("subEventName") or ""),
                    "period": period,
                    "period_index": PERIOD_ORDER[period],
                    "event_seconds": float(event["eventSec"]),
                    "absolute_seconds": float(absolute_seconds),
                    "delta_from_previous_seconds": delta,
                    "period_break": period_break,
                    "team_id": int(event["teamId"]),
                    "player_id": player_id,
                    "player_known": player_id != 0 and player_id in known_players,
                    "position_count": len(event.get("positions") or []),
                    "start_x_100": start[0],
                    "start_y_100": start[1],
                    "start_x": start[2],
                    "start_y": start[3],
                    "start_position_valid": start[4],
                    "end_x_100": end[0],
                    "end_y_100": end[1],
                    "end_x": end[2],
                    "end_y": end[3],
                    "end_position_valid": end[4],
                    "tag_count": len(raw_tags),
                }
            )
            for tag_order, item in enumerate(raw_tags):
                tag_id = int(item["id"])
                metadata = tag_mapping.get(tag_id)
                if metadata is None:
                    raise ValueError(f"Undefined tag {tag_id}")
                tag_rows.append(
                    {
                        "competition": competition,
                        "match_id": match_id,
                        "event_uid": event_uid,
                        "event_index": event_index,
                        "tag_order": tag_order,
                        "tag_id": tag_id,
                        "tag_label": metadata["Label"],
                        "tag_description": metadata["Description"],
                    }
                )
            previous_absolute = absolute_seconds
            previous_period = period

    match_rows: list[dict[str, Any]] = []
    match_team_rows: list[dict[str, Any]] = []
    for match in sorted(raw_matches, key=lambda row: int(row["wyId"])):
        match_id = int(match["wyId"])
        if match_id not in indexed_by_match:
            continue
        match_rows.append(
            {
                "competition": competition,
                "competition_id": int(match["competitionId"]),
                "match_id": match_id,
                "season_id": int(match.get("seasonId") or 0),
                "round_id": int(match.get("roundId") or 0),
                "gameweek": int(match.get("gameweek") or 0),
                "date": str(match.get("date") or ""),
                "date_utc": str(match.get("dateutc") or ""),
                "duration": str(match.get("duration") or ""),
                "status": str(match.get("status") or ""),
                "winner_team_id": int(match.get("winner") or 0),
                "venue": str(match.get("venue") or ""),
                "label": str(match.get("label") or ""),
                "group_name": str(match.get("groupName") or ""),
            }
        )
        for team_key, team in sorted(
            (match.get("teamsData") or {}).items(), key=lambda item: int(item[0])
        ):
            match_team_rows.append(
                {
                    "competition": competition,
                    "match_id": match_id,
                    "team_id": int(team.get("teamId") or team_key),
                    "side": str(team.get("side") or ""),
                    "coach_id": int(team.get("coachId") or 0),
                    "has_formation": bool(team.get("hasFormation")),
                    "score": int(team.get("score") or 0),
                    "score_ht": int(team.get("scoreHT") or 0),
                    "score_et": int(team.get("scoreET") or 0),
                    "score_p": int(team.get("scoreP") or 0),
                }
            )

    events_frame = pd.DataFrame(event_rows)
    _nullable_int(events_frame, ("subevent_id",))
    tags_frame = pd.DataFrame(tag_rows)
    matches_frame = pd.DataFrame(match_rows)
    match_teams_frame = pd.DataFrame(match_team_rows)

    errors: list[str] = []
    if len(events_frame) != len(raw_events):
        errors.append("normalized event count differs from raw input")
    if int(events_frame["tag_count"].sum()) != len(tags_frame):
        errors.append("event tag count differs from raw tag bridge count")
    if events_frame["event_uid"].duplicated().any():
        errors.append("event_uid is not unique")
    if events_frame[["match_id", "event_index"]].duplicated().any():
        errors.append("match event index is not unique")
    if not events_frame["start_x"].dropna().between(0.0, 1.0).all():
        errors.append("start x coordinate outside [0, 1]")
    if not events_frame["start_y"].dropna().between(0.0, 1.0).all():
        errors.append("start y coordinate outside [0, 1]")
    if not events_frame["end_x"].dropna().between(0.0, 1.0).all():
        errors.append("end x coordinate outside [0, 1]")
    if not events_frame["end_y"].dropna().between(0.0, 1.0).all():
        errors.append("end y coordinate outside [0, 1]")
    if errors:
        raise ValueError("Invalid normalized event tables: " + "; ".join(errors))

    _write_parquet(events_frame, competition_root / "events.parquet")
    _write_parquet(tags_frame, competition_root / "event_tags.parquet")
    _write_parquet(matches_frame, competition_root / "matches.parquet")
    _write_parquet(match_teams_frame, competition_root / "match_teams.parquet")

    schema = {
        "schema_version": EVENT_TABLE_SCHEMA_VERSION,
        "competition": competition,
        "event_order": ["period_index", "event_seconds", "source_event_index"],
        "absolute_time_policy": "active play clock matching graph schema 1.1.0",
        "coordinate_policy": "raw 0-100 and normalized 0-1; missing values remain null",
        "tag_policy": "all raw tags retained without graph pipeline filtering",
        "tables": {
            "events": list(events_frame.columns),
            "event_tags": list(tags_frame.columns),
            "matches": list(matches_frame.columns),
            "match_teams": list(match_teams_frame.columns),
        },
    }
    _write_json(schema, competition_root / "metadata/schema.json")
    validation = {
        "valid": True,
        "errors": [],
        "events": len(events_frame),
        "matches": int(events_frame["match_id"].nunique()),
        "event_tags": len(tags_frame),
        "unknown_players": int((~events_frame["player_known"]).sum()),
        "missing_end_positions": int((~events_frame["end_position_valid"]).sum()),
    }
    _write_json(validation, competition_root / "metadata/validation_report.json")

    analysis_root = competition_root / "analysis"
    for columns, filename in (
        (("event_type_id", "event_name"), "event_type_counts.csv"),
        (("event_type_id", "event_name", "subevent_id", "subevent_name"), "subevent_type_counts.csv"),
        (("period",), "period_counts.csv"),
    ):
        counts = (
            events_frame.groupby(list(columns), dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        analysis_root.mkdir(parents=True, exist_ok=True)
        counts.to_csv(analysis_root / filename, index=False)
    (
        tags_frame.groupby(["tag_id", "tag_label", "tag_description"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .to_csv(analysis_root / "tag_counts.csv", index=False)
    )
    pd.DataFrame(
        [
            {"field": column, "missing_count": int(events_frame[column].isna().sum())}
            for column in events_frame.columns
        ]
    ).to_csv(analysis_root / "missingness.csv", index=False)

    source_paths = (
        event_path,
        match_path,
        event_mapping_path,
        tag_mapping_path,
        players_path,
    )
    manifest = {
        "schema_version": EVENT_TABLE_SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": competition,
        "data_root": str(data_root),
        "output_root": str(competition_root),
        "totals": {
            "events": len(events_frame),
            "event_tags": len(tags_frame),
            "matches": len(matches_frame),
            "match_teams": len(match_teams_frame),
        },
        "source_files": [
            {
                "path": str(path.relative_to(data_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "output_files": [
            {
                "path": filename,
                "size_bytes": (competition_root / filename).stat().st_size,
                "sha256": _sha256(competition_root / filename),
            }
            for filename in (
                "events.parquet",
                "event_tags.parquet",
                "matches.parquet",
                "match_teams.parquet",
            )
        ],
    }
    _write_json(manifest, manifest_path)
    return manifest
