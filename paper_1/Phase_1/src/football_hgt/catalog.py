"""Load global entity vocabularies and graph-construction metadata."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    RESULT_DIRECTION_TO_INDEX,
    ResultDirectionRule,
    TagOverride,
    UNKNOWN_PLAYER_ID,
    UNKNOWN_SUBEVENT_ID,
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _id_vocab(ids: list[int], unknown_id: int | None = None) -> dict[int, int]:
    ordered = sorted(set(ids))
    if unknown_id is not None:
        ordered = [unknown_id] + [value for value in ordered if value != unknown_id]
    return {raw_id: index for index, raw_id in enumerate(ordered)}


@dataclass(frozen=True)
class Catalog:
    players: dict[int, dict[str, Any]]
    teams: dict[int, dict[str, Any]]
    competitions: dict[int, dict[str, Any]]
    matches: dict[int, dict[str, Any]]
    event_types: dict[int, str]
    subevent_types: dict[str, dict[str, Any]]
    tags: dict[int, dict[str, str]]
    tag_overrides: dict[tuple[str, int], TagOverride]
    result_direction_rules: dict[tuple[str, int], ResultDirectionRule]
    player_vocab: dict[int, int]
    team_vocab: dict[int, int]
    competition_vocab: dict[int, int]
    match_vocab: dict[int, int]
    event_type_vocab: dict[int, int]
    subevent_type_vocab: dict[str, int]
    tag_vocab: dict[int, int]

    def to_serializable_vocabularies(self) -> dict[str, Any]:
        def ordered_values(vocab: dict[Any, int]) -> list[Any]:
            return [raw_id for raw_id, _ in sorted(vocab.items(), key=lambda item: item[1])]

        return {
            "player_ids": ordered_values(self.player_vocab),
            "team_ids": ordered_values(self.team_vocab),
            "competition_ids": ordered_values(self.competition_vocab),
            "match_ids": ordered_values(self.match_vocab),
            "event_type_ids": ordered_values(self.event_type_vocab),
            "subevent_type_ids": ordered_values(self.subevent_type_vocab),
            "tag_ids": ordered_values(self.tag_vocab),
        }


def load_catalog(data_root: Path) -> Catalog:
    """Load raw entity tables, mappings, taxonomy, overrides, and matches."""

    players_list = _read_json(data_root / "raw/entities/players.json")
    teams_list = _read_json(data_root / "raw/entities/teams.json")
    competitions_list = _read_json(data_root / "raw/metadata/competitions.json")

    matches_list: list[dict[str, Any]] = []
    for path in sorted((data_root / "raw/matches").glob("matches_*.json")):
        matches_list.extend(_read_json(path))

    players = {int(item["wyId"]): item for item in players_list}
    teams = {int(item["wyId"]): item for item in teams_list}
    competitions = {int(item["wyId"]): item for item in competitions_list}
    matches = {int(item["wyId"]): item for item in matches_list}

    event_types: dict[int, str] = {}
    subevent_types: dict[str, dict[str, Any]] = {
        UNKNOWN_SUBEVENT_ID: {
            "raw_id": None,
            "event_id": None,
            "label": "Unknown subevent",
        }
    }
    with (data_root / "raw/mappings/eventid2name.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            event_id = int(row["event"])
            subevent_id = str(int(row["subevent"]))
            event_types[event_id] = row["event_label"]
            subevent_types[subevent_id] = {
                "raw_id": int(row["subevent"]),
                "event_id": event_id,
                "label": row["subevent_label"],
            }

    tag_metadata: dict[int, dict[str, str]] = {}
    with (data_root / "raw/mappings/tags2name.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            tag_metadata[int(row["Tag"])] = {
                "label": row["Label"],
                "description": row["Description"],
                "category": "",
            }

    with (data_root / "analysis/tag_taxonomy.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            tag_id = int(row["tagId"])
            if tag_id not in tag_metadata:
                raise ValueError(f"Taxonomy references undefined tag {tag_id}")
            tag_metadata[tag_id]["category"] = row["tag_category"]

    tag_overrides: dict[tuple[str, int], TagOverride] = {}
    with (data_root / "analysis/event_tag_pipeline_overrides.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            category = row["effective_tag_category"].strip() or None
            tag_overrides[(row["eventName"], int(row["tagId"]))] = TagOverride(
                build_tag_edge=row["build_tag_edge"].strip() not in {"0", "false", "False"},
                effective_category=category,
            )

    result_direction_rules: dict[tuple[str, int], ResultDirectionRule] = {}
    rules_path = data_root / "analysis/event_result_direction_rules.csv"
    with rules_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_name = row["eventName"]
            event_id = int(row["eventId"])
            tag_id = int(row["tagId"])
            direction = row["direction"]
            key = (event_name, tag_id)
            if key in result_direction_rules:
                raise ValueError(f"Duplicate result-direction rule for {key}")
            if event_types.get(event_id) != event_name:
                raise ValueError(
                    f"Result rule event mismatch: {event_id} maps to "
                    f"{event_types.get(event_id)!r}, not {event_name!r}"
                )
            if tag_id not in tag_metadata:
                raise ValueError(f"Result rule references undefined tag {tag_id}")
            if tag_metadata[tag_id]["category"] != "event_result":
                raise ValueError(f"Result rule references non-result tag {tag_id}")
            if direction not in RESULT_DIRECTION_TO_INDEX:
                raise ValueError(f"Unknown result direction {direction!r}")
            result_direction_rules[key] = ResultDirectionRule(
                direction=direction,
                priority=int(row["priority"]),
                note=row["rule_note"],
            )

    observed_result_pairs: set[tuple[str, int]] = set()
    occurrence_path = data_root / "analysis/event_tag_occurrence_with_taxonomy.csv"
    with occurrence_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["tag_category"] == "event_result":
                observed_result_pairs.add((row["eventName"], int(row["tagId"])))
    missing_rules = observed_result_pairs - set(result_direction_rules)
    extra_rules = set(result_direction_rules) - observed_result_pairs
    if missing_rules or extra_rules:
        raise ValueError(
            "Result-direction rule coverage mismatch: "
            f"missing={sorted(missing_rules)}, extra={sorted(extra_rules)}"
        )

    subevent_ids = list(subevent_types)
    subevent_vocab = {UNKNOWN_SUBEVENT_ID: 0}
    numeric_subevents = sorted(
        (value for value in subevent_ids if value != UNKNOWN_SUBEVENT_ID),
        key=int,
    )
    subevent_vocab.update(
        {raw_id: index for index, raw_id in enumerate(numeric_subevents, start=1)}
    )

    return Catalog(
        players=players,
        teams=teams,
        competitions=competitions,
        matches=matches,
        event_types=event_types,
        subevent_types=subevent_types,
        tags=tag_metadata,
        tag_overrides=tag_overrides,
        result_direction_rules=result_direction_rules,
        player_vocab=_id_vocab(list(players), unknown_id=UNKNOWN_PLAYER_ID),
        team_vocab=_id_vocab(list(teams)),
        competition_vocab=_id_vocab(list(competitions)),
        match_vocab=_id_vocab(list(matches)),
        event_type_vocab=_id_vocab(list(event_types)),
        subevent_type_vocab=subevent_vocab,
        tag_vocab=_id_vocab(list(tag_metadata)),
    )
