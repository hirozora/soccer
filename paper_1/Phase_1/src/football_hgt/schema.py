"""Stable schema and label policies for match-level heterogeneous graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final = "1.1.0"

NODE_TYPES: Final = (
    "event",
    "player",
    "team",
    "match",
    "competition",
    "event_type",
    "tag",
)

EDGE_TYPES: Final = (
    ("event", "next", "event"),
    ("player", "performs", "event"),
    ("event", "performed_by", "player"),
    ("team", "performs", "event"),
    ("event", "performed_by_team", "team"),
    ("event", "in_match", "match"),
    ("match", "contains", "event"),
    ("match", "in_competition", "competition"),
    ("competition", "contains", "match"),
    ("event", "has_type", "event_type"),
    ("event_type", "describes", "event"),
    ("event", "has_tag", "tag"),
    ("tag", "describes", "event"),
)

PERIOD_ORDER: Final = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
PERIOD_NOMINAL_SECONDS: Final = {
    "1H": 45.0 * 60.0,
    "2H": 45.0 * 60.0,
    "E1": 15.0 * 60.0,
    "E2": 15.0 * 60.0,
    "P": 0.0,
}

RESULT_DIRECTION_TO_INDEX: Final = {
    "neutral_or_unknown": 0,
    "favorable": 1,
    "unfavorable": 2,
}

# Generic signal sets are retained for conflict diagnostics. Production labels
# are resolved by event-specific rules loaded from the analysis rule table.
HIGH_PRIORITY_FAVORABLE_TAGS: Final = frozenset({101})
HIGH_PRIORITY_UNFAVORABLE_TAGS: Final = frozenset(
    {102, 1302, 1701, 1702, 1703, 2001, 2101}
)
FAVORABLE_TAGS: Final = frozenset({301, 302, 703, 1401, 1801})
UNFAVORABLE_TAGS: Final = frozenset({701, 1802})

UNKNOWN_PLAYER_ID: Final = 0
UNKNOWN_SUBEVENT_ID: Final = "__UNKNOWN_SUBEVENT__"


def edge_type_key(edge_type: tuple[str, str, str]) -> str:
    """Convert a typed edge triplet to its portable serialized key."""

    return "__".join(edge_type)


def classify_result_direction(
    event_name: str,
    tag_ids: set[int],
    rules: dict[tuple[str, int], "ResultDirectionRule"],
) -> int:
    """Resolve event-conditioned result tags using the highest-priority rule."""

    matched = [
        rules[(event_name, tag_id)]
        for tag_id in tag_ids
        if (event_name, tag_id) in rules
    ]
    if not matched:
        return RESULT_DIRECTION_TO_INDEX["neutral_or_unknown"]

    highest_priority = max(rule.priority for rule in matched)
    directions = {
        rule.direction for rule in matched if rule.priority == highest_priority
    }
    if len(directions) != 1:
        raise ValueError(
            f"Conflicting result-direction rules for {event_name!r} at priority "
            f"{highest_priority}: {sorted(directions)}"
        )
    return RESULT_DIRECTION_TO_INDEX[directions.pop()]


@dataclass(frozen=True)
class TagOverride:
    """Pipeline behavior override for one event-name/tag pair."""

    build_tag_edge: bool
    effective_category: str | None = None


@dataclass(frozen=True)
class ResultDirectionRule:
    """Event-specific polarity and precedence for one result tag."""

    direction: str
    priority: int
    note: str
