"""Deterministic causal possession inference for normalized Wyscout events."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .event_tables import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT as EVENT_OUTPUT_ROOT


POSSESSION_SCHEMA_VERSION = "1.0.0"
DEFAULT_POSSESSION_ROOT = DEFAULT_DATA_ROOT / "processed/inferred_possessions/v1"
DEFAULT_RULES_PATH = DEFAULT_DATA_ROOT / "analysis/possession_rules.csv"
STATES = frozenset({"DEAD_BALL", "CONTROL", "CONTESTED"})
BASE_SIGNALS = frozenset({"restart", "control", "contest", "boundary", "neutral"})
POST_EFFECTS = frozenset({"hold", "contest", "close", "keep"})
TEAM_CANDIDATE_TAGS = frozenset({703, 1401, 1501})
OPPONENT_CANDIDATE_TAGS = frozenset({701, 1302, 1802, 2001})

STATE_STRING_COLUMNS = (
    "period",
    "possession_uid",
    "event_role",
    "state_before",
    "state_after",
    "active_possession_after_event",
    "closed_possession_before_event",
    "closed_possession_after_event",
    "boundary_reason",
    "evidence",
    "confidence",
)
STATE_NULLABLE_INT_COLUMNS = (
    "possession_index",
    "owner_team_id",
    "candidate_team_id",
    "owner_before_event",
    "candidate_before_event",
    "event_count_so_far",
)
STATE_FLOAT_COLUMNS = (
    "duration_so_far_seconds",
    "start_x",
    "start_y",
    "current_x",
    "current_y",
)
POSSESSION_STRING_COLUMNS = (
    "possession_uid",
    "period",
    "start_reason",
    "previous_possession_uid",
    "next_possession_uid_audit_only",
    "close_reason_audit_only",
)
POSSESSION_NULLABLE_INT_COLUMNS = (
    "close_evidence_event_uid_audit_only",
    "close_evidence_event_index_audit_only",
    "last_event_uid_audit_only",
    "last_event_index_audit_only",
)
POSSESSION_FLOAT_COLUMNS = (
    "start_seconds",
    "start_x",
    "start_y",
    "duration_seconds_audit_only",
    "last_x_audit_only",
    "last_y_audit_only",
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SubeventRule:
    event_id: int
    subevent_id: int | None
    signal: str
    post_effect: str
    note: str


@dataclass(frozen=True)
class TagRule:
    tag_id: int
    signal: str
    post_effect: str
    priority: int
    note: str


@dataclass(frozen=True)
class PossessionRules:
    events: dict[int, SubeventRule]
    subevents: dict[tuple[int, int], SubeventRule]
    tags: dict[int, TagRule]


def load_possession_rules(path: Path = DEFAULT_RULES_PATH) -> PossessionRules:
    events: dict[int, SubeventRule] = {}
    subevents: dict[tuple[int, int], SubeventRule] = {}
    tags: dict[int, TagRule] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = row["rule_kind"]
            signal = row["signal"]
            effect = row["post_effect"]
            if effect not in POST_EFFECTS:
                raise ValueError(f"Unknown possession post effect {effect!r}")
            if kind in {"event", "subevent"}:
                if signal not in BASE_SIGNALS:
                    raise ValueError(f"Unknown possession signal {signal!r}")
                event_id = int(row["event_id"])
                if kind == "event":
                    if event_id in events:
                        raise ValueError(f"Duplicate possession event rule {event_id}")
                    events[event_id] = SubeventRule(
                        event_id=event_id,
                        subevent_id=None,
                        signal=signal,
                        post_effect=effect,
                        note=row["note"],
                    )
                    continue
                key = (event_id, int(row["subevent_id"]))
                if key in subevents:
                    raise ValueError(f"Duplicate possession subevent rule {key}")
                subevents[key] = SubeventRule(
                    event_id=key[0],
                    subevent_id=key[1],
                    signal=signal,
                    post_effect=effect,
                    note=row["note"],
                )
            elif kind == "tag":
                tag_id = int(row["tag_id"])
                if tag_id in tags:
                    raise ValueError(f"Duplicate possession tag rule {tag_id}")
                tags[tag_id] = TagRule(
                    tag_id=tag_id,
                    signal=signal,
                    post_effect=effect,
                    priority=int(row["priority"]),
                    note=row["note"],
                )
            else:
                raise ValueError(f"Unknown possession rule kind {kind!r}")
    return PossessionRules(events=events, subevents=subevents, tags=tags)


@dataclass
class RuntimePossession:
    match_id: int
    possession_index: int
    owner_team_id: int
    period: str
    start_event_uid: int
    start_event_index: int
    start_seconds: float
    start_x: float | None
    start_y: float | None
    start_reason: str
    previous_possession_uid: str | None
    event_count: int = 0
    contested_event_count: int = 0
    low_confidence_event_count: int = 0
    last_event_uid: int | None = None
    last_event_index: int | None = None
    last_seconds: float | None = None
    current_x: float | None = None
    current_y: float | None = None
    is_closed: bool = False
    close_evidence_event_uid: int | None = None
    close_evidence_event_index: int | None = None
    close_reason: str | None = None
    next_possession_uid: str | None = None

    @property
    def uid(self) -> str:
        return f"{self.match_id}:{self.possession_index}"

    def add_event(
        self,
        event_uid: int,
        event_index: int,
        seconds: float,
        x: float | None,
        y: float | None,
        contested: bool,
        confidence: str,
    ) -> None:
        self.event_count += 1
        self.contested_event_count += int(contested)
        self.low_confidence_event_count += int(confidence == "low")
        self.last_event_uid = event_uid
        self.last_event_index = event_index
        self.last_seconds = seconds
        if x is not None and y is not None:
            self.current_x = x
            self.current_y = y


@dataclass
class Machine:
    match_id: int
    team_ids: tuple[int, ...]
    rules: PossessionRules
    state: str = "DEAD_BALL"
    active: RuntimePossession | None = None
    candidate_team_id: int | None = None
    possessions: list[RuntimePossession] = field(default_factory=list)
    last_closed: RuntimePossession | None = None

    def opponent(self, team_id: int) -> int | None:
        opponents = [value for value in self.team_ids if value != team_id]
        return opponents[0] if len(opponents) == 1 else None

    def start(self, event: pd.Series, reason: str) -> RuntimePossession:
        team_id = int(event.team_id)
        previous_uid = self.last_closed.uid if self.last_closed is not None else None
        possession = RuntimePossession(
            match_id=self.match_id,
            possession_index=len(self.possessions),
            owner_team_id=team_id,
            period=str(event.period),
            start_event_uid=int(event.event_uid),
            start_event_index=int(event.event_index),
            start_seconds=float(event.absolute_seconds),
            start_x=_nullable_float(event.start_x),
            start_y=_nullable_float(event.start_y),
            start_reason=reason,
            previous_possession_uid=previous_uid,
        )
        if self.last_closed is not None:
            self.last_closed.next_possession_uid = possession.uid
        self.possessions.append(possession)
        self.active = possession
        self.state = "CONTROL"
        self.candidate_team_id = None
        return possession

    def close(self, event: pd.Series, reason: str) -> str | None:
        if self.active is None:
            self.state = "DEAD_BALL"
            self.candidate_team_id = None
            return None
        closed = self.active
        closed.is_closed = True
        closed.close_evidence_event_uid = int(event.event_uid)
        closed.close_evidence_event_index = int(event.event_index)
        closed.close_reason = reason
        self.active = None
        self.last_closed = closed
        self.state = "DEAD_BALL"
        self.candidate_team_id = None
        return closed.uid


def _nullable_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _representative_position(event: pd.Series) -> tuple[float | None, float | None]:
    if bool(event.end_position_valid):
        return _nullable_float(event.end_x), _nullable_float(event.end_y)
    if bool(event.start_position_valid):
        return _nullable_float(event.start_x), _nullable_float(event.start_y)
    return None, None


def _effective_post_effect(
    base: SubeventRule, tag_ids: set[int], rules: PossessionRules
) -> tuple[str, list[str]]:
    matched = sorted(
        (rules.tags[tag_id] for tag_id in tag_ids if tag_id in rules.tags),
        key=lambda rule: (-rule.priority, rule.tag_id),
    )
    rule_level = "event" if base.subevent_id is None else "subevent"
    rule_id = base.event_id if base.subevent_id is None else f"{base.event_id}/{base.subevent_id}"
    evidence = [f"{rule_level}:{rule_id}:{base.signal}"]
    evidence.extend(f"tag:{rule.tag_id}:{rule.signal}" for rule in matched)
    close = [rule for rule in matched if rule.post_effect == "close"]
    if close:
        return "close", evidence
    contest = [rule for rule in matched if rule.post_effect == "contest"]
    if contest:
        return "contest", evidence
    return base.post_effect, evidence


def _candidate_after_contest(
    base: SubeventRule,
    event_team_id: int,
    tag_ids: set[int],
    machine: Machine,
) -> int | None:
    """Return an auditable candidate without treating it as confirmed control."""

    if tag_ids & TEAM_CANDIDATE_TAGS:
        return event_team_id
    if tag_ids & OPPONENT_CANDIDATE_TAGS:
        return machine.opponent(event_team_id)
    if base.signal == "contest" and base.event_id == 9:
        return event_team_id
    return None


def _event_tags(tags: pd.DataFrame) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for event_uid, group in tags.groupby("event_uid", sort=False):
        result[int(event_uid)] = set(int(value) for value in group["tag_id"])
    return result


def _coerce_state_schema(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("match_id", "event_uid", "event_index", "event_team_id"):
        frame[column] = frame[column].astype("int64")
    for column in STATE_NULLABLE_INT_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    for column in STATE_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in STATE_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in ("switch_confirmed", "is_closed_as_of_event"):
        frame[column] = frame[column].astype("bool")
    return frame


def _coerce_possession_schema(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in (
        "match_id",
        "possession_index",
        "owner_team_id",
        "start_event_uid",
        "start_event_index",
        "event_count_audit_only",
        "contested_event_count_audit_only",
        "low_confidence_event_count_audit_only",
    ):
        frame[column] = frame[column].astype("int64")
    for column in POSSESSION_NULLABLE_INT_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    for column in POSSESSION_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in POSSESSION_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    frame["is_closed_audit_only"] = frame["is_closed_audit_only"].astype("bool")
    return frame


def infer_match_possessions(
    events: pd.DataFrame,
    tags_by_event: dict[int, set[int]],
    team_ids: Iterable[int],
    rules: PossessionRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer one match left-to-right without finalizing an open last possession."""

    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    ordered = events.sort_values("event_index").reset_index(drop=True)
    match_id = int(ordered.iloc[0].match_id)
    machine = Machine(match_id, tuple(sorted(int(value) for value in team_ids)), rules)
    state_rows: list[dict[str, Any]] = []
    previous_period: str | None = None

    for _, event in ordered.iterrows():
        event_uid = int(event.event_uid)
        event_index = int(event.event_index)
        event_team = int(event.team_id)
        state_before = machine.state
        owner_before = machine.active.owner_team_id if machine.active else None
        candidate_before = machine.candidate_team_id
        closed_before: str | None = None
        closed_after: str | None = None
        closed_before_reason: str | None = None
        closed_after_reason: str | None = None

        period = str(event.period)
        if previous_period is not None and period != previous_period:
            closed_before_reason = "period_break"
            closed_before = machine.close(event, closed_before_reason)

        subevent_id = None if pd.isna(event.subevent_id) else int(event.subevent_id)
        key = (int(event.event_type_id), subevent_id)
        base = rules.subevents.get(key) if subevent_id is not None else None
        if base is None:
            base = rules.events.get(int(event.event_type_id))
        if base is None:
            raise ValueError(f"No possession rule for event/subevent {key}")
        tag_ids = tags_by_event.get(event_uid, set())
        post_effect, evidence_parts = _effective_post_effect(base, tag_ids, rules)
        evidence = "|".join(evidence_parts)
        role = "unassigned"
        confidence = "high"
        event_possession: RuntimePossession | None = None

        if base.signal == "boundary":
            if machine.active is not None:
                event_possession = machine.active
                role = "boundary"
                confidence = "high"
                x, y = _representative_position(event)
                event_possession.add_event(
                    event_uid, event_index, float(event.absolute_seconds), x, y, False, confidence
                )
            closed_after_reason = f"boundary:{event.subevent_name or event.event_name}"
            closed_after = machine.close(event, closed_after_reason)
        elif base.signal == "restart":
            if machine.active is not None:
                closed_before_reason = f"restart:{event.subevent_name or event.event_name}"
                closed_before = machine.close(event, closed_before_reason)
            event_possession = machine.start(event, f"restart:{event.subevent_name}")
            role = "restart"
        elif base.signal == "control":
            if machine.active is None:
                event_possession = machine.start(event, f"control:{event.subevent_name}")
            elif machine.active.owner_team_id != event_team:
                closed_before_reason = "confirmed_opponent_control"
                closed_before = machine.close(event, closed_before_reason)
                event_possession = machine.start(event, f"control:{event.subevent_name}")
            else:
                event_possession = machine.active
                machine.state = "CONTROL"
                machine.candidate_team_id = None
            role = "owner_action"
        elif base.signal == "contest":
            confidence = "medium" if machine.active is not None else "low"
            machine.state = "CONTESTED"
            machine.candidate_team_id = _candidate_after_contest(
                base, event_team, tag_ids, machine
            )
            if machine.active is not None:
                event_possession = machine.active
                role = (
                    "owner_contest"
                    if machine.active.owner_team_id == event_team
                    else "opponent_contest"
                )
            else:
                role = "unassigned_contest"
        else:
            confidence = "low"
            role = "neutral"

        if event_possession is not None and base.signal != "boundary":
            x, y = _representative_position(event)
            event_possession.add_event(
                event_uid,
                event_index,
                float(event.absolute_seconds),
                x,
                y,
                base.signal == "contest" or post_effect == "contest",
                confidence,
            )

        if event_possession is not None and post_effect == "close" and closed_after is None:
            closed_after_reason = "tag_confirmed_close"
            closed_after = machine.close(event, closed_after_reason)
        elif event_possession is not None and post_effect == "contest":
            machine.state = "CONTESTED"
            machine.candidate_team_id = _candidate_after_contest(
                base, event_team, tag_ids, machine
            )
            if confidence == "high":
                confidence = "medium"
        elif event_possession is not None and machine.active is not None:
            machine.state = "CONTROL"
            machine.candidate_team_id = None

        possession_uid = event_possession.uid if event_possession else None
        duration_so_far = (
            float(event.absolute_seconds) - event_possession.start_seconds
            if event_possession is not None
            else None
        )
        state_rows.append(
            {
                "match_id": match_id,
                "event_uid": event_uid,
                "event_index": event_index,
                "period": period,
                "event_team_id": event_team,
                "possession_uid": possession_uid,
                "possession_index": (
                    event_possession.possession_index if event_possession else None
                ),
                "owner_team_id": (
                    event_possession.owner_team_id if event_possession else None
                ),
                "candidate_team_id": machine.candidate_team_id,
                "event_role": role,
                "state_before": state_before,
                "state_after": machine.state,
                "owner_before_event": owner_before,
                "candidate_before_event": candidate_before,
                "active_possession_after_event": (
                    machine.active.uid if machine.active is not None else None
                ),
                "closed_possession_before_event": closed_before,
                "closed_possession_after_event": closed_after,
                "switch_confirmed": closed_before_reason == "confirmed_opponent_control",
                "boundary_reason": closed_after_reason or closed_before_reason,
                "evidence": evidence,
                "confidence": confidence,
                "duration_so_far_seconds": duration_so_far,
                "event_count_so_far": (
                    event_possession.event_count if event_possession else None
                ),
                "start_x": event_possession.start_x if event_possession else None,
                "start_y": event_possession.start_y if event_possession else None,
                "current_x": event_possession.current_x if event_possession else None,
                "current_y": event_possession.current_y if event_possession else None,
                "is_closed_as_of_event": (
                    event_possession.is_closed if event_possession else False
                ),
            }
        )
        previous_period = period

    possession_rows = [
        {
            "match_id": possession.match_id,
            "possession_uid": possession.uid,
            "possession_index": possession.possession_index,
            "owner_team_id": possession.owner_team_id,
            "period": possession.period,
            "start_event_uid": possession.start_event_uid,
            "start_event_index": possession.start_event_index,
            "start_seconds": possession.start_seconds,
            "start_x": possession.start_x,
            "start_y": possession.start_y,
            "start_reason": possession.start_reason,
            "previous_possession_uid": possession.previous_possession_uid,
            "next_possession_uid_audit_only": possession.next_possession_uid,
            "is_closed_audit_only": possession.is_closed,
            "close_evidence_event_uid_audit_only": possession.close_evidence_event_uid,
            "close_evidence_event_index_audit_only": possession.close_evidence_event_index,
            "close_reason_audit_only": possession.close_reason,
            "last_event_uid_audit_only": possession.last_event_uid,
            "last_event_index_audit_only": possession.last_event_index,
            "duration_seconds_audit_only": (
                None
                if possession.last_seconds is None
                else possession.last_seconds - possession.start_seconds
            ),
            "event_count_audit_only": possession.event_count,
            "contested_event_count_audit_only": possession.contested_event_count,
            "low_confidence_event_count_audit_only": possession.low_confidence_event_count,
            "last_x_audit_only": possession.current_x,
            "last_y_audit_only": possession.current_y,
        }
        for possession in machine.possessions
    ]
    return (
        _coerce_state_schema(pd.DataFrame(state_rows)),
        _coerce_possession_schema(pd.DataFrame(possession_rows)),
    )


def _validate_rule_coverage(event_root: Path, rules: PossessionRules) -> list[str]:
    errors: list[str] = []
    events = pd.read_parquet(
        event_root / "events.parquet", columns=["event_type_id", "subevent_id"]
    )
    observed = {
        (int(row.event_type_id), int(row.subevent_id))
        for row in events.itertuples()
        if not pd.isna(row.subevent_id)
    }
    missing = observed - set(rules.subevents)
    uncovered = {value for value in missing if value[0] not in rules.events}
    if uncovered:
        errors.append(f"missing subevent rules: {sorted(uncovered)}")
    return errors


def infer_possessions(
    event_root: Path = EVENT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_POSSESSION_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    competition: str = "England",
    rules_path: Path = DEFAULT_RULES_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Infer possession states for one normalized competition."""

    event_competition_root = Path(event_root).resolve() / competition
    competition_root = Path(output_root).resolve() / competition
    manifest_path = competition_root / "metadata/manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in ("events.parquet", "event_tags.parquet", "match_teams.parquet"):
        if not (event_competition_root / filename).exists():
            raise FileNotFoundError(event_competition_root / filename)

    rules = load_possession_rules(rules_path)
    coverage_errors = _validate_rule_coverage(event_competition_root, rules)
    if coverage_errors:
        raise ValueError("Invalid possession rules: " + "; ".join(coverage_errors))
    events = pd.read_parquet(event_competition_root / "events.parquet")
    tags = pd.read_parquet(event_competition_root / "event_tags.parquet")
    match_teams = pd.read_parquet(event_competition_root / "match_teams.parquet")
    tags_by_event = _event_tags(tags)
    teams_by_match = {
        int(match_id): tuple(int(value) for value in group["team_id"])
        for match_id, group in match_teams.groupby("match_id", sort=False)
    }

    state_frames: list[pd.DataFrame] = []
    possession_frames: list[pd.DataFrame] = []
    for match_id, match_events in events.groupby("match_id", sort=True):
        states, possessions = infer_match_possessions(
            match_events,
            tags_by_event,
            teams_by_match[int(match_id)],
            rules,
        )
        state_frames.append(states)
        possession_frames.append(possessions)
    states = pd.concat(state_frames, ignore_index=True)
    possessions = pd.concat(possession_frames, ignore_index=True)

    _write_parquet(states, competition_root / "event_possession_states.parquet")
    _write_parquet(possessions, competition_root / "possessions.parquet")
    competition_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rules_path, competition_root / "possession_rules.csv")
    validation = validate_possessions(
        events,
        tags,
        match_teams,
        states,
        possessions,
        rules,
        prefix_checks=100,
    )
    _write_json(validation, competition_root / "metadata/validation_report.json")
    if not validation["valid"]:
        raise ValueError("Invalid inferred possessions: " + "; ".join(validation["errors"]))

    analysis_root = competition_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    for field, filename in (
        ("event_role", "event_role_counts.csv"),
        ("state_after", "state_counts.csv"),
        ("confidence", "confidence_counts.csv"),
        ("boundary_reason", "boundary_reason_counts.csv"),
    ):
        (
            states.groupby(field, dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
            .to_csv(analysis_root / filename, index=False)
        )
    possession_distribution = possessions.groupby("match_id").size()
    pd.DataFrame(
        {
            "metric": [
                "possessions_per_match_mean",
                "possessions_per_match_median",
                "events_per_possession_mean",
                "duration_seconds_mean",
                "unassigned_event_fraction",
                "contested_event_fraction",
                "low_confidence_event_fraction",
            ],
            "value": [
                float(possession_distribution.mean()),
                float(possession_distribution.median()),
                float(possessions["event_count_audit_only"].mean()),
                float(possessions["duration_seconds_audit_only"].mean()),
                float(states["possession_uid"].isna().mean()),
                float((states["state_after"] == "CONTESTED").mean()),
                float((states["confidence"] == "low").mean()),
            ],
        }
    ).to_csv(analysis_root / "summary_statistics.csv", index=False)
    state_by_match = states.groupby("match_id", sort=True).agg(
        event_count=("event_uid", "size"),
        assigned_event_count=("possession_uid", "count"),
        contested_event_count=(
            "state_after",
            lambda values: int((values == "CONTESTED").sum()),
        ),
        low_confidence_event_count=(
            "confidence",
            lambda values: int((values == "low").sum()),
        ),
        confirmed_switch_count=("switch_confirmed", "sum"),
    )
    possession_by_match = possessions.groupby("match_id", sort=True).agg(
        possession_count=("possession_uid", "size"),
        mean_events_per_possession=("event_count_audit_only", "mean"),
        median_events_per_possession=("event_count_audit_only", "median"),
        mean_duration_seconds=("duration_seconds_audit_only", "mean"),
        median_duration_seconds=("duration_seconds_audit_only", "median"),
    )
    per_match = state_by_match.join(possession_by_match, how="outer").reset_index()
    per_match["unassigned_event_count"] = (
        per_match["event_count"] - per_match["assigned_event_count"]
    )
    for numerator in (
        "unassigned_event_count",
        "contested_event_count",
        "low_confidence_event_count",
    ):
        per_match[numerator.replace("_count", "_fraction")] = (
            per_match[numerator] / per_match["event_count"]
        )
    per_match.to_csv(analysis_root / "per_match_statistics.csv", index=False)

    distribution_rows: list[dict[str, Any]] = []
    for metric, column in (
        ("events_per_possession", "event_count_audit_only"),
        ("duration_seconds", "duration_seconds_audit_only"),
    ):
        values = possessions[column].dropna()
        for quantile in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
            distribution_rows.append(
                {
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(values.quantile(quantile)),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(
        analysis_root / "possession_distributions.csv", index=False
    )
    _write_audit_sample(states, events, analysis_root / "manual_audit_sample.csv")

    schema = {
        "schema_version": POSSESSION_SCHEMA_VERSION,
        "competition": competition,
        "causal_contract": {
            "streaming_only": True,
            "future_revision": False,
            "open_final_possession_is_not_force_closed": True,
            "audit_only_suffix_forbidden_in_model_inputs": True,
        },
        "tables": {
            "event_possession_states": list(states.columns),
            "possessions": list(possessions.columns),
        },
        "states": sorted(STATES),
    }
    _write_json(schema, competition_root / "metadata/schema.json")
    manifest = {
        "schema_version": POSSESSION_SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": competition,
        "event_source_root": str(event_competition_root),
        "output_root": str(competition_root),
        "rules_path": str(Path(rules_path).resolve()),
        "rules_sha256": _sha256(Path(rules_path)),
        "totals": {
            "events": len(states),
            "possessions": len(possessions),
            "matches": int(states["match_id"].nunique()),
            "assigned_events": int(states["possession_uid"].notna().sum()),
            "unassigned_events": int(states["possession_uid"].isna().sum()),
        },
        "validation": {
            "valid": validation["valid"],
            "prefix_checks": validation["prefix_checks"],
        },
        "output_files": [
            {
                "path": filename,
                "size_bytes": (competition_root / filename).stat().st_size,
                "sha256": _sha256(competition_root / filename),
            }
            for filename in (
                "event_possession_states.parquet",
                "possessions.parquet",
                "possession_rules.csv",
            )
        ],
    }
    _write_json(manifest, manifest_path)
    return manifest


def _write_audit_sample(states: pd.DataFrame, events: pd.DataFrame, path: Path) -> None:
    joined = states.merge(
        events[
            [
                "match_id",
                "event_uid",
                "event_name",
                "subevent_name",
                "event_seconds",
                "team_id",
            ]
        ],
        on=["match_id", "event_uid"],
        how="left",
        validate="one_to_one",
    )
    candidates = joined[
        joined["switch_confirmed"]
        | joined["boundary_reason"].notna()
        | (joined["confidence"] == "low")
    ].copy()
    if candidates.empty:
        candidates = joined.head(0)
    else:
        candidates["audit_stratum"] = candidates["boundary_reason"].fillna(
            candidates["confidence"]
        )
        candidates = (
            candidates.sort_values(["audit_stratum", "match_id", "event_index"])
            .groupby("audit_stratum", group_keys=False, dropna=False)
            .head(25)
        )
    candidates.to_csv(path, index=False)


def validate_possessions(
    events: pd.DataFrame,
    tags: pd.DataFrame,
    match_teams: pd.DataFrame,
    states: pd.DataFrame,
    possessions: pd.DataFrame,
    rules: PossessionRules,
    prefix_checks: int = 100,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(states) != len(events):
        errors.append("state row count differs from event row count")
    event_keys = set(
        zip(events["match_id"].astype(int), events["event_uid"].astype(int), strict=True)
    )
    state_keys = set(
        zip(states["match_id"].astype(int), states["event_uid"].astype(int), strict=True)
    )
    if event_keys != state_keys:
        errors.append("event state keys differ from normalized event keys")
    if states[["match_id", "event_uid"]].duplicated().any():
        errors.append("event state mapping is not one-to-one")
    if possessions["possession_uid"].isna().any():
        errors.append("possession_uid contains null values")
    if possessions["possession_uid"].duplicated().any():
        errors.append("possession_uid is not unique")
    if possessions[["match_id", "possession_index"]].duplicated().any():
        errors.append("possession index is not unique within a match")
    if not set(states["state_before"]).issubset(STATES):
        errors.append("unknown state_before value")
    if not set(states["state_after"]).issubset(STATES):
        errors.append("unknown state_after value")
    assigned = states[states["possession_uid"].notna()]
    possession_ids = set(possessions["possession_uid"].astype(str))
    assigned_ids = set(assigned["possession_uid"].astype(str))
    if not assigned_ids.issubset(possession_ids):
        errors.append("event state references an unknown possession")
    active_ids = set(states["active_possession_after_event"].dropna().astype(str))
    closed_ids = set(
        pd.concat(
            [
                states["closed_possession_before_event"],
                states["closed_possession_after_event"],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
    )
    if not (active_ids | closed_ids).issubset(possession_ids):
        errors.append("state transition references an unknown possession")
    owner_counts = possessions.groupby("possession_uid")["owner_team_id"].nunique()
    if (owner_counts != 1).any():
        errors.append("a possession has multiple owners")
    possession_periods = assigned.groupby("possession_uid")["period"].nunique()
    if (possession_periods > 1).any():
        errors.append("a possession crosses periods")
    if not assigned.empty:
        assigned_contract = assigned.merge(
            possessions[
                [
                    "possession_uid",
                    "match_id",
                    "owner_team_id",
                    "period",
                    "start_seconds",
                ]
            ],
            on="possession_uid",
            how="left",
            suffixes=("_state", "_possession"),
            validate="many_to_one",
        )
        if (
            assigned_contract["match_id_state"].astype(int)
            != assigned_contract["match_id_possession"].astype(int)
        ).any():
            errors.append("event and possession match IDs disagree")
        if (
            assigned_contract["owner_team_id_state"].astype(int)
            != assigned_contract["owner_team_id_possession"].astype(int)
        ).any():
            errors.append("event and possession owners disagree")
        if (
            assigned_contract["period_state"].astype(str)
            != assigned_contract["period_possession"].astype(str)
        ).any():
            errors.append("event and possession periods disagree")

        causal = assigned_contract.merge(
            events[["match_id", "event_uid", "absolute_seconds"]],
            left_on=["match_id_state", "event_uid"],
            right_on=["match_id", "event_uid"],
            how="left",
            validate="one_to_one",
        )
        expected_duration = causal["absolute_seconds"] - causal["start_seconds"]
        if (expected_duration < -1e-9).any() or not (
            (causal["duration_so_far_seconds"] - expected_duration).abs() <= 1e-7
        ).all():
            errors.append("duration-so-far is not a causal elapsed-time snapshot")
        expected_counts = (
            causal.sort_values(["match_id_state", "event_index"])
            .groupby("possession_uid")
            .cumcount()
            + 1
        )
        actual_counts = causal.sort_values(
            ["match_id_state", "event_index"]
        )["event_count_so_far"].astype(int)
        if not actual_counts.reset_index(drop=True).equals(
            expected_counts.reset_index(drop=True).astype(int)
        ):
            errors.append("event-count-so-far is not a causal prefix count")
    if states["evidence"].isna().any() or (states["evidence"] == "").any():
        errors.append("state transition without evidence")
    known_teams = {
        int(match_id): set(int(value) for value in group["team_id"])
        for match_id, group in match_teams.groupby("match_id")
    }
    for row in possessions[["match_id", "owner_team_id"]].itertuples(index=False):
        if int(row.owner_team_id) not in known_teams[int(row.match_id)]:
            errors.append(f"possession owner outside match teams for match {row.match_id}")
            break
    for row in states[
        ["match_id", "event_team_id", "candidate_team_id", "owner_before_event"]
    ].itertuples(index=False):
        valid = known_teams[int(row.match_id)]
        if int(row.event_team_id) not in valid:
            errors.append(f"event team outside match teams for match {row.match_id}")
            break
        for field in (row.candidate_team_id, row.owner_before_event):
            if not pd.isna(field) and int(field) not in valid:
                errors.append(f"state team reference outside match teams for match {row.match_id}")
                break
        if errors and errors[-1].startswith("state team reference"):
            break
    duel_switches = states[
        states["switch_confirmed"] & states["evidence"].str.contains("subevent:1/", regex=False)
    ]
    if len(duel_switches):
        errors.append("duel confirmed a possession switch")
    first_evidence = states["evidence"].str.split("|", regex=False).str[0]
    non_control_switches = states[
        states["switch_confirmed"] & ~first_evidence.str.endswith(":control")
    ]
    if len(non_control_switches):
        errors.append("a non-control event confirmed a possession switch")
    owner_actions = assigned[assigned["event_role"] == "owner_action"]
    if (
        owner_actions["owner_team_id"].astype(int)
        != owner_actions["event_team_id"].astype(int)
    ).any():
        errors.append("a control event did not synchronize the possession owner")
    restarts = assigned[assigned["event_role"] == "restart"]
    if (
        restarts["owner_team_id"].astype(int)
        != restarts["event_team_id"].astype(int)
    ).any():
        errors.append("a restart did not synchronize the possession owner")
    model_safe_columns = set(states.columns)
    forbidden = {
        "final_duration",
        "final_event_count",
        "final_position",
        "next_owner_team_id",
    }
    if model_safe_columns & forbidden or any("audit_only" in value for value in model_safe_columns):
        errors.append("model-safe event state table contains future or audit-only fields")

    previous_links = possessions[
        possessions["previous_possession_uid"].notna()
    ][["possession_uid", "previous_possession_uid"]]
    next_by_uid = possessions.set_index("possession_uid")[
        "next_possession_uid_audit_only"
    ].to_dict()
    if any(
        next_by_uid.get(row.previous_possession_uid) != row.possession_uid
        for row in previous_links.itertuples(index=False)
    ):
        errors.append("previous/next possession links are inconsistent")

    prefix_errors = _prefix_equivalence_checks(
        events, tags, match_teams, rules, prefix_checks
    )
    errors.extend(prefix_errors)
    return {
        "valid": not errors,
        "errors": errors,
        "events": len(events),
        "state_rows": len(states),
        "possessions": len(possessions),
        "matches": int(events["match_id"].nunique()),
        "assigned_events": int(states["possession_uid"].notna().sum()),
        "unassigned_events": int(states["possession_uid"].isna().sum()),
        "contested_events": int((states["state_after"] == "CONTESTED").sum()),
        "low_confidence_events": int((states["confidence"] == "low").sum()),
        "unassigned_event_fraction": float(states["possession_uid"].isna().mean()),
        "contested_event_fraction": float(
            (states["state_after"] == "CONTESTED").mean()
        ),
        "low_confidence_event_fraction": float(
            (states["confidence"] == "low").mean()
        ),
        "confirmed_switches": int(states["switch_confirmed"].sum()),
        "duel_confirmed_switches": len(duel_switches),
        "non_control_confirmed_switches": len(non_control_switches),
        "prefix_checks": prefix_checks,
    }


def _prefix_equivalence_checks(
    events: pd.DataFrame,
    tags: pd.DataFrame,
    match_teams: pd.DataFrame,
    rules: PossessionRules,
    checks: int,
) -> list[str]:
    if checks <= 0:
        return []
    rng = random.Random(20260715)
    matches = sorted(int(value) for value in events["match_id"].unique())
    tags_by_event = _event_tags(tags)
    teams_by_match = {
        int(match_id): tuple(int(value) for value in group["team_id"])
        for match_id, group in match_teams.groupby("match_id")
    }
    errors: list[str] = []
    for _ in range(checks):
        match_id = rng.choice(matches)
        match_events = events[events["match_id"] == match_id].sort_values("event_index")
        cutoff = rng.randrange(1, len(match_events) + 1)
        full_states, _ = infer_match_possessions(
            match_events, tags_by_event, teams_by_match[match_id], rules
        )
        prefix_states, _ = infer_match_possessions(
            match_events.iloc[:cutoff], tags_by_event, teams_by_match[match_id], rules
        )
        columns = list(prefix_states.columns)
        if not prefix_states.reset_index(drop=True).equals(
            full_states.iloc[:cutoff][columns].reset_index(drop=True)
        ):
            errors.append(f"prefix equivalence failed for match {match_id} cutoff {cutoff}")
            break
    return errors
