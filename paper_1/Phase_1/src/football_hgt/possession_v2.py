"""Graph-ready V2 views over the deterministic V1 possession segmentation."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .event_tables import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT as EVENT_OUTPUT_ROOT
from .possession import (
    PossessionRules,
    SubeventRule,
    _event_tags,
    _sha256,
    _write_json,
    _write_parquet,
    infer_match_possessions,
    load_possession_rules,
)


POSSESSION_SCHEMA_VERSION_V2 = "2.0.0"
DEFAULT_POSSESSION_ROOT_V2 = DEFAULT_DATA_ROOT / "processed/inferred_possessions/v2"
DEFAULT_POSSESSION_ROOT_V1 = DEFAULT_DATA_ROOT / "processed/inferred_possessions/v1"
DEFAULT_RULES_PATH_V2 = DEFAULT_DATA_ROOT / "analysis/possession_rules_v2.csv"
DEFAULT_CANDIDATE_RULES_PATH_V2 = (
    DEFAULT_DATA_ROOT / "analysis/possession_candidate_rules_v2.csv"
)

CANDIDATE_STATUSES = frozenset({"none", "single", "conflicting"})
CANDIDATE_DIRECTIONS = frozenset({"event_team", "opponent"})
EVENT_ROLES = frozenset({"control", "restart", "contest", "boundary", "neutral"})
ACTOR_RELATIONS = frozenset({"owner", "opponent", "unknown"})
EXPECTED_ENGLAND_TOTALS = {
    "matches": 380,
    "events": 643_150,
    "possessions": 117_607,
    "confirmed_switches": 79_327,
    "assigned_events": 641_040,
    "unassigned_events": 2_110,
}


@dataclass(frozen=True)
class CandidateRuleV2:
    event_id: int
    subevent_id: int | None
    tag_id: int | None
    direction: str
    rule_scope: str
    note: str

    @property
    def rule_id(self) -> str:
        subevent = "*" if self.subevent_id is None else str(self.subevent_id)
        tag = "*" if self.tag_id is None else str(self.tag_id)
        return f"candidate:{self.event_id}/{subevent}/{tag}:{self.direction}"


@dataclass(frozen=True)
class PossessionRulesV2:
    base: PossessionRules
    candidates: tuple[CandidateRuleV2, ...]


@dataclass(frozen=True)
class PossessionInferenceV2:
    states: pd.DataFrame
    evidence: pd.DataFrame
    possessions: pd.DataFrame
    transitions: pd.DataFrame
    audit: pd.DataFrame


def load_possession_rules_v2(
    rules_path: Path = DEFAULT_RULES_PATH_V2,
    candidate_rules_path: Path = DEFAULT_CANDIDATE_RULES_PATH_V2,
) -> PossessionRulesV2:
    """Load the unchanged V1 segmentation rules and contextual V2 candidates."""

    candidates: list[CandidateRuleV2] = []
    seen: set[tuple[int, int | None, int | None, str]] = set()
    with Path(candidate_rules_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            direction = row["direction"]
            scope = row["rule_scope"]
            if direction not in CANDIDATE_DIRECTIONS:
                raise ValueError(f"Unknown candidate direction {direction!r}")
            if scope not in {"tag", "fallback"}:
                raise ValueError(f"Unknown candidate rule scope {scope!r}")
            subevent_id = int(row["subevent_id"]) if row["subevent_id"] else None
            tag_id = int(row["tag_id"]) if row["tag_id"] else None
            if scope == "tag" and tag_id is None:
                raise ValueError("A tag candidate rule requires tag_id")
            if scope == "fallback" and tag_id is not None:
                raise ValueError("A fallback candidate rule cannot have tag_id")
            key = (int(row["event_id"]), subevent_id, tag_id, scope)
            if key in seen:
                raise ValueError(f"Duplicate V2 candidate rule {key}")
            seen.add(key)
            candidates.append(
                CandidateRuleV2(
                    event_id=key[0],
                    subevent_id=subevent_id,
                    tag_id=tag_id,
                    direction=direction,
                    rule_scope=scope,
                    note=row["note"],
                )
            )
    return PossessionRulesV2(
        base=load_possession_rules(Path(rules_path)),
        candidates=tuple(candidates),
    )


def _base_rule(event: pd.Series, rules: PossessionRules) -> SubeventRule:
    subevent_id = None if pd.isna(event.subevent_id) else int(event.subevent_id)
    base = (
        rules.subevents.get((int(event.event_type_id), subevent_id))
        if subevent_id is not None
        else None
    )
    if base is None:
        base = rules.events.get(int(event.event_type_id))
    if base is None:
        raise ValueError(
            f"No V2 possession rule for event/subevent "
            f"{(int(event.event_type_id), subevent_id)}"
        )
    return base


def _matched_tag_effects(
    tag_ids: set[int], rules: PossessionRules
) -> list[Any]:
    return sorted(
        (rules.tags[tag_id] for tag_id in tag_ids if tag_id in rules.tags),
        key=lambda rule: (-rule.priority, rule.tag_id),
    )


def _effective_post_effect(base: SubeventRule, tag_effects: list[Any]) -> str:
    if any(rule.post_effect == "close" for rule in tag_effects):
        return "close"
    if any(rule.post_effect == "contest" for rule in tag_effects):
        return "contest"
    return base.post_effect


def _matching_candidate_rules(
    event: pd.Series,
    tag_ids: set[int],
    rules: PossessionRulesV2,
) -> list[CandidateRuleV2]:
    event_id = int(event.event_type_id)
    subevent_id = None if pd.isna(event.subevent_id) else int(event.subevent_id)
    contextual = [
        rule
        for rule in rules.candidates
        if rule.event_id == event_id
        and (rule.subevent_id is None or rule.subevent_id == subevent_id)
    ]
    tagged = [
        rule
        for rule in contextual
        if rule.rule_scope == "tag" and rule.tag_id in tag_ids
    ]
    if tagged:
        return tagged
    return [rule for rule in contextual if rule.rule_scope == "fallback"]


def _resolve_candidate(
    event: pd.Series,
    team_ids: tuple[int, ...],
    matched: list[CandidateRuleV2],
) -> tuple[int | None, str]:
    directions = {rule.direction for rule in matched}
    if not directions:
        return None, "none"
    if len(directions) > 1:
        return None, "conflicting"
    direction = next(iter(directions))
    event_team = int(event.team_id)
    if direction == "event_team":
        return event_team, "single"
    opponents = [team_id for team_id in team_ids if team_id != event_team]
    if len(opponents) != 1:
        return None, "conflicting"
    return opponents[0], "single"


def _coerce_nullable_int(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = frame[column].astype("Int64")


def _build_possession_tables(
    v1_possessions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_columns = [
        "match_id",
        "possession_uid",
        "possession_index",
        "owner_team_id",
        "period",
        "start_event_uid",
        "start_event_index",
        "start_seconds",
        "start_x",
        "start_y",
        "start_reason",
        "previous_possession_uid",
    ]
    possessions = v1_possessions[core_columns].copy()

    audit_columns = [
        "match_id",
        "possession_uid",
        "possession_index",
        "next_possession_uid_audit_only",
        "is_closed_audit_only",
        "close_evidence_event_uid_audit_only",
        "close_evidence_event_index_audit_only",
        "close_reason_audit_only",
        "last_event_uid_audit_only",
        "last_event_index_audit_only",
        "duration_seconds_audit_only",
        "event_count_audit_only",
        "contested_event_count_audit_only",
        "low_confidence_event_count_audit_only",
        "last_x_audit_only",
        "last_y_audit_only",
    ]
    audit = v1_possessions[audit_columns].copy().rename(
        columns={column: column.removesuffix("_audit_only") for column in audit_columns}
    )

    by_uid = v1_possessions.set_index("possession_uid")
    transition_rows: list[dict[str, Any]] = []
    for current in v1_possessions.itertuples(index=False):
        previous_uid = current.previous_possession_uid
        if pd.isna(previous_uid):
            continue
        previous = by_uid.loc[str(previous_uid)]
        transition_rows.append(
            {
                "match_id": int(current.match_id),
                "period": str(current.period),
                "previous_possession_uid": str(previous_uid),
                "next_possession_uid": str(current.possession_uid),
                "previous_owner_team_id": int(previous.owner_team_id),
                "next_owner_team_id": int(current.owner_team_id),
                "transition_event_uid": int(current.start_event_uid),
                "transition_event_index": int(current.start_event_index),
                "transition_reason": str(previous.close_reason_audit_only)
                if not pd.isna(previous.close_reason_audit_only)
                else str(current.start_reason),
            }
        )
    transitions = pd.DataFrame(transition_rows)
    if not transitions.empty:
        transitions = transitions.astype(
            {
                "match_id": "int64",
                "previous_owner_team_id": "int64",
                "next_owner_team_id": "int64",
                "transition_event_uid": "int64",
                "transition_event_index": "int64",
            }
        )
        for column in (
            "period",
            "previous_possession_uid",
            "next_possession_uid",
            "transition_reason",
        ):
            transitions[column] = transitions[column].astype("string")
    return possessions, transitions, audit


def infer_match_possessions_v2(
    events: pd.DataFrame,
    tags_by_event: dict[int, set[int]],
    team_ids: Iterable[int],
    rules: PossessionRulesV2,
) -> PossessionInferenceV2:
    """Build V2 causal views without changing the V1 segmentation."""

    team_tuple = tuple(sorted(int(team_id) for team_id in team_ids))
    ordered = events.sort_values("event_index").reset_index(drop=True)
    v1_states, v1_possessions = infer_match_possessions(
        ordered, tags_by_event, team_tuple, rules.base
    )
    v1_states = v1_states.sort_values("event_index").reset_index(drop=True)
    if ordered["event_uid"].astype(int).tolist() != v1_states["event_uid"].astype(int).tolist():
        raise ValueError("V1 state output is not aligned with normalized events")

    state_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    candidate_team: int | None = None
    candidate_status = "none"
    previous_period: str | None = None

    for event, v1 in zip(ordered.itertuples(index=False), v1_states.itertuples(index=False), strict=True):
        event_series = pd.Series(event._asdict())
        period = str(event.period)
        period_changed = previous_period is not None and period != previous_period
        if period_changed:
            state_before = "DEAD_BALL"
            owner_before = None
            candidate_team = None
            candidate_status = "none"
        else:
            state_before = str(v1.state_before)
            owner_before = None if pd.isna(v1.owner_before_event) else int(v1.owner_before_event)

        candidate_before = candidate_team
        candidate_status_before = candidate_status
        base = _base_rule(event_series, rules.base)
        tag_ids = tags_by_event.get(int(event.event_uid), set())
        tag_effects = _matched_tag_effects(tag_ids, rules.base)
        post_effect = _effective_post_effect(base, tag_effects)
        candidate_rules = _matching_candidate_rules(event_series, tag_ids, rules)

        candidate_triggered = base.signal == "contest" or (
            base.signal in {"control", "restart"} and post_effect == "contest"
        )
        if str(v1.state_after) == "DEAD_BALL":
            candidate_team, candidate_status = None, "none"
        elif candidate_triggered:
            candidate_team, candidate_status = _resolve_candidate(
                event_series, team_tuple, candidate_rules
            )
        elif base.signal != "neutral":
            candidate_team, candidate_status = None, "none"

        possession_owner = None if pd.isna(v1.owner_team_id) else int(v1.owner_team_id)
        if possession_owner is None:
            actor_relation = "unknown"
        elif int(event.team_id) == possession_owner:
            actor_relation = "owner"
        else:
            actor_relation = "opponent"

        state_rows.append(
            {
                "match_id": int(event.match_id),
                "event_uid": int(event.event_uid),
                "event_index": int(event.event_index),
                "period": period,
                "event_team_id": int(event.team_id),
                "possession_uid": None if pd.isna(v1.possession_uid) else str(v1.possession_uid),
                "possession_index": None if pd.isna(v1.possession_index) else int(v1.possession_index),
                "possession_owner_team_id": possession_owner,
                "event_role": base.signal,
                "actor_relation_to_owner": actor_relation,
                "control_state_before": state_before,
                "control_state_after": str(v1.state_after),
                "owner_team_before_event": owner_before,
                "candidate_team_before_event": candidate_before,
                "candidate_status_before_event": candidate_status_before,
                "candidate_team_after_event": candidate_team,
                "candidate_status_after_event": candidate_status,
                "active_possession_after_event": None
                if pd.isna(v1.active_possession_after_event)
                else str(v1.active_possession_after_event),
                "closed_possession_before_event": None
                if pd.isna(v1.closed_possession_before_event)
                else str(v1.closed_possession_before_event),
                "closed_possession_after_event": None
                if pd.isna(v1.closed_possession_after_event)
                else str(v1.closed_possession_after_event),
                "switch_confirmed": bool(v1.switch_confirmed),
                "boundary_reason": None
                if pd.isna(v1.boundary_reason)
                else str(v1.boundary_reason),
                "duration_so_far_seconds": None
                if pd.isna(v1.duration_so_far_seconds)
                else float(v1.duration_so_far_seconds),
                "event_count_so_far": None
                if pd.isna(v1.event_count_so_far)
                else int(v1.event_count_so_far),
                "start_x": None if pd.isna(v1.start_x) else float(v1.start_x),
                "start_y": None if pd.isna(v1.start_y) else float(v1.start_y),
                "current_x": None if pd.isna(v1.current_x) else float(v1.current_x),
                "current_y": None if pd.isna(v1.current_y) else float(v1.current_y),
                "is_closed_as_of_event": bool(v1.is_closed_as_of_event),
            }
        )

        evidence_index = 0
        evidence_rows.append(
            {
                "match_id": int(event.match_id),
                "event_uid": int(event.event_uid),
                "event_index": int(event.event_index),
                "evidence_index": evidence_index,
                "evidence_kind": "base",
                "rule_id": f"base:{base.event_id}/{base.subevent_id or '*'}",
                "tag_id": None,
                "signal": base.signal,
                "post_effect": base.post_effect,
                "candidate_direction": None,
            }
        )
        evidence_index += 1
        for effect in tag_effects:
            evidence_rows.append(
                {
                    "match_id": int(event.match_id),
                    "event_uid": int(event.event_uid),
                    "event_index": int(event.event_index),
                    "evidence_index": evidence_index,
                    "evidence_kind": "tag_effect",
                    "rule_id": f"tag_effect:{effect.tag_id}",
                    "tag_id": effect.tag_id,
                    "signal": effect.signal,
                    "post_effect": effect.post_effect,
                    "candidate_direction": None,
                }
            )
            evidence_index += 1
        for candidate_rule in candidate_rules:
            evidence_rows.append(
                {
                    "match_id": int(event.match_id),
                    "event_uid": int(event.event_uid),
                    "event_index": int(event.event_index),
                    "evidence_index": evidence_index,
                    "evidence_kind": f"candidate_{candidate_rule.rule_scope}",
                    "rule_id": candidate_rule.rule_id,
                    "tag_id": candidate_rule.tag_id,
                    "signal": None,
                    "post_effect": None,
                    "candidate_direction": candidate_rule.direction,
                }
            )
            evidence_index += 1
        previous_period = period

    states = pd.DataFrame(state_rows)
    evidence = pd.DataFrame(evidence_rows)
    for column in ("match_id", "event_uid", "event_index"):
        states[column] = states[column].astype("int64")
        evidence[column] = evidence[column].astype("int64")
    evidence["evidence_index"] = evidence["evidence_index"].astype("int64")
    _coerce_nullable_int(
        states,
        (
            "possession_index",
            "possession_owner_team_id",
            "owner_team_before_event",
            "candidate_team_before_event",
            "candidate_team_after_event",
            "event_count_so_far",
        ),
    )
    _coerce_nullable_int(evidence, ("tag_id",))
    for column in (
        "period",
        "possession_uid",
        "event_role",
        "actor_relation_to_owner",
        "control_state_before",
        "control_state_after",
        "candidate_status_before_event",
        "candidate_status_after_event",
        "active_possession_after_event",
        "closed_possession_before_event",
        "closed_possession_after_event",
        "boundary_reason",
    ):
        states[column] = states[column].astype("string")
    for column in (
        "evidence_kind",
        "rule_id",
        "signal",
        "post_effect",
        "candidate_direction",
    ):
        evidence[column] = evidence[column].astype("string")
    states["switch_confirmed"] = states["switch_confirmed"].astype("bool")
    states["is_closed_as_of_event"] = states["is_closed_as_of_event"].astype("bool")
    for column in (
        "duration_so_far_seconds",
        "start_x",
        "start_y",
        "current_x",
        "current_y",
    ):
        states[column] = states[column].astype("float64")

    possessions, transitions, audit = _build_possession_tables(v1_possessions)
    return PossessionInferenceV2(states, evidence, possessions, transitions, audit)


def _null_safe_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def _segmentation_identity(
    v1_states: pd.DataFrame,
    v1_possessions: pd.DataFrame,
    result: PossessionInferenceV2,
) -> dict[str, Any]:
    state_projection = result.states.rename(
        columns={"possession_owner_team_id": "owner_team_id"}
    )
    state_columns = [
        "match_id",
        "event_uid",
        "event_index",
        "possession_uid",
        "possession_index",
        "owner_team_id",
        "switch_confirmed",
    ]
    left = v1_states[state_columns].sort_values(state_columns[:3]).reset_index(drop=True)
    right = state_projection[state_columns].sort_values(state_columns[:3]).reset_index(drop=True)
    state_mismatches: list[dict[str, Any]] = []
    if len(left) != len(right):
        state_mismatch_count = abs(len(left) - len(right)) + min(len(left), len(right))
    else:
        mismatch_mask = pd.Series(False, index=left.index)
        for column in state_columns:
            mismatch_mask |= ~_null_safe_equal(left[column], right[column])
        state_mismatch_count = int(mismatch_mask.sum())
        if state_mismatch_count:
            examples = pd.concat(
                {
                    "v1": left[mismatch_mask].head(10),
                    "v2": right[mismatch_mask].head(10),
                },
                names=["source"],
            )
            state_mismatches = examples.reset_index().to_dict("records")

    possession_columns = [
        "match_id",
        "possession_uid",
        "possession_index",
        "owner_team_id",
        "period",
        "start_event_uid",
        "start_event_index",
        "start_reason",
        "previous_possession_uid",
    ]
    p_left = v1_possessions[possession_columns].sort_values(
        ["match_id", "possession_index"]
    ).reset_index(drop=True)
    p_right = result.possessions[possession_columns].sort_values(
        ["match_id", "possession_index"]
    ).reset_index(drop=True)
    possession_mismatch_count = abs(len(p_left) - len(p_right))
    possession_mismatches: list[dict[str, Any]] = []
    if len(p_left) == len(p_right):
        mismatch_mask = pd.Series(False, index=p_left.index)
        for column in possession_columns:
            mismatch_mask |= ~_null_safe_equal(p_left[column], p_right[column])
        possession_mismatch_count = int(mismatch_mask.sum())
        if possession_mismatch_count:
            examples = pd.concat(
                {
                    "v1": p_left[mismatch_mask].head(10),
                    "v2": p_right[mismatch_mask].head(10),
                },
                names=["source"],
            )
            possession_mismatches = examples.reset_index().to_dict("records")
    return {
        "valid": state_mismatch_count == 0 and possession_mismatch_count == 0,
        "state_mismatch_count": state_mismatch_count,
        "possession_mismatch_count": possession_mismatch_count,
        "state_mismatch_examples": state_mismatches,
        "possession_mismatch_examples": possession_mismatches,
    }


def validate_possessions_v2(
    events: pd.DataFrame,
    tags: pd.DataFrame,
    match_teams: pd.DataFrame,
    result: PossessionInferenceV2,
    rules: PossessionRulesV2,
    v1_states: pd.DataFrame | None = None,
    v1_possessions: pd.DataFrame | None = None,
    prefix_checks: int = 0,
) -> dict[str, Any]:
    """Validate V2 causality, graph contracts, and optional V1 identity."""

    states = result.states
    possessions = result.possessions
    errors: list[str] = []
    if len(states) != len(events):
        errors.append("state row count differs from event row count")
    if states[["match_id", "event_uid", "event_index"]].duplicated().any():
        errors.append("event state mapping is not one-to-one")
    if set(states["event_role"].dropna()) - EVENT_ROLES:
        errors.append("unknown event_role")
    if set(states["actor_relation_to_owner"].dropna()) - ACTOR_RELATIONS:
        errors.append("unknown actor_relation_to_owner")
    for column in ("candidate_status_before_event", "candidate_status_after_event"):
        if set(states[column].dropna()) - CANDIDATE_STATUSES:
            errors.append(f"unknown {column}")
    conflicting = states["candidate_status_after_event"] == "conflicting"
    if states.loc[conflicting, "candidate_team_after_event"].notna().any():
        errors.append("conflicting candidate has a concrete team")
    single = states["candidate_status_after_event"] == "single"
    if states.loc[single, "candidate_team_after_event"].isna().any():
        errors.append("single candidate is missing a concrete team")

    first_in_period = (
        states.sort_values(["match_id", "event_index"])
        .groupby(["match_id", "period"], sort=False)
        .head(1)
    )
    period_reset_valid = (
        (first_in_period["control_state_before"] == "DEAD_BALL")
        & first_in_period["owner_team_before_event"].isna()
        & first_in_period["candidate_team_before_event"].isna()
        & (first_in_period["candidate_status_before_event"] == "none")
    )
    if not period_reset_valid.all():
        errors.append("period-first state was not reset")

    assigned = states[states["possession_uid"].notna()]
    possession_ids = set(possessions["possession_uid"].astype(str))
    if not set(assigned["possession_uid"].astype(str)).issubset(possession_ids):
        errors.append("event state references an unknown possession")
    if assigned.groupby("possession_uid")["period"].nunique().gt(1).any():
        errors.append("a possession crosses periods")
    if possessions["possession_uid"].duplicated().any():
        errors.append("possession_uid is not unique")

    known_teams = {
        int(match_id): set(int(team_id) for team_id in group["team_id"])
        for match_id, group in match_teams.groupby("match_id")
    }
    for row in states[
        [
            "match_id",
            "event_team_id",
            "possession_owner_team_id",
            "candidate_team_after_event",
        ]
    ].itertuples(index=False):
        valid = known_teams[int(row.match_id)]
        values = (row.event_team_id, row.possession_owner_team_id, row.candidate_team_after_event)
        if any(not pd.isna(value) and int(value) not in valid for value in values):
            errors.append(f"team reference outside match teams for match {row.match_id}")
            break

    transition_ids = set(result.transitions["previous_possession_uid"].astype(str)) | set(
        result.transitions["next_possession_uid"].astype(str)
    )
    if not transition_ids.issubset(possession_ids):
        errors.append("transition references an unknown possession")
    if not result.transitions.empty:
        next_starts = possessions.set_index("possession_uid")["start_event_index"]
        expected = result.transitions["next_possession_uid"].map(next_starts).astype(int)
        if not expected.equals(result.transitions["transition_event_index"].astype(int)):
            errors.append("transition availability does not equal next-possession start")

    forbidden_tokens = ("final_", "audit_only", "next_possession_uid")
    for table_name, frame in (
        ("event_possession_states", states),
        ("possessions", possessions),
    ):
        if any(
            token in column
            for column in frame.columns
            for token in forbidden_tokens
        ):
            errors.append(f"{table_name} contains a forbidden future-facing field")

    identity = None
    if v1_states is not None and v1_possessions is not None:
        identity = _segmentation_identity(v1_states, v1_possessions, result)
        if not identity["valid"]:
            errors.append("V1/V2 segmentation identity mismatch")

    prefix_errors: list[str] = []
    if prefix_checks:
        rng = random.Random(20260715)
        tags_by_event = _event_tags(tags)
        teams_by_match = {
            int(match_id): tuple(int(team_id) for team_id in group["team_id"])
            for match_id, group in match_teams.groupby("match_id")
        }
        matches = sorted(int(match_id) for match_id in events["match_id"].unique())
        full_by_match = {
            int(match_id): group.sort_values("event_index").reset_index(drop=True)
            for match_id, group in states.groupby("match_id", sort=False)
        }
        for _ in range(prefix_checks):
            match_id = rng.choice(matches)
            match_events = events[events["match_id"] == match_id].sort_values("event_index")
            cutoff = rng.randrange(1, len(match_events) + 1)
            prefix = infer_match_possessions_v2(
                match_events.iloc[:cutoff],
                tags_by_event,
                teams_by_match[match_id],
                rules,
            ).states
            expected = full_by_match[match_id].iloc[:cutoff][prefix.columns]
            if not prefix.reset_index(drop=True).equals(expected.reset_index(drop=True)):
                prefix_errors.append(
                    f"prefix equivalence failed for match {match_id} cutoff {cutoff}"
                )
                break
        errors.extend(prefix_errors)

    return {
        "valid": not errors,
        "errors": errors,
        "events": len(states),
        "possessions": len(possessions),
        "matches": int(states["match_id"].nunique()),
        "assigned_events": int(states["possession_uid"].notna().sum()),
        "unassigned_events": int(states["possession_uid"].isna().sum()),
        "confirmed_switches": int(states["switch_confirmed"].sum()),
        "conflicting_candidates": int(conflicting.sum()),
        "period_starts": len(first_in_period),
        "period_resets_valid": int(period_reset_valid.sum()),
        "prefix_checks": prefix_checks,
        "prefix_errors": prefix_errors,
        "segmentation_identity": identity,
    }


def _write_analysis(
    result: PossessionInferenceV2,
    events: pd.DataFrame,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    states = result.states
    for fields, filename in (
        (["candidate_status_after_event"], "candidate_status_counts.csv"),
        (["event_role", "actor_relation_to_owner"], "event_role_actor_relation_counts.csv"),
        (["control_state_after"], "control_state_counts.csv"),
        (["boundary_reason"], "boundary_reason_counts.csv"),
    ):
        (
            states.groupby(fields, dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
            .to_csv(output_root / filename, index=False)
        )
    candidate_evidence = result.evidence[
        result.evidence["evidence_kind"].str.startswith("candidate", na=False)
    ]
    (
        candidate_evidence.groupby(
            ["rule_id", "candidate_direction"], dropna=False
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .to_csv(output_root / "candidate_rule_counts.csv", index=False)
    )

    joined = states.merge(
        events[["match_id", "event_uid", "event_name", "subevent_name"]],
        on=["match_id", "event_uid"],
        how="left",
        validate="one_to_one",
    )
    (
        joined[joined["switch_confirmed"]]
        .groupby(["event_name", "subevent_name"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .to_csv(output_root / "switch_action_distribution.csv", index=False)
    )
    period_starts = (
        joined.sort_values(["match_id", "event_index"])
        .groupby(["match_id", "period"], sort=False)
        .head(1)
    )
    period_starts.to_csv(output_root / "period_start_states.csv", index=False)

    conflicting_ids = set(
        states.loc[
            states["candidate_status_after_event"] == "conflicting", "event_uid"
        ].astype(int)
    )
    conflict_evidence = candidate_evidence[
        candidate_evidence["event_uid"].isin(conflicting_ids)
    ]
    if conflict_evidence.empty:
        conflict_combinations = pd.DataFrame(
            columns=["event_name", "subevent_name", "directions", "rule_ids", "count"]
        )
    else:
        conflict_events = (
            conflict_evidence.groupby(["match_id", "event_uid", "event_index"])
            .agg(
                directions=(
                    "candidate_direction",
                    lambda values: "|".join(sorted(set(str(value) for value in values))),
                ),
                rule_ids=(
                    "rule_id",
                    lambda values: "|".join(sorted(set(str(value) for value in values))),
                ),
            )
            .reset_index()
            .merge(
                events[["match_id", "event_uid", "event_name", "subevent_name"]],
                on=["match_id", "event_uid"],
                how="left",
                validate="one_to_one",
            )
        )
        conflict_combinations = (
            conflict_events.groupby(
                ["event_name", "subevent_name", "directions", "rule_ids"],
                dropna=False,
            )
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
    conflict_combinations.to_csv(
        output_root / "conflicting_candidate_combinations.csv", index=False
    )

    strata = pd.Series("regular", index=joined.index, dtype="string")
    strata[joined["switch_confirmed"]] = "confirmed_switch"
    strata[joined["candidate_status_after_event"] == "conflicting"] = "candidate_conflict"
    strata[joined["event_role"] == "restart"] = "restart"
    strata[joined["event_role"] == "boundary"] = "boundary"
    strata[joined["boundary_reason"] == "period_break"] = "period_break"
    joined["audit_stratum"] = strata
    sample = (
        joined[joined["audit_stratum"] != "regular"]
        .sort_values(["audit_stratum", "match_id", "event_index"])
        .groupby("audit_stratum", group_keys=False)
        .head(25)
    )
    sample.to_csv(output_root / "manual_audit_sample.csv", index=False)

    audit = result.audit
    possession_audit = result.possessions.merge(
        audit,
        on=["match_id", "possession_uid", "possession_index"],
        how="inner",
        validate="one_to_one",
    )
    one_event = possession_audit[possession_audit["event_count"] == 1]
    (
        one_event.groupby(["start_reason", "close_reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .to_csv(output_root / "one_event_possessions_by_reason.csv", index=False)
    )
    possession_samples: list[pd.DataFrame] = []
    if not one_event.empty:
        sample = one_event.sort_values(["match_id", "possession_index"]).head(50).copy()
        sample["audit_stratum"] = "one_event"
        possession_samples.append(sample)
    long_possessions = possession_audit[
        (possession_audit["event_count"] >= 25)
        | (possession_audit["duration_seconds"] >= 60.0)
    ]
    if not long_possessions.empty:
        sample = long_possessions.sort_values(
            ["event_count", "duration_seconds"], ascending=False
        ).head(50).copy()
        sample["audit_stratum"] = "long_possession"
        possession_samples.append(sample)
    if possession_samples:
        possession_sample = pd.concat(possession_samples, ignore_index=True)
    else:
        possession_sample = possession_audit.head(0).copy()
        possession_sample["audit_stratum"] = pd.Series(dtype="string")
    possession_sample.to_csv(output_root / "possession_audit_sample.csv", index=False)

    distribution_rows: list[dict[str, Any]] = []
    for metric, column in (
        ("events_per_possession", "event_count"),
        ("duration_seconds", "duration_seconds"),
    ):
        values = audit[column].dropna()
        for quantile in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
            distribution_rows.append(
                {
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(values.quantile(quantile)),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(
        output_root / "possession_distributions.csv", index=False
    )


def _assert_runtime_versions() -> dict[str, str]:
    import pyarrow

    major = int(pyarrow.__version__.split(".", 1)[0])
    if not 24 <= major < 26:
        raise RuntimeError(
            f"Possession V2 requires pyarrow>=24,<26, found {pyarrow.__version__}"
        )
    return {"pandas": pd.__version__, "pyarrow": pyarrow.__version__}


def infer_possessions_v2(
    event_root: Path = EVENT_OUTPUT_ROOT,
    output_root: Path = DEFAULT_POSSESSION_ROOT_V2,
    v1_root: Path = DEFAULT_POSSESSION_ROOT_V1,
    competition: str = "England",
    rules_path: Path = DEFAULT_RULES_PATH_V2,
    candidate_rules_path: Path = DEFAULT_CANDIDATE_RULES_PATH_V2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, validate, identity-check, and atomically publish possession V2."""

    versions = _assert_runtime_versions()
    event_competition_root = Path(event_root).resolve() / competition
    final_root = Path(output_root).resolve() / competition
    final_manifest = final_root / "metadata/manifest.json"
    if final_manifest.exists() and not overwrite:
        return json.loads(final_manifest.read_text(encoding="utf-8"))
    v1_competition_root = Path(v1_root).resolve() / competition
    required = [
        event_competition_root / "events.parquet",
        event_competition_root / "event_tags.parquet",
        event_competition_root / "match_teams.parquet",
        v1_competition_root / "event_possession_states.parquet",
        v1_competition_root / "possessions.parquet",
        Path(rules_path),
        Path(candidate_rules_path),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V2 inputs: " + ", ".join(missing))

    events = pd.read_parquet(event_competition_root / "events.parquet")
    tags = pd.read_parquet(event_competition_root / "event_tags.parquet")
    match_teams = pd.read_parquet(event_competition_root / "match_teams.parquet")
    v1_states = pd.read_parquet(v1_competition_root / "event_possession_states.parquet")
    v1_possessions = pd.read_parquet(v1_competition_root / "possessions.parquet")
    rules = load_possession_rules_v2(rules_path, candidate_rules_path)
    tags_by_event = _event_tags(tags)
    teams_by_match = {
        int(match_id): tuple(int(team_id) for team_id in group["team_id"])
        for match_id, group in match_teams.groupby("match_id", sort=False)
    }

    frames: dict[str, list[pd.DataFrame]] = {
        "states": [],
        "evidence": [],
        "possessions": [],
        "transitions": [],
        "audit": [],
    }
    for match_id, match_events in events.groupby("match_id", sort=True):
        match_result = infer_match_possessions_v2(
            match_events,
            tags_by_event,
            teams_by_match[int(match_id)],
            rules,
        )
        for field in frames:
            frames[field].append(getattr(match_result, field))
    result = PossessionInferenceV2(
        **{field: pd.concat(values, ignore_index=True) for field, values in frames.items()}
    )

    validation = validate_possessions_v2(
        events,
        tags,
        match_teams,
        result,
        rules,
        v1_states=v1_states,
        v1_possessions=v1_possessions,
        prefix_checks=100,
    )
    if competition == "England":
        observed = {
            "matches": validation["matches"],
            "events": validation["events"],
            "possessions": validation["possessions"],
            "confirmed_switches": validation["confirmed_switches"],
            "assigned_events": validation["assigned_events"],
            "unassigned_events": validation["unassigned_events"],
        }
        if observed != EXPECTED_ENGLAND_TOTALS:
            validation["valid"] = False
            validation["errors"].append(
                f"England totals differ: expected {EXPECTED_ENGLAND_TOTALS}, observed {observed}"
            )

    build_root = final_root.parent / f".{competition}.build-{uuid.uuid4().hex}"
    build_root.mkdir(parents=True, exist_ok=False)
    _write_json(validation, build_root / "metadata/validation_report.json")
    identity = validation["segmentation_identity"] or {}
    _write_json(identity, build_root / "metadata/segmentation_identity.json")
    if not validation["valid"]:
        _write_json(
            {
                "schema_version": POSSESSION_SCHEMA_VERSION_V2,
                "status": "failed",
                "errors": validation["errors"],
            },
            build_root / "metadata/manifest.json",
        )
        raise ValueError(
            f"V2 validation failed; diagnostic build retained at {build_root}: "
            + "; ".join(validation["errors"])
        )

    tables = {
        "event_possession_states.parquet": result.states,
        "event_possession_evidence.parquet": result.evidence,
        "possessions.parquet": result.possessions,
        "possession_transitions.parquet": result.transitions,
        "possession_audit.parquet": result.audit,
    }
    for filename, frame in tables.items():
        _write_parquet(frame, build_root / filename)
    shutil.copyfile(rules_path, build_root / "possession_rules_v2.csv")
    shutil.copyfile(candidate_rules_path, build_root / "possession_candidate_rules_v2.csv")
    _write_analysis(result, events, build_root / "analysis")

    model_safe_contract = {
        "schema_version": POSSESSION_SCHEMA_VERSION_V2,
        "allowed_tables": [
            "event_possession_states.parquet",
            "event_possession_evidence.parquet",
            "possessions.parquet",
            "possession_transitions.parquet",
        ],
        "forbidden_tables": ["possession_audit.parquet"],
        "open_possession_snapshot_source": "anchor event row in event_possession_states.parquet",
        "transition_visibility_rule": "transition_event_index <= anchor_event_index",
        "candidate_is_owner": False,
        "candidate_status_values": sorted(CANDIDATE_STATUSES),
        "candidate_status_model_encoding": "categorical embedding",
        "event_role_and_actor_relation_usage": "Event categorical features",
    }
    _write_json(model_safe_contract, build_root / "metadata/model_safe_contract.json")
    schema = {
        "schema_version": POSSESSION_SCHEMA_VERSION_V2,
        "competition": competition,
        "tables": {filename.removesuffix(".parquet"): list(frame.columns) for filename, frame in tables.items()},
        "states": ["CONTROL", "CONTESTED", "DEAD_BALL"],
        "candidate_statuses": sorted(CANDIDATE_STATUSES),
        "event_roles": sorted(EVENT_ROLES),
        "actor_relations": sorted(ACTOR_RELATIONS),
    }
    _write_json(schema, build_root / "metadata/schema.json")

    output_files = []
    for path in sorted(build_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            output_files.append(
                {
                    "path": str(path.relative_to(build_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    event_manifest = event_competition_root / "metadata/manifest.json"
    v1_manifest = v1_competition_root / "metadata/manifest.json"
    manifest = {
        "schema_version": POSSESSION_SCHEMA_VERSION_V2,
        "status": "complete",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": competition,
        "event_source_root": str(event_competition_root),
        "v1_identity_source_root": str(v1_competition_root),
        "output_root": str(final_root),
        "runtime_versions": versions,
        "source_hashes": {
            "event_manifest": _sha256(event_manifest) if event_manifest.exists() else None,
            "v1_manifest": _sha256(v1_manifest) if v1_manifest.exists() else None,
            "rules": _sha256(Path(rules_path)),
            "candidate_rules": _sha256(Path(candidate_rules_path)),
            "implementation": _sha256(Path(__file__)),
        },
        "totals": {
            "matches": validation["matches"],
            "events": validation["events"],
            "possessions": validation["possessions"],
            "confirmed_switches": validation["confirmed_switches"],
            "assigned_events": validation["assigned_events"],
            "unassigned_events": validation["unassigned_events"],
            "conflicting_candidates": validation["conflicting_candidates"],
        },
        "validation": {
            "valid": True,
            "prefix_checks": validation["prefix_checks"],
            "segmentation_identity": identity,
        },
        "output_files": output_files,
    }
    _write_json(manifest, build_root / "metadata/manifest.json")

    backup_root = final_root.parent / f".{competition}.backup-{uuid.uuid4().hex}"
    if final_root.exists():
        if not overwrite:
            raise FileExistsError(final_root)
        os.replace(final_root, backup_root)
    try:
        os.replace(build_root, final_root)
    except Exception:
        if backup_root.exists() and not final_root.exists():
            os.replace(backup_root, final_root)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)
    return manifest
