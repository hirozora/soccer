"""Deterministic mappings from Open Wyscout events to benchmark labels."""

from __future__ import annotations

from collections.abc import Collection

import torch

from .constants import (
    ACTION_TAGS,
    ACTION_TO_INDEX,
    FAIRPLAY_TAG,
    FINE_EVENT_TO_INDEX,
    HEAD_BODY_TAG,
    INTERCEPTION_TAG,
    LEFT_FOOT_TAG,
    OWN_GOAL_TAG,
    RED_CARD_TAG,
    RIGHT_FOOT_TAG,
    YELLOW_CARD_TAGS,
    ZONE_CENTERS_100,
)


def action4_label(
    event_id: int, subevent_id: int | None, tag_ids: Collection[int]
) -> tuple[int, bool]:
    """Map one real raw event to the shared Seq2Event/NMSTPP action target."""

    tags = set(tag_ids)
    if event_id == 8:
        if subevent_id == 80:
            return ACTION_TO_INDEX["cross"], True
        return ACTION_TO_INDEX["pass"], True
    if event_id == 3:
        if subevent_id in {30, 32}:
            return ACTION_TO_INDEX["cross"], True
        if subevent_id in {33, 35}:
            return ACTION_TO_INDEX["shot"], True
        if subevent_id in {31, 34, 36}:
            return ACTION_TO_INDEX["pass"], True
    if event_id == 10:
        return ACTION_TO_INDEX["shot"], True
    if event_id == 1 and subevent_id == 11 and tags.intersection(ACTION_TAGS):
        return ACTION_TO_INDEX["dribble"], True
    if event_id == 7:
        if subevent_id == 70:
            return ACTION_TO_INDEX["dribble"], True
        if subevent_id == 71:
            return ACTION_TO_INDEX["pass"], True
        if subevent_id == 72 and not tags:
            return ACTION_TO_INDEX["dribble"], True
    return -1, False


def _shot_body_label(tag_ids: set[int]) -> str:
    if HEAD_BODY_TAG in tag_ids:
        return "head_shot"
    if LEFT_FOOT_TAG in tag_ids:
        return "left_foot_shot"
    # The official preprocessor defaults unresolved shots to right foot.
    return "right_foot_shot"


def unified_fine_label(
    event_id: int, subevent_id: int | None, tag_ids: Collection[int]
) -> int:
    """Apply a Wyscout-Open adapter followed by official LEM override order.

    Carry and synthetic period-end events are deliberately never generated.
    The returned ID always uses the official 32-entry tokenizer order.
    """

    tags = set(tag_ids)

    # Approximate the V3 primary type using only deterministic Open fields.
    if OWN_GOAL_TAG in tags:
        label = "own_goal"
    elif FAIRPLAY_TAG in tags:
        label = "fairplay"
    elif INTERCEPTION_TAG in tags:
        label = "interception"
    elif event_id == 1:
        label = {
            10: "aerial_duel",
            11: "offensive_duel",
            12: "defensive_duel",
            13: "loose_ball_duel",
        }.get(subevent_id, "loose_ball_duel")
    elif event_id == 2:
        label = "infraction"
    elif event_id == 3:
        label = {
            30: "corner",
            31: "free_kick",
            32: "cross",
            33: _shot_body_label(tags),
            34: "goal_kick",
            35: "free_kick",
            36: "throw_in",
        }.get(subevent_id, "free_kick")
    elif event_id == 4:
        label = "goalkeeper_exit"
    elif event_id == 5:
        label = "game_interruption"
    elif event_id == 6:
        label = "offside"
    elif event_id == 7:
        label = {
            70: "acceleration",
            71: "clearance",
            72: "touch",
        }.get(subevent_id, "touch")
    elif event_id == 8:
        label = "pass"
    elif event_id == 9:
        label = "save" if subevent_id == 90 else "shot_against"
    elif event_id == 10:
        label = _shot_body_label(tags)
    else:
        raise ValueError(f"Unsupported Wyscout event ID {event_id}")

    # Official process_event_types override order.
    if event_id == 1:
        label = {
            10: "aerial_duel",
            11: "offensive_duel",
            12: "defensive_duel",
            13: "loose_ball_duel",
        }.get(subevent_id, label)
        if subevent_id == 11 and tags.intersection(ACTION_TAGS):
            label = "dribble"
    if (event_id == 8 and subevent_id == 80) or (
        event_id == 3 and subevent_id == 32
    ):
        label = "cross"
    elif event_id == 8 and subevent_id in {83, 84}:
        label = "long_pass"
    if event_id == 10 or (event_id == 3 and subevent_id == 33):
        label = _shot_body_label(tags)
    if event_id == 9:
        label = "save" if subevent_id == 90 else "shot_against"
    if tags.intersection(YELLOW_CARD_TAGS):
        label = "yellow_card"
    if RED_CARD_TAG in tags:
        label = "red_card"
    return FINE_EVENT_TO_INDEX[label]


def position_to_zone(position_xy: torch.Tensor) -> torch.Tensor:
    """Return zero-based nearest official NMSTPP zone IDs."""

    if position_xy.shape[-1] != 2:
        raise ValueError("position_xy must have final dimension 2")
    centers = torch.tensor(
        ZONE_CENTERS_100, dtype=position_xy.dtype, device=position_xy.device
    ) / 100.0
    distances = torch.sum((position_xy.unsqueeze(-2) - centers) ** 2, dim=-1)
    return distances.argmin(dim=-1)


def zones_to_centers(zone_ids: torch.Tensor) -> torch.Tensor:
    """Convert zero-based zone IDs to normalized x/y centre coordinates."""

    centers = torch.tensor(
        ZONE_CENTERS_100, dtype=torch.float32, device=zone_ids.device
    ) / 100.0
    return centers[zone_ids.long()]


def build_fold_matrix(
    raw_targets: torch.Tensor, fine_targets: torch.Tensor, num_raw: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate P(raw|fine) from training targets only."""

    if raw_targets.shape != fine_targets.shape:
        raise ValueError("raw_targets and fine_targets must have matching shapes")
    num_fine = len(FINE_EVENT_TO_INDEX)
    counts = torch.zeros((num_raw, num_fine), dtype=torch.float64)
    flat = raw_targets.long() * num_fine + fine_targets.long()
    counts.view(-1).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=counts.dtype)
    )
    support = counts.sum(dim=0)
    active = support > 0
    matrix = torch.zeros_like(counts)
    matrix[:, active] = counts[:, active] / support[active]
    return matrix.float(), active


def fold_fine_probabilities(
    fine_probabilities: torch.Tensor, fold_matrix: torch.Tensor
) -> torch.Tensor:
    """Fold [...,32] LEM probabilities into [...,10] raw probabilities."""

    if fine_probabilities.shape[-1] != fold_matrix.shape[1]:
        raise ValueError("Fine probability width does not match fold matrix")
    return fine_probabilities @ fold_matrix.T

