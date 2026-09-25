"""Validation-only method selection for the lightweight confirmation."""

from __future__ import annotations

from typing import Any

from .constants import CONFIRMATION_METHODS


def select_confirmed_methods(
    means: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, str], float]:
    inverse_accuracy = means["event"]["inverse_ce"]["accuracy"]
    accuracy_floor = inverse_accuracy - 0.01
    eligible_event = [
        method
        for method in CONFIRMATION_METHODS["event"]
        if means["event"][method]["accuracy"] >= accuracy_floor
    ]
    if not eligible_event:
        raise ValueError("No Event method satisfies the inverse-CE accuracy guard")
    winners = {
        "event": max(
            eligible_event,
            key=lambda method: means["event"][method]["macro_f1"],
        ),
        "time": min(
            CONFIRMATION_METHODS["time"],
            key=lambda method: means["time"][method]["mae_seconds"],
        ),
        "position": min(
            CONFIRMATION_METHODS["position"],
            key=lambda method: means["position"][method]["distance_mae_m"],
        ),
    }
    return winners, accuracy_floor

