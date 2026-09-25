"""Definitions and validation-only selection for the Possession regression study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import CONFIRMATION_SEEDS, EXPERIMENT_ROOT


POSSESSION_EXPERIMENT_ROOT = EXPERIMENT_ROOT / "possession_regression"
TOPOLOGY_VARIANTS = {
    "t0_membership": ("membership", "topology"),
    "t1_owner": ("owner", "topology"),
    "t2_transition": ("transition", "topology"),
}
FEATURE_VARIANTS = {
    "c1_categorical": "categorical",
    "d2_dynamic": "dynamic",
}


def b0_dir(seed: int) -> Path:
    return EXPERIMENT_ROOT / "loss_balance_validation/j2_020" / f"seed{seed}"


def validation_dir(variant: str, seed: int) -> Path:
    if variant in TOPOLOGY_VARIANTS:
        stage = "topology"
    elif variant in FEATURE_VARIANTS:
        stage = "features"
    elif variant == "v3_j0":
        stage = "j0"
    else:
        raise ValueError(f"Unknown Possession variant {variant!r}")
    return POSSESSION_EXPERIMENT_ROOT / "validation" / stage / variant / f"seed{seed}"


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def metric_row(variant: str, seed: int, result: dict[str, Any], split: str) -> dict[str, Any]:
    metrics = result[split]
    return {
        "variant": variant,
        "seed": seed,
        "split": split,
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
    }


def _select_against_b0(frame: pd.DataFrame, candidates: list[str]) -> tuple[str, dict[str, Any]]:
    means = frame.groupby("variant").mean(numeric_only=True)
    reference = means.loc["b0"]
    decisions: dict[str, Any] = {}
    eligible: list[str] = []
    for variant in candidates:
        row = means.loc[variant]
        criteria = {
            "event_accuracy_within_1pp": bool(row.event_accuracy >= reference.event_accuracy - 0.01),
            "time_mae_within_0.05s": bool(row.time_mae_seconds <= reference.time_mae_seconds + 0.05),
            "position_within_0.5m": bool(
                row.position_distance_mae_m <= reference.position_distance_mae_m + 0.5
            ),
        }
        criteria["eligible"] = all(criteria.values())
        if criteria["eligible"]:
            eligible.append(variant)
        decisions[variant] = {
            "criteria": criteria,
            "difference_minus_b0": {
                name: float(row[name] - reference[name])
                for name in (
                    "event_accuracy",
                    "event_macro_f1",
                    "time_mae_seconds",
                    "position_distance_mae_m",
                )
            },
        }
    pool = eligible or candidates
    selected = max(pool, key=lambda name: float(means.loc[name].event_macro_f1))
    return selected, {"decisions": decisions, "any_eligible": bool(eligible)}


def select_topology() -> dict[str, Any]:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        rows.append(metric_row("b0", seed, _read_result(b0_dir(seed)), "validation"))
        for variant in TOPOLOGY_VARIANTS:
            result = _read_result(validation_dir(variant, seed))
            if result.get("test_accessed"):
                raise RuntimeError("Topology selection result accessed test")
            rows.append(metric_row(variant, seed, result, "validation"))
    frame = pd.DataFrame(rows)
    selected, details = _select_against_b0(frame, list(TOPOLOGY_VARIANTS))
    root = POSSESSION_EXPERIMENT_ROOT / "selection"
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "topology_by_seed.csv", index=False)
    state = {
        "selected_topology_variant": selected,
        "selected_topology": TOPOLOGY_VARIANTS[selected][0],
        "selection_split": "validation",
        **details,
    }
    (root / "topology.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def select_final_v3() -> dict[str, Any]:
    topology = select_topology()
    topology_variant = topology["selected_topology_variant"]
    candidates = [topology_variant, *FEATURE_VARIANTS]
    rows = []
    for seed in CONFIRMATION_SEEDS:
        rows.append(metric_row("b0", seed, _read_result(b0_dir(seed)), "validation"))
        for variant in candidates:
            result = _read_result(validation_dir(variant, seed))
            if result.get("test_accessed"):
                raise RuntimeError("V3 selection result accessed test")
            rows.append(metric_row(variant, seed, result, "validation"))
    frame = pd.DataFrame(rows)
    selected, details = _select_against_b0(frame, candidates)
    if selected in TOPOLOGY_VARIANTS:
        selected_topology, selected_features = TOPOLOGY_VARIANTS[selected]
    else:
        selected_topology = topology["selected_topology"]
        selected_features = FEATURE_VARIANTS[selected]
    root = POSSESSION_EXPERIMENT_ROOT / "selection"
    frame.to_csv(root / "final_v3_by_seed.csv", index=False)
    state = {
        "selected_variant": selected,
        "selected_topology": selected_topology,
        "selected_feature_level": selected_features,
        "selection_split": "validation",
        "v3_is_eligible": details["any_eligible"],
        **details,
    }
    (root / "final_v3.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state
