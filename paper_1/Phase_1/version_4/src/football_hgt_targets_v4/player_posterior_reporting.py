"""Validation lock, Oracle recovery, and reports for posterior conditioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .oracle_dependency_reporting import cluster_bootstrap
from .oracle_dependency_study import ORACLE_DEPENDENCY_ROOT
from .player_posterior import family_metrics, load_condition_cache
from .player_posterior_study import (
    CONDITIONS,
    FAMILIES,
    PLAYER_POSTERIOR_ROOT,
    decision_path,
    test_dir,
    training_dir,
)


FAMILY_ALIAS = {
    "event": "event_player_state",
    "position": "position_player_state",
}
THRESHOLDS = {"event": 0.005, "position": 0.25}


def _prediction_path(split: str, family: str, condition: str, seed: int) -> Path:
    root = training_dir(family, condition, seed) if split == "validation" else test_dir(family, condition, seed)
    return root / f"{split}_predictions.parquet"


def _load(split: str, family: str, condition: str) -> pd.DataFrame:
    parts = []
    for seed in CONFIRMATION_SEEDS:
        frame = pd.read_parquet(_prediction_path(split, family, condition, seed)).copy()
        frame["seed"] = seed
        frame["condition"] = condition
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _draws(frame: pd.DataFrame, seed: int) -> np.ndarray:
    matches = np.sort(frame.match_id.unique())
    sampled = np.random.default_rng(seed).integers(
        0, len(matches), size=(10_000, len(matches))
    )
    return np.stack(
        [(sampled == index).sum(axis=1) for index in range(len(matches))], axis=1
    )


def _metric(frame: pd.DataFrame, family: str) -> float:
    values = family_metrics(family, frame)
    return float(
        values["known_player_macro_f1"]
        if family == "event"
        else values["known_player_distance_mae_m"]
    )


def _improvement(reference: float, candidate: float, family: str) -> float:
    return reference - candidate if family == "position" else candidate - reference


def _seed_directions(split: str, family: str, reference: str, candidate: str) -> int:
    improving = 0
    for seed in CONFIRMATION_SEEDS:
        left = pd.read_parquet(_prediction_path(split, family, reference, seed))
        right = pd.read_parquet(_prediction_path(split, family, candidate, seed))
        improving += int(_improvement(_metric(left, family), _metric(right, family), family) > 0)
    return improving


def _comparison(
    split: str, family: str, reference: str, candidate: str, draws: np.ndarray
) -> dict[str, Any]:
    left, right = _load(split, family, reference), _load(split, family, candidate)
    result = cluster_bootstrap(left, right, FAMILY_ALIAS[family], draws)
    result["improving_seeds"] = _seed_directions(split, family, reference, candidate)
    result["effective"] = bool(
        result["improvement"] >= THRESHOLDS[family]
        and result["ci95"][0] > 0
        and result["improving_seeds"] >= 2
    )
    result["all_sample_reference"] = (
        family_metrics(family, left)["macro_f1"]
        if family == "event"
        else family_metrics(family, left)["distance_mae_m"]
    )
    result["all_sample_candidate"] = (
        family_metrics(family, right)["macro_f1"]
        if family == "event"
        else family_metrics(family, right)["distance_mae_m"]
    )
    return result


def _all_comparisons(split: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in FAMILIES:
        sample = _load(split, family, "null")
        draws = _draws(sample, 20260816 if split == "validation" else 20260817)
        output[family] = {
            "base_post_minus_null": _comparison(
                split, family, "null", "base_post", draws
            ),
            "team_post_minus_base_post": _comparison(
                split, family, "base_post", "team_post", draws
            ),
            "team_post_minus_null": _comparison(
                split, family, "null", "team_post", draws
            ),
        }
    return output


def lock_dependency_decision() -> Path:
    initial_hashes: dict[str, Any] = {}
    for family in FAMILIES:
        initial_hashes[family] = {}
        for seed in CONFIRMATION_SEEDS:
            values = {}
            for condition in CONDITIONS:
                result = json.loads((training_dir(family, condition, seed) / "result.json").read_text())
                if result["test_accessed"]:
                    raise RuntimeError("Validation training accessed test")
                values[condition] = result["initial_hash"]
            if len(set(values.values())) != 1:
                raise RuntimeError(f"Head initialization differs for {family} seed {seed}")
            initial_hashes[family][str(seed)] = values
    comparisons = _all_comparisons("validation")
    transmitted = {
        family: comparisons[family]["team_post_minus_base_post"]["effective"]
        for family in FAMILIES
    }
    payload = {
        "split": "validation",
        "primary_population": "task_mask & player_mask",
        "secondary_population": "all task-valid samples",
        "comparisons": comparisons,
        "team_dependency_transmitted": transmitted,
        "confirm_probability_chain": any(transmitted.values()),
        "initial_hashes": initial_hashes,
        "test_accessed": False,
    }
    path = decision_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _old_oracle_denominators(split: str) -> dict[str, float]:
    report = json.loads((ORACLE_DEPENDENCY_ROOT / "report/report.json").read_text())
    section = report["validation_decision"] if split == "validation" else report["test_confirmation"]
    return {
        "event": float(section["comparisons"]["event_player_state"]["oracle_minus_null"]["improvement"]),
        "position": float(section["comparisons"]["position_player_state"]["oracle_minus_null"]["improvement"]),
    }


def _diagnostics(split: str) -> dict[str, Any]:
    rows = []
    names = (
        "base_true_probability", "team_true_probability", "base_true_rank",
        "team_true_rank", "base_entropy", "team_entropy", "base_ess", "team_ess",
        "base_norm", "team_norm", "base_true_cosine", "team_true_cosine",
        "base_true_l2", "team_true_l2",
    )
    for seed in CONFIRMATION_SEEDS:
        cache = load_condition_cache(seed, split)
        mask = cache["player_mask"].bool()
        row: dict[str, Any] = {"seed": seed, "samples": int(mask.sum())}
        for name in names:
            row[name] = float(cache[name][mask].float().mean())
        row["base_mrr"] = float((1.0 / cache["base_true_rank"][mask]).mean())
        row["team_mrr"] = float((1.0 / cache["team_true_rank"][mask]).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = PLAYER_POSTERIOR_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{split}_posterior_diagnostics.csv", index=False)
    return {
        name: float(frame[name].mean())
        for name in frame.columns if name not in {"seed", "samples"}
    }


def _group_reports(split: str, family: str, output: Path) -> None:
    base = _load(split, family, "base_post").sort_values(["seed", "sample_id"]).reset_index(drop=True)
    team = _load(split, family, "team_post").sort_values(["seed", "sample_id"]).reset_index(drop=True)
    cache_parts = []
    for seed in CONFIRMATION_SEEDS:
        cache = load_condition_cache(seed, split)
        probabilities = cache["team_logits"].softmax(-1)
        cache_parts.append(pd.DataFrame({
            "seed": seed,
            "sample_id": cache["sample_ids"],
            "team_correct": (probabilities.argmax(-1) == cache["team_true"]).numpy(),
            "team_confidence": probabilities.max(-1).values.numpy(),
            "candidate_count": cache["candidate_count"].numpy(),
            "target_seen": cache["target_seen"].numpy(),
        }))
    metadata = pd.concat(cache_parts, ignore_index=True).sort_values(["seed", "sample_id"]).reset_index(drop=True)
    if base[["seed", "sample_id"]].to_dict("list") != metadata[["seed", "sample_id"]].to_dict("list"):
        raise RuntimeError("Grouped metadata is not aligned")
    frame = team.copy()
    for column in ("team_correct", "team_confidence", "candidate_count", "target_seen"):
        frame[column] = metadata[column]
    frame["confidence_bin"] = pd.cut(
        frame.team_confidence, [0.5, 0.6, 0.7, 0.8, 0.9, 1.000001], right=False
    )
    if family == "event":
        frame["value"] = (team.event_pred == team.event_true).astype(float) - (base.event_pred == base.event_true).astype(float)
    else:
        def distance(values: pd.DataFrame) -> np.ndarray:
            return np.sqrt(
                ((values.position_pred_x - values.position_true_x) * 105.0) ** 2
                + ((values.position_pred_y - values.position_true_y) * 68.0) ** 2
            )
        frame["value"] = distance(base) - distance(team)
        frame = frame[frame.position_mask.astype(bool)]
    for column in (
        "team_correct", "confidence_bin", "candidate_count", "target_seen",
        "event_true", "zone_true", "control_state", "event_role", "switch_confirmed",
    ):
        frame.groupby(column, observed=True).value.agg(["count", "mean"]).to_csv(
            output / f"{split}_{family}_by_{column}.csv"
        )


def build_final_report() -> Path:
    if not decision_path().exists():
        raise RuntimeError("Validation dependency decision required")
    validation = json.loads(decision_path().read_text())
    test = _all_comparisons("test")
    recovery: dict[str, Any] = {}
    for split, comparisons in (("validation", validation["comparisons"]), ("test", test)):
        denominators = _old_oracle_denominators(split)
        recovery[split] = {}
        for family in FAMILIES:
            total = comparisons[family]["team_post_minus_null"]["improvement"]
            incremental = comparisons[family]["team_post_minus_base_post"]["improvement"]
            recovery[split][family] = {
                "old_oracle_internal_denominator": denominators[family],
                "total_recovery": total / denominators[family],
                "team_increment_recovery": incremental / denominators[family],
            }
    output = PLAYER_POSTERIOR_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        _group_reports("validation", family, output)
        _group_reports("test", family, output)
    payload = {
        "validation_decision": validation,
        "test_confirmation": test,
        "oracle_recovery": recovery,
        "posterior_diagnostics": {
            "validation": _diagnostics("validation"),
            "test": _diagnostics("test"),
        },
        "final_decision_unchanged_by_test": True,
        "oracle_denominator_contract": "old Oracle minus old Null within oracle_dependency_probe_v1",
    }
    path = output / "report.json"
    path.write_text(json.dumps(payload, indent=2))
    return path

