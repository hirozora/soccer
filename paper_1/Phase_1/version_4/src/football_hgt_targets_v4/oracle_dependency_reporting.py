"""Validation locking and match-cluster reporting for Oracle probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .constants import CONFIRMATION_SEEDS
from .oracle_dependency import (
    OracleHead,
    family_metrics,
    hard_true_team_frame,
    load_oracle_cache,
    player_scores_from_head,
)
from .oracle_dependency_study import (
    CONDITIONS,
    FAMILIES,
    ORACLE_DEPENDENCY_ROOT,
    test_dir,
    training_dir,
    validation_lock_path,
)


THRESHOLDS = {
    "player_team": 0.05,
    "position_player_state": 1.0,
    "event_player_state": 0.01,
}


def _prediction_path(split: str, family: str, condition: str, seed: int) -> Path:
    root = training_dir(family, condition, seed) if split == "validation" else test_dir(family, condition, seed)
    return root / f"{split}_predictions.parquet"


def _load_frames(split: str, family: str, condition: str) -> pd.DataFrame:
    parts = []
    for seed in CONFIRMATION_SEEDS:
        frame = pd.read_parquet(_prediction_path(split, family, condition, seed))
        frame["seed"] = seed; frame["condition"] = condition
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _event_confusions(frame: pd.DataFrame, matches: np.ndarray) -> np.ndarray:
    active = frame[frame.player_mask.astype(bool)]
    result = np.zeros((len(matches), 10, 10), dtype=np.float64)
    match_index = {int(value): index for index, value in enumerate(matches)}
    for row in active.itertuples():
        result[match_index[int(row.match_id)], int(row.event_true), int(row.event_pred)] += 1
    return result


def _macro_f1(confusion: np.ndarray) -> np.ndarray:
    true = confusion.sum(axis=2)
    predicted = confusion.sum(axis=1)
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    denominator = true + predicted
    values = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)
    return values.mean(axis=1)


def _linear_match_stats(frame: pd.DataFrame, family: str, matches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if family == "position_player_state":
        active = frame[frame.position_mask.astype(bool) & frame.player_mask.astype(bool)].copy()
        active["value"] = np.sqrt(
            ((active.position_pred_x - active.position_true_x) * 105.0) ** 2
            + ((active.position_pred_y - active.position_true_y) * 68.0) ** 2
        )
    elif family == "player_team":
        active = frame[frame.player_mask.astype(bool)].copy()
        active["value"] = (active.player_rank <= 1).astype(float)
    else:
        raise ValueError(family)
    grouped = active.groupby("match_id").value.agg(["sum", "count"])
    sums = np.array([grouped.loc[value, "sum"] if value in grouped.index else 0 for value in matches], dtype=float)
    counts = np.array([grouped.loc[value, "count"] if value in grouped.index else 0 for value in matches], dtype=float)
    return sums, counts


def _metric(frame: pd.DataFrame, family: str) -> float:
    metrics = family_metrics(family, frame)
    if family == "event_player_state": return float(metrics["known_player_macro_f1"])
    if family == "position_player_state": return float(metrics["known_player_distance_mae_m"])
    return float(metrics["top1"])


def _improvement(reference: float, candidate: float, family: str) -> float:
    return reference - candidate if family == "position_player_state" else candidate - reference


def cluster_bootstrap(
    reference: pd.DataFrame, candidate: pd.DataFrame, family: str,
    draws: np.ndarray,
) -> dict[str, Any]:
    keys = ["seed", "sample_id", "match_id"]
    if reference[keys].sort_values(keys).reset_index(drop=True).to_dict("list") != candidate[keys].sort_values(keys).reset_index(drop=True).to_dict("list"):
        raise RuntimeError("Paired probe frames are not aligned")
    matches = np.sort(reference.match_id.unique())
    if draws.shape[1] != len(matches):
        raise ValueError("Bootstrap draw width does not match matches")
    if family == "event_player_state":
        ref_values = _macro_f1((draws @ _event_confusions(reference, matches).reshape(len(matches), -1)).reshape(-1, 10, 10))
        cand_values = _macro_f1((draws @ _event_confusions(candidate, matches).reshape(len(matches), -1)).reshape(-1, 10, 10))
        differences = cand_values - ref_values
    else:
        ref_sum, ref_count = _linear_match_stats(reference, family, matches)
        cand_sum, cand_count = _linear_match_stats(candidate, family, matches)
        ref_values = (draws @ ref_sum) / np.maximum(draws @ ref_count, 1)
        cand_values = (draws @ cand_sum) / np.maximum(draws @ cand_count, 1)
        differences = ref_values - cand_values if family == "position_player_state" else cand_values - ref_values
    reference_metric, candidate_metric = _metric(reference, family), _metric(candidate, family)
    return {
        "reference": reference_metric,
        "candidate": candidate_metric,
        "improvement": _improvement(reference_metric, candidate_metric, family),
        "ci95": np.quantile(differences, (0.025, 0.975)).tolist(),
        "resampling_unit": "match",
        "replicates": int(draws.shape[0]),
    }


def _seed_directions(split: str, family: str, reference: str, candidate: str) -> int:
    improving = 0
    for seed in CONFIRMATION_SEEDS:
        ref = pd.read_parquet(_prediction_path(split, family, reference, seed))
        cand = pd.read_parquet(_prediction_path(split, family, candidate, seed))
        improving += _improvement(_metric(ref, family), _metric(cand, family), family) > 0
    return int(improving)


def _draws(split: str) -> np.ndarray:
    frame = _load_frames(split, FAMILIES[0], CONDITIONS[0])
    count = frame.match_id.nunique()
    generator = np.random.default_rng(20260814 if split == "validation" else 20260815)
    sampled = generator.integers(0, count, size=(10_000, count))
    return np.stack([(sampled == index).sum(axis=1) for index in range(count)], axis=1)


def hard_oracle_results(split: str, device_name: str = "cuda:0") -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    for seed in CONFIRMATION_SEEDS:
        cache = load_oracle_cache(seed, split)
        base = hard_true_team_frame(cache, cache["candidate_scores"], "partial_l2_f80")
        base["seed"] = seed; frames.append(base)
        checkpoint = training_dir("player_team", "null", seed) / "best.pt"
        state = torch.load(checkpoint, map_location=device_name, weights_only=False)
        model = OracleHead("player_team").to(device_name); model.load_state_dict(state["model"])
        null_scores = player_scores_from_head(model, cache, "null", torch.device(device_name))
        null = hard_true_team_frame(cache, null_scores, "retrained_null")
        null["seed"] = seed; frames.append(null)
    frame = pd.concat(frames, ignore_index=True)
    total_samples = sum(len(load_oracle_cache(seed, split)["sample_ids"]) for seed in CONFIRMATION_SEEDS)
    results: dict[str, Any] = {}
    for source, part in frame.groupby("source"):
        unmasked = float(np.mean(part.unmasked_rank <= 1))
        hard = float(np.mean(part.hard_rank <= 1))
        matches = np.sort(part.match_id.unique())
        draws = _draws(split)
        ref = part.rename(columns={"unmasked_rank": "player_rank"}).copy()
        cand = part.rename(columns={"hard_rank": "player_rank"}).copy()
        ref["player_mask"] = True; cand["player_mask"] = True
        results[source] = {
            "unmasked_top1": unmasked,
            "hard_top1": hard,
            "improvement": hard - unmasked,
            "eligible_samples": len(part),
            "coverage": len(part) / max(total_samples, 1),
            "bootstrap": cluster_bootstrap(ref, cand, "player_team", draws),
        }
    return frame, results


def _decision_payload(split: str, include_hard: bool = True) -> dict[str, Any]:
    draws = _draws(split)
    comparisons: dict[str, Any] = {}
    dependencies: dict[str, bool] = {}
    for family in FAMILIES:
        null = _load_frames(split, family, "null")
        oracle = _load_frames(split, family, "oracle")
        shuffled = _load_frames(split, family, "shuffled")
        versus_null = cluster_bootstrap(null, oracle, family, draws)
        versus_shuffled = cluster_bootstrap(shuffled, oracle, family, draws)
        improving = _seed_directions(split, family, "null", "oracle")
        effective = bool(
            improving >= 2
            and versus_null["ci95"][0] > 0
            and versus_shuffled["ci95"][0] > 0
            and versus_null["improvement"] >= THRESHOLDS[family]
        )
        comparisons[family] = {
            "oracle_minus_null": versus_null,
            "oracle_minus_shuffled": versus_shuffled,
            "improving_seeds": improving,
            "threshold": THRESHOLDS[family],
            "effective": effective,
        }
        dependencies[family] = effective
    hard = None
    if include_hard:
        _, hard = hard_oracle_results(split)
        hard_effective = hard["retrained_null"]["improvement"] >= 0.05 and hard["retrained_null"]["bootstrap"]["ci95"][0] > 0
    else:
        hard_effective = False
    if dependencies["position_player_state"] or dependencies["event_player_state"]:
        priority = "player_centric_history"
    elif dependencies["player_team"]:
        priority = "team_conditioned_player_scorer"
    elif hard_effective:
        priority = "explicit_team_aware_candidate_masking"
    else:
        priority = "position_zone_xy_residual"
    return {
        "split": split,
        "comparisons": comparisons,
        "hard_true_team": hard,
        "dependencies": dependencies,
        "hard_team_effective": hard_effective,
        "research_priority": priority,
        "bootstrap": {"unit": "match", "replicates": 10_000, "shared_draws": True},
    }


def lock_validation_decision() -> Path:
    initial_hashes: dict[str, dict[str, list[str]]] = {}
    for family in FAMILIES:
        initial_hashes[family] = {}
        for seed in CONFIRMATION_SEEDS:
            values = []
            for condition in CONDITIONS:
                result = json.loads((training_dir(family, condition, seed) / "result.json").read_text())
                if result.get("test_accessed"):
                    raise RuntimeError("Validation result accessed test")
                values.append(result["initial_hash"])
            if len(set(values)) != 1:
                raise RuntimeError(f"Head initialization mismatch: {family} seed {seed}")
            initial_hashes[family][str(seed)] = values
    payload = _decision_payload("validation")
    payload.update(test_accessed=False, initial_hashes=initial_hashes)
    path = validation_lock_path(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _group_reports(split: str) -> None:
    output = ORACLE_DEPENDENCY_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    position = _load_frames(split, "position_player_state", "oracle")
    position = position[position.position_mask.astype(bool) & position.player_mask.astype(bool)].copy()
    position["distance_m"] = np.sqrt(
        ((position.position_pred_x - position.position_true_x) * 105.0) ** 2
        + ((position.position_pred_y - position.position_true_y) * 68.0) ** 2
    )
    position.groupby("event_true").distance_m.agg(["count", "mean", "median"]).to_csv(output / f"{split}_position_by_event.csv")
    position.groupby("zone_true").distance_m.agg(["count", "mean", "median"]).to_csv(output / f"{split}_position_by_zone.csv")
    player = _load_frames(split, "player_team", "oracle")
    active = player[player.player_mask.astype(bool)].copy()
    active["candidate_bin"] = pd.cut(active.candidate_count, [0, 20, 30, 35, 40, np.inf], right=True)
    active.assign(correct=(active.player_rank <= 1)).groupby("candidate_bin", observed=True).correct.agg(["count", "mean"]).to_csv(output / f"{split}_player_by_candidate_count.csv")


def build_final_report() -> Path:
    if not validation_lock_path().exists():
        raise RuntimeError("Validation decision is not locked")
    validation = json.loads(validation_lock_path().read_text())
    test = _decision_payload("test")
    output = ORACLE_DEPENDENCY_ROOT / "report"; output.mkdir(parents=True, exist_ok=True)
    hard_frame, _ = hard_oracle_results("test")
    hard_frame.to_parquet(output / "test_hard_true_team_predictions.parquet", index=False)
    _group_reports("validation"); _group_reports("test")
    payload = {
        "validation_decision": validation,
        "test_confirmation": test,
        "final_research_priority": validation["research_priority"],
        "true_player_interpretation": "true next player plus its frozen causal private-L2 state; not ID-only",
        "test_does_not_reselect_priority": True,
    }
    path = output / "report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
