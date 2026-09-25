"""Validation lock, clustered bootstrap, and reports for Player histories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .oracle_dependency_study import ORACLE_DEPENDENCY_ROOT
from .partial_sharing_study import test_dir as partial_test_dir
from .partial_sharing_study import training_dir as partial_training_dir
from .player_history_study import (
    CONDITIONS,
    PLAYER_HISTORY_ROOT,
    decision_path,
    test_dir,
    training_dir,
)


def _path(split: str, condition: str, seed: int) -> Path:
    root = training_dir(condition, seed) if split == "validation" else test_dir(condition, seed)
    return root / f"{split}_predictions.parquet"


def _baseline_path(split: str, seed: int) -> Path:
    root = partial_training_dir(seed) if split == "validation" else partial_test_dir(seed)
    name = "validation_predictions_guarded_core.parquet" if split == "validation" else "test_predictions.parquet"
    return root / name


def _load(split: str, condition: str) -> pd.DataFrame:
    frames = []
    for seed in CONFIRMATION_SEEDS:
        frame = pd.read_parquet(_path(split, condition, seed)).copy()
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _baseline(split: str) -> pd.DataFrame:
    frames = []
    for seed in CONFIRMATION_SEEDS:
        frame = pd.read_parquet(_baseline_path(split, seed)).copy()
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["seed", "sample_id"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if left[keys].to_dict("records") != right[keys].to_dict("records"):
        raise RuntimeError("Player-history prediction samples are not aligned")
    if not np.array_equal(left.player_mask.to_numpy(), right.player_mask.to_numpy()):
        raise RuntimeError("Player-history masks are not aligned")
    return left, right


def _metric(frame: pd.DataFrame, metric: str) -> float:
    ranks = frame.player_rank.to_numpy(dtype=float)
    if metric == "top1":
        return float(np.mean(ranks <= 1))
    if metric == "mrr":
        return float(np.mean(1.0 / ranks))
    raise ValueError(metric)


def _comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    shuffled_subset: bool = False,
    iterations: int = 10_000,
) -> dict[str, Any]:
    reference, candidate = _aligned(reference, candidate)
    player_mask = reference.player_mask.astype(bool).to_numpy()
    mask = player_mask.copy()
    if shuffled_subset:
        required = (
            reference.target_team_mapping_valid.astype(bool).to_numpy()
            & reference.target_shuffle_eligible.astype(bool).to_numpy()
            & candidate.target_team_mapping_valid.astype(bool).to_numpy()
            & candidate.target_shuffle_eligible.astype(bool).to_numpy()
        )
        mask &= required
    reference = reference.loc[mask].copy()
    candidate = candidate.loc[mask].copy()
    matches = np.asarray(sorted(reference.match_id.astype(int).unique()))
    rng = np.random.default_rng(20260815)
    match_index = {int(match): index for index, match in enumerate(matches)}
    counts = np.zeros(len(matches), dtype=float)
    left_top1 = np.zeros(len(matches), dtype=float)
    right_top1 = np.zeros(len(matches), dtype=float)
    left_mrr = np.zeros(len(matches), dtype=float)
    right_mrr = np.zeros(len(matches), dtype=float)
    for match, group in reference.groupby(reference.match_id.astype(int)):
        index = match_index[int(match)]
        other = candidate.loc[group.index]
        counts[index] = len(group)
        left_top1[index] = np.sum(group.player_rank.to_numpy() <= 1)
        right_top1[index] = np.sum(other.player_rank.to_numpy() <= 1)
        left_mrr[index] = np.sum(1.0 / group.player_rank.to_numpy(dtype=float))
        right_mrr[index] = np.sum(1.0 / other.player_rank.to_numpy(dtype=float))
    draws = rng.integers(0, len(matches), size=(iterations, len(matches)))
    denominators = counts[draws].sum(axis=1)
    top1 = (
        right_top1[draws].sum(axis=1) - left_top1[draws].sum(axis=1)
    ) / denominators
    mrr = (
        right_mrr[draws].sum(axis=1) - left_mrr[draws].sum(axis=1)
    ) / denominators
    return {
        "reference_top1": _metric(reference, "top1"),
        "candidate_top1": _metric(candidate, "top1"),
        "top1_improvement": _metric(candidate, "top1") - _metric(reference, "top1"),
        "top1_ci95": [float(value) for value in np.quantile(top1, (0.025, 0.975))],
        "mrr_improvement": _metric(candidate, "mrr") - _metric(reference, "mrr"),
        "mrr_ci95": [float(value) for value in np.quantile(mrr, (0.025, 0.975))],
        "eligible_samples": int(len(reference)),
        "coverage": float(len(reference) / max(int(player_mask.sum()), 1)),
        "resampling_unit": "match",
        "replicates": iterations,
    }


def _improving_seeds(split: str, reference: str, candidate: str) -> int:
    count = 0
    for seed in CONFIRMATION_SEEDS:
        left = pd.read_parquet(_path(split, reference, seed))
        right = pd.read_parquet(_path(split, candidate, seed))
        left, right = _aligned(
            left.assign(seed=seed), right.assign(seed=seed)
        )
        mask = left.player_mask.astype(bool)
        count += int(_metric(right[mask], "top1") > _metric(left[mask], "top1"))
    return count


def _decision(split: str) -> dict[str, Any]:
    baseline = _baseline(split)
    null = _load(split, "ph_null")
    history = _load(split, "ph_hist_k5")
    shuffled = _load(split, "ph_shuffled_team")
    comparisons = {
        "null_minus_partial_l2": _comparison(baseline, null),
        "hist_k5_minus_null": _comparison(null, history),
        "hist_k5_minus_shuffled_team": _comparison(
            shuffled, history, shuffled_subset=True
        ),
        "shuffled_team_minus_null": _comparison(null, shuffled),
    }
    primary = comparisons["hist_k5_minus_null"]
    identity = comparisons["hist_k5_minus_shuffled_team"]
    effective = bool(
        _improving_seeds(split, "ph_null", "ph_hist_k5") >= 2
        and primary["top1_improvement"] >= 0.01
        and primary["top1_ci95"][0] > 0
        and primary["mrr_improvement"] >= -0.005
        and identity["top1_ci95"][0] > 0
    )
    return {
        "split": split,
        "comparisons": comparisons,
        "improving_seeds": _improving_seeds(split, "ph_null", "ph_hist_k5"),
        "history_effective": effective,
        "enable_stage_b": effective,
        "next_step": (
            "predicted_player_representation_to_event_position"
            if effective
            else "retain_partial_l2_f80"
        ),
        "test_accessed": split == "test",
    }


def lock_player_history_decision() -> Path:
    initial_hashes: dict[str, dict[str, str]] = {}
    for seed in CONFIRMATION_SEEDS:
        initial_hashes[str(seed)] = {}
        for condition in CONDITIONS:
            result = json.loads((training_dir(condition, seed) / "result.json").read_text())
            if result.get("test_accessed"):
                raise RuntimeError("Validation training accessed test")
            initial_hashes[str(seed)][condition] = result["initial_trainable_sha256"]
        if len(set(initial_hashes[str(seed)].values())) != 1:
            raise RuntimeError(f"Condition initialization differs for seed {seed}")
    payload = _decision("validation")
    payload["initial_trainable_hashes"] = initial_hashes
    path = decision_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _group_reports(split: str, output: Path) -> None:
    frame = _load(split, "ph_hist_k5")
    active = frame[frame.player_mask.astype(bool)].copy()
    active["correct"] = active.player_rank <= 1
    active["history_bin"] = pd.cut(
        active.target_history_count, [-1, 0, 4, 9, 19, np.inf],
        labels=["0", "1-4", "5-9", "10-19", "20+"],
    )
    active["recency_bin"] = pd.cut(
        active.target_recency_seconds,
        [-1, 5, 30, 120, 300, np.inf],
        labels=["0-5", "5-30", "30-120", "120-300", "300+"],
        right=False,
    )
    for column in (
        "history_bin",
        "recency_bin",
        "target_current_possession_participated",
        "target_shuffle_eligible",
        "event_true",
        "candidate_count",
    ):
        active.groupby(column, observed=True).correct.agg(["count", "mean"]).to_csv(
            output / f"{split}_by_{column}.csv"
        )


def build_player_history_report() -> Path:
    if not decision_path().exists():
        raise RuntimeError("Validation decision is not locked")
    validation = json.loads(decision_path().read_text())
    test = _decision("test")
    output = PLAYER_HISTORY_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    _group_reports("validation", output)
    _group_reports("test", output)
    oracle_reference = ORACLE_DEPENDENCY_ROOT / "report/report.json"
    true_team = None
    if oracle_reference.exists():
        oracle = json.loads(oracle_reference.read_text())
        true_team = oracle["test_confirmation"]["comparisons"]["player_team"]
    payload = {
        "validation_decision": validation,
        "test_confirmation": test,
        "final_decision_unchanged_by_test": True,
        "true_team_soft_hard_reference": true_team,
        "interpretation": "PH-HistK5 is K5 recent sequence plus cumulative causal player statistics; TrueTeam is a reference uplift, not a Player-history ceiling.",
    }
    path = output / "report.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
