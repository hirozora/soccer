"""Cross-fitted validation, locking, and reporting for Team candidate priors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .oracle_dependency_study import ORACLE_DEPENDENCY_ROOT
from .partial_sharing_study import test_dir as partial_test_dir
from .partial_sharing_study import training_dir as partial_training_dir
from .team_candidate_prior import load_team_candidate_cache, metrics, prediction_frame
from .team_candidate_prior_study import (
    FOLD_COUNT,
    LAMBDA_GRID,
    SELECTOR_SEED,
    TEAM_CANDIDATE_PRIOR_ROOT,
    lock_path,
    test_prediction_path,
    validation_prediction_path,
)


def _fold_map(match_ids: set[int]) -> dict[int, int]:
    def key(match_id: int) -> str:
        return hashlib.sha256(f"{SELECTOR_SEED}:{match_id}".encode()).hexdigest()

    ordered = sorted(match_ids, key=key)
    return {match_id: index % FOLD_COUNT for index, match_id in enumerate(ordered)}


def _seed_mean_metric(frame: pd.DataFrame, column: str) -> float:
    active = frame[frame.player_mask.astype(bool)].copy()
    if column == "top1":
        active["value"] = (active.player_rank <= 1).astype(float)
    elif column == "mrr":
        active["value"] = active.reciprocal_rank.astype(float)
    else:
        raise ValueError(column)
    return float(active.groupby("seed").value.mean().mean())


def _best_lambda(frames: dict[float, pd.DataFrame]) -> tuple[float, pd.DataFrame]:
    rows = []
    for value, frame in frames.items():
        rows.append({
            "lambda": float(value),
            "top1": _seed_mean_metric(frame, "top1"),
            "mrr": _seed_mean_metric(frame, "mrr"),
        })
    table = pd.DataFrame(rows).sort_values(
        ["top1", "mrr", "lambda"], ascending=[False, False, True]
    ).reset_index(drop=True)
    return float(table.iloc[0]["lambda"]), table


def _all_frames(split: str, method: str, lambda_value: float = 0.0) -> pd.DataFrame:
    return pd.concat(
        [
            prediction_frame(
                load_team_candidate_cache(seed, split), method, lambda_value=lambda_value
            )
            for seed in CONFIRMATION_SEEDS
        ],
        ignore_index=True,
    )


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["seed", "sample_id"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if left[keys].to_dict("records") != right[keys].to_dict("records"):
        raise RuntimeError("Candidate-prior predictions are not aligned")
    if not np.array_equal(left.player_mask.to_numpy(), right.player_mask.to_numpy()):
        raise RuntimeError("Candidate-prior masks are not aligned")
    return left, right


def paired_bootstrap(
    reference: pd.DataFrame, candidate: pd.DataFrame, iterations: int = 10_000
) -> dict[str, Any]:
    reference, candidate = _aligned(reference, candidate)
    active = reference.player_mask.astype(bool).to_numpy()
    reference, candidate = reference.loc[active].copy(), candidate.loc[active].copy()
    matches = np.asarray(sorted(reference.match_id.astype(int).unique()))
    index = {int(match): offset for offset, match in enumerate(matches)}
    counts = np.zeros(len(matches), dtype=float)
    left_top1 = np.zeros(len(matches), dtype=float)
    right_top1 = np.zeros(len(matches), dtype=float)
    left_mrr = np.zeros(len(matches), dtype=float)
    right_mrr = np.zeros(len(matches), dtype=float)
    for match, group in reference.groupby(reference.match_id.astype(int)):
        offset = index[int(match)]
        other = candidate.loc[group.index]
        counts[offset] = len(group)
        left_top1[offset] = np.sum(group.player_rank.to_numpy() <= 1)
        right_top1[offset] = np.sum(other.player_rank.to_numpy() <= 1)
        left_mrr[offset] = group.reciprocal_rank.sum()
        right_mrr[offset] = other.reciprocal_rank.sum()
    draws = np.random.default_rng(SELECTOR_SEED).integers(
        0, len(matches), size=(iterations, len(matches))
    )
    denominator = counts[draws].sum(axis=1)
    top1_delta = (
        right_top1[draws].sum(axis=1) - left_top1[draws].sum(axis=1)
    ) / denominator
    mrr_delta = (
        right_mrr[draws].sum(axis=1) - left_mrr[draws].sum(axis=1)
    ) / denominator
    left_metrics, right_metrics = metrics(reference), metrics(candidate)
    return {
        "reference_top1": left_metrics["top1"],
        "candidate_top1": right_metrics["top1"],
        "top1_improvement": right_metrics["top1"] - left_metrics["top1"],
        "top1_ci95": [float(value) for value in np.quantile(top1_delta, (0.025, 0.975))],
        "mrr_improvement": right_metrics["mrr"] - left_metrics["mrr"],
        "mrr_ci95": [float(value) for value in np.quantile(mrr_delta, (0.025, 0.975))],
        "eligible_samples": int(len(reference)),
        "resampling_unit": "match",
        "replicates": iterations,
    }


def _improving_seeds(reference: pd.DataFrame, candidate: pd.DataFrame) -> int:
    reference, candidate = _aligned(reference, candidate)
    count = 0
    for seed in CONFIRMATION_SEEDS:
        left = reference[(reference.seed == seed) & reference.player_mask.astype(bool)]
        right = candidate[(candidate.seed == seed) & candidate.player_mask.astype(bool)]
        count += int(metrics(right)["top1"] > metrics(left)["top1"])
    return count


def _source_baseline_path(split: str, seed: int) -> Path:
    if split == "validation":
        return partial_training_dir(seed) / "validation_predictions_guarded_core.parquet"
    if split == "test":
        return partial_test_dir(seed) / "test_predictions.parquet"
    raise ValueError(split)


def _verify_base_regression(base: pd.DataFrame, split: str) -> None:
    expected = []
    for seed in CONFIRMATION_SEEDS:
        frame = pd.read_parquet(_source_baseline_path(split, seed)).copy()
        frame["seed"] = seed
        expected.append(frame)
    expected_frame = pd.concat(expected, ignore_index=True)
    keys = ["seed", "sample_id"]
    base = base.sort_values(keys).reset_index(drop=True)
    expected_frame = expected_frame.sort_values(keys).reset_index(drop=True)
    if base[keys].to_dict("records") != expected_frame[keys].to_dict("records"):
        raise RuntimeError("TC-Base samples differ from Partial-L2-F80")
    active = expected_frame.player_mask.astype(bool).to_numpy()
    if not np.array_equal(
        base.player_rank.to_numpy()[active], expected_frame.player_rank.to_numpy()[active]
    ):
        raise RuntimeError("TC-Base ranks differ from Partial-L2-F80")
    if not np.array_equal(base.team_pred.to_numpy(), expected_frame.team_pred.to_numpy()):
        raise RuntimeError("TC-Base Team predictions differ from Partial-L2-F80")
    difference = np.max(np.abs(
        base.team_same_probability.to_numpy()
        - expected_frame.team_same_probability.to_numpy()
    ))
    if difference >= 1e-6:
        raise RuntimeError(f"TC-Base Team probabilities differ by {difference}")


def lock_validation_decision() -> Path:
    base = _all_frames("validation", "base")
    hard = _all_frames("validation", "hard")
    _verify_base_regression(base, "validation")
    fold_by_match = _fold_map(set(base.match_id.astype(int)))
    base["fold"] = base.match_id.astype(int).map(fold_by_match)
    hard["fold"] = hard.match_id.astype(int).map(fold_by_match)
    soft_by_lambda = {
        value: _all_frames("validation", "soft", value).assign(
            fold=lambda frame: frame.match_id.astype(int).map(fold_by_match)
        )
        for value in LAMBDA_GRID
    }

    oof_parts = []
    sweep_parts = []
    fold_lambdas: dict[str, float] = {}
    for fold in range(FOLD_COUNT):
        training = {
            value: frame[frame.fold != fold].copy()
            for value, frame in soft_by_lambda.items()
        }
        selected, table = _best_lambda(training)
        table["held_out_fold"] = fold
        sweep_parts.append(table)
        fold_lambdas[str(fold)] = selected
        oof_parts.append(soft_by_lambda[selected][soft_by_lambda[selected].fold == fold])
    soft_oof = pd.concat(oof_parts, ignore_index=True)
    base = base.drop(columns="fold")
    hard = hard.drop(columns="fold")
    soft_oof = soft_oof.drop(columns="fold")
    validation_prediction_path("base").parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(validation_prediction_path("base"), index=False)
    hard.to_parquet(validation_prediction_path("hard_pred"), index=False)
    soft_oof.to_parquet(validation_prediction_path("soft_pred_oof"), index=False)
    pd.concat(sweep_parts, ignore_index=True).to_csv(
        TEAM_CANDIDATE_PRIOR_ROOT / "validation/lambda_crossfit.csv", index=False
    )

    final_lambda, full_sweep = _best_lambda(soft_by_lambda)
    full_sweep.to_csv(
        TEAM_CANDIDATE_PRIOR_ROOT / "validation/lambda_full_validation.csv", index=False
    )
    soft_comparison = paired_bootstrap(base, soft_oof)
    hard_comparison = paired_bootstrap(base, hard)
    soft_vs_hard = paired_bootstrap(hard, soft_oof)
    soft_effective = bool(
        soft_comparison["top1_improvement"] >= 0.01
        and soft_comparison["top1_ci95"][0] > 0
        and soft_comparison["mrr_improvement"] >= -0.005
        and _improving_seeds(base, soft_oof) >= 2
    )
    hard_effective = bool(
        hard_comparison["top1_improvement"] >= 0.01
        and hard_comparison["top1_ci95"][0] > 0
        and hard_comparison["mrr_improvement"] >= -0.005
        and _improving_seeds(base, hard) >= 2
    )
    if soft_effective and hard_effective:
        selected_method = (
            "hard_pred"
            if hard_comparison["candidate_top1"] - soft_comparison["candidate_top1"] >= 0.005
            else "soft_pred"
        )
    elif soft_effective:
        selected_method = "soft_pred"
    elif hard_effective:
        selected_method = "hard_pred"
    else:
        selected_method = "base"

    payload = {
        "split": "validation",
        "cross_fitting": {
            "folds": FOLD_COUNT,
            "selector_seed": SELECTOR_SEED,
            "fold_lambdas": fold_lambdas,
            "fold_assignment": {str(key): value for key, value in fold_by_match.items()},
        },
        "final_lambda": final_lambda,
        "comparisons": {
            "soft_pred_minus_base": soft_comparison,
            "hard_pred_minus_base": hard_comparison,
            "soft_pred_minus_hard_pred": soft_vs_hard,
        },
        "improving_seeds": {
            "soft_pred": _improving_seeds(base, soft_oof),
            "hard_pred": _improving_seeds(base, hard),
        },
        "effective": {"soft_pred": soft_effective, "hard_pred": hard_effective},
        "selected_method": selected_method,
        "enable_posterior_stage_b": selected_method != "base",
        "next_step": (
            "team_aware_player_posterior_to_event_position"
            if selected_method != "base"
            else "retain_partial_l2_f80"
        ),
        "test_accessed": False,
    }
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def evaluate_locked_test() -> Path:
    if not lock_path().exists():
        raise RuntimeError("Validation method lock is required")
    lock = json.loads(lock_path().read_text())
    method = lock["selected_method"]
    base = _all_frames("test", "base")
    _verify_base_regression(base, "test")
    candidate = (
        base.copy()
        if method == "base"
        else _all_frames("test", "soft", lock["final_lambda"])
        if method == "soft_pred"
        else _all_frames("test", "hard")
    )
    test_prediction_path("base").parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(test_prediction_path("base"), index=False)
    candidate.to_parquet(test_prediction_path(method), index=False)
    comparison = paired_bootstrap(base, candidate)
    payload = {
        "selected_method": method,
        "lambda": lock["final_lambda"] if method == "soft_pred" else None,
        "comparison_to_base": comparison,
        "base_metrics": metrics(base[base.player_mask.astype(bool)]),
        "selected_metrics": metrics(candidate[candidate.player_mask.astype(bool)]),
        "test_accessed": True,
        "test_does_not_reselect_method": True,
    }
    path = TEAM_CANDIDATE_PRIOR_ROOT / "test/test_confirmation.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _group_report(base: pd.DataFrame, candidate: pd.DataFrame, output: Path) -> None:
    base, candidate = _aligned(base, candidate)
    active = base.player_mask.astype(bool).to_numpy()
    frame = candidate.loc[active].copy()
    frame["base_correct"] = (base.loc[active].player_rank <= 1).to_numpy()
    frame["candidate_correct"] = frame.player_rank <= 1
    frame["delta"] = frame.candidate_correct.astype(float) - frame.base_correct.astype(float)
    frame["confidence_bin"] = pd.cut(
        frame.team_confidence, [0.5, 0.6, 0.7, 0.8, 0.9, 1.000001], right=False
    )
    for column in (
        "team_correct", "confidence_bin", "team_true", "candidate_count",
        "event_true", "control_state", "event_role", "switch_confirmed",
    ):
        frame.groupby(column, observed=True).agg(
            count=("delta", "size"),
            base_top1=("base_correct", "mean"),
            candidate_top1=("candidate_correct", "mean"),
            top1_delta=("delta", "mean"),
        ).to_csv(output / f"test_by_{column}.csv")


def build_final_report() -> Path:
    if not lock_path().exists():
        raise RuntimeError("Validation method lock is required")
    confirmation_path = TEAM_CANDIDATE_PRIOR_ROOT / "test/test_confirmation.json"
    if not confirmation_path.exists():
        raise RuntimeError("Locked test confirmation is missing")
    lock = json.loads(lock_path().read_text())
    test = json.loads(confirmation_path.read_text())
    oracle_path = ORACLE_DEPENDENCY_ROOT / "report/report.json"
    oracle = json.loads(oracle_path.read_text())
    hard_reference = oracle["test_confirmation"]["hard_true_team"]["partial_l2_f80"]
    uplift = float(test["comparison_to_base"]["top1_improvement"])
    hard_uplift = float(hard_reference["improvement"])
    report = {
        "validation_lock": lock,
        "test_confirmation": test,
        "true_team_hard_reference": hard_reference,
        "oracle_recovery_ratio": uplift / hard_uplift if hard_uplift > 0 else None,
        "final_decision_unchanged_by_test": True,
        "interpretation": (
            "The experiment changes the Player posterior/ranking, not the candidate Player states."
        ),
    }
    output = TEAM_CANDIDATE_PRIOR_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(test_prediction_path("base"))
    selected = pd.read_parquet(test_prediction_path(lock["selected_method"]))
    _group_report(base, selected, output)
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
