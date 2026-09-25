"""Aggregate frozen-probe results and evaluate pre-registered dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_benchmark.constants import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS

from .constants import CONFIRMATION_SEEDS
from .dependency_study import CONFIGS_BY_FAMILY, DEPENDENCY_ROOT


def _error(frame: pd.DataFrame, family: str) -> np.ndarray:
    if family == "position":
        return np.sqrt(
            ((frame.position_pred_x - frame.position_true_x) * PITCH_LENGTH_METERS) ** 2
            + ((frame.position_pred_y - frame.position_true_y) * PITCH_WIDTH_METERS) ** 2
        ).to_numpy()
    active = frame.time_mask.to_numpy(bool)
    result = np.full(len(frame), np.nan)
    result[active] = np.abs(frame.loc[active, "time_pred"] - frame.loc[active, "time_true"])
    return result


def _sufficient(frame: pd.DataFrame, family: str, match_ids: list[int]) -> np.ndarray:
    errors = _error(frame, family)
    result = np.zeros((len(match_ids), 2), dtype=np.float64)
    matches = frame.match_id.to_numpy(int)
    for index, match_id in enumerate(match_ids):
        selected = (matches == match_id) & np.isfinite(errors)
        result[index] = [errors[selected].sum(), selected.sum()]
    return result


def _bootstrap_improvement(
    baseline: np.ndarray, candidate: np.ndarray, iterations: int = 10_000, seed: int = 20260715
) -> dict[str, Any]:
    if baseline.shape != candidate.shape:
        raise ValueError("Paired statistics differ")
    num_seeds, num_matches, _ = baseline.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for iteration in range(iterations):
        seed_indices = rng.integers(0, num_seeds, num_seeds)
        base_sum = np.zeros(2); candidate_sum = np.zeros(2)
        for selected_seed in seed_indices:
            match_indices = rng.integers(0, num_matches, num_matches)
            base_sum += baseline[selected_seed, match_indices].sum(axis=0)
            candidate_sum += candidate[selected_seed, match_indices].sum(axis=0)
        draws[iteration] = base_sum[0] / base_sum[1] - candidate_sum[0] / candidate_sum[1]
    observed_base = baseline.sum(axis=(0, 1)); observed_candidate = candidate.sum(axis=(0, 1))
    improvement = observed_base[0] / observed_base[1] - observed_candidate[0] / observed_candidate[1]
    return {
        "baseline_error": float(observed_base[0] / observed_base[1]),
        "candidate_error": float(observed_candidate[0] / observed_candidate[1]),
        "improvement": float(improvement),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _read_result(family: str, name: str, seed: int) -> dict[str, Any]:
    path = DEPENDENCY_ROOT / "test" / family / name / f"seed{seed}/result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_dependency_report(output_dir: Path | None = None) -> Path:
    output = output_dir or DEPENDENCY_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats: dict[tuple[str, str], list[np.ndarray]] = {}
    for family, names in CONFIGS_BY_FAMILY.items():
        expected_ids: list[str] | None = None
        match_ids: list[int] | None = None
        for name in names:
            stats[(family, name)] = []
            for seed in CONFIRMATION_SEEDS:
                result = _read_result(family, name, seed)
                metrics = result["test"][family]
                primary = metrics["distance_mae_m"] if family == "position" else metrics["mae_seconds"]
                rows.append({"family": family, "configuration": name, "seed": seed, "primary_error": primary})
                path = DEPENDENCY_ROOT / "test" / family / name / f"seed{seed}/test_predictions.parquet"
                frame = pd.read_parquet(path).sort_values("sample_id")
                ids = frame.sample_id.tolist()
                if expected_ids is None:
                    expected_ids = ids
                    match_ids = sorted(frame.match_id.astype(int).unique().tolist())
                elif ids != expected_ids:
                    raise RuntimeError(f"Unpaired probe predictions in {family}")
                stats[(family, name)].append(_sufficient(frame, family, match_ids or []))
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "metrics_by_seed.csv", index=False)
    by_seed.groupby(["family", "configuration"]).primary_error.agg(["mean", "std"]).to_csv(
        output / "metrics_summary.csv"
    )
    comparisons = {
        "event_to_position_oracle": ("position", "p0_null", "p1_event_oracle"),
        "event_to_position_predicted": ("position", "p0_null", "p2_event_predicted"),
        "time_to_position_oracle": ("position", "p0_null", "p3_time_oracle"),
        "time_to_position_predicted": ("position", "p0_null", "p4_time_predicted"),
        "event_time_to_position_oracle": ("position", "p0_null", "p5_event_time_oracle"),
        "event_time_to_position_predicted": ("position", "p0_null", "p6_event_time_predicted"),
        "event_to_position_shuffled": ("position", "p0_null", "p7_event_shuffled"),
        "time_to_position_shuffled": ("position", "p0_null", "p8_time_shuffled"),
        "event_time_to_position_shuffled": ("position", "p0_null", "p9_event_time_shuffled"),
        "event_to_time_oracle": ("time", "t0_null", "t1_event_oracle"),
        "event_to_time_predicted": ("time", "t0_null", "t2_event_predicted"),
        "event_to_time_shuffled": ("time", "t0_null", "t3_event_shuffled"),
    }
    bootstrap: dict[str, Any] = {}
    for label, (family, baseline, candidate) in comparisons.items():
        bootstrap[label] = _bootstrap_improvement(
            np.stack(stats[(family, baseline)]), np.stack(stats[(family, candidate)])
        )
    (output / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")

    def significant(label: str, threshold: float) -> bool:
        value = bootstrap[label]
        return value["improvement"] >= threshold and value["ci95"][0] > 0

    event_position_oracle = bootstrap["event_to_position_oracle"]["improvement"]
    event_position_predicted = bootstrap["event_to_position_predicted"]["improvement"]
    event_time_oracle = bootstrap["event_to_time_oracle"]["improvement"]
    event_time_predicted = bootstrap["event_to_time_predicted"]["improvement"]
    time_position_oracle = bootstrap["time_to_position_oracle"]["improvement"]
    decisions = {
        "event_to_position_worth_modeling": significant("event_to_position_oracle", 0.5),
        "event_to_time_worth_modeling": significant("event_to_time_oracle", 0.05),
        "time_to_position_worth_modeling": significant("time_to_position_oracle", 0.5),
        "event_position_oracle_improves_at_least_2_90m": event_position_oracle >= 2.90,
        "event_position_predicted_retention": (
            event_position_predicted / event_position_oracle if event_position_oracle > 0 else None
        ),
        "event_time_predicted_retention": (
            event_time_predicted / event_time_oracle if event_time_oracle > 0 else None
        ),
    }
    decisions["event_position_directly_deployable"] = bool(
        bootstrap["event_to_position_predicted"]["ci95"][0] > 0
        and (decisions["event_position_predicted_retention"] or 0) >= 0.5
    )
    decisions["event_time_directly_deployable"] = bool(
        bootstrap["event_to_time_predicted"]["ci95"][0] > 0
        and (decisions["event_time_predicted_retention"] or 0) >= 0.5
    )
    attribution_path = DEPENDENCY_ROOT / "position_attribution/summary.json"
    if attribution_path.exists():
        attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
        decisions["nmstpp_reported_2_90m_gap_is_comparable"] = False
        decisions["fair_continuous_hgt_minus_nmstpp_m"] = attribution["overall_gap_m"]
        decisions["fair_equal_granularity_hgt_minus_nmstpp_m"] = attribution[
            "equal_granularity_gap_m"
        ]
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return output
