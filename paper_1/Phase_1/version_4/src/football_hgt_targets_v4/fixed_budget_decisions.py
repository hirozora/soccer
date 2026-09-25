"""Validation-only gates for fixed-budget extension, conflict, and locking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .fixed_budget_reporting import paired_bootstrap
from .fixed_budget_study import (
    FIXED_BUDGET_ROOT,
    MATCHED_PAIRS,
    training_dir,
)


CORE_THRESHOLDS = {
    "event_macro_f1": 0.02,
    "time_mae_seconds": 0.05,
    "position_distance_mae_m": 0.50,
}


def read_result(configuration: str, seed: int) -> dict[str, Any]:
    return json.loads((training_dir(configuration, seed) / "result.json").read_text(encoding="utf-8"))


def metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
        "team_accuracy": float(metrics["team"]["accuracy"]),
        "team_macro_f1": float(metrics["team"]["macro_f1"]),
        "player_top1": float(metrics["player"]["top1_accuracy"]),
        "player_top3": float(metrics["player"]["top3_accuracy"]),
        "player_top5": float(metrics["player"]["top5_accuracy"]),
        "player_mrr": float(metrics["player"]["mrr"]),
        "core_etp_loss": float(metrics["core_etp_loss"]),
        "joint_active_loss": float(metrics["joint_active_loss"]),
    }


def _write(name: str, value: Any) -> Path:
    output = FIXED_BUDGET_ROOT / "decisions" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def training_boundary_decision() -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    for configuration in {name for pair in MATCHED_PAIRS for name in pair}:
        seeds = []
        for seed in CONFIRMATION_SEEDS:
            result = read_result(configuration, seed)
            metric = "core_etp_loss" if configuration.startswith("three_") else "joint_active_loss"
            history = result["history"]
            if len(history) < 16:
                raise RuntimeError(f"{configuration}/seed{seed} did not complete 16 epochs")
            first_budget = history[:16]
            best_epoch = min(first_budget, key=lambda row: row[metric])["epoch"]
            previous = float(np.mean([row[metric] for row in first_budget[10:13]]))
            recent = float(np.mean([row[metric] for row in first_budget[13:16]]))
            hit = best_epoch in {15, 16} and recent < previous - 1e-4
            seeds.append({
                "seed": seed,
                "metric": metric,
                "best_epoch": best_epoch,
                "epoch11_13_mean": previous,
                "epoch14_16_mean": recent,
                "hit": hit,
            })
        configurations[configuration] = {
            "seeds": seeds,
            "hit_count": sum(item["hit"] for item in seeds),
            "training_boundary_hit": sum(item["hit"] for item in seeds) >= 2,
        }
    pairs = []
    for three, five in MATCHED_PAIRS:
        extend = configurations[three]["training_boundary_hit"] or configurations[five]["training_boundary_hit"]
        pairs.append({"three": three, "five": five, "extend_to_24": extend})
    decision = {"configurations": configurations, "pairs": pairs}
    _write("training_boundary.json", decision)
    return decision


def _selected_metrics(configuration: str, result: dict[str, Any]) -> dict[str, Any]:
    return result["validation"]


def stage2_trigger_decision() -> dict[str, Any]:
    pairs = []
    trigger = False
    for three, five in MATCHED_PAIRS:
        rows = []
        for seed in CONFIRMATION_SEEDS:
            left_result = read_result(three, seed)
            right_result = read_result(five, seed)
            if left_result["initial_common_sha256"] != right_result["initial_common_sha256"]:
                raise RuntimeError(f"Matched initialization differs: {three}/{five}/seed{seed}")
            if left_result["epochs_completed"] != right_result["epochs_completed"]:
                raise RuntimeError(f"Matched budgets differ: {three}/{five}/seed{seed}")
            left_frame = pd.read_parquet(training_dir(three, seed) / "validation_predictions.parquet")
            right_frame = pd.read_parquet(training_dir(five, seed) / "validation_predictions.parquet")
            if left_frame.sample_id.tolist() != right_frame.sample_id.tolist():
                raise RuntimeError(f"Matched samples differ: {three}/{five}/seed{seed}")
            for field in ("event_true", "time_true", "time_mask", "position_true_x", "position_true_y", "position_mask"):
                if not np.array_equal(left_frame[field].to_numpy(), right_frame[field].to_numpy()):
                    raise RuntimeError(f"Matched targets differ: {three}/{five}/{field}/seed{seed}")
            left = metric_values(_selected_metrics(three, left_result))
            right = metric_values(_selected_metrics(five, right_result))
            rows.append({"seed": seed, **{name: right[name] - left[name] for name in CORE_THRESHOLDS}})
        frame = pd.DataFrame(rows)
        checks = {
            "event": int((frame.event_macro_f1 <= -CORE_THRESHOLDS["event_macro_f1"]).sum()) >= 2,
            "time": int((frame.time_mae_seconds >= CORE_THRESHOLDS["time_mae_seconds"]).sum()) >= 2,
            "position": int((frame.position_distance_mae_m >= CORE_THRESHOLDS["position_distance_mae_m"]).sum()) >= 2,
        }
        pair_trigger = any(checks.values())
        trigger = trigger or pair_trigger
        pairs.append({"three": three, "five": five, "seed_differences": rows, "checks": checks, "trigger": pair_trigger})
    result = {"thresholds": CORE_THRESHOLDS, "pairs": pairs, "run_stage2": trigger}
    _write("stage2_trigger.json", result)
    return result


def _conflict_effect(reference: str, candidate: str) -> dict[str, Any]:
    differences = []
    for seed in CONFIRMATION_SEEDS:
        left = metric_values(read_result(reference, seed)["validation_core"])
        right = metric_values(read_result(candidate, seed)["validation_core"])
        differences.append({"seed": seed, **{name: right[name] - left[name] for name in CORE_THRESHOLDS}})
    frame = pd.DataFrame(differences)
    bootstrap = paired_bootstrap(reference, candidate, split="validation", checkpoint="core")
    checks = {}
    for metric, threshold in CORE_THRESHOLDS.items():
        values = frame[metric]
        low, high = bootstrap[metric]["ci95"]
        if metric == "event_macro_f1":
            checks[metric] = int((values < 0).sum()) >= 2 and high < 0 and float(values.mean()) <= -threshold
        else:
            checks[metric] = int((values > 0).sum()) >= 2 and low > 0 and float(values.mean()) >= threshold
    return {"seed_differences": differences, "bootstrap": bootstrap, "checks": checks, "conflict": any(checks.values())}


def conflict_decision() -> dict[str, Any]:
    comparisons = {
        "t4_team_minus_t3": _conflict_effect("three_f80", "t4_team"),
        "t4_player_minus_t3": _conflict_effect("three_f80", "t4_player"),
        "t5_minus_t3": _conflict_effect("three_f80", "five_f80"),
        "t5_minus_t4_team": _conflict_effect("t4_team", "five_f80"),
        "t5_minus_t4_player": _conflict_effect("t4_player", "five_f80"),
    }
    player = comparisons["t4_player_minus_t3"]["conflict"] or comparisons["t5_minus_t4_team"]["conflict"]
    team = comparisons["t4_team_minus_t3"]["conflict"]
    interaction = (
        comparisons["t5_minus_t3"]["conflict"]
        and not comparisons["t4_team_minus_t3"]["conflict"]
        and not comparisons["t4_player_minus_t3"]["conflict"]
    )
    result = {
        "comparisons": comparisons,
        "player_participates": player,
        "team_participates": team,
        "joint_interaction_only": interaction,
        "run_player_adapter": player,
    }
    _write("conflict.json", result)
    return result


def lock_baseline() -> dict[str, Any]:
    mapping = {
        "five_f80": "three_f80",
        "five_hard": "three_fixedb",
        "five_soft": "three_sfb",
    }
    if all((training_dir("t5_player_adapter", seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS):
        mapping["t5_player_adapter"] = "three_f80"
    candidates = []
    for candidate, reference in mapping.items():
        rows = []
        for seed in CONFIRMATION_SEEDS:
            candidate_result = read_result(candidate, seed)
            reference_result = read_result(reference, seed)
            if candidate_result["epochs_completed"] != reference_result["epochs_completed"]:
                raise RuntimeError(f"Candidate/reference budget mismatch: {candidate}/{reference}/seed{seed}")
            candidate_metrics = metric_values(candidate_result["validation"])
            reference_metrics = metric_values(reference_result["validation"])
            f80_metrics = metric_values(read_result("five_f80", seed)["validation"])
            rows.append({
                "seed": seed,
                "event_gap": candidate_metrics["event_macro_f1"] - reference_metrics["event_macro_f1"],
                "time_gap": candidate_metrics["time_mae_seconds"] - reference_metrics["time_mae_seconds"],
                "position_gap": candidate_metrics["position_distance_mae_m"] - reference_metrics["position_distance_mae_m"],
                "team_gap_vs_f80": candidate_metrics["team_accuracy"] - f80_metrics["team_accuracy"],
                "player_gap_vs_f80": candidate_metrics["player_top1"] - f80_metrics["player_top1"],
                "core_loss": candidate_metrics["core_etp_loss"],
                "joint_loss": candidate_metrics["joint_active_loss"],
            })
        frame = pd.DataFrame(rows)
        checks = {
            "event": float(frame.event_gap.mean()) > -0.02,
            "time": float(frame.time_gap.mean()) < 0.05,
            "position": float(frame.position_gap.mean()) < 0.50,
            "team": float(frame.team_gap_vs_f80.mean()) >= -0.01,
            "player": float(frame.player_gap_vs_f80.mean()) >= -0.01,
        }
        candidates.append({
            "candidate": candidate,
            "reference": reference,
            "rows": rows,
            "checks": checks,
            "accepted": all(checks.values()),
            "core_loss_mean": float(frame.core_loss.mean()),
            "joint_loss_mean": float(frame.joint_loss.mean()),
        })
    accepted = [item for item in candidates if item["accepted"]]
    if accepted:
        best_core = min(item["core_loss_mean"] for item in accepted)
        tied = [item for item in accepted if item["core_loss_mean"] - best_core < 1e-4]
        winner = min(tied, key=lambda item: item["joint_loss_mean"])["candidate"]
    else:
        winner = None
    result = {
        "candidates": candidates,
        "winner": winner,
        "five_task_baseline_established": winner is not None,
        "next_research_question": None if winner else "which HGT layer should remain shared",
        "test_accessed": False,
    }
    _write("baseline_lock.json", result)
    return result
