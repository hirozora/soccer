"""Validation-only model lock and final reporting for relation gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .partial_sharing_study import test_dir as g0_test_dir
from .partial_sharing_study import training_dir as g0_training_dir
from .state_gating_study import STATE_GATING_ROOT, test_dir, training_dir


CORE_METRICS = {
    "event": "event_macro_f1",
    "time": "time_mae_seconds",
    "position": "position_distance_mae_m",
}


def _read(root: Path) -> dict[str, Any]:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "event_accuracy": float(metrics["event"]["accuracy"]),
        "event_macro_f1": float(metrics["event"]["macro_f1"]),
        "time_mae_seconds": float(metrics["time"]["mae_seconds"]),
        "position_distance_mae_m": float(metrics["position"]["distance_mae_m"]),
        "position_median_m": float(metrics["position"]["distance_median_m"]),
        "team_accuracy": float(metrics["team"]["accuracy"]),
        "team_macro_f1": float(metrics["team"]["macro_f1"]),
        "player_top1": float(metrics["player"]["top1_accuracy"]),
        "player_top3": float(metrics["player"]["top3_accuracy"]),
        "player_top5": float(metrics["player"]["top5_accuracy"]),
        "player_mrr": float(metrics["player"]["mrr"]),
        "player_coverage": float(metrics["player"]["coverage"]),
    }


def _validation_prediction(candidate: bool, seed: int) -> pd.DataFrame:
    root = training_dir(seed) if candidate else g0_training_dir(seed)
    path = root / "validation_predictions_guarded_core.parquet"
    return pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)


def _test_prediction(candidate: bool, seed: int) -> pd.DataFrame:
    root = test_dir(seed) if candidate else g0_test_dir(seed)
    return pd.read_parquet(root / "test_predictions.parquet").sort_values(
        "sample_id"
    ).reset_index(drop=True)


def _paired(candidate: bool, split: str) -> dict[str, Any]:
    reference_statistics = []
    candidate_statistics = []
    expected_ids = None
    for seed in CONFIRMATION_SEEDS:
        reference = (
            _validation_prediction(False, seed)
            if split == "validation" else _test_prediction(False, seed)
        )
        contender = (
            _validation_prediction(candidate, seed)
            if split == "validation" else _test_prediction(candidate, seed)
        )
        if reference.sample_id.tolist() != contender.sample_id.tolist():
            raise RuntimeError(f"G0/G1 {split} sample IDs differ for seed {seed}")
        if expected_ids is None:
            expected_ids = reference.sample_id.tolist()
        elif expected_ids != reference.sample_id.tolist():
            raise RuntimeError(f"{split} sample IDs differ across seeds")
        matches = sorted(reference.match_id.astype(int).unique().tolist())
        reference_statistics.append(_match_statistics(reference, matches))
        candidate_statistics.append(_match_statistics(contender, matches))
    return _bootstrap(
        np.stack(reference_statistics), np.stack(candidate_statistics), 10_000
    )


def _seed_rows(split: str) -> pd.DataFrame:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        for model, root in (
            ("G0", g0_training_dir(seed) if split == "validation" else g0_test_dir(seed)),
            ("G1", training_dir(seed) if split == "validation" else test_dir(seed)),
        ):
            result = _read(root)
            metrics = result["validation_guarded_core"] if split == "validation" else result["test"]
            rows.append({"model": model, "seed": seed, "split": split, **_metric_values(metrics)})
    return pd.DataFrame(rows)


def lock_state_gating_method() -> Path:
    """Use validation only to lock G0 or G1."""

    for seed in CONFIRMATION_SEEDS:
        candidate_root = training_dir(seed)
        result = _read(candidate_root)
        if result.get("test_accessed") or result.get("test") is not None:
            raise RuntimeError(f"G1 validation accessed test data: {candidate_root}")
        if not (candidate_root / "best_guarded_core.pt").exists():
            raise RuntimeError(f"Missing guarded G1 checkpoint: {candidate_root}")
        reference = _read(g0_training_dir(seed))
        if result["initial_common_sha256"] != reference["initial_common_sha256"]:
            raise RuntimeError(f"G0/G1 common initialization differs for seed {seed}")

    rows = _seed_rows("validation")
    pivot = rows.set_index(["model", "seed"])
    numeric_columns = [
        name for name in rows.columns
        if name not in {"model", "seed", "split"}
    ]
    differences = pivot.loc["G1", numeric_columns] - pivot.loc["G0", numeric_columns]
    bootstrap = _paired(True, "validation")
    effective = {
        "event": bool(
            (differences.event_macro_f1 > 0).sum() >= 2
            and float(differences.event_macro_f1.mean()) >= 0.005
            and bootstrap["event_macro_f1"]["ci95"][0] > 0
        ),
        "time": bool(
            (differences.time_mae_seconds < 0).sum() >= 2
            and float(differences.time_mae_seconds.mean()) <= -0.01
            and bootstrap["time_mae_seconds"]["ci95"][1] < 0
        ),
        "position": bool(
            (differences.position_distance_mae_m < 0).sum() >= 2
            and float(differences.position_distance_mae_m.mean()) <= -0.25
            and bootstrap["position_distance_mae_m"]["ci95"][1] < 0
        ),
    }
    guards = {
        "event_accuracy": float(differences.event_accuracy.mean()) >= -0.01,
        "event_macro_f1": float(differences.event_macro_f1.mean()) > -0.02,
        "time": float(differences.time_mae_seconds.mean()) < 0.05,
        "position": float(differences.position_distance_mae_m.mean()) < 0.50,
        "team": float(differences.team_accuracy.mean()) >= -0.01,
        "player": float(differences.player_top1.mean()) >= -0.01,
    }
    selected = "G1" if any(effective.values()) and all(guards.values()) else "G0"
    payload = {
        "selection_split": "validation",
        "selected_model": selected,
        "g1_effective_tasks": effective,
        "regression_guards": guards,
        "validation_bootstrap": bootstrap,
        "validation_seed_differences": differences.reset_index().to_dict("records"),
        "test_accessed": False,
        "test_policy": (
            "G1 is locked and will be tested without test-time reselection"
            if selected == "G1"
            else "G0 remains locked; G1 test predictions must not be generated"
        ),
    }
    output = STATE_GATING_ROOT / "selection/final_propagation_lock.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def selected_model() -> str:
    path = STATE_GATING_ROOT / "selection/final_propagation_lock.json"
    if not path.exists():
        raise RuntimeError("Validation propagation lock does not exist")
    return json.loads(path.read_text(encoding="utf-8"))["selected_model"]


def build_state_gating_report() -> Path:
    lock_path = STATE_GATING_ROOT / "selection/final_propagation_lock.json"
    if not lock_path.exists():
        raise RuntimeError("Validation propagation lock does not exist")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    output = STATE_GATING_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    validation = _seed_rows("validation")
    validation.to_csv(output / "validation_metrics_by_seed.csv", index=False)

    gate_rows = []
    for seed in CONFIRMATION_SEEDS:
        diagnostics = json.loads(
            (training_dir(seed) / "gate_diagnostics.json").read_text(encoding="utf-8")
        )
        for layer, relations in diagnostics["layers"].items():
            for relation, values in relations.items():
                for group, stats in values["groups"].items():
                    gate_rows.append({
                        "seed": seed,
                        "layer": layer,
                        "relation": relation,
                        "state_group": group,
                        **stats,
                        "edge_weighted_mean": values["edge_weighted_mean"],
                        "edge_count": values["edge_count"],
                    })
    pd.DataFrame(gate_rows).to_csv(output / "gate_diagnostics.csv", index=False)

    report: dict[str, Any] = {
        "validation_lock": lock,
        "selected_model": lock["selected_model"],
        "test_was_not_used_for_selection": True,
        "gate_diagnostics": str(output / "gate_diagnostics.csv"),
    }
    if lock["selected_model"] == "G1":
        test = _seed_rows("test")
        test.to_csv(output / "test_metrics_by_seed.csv", index=False)
        test.groupby("model").mean(numeric_only=True).to_csv(output / "test_summary.csv")
        test_bootstrap = _paired(True, "test")
        (output / "test_paired_bootstrap.json").write_text(
            json.dumps({"iterations": 10_000, "G1_minus_G0": test_bootstrap}, indent=2),
            encoding="utf-8",
        )
        report["test_bootstrap"] = test_bootstrap
        report["conclusion"] = (
            "G1 was selected on validation; test reports whether the state-aware propagation gain generalized"
        )
    else:
        if any((test_dir(seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS):
            raise RuntimeError("G1 test data exists although validation locked G0")
        report["conclusion"] = (
            "Sample-state relation gating did not pass validation; retain standard Partial-L2-F80 and stop this gating line"
        )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
