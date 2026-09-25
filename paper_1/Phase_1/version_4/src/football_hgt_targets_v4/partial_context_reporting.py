"""Validation-only context selection and locked-test reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .partial_context_study import (
    ALL_CONFIGURATIONS,
    COST_ORDER,
    PARTIAL_CONTEXT_ROOT,
    TRAINED_CONFIGURATIONS,
    checkpoint_path,
    final_lock_path,
    test_dir,
    training_dir,
)


TASK_METRICS = {
    "event": "event_macro_f1",
    "time": "time_mae_seconds",
    "position": "position_distance_mae_m",
}
PRACTICAL = {"event": 0.005, "time": 0.01, "position": 0.25}


def _read(root: Path) -> dict[str, Any]:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _metrics(payload: dict[str, Any], split: str) -> dict[str, float]:
    metrics = payload[split]
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
        "core_etp_loss": float(metrics["core_etp_loss"]),
    }


def _prediction(configuration: str, seed: int, split: str) -> pd.DataFrame:
    name = "validation_predictions_guarded_core.parquet" if split == "validation" else "test_predictions.parquet"
    return pd.read_parquet((training_dir if split == "validation" else test_dir)(configuration, seed) / name).sort_values("sample_id").reset_index(drop=True)


def _statistics(
    split: str, configurations: tuple[str, ...] = ALL_CONFIGURATIONS
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {name: [] for name in configurations}
    expected_ids = None
    matches = None
    for seed in CONFIRMATION_SEEDS:
        for configuration in configurations:
            frame = _prediction(configuration, seed, split)
            ids = frame.sample_id.tolist()
            if expected_ids is None:
                expected_ids = ids
                matches = sorted(frame.match_id.astype(int).unique())
            elif ids != expected_ids:
                raise RuntimeError(f"Unpaired {split} sample IDs: {configuration}/seed{seed}")
            values[configuration].append(_match_statistics(frame, matches))
    return {name: np.stack(rows) for name, rows in values.items()}


def _task_decision(
    task: str,
    seed_differences: np.ndarray,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    lower_is_better = task in {"time", "position"}
    improving = int(np.sum(seed_differences < 0 if lower_is_better else seed_differences > 0))
    low, high = comparison[TASK_METRICS[task]]["ci95"]
    mean = float(seed_differences.mean())
    practical = mean <= -PRACTICAL[task] if lower_is_better else mean >= PRACTICAL[task]
    directional_ci = high < 0 if lower_is_better else low > 0
    return {
        "mean_difference": mean,
        "improving_seeds": improving,
        "ci95": [float(low), float(high)],
        "practical": practical,
        "effective": improving >= 2 and directional_ci and practical,
    }


def lock_context_from_validation() -> Path:
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in ALL_CONFIGURATIONS}
    initial_hashes: dict[int, str] = {}
    failed_configurations: dict[str, dict[str, Any]] = {}
    for seed in CONFIRMATION_SEEDS:
        f80 = _read(training_dir("partial_l2", seed))
        initial_hashes[seed] = f80["initial_common_sha256"]
        for configuration in ALL_CONFIGURATIONS:
            root = training_dir(configuration, seed)
            if not (root / "result.json").exists():
                history_path = root / "history.json"
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
                expected_guard_failure = (
                    len(history) == 24
                    and not any(item.get("guarded_core_eligible", False) for item in history)
                )
                failed_configurations.setdefault(configuration, {
                    "reason": "no_seed_checkpoint_satisfied_dual_actor_guards" if expected_guard_failure else "incomplete_training",
                    "seeds": [],
                })["seeds"].append(seed)
                continue
            payload = _read(root)
            if payload.get("test_accessed") or payload.get("test") is not None:
                raise RuntimeError(f"Validation accessed test data: {root}")
            if payload["initial_common_sha256"] != initial_hashes[seed]:
                raise RuntimeError(f"Common initialization mismatch: {configuration}/seed{seed}")
            if not checkpoint_path(configuration, seed).exists():
                raise RuntimeError(f"Missing guarded checkpoint: {configuration}/seed{seed}")
            metrics = _metrics(payload, "validation_guarded_core")
            rows[configuration].append(metrics)
            if configuration in TRAINED_CONFIGURATIONS:
                references = {item["name"]: item for item in payload["guard_references"]}
                if set(references) != {"five_f80", "partial_l2_f80"}:
                    raise RuntimeError(f"Dual actor guards were not applied: {root}")
                for reference in references.values():
                    if metrics["team_accuracy"] < reference["team_accuracy"] - reference["margin"] - 1e-12:
                        raise RuntimeError(f"Team guard violation: {root}")
                    if metrics["player_top1"] < reference["player_top1"] - reference["margin"] - 1e-12:
                        raise RuntimeError(f"Player guard violation: {root}")

    complete_configurations = tuple(
        name for name in ALL_CONFIGURATIONS
        if name not in failed_configurations and len(rows[name]) == len(CONFIRMATION_SEEDS)
    )
    arrays = _statistics("validation", complete_configurations)
    bootstraps = {
        f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right])
        for left, right in (
            ("partial_l2", "partial_l2_hard"),
            ("partial_l2", "partial_l2_soft"),
            ("partial_l2_hard", "partial_l2_soft"),
        )
        if left in arrays and right in arrays
    }
    f80 = pd.DataFrame(rows["partial_l2"])
    decisions: dict[str, Any] = {}
    candidates = ["partial_l2"]
    for configuration in TRAINED_CONFIGURATIONS:
        if configuration in failed_configurations:
            decisions[configuration] = {
                "tasks": {},
                "core_guardrails": {},
                "eligible": False,
                "failure": failed_configurations[configuration],
            }
            continue
        candidate = pd.DataFrame(rows[configuration])
        differences = candidate - f80
        comparison = bootstraps[f"{configuration}_minus_partial_l2"]
        tasks = {
            task: _task_decision(task, differences[metric].to_numpy(), comparison)
            for task, metric in TASK_METRICS.items()
        }
        guards = {
            "event_accuracy": float(differences.event_accuracy.mean()) >= -0.01,
            "event_macro_f1": float(differences.event_macro_f1.mean()) > -0.02,
            "time": float(differences.time_mae_seconds.mean()) < 0.05,
            "position": float(differences.position_distance_mae_m.mean()) < 0.50,
        }
        eligible = all(guards.values()) and any(item["effective"] for item in tasks.values())
        decisions[configuration] = {
            "tasks": tasks,
            "core_guardrails": guards,
            "eligible": eligible,
        }
        if eligible:
            candidates.append(configuration)

    losses = {name: float(np.mean([row["core_etp_loss"] for row in rows[name]])) for name in candidates}
    best_loss = min(losses.values())
    tied = [name for name, value in losses.items() if value <= best_loss + 1e-4]
    selected = min(tied, key=COST_ORDER.__getitem__)
    lock = {
        "selection_split": "validation",
        "selected_configuration": selected,
        "candidate_configurations": candidates,
        "mean_validation_core_etp_loss": losses,
        "decisions": decisions,
        "failed_configurations": failed_configurations,
        "validation_bootstrap": bootstraps,
        "initial_common_sha256": initial_hashes,
        "checkpoints": {
            str(seed): str(checkpoint_path(selected, seed)) for seed in CONFIRMATION_SEEDS
        },
        "test_accessed": False,
    }
    output = final_lock_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return output


def build_partial_context_report() -> Path:
    lock_path = final_lock_path()
    if not lock_path.exists():
        raise RuntimeError("Context is not locked on validation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = lock["selected_configuration"]
    output = PARTIAL_CONTEXT_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("validation", "test"):
        configurations = (
            tuple(name for name in ALL_CONFIGURATIONS if name not in lock["failed_configurations"])
            if split == "validation"
            else ("partial_l2", selected)
        )
        for configuration in dict.fromkeys(configurations):
            for seed in CONFIRMATION_SEEDS:
                rows.append({"configuration": configuration, "seed": seed, "split": split, **_metrics(_read((training_dir if split == "validation" else test_dir)(configuration, seed)), "validation_guarded_core" if split == "validation" else "test")})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    frame.groupby(["split", "configuration"]).agg(["mean", "std"]).to_csv(output / "summary.csv")
    test_bootstrap = None
    if selected != "partial_l2":
        arrays = _statistics("test", ("partial_l2", selected))
        test_bootstrap = _bootstrap(arrays["partial_l2"], arrays[selected])
    fusion = {}
    soft_results = [_read(training_dir("partial_l2_soft", seed)) for seed in CONFIRMATION_SEEDS]
    for task in ("event", "time", "position"):
        per_seed = []
        contexts = []
        for result in soft_results:
            epoch = result["best_epoch"]
            record = next(item for item in result["history"] if item["epoch"] == epoch)
            per_seed.append(record["fusion"]["weights"][task])
            contexts.append(record["fusion"]["context"])
        fusion[task] = {"per_seed": per_seed, "mean": np.mean(per_seed, axis=0).tolist()}
        fusion[task]["context"] = contexts
    report = {
        "selection": lock,
        "test_bootstrap_locked_minus_f80": test_bootstrap,
        "soft_fusion": fusion,
        "conclusion": (
            "Task-specific context was locked from validation and confirmed on test"
            if selected != "partial_l2"
            else "Neither Hard nor Soft produced a preregistered practical validation gain; Partial-L2-F80 remains locked"
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
