"""Validation lock and test-only confirmation for age-aware readout."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .age_pooling_study import (
    AGE_POOL_ROOT, ALL, COMPARISONS, TRAINED,
    checkpoint_path, lock_path, test_dir, training_dir,
)
from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .partial_context_reporting import PRACTICAL, TASK_METRICS, _metrics, _task_decision


def _read(root: Any) -> dict[str, Any]:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _prediction(configuration: str, seed: int, split: str) -> pd.DataFrame:
    filename = "validation_predictions_guarded_core.parquet" if split == "validation" else "test_predictions.parquet"
    root = training_dir(configuration, seed) if split == "validation" else test_dir(configuration, seed)
    return pd.read_parquet(root / filename).sort_values("sample_id").reset_index(drop=True)


def _statistics(split: str, configurations: tuple[str, ...]) -> dict[str, np.ndarray]:
    values = {name: [] for name in configurations}
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
                raise RuntimeError(f"Unpaired {split} predictions: {configuration}/seed{seed}")
            values[configuration].append(_match_statistics(frame, matches))
    return {name: np.stack(rows) for name, rows in values.items()}


def _core_guards(difference: pd.DataFrame) -> dict[str, bool]:
    return {
        "event_accuracy": float(difference.event_accuracy.mean()) >= -0.01,
        "event_macro_f1": float(difference.event_macro_f1.mean()) > -0.02,
        "time": float(difference.time_mae_seconds.mean()) < 0.05,
        "position": float(difference.position_distance_mae_m.mean()) < 0.50,
    }


def _profile_preference() -> dict[str, Any]:
    expected = {task: [] for task in ("event", "time", "position")}
    for seed in CONFIRMATION_SEEDS:
        diagnostic = json.loads(
            (training_dir("ap_task", seed) / "age_pooling_diagnostics.json").read_text()
        )
        if diagnostic["source_age_mismatch_count"]:
            raise RuntimeError(f"Age mismatch in AP-Task seed {seed}")
        for task in expected:
            expected[task].append(float(diagnostic["sample_summary"][task]["expected_age"]["mean"]))
    pairs = {}
    tasks = tuple(expected)
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1:]:
            differences = np.asarray(expected[left]) - np.asarray(expected[right])
            direction = int(np.sum(differences > 0))
            pairs[f"{left}_minus_{right}"] = {
                "differences_by_seed": differences.tolist(),
                "mean_difference_events": float(differences.mean()),
                "same_direction_seeds": max(direction, len(differences) - direction),
                "stable_separation": bool(
                    max(direction, len(differences) - direction) >= 2
                    and abs(float(differences.mean())) >= 2.0
                ),
            }
    return {"expected_age_by_seed": expected, "pairs": pairs}


def lock_validation() -> Any:
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in ALL}
    failures: dict[str, Any] = {}
    hashes: dict[int, str] = {}
    for seed in CONFIRMATION_SEEDS:
        baseline = _read(training_dir("ap_mean", seed))
        hashes[seed] = baseline["initial_common_sha256"]
        rows["ap_mean"].append(_metrics(baseline, "validation_guarded_core"))
        for configuration in TRAINED:
            root = training_dir(configuration, seed)
            if not (root / "result.json").exists():
                history = json.loads((root / "history.json").read_text()) if (root / "history.json").exists() else []
                reason = "no_dual_guard_epoch" if len(history) == 24 and not any(row.get("guarded_core_eligible") for row in history) else "incomplete"
                failures.setdefault(configuration, {"reason": reason, "seeds": []})["seeds"].append(seed)
                continue
            payload = _read(root)
            if payload.get("test_accessed"):
                raise RuntimeError("Validation accessed test data")
            if payload["initial_common_sha256"] != hashes[seed]:
                raise RuntimeError(f"Common initialization mismatch: {configuration}/seed{seed}")
            if not checkpoint_path(configuration, seed).exists():
                raise RuntimeError(f"Missing guarded checkpoint: {configuration}/seed{seed}")
            if {item["name"] for item in payload["guard_references"]} != {"five_f80", "partial_l2_f80"}:
                raise RuntimeError("Dual actor guards were not applied")
            if payload["age_pooling_diagnostics"]["source_age_mismatch_count"]:
                raise RuntimeError("Local/source age mismatch")
            rows[configuration].append(_metrics(payload, "validation_guarded_core"))

    complete = tuple(name for name in ALL if name not in failures and len(rows[name]) == 3)
    arrays = _statistics("validation", complete)
    bootstraps = {
        f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right])
        for left, right in COMPARISONS if left in complete and right in complete
    }
    decisions = {}
    for left, right in COMPARISONS:
        key = f"{right}_minus_{left}"
        if key not in bootstraps:
            continue
        difference = pd.DataFrame(rows[right]) - pd.DataFrame(rows[left])
        tasks = {
            task: _task_decision(task, difference[metric].to_numpy(), bootstraps[key])
            for task, metric in TASK_METRICS.items()
        }
        guards = _core_guards(difference)
        decisions[key] = {
            "tasks": tasks,
            "core_guardrails": guards,
            "effective": all(guards.values()) and any(value["effective"] for value in tasks.values()),
        }
    output = lock_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "selection_split": "validation",
        "complete_configurations": complete,
        "ineligible": failures,
        "decisions": decisions,
        "profile_preference": _profile_preference() if "ap_task" in complete else None,
        "bootstrap": bootstraps,
        "initial_common_sha256": hashes,
        "test_accessed": False,
    }, indent=2), encoding="utf-8")
    return output


def _write_profile_tables(configurations: tuple[str, ...], output: Any) -> None:
    score_rows, sample_rows = [], []
    for configuration in configurations:
        if configuration == "ap_mean":
            continue
        for seed in CONFIRMATION_SEEDS:
            diagnostic = json.loads(
                (training_dir(configuration, seed) / "age_pooling_diagnostics.json").read_text()
            )
            for task in ("event", "time", "position"):
                profile = diagnostic["canonical"][task]
                for age, (score, weight) in enumerate(zip(profile["relative_score"], profile["canonical_weight"])):
                    score_rows.append({
                        "configuration": configuration, "seed": seed, "task": task,
                        "age": age, "relative_score": score, "canonical_weight": weight,
                    })
                summary = diagnostic["sample_summary"][task]
                sample_rows.append({
                    "configuration": configuration, "seed": seed, "task": task,
                    **{f"{name}_{stat}": value for name, values in summary.items() if name != "bucket_mass" for stat, value in values.items()},
                })
    pd.DataFrame(score_rows).to_csv(output / "age_score_profiles.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "sample_age_summary.csv", index=False)


def build_report() -> Any:
    if not lock_path().exists():
        raise RuntimeError("Validation lock is required")
    lock = json.loads(lock_path().read_text())
    complete = tuple(lock["complete_configurations"])
    tested = tuple(name for name in complete if all((test_dir(name, seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS))
    rows = []
    for split in ("validation_guarded_core", "test"):
        for configuration in tested:
            for seed in CONFIRMATION_SEEDS:
                root = training_dir(configuration, seed) if split.startswith("validation") else test_dir(configuration, seed)
                rows.append({"configuration": configuration, "seed": seed, "split": split, **_metrics(_read(root), split)})
    frame = pd.DataFrame(rows)
    output = AGE_POOL_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    frame.groupby(["split", "configuration"]).agg(["mean", "std"]).to_csv(output / "summary.csv")
    arrays = _statistics("test", tested)
    test_bootstrap = {
        f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right])
        for left, right in COMPARISONS if left in tested and right in tested
    }
    (output / "test_bootstrap.json").write_text(json.dumps(test_bootstrap, indent=2), encoding="utf-8")
    _write_profile_tables(tested, output)
    (output / "report.json").write_text(json.dumps({
        "validation_decisions": lock["decisions"],
        "profile_preference": lock["profile_preference"],
        "test_only_confirms_locked_hypotheses": True,
        "ineligible": lock["ineligible"],
        "practical_thresholds": PRACTICAL,
    }, indent=2), encoding="utf-8")
    return output
