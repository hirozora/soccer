"""Validation lock and test report for strict receptive-field models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .partial_context_reporting import PRACTICAL, TASK_METRICS, _metrics, _task_decision
from .receptive_field_study import ALL, RF_ROOT, TRAINED, checkpoint_path, lock_path, test_dir, training_dir


COMPARISONS = (
    ("rf_f80", "rf_core_u5"),
    ("rf_f80", "rf_core_u10"),
    ("rf_f80", "rf_task"),
    ("rf_core_u5", "rf_task"),
    ("rf_core_u10", "rf_task"),
)


def _read(root: Path) -> dict[str, Any]:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _prediction(configuration: str, seed: int, split: str) -> pd.DataFrame:
    name = "validation_predictions_guarded_core.parquet" if split == "validation" else "test_predictions.parquet"
    root = training_dir(configuration, seed) if split == "validation" else test_dir(configuration, seed)
    return pd.read_parquet(root / name).sort_values("sample_id").reset_index(drop=True)


def _statistics(split: str, configurations: tuple[str, ...]) -> dict[str, np.ndarray]:
    values = {name: [] for name in configurations}
    expected = None
    matches = None
    for seed in CONFIRMATION_SEEDS:
        for configuration in configurations:
            frame = _prediction(configuration, seed, split)
            ids = frame.sample_id.tolist()
            if expected is None:
                expected = ids
                matches = sorted(frame.match_id.astype(int).unique())
            elif ids != expected:
                raise RuntimeError(f"Unpaired {split} predictions: {configuration}/seed{seed}")
            values[configuration].append(_match_statistics(frame, matches))
    return {name: np.stack(rows) for name, rows in values.items()}


def lock_validation() -> Path:
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in ALL}
    failures: dict[str, Any] = {}
    hashes: dict[int, str] = {}
    for seed in CONFIRMATION_SEEDS:
        f80 = _read(training_dir("rf_f80", seed))
        hashes[seed] = f80["initial_common_sha256"]
        rows["rf_f80"].append(_metrics(f80, "validation_guarded_core"))
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
                raise RuntimeError(f"Initialization mismatch: {configuration}/seed{seed}")
            if not checkpoint_path(configuration, seed).exists():
                raise RuntimeError(f"Missing guarded checkpoint: {configuration}/seed{seed}")
            if {item["name"] for item in payload["guard_references"]} != {"five_f80", "partial_l2_f80"}:
                raise RuntimeError("Dual actor guards were not applied")
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
        left_frame, right_frame = pd.DataFrame(rows[left]), pd.DataFrame(rows[right])
        difference = right_frame - left_frame
        decisions[key] = {
            task: _task_decision(task, difference[metric].to_numpy(), bootstraps[key])
            for task, metric in TASK_METRICS.items()
        }
    output = lock_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "selection_split": "validation",
        "complete_configurations": complete,
        "ineligible": failures,
        "decisions": decisions,
        "bootstrap": bootstraps,
        "initial_common_sha256": hashes,
        "test_accessed": False,
    }, indent=2), encoding="utf-8")
    return output


def build_report() -> Path:
    if not lock_path().exists():
        raise RuntimeError("Validation lock is required")
    lock = json.loads(lock_path().read_text())
    complete = tuple(lock["complete_configurations"])
    tested = tuple(name for name in complete if all((test_dir(name, seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS))
    rows = []
    for split in ("validation_guarded_core", "test"):
        for configuration in tested:
            for seed in CONFIRMATION_SEEDS:
                payload = _read(training_dir(configuration, seed) if split.startswith("validation") else test_dir(configuration, seed))
                rows.append({"configuration": configuration, "seed": seed, "split": split, **_metrics(payload, split)})
    frame = pd.DataFrame(rows)
    output = RF_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    frame.groupby(["split", "configuration"]).agg(["mean", "std"]).to_csv(output / "summary.csv")
    arrays = _statistics("test", tested)
    bootstraps = {
        f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right])
        for left, right in COMPARISONS if left in tested and right in tested
    }
    (output / "test_bootstrap.json").write_text(json.dumps(bootstraps, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps({
        "validation_decisions": lock["decisions"],
        "test_only_confirms_locked_conclusions": True,
        "ineligible": lock["ineligible"],
        "practical_thresholds": PRACTICAL,
    }, indent=2), encoding="utf-8")
    return output
