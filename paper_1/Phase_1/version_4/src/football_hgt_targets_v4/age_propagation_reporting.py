"""Validation decisions and locked test report for age propagation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .age_propagation_study import (
    AGE_PROPAGATION_ROOT,
    ALL,
    COMPARISONS,
    COST_ORDER,
    TRAINED,
    checkpoint_path,
    lock_path,
    test_dir,
    training_dir,
)
from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .partial_context_reporting import (
    PRACTICAL,
    TASK_METRICS,
    _metrics,
    _task_decision,
)


def _read(root: Path) -> dict[str, Any]:
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _prediction(configuration: str, seed: int, split: str) -> pd.DataFrame:
    filename = (
        "validation_predictions_guarded_core.parquet"
        if split == "validation"
        else "test_predictions.parquet"
    )
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
                raise RuntimeError(
                    f"Unpaired {split} predictions: {configuration}/seed{seed}"
                )
            values[configuration].append(_match_statistics(frame, matches))
    return {name: np.stack(rows) for name, rows in values.items()}


def _core_guards(difference: pd.DataFrame) -> dict[str, bool]:
    return {
        "event_accuracy": float(difference.event_accuracy.mean()) >= -0.01,
        "event_macro_f1": float(difference.event_macro_f1.mean()) > -0.02,
        "time": float(difference.time_mae_seconds.mean()) < 0.05,
        "position": float(difference.position_distance_mae_m.mean()) < 0.50,
    }


def lock_validation() -> Path:
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in ALL}
    failures: dict[str, Any] = {}
    hashes: dict[int, str] = {}
    for seed in CONFIRMATION_SEEDS:
        baseline = _read(training_dir("partial_l2_f80", seed))
        hashes[seed] = baseline["initial_common_sha256"]
        for configuration in ALL:
            root = training_dir(configuration, seed)
            if not (root / "result.json").exists():
                history_path = root / "history.json"
                history = json.loads(history_path.read_text()) if history_path.exists() else []
                reason = (
                    "no_dual_guard_epoch"
                    if len(history) == 24
                    and not any(row.get("guarded_core_eligible") for row in history)
                    else "incomplete"
                )
                failures.setdefault(configuration, {"reason": reason, "seeds": []})[
                    "seeds"
                ].append(seed)
                continue
            payload = _read(root)
            if payload.get("test_accessed"):
                raise RuntimeError(f"Validation accessed test data: {root}")
            if payload["initial_common_sha256"] != hashes[seed]:
                raise RuntimeError(
                    f"Common initialization mismatch: {configuration}/seed{seed}"
                )
            if not checkpoint_path(configuration, seed).exists():
                raise RuntimeError(f"Missing guarded checkpoint: {configuration}/seed{seed}")
            rows[configuration].append(_metrics(payload, "validation_guarded_core"))
            if configuration in TRAINED:
                references = {item["name"] for item in payload["guard_references"]}
                if references != {"five_f80", "partial_l2_f80"}:
                    raise RuntimeError(f"Dual actor guards missing: {root}")
                diagnostics = payload.get("age_propagation_diagnostics")
                if not diagnostics or diagnostics["source_age_mismatch_count"]:
                    raise RuntimeError(f"Invalid propagation diagnostics: {root}")

    complete = tuple(
        name
        for name in ALL
        if name not in failures and len(rows[name]) == len(CONFIRMATION_SEEDS)
    )
    arrays = _statistics("validation", complete)
    bootstraps = {
        f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right])
        for left, right in COMPARISONS
        if left in arrays and right in arrays
    }
    decisions: dict[str, Any] = {}
    for left, right in COMPARISONS:
        key = f"{right}_minus_{left}"
        if key not in bootstraps:
            continue
        difference = pd.DataFrame(rows[right]) - pd.DataFrame(rows[left])
        tasks = {
            task: _task_decision(
                task, difference[metric].to_numpy(), bootstraps[key]
            )
            for task, metric in TASK_METRICS.items()
        }
        guards = _core_guards(difference)
        decisions[key] = {
            "tasks": tasks,
            "core_guardrails": guards,
            "effective": all(guards.values())
            and any(value["effective"] for value in tasks.values()),
        }

    candidates = ["partial_l2_f80"]
    for configuration in TRAINED:
        key = f"{configuration}_minus_partial_l2_f80"
        if configuration in complete and decisions.get(key, {}).get("effective"):
            candidates.append(configuration)
    losses = {
        name: float(np.mean([row["core_etp_loss"] for row in rows[name]]))
        for name in candidates
    }
    best_loss = min(losses.values())
    tied = [name for name, value in losses.items() if value <= best_loss + 1e-4]
    selected = min(tied, key=COST_ORDER.__getitem__)
    output = lock_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "complete_configurations": complete,
                "ineligible": failures,
                "decisions": decisions,
                "validation_bootstrap": bootstraps,
                "candidate_configurations": candidates,
                "mean_validation_core_etp_loss": losses,
                "selected_configuration": selected,
                "initial_common_sha256": hashes,
                "test_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def _write_gate_tables(configurations: tuple[str, ...], output: Path) -> None:
    rows = []
    profile_rows = []
    node_rows = []
    relation_rows = []
    embedding_rows = []
    for configuration in configurations:
        if configuration not in TRAINED:
            continue
        for seed in CONFIRMATION_SEEDS:
            diagnostic = json.loads(
                (training_dir(configuration, seed) / "age_propagation_diagnostics.json").read_text()
            )
            for layer, task_profiles in diagnostic["canonical"].items():
                for task, profile in task_profiles.items():
                    profile_rows.append({
                        "configuration": configuration,
                        "seed": seed,
                        "layer": layer,
                        "task": task,
                        "expected_age": profile["expected_age"],
                        "normalized_entropy": profile["normalized_entropy"],
                        "effective_sample_size": profile["effective_sample_size"],
                        **{
                            f"bucket_{name}": value
                            for name, value in profile["bucket_gate_mean"].items()
                        },
                    })
                    for age, gate in enumerate(profile["gate"]):
                        rows.append(
                            {
                                "configuration": configuration,
                                "seed": seed,
                                "layer": layer,
                                "task": task,
                                "age": age,
                                "gate": gate,
                                "relative_gate": profile["relative_gate"][age],
                                "normalized_gate": profile["normalized_gate"][age],
                            }
                        )
            for task, layers in diagnostic["residual"].items():
                for layer, values in layers.items():
                    for node_type, metrics in values["node"].items():
                        node_rows.append({
                            "configuration": configuration,
                            "seed": seed,
                            "task": task,
                            "layer": layer,
                            "node_type": node_type,
                            **metrics,
                        })
                    for relation, norm in values["relation_message_norm"].items():
                        relation_rows.append({
                            "configuration": configuration,
                            "seed": seed,
                            "task": task,
                            "layer": layer,
                            "relation": relation,
                            "message_norm": norm,
                        })
            for pair, metrics in diagnostic["embedding"].get("pairs", {}).items():
                embedding_rows.append({
                    "configuration": configuration,
                    "seed": seed,
                    "pair": pair,
                    **metrics,
                })
    pd.DataFrame(rows).to_csv(output / "gate_profiles.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "gate_profile_summary.csv", index=False)
    pd.DataFrame(node_rows).to_csv(output / "residual_node_statistics.csv", index=False)
    pd.DataFrame(relation_rows).to_csv(output / "relation_contributions.csv", index=False)
    pd.DataFrame(embedding_rows).to_csv(output / "task_embedding_statistics.csv", index=False)


def build_report() -> Path:
    if not lock_path().exists():
        raise RuntimeError("Validation lock is required")
    lock = json.loads(lock_path().read_text())
    complete = tuple(lock["complete_configurations"])
    tested = tuple(
        name
        for name in complete
        if all((test_dir(name, seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS)
    )
    rows = []
    for split in ("validation_guarded_core", "test"):
        for configuration in tested:
            for seed in CONFIRMATION_SEEDS:
                root = (
                    training_dir(configuration, seed)
                    if split.startswith("validation")
                    else test_dir(configuration, seed)
                )
                rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "split": split,
                        **_metrics(_read(root), split),
                    }
                )
    frame = pd.DataFrame(rows)
    output = AGE_PROPAGATION_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    frame.groupby(["split", "configuration"]).agg(["mean", "std"]).to_csv(
        output / "summary.csv"
    )
    arrays = _statistics("test", tested)
    test_bootstrap = {
        f"{right}_minus_{left}": _bootstrap(arrays[left], arrays[right])
        for left, right in COMPARISONS
        if left in arrays and right in arrays
    }
    (output / "test_bootstrap.json").write_text(
        json.dumps(test_bootstrap, indent=2), encoding="utf-8"
    )
    _write_gate_tables(tested, output)
    (output / "report.json").write_text(
        json.dumps(
            {
                "validation_lock": lock,
                "test_only_confirms_validation_decisions": True,
                "ap_task_test_was_visible_before_pg_design": True,
                "ap_comparison_is_not_a_pristine_holdout": True,
                "practical_thresholds": PRACTICAL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
