"""Paired evaluation and final report for fixed-budget conflict experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CONFIRMATION_SEEDS
from .five_task_reporting import _bootstrap, _match_statistics
from .fixed_budget_study import FIXED_BUDGET_ROOT, STAGE1, test_dir, training_dir


def _prediction(configuration: str, seed: int, split: str, checkpoint: str) -> pd.DataFrame:
    if split == "validation":
        suffix = "core" if checkpoint == "core" else "joint"
        source = "five_f80" if configuration == "t5_core" else configuration
        path = training_dir(source, seed) / f"validation_predictions_{suffix}.parquet"
    else:
        path = test_dir(configuration, seed) / "test_predictions.parquet"
    return pd.read_parquet(path).sort_values("sample_id").reset_index(drop=True)


def paired_bootstrap(
    reference: str,
    candidate: str,
    *,
    split: str,
    checkpoint: str,
) -> dict[str, Any]:
    left_stats, right_stats = [], []
    expected_ids = None
    for seed in CONFIRMATION_SEEDS:
        left = _prediction(reference, seed, split, checkpoint)
        right = _prediction(candidate, seed, split, checkpoint)
        if left.sample_id.tolist() != right.sample_id.tolist():
            raise RuntimeError("Fixed-budget paired predictions are misaligned")
        if expected_ids is None:
            expected_ids = left.sample_id.tolist()
        elif expected_ids != left.sample_id.tolist():
            raise RuntimeError("Fixed-budget seeds use different samples")
        matches = sorted(left.match_id.astype(int).unique().tolist())
        left_stats.append(_match_statistics(left, matches))
        right_stats.append(_match_statistics(right, matches))
    return _bootstrap(np.stack(left_stats), np.stack(right_stats), iterations=10_000)


def _result(configuration: str, seed: int, split: str) -> dict[str, Any]:
    source = "five_f80" if configuration == "t5_core" and split == "validation" else configuration
    root = training_dir(source, seed) if split == "validation" else test_dir(source, seed)
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _row(configuration: str, seed: int, split: str) -> dict[str, Any]:
    result = _result(configuration, seed, split)
    metrics = (
        result["validation_core"]
        if split == "validation" and configuration == "t5_core"
        else result["validation"] if split == "validation" else result["test"]
    )
    return {
        "configuration": configuration,
        "seed": seed,
        "split": split,
        "best_epoch": result["best_core_epoch"] if split == "validation" and configuration == "t5_core" else result["best_epoch"],
        "event_accuracy": metrics["event"]["accuracy"],
        "event_macro_f1": metrics["event"]["macro_f1"],
        "time_mae_seconds": metrics["time"]["mae_seconds"],
        "position_distance_mae_m": metrics["position"]["distance_mae_m"],
        "team_accuracy": metrics["team"]["accuracy"],
        "team_macro_f1": metrics["team"]["macro_f1"],
        "player_top1": metrics["player"]["top1_accuracy"],
        "player_top3": metrics["player"]["top3_accuracy"],
        "player_top5": metrics["player"]["top5_accuracy"],
        "player_mrr": metrics["player"]["mrr"],
        "core_etp_loss": metrics["core_etp_loss"],
        "joint_active_loss": metrics["joint_active_loss"],
    }


def build_fixed_budget_report() -> Path:
    lock_path = FIXED_BUDGET_ROOT / "decisions/baseline_lock.json"
    if not lock_path.exists():
        raise RuntimeError("Fixed-budget baseline is not locked")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    configurations = list(STAGE1)
    stage2 = FIXED_BUDGET_ROOT / "decisions/stage2_trigger.json"
    if stage2.exists() and json.loads(stage2.read_text())["run_stage2"]:
        configurations.extend(("t4_team", "t4_player", "t5_core"))
    if all((training_dir("t5_player_adapter", seed) / "result.json").exists() for seed in CONFIRMATION_SEEDS):
        configurations.append("t5_player_adapter")
    output = FIXED_BUDGET_ROOT / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("validation", "test"):
        for configuration in configurations:
            for seed in CONFIRMATION_SEEDS:
                if (test_dir(configuration, seed) / "result.json").exists() or split == "validation":
                    rows.append(_row(configuration, seed, split))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    test = frame[frame.split == "test"]
    numeric = [name for name in test.columns if name not in {"configuration", "seed", "split"}]
    test.groupby("configuration")[numeric].agg(["mean", "std"]).to_csv(output / "test_summary.csv")

    comparisons = {}
    registered = [
        ("three_f80", "five_f80"),
        ("three_fixedb", "five_hard"),
        ("three_sfb", "five_soft"),
    ]
    if "t4_team" in configurations:
        registered.extend((
            ("three_f80", "t4_team"),
            ("three_f80", "t4_player"),
            ("three_f80", "t5_core"),
            ("t4_team", "t5_core"),
            ("t4_player", "t5_core"),
        ))
    if "t5_player_adapter" in configurations:
        registered.extend((("five_f80", "t5_player_adapter"), ("three_f80", "t5_player_adapter")))
    for reference, candidate in registered:
        comparisons[f"{candidate}_minus_{reference}"] = paired_bootstrap(
            reference, candidate, split="test", checkpoint="selected"
        )
    (output / "paired_bootstrap.json").write_text(
        json.dumps({"iterations": 10_000, "comparisons": comparisons}, indent=2), encoding="utf-8"
    )

    old = FIXED_BUDGET_ROOT.parent / "five_task_view_v1/report/test_summary.csv"
    report = {
        "baseline_lock": lock,
        "training_boundary": json.loads((FIXED_BUDGET_ROOT / "decisions/training_boundary.json").read_text()),
        "stage2_trigger": json.loads(stage2.read_text()) if stage2.exists() else None,
        "conflict": json.loads((FIXED_BUDGET_ROOT / "decisions/conflict.json").read_text()) if (FIXED_BUDGET_ROOT / "decisions/conflict.json").exists() else None,
        "old_eight_epoch_reference": str(old),
        "old_reference_exists": old.exists(),
        "test_was_locked_before_access": True,
        "partial_sharing_recommendation": (
            "branch HGT layer 2 into main and player paths"
            if lock["winner"] is None and "t5_player_adapter" in configurations
            else None
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
