#!/usr/bin/env python3
"""Build a compact final comparison from controlled and full-validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "event_macro_f1",
    "event_accuracy",
    "time_mae_seconds",
    "position_euclidean_distance",
    "side_accuracy",
    "player_top3_accuracy",
    "advantage_accuracy",
)


def compact(metrics: dict) -> dict:
    return {name: metrics[name] for name in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.controlled_root / "summary.json").read_text(encoding="utf-8"))
    full = json.loads(args.full_result.read_text(encoding="utf-8"))
    e6 = json.loads((args.controlled_root / "e6" / "result.json").read_text(encoding="utf-8"))

    controlled_rows = {
        row["stage"]: {
            "selected_epoch": row["best_epoch"],
            "selected": {name: row[name] for name in METRICS},
            "event_best_trained_checkpoint": {
                "epoch": row["diagnostic_epoch"],
                "feasible": row["diagnostic_feasible"],
                "event_macro_f1": row["diagnostic_event_macro_f1"],
                "time_mae_seconds": row["diagnostic_time_mae_seconds"],
                "position_euclidean_distance": row["diagnostic_position_distance"],
                "player_top3_accuracy": row["diagnostic_player_top3"],
            },
            "independent_task_peaks": {
                "time_mae_seconds": row["peak_time_mae_seconds"],
                "position_euclidean_distance": row["peak_position_distance"],
                "player_top3_accuracy": row["peak_player_top3"],
            },
        }
        for row in summary["rows"]
    }

    diagnostic = e6["diagnostic_best_trained_epoch"]["validation"]
    leaveout = {}
    for expert, metrics in e6["diagnostic_leave_one_expert_out"].items():
        leaveout[expert] = {
            "event_macro_f1_drop": diagnostic["event_macro_f1"] - metrics["event_macro_f1"],
            "time_mae_increase_seconds": metrics["time_mae_seconds"] - diagnostic["time_mae_seconds"],
            "position_distance_increase": (
                metrics["position_euclidean_distance"] - diagnostic["position_euclidean_distance"]
            ),
            "player_top3_drop": diagnostic["player_top3_accuracy"] - metrics["player_top3_accuracy"],
        }

    report = {
        "selection_policy": {
            "primary": "maximum Event Macro-F1 among feasible checkpoints, including epoch 0",
            "selected_stage": summary["best_overall_stage"],
            "selected_core_stage": summary["best_core_stage"],
            "best_trained_event_stage": summary["best_trained_event_stage"],
        },
        "controlled_validation": controlled_rows,
        "full_validation": {
            "stage": full["stage"],
            "epoch": full["best_epoch"]["epoch"],
            "num_samples": full["best_epoch"]["validation"]["num_samples"],
            "metrics": compact(full["best_epoch"]["validation"]),
            "initial_equivalence_max_error": full["initial_equivalence"][
                "maximum_absolute_output_difference"
            ],
        },
        "e6_event_best_checkpoint_leave_one_expert_out": {
            "epoch": e6["diagnostic_best_trained_epoch"]["epoch"],
            "baseline": compact(diagnostic),
            "deltas_when_removed": leaveout,
        },
        "interpretation": {
            "primary_result": "No trained V3 checkpoint exceeded V1 on controlled Event Macro-F1.",
            "task_result": (
                "Position and Player corrections produced large task-specific gains, "
                "but the combined model did not pass the Event-primary selection rule."
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "selected_stage": summary["best_overall_stage"]}, indent=2))


if __name__ == "__main__":
    main()
