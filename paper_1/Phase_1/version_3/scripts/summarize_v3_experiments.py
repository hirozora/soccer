#!/usr/bin/env python3
"""Summarize Version 3 stage results and select the best feasible core."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE_ORDER = ("E0", "E1-S", "E1-T", "E2", "E3", "E4", "E5-POS", "E5-PLAYER", "E6")
CORE_STAGES = {"E2", "E3", "E4"}
METRICS = (
    "event_macro_f1",
    "event_accuracy",
    "time_mae_seconds",
    "position_euclidean_distance",
    "side_accuracy",
    "player_accuracy",
    "player_top3_accuracy",
    "player_mrr",
    "advantage_accuracy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    documents = {}
    for result_path in args.result_root.glob("*/result.json"):
        document = json.loads(result_path.read_text(encoding="utf-8"))
        documents[document["stage"]] = document
    if "E0" not in documents:
        raise ValueError("E0 result is required as the comparison reference")
    reference = documents["E0"]["best_epoch"]["validation"]
    rows = []
    for stage in STAGE_ORDER:
        if stage not in documents:
            continue
        document = documents[stage]
        best = document["best_epoch"]
        metrics = best["validation"]
        row = {
            "stage": stage,
            "best_epoch": best["epoch"],
            "feasible": best["feasible"],
            "num_parameters": document["num_parameters"],
            "num_trainable_parameters": document["num_trainable_parameters"],
        }
        for metric in METRICS:
            row[metric] = metrics[metric]
            row[f"delta_{metric}"] = metrics[metric] - reference[metric]
        trained_entries = [entry for entry in document["history"] if entry["epoch"] > 0]
        trained = [entry["validation"] for entry in trained_entries]
        diagnostic_entry = (
            max(trained_entries, key=lambda entry: entry["validation"]["event_macro_f1"])
            if trained_entries
            else best
        )
        diagnostic = diagnostic_entry["validation"]
        row["diagnostic_epoch"] = diagnostic_entry["epoch"]
        row["diagnostic_feasible"] = diagnostic_entry["feasible"]
        row["diagnostic_event_macro_f1"] = diagnostic["event_macro_f1"]
        row["diagnostic_time_mae_seconds"] = diagnostic["time_mae_seconds"]
        row["diagnostic_position_distance"] = diagnostic["position_euclidean_distance"]
        row["diagnostic_player_top3"] = diagnostic["player_top3_accuracy"]
        # These are independent per-task extrema and may come from different epochs.
        row["peak_time_mae_seconds"] = (
            min(value["time_mae_seconds"] for value in trained) if trained else metrics["time_mae_seconds"]
        )
        row["peak_position_distance"] = (
            min(value["position_euclidean_distance"] for value in trained)
            if trained
            else metrics["position_euclidean_distance"]
        )
        row["peak_player_top3"] = (
            max(value["player_top3_accuracy"] for value in trained)
            if trained
            else metrics["player_top3_accuracy"]
        )
        rows.append(row)
    core_rows = [row for row in rows if row["stage"] in CORE_STAGES and row["feasible"]]
    best_core = max(core_rows, key=lambda row: row["event_macro_f1"]) if core_rows else None
    feasible_rows = [row for row in rows if row["feasible"]]
    best_overall = max(feasible_rows, key=lambda row: row["event_macro_f1"])
    trained_rows = [row for row in rows if row["diagnostic_epoch"] > 0]
    best_trained = max(trained_rows, key=lambda row: row["diagnostic_event_macro_f1"]) if trained_rows else None
    summary = {
        "result_root": str(args.result_root.resolve()),
        "reference_stage": "E0",
        "best_core_stage": best_core["stage"] if best_core else None,
        "best_overall_stage": best_overall["stage"],
        "best_trained_event_stage": best_trained["stage"] if best_trained else None,
        "rows": rows,
    }
    output = args.output or args.result_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": str(output), "best_core": summary["best_core_stage"], "best_overall": summary["best_overall_stage"]}, indent=2))


if __name__ == "__main__":
    main()
