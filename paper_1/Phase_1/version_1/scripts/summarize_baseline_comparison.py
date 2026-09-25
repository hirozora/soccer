#!/usr/bin/env python3
"""Aggregate the completed same-seed HGT and baseline validation comparison."""

from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = VERSION_ROOT / "experiments/history"
OUTPUT_PATH = EXPERIMENT_ROOT / "baseline_comparison_summary_v1.json"
SEED = 20260715


def metric_row(
    model_name: str,
    sequence_length: int,
    num_parameters: int,
    metrics: dict,
) -> dict:
    return {
        "model": model_name,
        "seed": SEED,
        "sequence_length": sequence_length,
        "num_parameters": num_parameters,
        "event_macro_f1": metrics["event_macro_f1"],
        "event_accuracy": metrics["event_accuracy"],
        "time_mae_seconds": metrics["time_mae_seconds"],
        "position_euclidean_distance": metrics["position_euclidean_distance"],
    }


def main() -> None:
    hgt_path = EXPERIMENT_ROOT / "window_ablation_stage2/k_80.json"
    hgt = json.loads(hgt_path.read_text(encoding="utf-8"))
    hgt_metrics = hgt["best_epoch"]["validation"]
    rows = [
        metric_row("hgt_v1", 80, hgt["num_parameters"], hgt_metrics)
    ]
    for model_name in (
        "soccer_seq2event",
        "unified_lem",
        "og_lem_extended",
    ):
        path = (
            EXPERIMENT_ROOT
            / "baseline_comparison"
            / f"seed_{SEED}"
            / f"{model_name}.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            metric_row(
                model_name,
                data["sequence_length"],
                data["num_parameters"],
                data["best_epoch"]["validation"],
            )
        )
    ranking = [
        row["model"]
        for row in sorted(rows, key=lambda row: row["event_macro_f1"], reverse=True)
    ]
    output = {
        "experiment": "controlled_small_scale_validation_comparison_v1",
        "status": "end-to-end flow validation",
        "shared_protocol": {
            "competition": "England",
            "train_matches": 48,
            "validation_matches": 12,
            "steps_per_match": 96,
            "epochs": 8,
            "seed": SEED,
            "event_label_space": "common 10-class Wyscout eventName vocabulary",
        },
        "ranking_by_event_macro_f1": ranking,
        "results": rows,
        "conclusion": (
            "The fixed-window HGT pipeline and adapted existing methods are in "
            "the same broad performance range on the controlled smoke protocol."
        ),
        "limitations": [
            "Validation-only smoke comparison; the held-out test split was not used.",
            "This summary uses one fully regenerated common seed.",
            "Adapters preserve core architectures but use the common label space.",
            "HGT is trained for six tasks, Soccer-SEQ2Event for three, and LEM adapters for event type only.",
            "The subset and eight-epoch budget are too small for paper-level claims.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
