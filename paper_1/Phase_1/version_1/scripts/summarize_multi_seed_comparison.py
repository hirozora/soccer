#!/usr/bin/env python3
"""Summarize the five requested baseline configurations over three seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


VERSION_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = VERSION_ROOT / "experiments/history"
SEEDS = (20260715, 20260716, 20260717)
METRICS = (
    "event_macro_f1",
    "event_accuracy",
    "time_mae_seconds",
    "position_euclidean_distance",
)


def result_path(configuration: str, seed: int) -> Path:
    if configuration == "hgt_v1_k80":
        if seed == SEEDS[0]:
            return EXPERIMENT_ROOT / "window_ablation_stage2/k_80.json"
        return (
            EXPERIMENT_ROOT
            / "baseline_comparison_multi_seed/hgt"
            / f"seed_{seed}/k_80.json"
        )
    model, window = {
        "soccer_seq2event_default_k40": ("soccer_seq2event", "default"),
        "soccer_seq2event_k80": ("soccer_seq2event", "k80"),
        "unified_lem_default_k9": ("unified_lem", "default"),
        "unified_lem_k80": ("unified_lem", "k80"),
    }[configuration]
    root = (
        EXPERIMENT_ROOT / "baseline_comparison"
        if window == "default"
        else EXPERIMENT_ROOT / "baseline_comparison_equal_window_k80"
    )
    return root / f"seed_{seed}/{model}.json"


def read_run(configuration: str, seed: int) -> dict:
    path = result_path(configuration, seed)
    document = json.loads(path.read_text(encoding="utf-8"))
    metrics = document["best_epoch"]["validation"]
    sequence_length = document.get("window_size", document.get("sequence_length"))
    return {
        "seed": seed,
        "path": str(path.relative_to(VERSION_ROOT)),
        "sequence_length": sequence_length,
        "num_parameters": document["num_parameters"],
        "best_epoch": document["best_epoch"]["epoch"],
        **{metric: metrics[metric] for metric in METRICS},
    }


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    configurations = (
        "hgt_v1_k80",
        "soccer_seq2event_default_k40",
        "soccer_seq2event_k80",
        "unified_lem_default_k9",
        "unified_lem_k80",
    )
    runs = {
        configuration: [read_run(configuration, seed) for seed in SEEDS]
        for configuration in configurations
    }
    summary = []
    reference = runs["hgt_v1_k80"]
    for configuration in configurations:
        configuration_runs = runs[configuration]
        metrics = {}
        for metric in METRICS:
            values = [run[metric] for run in configuration_runs if run[metric] is not None]
            metrics[metric] = aggregate(values) if values else None
        paired = {}
        if configuration != "hgt_v1_k80":
            for metric in METRICS:
                differences = [
                    run[metric] - base[metric]
                    for run, base in zip(configuration_runs, reference, strict=True)
                    if run[metric] is not None and base[metric] is not None
                ]
                paired[metric] = aggregate(differences) if differences else None
        summary.append(
            {
                "configuration": configuration,
                "sequence_length": configuration_runs[0]["sequence_length"],
                "num_parameters": configuration_runs[0]["num_parameters"],
                "metrics": metrics,
                "paired_delta_vs_hgt": paired or None,
                "runs": configuration_runs,
            }
        )

    ranked = sorted(
        summary,
        key=lambda item: item["metrics"]["event_macro_f1"]["mean"],
        reverse=True,
    )
    report = {
        "experiment": "five_configuration_three_seed_comparison_v1",
        "protocol": {
            "competition": "England",
            "train_matches": 48,
            "validation_matches": 12,
            "steps_per_match": 96,
            "epochs": 8,
            "batch_size": 48,
            "seeds": list(SEEDS),
        },
        "ranking_by_mean_event_macro_f1": [
            item["configuration"] for item in ranked
        ],
        "configurations": summary,
        "unified_lem_k80_status": (
            "majority-class collapse in all three seeds; report for completeness, "
            "not as a converged capacity estimate"
        ),
        "limitations": [
            "Three seeds reduce random-initialization uncertainty but do not replace a held-out test evaluation.",
            "The comparison uses a controlled validation subset rather than every event.",
            "The models differ in parameter count and number of prediction tasks.",
            "Unified LEM K=80 does not converge under the shared eight-epoch budget.",
        ],
    }
    output = EXPERIMENT_ROOT / "baseline_comparison_multi_seed_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["configuration", "sequence_length", "num_parameters"]
        for metric in METRICS:
            fieldnames.extend((f"{metric}_mean", f"{metric}_sample_std"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            row = {
                "configuration": item["configuration"],
                "sequence_length": item["sequence_length"],
                "num_parameters": item["num_parameters"],
            }
            for metric in METRICS:
                aggregate_value = item["metrics"][metric]
                row[f"{metric}_mean"] = (
                    aggregate_value["mean"] if aggregate_value else None
                )
                row[f"{metric}_sample_std"] = (
                    aggregate_value["sample_std"] if aggregate_value else None
                )
            writer.writerow(row)
    print(json.dumps({"json": str(output), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
