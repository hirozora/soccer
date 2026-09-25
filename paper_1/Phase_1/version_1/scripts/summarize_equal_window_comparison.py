#!/usr/bin/env python3
"""Compare K=80 HGT with K=80 sequence baseline adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("soccer_seq2event", "unified_lem")


def row(model: str, document: dict) -> dict:
    metrics = document["best_epoch"]["validation"]
    return {
        "model": model,
        "sequence_length": document["sequence_length"],
        "num_parameters": document["num_parameters"],
        "best_epoch": document["best_epoch"]["epoch"],
        "event_macro_f1": metrics["event_macro_f1"],
        "event_accuracy": metrics["event_accuracy"],
        "time_mae_seconds": metrics["time_mae_seconds"],
        "position_euclidean_distance": metrics["position_euclidean_distance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--hgt-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hgt = json.loads(args.hgt_result.read_text(encoding="utf-8"))
    hgt_metrics = hgt["best_epoch"]["validation"]
    rows = [
        {
            "model": "hgt_v1",
            "sequence_length": hgt["window_size"],
            "num_parameters": hgt["num_parameters"],
            "best_epoch": hgt["best_epoch"]["epoch"],
            "event_macro_f1": hgt_metrics["event_macro_f1"],
            "event_accuracy": hgt_metrics["event_accuracy"],
            "time_mae_seconds": hgt_metrics["time_mae_seconds"],
            "position_euclidean_distance": hgt_metrics["position_euclidean_distance"],
        }
    ]
    for model in MODELS:
        path = args.experiment_root / f"seed_{args.seed}" / f"{model}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["sequence_length"] != 80:
            raise ValueError(f"{path} is not a K=80 result")
        rows.append(row(model, document))

    unified_sweep = []
    sweep_prefix = args.experiment_root.name + "_lr_"
    for root in sorted(args.experiment_root.parent.glob(sweep_prefix + "*")):
        path = root / f"seed_{args.seed}" / "unified_lem.json"
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        metrics = document["best_epoch"]["validation"]
        unified_sweep.append(
            {
                "learning_rate": document["learning_rate"],
                "event_macro_f1": metrics["event_macro_f1"],
                "event_accuracy": metrics["event_accuracy"],
                "best_epoch": document["best_epoch"]["epoch"],
            }
        )
    unified_default = json.loads(
        (args.experiment_root / f"seed_{args.seed}" / "unified_lem.json").read_text(
            encoding="utf-8"
        )
    )
    unified_default_metrics = unified_default["best_epoch"]["validation"]
    unified_sweep.append(
        {
            "learning_rate": unified_default["learning_rate"],
            "event_macro_f1": unified_default_metrics["event_macro_f1"],
            "event_accuracy": unified_default_metrics["event_accuracy"],
            "best_epoch": unified_default["best_epoch"]["epoch"],
        }
    )
    unified_sweep.sort(key=lambda item: item["learning_rate"])

    reference = rows[0]
    for item in rows[1:]:
        item["delta_event_macro_f1_vs_hgt"] = (
            item["event_macro_f1"] - reference["event_macro_f1"]
        )
        item["delta_event_accuracy_vs_hgt"] = (
            item["event_accuracy"] - reference["event_accuracy"]
        )
    report = {
        "experiment": "equal_k80_controlled_baseline_comparison_v1",
        "protocol": {
            "competition": "England",
            "train_matches": 48,
            "validation_matches": 12,
            "steps_per_match": 96,
            "epochs": 8,
            "batch_size": 48,
            "seed": args.seed,
            "sequence_length": 80,
        },
        "ranking_by_event_macro_f1": [
            item["model"]
            for item in sorted(rows, key=lambda item: item["event_macro_f1"], reverse=True)
        ],
        "results": rows,
        "unified_lem_k80_learning_rate_sweep": unified_sweep,
        "unified_lem_k80_status": (
            "majority-class collapse across all tested learning rates; "
            "not an informative estimate of converged model capacity"
        ),
        "conclusion": (
            "Equalizing context at K=80 did not improve baseline event prediction. "
            "Soccer-SEQ2Event remained below HGT, while Unified LEM did not train "
            "successfully with its flattened K=80 input under this budget."
        ),
        "limitations": [
            "Controlled validation subset only; the held-out test split is not used.",
            "This result uses one seed.",
            "Increasing Unified LEM context also increases its parameter count.",
            "Unified LEM K=80 is an optimization failure and must not be treated as a stable lower bound.",
            "HGT predicts six tasks, Soccer-SEQ2Event three, and Unified LEM event type only.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
