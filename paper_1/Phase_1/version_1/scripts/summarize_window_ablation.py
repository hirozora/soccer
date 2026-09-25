#!/usr/bin/env python3
"""Summarize completed fixed-window smoke experiments."""

from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = VERSION_ROOT / "experiments/history"
OUTPUT_PATH = EXPERIMENT_ROOT / "window_ablation_summary_v1.json"


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    best = data["best_epoch"]["validation"]
    return {
        "path": str(path.relative_to(VERSION_ROOT)),
        "window_size": data["window_size"],
        "seed": data["config"]["seed"],
        "event_macro_f1": best["event_macro_f1"],
        "event_accuracy": best["event_accuracy"],
        "time_mae_seconds": best["time_mae_seconds"],
        "position_euclidean_distance": best["position_euclidean_distance"],
        "elapsed_seconds": data["elapsed_seconds"],
    }


def main() -> None:
    stage1 = [
        load_result(path)
        for path in sorted((EXPERIMENT_ROOT / "window_ablation").glob("k_*.json"))
    ]
    stage2 = [
        load_result(path)
        for path in sorted(
            (EXPERIMENT_ROOT / "window_ablation_stage2").glob("k_*.json")
        )
    ]
    selected = max(stage2, key=lambda row: row["event_macro_f1"])
    output = {
        "selection_metric": "validation.event_macro_f1",
        "selected_window_size": selected["window_size"],
        "selection_status": "single-seed small-scale flow validation",
        "implementation_note": (
            "Results were regenerated after removing an external residual around "
            "HGTConv, which already includes an internal learned skip connection."
        ),
        "stage1_all_windows": stage1,
        "stage2_all_windows": stage2,
        "decision_note": (
            "K=80 is retained as the current provisional setting. The completed "
            "experiment confirms that fixed-window HGT is viable; it is not a "
            "paper-level window selection claim."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
